"""Behavioral contracts of the attentive probes and metadata-only baselines."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.features import FLOAT16_MAX, _float16_safe, layerwise_states
from src.evaluation.pipeline import METADATA_METHODS, PartialFits, _sampled_tokens
from src.evaluation.probes import (
    ATTENTIVE,
    AttentiveProbe,
    attentive_predictions,
    fit_attentive_layer,
    fit_mlp_layer,
    fit_ridge_layer,
    group_mean,
    local_queries,
    predict_attentive,
    select_layers,
)
from src.evaluation.protocol import (
    Protocol,
    governing_names,
    governing_values,
    position_metadata,
    regime_metadata,
    token_coordinates,
)
from src.models.vit import sincos_3d


def _context(rng, samples=10, tokens=12, dim=16):
    return torch.tensor(rng.normal(size=(samples, tokens, dim)), dtype=torch.float16)


def _samples(count):
    return [
        {
            "trajectory": i,
            "parameters": {"Rayleigh": 10.0 ** (5 + i % 3), "Prandtl": 0.5 + 0.25 * i},
            "age": i / max(count - 1, 1),
        }
        for i in range(count)
    ]


class GlobalAttentiveTests(unittest.TestCase):
    def test_factorized_query_matches_standard_cross_attention(self):
        torch.manual_seed(3)
        probe = AttentiveProbe(dim=16, heads=4, ffn_hidden=8, outputs=2).eval()
        x = torch.randn(5, 9, 16)
        with torch.no_grad():
            normalized = probe.norm_context(x)
            q = (
                probe.q_proj(probe.query)
                .reshape(1, 1, 4, 4)
                .transpose(1, 2)
                .expand(5, -1, -1, -1)
            )
            k, v = (
                projection(normalized).reshape(5, 9, 4, 4).transpose(1, 2)
                for projection in (probe.k_proj, probe.v_proj)
            )
            pooled = (
                F.scaled_dot_product_attention(q, k, v)
                .transpose(1, 2)
                .reshape(5, 1, 16)
            )
            summary = probe.query.expand(5, -1, -1) + probe.proj(pooled)
            reference = probe.head(summary + probe.ffn(probe.norm_summary(summary)))
            torch.testing.assert_close(probe(x), reference, atol=1e-6, rtol=1e-5)

    def test_prediction_uses_the_whole_context_not_one_token(self):
        torch.manual_seed(5)
        probe = AttentiveProbe(dim=16, heads=4, ffn_hidden=8).eval()
        x = torch.randn(2, 9, 16)
        altered = x.clone()
        altered[:, 4:, 0] += 3.0
        with torch.no_grad():
            self.assertGreater(float((probe(x) - probe(altered)).abs().max()), 1e-6)

    def test_capacity_follows_the_cited_single_block_design(self):
        probe = AttentiveProbe()
        self.assertEqual((probe.heads, ATTENTIVE["blocks"]), (8, 1))
        self.assertEqual(probe.query.shape, (1, 1, 384))
        self.assertEqual(probe.ffn[0].out_features, 96)
        self.assertEqual(probe.head.out_features, 1)
        self.assertFalse(
            any(isinstance(m, torch.nn.Dropout) for m in probe.modules())
        )
        self.assertEqual(ATTENTIVE["dropout"], 0.0)


class LocalAttentiveTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.grid = (1, 3, 4)
        self.tokens = int(np.prod(self.grid))
        self.context = _context(self.rng, samples=8, tokens=self.tokens)
        self.positions = np.stack(
            [
                np.sort(self.rng.choice(self.tokens, 4, replace=False))
                for _ in range(len(self.context))
            ]
        )

    def test_queries_are_location_conditioned_and_independent(self):
        torch.manual_seed(7)
        probe = AttentiveProbe(dim=16, heads=4, ffn_hidden=8, local=True).eval()
        coordinates = sincos_3d(16, *self.grid)
        context = self.context.float()
        index = torch.as_tensor(self.positions, dtype=torch.long)
        with torch.no_grad():
            queries = local_queries(context, index, coordinates)
            outputs = probe(context, queries)
            self.assertEqual(tuple(outputs.shape), (8, 4, 1))
            # The same frozen token asked about two locations differs only by
            # the fixed coordinate encoding, and must still change the answer.
            shifted = index.clone()
            shifted[:, 1] = shifted[:, 0]
            same_token = local_queries(context, shifted, coordinates)
            same_token[:, 1] = queries[:, 1] - coordinates[index[:, 1]] + coordinates[
                shifted[:, 1]
            ]
            self.assertGreater(
                float((probe(context, same_token)[:, 1] - outputs[:, 1]).abs().max()),
                1e-5,
            )
            # Replacing one query never moves the prediction of another.
            replaced = queries.clone()
            replaced[:, 0] += 5.0
            moved = probe(context, replaced)
            torch.testing.assert_close(moved[:, 1:], outputs[:, 1:])
            self.assertGreater(float((moved[:, 0] - outputs[:, 0]).abs().max()), 1e-5)

    def test_context_outside_the_queried_location_changes_the_prediction(self):
        torch.manual_seed(9)
        probe = AttentiveProbe(dim=16, heads=4, ffn_hidden=8, local=True).eval()
        coordinates = sincos_3d(16, *self.grid)
        context = self.context.float()
        index = torch.as_tensor(self.positions, dtype=torch.long)
        elsewhere = context.clone()
        mask = torch.ones(len(context), self.tokens, dtype=torch.bool)
        mask.scatter_(1, index, False)
        elsewhere[..., 0][mask] += 4.0
        with torch.no_grad():
            queries = local_queries(context, index, coordinates)
            torch.testing.assert_close(
                queries, local_queries(elsewhere, index, coordinates)
            )
            self.assertGreater(
                float(
                    (probe(context, queries) - probe(elsewhere, queries)).abs().max()
                ),
                1e-4,
            )

    def test_fit_preserves_every_sampled_location(self):
        target = (
            self.context.float()[np.arange(8)[:, None], self.positions]
            .mean(-1)
            .numpy()
            .astype(np.float64)
        )
        entry = fit_attentive_layer(
            3,
            self.context,
            {"enstrophy": target},
            self.context,
            {"enstrophy": target},
            positions=self.positions,
            valid_positions=self.positions,
            grid=self.grid,
            epochs=2,
            batch_size=4,
        )
        self.assertEqual(entry["layer"], 3)
        self.assertEqual(entry["fit"]["queries"], 4)
        self.assertTrue(entry["fit"]["local"])
        predictions = attentive_predictions(
            entry["fit"], self.context, self.positions
        )["enstrophy"]
        self.assertEqual(predictions.shape, (8 * 4,))

    def test_query_scale_is_stable_across_encoder_depth(self):
        coordinates = sincos_3d(16, *self.grid)
        index = torch.as_tensor(self.positions, dtype=torch.long)
        context = self.context.float()
        # Deeper encoder outputs are larger and offset; the coordinate must
        # keep the same relative weight in the query regardless.
        deeper = 40.0 * context + 7.0
        torch.testing.assert_close(
            local_queries(context, index, coordinates),
            local_queries(deeper, index, coordinates),
            atol=1e-4,
            rtol=1e-4,
        )
        queries = local_queries(context, index, coordinates)
        token_part = queries - coordinates[index]
        torch.testing.assert_close(
            token_part.mean(-1), torch.zeros_like(token_part.mean(-1)), atol=1e-5, rtol=0
        )
        self.assertGreater(float(coordinates[index].abs().max()), 0.1)


class LeakageTests(unittest.TestCase):
    def test_probe_inputs_are_the_input_context_tokens_only(self):
        from src.data.well import ClipSpec
        from src.models.vit import build_encoder

        protocol = Protocol("rayleigh_benard", n_frames=2, patch=(1, 64, 32))
        spec = ClipSpec(4, 2, 512, 128)
        encoder = build_encoder(
            spec,
            dict(patch_t=1, patch_h=64, patch_w=32, embed_dim=16, depth=2, num_heads=2),
        ).eval()
        context = torch.randn(2, 4, 2, 512, 128)
        future = torch.randn(2, 4, 2, 512, 128)
        states = layerwise_states(encoder, context)
        # Every staged encoder output has exactly the context's token count.
        expected = (2 // protocol.patch[0]) * (512 // 64) * (128 // 32)
        self.assertEqual({tuple(s.shape) for s in states}, {(2, expected, 16)})
        self.assertEqual(len(states), len(encoder.blocks) + 1)
        # A different future clip cannot change any probe input.
        for staged, again in zip(states, layerwise_states(encoder, context)):
            torch.testing.assert_close(staged, again)
        self.assertFalse(
            any(
                torch.allclose(a, b)
                for a, b in zip(states, layerwise_states(encoder, future))
            )
        )

    def test_sampled_tokens_come_from_the_staged_context(self):
        rng = np.random.default_rng(2)
        context = _context(rng, samples=4, tokens=6, dim=8)
        positions = np.stack(
            [np.sort(rng.choice(6, 3, replace=False)) for _ in range(4)]
        )
        gathered = _sampled_tokens(context, positions)
        self.assertEqual(gathered.shape, (12, 8))
        np.testing.assert_array_equal(
            gathered,
            context.float().numpy()[np.arange(4)[:, None], positions].reshape(12, 8),
        )


class MetadataBaselineTests(unittest.TestCase):
    def test_pooled_and_local_metadata_have_exactly_three_and_six_inputs(self):
        samples = _samples(6)
        pooled = regime_metadata(samples, "rayleigh_benard")
        self.assertEqual(pooled.shape, (6, 3))
        np.testing.assert_allclose(
            pooled[:, 0], [5 + i % 3 for i in range(6)], atol=1e-12
        )
        np.testing.assert_allclose(
            pooled[:, 2], [i / 5 for i in range(6)], atol=1e-12
        )
        grid = (1, 3, 4)
        positions = np.tile(np.array([0, 11]), (6, 1))
        local = position_metadata(samples, positions, grid, "rayleigh_benard")
        self.assertEqual(local.shape, (12, 6))
        np.testing.assert_allclose(local[:, :3], np.repeat(pooled, 2, axis=0))
        np.testing.assert_allclose(
            local[:2, 3:], [[0, 0, 0], [0, 1, 1]], atol=1e-12
        )
        np.testing.assert_allclose(
            token_coordinates(np.array([0, 11]), grid), local[:2, 3:]
        )

    def test_metadata_mlp_sees_no_encoder_feature(self):
        samples = _samples(12)
        metadata = regime_metadata(samples, "rayleigh_benard")
        target = 2.0 * metadata[:, 0] - metadata[:, 2]
        entry = fit_mlp_layer(
            0, metadata, target, metadata, target, max_steps=40, min_steps=20
        )
        self.assertEqual(entry["fit"]["states"][0]["network.0.weight"].shape[1], 3)
        self.assertEqual(entry["fit"]["hidden"], 128)
        self.assertEqual(entry["fit"]["dropout"], 0.1)
        result = select_layers([entry], include_depth=False)
        self.assertEqual(result["selected_layer"], 0)
        self.assertEqual(
            set(METADATA_METHODS.values()),
            {"regime_time_mlp", "regime_time_position_mlp"},
        )


class GoverningTargetTests(unittest.TestCase):
    def test_paper_convention_logs_rayleigh_and_keeps_prandtl_linear(self):
        self.assertEqual(
            governing_names("rayleigh_benard"), ("log10_Rayleigh", "Prandtl")
        )
        values = governing_values(
            "rayleigh_benard", {"Rayleigh": 1e6, "Prandtl": 0.7}
        )
        self.assertAlmostEqual(values["log10_Rayleigh"], 6.0)
        self.assertAlmostEqual(values["Prandtl"], 0.7)
        self.assertEqual(
            list(Protocol("rayleigh_benard").to_dict()["governing_targets"]),
            ["log10_Rayleigh", "Prandtl"],
        )

    def test_joint_fits_report_each_parameter_and_a_normalized_mse(self):
        rng = np.random.default_rng(4)
        features = rng.normal(size=(24, 5)).astype(np.float32)
        rayleigh = 6 + 0.5 * features[:, 0].astype(np.float64)
        prandtl = 0.7 + 0.01 * features[:, 1].astype(np.float64)
        targets = {"log10_Rayleigh": rayleigh, "Prandtl": prandtl}
        entry = fit_ridge_layer(2, features, targets, features, targets, joint=True)
        self.assertEqual(entry["fit"]["weights"].shape, (5, 2))
        self.assertEqual(sorted(entry["outputs"]), ["Prandtl", "log10_Rayleigh"])
        self.assertAlmostEqual(
            entry["valid_normalized_mse"],
            float(
                np.mean(
                    [
                        entry["outputs"][name]["valid_normalized_mse"]
                        for name in targets
                    ]
                )
            ),
        )
        # Metrics stay in each parameter's own units, while normalized MSE
        # divides by that parameter's train variance.
        for name, values in targets.items():
            self.assertAlmostEqual(
                entry["outputs"][name]["valid_normalized_mse"],
                entry["outputs"][name]["valid_mse"] / np.var(values),
            )
        # Standardization uses train statistics only.
        np.testing.assert_allclose(
            entry["fit"]["target_mean"].numpy(), [rayleigh.mean(), prandtl.mean()]
        )
        # The governing attentive head sees every real sampled window, never
        # an element-wise average of token sequences, and its per-window
        # predictions are averaged per trajectory before scoring.
        context = _context(np.random.default_rng(6), samples=24, tokens=6)
        groups = [np.arange(3 * i, 3 * i + 3) for i in range(8)]
        windows = {
            name: np.repeat(values[:8], 3) for name, values in targets.items()
        }
        per_trajectory = {name: values[:8] for name, values in targets.items()}
        joint = fit_attentive_layer(
            2,
            context,
            windows,
            context,
            per_trajectory,
            valid_groups=groups,
            epochs=2,
            batch_size=6,
            joint=True,
        )
        self.assertEqual(joint["fit"]["outputs"], ["log10_Rayleigh", "Prandtl"])
        self.assertEqual(joint["fit"]["state"]["head.weight"].shape[0], 2)
        self.assertIn("valid_normalized_mse", joint)
        # Train-only standardization comes from the repeated window labels.
        np.testing.assert_allclose(
            joint["fit"]["target_mean"],
            [windows["log10_Rayleigh"].mean(), windows["Prandtl"].mean()],
        )
        windowed = predict_attentive(joint["fit"], context)
        averaged = predict_attentive(joint["fit"], context, groups=groups)
        self.assertEqual(windowed.shape, (24, 1, 2))
        self.assertEqual(averaged.shape, (8, 1, 2))
        np.testing.assert_allclose(
            averaged, group_mean(windowed, groups), atol=1e-10
        )
        self.assertEqual(
            attentive_predictions(joint["fit"], context, groups=groups)[
                "Prandtl"
            ].shape,
            (8,),
        )
        with self.assertRaisesRegex(ValueError, "never grouped"):
            fit_attentive_layer(
                2,
                context,
                {"enstrophy": np.zeros((24, 2))},
                context,
                {"enstrophy": np.zeros((24, 2))},
                positions=np.zeros((24, 2), dtype=int),
                valid_positions=np.zeros((24, 2), dtype=int),
                grid=(1, 2, 3),
                valid_groups=groups,
                epochs=1,
            )

    def test_target_offsets_include_the_new_horizon(self):
        protocol = Protocol("rayleigh_benard")
        self.assertEqual(protocol.target_offsets, (0, 16, 24, 40))
        self.assertEqual(
            [protocol.horizon(offset) for offset in protocol.target_offsets],
            [0, 8, 16, 32],
        )


class ShardSafetyTests(unittest.TestCase):
    def test_float16_shards_reject_unrepresentable_encoder_outputs(self):
        state = torch.full((2, 3, 4), 1000.0)
        self.assertAlmostEqual(_float16_safe(state, "pooled", 7), 1000.0)
        state[0, 1, 2] = 2 * FLOAT16_MAX
        with self.assertRaisesRegex(ValueError, "pooled encoder output 7"):
            _float16_safe(state, "pooled", 7)
        state[0, 1, 2] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite token encoder output 3"):
            _float16_safe(state, "token", 3)


class PartialFitsTests(unittest.TestCase):
    def test_resume_requires_identical_inputs_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fits"
            identity = dict(checkpoint="abc", probe_settings={"attentive_epochs": 100})
            partial = PartialFits(output, identity)
            self.assertFalse(partial.completed("layer0"))
            partial.publish("layer0", dict(entries={"cell": {"layer": 0}}, plans={}))
            self.assertTrue(partial.completed("layer0"))
            self.assertEqual(
                PartialFits(output, dict(identity)).load("layer0")["entries"],
                {"cell": {"layer": 0}},
            )
            moved = dict(identity, probe_settings={"attentive_epochs": 50})
            with self.assertRaisesRegex(ValueError, "different inputs"):
                PartialFits(output, moved)
            # Only a sealed result clears the partial work.
            self.assertTrue(partial.root.is_dir())
            partial.discard()
            self.assertFalse(partial.root.exists())


if __name__ == "__main__":
    unittest.main()
