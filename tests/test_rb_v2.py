import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts import rb_eval_v2 as V2
from scripts import rb_targets_v2 as T


def regime_records():
    records = []
    for ra in (1e6, 1e7, 1e8, 1e9, 1e10):
        for pr in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            for replicate in range(5):
                records.append(
                    {
                        "trajectory": len(records),
                        "Rayleigh": ra,
                        "Prandtl": pr,
                        "replicate": replicate,
                    }
                )
    return records


class SamplingTests(unittest.TestCase):
    def test_original_and_developed_offsets_share_all_horizons(self):
        self.assertEqual(V2.pooled_offsets("original_support", onset=99), [0, 26, 53])
        self.assertEqual(V2.pooled_offsets("developed", onset=40), [40, 96, 152])
        for offset in V2.pooled_offsets("original_support", onset=99):
            self.assertLessEqual(V2.target_start(offset, 32) + 8, 101)
        for offset in V2.pooled_offsets("developed", onset=40):
            self.assertLessEqual(V2.target_start(offset, 32) + 8, 200)

    def test_token_offsets_cover_five_temporal_quantiles_per_regime(self):
        got = [V2.token_offset("original_support", onset=0, replicate=i) for i in range(5)]
        self.assertEqual(got, [0, 13, 26, 40, 53])

    def test_regime_folds_hold_out_one_run_per_regime(self):
        records = regime_records()
        folds = V2.regime_folds(records)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertEqual(len(fold["fit"]), 140)
            self.assertEqual(len(fold["select"]), 35)
            held_regimes = {
                (records[i]["Rayleigh"], records[i]["Prandtl"])
                for i in fold["select"]
            }
            self.assertEqual(len(held_regimes), 35)

    def test_token_subsample_is_unique_deterministic_and_trajectory_specific(self):
        a = V2.token_indices(trajectory=3, n_tokens=1024, n_select=64)
        b = V2.token_indices(trajectory=3, n_tokens=1024, n_select=64)
        c = V2.token_indices(trajectory=4, n_tokens=1024, n_select=64)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(np.unique(a)), 64)
        self.assertFalse(np.array_equal(a, c))

    def test_manifest_hash_is_order_stable(self):
        one = {"b": [2, 3], "a": {"x": 1}}
        two = {"a": {"x": 1}, "b": [2, 3]}
        self.assertEqual(V2.canonical_hash(one), V2.canonical_hash(two))


class TargetTests(unittest.TestCase):
    def test_volume_mean_respects_chebyshev_cell_volume(self):
        y = torch.as_tensor(T.R.cheb_gauss_nodes(), dtype=torch.float64)
        field = (y**2).reshape(1, 1, 1, -1).expand(1, 2, 3, -1)
        got = T.volume_mean(field)
        self.assertAlmostEqual(float(got[0]), 1.0 / 3.0, places=11)

    def test_weighted_patch_map_matches_encoder_token_order_and_constant(self):
        field = torch.ones(1, 8, 512, 128, dtype=torch.float64)
        got = T.weighted_patch_mean(field, (2, 16, 16))
        self.assertEqual(got.shape, (1, 1024))
        torch.testing.assert_close(got, torch.ones_like(got))

    def test_target_schema_excludes_invalid_quantities(self):
        self.assertEqual(
            set(T.PRIMARY_TARGETS),
            {
                "enstrophy",
                "buoyancy_gradient_energy",
                "convective_flux",
                "pressure_gradient_magnitude",
                "buoyancy_laplacian_magnitude",
                "deformation_energy",
            },
        )
        self.assertEqual(set(T.TOKEN_ONLY_TARGETS), {"vorticity_signed", "okubo_weiss_signed"})
        invalid = {"divergence", "velocity_buoyancy_coherence", "okubo_weiss"}
        self.assertTrue(invalid.isdisjoint(T.PRIMARY_TARGETS))

    def test_onset_requires_sustained_half_plateau(self):
        energy = np.r_[np.zeros(20), np.full(2, 0.7), np.zeros(3), np.ones(75)]
        onset = T.detect_onset_from_energy(energy, threshold=0.5, smooth=1, sustain=5)
        self.assertEqual(onset, 25)


class ProbeTests(unittest.TestCase):
    def test_ridge_selection_uses_folds_and_finds_informative_layer(self):
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(25), 2)
        y = torch.tensor(np.repeat(np.linspace(-2, 2, 25), 2), dtype=torch.float64)
        bad = torch.tensor(rng.normal(size=(50, 3)), dtype=torch.float64)
        good = torch.stack([y, y**2, torch.ones_like(y)], dim=1)
        folds = V2.group_folds(groups, n_folds=5)
        selected = V2.select_ridge_cv([bad, good], y, folds, alphas=(1e-4, 1e-2))
        self.assertEqual(selected["layer"], 1)
        self.assertGreater(selected["cv_r2"], 0.99)

    def test_nuisance_basis_has_fixed_documented_columns(self):
        x = V2.nuisance_basis(
            rayleigh=np.array([1e6, 1e8]),
            prandtl=np.array([0.1, 1.0]),
            age=np.array([0.0, 1.0]),
        )
        self.assertEqual(x.shape, (2, 10))
        self.assertTrue(np.allclose(x[:, 0], 1.0))

    def test_noise_is_paired_by_seed(self):
        clip = torch.zeros(2, 4, 8, 3, 2)
        a = V2.add_standardized_noise(clip, sigma=0.2, seed=7)
        b = V2.add_standardized_noise(clip, sigma=0.2, seed=7)
        c = V2.add_standardized_noise(clip, sigma=0.2, seed=8)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))


class CheckpointTests(unittest.TestCase):
    def _checkpoint(self, objective, seed):
        objective_cfg = {"mask_ratio": 0.9}
        if objective == "jepa":
            objective_cfg.update(
                {
                    "predictor_dim": 192,
                    "predictor_depth": 6,
                    "predictor_heads": 6,
                    "ema_start": 0.996,
                    "ema_end": 1.0,
                }
            )
            lr = 5e-5
        else:
            objective_cfg.update(
                {
                    "decoder_dim": 192,
                    "decoder_depth": 4,
                    "decoder_heads": 6,
                    "norm_pix": True,
                }
            )
            lr = 1e-4
        return {
            "config": {
                "objective_name": objective,
                "seed": seed,
                "data": {"dataset_name": "rayleigh_benard", "n_frames": 8},
                "encoder": {
                    "patch_t": 2,
                    "patch_h": 16,
                    "patch_w": 16,
                    "embed_dim": 384,
                    "depth": 12,
                    "num_heads": 6,
                },
                "objective": objective_cfg,
                "optim": {"lr": lr, "total_steps": 100_000},
            },
            "step": 100_000,
            "spec": {"n_channels": 4, "n_frames": 8, "height": 512, "width": 128},
            "encoder": {},
        }

    def test_checkpoint_contract_accepts_final_tuned_endpoint(self):
        meta = V2.validate_checkpoint_payload(self._checkpoint("jepa", 2))
        self.assertEqual(meta["objective"], "jepa")
        self.assertEqual(meta["seed"], 2)

    def test_checkpoint_contract_rejects_old_wide_predictor(self):
        payload = self._checkpoint("jepa", 1)
        payload["config"]["objective"]["predictor_dim"] = 384
        with self.assertRaisesRegex(ValueError, "predictor_dim"):
            V2.validate_checkpoint_payload(payload)

    def test_checkpoint_contract_rejects_wrong_head_depth(self):
        payload = self._checkpoint("mae", 1)
        payload["config"]["objective"]["decoder_depth"] = 6
        with self.assertRaisesRegex(ValueError, "decoder_depth"):
            V2.validate_checkpoint_payload(payload)

    def test_result_envelope_records_schema_and_manifest_hash(self):
        result = V2.result_envelope({"x": 1}, manifest={"samples": [1, 2]})
        self.assertEqual(result["schema_version"], "rb-probe-v2")
        self.assertEqual(result["manifest_sha256"], V2.canonical_hash({"samples": [1, 2]}))
        json.dumps(result)

    def test_cli_exposes_every_rerun_stage(self):
        parser = V2.build_parser()
        actions = next(action for action in parser._actions if action.dest == "command")
        self.assertTrue(
            {
                "validate-checkpoints",
                "prepare-cache",
                "extract-features",
                "fit-probes",
                "aggregate",
                "plot",
            }.issubset(actions.choices)
        )

    def test_cli_does_not_expose_incomplete_no_noise_mode(self):
        parser = V2.build_parser()
        commands = next(action for action in parser._actions if action.dest == "command")
        extract = commands.choices["extract-features"]
        options = {option for action in extract._actions for option in action.option_strings}
        self.assertNotIn("--no-noise", options)

    def test_direct_cli_execution_imports_pipeline(self):
        root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "rb_eval_v2.py"),
                    "fit-probes",
                    "--feature-dir",
                    str(Path(tmp) / "missing-features"),
                    "--cache-root",
                    str(Path(tmp) / "cache"),
                    "--output",
                    str(Path(tmp) / "result.json"),
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest.json", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
