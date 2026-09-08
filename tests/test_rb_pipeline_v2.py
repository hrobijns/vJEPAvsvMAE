import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts import rb_eval_v2 as E
from scripts import rb_pipeline_v2 as P


_PERSISTENCE_TARGETS = (
    "enstrophy",
    "buoyancy_gradient_energy",
    "convective_flux",
    "pressure_gradient_magnitude",
    "buoyancy_laplacian_magnitude",
)


def _write_persistence_cache(cache_root: Path) -> None:
    split_root = cache_root / "test"
    target_root = split_root / "original_support"
    target_root.mkdir(parents=True)
    manifest = {"schema_version": P.SCHEMA_VERSION, "split": "test"}
    manifest["manifest_sha256"] = P.canonical_hash(manifest)
    (split_root / "manifest.json").write_text(json.dumps(manifest))
    samples = [
        {"trajectory": 0, "Rayleigh": 1e6, "Prandtl": 1.0},
        {"trajectory": 1, "Rayleigh": 1e7, "Prandtl": 1.0},
    ]
    for representation in ("pooled", "token"):
        (target_root / f"{representation}_samples.json").write_text(json.dumps(samples))
    for target in _PERSISTENCE_TARGETS:
        for representation, current, later in (
            ("pooled", np.array([0.0, 1.0]), np.array([0.0, 2.0])),
            (
                "token",
                np.array([[0.0, 1.0], [2.0, 3.0]]),
                np.array([[0.0, 2.0], [4.0, 6.0]]),
            ),
        ):
            np.save(target_root / f"{representation}_gap0_{target}.npy", current)
            np.save(target_root / f"{representation}_gap8_{target}.npy", current)
            np.save(target_root / f"{representation}_gap32_{target}.npy", later)


def _records():
    rows = []
    for regime in range(35):
        for replicate in range(5):
            rows.append(
                {
                    "trajectory": len(rows),
                    "file": f"rayleigh_{regime}_prandtl_1.hdf5",
                    "Rayleigh": float(10 ** (5 + regime / 10)),
                    "Prandtl": 1.0,
                    "replicate": replicate,
                }
            )
    return rows


class _AddOne(nn.Module):
    def forward(self, x):
        return x + 1


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.embed_dim = 4
        self.blocks = nn.ModuleList([_AddOne(), _AddOne()])
        self.norm = nn.Identity()

    def tokenize(self, clip):
        return clip


class PipelineContractTests(unittest.TestCase):
    def test_validate_records_requires_balanced_official_split(self):
        rows = _records()
        P.validate_official_split(rows)
        with self.assertRaisesRegex(ValueError, "175"):
            P.validate_official_split(rows[:-1])

    def test_sample_tables_have_expected_rows_and_offsets(self):
        rows = _records()
        onsets = np.full(175, 60)
        pooled = P.build_sample_table(rows, onsets, "developed", "pooled")
        token = P.build_sample_table(rows, onsets, "original_support", "token")
        self.assertEqual(len(pooled), 525)
        self.assertEqual(len(token), 175)
        self.assertEqual([r["context_start"] for r in pooled[:3]], [60, 106, 152])
        self.assertEqual([r["context_start"] for r in token[:5]], [0, 13, 26, 40, 53])

    def test_validation_folds_balance_token_times_without_splitting_trajectories(self):
        samples = P.build_sample_table(_records(), np.full(175, 60), "original_support", "token")
        repeated = P._repeat_samples(samples, 64)

        folds = P._balanced_regime_folds(repeated)

        self.assertEqual(len(folds), 5)
        for fold in folds:
            held = [repeated[index] for index in fold["select"]]
            held_trajectories = {row["trajectory"] for row in held}
            self.assertEqual(len(held_trajectories), 35)
            self.assertEqual(
                {time: sum(row["context_start"] == time for row in held) for time in (0, 13, 26, 40, 53)},
                {time: 7 * 64 for time in (0, 13, 26, 40, 53)},
            )
            held_regimes = {(row["Rayleigh"], row["Prandtl"]) for row in held}
            self.assertEqual(len(held_regimes), 35)

    def test_normalization_is_channelwise_and_round_trips(self):
        raw = torch.arange(2 * 4 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 4, 2, 3, 2)
        means = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stds = torch.tensor([2.0, 4.0, 5.0, 10.0])
        normalized = P.normalize_clip(raw, means, stds)
        restored = P.denormalize_clip(normalized, means, stds)
        torch.testing.assert_close(restored, raw)

    def test_layerwise_features_include_each_block_and_final_norm(self):
        encoder = _FakeEncoder()
        clips = torch.zeros(2, 3, 4)
        pooled, tokens = P.layerwise_features(encoder, clips, token_positions=np.array([[0, 2], [1, 2]]))
        self.assertEqual(pooled.shape, (2, 3, 4))
        self.assertEqual(tokens.shape, (2, 2, 3, 4))
        torch.testing.assert_close(pooled[:, 0], torch.ones(2, 4))
        torch.testing.assert_close(pooled[:, 1], torch.full((2, 4), 2.0))

    def test_noise_is_paired_across_repeated_extractions(self):
        clips = torch.zeros(3, 2, 2, 2, 2)
        indices = np.array([10, 11, 12])
        a = P.paired_noise_batch(clips, sigma=0.2, corruption_seed=2, sample_indices=indices)
        b = P.paired_noise_batch(clips, sigma=0.2, corruption_seed=2, sample_indices=indices)
        c = P.paired_noise_batch(clips, sigma=0.2, corruption_seed=1, sample_indices=indices)
        torch.testing.assert_close(a, b)
        self.assertFalse(torch.equal(a, c))

    def test_feature_extraction_writes_noise_for_both_sampling_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pt"
            torch.save({"stub": True}, checkpoint)
            cache_root = root / "cache"
            for split in ("valid", "test"):
                split_root = cache_root / split
                manifest = {"schema_version": P.SCHEMA_VERSION, "split": split}
                manifest["manifest_sha256"] = P.canonical_hash(manifest)
                split_root.mkdir(parents=True)
                (split_root / "manifest.json").write_text(json.dumps(manifest))
                for stratum in ("original_support", "developed"):
                    stratum_root = split_root / stratum
                    stratum_root.mkdir()
                    context = np.zeros((1, 3, 4), dtype=np.float32)
                    np.save(stratum_root / "pooled_context.npy", context)
                    np.save(stratum_root / "token_context.npy", context)
                    np.save(stratum_root / "token_positions.npy", np.zeros((1, 64), dtype=np.int64))

            feature_root = root / "features"
            with (
                mock.patch.object(
                    P,
                    "validate_checkpoint_payload",
                    return_value={"objective": "jepa", "seed": 1, "step": 100_000, "lr": 5e-5},
                ),
                mock.patch("scripts.load_encoder.load_encoder", return_value=(_FakeEncoder(), None, None)),
                mock.patch.object(P, "NOISE_SIGMAS", (0.0,)),
                mock.patch.object(P, "NOISE_SEEDS", (0,)),
            ):
                output = P.extract_checkpoint_features(
                    checkpoint=checkpoint,
                    cache_root=cache_root,
                    feature_root=feature_root,
                    batch_size=1,
                )

            for stratum in ("original_support", "developed"):
                self.assertTrue(
                    (output / "test" / stratum / "pooled_noise_sigma0_seed0.npy").exists()
                )

    def test_clean_fit_noise_evaluation_reports_both_sampling_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp)
            clean_models = {}
            original_target = np.arange(6, dtype=np.float64)
            developed_target = np.array([10, 12, 11, 15, 14, 13], dtype=np.float64)
            fixtures = {
                "original_support": {
                    "layer": 0,
                    "alpha": 0.0,
                    "target": original_target,
                    "features": np.stack(
                        [original_target, np.array([2, 5, 1, 4, 0, 3])], axis=1
                    )[:, :, None],
                },
                "developed": {
                    "layer": 1,
                    "alpha": 0.5,
                    "target": developed_target,
                    "features": np.stack(
                        [np.array([5, 0, 4, 1, 3, 2]), developed_target], axis=1
                    )[:, :, None],
                },
            }
            for stratum, fixture in fixtures.items():
                output_dir = feature_dir / "test" / stratum
                output_dir.mkdir(parents=True)
                for corruption_seed in (0, 1, 2):
                    np.save(
                        output_dir / f"pooled_noise_sigma0_seed{corruption_seed}.npy",
                        fixture["features"].astype(np.float32),
                    )
                clean_models[(stratum, "enstrophy")] = {
                    "layer": fixture["layer"],
                    "alpha": fixture["alpha"],
                    "valid_features": fixture["features"].astype(np.float32),
                    "valid_target": fixture["target"],
                    "test_target": fixture["target"],
                }

            with (
                mock.patch.object(P, "NOISE_SIGMAS", (0.0,)),
                mock.patch.object(P, "NOISE_SEEDS", (0, 1, 2)),
            ):
                rows = P._evaluate_clean_fit_noise(
                    clean_models,
                    feature_dir=feature_dir,
                    objective="jepa",
                    checkpoint_seed=1,
                )

            self.assertEqual(len(rows), 2)
            by_stratum = {row["stratum"]: row for row in rows}
            self.assertEqual(set(by_stratum), {"original_support", "developed"})
            self.assertEqual(by_stratum["original_support"]["selected_layer"], 0)
            self.assertEqual(by_stratum["original_support"]["selected_alpha"], 0.0)
            self.assertAlmostEqual(by_stratum["original_support"]["test_r2"], 1.0)
            self.assertEqual(by_stratum["developed"]["selected_layer"], 1)
            self.assertEqual(by_stratum["developed"]["selected_alpha"], 0.5)
            self.assertAlmostEqual(by_stratum["developed"]["test_r2"], 8.0 / 9.0)
            for stratum, row in by_stratum.items():
                self.assertEqual(row["family"], "noise")
                self.assertEqual(row["method"], "ridge_encoder_clean_fit")
                self.assertEqual(
                    row["cell_id"],
                    "family=noise|method=ridge_encoder_clean_fit|representation=pooled|"
                    f"stratum={stratum}|gap=0|target=enstrophy|sigma=0.0",
                )

    def test_stratified_bootstrap_resamples_trajectory_blocks(self):
        target = np.arange(20, dtype=float)
        prediction = target + np.tile([0.0, 1.0], 10)
        groups = np.repeat(np.arange(10), 2)
        regimes = np.repeat(np.arange(5), 4)
        a = P.stratified_block_bootstrap_r2(prediction, target, groups, regimes, n_boot=20, seed=4)
        b = P.stratified_block_bootstrap_r2(prediction, target, groups, regimes, n_boot=20, seed=4)
        self.assertEqual(a, b)
        self.assertEqual(a["n_boot"], 20)
        self.assertLessEqual(a["ci_low"], a["median"])
        self.assertGreaterEqual(a["ci_high"], a["median"])

    def test_persistence_baseline_uses_current_target_for_future_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_persistence_cache(cache_root)
            output = Path(tmp) / "persistence.json"
            P.compute_persistence_baseline(
                cache_root=cache_root,
                output=output,
                n_boot=5,
            )
            payload = json.loads(output.read_text())

        self.assertEqual(payload["kind"], "rb-persistence-baseline")
        self.assertEqual(len(payload["rows"]), 20)
        self.assertEqual(len(payload["summary"]), 4)
        self.assertEqual({row["gap"] for row in payload["rows"]}, {8, 32})
        self.assertNotIn("deformation_energy", {row["target"] for row in payload["rows"]})
        self.assertTrue(
            all(row["test_r2"] == 1.0 for row in payload["rows"] if row["gap"] == 8)
        )
        pooled = next(
            row
            for row in payload["rows"]
            if row["representation"] == "pooled"
            and row["gap"] == 32
            and row["target"] == "enstrophy"
        )
        token = next(
            row
            for row in payload["rows"]
            if row["representation"] == "token"
            and row["gap"] == 32
            and row["target"] == "enstrophy"
        )
        self.assertAlmostEqual(pooled["test_r2"], 0.5)
        self.assertAlmostEqual(token["test_r2"], 0.3)
        self.assertEqual(pooled["n_test"], 2)
        self.assertEqual(token["n_test"], 4)
        self.assertEqual(pooled["bootstrap"]["n_boot"], 5)
        pooled_summary = next(
            row
            for row in payload["summary"]
            if row["representation"] == "pooled" and row["gap"] == 32
        )
        self.assertAlmostEqual(pooled_summary["five_target_r2_mean"], 0.5)

    def test_persistence_baseline_cli_writes_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            _write_persistence_cache(cache_root)
            output = Path(tmp) / "persistence.json"

            exit_code = E.main(
                [
                    "persistence-baseline",
                    "--cache-root",
                    str(cache_root),
                    "--output",
                    str(output),
                    "--n-boot",
                    "5",
                ]
            )
            payload = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["kind"], "rb-persistence-baseline")
        self.assertEqual(len(payload["rows"]), 20)

    def test_atomic_json_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            P.atomic_write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            with self.assertRaises(FileExistsError):
                P.atomic_write_json(path, {"a": 2})

    def test_ridge_cell_selects_on_valid_and_reports_test_once(self):
        generator = np.random.default_rng(7)
        valid_x = generator.normal(size=(175, 3, 6)).astype(np.float32)
        test_x = generator.normal(size=(175, 3, 6)).astype(np.float32)
        valid_y = valid_x[:, 1, 2] - 0.5 * valid_x[:, 1, 4]
        test_y = test_x[:, 1, 2] - 0.5 * test_x[:, 1, 4]
        rows = _records()
        result, prediction = P.fit_ridge_cell(
            valid_x,
            valid_y,
            test_x,
            test_y,
            valid_samples=rows,
            test_samples=rows,
            n_boot=10,
        )
        self.assertEqual(result["selected_layer"], 1)
        self.assertGreater(result["test_r2"], 0.99)
        self.assertEqual(prediction.shape, test_y.shape)
        self.assertEqual(len(result["depth_curve"]), 3)

    def test_batched_ridge_shares_design_but_selects_each_target_separately(self):
        generator = np.random.default_rng(8)
        valid_x = generator.normal(size=(175, 3, 5)).astype(np.float32)
        test_x = generator.normal(size=(175, 3, 5)).astype(np.float32)
        valid_targets = {"a": valid_x[:, 0, 0], "b": valid_x[:, 2, 3]}
        test_targets = {"a": test_x[:, 0, 0], "b": test_x[:, 2, 3]}
        results, predictions = P.fit_ridge_cells_many(
            valid_x,
            valid_targets,
            test_x,
            test_targets,
            valid_samples=_records(),
            test_samples=_records(),
            n_boot=5,
        )
        self.assertEqual(results["a"]["selected_layer"], 0)
        self.assertEqual(results["b"]["selected_layer"], 2)
        self.assertGreater(results["a"]["test_r2"], 0.99)
        self.assertEqual(set(predictions), {"a", "b"})

    def test_mlp_step_selection_never_returns_before_minimum(self):
        features = torch.linspace(-1.0, 1.0, 50).reshape(-1, 1)
        replicates = np.tile(np.arange(5), 10)
        target = features[:, 0].clone()
        target[torch.as_tensor(replicates == 0)] *= -1

        selected = P._select_mlp_steps(
            features,
            target,
            replicates,
            seed=0,
            hidden=4,
            dropout=0.0,
            lr=1e-2,
            weight_decay=0.0,
            max_steps=100,
            min_steps=60,
            patience=20,
            check_every=10,
        )

        self.assertGreaterEqual(selected, 60)

    def test_probe_family_is_selected_from_validation_not_test(self):
        ridge = {
            "method": "ridge_encoder",
            "selected_layer": 2,
            "valid_cv_r2": 0.61,
            "test_r2": 0.99,
        }
        mlp = {
            "method": "mlp_encoder_selected",
            "selected_layer": 7,
            "valid_cv_r2": 0.72,
            "test_r2": 0.10,
        }

        selected = P.select_probe_family(ridge, mlp)

        self.assertEqual(selected["selected_method"], "mlp_encoder_selected")
        self.assertAlmostEqual(selected["test_r2"], 0.10)
        self.assertEqual(selected["selection_rule"], "maximum validation CV R2")

    def test_aggregate_requires_three_checkpoint_seeds_per_objective(self):
        rows = []
        for objective in ("jepa", "mae"):
            for seed in (1, 2, 3):
                rows.append(
                    {
                        "objective": objective,
                        "checkpoint_seed": seed,
                        "cell_id": "physics|ridge|pooled|developed|gap0|enstrophy",
                        "test_r2": float(seed + (objective == "jepa")),
                    }
                )
        summary = P.aggregate_seed_rows(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["n_checkpoint_seeds"], 3)
        with self.assertRaisesRegex(ValueError, "three checkpoint seeds"):
            P.aggregate_seed_rows(rows[:-1])

    def test_aggregate_reports_deterministic_controls_once(self):
        rows = []
        for objective in ("jepa", "mae"):
            for seed in (1, 2, 3):
                rows.append(
                    {
                        "objective": objective,
                        "checkpoint_seed": seed,
                        "cell_id": "family=physics|method=ridge_nuisance|target=enstrophy",
                        "family": "physics",
                        "method": "ridge_nuisance",
                        "target": "enstrophy",
                        "test_r2": 0.25,
                        "bootstrap": {"ci_low": 0.2, "ci_high": 0.3},
                    }
                )

        summary = P.aggregate_seed_rows(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["objective"], "shared")
        self.assertEqual(summary[0]["test_r2_mean"], 0.25)
        self.assertNotIn("n_checkpoint_seeds", summary[0])
        self.assertNotIn("test_r2_std", summary[0])
        self.assertEqual(summary[0]["bootstrap"], {"ci_low": 0.2, "ci_high": 0.3})

    def test_plot_interface_writes_fixed_products(self):
        summary = []
        per_checkpoint = []
        for objective in ("jepa", "mae"):
            for target in P.PRIMARY_TARGETS:
                summary.append(
                    {
                        "objective": objective,
                        "family": "physics",
                        "method": "ridge_encoder",
                        "representation": "pooled",
                        "stratum": "developed",
                        "gap": 0,
                        "target": target,
                        "test_r2_mean": 0.5,
                        "test_r2_std": 0.1,
                    }
                )
                for seed in (1, 2, 3):
                    per_checkpoint.append(
                        {
                            "objective": objective,
                            "checkpoint_seed": seed,
                            "family": "physics",
                            "method": "ridge_encoder",
                            "representation": "pooled",
                            "stratum": "developed",
                            "gap": 0,
                            "target": target,
                            "depth_curve": [{"test_r2": 0.1 * layer} for layer in range(1, 14)],
                        }
                    )
                for stratum, baseline in (("developed", 0.5), ("original_support", 9.0)):
                    for sigma in P.NOISE_SIGMAS:
                        summary.append(
                            {
                                "objective": objective,
                                "family": "noise",
                                "stratum": stratum,
                                "target": target,
                                "sigma": sigma,
                                "test_r2_mean": baseline - 0.1 * sigma,
                            }
                        )
        with tempfile.TemporaryDirectory() as tmp:
            aggregate = Path(tmp) / "aggregate.json"
            aggregate.write_text(
                json.dumps(
                    {
                        "schema_version": "rb-probe-v2",
                        "kind": "rb-aggregate",
                        "summary": summary,
                        "per_checkpoint_rows": per_checkpoint,
                    }
                )
            )
            output = Path(tmp) / "figures"
            from matplotlib.axes import Axes

            noise_series = []
            real_plot = Axes.plot

            def record_plot(axis, x, y, *args, **kwargs):
                if kwargs.get("marker") == "o":
                    noise_series.append((list(x), list(y)))
                return real_plot(axis, x, y, *args, **kwargs)

            with mock.patch.object(Axes, "plot", record_plot):
                P.plot_aggregate(aggregate, output)
            self.assertTrue((output / "rb_physics_gap0.pdf").exists())
            self.assertTrue((output / "rb_depth_gap0.pdf").exists())
            self.assertTrue((output / "rb_noise.pdf").exists())
            self.assertTrue((output / "plots_manifest.json").exists())
            lines = (output / "rb_summary.tsv").read_text().splitlines()
            header = lines[0].split("\t")
            self.assertIn("sigma", header)
            table = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
            noise_sigmas = {float(row["sigma"]) for row in table if row["family"] == "noise"}
            self.assertEqual(noise_sigmas, set(P.NOISE_SIGMAS))
            self.assertEqual(len(noise_series), 2 * len(P.PRIMARY_TARGETS))
            for sigmas, scores in noise_series:
                self.assertEqual(sigmas, list(P.NOISE_SIGMAS))
                self.assertEqual(scores, [0.5 - 0.1 * sigma for sigma in P.NOISE_SIGMAS])


if __name__ == "__main__":
    unittest.main()
