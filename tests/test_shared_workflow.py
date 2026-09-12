import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.source import WellSource
from src.data.well import (
    ClipSpec,
    WellClipDataset,
    MemmapClipDataset,
    train_valid_trajectory_split,
)
from src.data.preprocess import preprocess
from src.evaluation.artifacts import Artifact, seal, write_json
from src.evaluation.cache import prepare_cache
from src.evaluation.features import extract_features, paired_noise_batch
from src.evaluation.pipeline import fit_probes, score_probes, evaluate_noise, _score_fit
from src.evaluation.selection import select_checkpoints
from src.evaluation.probes import (
    fit_ridge_many,
    fit_mlp,
    predict,
    metrics,
    selected_family,
)
from src.evaluation.protocol import (
    Protocol,
    token_indices,
    position_basis,
)
from src.evaluation.reporting import aggregate, plot
from src.models.vit import build_encoder
from src.objectives import OBJECTIVES
from src.physics.systems import SYSTEMS
from src.physics import rayleigh_benard as rb, rb_derivatives as R
from src.physics.quadrature import quad_weights
from src.train import training_identity, validate_resume
from tests.fixtures import write_well


class PhysicsTests(unittest.TestCase):
    def test_rb_operators_and_quadrature(self):
        np.testing.assert_allclose(
            quad_weights(), quad_weights(method="moment"), atol=1e-15
        )
        x = torch.arange(512, dtype=torch.float64)[:, None] * 4 / 512
        y = R.cheb_gauss_nodes()[None, :]
        f = torch.sin(2 * torch.pi * x / 4) * y**3
        torch.testing.assert_close(
            R.ddx(f),
            2 * torch.pi / 4 * torch.cos(2 * torch.pi * x / 4) * y**3,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            R.ddy(f), 3 * torch.sin(2 * torch.pi * x / 4) * y**2, atol=1e-10, rtol=1e-10
        )
        self.assertAlmostEqual(
            float(rb.volume_mean((y**2).expand(1, 2, 512, 128))), 1 / 3, places=12
        )
        self.assertGreater(abs(float((y**2).mean()) - 1 / 3), 0.01)

    def test_periodic_axes_and_physical_scaling(self):
        for dataset in ("active_matter", "shear_flow"):
            system = SYSTEMS[dataset]
            nx, ny = 32, 48
            x = torch.arange(nx, dtype=torch.float64)[:, None] * system.lengths[0] / nx
            y = torch.arange(ny, dtype=torch.float64)[None, :] * system.lengths[1] / ny
            u = torch.cos(2 * torch.pi * y / system.lengths[1]).expand(nx, ny)
            v = torch.sin(4 * torch.pi * x / system.lengths[0]).expand(nx, ny)
            raw = torch.zeros(1, system.channels, 2, nx, ny, dtype=torch.float64)
            if dataset == "active_matter":
                raw[:, 0] = 1
                raw[:, 3] = raw[:, 6] = 0.5
            raw[:, 1] = u
            raw[:, 2] = v
            expected = (
                4
                * torch.pi
                / system.lengths[0]
                * torch.cos(4 * torch.pi * x / system.lengths[0])
                + 2
                * torch.pi
                / system.lengths[1]
                * torch.sin(2 * torch.pi * y / system.lengths[1])
            ) ** 2
            actual = system.fields(raw, (1, 2))["enstrophy"]
            torch.testing.assert_close(
                actual, expected.expand(1, 2, nx, ny), atol=1e-9, rtol=1e-9
            )
            torch.testing.assert_close(
                system.reduce(actual, (1, 8, 8)).mean(1), system.reduce(actual)
            )

    def test_token_order_and_rb_weights(self):
        field = (
            torch.arange(4 * 32 * 8, dtype=torch.float64)
            .reshape(1, 4, 1, 32, 1, 8, 1)
            .expand(1, 4, 2, 32, 16, 8, 16)
            .reshape(1, 8, 512, 128)
        )
        torch.testing.assert_close(
            rb.weighted_patch_mean(field, (2, 16, 16)),
            torch.arange(1024, dtype=torch.float64)[None],
        )
        self.assertEqual(
            set(rb.PRIMARY_TARGETS),
            {
                "enstrophy",
                "buoyancy_gradient_energy",
                "convective_flux",
                "pressure_gradient_magnitude",
                "buoyancy_laplacian_magnitude",
            },
        )


class ProtocolTests(unittest.TestCase):
    def test_sampling_support_and_shorter_trajectories(self):
        p = Protocol("rayleigh_benard", frame_limit=101)
        self.assertEqual(p.offsets(200, 0)["pooled"], [0, 26, 53])
        self.assertEqual(
            [p.offsets(200, i)["token"][0] for i in range(5)], [0, 13, 26, 40, 53]
        )
        self.assertEqual(p.target_start(53, 40) + 8, 101)
        self.assertEqual(Protocol("active_matter").offsets(81, 4)["token"], [33])
        with self.assertRaises(ValueError):
            p.offsets(30, 0)

    def test_full_trajectory_evaluation_samples(self):
        for dataset in ("rayleigh_benard", "shear_flow"):
            p = Protocol(dataset)
            self.assertIsNone(p.frame_limit)
            self.assertEqual(p.offsets(200, 0)["pooled"], [0, 76, 152])
            self.assertEqual(
                [p.offsets(200, i)["token"][0] for i in range(5)],
                [0, 38, 76, 114, 152],
            )
            self.assertEqual(p.target_start(152, 40) + 8, 200)
            # Full support follows the source length, including beyond 200.
            self.assertEqual(p.target_start(p.offsets(240, 4)["token"][0], 40) + 8, 240)
            self.assertEqual(p.token_samples, 64)
        p = Protocol("active_matter")
        self.assertEqual(p.offsets(81, 0)["pooled"], [0, 16, 33])
        self.assertEqual(p.target_start(33, 40) + 8, 81)

    def test_tokens_noise_and_position_coordinates(self):
        a = token_indices(3, 1024)
        self.assertTrue(np.array_equal(a, token_indices(3, 1024)))
        self.assertFalse(np.array_equal(a, token_indices(4, 1024)))
        x = torch.zeros(3, 4, 2, 8, 8)
        full = paired_noise_batch(x, 0.1, 2, [0, 1, 2])
        batches = torch.cat(
            [
                paired_noise_batch(x[:1], 0.1, 2, [0]),
                paired_noise_batch(x[1:], 0.1, 2, [1, 2]),
            ]
        )
        torch.testing.assert_close(full, batches, atol=0, rtol=0)
        np.testing.assert_array_equal(
            position_basis(np.array([0, 1023]), (4, 32, 8))[:, 1:4],
            [[-1, -1, -1], [1, 1, 1]],
        )


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(71)
        self.x = self.rng.normal(size=(25, 2, 6)).astype("float32")
        self.xt = self.rng.normal(size=(25, 2, 6)).astype("float32")
        self.y = 2 * self.x[:, 1, 0].astype("float64") + 1

    def test_ridge_heldout_predictions_and_selection(self):
        fits = fit_ridge_many(
            self.x, {"y": self.y}, self.xt, {"y": 2 * self.xt[:, 1, 0] + 1}
        )["y"]
        self.assertEqual(fits["selected_layer"], 1)
        entry = fits["layers"][1]
        predicted = predict(entry["fit"], self.xt[:, 1])
        self.assertGreater(
            metrics(predicted, 2 * self.xt[:, 1, 0] + 1)["test_r2"], 0.999
        )
        # Test labels enter scoring only; selection APIs do not accept them.
        selected = selected_family(
            {"valid_vrmse": 0.2, "method": "ridge", "test_r2": -10},
            {"valid_vrmse": 0.3, "method": "mlp", "test_r2": 1},
        )
        self.assertEqual(selected["method"], "ridge")
        self.assertEqual(
            selected_family(
                {"valid_vrmse": 0.2, "method": "ridge"},
                {"valid_vrmse": 0.2, "method": "mlp"},
            )["method"],
            "ridge",
        )

    def test_saved_mlp_replay_for_low_variance_targets(self):
        for scale in (1.0, 1e-7, 1e-10):
            y = 1 + scale * self.y
            fitted = fit_mlp(
                self.x,
                y,
                self.xt,
                1 + scale * (2 * self.xt[:, 1, 0].astype("float64") + 1),
                max_steps=4,
                min_steps=2,
            )
            entry = next(
                r for r in fitted["layers"] if r["layer"] == fitted["selected_layer"]
            )
            expected = predict(entry["fit"], self.xt[:, entry["layer"]])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "probe.pt"
                torch.save(entry["fit"], path)
                loaded = torch.load(path, weights_only=False)
                np.testing.assert_array_equal(
                    predict(loaded, self.xt[:, entry["layer"]]), expected
                )
            self.assertTrue(
                np.isfinite(
                    metrics(
                        expected,
                        1 + scale * (2 * self.xt[:, 1, 0].astype("float64") + 1),
                    )["test_r2"]
                )
            )

    def test_constant_targets_are_explicitly_undefined(self):
        fit = fit_ridge_many(
            self.x, {"constant": np.ones(25)}, self.xt, {"constant": np.ones(25)}
        )["constant"]
        self.assertEqual(fit["status"], "undefined_validation_vrmse")
        self.assertEqual(
            metrics(np.zeros(3), np.ones(3))["metric_status"], "constant_target"
        )

    def test_selected_only_mlp_does_not_claim_a_depth_curve(self):
        fitted = fit_mlp(
            self.x,
            self.y,
            self.xt,
            2 * self.xt[:, 1, 0] + 1,
            max_steps=2,
            min_steps=2,
            include_depth=False,
            candidate_layers=(1,),
        )
        row, state = _score_fit(fitted, self.xt, self.y, {}, "mlp")
        self.assertEqual(row["selected_layer"], 1)
        self.assertEqual(row["depth_curve"], [])
        self.assertIsNotNone(state)


class ArtifactTests(unittest.TestCase):
    def test_aggregation_requires_matching_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, seed, step in (
                ("a", 1, 25000),
                ("b", 2, 25000),
                ("c", 2, 100000),
            ):
                path = root / name
                path.mkdir()
                write_json(
                    path / "rows.json",
                    [dict(cell_id="x", family="regime", target="x", test_r2=0.5)],
                )
                seal(
                    path,
                    "probes",
                    checkpoint=dict(
                        objective="jepa",
                        seed=seed,
                        step=step,
                        training_protocol={"total_steps": 100000},
                    ),
                    protocol={},
                    caches={},
                    probe_settings={},
                )
                paths.append(path)
            # Matching intermediate checkpoints are valid despite a longer plan.
            aggregate(paths[:2], root / "matched", objectives=("jepa",), seeds=(1, 2))
            with self.assertRaisesRegex(ValueError, "incompatible checkpoint step"):
                aggregate(
                    [paths[0], paths[2]],
                    root / "mixed",
                    objectives=("jepa",),
                    seeds=(1, 2),
                )

    def test_seed_summaries_preserve_selection_and_scientific_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed, scores in ((1, (0.1, 0.3)), (2, (0.5, 0.7))):
                path = root / f"run{seed}"
                path.mkdir()
                rows = [
                    dict(
                        cell_id=target,
                        family="physics",
                        representation="token",
                        method="mlp",
                        target_offset=0,
                        target=target,
                        test_r2=score,
                        selected_layer=seed,
                        depth_curve=[],
                        valid_r2=score + 0.1,
                    )
                    for target, score in zip(("a", "b"), scores)
                ]
                rows.append(
                    dict(
                        cell_id="control",
                        family="physics",
                        representation="token",
                        method="position",
                        target_offset=0,
                        target="a",
                        test_r2=0.05,
                        shared=True,
                    )
                )
                write_json(path / "rows.json", rows)
                seal(
                    path,
                    "probes",
                    checkpoint={"objective": "jepa", "seed": seed},
                    protocol={},
                    caches={},
                    probe_settings={},
                )
                paths.append(path)
            out = root / "out"
            aggregate(paths, out, objectives=("jepa",), seeds=(1, 2))
            result = Artifact(out, "aggregate")
            summaries = {r["cell_id"]: r for r in result.json("summary.json")}
            self.assertNotIn("selected_layer", summaries["a"])
            self.assertEqual(
                [v["selected_layer"] for v in summaries["a"]["selections"]], [1, 2]
            )
            self.assertEqual(summaries["control"]["metrics"]["test_r2"]["n"], 1)
            avg = result.json("target_means.json")[0]["metrics"]["test_r2"]
            self.assertAlmostEqual(avg["mean"], 0.4)
            self.assertAlmostEqual(avg["std"], np.std([0.2, 0.6], ddof=1))
            self.assertEqual(avg["n"], 2)

    def test_array_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "x.npy", np.arange(10))
            seal(root, "cache", protocol={})
            np.save(root / "x.npy", np.arange(10)[::-1])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Artifact(root, "cache").array("x.npy")
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["protocol"] = {"changed": True}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                Artifact(root, "cache")

    def test_aggregation_rejects_incompatible_and_missing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed in (1, 2):
                path = root / f"run{seed}"
                path.mkdir()
                write_json(
                    path / "rows.json",
                    [
                        {
                            "cell_id": "x",
                            "family": "physics",
                            "representation": "pooled",
                            "method": "ridge",
                            "target_offset": 0,
                            "target": "x",
                            "test_r2": 0.5,
                        }
                    ],
                )
                seal(
                    path,
                    "probes",
                    checkpoint={"objective": "jepa", "seed": seed},
                    protocol={},
                    caches={"test": str(seed)},
                    probe_settings={},
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "caches"):
                aggregate(paths, root / "out", objectives=("jepa",), seeds=(1, 2))
            with self.assertRaisesRegex(ValueError, "roster"):
                aggregate(paths[:1], root / "out", objectives=("jepa",), seeds=(1, 2))
            with self.assertRaisesRegex(ValueError, "roster"):
                aggregate(
                    [paths[0], paths[0]],
                    root / "out",
                    objectives=("jepa",),
                    seeds=(1, 2),
                )


class WorkflowTests(unittest.TestCase):
    def test_training_backends_match_and_bound_frames(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            write_well(tmp, "shear_flow", "train", frames=200)
            preprocess(tmp, "shear_flow")
            hdf = WellClipDataset(tmp, "shear_flow", frame_limit=101)
            mm = MemmapClipDataset(tmp, "shear_flow", "train", frame_limit=101)
            self.assertEqual(len(mm), 5 * 94)
            self.assertEqual(hdf.identity, mm.identity)
            for i in (0, 93, 94, 469):
                torch.testing.assert_close(
                    hdf[i]["clip"], mm[i]["clip"], rtol=0, atol=0
                )
            full_hdf = WellClipDataset(tmp, "shear_flow")
            full_mm = MemmapClipDataset(tmp, "shear_flow", "train")
            self.assertEqual(len(full_mm), 5 * 193)
            self.assertEqual(len(full_hdf), len(full_mm))
            self.assertEqual(full_mm.window(192), (0, 192))
            self.assertEqual(full_mm.window(193), (1, 0))
            self.assertEqual(full_mm.window(964), (4, 192))
            for i in (0, 101, 192, 193, 964):
                torch.testing.assert_close(
                    full_hdf[i]["clip"], full_mm[i]["clip"], rtol=0, atol=0
                )
            self.assertFalse(torch.equal(full_mm[93]["clip"], full_mm[192]["clip"]))
            for frames in (81, 200):
                paired_hdf = WellClipDataset(
                    tmp, "shear_flow", frame_limit=frames, future=True
                )
                paired_mm = MemmapClipDataset(
                    tmp, "shear_flow", "train", frame_limit=frames, future=True
                )
                count = frames - 15
                self.assertEqual(len(paired_hdf), 5 * count)
                self.assertEqual(len(paired_mm), len(paired_hdf))
                self.assertEqual(paired_hdf.spec.n_frames, 8)
                self.assertEqual(paired_hdf.window(count - 1), (0, frames - 16))
                self.assertEqual(paired_hdf.window(count), (1, 0))
                for index in (0, count - 1, count, len(paired_hdf) - 1):
                    trajectory, start = paired_hdf.window(index)
                    pair = paired_hdf[index]
                    for key, offset in (("clip", 0), ("target_clip", 8)):
                        torch.testing.assert_close(
                            pair[key], paired_mm[index][key], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            pair[key],
                            full_mm[trajectory * 193 + start + offset]["clip"],
                            rtol=0,
                            atol=0,
                        )
                with self.assertRaises(IndexError):
                    paired_hdf[len(paired_hdf)]
            for backend in (WellClipDataset, MemmapClipDataset):
                with self.assertRaises(ValueError):
                    backend(tmp, "shear_flow", "train", frame_limit=15, future=True)
                self.assertEqual(
                    len(
                        backend(tmp, "shear_flow", "train", frame_limit=16, future=True)
                    ),
                    5,
                )
            fit, held = train_valid_trajectory_split(5)
            self.assertFalse(set(fit) & set(held))
            with self.assertRaises(IndexError):
                mm[len(mm)]
            with self.assertRaises(FileExistsError):
                preprocess(tmp, "shear_flow")
            meta = Path(tmp) / "memmap/shear_flow/train.meta.json"
            value = json.loads(meta.read_text())
            value["shape"][2] = 101
            meta.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                MemmapClipDataset(tmp, "shear_flow", "train")

    def test_resume_rejects_changed_temporal_support(self):
        from types import SimpleNamespace

        config = dict(
            data={"n_frames": 8, "frame_limit": 101, "valid_stride": 8},
            objective_name="jepa",
            objective={},
            encoder={},
            optim={},
            seed=1,
        )
        dataset = SimpleNamespace(identity="data")
        identity = training_identity(config, dataset, [1], [0])
        validate_resume({"training_identity": identity}, identity)
        for objective in OBJECTIVES[1:]:
            config["objective_name"] = objective
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_resume(
                    {"training_identity": identity},
                    training_identity(config, dataset, [1], [0]),
                )
        config["objective_name"] = "jepa"
        config["data"]["frame_limit"] = None
        with self.assertRaises(ValueError):
            validate_resume(
                {"training_identity": identity},
                training_identity(config, dataset, [1], [0]),
            )

    def test_all_systems_use_the_same_end_to_end_stages(self):
        for dataset in SYSTEMS:
            with (
                self.subTest(dataset=dataset),
                tempfile.TemporaryDirectory() as tmp,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                root = Path(tmp)
                for split in ("train", "valid"):
                    write_well(root, dataset, split)
                patch_size = (1, 64, 32) if dataset == "rayleigh_benard" else (1, 8, 8)
                protocol = Protocol(
                    dataset,
                    frame_limit=6,
                    n_frames=2,
                    target_offsets=(0, 2, 3),
                    patch=patch_size,
                    token_samples=4,
                    noise_sigmas=(0.0, 0.1),
                    noise_seeds=(0,),
                )
                for split in ("train", "valid"):
                    prepare_cache(root, split, root / "cache", protocol)
                source = WellSource(root, dataset, "valid", n_frames=2)
                spec = ClipSpec(len(source.channels), 2, *source.shape)
                cfg = dict(
                    data={"dataset_name": dataset},
                    objective_name="mae",
                    seed=0,
                    optim={"total_steps": 2},
                    encoder=dict(
                        patch_t=1,
                        patch_h=patch_size[1],
                        patch_w=patch_size[2],
                        embed_dim=16,
                        depth=1,
                        num_heads=2,
                    ),
                )
                checkpoint = root / "encoder.pt"
                encoder = build_encoder(spec, cfg["encoder"])
                torch.save(
                    dict(
                        encoder=encoder.state_dict(),
                        config=cfg,
                        spec=asdict(spec),
                        step=1,
                    ),
                    checkpoint,
                )
                for split in ("train", "valid"):
                    features = extract_features(
                        checkpoint,
                        root / "cache",
                        root / "features",
                        split,
                        batch_size=5,
                    )
                fit_probes(
                    features,
                    root / "cache",
                    root / "fits",
                    mlp_max_steps=2,
                    mlp_min_steps=2,
                )
                self.assertFalse((root / "cache/test").exists())
                select_checkpoints([root / "fits"], root / "selection")
                write_well(root, dataset, "test")
                prepare_cache(root, "test", root / "cache", protocol)
                extract_features(
                    checkpoint,
                    root / "cache",
                    root / "features",
                    "test",
                    batch_size=5,
                    include_noise=True,
                )
                score_probes(
                    features,
                    root / "cache",
                    root / "fits",
                    root / "selection",
                    root / "probes",
                )
                evaluate_noise(
                    features, root / "cache", root / "probes", root / "noise"
                )
                aggregate(
                    [root / "probes"],
                    root / "aggregate",
                    objectives=("mae",),
                    seeds=(0,),
                )
                aggregate(
                    [root / "noise"],
                    root / "noise_aggregate",
                    objectives=("mae",),
                    seeds=(0,),
                    kind="noise",
                )
                if dataset == "shear_flow":
                    plot(root / "aggregate", root / "plots")
                    plot(root / "noise_aggregate", root / "noise_plots")
                    self.assertTrue((root / "plots/summary.tsv").is_file())
                self.assertTrue((root / "aggregate/target_means.json").is_file())
                rows = Artifact(root / "probes", "probes").json("rows.json")
                for row in rows:
                    if row["method"] == "selected":
                        self.assertNotIn("depth_curve", row)
                    elif row["family"] == "physics" and row["method"] in (
                        "ridge",
                        "mlp",
                    ):
                        self.assertEqual(
                            [p["layer"] for p in row["depth_curve"]],
                            list(range(cfg["encoder"]["depth"] + 1)),
                        )
                self.assertEqual(
                    {r["target"] for r in rows if r["family"] == "physics"},
                    set(SYSTEMS[dataset].targets),
                )
                self.assertTrue(any(r["method"] == "persistence" for r in rows))


if __name__ == "__main__":
    unittest.main()
