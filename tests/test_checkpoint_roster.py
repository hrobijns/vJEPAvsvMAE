"""The frozen roster is one best-validation encoder per training run."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.artifacts import provenance, seal, sha256_file, write_json
from src.evaluation.protocol import Protocol
from src.objectives import OBJECTIVES

REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "checkpoints/iclr2027/seed1"
# Rayleigh-Bénard best-validation steps recorded in the seed-1 handoff index.
RAYLEIGH_BENARD_STEPS = {
    "jepa": 100000,
    "mae": 100000,
    "jepa_future": 98000,
    "mae_future": 100000,
}


def sweep():
    spec = importlib.util.spec_from_file_location(
        "probe_sweep", REPO / "scripts/probe_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_row(objective, label, step, seed=1, total=100000, **overrides):
    return dict(
        path=f"shear_flow/{objective}/encoder_{label}.pt",
        dataset="shear_flow",
        objective=objective,
        seed=seed,
        step=step,
        candidate=label,
        bytes=1,
        sha256=f"{objective}_{seed}_{label}_checkpoint",
        encoder_state_sha256=f"{objective}_{seed}_{label}_state",
        config_sha256=f"{objective}_config",
        spec={"n_channels": 4, "n_frames": 8, "height": 256, "width": 512},
        training_protocol={"total_steps": total, "batch_size": 64},
        training_identity=f"{objective}_{seed}_identity",
        data_exposure={"clips_processed": step},
    ) | overrides


def synthetic_index(objectives=OBJECTIVES, seeds=(1,), milestones=True):
    """A handoff may still retain milestone files; only best_val is probed."""
    rows = []
    for seed in seeds:
        for objective in objectives:
            if milestones:
                for percent in (25, 50, 75, 100):
                    rows.append(
                        index_row(
                            objective, f"{percent:03d}pct", percent * 1000, seed=seed
                        )
                    )
            rows.append(index_row(objective, "best_val", 64000, seed=seed))
    return dict(training_provenance={}, checkpoints=rows)


class BestValidationRosterTests(unittest.TestCase):
    def setUp(self):
        self.sweep = sweep()

    def test_seed1_handoff_freezes_one_best_validation_encoder_per_run(self):
        index = json.loads((HANDOFF / "index.json").read_text())
        encoders = self.sweep.frozen_encoders(HANDOFF, index)
        groups = self.sweep.roster_groups(encoders)
        self.assertEqual((len(encoders), len(groups)), (12, 12))
        self.assertEqual({r["candidate"] for r in encoders}, {"best_val"})
        self.assertEqual(len({r["checkpoint_sha256"] for r in encoders}), 12)
        self.assertEqual(
            {r["dataset"] for r in encoders},
            {"rayleigh_benard", "active_matter", "shear_flow"},
        )
        self.assertEqual({r["seed"] for r in encoders}, {1})
        for row in encoders:
            self.assertTrue(row["checkpoint"].endswith("/encoder_best_val.pt"))
            self.assertTrue(0 < row["step"] <= row["total_steps"])

    def test_milestone_rows_are_ignored_when_present_or_absent(self):
        self.assertEqual(
            len(self.sweep.frozen_encoders("/handoff", synthetic_index())), 4
        )
        without = synthetic_index(milestones=False)
        self.assertEqual(len(self.sweep.frozen_encoders("/handoff", without)), 4)

    def test_dataset_scope_freezes_only_the_requested_complete_system(self):
        index = json.loads((HANDOFF / "index.json").read_text())
        encoders = self.sweep.frozen_encoders(
            HANDOFF, index, dataset="rayleigh_benard"
        )
        self.assertEqual(len(self.sweep.roster_groups(encoders)), 4)
        self.assertEqual({row["dataset"] for row in encoders}, {"rayleigh_benard"})
        self.assertEqual({row["objective"] for row in encoders}, set(OBJECTIVES))
        self.assertEqual(
            {row["objective"]: row["step"] for row in encoders},
            RAYLEIGH_BENARD_STEPS,
        )
        for row in encoders:
            self.assertTrue(
                row["checkpoint"].endswith(
                    f"rayleigh_benard/{row['objective']}/encoder_best_val.pt"
                )
            )

    def test_dataset_scope_rejects_a_dataset_absent_from_the_handoff(self):
        with self.assertRaisesRegex(ValueError, "is absent from the handoff"):
            self.sweep.frozen_encoders(
                "/handoff", synthetic_index(), dataset="rayleigh_benard"
            )

    def test_a_run_must_declare_exactly_one_best_validation_encoder(self):
        missing = synthetic_index()
        missing["checkpoints"] = [
            r
            for r in missing["checkpoints"]
            if not (r["objective"] == "mae" and r["candidate"] == "best_val")
        ]
        with self.assertRaisesRegex(ValueError, "declares 0 best_val encoders"):
            self.sweep.frozen_encoders("/handoff", missing)

        duplicate = synthetic_index()
        duplicate["checkpoints"].append(
            index_row("mae", "best_val", 64000, sha256="second_best_val")
        )
        with self.assertRaisesRegex(ValueError, "declares 2 best_val encoders"):
            self.sweep.frozen_encoders("/handoff", duplicate)

        with self.assertRaisesRegex(ValueError, "no checkpoints"):
            self.sweep.frozen_encoders("/handoff", dict(checkpoints=[]))

    def test_every_observed_dataset_and_seed_needs_all_four_objectives(self):
        index = synthetic_index()
        index["checkpoints"] = [
            r for r in index["checkpoints"] if r["objective"] != "mae_future"
        ]
        with self.assertRaisesRegex(
            ValueError, r"incomplete objective roster for \('shear_flow', 1\)"
        ):
            self.sweep.frozen_encoders("/handoff", index)

        # A later seed is validated on its own; no seed is hard-coded.
        self.assertEqual(
            len(self.sweep.frozen_encoders("/handoff", synthetic_index(seeds=(1, 2)))),
            8,
        )
        partial = synthetic_index(seeds=(1, 2))
        partial["checkpoints"] = [
            r
            for r in partial["checkpoints"]
            if r["seed"] == 1 or r["objective"] in ("jepa", "mae")
        ]
        with self.assertRaisesRegex(
            ValueError, r"incomplete objective roster for \('shear_flow', 2\)"
        ):
            self.sweep.frozen_encoders("/handoff", partial)

    def test_best_validation_step_must_be_inside_the_training_budget(self):
        for step in (0, 100001):
            index = synthetic_index()
            for row in index["checkpoints"]:
                if row["objective"] == "jepa" and row["candidate"] == "best_val":
                    row["step"] = step
            with self.assertRaisesRegex(
                ValueError, f"best_val encoder .* is at step {step}"
            ):
                self.sweep.frozen_encoders("/handoff", index)

        absent = synthetic_index()
        for row in absent["checkpoints"]:
            row["training_protocol"]["total_steps"] = None
        with self.assertRaisesRegex(ValueError, "missing training budget"):
            self.sweep.frozen_encoders("/handoff", absent)

    def test_inconsistent_run_metadata_in_the_handoff_is_rejected(self):
        for field, value in (
            ("config_sha256", "other_config"),
            ("training_identity", "other_identity"),
            ("spec", {"n_channels": 3}),
            ("training_protocol", {"total_steps": 100000, "batch_size": 32}),
        ):
            index = synthetic_index()
            for row in index["checkpoints"]:
                if row["objective"] == "jepa" and row["candidate"] == "100pct":
                    row[field] = value
            with self.assertRaisesRegex(ValueError, f"handoff {field} differs"):
                self.sweep.frozen_encoders("/handoff", index)

    def test_frozen_encoders_keep_size_exposure_and_run_identity(self):
        index = json.loads((HANDOFF / "index.json").read_text())
        row = next(
            r
            for r in self.sweep.frozen_encoders(HANDOFF, index, dataset="rayleigh_benard")
            if r["objective"] == "jepa"
        )
        source = next(
            r
            for r in index["checkpoints"]
            if r["dataset"] == "rayleigh_benard"
            and r["objective"] == "jepa"
            and r["candidate"] == "best_val"
        )
        for field in ("bytes", "spec", "training_identity", "data_exposure"):
            self.assertEqual(row[field], source[field])
        self.assertEqual(row["training_protocol"], source["training_protocol"])
        self.assertEqual(row["total_steps"], source["training_protocol"]["total_steps"])

    def test_frozen_protocol_comes_from_the_committed_evaluation_config(self):
        protocol = self.sweep.frozen_protocol(REPO, "rayleigh_benard")
        self.assertEqual(protocol.target_offsets, (0, 16, 24, 40))
        self.assertEqual((protocol.n_frames, protocol.patch), (8, (2, 16, 16)))
        self.assertIsNone(protocol.frame_limit)
        self.assertEqual(protocol.token_samples, 64)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "configs").mkdir()
            (repo / "configs/eval_shear_flow.yaml").write_text(
                "dataset: shear_flow\nn_frames: 8\n"
            )
            with self.assertRaisesRegex(ValueError, "explicit frame_limit"):
                self.sweep.frozen_protocol(repo, "shear_flow")
            (repo / "configs/eval_active_matter.yaml").write_text(
                "dataset: active_matter\nframe_limit: null\nbatch_size: 4\n"
            )
            with self.assertRaisesRegex(ValueError, "only supported protocol fields"):
                self.sweep.frozen_protocol(repo, "active_matter")

    def test_encoder_temporal_support_must_match_the_frozen_protocol(self):
        protocol = self.sweep.frozen_protocol(REPO, "rayleigh_benard")
        self.sweep.check_input_geometry(protocol, [dict(id="x", spec={"n_frames": 8})])
        with self.assertRaisesRegex(ValueError, "takes 4-frame clips"):
            self.sweep.check_input_geometry(
                protocol, [dict(id="x", spec={"n_frames": 4})]
            )

    def test_payload_verification_rejects_missing_short_and_altered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder_best_val.pt"
            row = dict(checkpoint=str(path), bytes=4, checkpoint_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "frozen encoder is missing"):
                self.sweep.verify_payloads([row])
            path.write_bytes(b"lfs")
            with self.assertRaisesRegex(ValueError, "not the declared 4 bytes"):
                self.sweep.verify_payloads([row])
            path.unlink()
            path.write_bytes(b"real")
            with self.assertRaisesRegex(ValueError, "content differs"):
                self.sweep.verify_payloads([row])
            self.sweep.verify_payloads(
                [dict(row, checkpoint_sha256=sha256_file(path))]
            )


def frozen_study(module, dataset="rayleigh_benard", objectives=OBJECTIVES, seeds=(1,)):
    encoders = [
        dict(
            dataset=dataset,
            objective=objective,
            seed=seed,
            candidate="best_val",
            step=RAYLEIGH_BENARD_STEPS[objective],
            total_steps=100000,
            checkpoint=f"/handoff/{dataset}/{objective}/encoder_best_val.pt",
            checkpoint_sha256=f"{objective}_{seed}_best_val",
            bytes=88377111,
            encoder_state_sha256=f"{objective}_{seed}_state",
            config_sha256=f"{objective}_config",
            training_identity=f"{objective}_{seed}_identity",
            training_protocol={"total_steps": 100000, "batch_size": 64},
            spec={"n_channels": 4, "n_frames": 8, "height": 512, "width": 128},
            data_exposure={"clips_processed": 6400000},
            id=f"{dataset}_{objective}_seed{seed}_best_val",
        )
        for seed in seeds
        for objective in objectives
    ]
    return dict(
        provenance=provenance(),
        script_sha256=sha256_file(module.__file__),
        base="/data",
        handoff_sha256="handoff",
        encoder_policy=copy.deepcopy(module.ENCODER_POLICY),
        probe_settings=copy.deepcopy(module.PROBE_SETTINGS),
        protocols={dataset: module.frozen_protocol(REPO, dataset).to_dict()},
        physical_targets={dataset: module.selected_physical_targets(dataset)},
        objectives=list(objectives),
        encoders=encoders,
    )


def real_checkpoints(output, study):
    """Stage verifiable checkpoint bytes so hash and size checks run for real."""
    for row in study["encoders"]:
        path = output / "handoff" / f"{row['objective']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row["objective"].encode())
        row["checkpoint"] = str(path)
        row["checkpoint_sha256"] = sha256_file(path)
        row["bytes"] = path.stat().st_size
    return study


def seal_stub(path, kind):
    """Stand in for a stage artifact the orchestration only has to recognize."""
    write_json(Path(path) / "manifest_input.json", {})
    seal(Path(path), kind)
    return Path(path)


def write_rows(path):
    write_json(
        path / "rows.json",
        [
            dict(
                family="physics",
                representation="pooled",
                target_offset=0,
                target="convective_flux",
                method=method,
                valid_vrmse=0.3,
            )
            for method in ("ridge", "attentive")
        ],
    )


def checkpoint_block(row):
    return dict(
        dataset=row["dataset"],
        objective=row["objective"],
        seed=row["seed"],
        step=row["step"],
        sha256=row["checkpoint_sha256"],
        config_sha256=row["config_sha256"],
        spec={},
        encoder={},
        training_protocol={"total_steps": row["total_steps"]},
    )


def write_fit(output, study, row, sha256=None):
    path = Path(output) / "fits" / row["id"]
    write_rows(path)
    seal(
        path,
        "probe_fits",
        protocol=study["protocols"][row["dataset"]],
        caches={"train": "train", "valid": "valid"},
        probe_settings={},
        checkpoint=checkpoint_block(row) | ({"sha256": sha256} if sha256 else {}),
    )
    return path


def write_scores(output, study, row):
    path = Path(output) / "test" / row["id"]
    write_rows(path)
    seal(
        path,
        "probes",
        protocol=study["protocols"][row["dataset"]],
        caches={"test": "test"},
        probe_settings={},
        checkpoint=checkpoint_block(row),
    )
    return path


class FrozenStudyTests(unittest.TestCase):
    def setUp(self):
        self.sweep = sweep()
        self.study = frozen_study(self.sweep)

    def test_load_binds_every_command_to_the_best_validation_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_json(output / "study.json", self.study)
            self.assertEqual(len(self.sweep.load(output)["encoders"]), 4)

            old = copy.deepcopy(self.study)
            old["encoder_policy"]["version"] = 2
            self.rewrite(output, old)
            with self.assertRaisesRegex(ValueError, "encoder policy"):
                self.sweep.load(output)

            retuned = copy.deepcopy(self.study)
            retuned["probe_settings"]["attentive_epochs"] = 5
            self.rewrite(output, retuned)
            with self.assertRaisesRegex(ValueError, "probe settings"):
                self.sweep.load(output)

            subset = copy.deepcopy(self.study)
            subset["physical_targets"]["rayleigh_benard"] = [
                "enstrophy",
                "convective_flux",
            ]
            self.rewrite(output, subset)
            self.assertEqual(
                self.sweep.load(output)["physical_targets"]["rayleigh_benard"],
                ["enstrophy", "convective_flux"],
            )

            invalid_targets = copy.deepcopy(self.study)
            invalid_targets["physical_targets"]["rayleigh_benard"] = ["not_a_target"]
            self.rewrite(output, invalid_targets)
            with self.assertRaisesRegex(ValueError, "physical targets"):
                self.sweep.load(output)


            single = frozen_study(self.sweep, objectives=("jepa",))
            self.rewrite(output, single)
            self.assertEqual(
                [row["objective"] for row in self.sweep.load(output)["encoders"]],
                ["jepa"],
            )
            milestone = copy.deepcopy(self.study)
            milestone["encoders"][0]["candidate"] = "100pct"
            self.rewrite(output, milestone)
            with self.assertRaisesRegex(ValueError, "outside the frozen roster"):
                self.sweep.load(output)

            repeated = copy.deepcopy(self.study)
            repeated["encoders"].append(copy.deepcopy(repeated["encoders"][0]))
            self.rewrite(output, repeated)
            with self.assertRaisesRegex(ValueError, "duplicate best_val encoder"):
                self.sweep.load(output)

            shared = copy.deepcopy(self.study)
            shared["encoders"][1]["checkpoint_sha256"] = shared["encoders"][0][
                "checkpoint_sha256"
            ]
            self.rewrite(output, shared)
            with self.assertRaisesRegex(ValueError, "duplicate encoder checkpoint"):
                self.sweep.load(output)

            dropped = copy.deepcopy(self.study)
            dropped["encoders"] = [
                r for r in dropped["encoders"] if r["objective"] != "jepa_future"
            ]
            self.rewrite(output, dropped)
            with self.assertRaisesRegex(ValueError, "incomplete objective roster"):
                self.sweep.load(output)

            slipped = copy.deepcopy(self.study)
            slipped["encoders"][0]["step"] = 100001
            self.rewrite(output, slipped)
            with self.assertRaisesRegex(ValueError, "outside the training budget"):
                self.sweep.load(output)

    def rewrite(self, output, study):
        (output / "study.json").unlink()
        write_json(output / "study.json", study)

    def test_task_indices_outside_the_frozen_roster_fail(self):
        with self.assertRaisesRegex(ValueError, "outside the frozen roster of 4"):
            self.sweep.encoder(self.study, 4)
        self.assertEqual(self.sweep.encoder(self.study, 0)["objective"], "jepa")

    def test_test_scoring_and_test_caching_need_every_validation_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calls = []
            self.sweep.prepare_cache = lambda *a, **k: calls.append(a)
            self.sweep.extract_features = lambda *a, **k: calls.append(a)
            self.sweep.score_probes = lambda *a, **k: calls.append(a)
            for row in self.study["encoders"][:3]:
                write_fit(output, self.study, row)
            with self.assertRaisesRegex(ValueError, "no probe fit"):
                self.sweep.test(output, self.study, 3)
            with self.assertRaisesRegex(ValueError, "no probe fit"):
                self.sweep.cache(output, self.study, "rayleigh_benard", "test")
            self.assertEqual(calls, [])
            # Train and validation data stay available before any fit exists.
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.cache(output, self.study, "rayleigh_benard", "valid")
            self.assertEqual(len(calls), 1)

    def test_a_fit_from_another_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for row in self.study["encoders"]:
                write_fit(output, self.study, row, sha256="some_other_encoder")
            with self.assertRaisesRegex(ValueError, "does not match the frozen"):
                self.sweep.completed_fits(output, self.study)

    def test_test_scoring_follows_the_fits_without_a_selection_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = real_checkpoints(output, self.study)
            fits = [write_fit(output, study, row) for row in study["encoders"]]
            row = study["encoders"][2]
            cache_root, _, feature, fit = self.sweep.encoder_paths(output, row)
            extracted, scored = [], []

            def fake_extract(checkpoint, cache, feature_root, split):
                extracted.append((checkpoint, cache, feature_root, split))
                seal_stub(feature / split, "features")

            def fake_score(features, cache, probes, destination):
                scored.append((features, cache, probes, destination))
                seal_stub(destination, "probes")

            self.sweep.extract_features = fake_extract
            self.sweep.score_probes = fake_score
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.test(output, study, 2)
            self.assertEqual(fit, fits[2])
            self.assertEqual(
                extracted,
                [
                    (row["checkpoint"], cache_root, feature.parent, split)
                    for split in ("train", "valid", "test")
                ],
            )
            self.assertEqual(
                scored, [(feature, cache_root, fit, output / "test" / row["id"])]
            )
            self.assertFalse((output / "selection").exists())

    def test_fitting_uses_the_probe_settings_frozen_in_the_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = real_checkpoints(output, self.study)
            row = study["encoders"][0]
            _, _, feature, _ = self.sweep.encoder_paths(output, row)
            calls = []

            def fake_extract(checkpoint, cache, feature_root, split):
                calls.append(split)
                seal_stub(feature / split, "features")

            def fake_fit(features, cache, destination, **settings):
                calls.append(settings)
                write_fit(output, study, row)

            self.sweep.extract_features = fake_extract
            self.sweep.fit_probes = fake_fit
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.run(output, study, 0)
            self.assertEqual(
                calls,
                [
                    "train",
                    "valid",
                    dict(
                        mlp_max_steps=2000,
                        mlp_min_steps=150,
                        governing_only=False,
                        feature_mlp=True,
                        attentive=False,
                        attentive_epochs=100,
                        attentive_batch_size=32,
                        attentive_min_epochs=15,
                        attentive_patience=10,
                        physical_targets=self.study["physical_targets"][
                            "rayleigh_benard"
                        ],
                    ),
                ],
            )

    def test_a_stage_that_seals_nothing_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = real_checkpoints(output, self.study)
            self.sweep.extract_features = lambda *a: None
            self.sweep.fit_probes = lambda *a, **k: None
            with self.assertRaisesRegex(ValueError, "sealed no train features"):
                self.sweep.run(output, study, 0)

    def test_a_sealed_fit_is_never_recomputed_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_fit(output, self.study, self.study["encoders"][0])
            calls = []
            self.sweep.extract_features = lambda *a: calls.append(a)
            self.sweep.fit_probes = lambda *a, **k: calls.append(a)
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.run(output, self.study, 0)
            self.assertEqual(calls, [])

    def test_report_aggregates_the_four_objective_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            aggregated, plotted = [], []
            self.sweep.aggregate = lambda *a, **k: aggregated.append((a, k))
            self.sweep.plot = lambda *a: plotted.append(a)
            scores = [write_scores(output, self.study, r) for r in self.study["encoders"][:3]]
            with self.assertRaisesRegex(ValueError, "without test scores"):
                self.sweep.report(output, self.study)
            scores.append(write_scores(output, self.study, self.study["encoders"][3]))
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.report(output, self.study)
            folder = output / "reports/rayleigh_benard"
            self.assertEqual(
                aggregated,
                [((scores, folder / "aggregate"), dict(objectives=list(OBJECTIVES), seeds=[1]))],
            )
            self.assertEqual(plotted, [(folder / "aggregate", folder / "plots")])

    def test_status_reports_pending_work_for_a_resumable_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.sweep.status(output, self.study)
            state = json.loads(stream.getvalue())
            self.assertEqual(state["pending_fits"], [0, 1, 2, 3])
            self.assertEqual(state["pending_tests"], [0, 1, 2, 3])
            self.assertEqual(
                state["caches"]["rayleigh_benard"],
                {"train": False, "valid": False, "test": False},
            )
            self.assertEqual(state["reports"], {"rayleigh_benard": False})

            for row in self.study["encoders"][:2]:
                write_fit(output, self.study, row)
            write_scores(output, self.study, self.study["encoders"][0])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.sweep.status(output, self.study)
            state = json.loads(stream.getvalue())
            self.assertEqual(state["pending_fits"], [2, 3])
            self.assertEqual(state["pending_tests"], [1, 2, 3])
            self.assertEqual(
                [r["feature_dir"] for r in state["encoders"]],
                [
                    str(self.sweep.encoder_paths(output, row)[2])
                    for row in self.study["encoders"]
                ],
            )


if __name__ == "__main__":
    unittest.main()
