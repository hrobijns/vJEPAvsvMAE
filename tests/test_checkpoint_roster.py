"""Stage-1 candidate policy: the frozen four-milestone roster and its boundary."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.artifacts import Artifact, provenance, seal, sha256_file, write_json
from src.evaluation.protocol import Protocol
from src.objectives import OBJECTIVES
from src.physics.systems import SYSTEMS

REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "checkpoints/iclr2027/seed1"


def sweep():
    spec = importlib.util.spec_from_file_location(
        "probe_sweep", REPO / "scripts/probe_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_row(objective, label, percent, seed=1, total=100000, state=None, **overrides):
    return dict(
        path=f"shear_flow/{objective}/seed{seed}/encoder_{label}.pt",
        dataset="shear_flow",
        objective=objective,
        seed=seed,
        step=percent * total // 100,
        candidate=label,
        bytes=1,
        sha256=f"{objective}_{seed}_{label}_checkpoint",
        encoder_state_sha256=state or f"{objective}_{seed}_{label}_state",
        config_sha256=f"{objective}_config",
        spec={"n_channels": 4, "n_frames": 8, "height": 256, "width": 512},
        training_protocol={"total_steps": total, "batch_size": 64},
        training_identity=f"{objective}_{seed}_identity",
        data_exposure={"clips_processed": percent * 1000},
        **overrides,
    )


def synthetic_index(objectives=OBJECTIVES, seeds=(1,)):
    rows = []
    for seed in seeds:
        for objective in objectives:
            for percent in (25, 50, 75, 100):
                rows.append(
                    index_row(objective, f"{percent:03d}pct", percent, seed=seed)
                )
            rows.append(
                index_row(
                    objective,
                    "best_val",
                    100,
                    seed=seed,
                    state=f"{objective}_{seed}_100pct_state",
                )
            )
    return dict(training_provenance={}, checkpoints=rows)


class CandidatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.sweep = sweep()

    def test_seed1_handoff_freezes_three_systems_by_four_objectives_by_four_steps(self):
        index = json.loads((HANDOFF / "index.json").read_text())
        candidates = self.sweep.frozen_candidates(HANDOFF, index)
        groups = self.sweep.roster_groups(candidates)
        self.assertEqual((len(candidates), len(groups)), (48, 12))
        self.assertEqual(
            sorted({r["candidate"] for r in candidates}),
            ["025pct", "050pct", "075pct", "100pct"],
        )
        self.assertEqual(
            sorted({r["step"] for r in candidates}), [25000, 50000, 75000, 100000]
        )
        self.assertEqual(len({r["checkpoint_sha256"] for r in candidates}), 48)
        self.assertEqual(len({(r["dataset"], r["objective"]) for r in candidates}), 12)
        for row in candidates:
            self.assertEqual(row["aliases"], [])
            self.assertNotIn("best_val", row["checkpoint"])
            self.assertEqual(
                row["step"] * 100, row["total_steps"] * int(row["candidate"][:3])
            )

    def test_dataset_scope_freezes_only_the_requested_complete_system(self):
        index = json.loads((HANDOFF / "index.json").read_text())
        candidates = self.sweep.frozen_candidates(
            HANDOFF, index, dataset="rayleigh_benard"
        )
        groups = self.sweep.roster_groups(candidates)
        self.assertEqual((len(candidates), len(groups)), (16, 4))
        self.assertEqual({row["dataset"] for row in candidates}, {"rayleigh_benard"})
        self.assertEqual(
            {row["objective"] for row in candidates}, set(OBJECTIVES)
        )
        self.assertEqual(
            sorted({row["candidate"] for row in candidates}),
            ["025pct", "050pct", "075pct", "100pct"],
        )

    def test_dataset_scope_rejects_a_dataset_absent_from_the_handoff(self):
        with self.assertRaisesRegex(ValueError, "is absent from the handoff"):
            self.sweep.frozen_candidates(
                "/handoff", synthetic_index(), dataset="rayleigh_benard"
            )

    def test_excluded_best_val_never_enters_the_roster(self):
        candidates = self.sweep.frozen_candidates("/handoff", synthetic_index())
        self.assertEqual(len(candidates), 16)
        labels = {r["candidate"] for r in candidates}
        labels.update(a["candidate"] for r in candidates for a in r["aliases"])
        self.assertEqual(labels, {"025pct", "050pct", "075pct", "100pct"})
        self.assertEqual(
            self.sweep.CANDIDATE_POLICY["excluded_candidates"], ["best_val"]
        )

    def test_missing_duplicate_and_unexpected_candidates_are_rejected(self):
        index = synthetic_index()
        missing = copy.deepcopy(index)
        missing["checkpoints"] = [
            r for r in missing["checkpoints"] if r["candidate"] != "075pct"
        ]
        with self.assertRaisesRegex(ValueError, "incomplete milestone roster"):
            self.sweep.frozen_candidates("/handoff", missing)

        duplicate = copy.deepcopy(index)
        duplicate["checkpoints"].append(index_row("jepa", "050pct", 50))
        with self.assertRaisesRegex(ValueError, "duplicate 050pct"):
            self.sweep.frozen_candidates("/handoff", duplicate)

        unexpected = copy.deepcopy(index)
        unexpected["checkpoints"].append(index_row("jepa", "090pct", 90))
        with self.assertRaisesRegex(ValueError, "unexpected candidate label"):
            self.sweep.frozen_candidates("/handoff", unexpected)

        with self.assertRaisesRegex(ValueError, "no checkpoints"):
            self.sweep.frozen_candidates("/handoff", dict(checkpoints=[]))

    def test_a_run_group_holding_only_excluded_candidates_is_still_rejected(self):
        index = synthetic_index()
        index["checkpoints"] = [
            r
            for r in index["checkpoints"]
            if r["objective"] != "mae" or r["candidate"] == "best_val"
        ]
        with self.assertRaisesRegex(
            ValueError, r"incomplete milestone roster for \('shear_flow', 'mae', 1\)"
        ):
            self.sweep.frozen_candidates("/handoff", index)

    def test_every_observed_dataset_and_seed_needs_all_four_objectives(self):
        index = synthetic_index()
        index["checkpoints"] = [
            r for r in index["checkpoints"] if r["objective"] != "mae_future"
        ]
        with self.assertRaisesRegex(
            ValueError, r"incomplete objective roster for \('shear_flow', 1\)"
        ):
            self.sweep.frozen_candidates("/handoff", index)

        # A later seed is validated on its own; no seed is hard-coded.
        two_seeds = synthetic_index(seeds=(1, 2))
        self.assertEqual(
            len(self.sweep.frozen_candidates("/handoff", two_seeds)), 32
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
            self.sweep.frozen_candidates("/handoff", partial)

    def test_aliasing_cannot_hide_a_repeated_or_absent_milestone(self):
        index = synthetic_index()
        # Byte-identical duplicates would collapse into one job; the declared
        # roster must still be rejected before deduplication.
        index["checkpoints"].append(
            index_row("jepa", "050pct", 50, state="jepa_1_050pct_state")
        )
        with self.assertRaisesRegex(ValueError, "duplicate 050pct"):
            self.sweep.frozen_candidates("/handoff", index)

        aliased = synthetic_index()
        aliased["checkpoints"] = [
            r
            for r in aliased["checkpoints"]
            if not (r["objective"] == "mae" and r["candidate"] == "100pct")
        ]
        for row in aliased["checkpoints"]:
            if row["objective"] == "mae" and row["candidate"] == "best_val":
                row["encoder_state_sha256"] = "mae_1_075pct_state"
        with self.assertRaisesRegex(ValueError, "incomplete milestone roster"):
            self.sweep.frozen_candidates("/handoff", aliased)

    def test_candidate_step_must_equal_its_declared_training_fraction(self):
        for label, step in (("050pct", 40000), ("100pct", 99000)):
            index = synthetic_index()
            for row in index["checkpoints"]:
                if row["objective"] == "jepa" and row["candidate"] == label:
                    row["step"] = step
            with self.assertRaisesRegex(ValueError, f"{label} candidate .* is at step"):
                self.sweep.frozen_candidates("/handoff", index)

        rescaled = synthetic_index()
        for row in rescaled["checkpoints"]:
            if row["objective"] == "jepa":
                row["training_protocol"]["total_steps"] = 200000
        with self.assertRaisesRegex(ValueError, "is at step"):
            self.sweep.frozen_candidates("/handoff", rescaled)

        absent = synthetic_index()
        for row in absent["checkpoints"]:
            row["training_protocol"]["total_steps"] = None
        with self.assertRaisesRegex(ValueError, "missing training budget"):
            self.sweep.frozen_candidates("/handoff", absent)

    def test_inconsistent_run_metadata_is_rejected(self):
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
            with self.assertRaisesRegex(ValueError, f"candidate {field} differs"):
                self.sweep.frozen_candidates("/handoff", index)


MILESTONE_STEPS = (25000, 50000, 75000, 100000)


def frozen_study(sweep_module, objectives=OBJECTIVES, steps=MILESTONE_STEPS):
    candidates = [
        dict(
            dataset="shear_flow",
            objective=objective,
            seed=1,
            candidate=f"{step // 1000:03d}pct",
            step=step,
            total_steps=100000,
            checkpoint=f"/handoff/shear_flow/{objective}/encoder_{step}.pt",
            checkpoint_sha256=f"{objective}_{step}",
            config_sha256="config",
            id=f"shear_flow_{objective}_seed1_{step}",
            aliases=[],
        )
        for objective in objectives
        for step in steps
    ]
    return dict(
        provenance=provenance(),
        script_sha256=sha256_file(sweep_module.__file__),
        base="/data",
        handoff_sha256="handoff",
        candidate_policy=copy.deepcopy(sweep_module.CANDIDATE_POLICY),
        protocols={"shear_flow": Protocol("shear_flow").to_dict()},
        candidates=candidates,
    )


def write_fit(path, sha256, step, valid_vrmse, objective="jepa"):
    protocol = Protocol("shear_flow")
    path.mkdir(parents=True)
    write_json(
        path / "rows.json",
        [
            dict(
                family="physics",
                method="selected",
                representation=rep,
                target_offset=offset,
                target=target,
                valid_vrmse=valid_vrmse,
            )
            for rep in ("pooled", "token")
            for offset in protocol.target_offsets
            for target in SYSTEMS["shear_flow"].targets
        ],
    )
    seal(
        path,
        "probe_fits",
        protocol=protocol.to_dict(),
        caches={"train": "train", "valid": "valid"},
        probe_settings={},
        checkpoint=dict(
            dataset="shear_flow",
            objective=objective,
            seed=1,
            step=step,
            sha256=sha256,
            config_sha256="config",
            spec={},
            encoder={},
            training_protocol={"total_steps": 100000},
        ),
    )


def write_frozen_fits(output, study, scores, skip=()):
    for row in study["candidates"]:
        if row["id"] in skip:
            continue
        write_fit(
            output / "fits" / row["id"],
            row["checkpoint_sha256"],
            row["step"],
            scores[row["step"]],
            objective=row["objective"],
        )


class FrozenStudyTests(unittest.TestCase):
    def setUp(self):
        self.sweep = sweep()

    def test_load_binds_every_command_to_the_frozen_candidate_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = frozen_study(self.sweep)
            write_json(output / "study.json", study)
            self.assertEqual(len(self.sweep.load(output)["candidates"]), 16)

            best_val = dict(
                study["candidates"][-1],
                candidate="best_val",
                step=16000,
                checkpoint_sha256="mae_future_best_val",
                id="shear_flow_mae_future_seed1_best_val",
            )
            old = copy.deepcopy(study)
            old["candidate_policy"]["version"] = 0
            old["candidate_policy"]["excluded_candidates"] = []
            old["candidates"].append(best_val)
            self.rewrite(output, old)
            with self.assertRaisesRegex(ValueError, "candidate policy"):
                self.sweep.load(output)

            mutated = copy.deepcopy(study)
            mutated["candidates"].append(best_val)
            self.rewrite(output, mutated)
            with self.assertRaisesRegex(ValueError, "outside the policy roster"):
                self.sweep.load(output)

            short = copy.deepcopy(study)
            short["candidates"] = [
                c
                for c in short["candidates"]
                if not (c["objective"] == "mae" and c["step"] == 75000)
            ]
            self.rewrite(output, short)
            with self.assertRaisesRegex(ValueError, "incomplete milestone roster"):
                self.sweep.load(output)

            dropped = copy.deepcopy(study)
            dropped["candidates"] = [
                c for c in dropped["candidates"] if c["objective"] != "jepa_future"
            ]
            self.rewrite(output, dropped)
            with self.assertRaisesRegex(ValueError, "incomplete objective roster"):
                self.sweep.load(output)

            slipped = copy.deepcopy(study)
            slipped["candidates"][1]["step"] = 60000
            self.rewrite(output, slipped)
            with self.assertRaisesRegex(ValueError, "is at step 60000"):
                self.sweep.load(output)

    def rewrite(self, output, study):
        (output / "study.json").unlink()
        write_json(output / "study.json", study)

    def test_task_indices_outside_the_frozen_roster_fail(self):
        study = frozen_study(self.sweep)
        with self.assertRaisesRegex(ValueError, "outside the frozen roster of 16"):
            self.sweep.candidate(study, 55)
        self.assertEqual(self.sweep.candidate(study, 3)["step"], 100000)

    def test_collect_scores_only_frozen_fits_and_picks_one_winner_per_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = frozen_study(self.sweep)
            scores = {25000: 0.5, 50000: 0.2, 75000: 0.4, 100000: 0.9}
            write_frozen_fits(output, study, scores)
            # An unfrozen fit on disk would win on validation but is not a candidate.
            write_fit(
                output / "fits/shear_flow_jepa_seed1_best_val",
                "jepa_best_val",
                16000,
                0.01,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.sweep.collect(output, study)
            selection = Artifact(output / "selection", "checkpoint_selection")
            winners = selection.json("selections.json")
            rows = selection.json("rows.json")
            self.assertEqual(len(winners), 4)
            self.assertEqual({w["objective"] for w in winners}, set(OBJECTIVES))
            self.assertEqual({w["step"] for w in winners}, {50000})
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(selection.manifest["candidates"]), 16)
            for row in rows:
                self.assertNotIn("test_vrmse", row)

    def test_collect_requires_a_fit_for_every_frozen_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = frozen_study(self.sweep)
            write_frozen_fits(
                output,
                study,
                dict.fromkeys(MILESTONE_STEPS, 0.3),
                skip={study["candidates"][-1]["id"]},
            )
            with self.assertRaisesRegex(ValueError, "no probe fit"):
                self.sweep.collect(output, study)

    def test_collect_rejects_a_fit_from_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            study = frozen_study(self.sweep)
            for row in study["candidates"]:
                write_fit(
                    output / "fits" / row["id"],
                    "jepa_best_val"
                    if row["id"] == study["candidates"][0]["id"]
                    else row["checkpoint_sha256"],
                    row["step"],
                    0.3,
                    objective=row["objective"],
                )
            with self.assertRaisesRegex(ValueError, "does not match prepared"):
                self.sweep.collect(output, study)

    def test_selected_row_must_come_from_the_frozen_roster(self):
        study = frozen_study(self.sweep)
        self.assertEqual(
            self.sweep.selected_row(study, {"checkpoint_sha256": "jepa_50000"})["step"],
            50000,
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen candidate roster"):
            self.sweep.selected_row(study, {"checkpoint_sha256": "jepa_best_val"})


if __name__ == "__main__":
    unittest.main()
