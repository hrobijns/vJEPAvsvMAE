"""Select by the agreed validation history, with complete comparable pilots."""
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.lr_sweep import prepare, score_history, VALIDATION_STEPS


class LearningRateSweepTests(unittest.TestCase):
    def test_selection_uses_best_recorded_validation_and_retains_diagnostics(self):
        rows = [dict(phase="val", step=step, loss=loss, context_feat_std=std)
                for step, loss, std in zip(VALIDATION_STEPS, (0.5, 0.2, 0.3, 0.4), (0.8, 0.6, 0.04, 0.1))]
        result = score_history(rows)
        self.assertEqual((result["best_val_loss"], result["best_val_step"]), (0.2, 4000))
        self.assertEqual(result["final_val_loss"], 0.4)
        self.assertEqual(result["collapse_flags"], ["context_feat_std"])
        for invalid in (rows[:-1], rows + rows[-1:]):
            with self.assertRaisesRegex(ValueError, "history"):
                score_history(invalid)
        invalid = copy.deepcopy(rows)
        invalid[1]["loss"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_history(invalid)

    def test_preparation_produces_independent_full_budget_pilots(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare("rayleigh_benard", tmp, tmp, workers=8)
            candidates = manifest["candidates"]
            self.assertEqual(len(candidates), 12)
            self.assertEqual(len({c["run_dir"] for c in candidates}), 12)
            self.assertEqual(len({c["objective"] for c in candidates[:4]}), 4)
            for candidate in candidates:
                cfg = yaml.safe_load(Path(candidate["config"]).read_text())
                self.assertEqual(cfg["optim"]["lr"], candidate["lr"])
                self.assertEqual(cfg["optim"]["total_steps"], 8000)
                self.assertEqual(cfg["optim"]["warmup_steps"], 5000)
                self.assertEqual(cfg["seed"], 0)
                self.assertNotIn("val_max_batches", cfg)
                self.assertIsNone(cfg["data"]["frame_limit"])
                self.assertEqual(cfg["data"]["num_workers"], 8)
            with self.assertRaises(FileExistsError):
                prepare("rayleigh_benard", tmp, tmp)
