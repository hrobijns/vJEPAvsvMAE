import tempfile
import unittest
from pathlib import Path

from scripts import preprocess_memmap
from scripts import workshop_test_eval


class PreprocessTests(unittest.TestCase):
    def test_rb_full_trajectory_requests_all_199_output_frames(self):
        self.assertEqual(preprocess_memmap.full_trajectory_max_rollout("rayleigh_benard"), 199)

    def test_full_trajectory_length_rejects_the_legacy_101_frame_cache(self):
        with self.assertRaisesRegex(ValueError, "expected 200"):
            preprocess_memmap.validate_trajectory_length("rayleigh_benard", 101)
        preprocess_memmap.validate_trajectory_length("rayleigh_benard", 200)

    def test_completed_memmap_is_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test.npy"
            metadata = Path(tmp) / "test.meta.json"
            progress = Path(tmp) / "test.progress.json"
            output.touch()
            metadata.write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "completed"):
                preprocess_memmap.validate_memmap_state(output, metadata, progress)

    def test_memmap_metadata_marks_normalized_full_rollout(self):
        metadata = preprocess_memmap.memmap_metadata(
            dataset="rayleigh_benard",
            split="test",
            shape=(175, 4, 200, 512, 128),
            source_contract={"stats_sha256": "abc"},
        )
        self.assertEqual(metadata["schema_version"], "well-memmap-v2")
        self.assertEqual(metadata["layout"], "NCTHW")
        self.assertEqual(metadata["shape"][2], 200)
        self.assertEqual(metadata["normalization"], "the_well_zscore")


class LegacyGuardTests(unittest.TestCase):
    def test_rb_legacy_pipeline_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(SystemExit, "legacy RB"):
            workshop_test_eval.guard_legacy_rb("rayleigh_benard", allow_legacy=False)
        workshop_test_eval.guard_legacy_rb("rayleigh_benard", allow_legacy=True)
        workshop_test_eval.guard_legacy_rb("shear_flow", allow_legacy=False)


if __name__ == "__main__":
    unittest.main()
