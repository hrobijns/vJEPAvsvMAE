"""Local staging preserves data and shares one resumable copy across callers."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.stage_training_cache import stage
from src.data.preprocess import preprocess
from src.data.well import MemmapClipDataset
from tests.fixtures import write_well


class TrainingCacheStagingTests(unittest.TestCase):
    def test_concurrent_calls_resume_one_copy_and_preserve_training_clips(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            write_well(root / "source", "shear_flow", "train", frames=16)
            preprocess(root / "source", "shear_flow")
            source = root / "source/memmap/shear_flow/train.npy"
            meta = json.loads(source.with_suffix(".meta.json").read_text())
            target = root / "ssd" / meta["array_sha256"] / "memmap/shear_flow/train.npy"
            target.parent.mkdir(parents=True)
            target.with_suffix(".partial.npy").write_bytes(source.read_bytes()[:1234])
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(stage, root / "source", "shear_flow", root / "ssd") for _ in range(2)]
                results = [f.result() for f in futures]
            self.assertEqual(sorted(r["reused"] for r in results), [False, True])
            self.assertEqual(next(r for r in results if not r["reused"])["resumed_bytes"], 1234)
            original = MemmapClipDataset(root / "source", "shear_flow", "train", future=True)
            local = MemmapClipDataset(results[0]["base_path"], "shear_flow", "train", future=True)
            self.assertEqual(original.identity, local.identity)
            self.assertEqual(len(original), len(local))
            for index in (0, len(original) - 1):
                for key in ("clip", "target_clip"):
                    self.assertTrue(torch.equal(original[index][key], local[index][key]))

    def test_damaged_copy_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            write_well(root / "source", "shear_flow", "train", frames=16)
            preprocess(root / "source", "shear_flow")
            source = root / "source/memmap/shear_flow/train.npy"
            meta = json.loads(source.with_suffix(".meta.json").read_text())
            with source.open("r+b") as handle:
                handle.seek(-1, 2)
                old = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([old[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "content hash"):
                stage(root / "source", "shear_flow", root / "ssd")
            self.assertFalse((root / "ssd" / meta["array_sha256"] / "memmap/shear_flow/train.npy").exists())
