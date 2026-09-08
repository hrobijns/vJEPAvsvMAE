"""Exercise exposure reporting through a saved checkpoint and a fresh process."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.models.checkpoints import load_encoder
from tests.fixtures import write_well


class TrainingExposureTests(unittest.TestCase):
    def test_completed_updates_survive_resume(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_well(root, "shear_flow", "train", frames=16)
            cfg = yaml.safe_load((repo / "configs/shear_flow_mae.yaml").read_text())
            cfg.update(
                run_name="exposure",
                out_dir=str(root / "runs"),
                log_every=1,
                save_every=2,
                val_every=2,
                bf16=False,
            )
            cfg["data"].update(base_path=str(root), memmap=False, num_workers=0)
            cfg["encoder"].update(
                patch_h=8, patch_w=8, embed_dim=16, depth=1, num_heads=2
            )
            cfg["objective"].update(decoder_dim=16, decoder_depth=1, decoder_heads=2)
            cfg["optim"].update(batch_size=2, total_steps=8, warmup_steps=2)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            # Exit immediately after a durable step-2 checkpoint; the next
            # invocation is the ordinary trainer with the same run budget.
            interrupted = """
import os, torch
from pathlib import Path
from src.train import main
save = torch.save
def save_then_exit(payload, path, *args, **kwargs):
    save(payload, path, *args, **kwargs)
    if Path(path).name == 'latest.pt' and payload['step'] == 2:
        os._exit(0)
torch.save = save_then_exit
main()
"""
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
            }
            args = ["--config", str(config), "--no-wandb"]
            for entry in (["-c", interrupted], ["-m", "src.train"]):
                result = subprocess.run(
                    [sys.executable, "-B", *entry, *args],
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                checkpoint = torch.load(
                    root / "runs/exposure/latest.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                step = 2 if entry[0] == "-c" else 8
                self.assertEqual(checkpoint["step"], step)
                self.assertEqual(
                    {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()},
                    {step},
                )
                exposure = checkpoint["data_exposure"]
                self.assertEqual(exposure["eligible_train_clips"], 4 * 9)
                self.assertEqual(exposure["eligible_validation_clips"], 9)
                self.assertEqual(exposure["planned_clips"], 16)
                self.assertEqual(exposure["clips_processed"], step * 2)
                self.assertAlmostEqual(exposure["equivalent_passes"], step * 2 / 36)
                self.assertAlmostEqual(exposure["planned_equivalent_passes"], 16 / 36)
            run = root / "runs/exposure"
            history = [
                json.loads(line)
                for line in (run / "history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [r["step"] for r in history if r["phase"] == "train"], list(range(1, 9))
            )
            for row in history:
                self.assertEqual(
                    row["data_exposure"]["clips_processed"], row["step"] * 2
                )
            _, _, metadata = load_encoder(run / "encoder_100pct.pt")
            self.assertEqual(metadata["data_exposure"], checkpoint["data_exposure"])
            self.assertIn("at step 2", result.stdout)
