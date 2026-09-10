"""Continuation must reproduce training, including prefetch and validation RNG."""

import contextlib
import fcntl
import io
import json
import os
import shutil
import signal
import time
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.data.preprocess import preprocess
from src.models.checkpoints import load_encoder
from src.train import atomic_save, restore_history
from tests.fixtures import write_well

REPO = Path(__file__).resolve().parents[1]
# Send the real trainer signal after a chosen update, before scheduled validation.
# Also exercise the Python/NumPy streams, which current objectives do not use.
ENTRY = """
import os, random, signal, numpy as np
import src.train as training
build = training.build_model
def build_checked(*args):
    model = build(*args)
    post_step = model.post_step
    def checked(step, total):
        post_step(step, total)
        random.random()
        np.random.random()
        if step + 1 == int(os.environ.get('STOP_STEP', '0')):
            os.kill(os.getpid(), signal.SIGUSR1)
    model.post_step = checked
    if os.environ.get('STOP_STEP') == '-1':
        os.kill(os.getpid(), signal.SIGUSR1)
    return model
training.build_model = build_checked
raise SystemExit(training.main())
"""


class TrainingExposureTests(unittest.TestCase):
    def run_training(self, config, stop=0, expected=0):
        result = subprocess.run(
            [sys.executable, "-B", "-c", ENTRY, "--config", str(config), "--no-wandb"],
            cwd=REPO,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                 "MKL_NUM_THREADS": "2", "STOP_STEP": str(stop)},
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_all_objectives_resume_exactly_across_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_well(root, "shear_flow", "train", frames=17)
            with contextlib.redirect_stdout(io.StringIO()):
                preprocess(root, "shear_flow")
            shutil.copytree(root / "memmap", root / "relocated/memmap")
            for index, objective in enumerate(("jepa", "mae", "jepa_future", "mae_future")):
                with self.subTest(objective=objective):
                    cfg = yaml.safe_load((REPO / f"configs/shear_flow_{objective}.yaml").read_text())
                    cfg.update(out_dir=str(root / "runs"), log_every=1, save_every=5,
                               val_every=2, bf16=False)
                    memmap = index in (1, 2)
                    workers = 2 if index in (0, 2) else 0
                    cfg["data"].update(base_path=str(root), memmap=memmap, num_workers=workers)
                    cfg["encoder"].update(patch_h=8, patch_w=8, embed_dim=16, depth=1, num_heads=2)
                    prefix = "predictor" if objective.startswith("jepa") else "decoder"
                    cfg["objective"].update({f"{prefix}_dim": 16, f"{prefix}_depth": 1, f"{prefix}_heads": 2})
                    cfg["optim"].update(batch_size=4, total_steps=12, warmup_steps=2)
                    config = root / "config.yaml"
                    cfg["run_name"] = f"{objective}_continuous"
                    config.write_text(yaml.safe_dump(cfg))
                    if index == 0:
                        self.run_training(config, stop=-1, expected=1)
                        self.assertFalse((root / "runs" / cfg["run_name"] / "latest.pt").exists())
                    self.run_training(config)
                    continuous = root / "runs" / cfg["run_name"]
                    expected = torch.load(continuous / "latest.pt", weights_only=False)

                    cfg["run_name"] = f"{objective}_resumed"
                    config.write_text(yaml.safe_dump(cfg))
                    run = root / "runs" / cfg["run_name"]
                    # 40 current clips / batch 4 = 10 updates per pass;
                    # 8 future pairs / batch 4 = 2 updates per pass.
                    stops = (1, 2) if objective.endswith("future") else (3, 10)
                    for stop in stops:
                        self.run_training(config, stop, expected=75)
                        saved = torch.load(run / "latest.pt", weights_only=False)
                        self.assertEqual(saved["step"], stop)
                        self.assertEqual({int(v["step"]) for v in saved["optimizer"]["state"].values()}, {stop})
                        self.assertEqual(saved["data_exposure"]["clips_processed"], stop * 4)
                        # Simulate a crash after latest replaced but before an
                        # applicable encoder export, plus a partial history tail.
                        for export in run.glob("encoder_*.pt"):
                            if torch.load(export, weights_only=False)["step"] == stop:
                                export.unlink()
                        with (run / "history.jsonl").open("ab") as handle:
                            handle.write(b'{"step":999,"phase":"train"}\n{"step":')
                        cfg["data"]["num_workers"] = 2 - workers
                        if memmap:
                            cfg["data"]["base_path"] = str(root / "relocated")
                        config.write_text(yaml.safe_dump(cfg))
                    # A signal on the final update is successful completion.
                    result = self.run_training(config, stop=12)
                    self.assertIn(f"at step {stops[-1]}", result.stdout)
                    actual = torch.load(run / "latest.pt", weights_only=False)
                    for key in ("model", "optimizer", "rng", "first_val_feat_std", "best_val_loss",
                                "best_val_step", "data_exposure", "training_identity"):
                        self.assert_state_equal(expected[key], actual[key])
                    self.assertEqual((continuous / "history.jsonl").read_bytes(), (run / "history.jsonl").read_bytes())
                    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
                    self.assertEqual([r["step"] for r in history if r["phase"] == "train"], list(range(1, 13)))
                    self.assertEqual(actual["data_exposure"]["eligible_train_clips"], 8 if objective.endswith("future") else 40)
                    self.assertEqual(actual["data_exposure"]["planned_clips"], 48)
                    for name in ("encoder_best_val.pt", "encoder_025pct.pt", "encoder_050pct.pt", "encoder_075pct.pt", "encoder_100pct.pt"):
                        expected_export = torch.load(continuous / name, weights_only=False)
                        actual_export = torch.load(run / name, weights_only=False)
                        self.assertEqual(expected_export["step"], actual_export["step"])
                        self.assert_state_equal(expected_export["encoder"], actual_export["encoder"])
                    _, _, metadata = load_encoder(run / "encoder_100pct.pt")
                    self.assertEqual(metadata["data_exposure"], actual["data_exposure"])
                    (run / "encoder_100pct.pt").unlink()
                    before = (run / "latest.pt").read_bytes()
                    self.assertIn("already complete", self.run_training(config).stdout)
                    self.assertEqual(before, (run / "latest.pt").read_bytes())
                    self.assertTrue((run / "encoder_100pct.pt").exists())

                    if index == 0:
                        # Resume rejects changed random-consuming settings,
                        # changed source code, old state and concurrent writers.
                        cfg["val_every"] = 3
                        config.write_text(yaml.safe_dump(cfg))
                        self.assertIn("differs", self.run_training(config, expected=1).stderr)
                        cfg["val_every"] = 2
                        config.write_text(yaml.safe_dump(cfg))
                        incompatible = dict(actual, training_identity="changed source hash")
                        atomic_save(incompatible, run / "latest.pt")
                        self.assertIn("differs", self.run_training(config, expected=1).stderr)
                        incompatible.pop("rng")
                        atomic_save(incompatible, run / "latest.pt")
                        self.assertIn("lacks complete", self.run_training(config, expected=1).stderr)
                        atomic_save(actual, run / "latest.pt")
                        with (run / ".train.lock").open("a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            self.assertIn("another trainer", self.run_training(config, expected=1).stderr)

    def test_failed_atomic_write_and_short_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest = root / "latest.pt"
            atomic_save({"step": 3}, latest)
            previous = latest.read_bytes()

            def interrupted_save(payload, handle):
                handle.write(b"incomplete checkpoint")
                raise RuntimeError("simulated interruption")

            with patch("src.train.torch.save", side_effect=interrupted_save):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    atomic_save({"step": 4}, latest)
            self.assertEqual(previous, latest.read_bytes())
            self.assertEqual(list(root.iterdir()), [latest])
            history = root / "history.jsonl"
            for content in (None, b"short"):
                if content is not None:
                    history.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "missing or shorter"):
                    restore_history(history, {"history_bytes": 100})

    def test_slurm_wrapper_requeues_only_planned_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv/bin").mkdir(parents=True)
            python = root / ".venv/bin/python"
            python.write_text("""#!/usr/bin/env bash
if [[ "$3" == - ]]; then
  cat >/dev/null
  echo /tmp/test-cache
  exit 0
fi
if [[ "$SEND_SIGNAL" == 0 ]]; then exit "$TRAINER_EXIT"; fi
trap 'exit "$TRAINER_EXIT"' USR1
touch ready
while true; do sleep 0.05 & wait $!; done
""")
            scontrol = root / "scontrol"
            scontrol.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" > requeue-call\n')
            python.chmod(0o755)
            scontrol.chmod(0o755)
            # Model the shell being descheduled after a signal interrupts wait,
            # allowing the trainer to exit before the shell checks its status.
            bash_env = root / "bash-env"
            bash_env.write_text("""wait() {
  builtin wait "$@"
  local waited_status=$?
  if (( waited_status > 128 )); then sleep 0.1; fi
  return "$waited_status"
}
""")
            cases = ((0, False, False), (1, False, False), (75, False, False),
                     (1, True, False), (75, True, False),
                     (0, True, True), (1, True, True), (75, True, True))
            for code, send_signal, delayed_wait in cases:
                with self.subTest(code=code, signal=send_signal, delayed_wait=delayed_wait):
                    (root / "ready").unlink(missing_ok=True)
                    (root / "requeue-call").unlink(missing_ok=True)
                    env = {**os.environ, "SLURM_SUBMIT_DIR": str(root), "SLURM_JOB_ID": "123",
                           "PATH": f"{root}:{os.environ['PATH']}", "TRAINER_EXIT": str(code),
                           "SEND_SIGNAL": str(int(send_signal)),
                           "BASH_ENV": str(bash_env) if delayed_wait else "/dev/null"}
                    process = subprocess.Popen(["bash", str(REPO / "scripts/slurm_train.sbatch"), "config.yaml"],
                                               cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    try:
                        if send_signal:
                            deadline = time.monotonic() + 10
                            while not (root / "ready").exists() and time.monotonic() < deadline:
                                time.sleep(0.02)
                            self.assertTrue((root / "ready").exists())
                            os.kill(process.pid, signal.SIGUSR1)
                        stdout, stderr = process.communicate(timeout=10)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.communicate()
                    requeued = code == 75 and send_signal
                    self.assertEqual(process.returncode, 0 if requeued else code, stdout + stderr)
                    self.assertEqual((root / "requeue-call").exists(), requeued)
                    if requeued:
                        self.assertEqual((root / "requeue-call").read_text().strip(), "requeue 123")
