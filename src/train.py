"""Unified training entrypoint for both objectives.

Usage:
    python -m src.train --config configs/active_matter_jepa.yaml
    python -m src.train --config configs/debug_mae.yaml --data-root ~/well_data

Everything except the `objective` section of the config is shared between the
JEPA and MAE runs of a pair.
"""

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

# DataLoader workers pass tensors back to the main process via /dev/shm by
# default -- with the training data itself also served from /dev/shm (for
# throughput, see src/data/well.py), and several concurrent training
# processes each running many workers, that default strategy exhausts shm
# ("unable to allocate shared memory... Resource temporarily unavailable").
# file_system sharing uses regular temp files instead, sidestepping it.
torch.multiprocessing.set_sharing_strategy("file_system")

from src.data.well import ClipSpec, MemmapClipDataset, WellClipDataset, train_valid_trajectory_split
from src.models.vit import build_encoder
from src.objectives.jepa import JEPAModel
from src.objectives.mae import MAEModel

MILESTONE_FRACS = (0.25, 0.5, 0.75, 1.0)


def retry_io(fn, *args, retries=15, delay=3.0, max_delay=60.0, **kwargs):
    """The network filesystem backing /workspace has shown transient write
    failures (observed: "Disk quota Exceeded" that cleared on the very next
    retry, and sustained "[Errno 5] Input/output error" spells lasting well
    past a minute) severe enough to raise an uncaught OSError and kill the
    whole process -- expensive given runs take hours, and especially costly
    during unattended stretches with no one to notice and relaunch a dead
    run. Retry checkpoint/log writes with exponential backoff (capped) before
    giving up for real; the original flat 8-retries/3s (24s total) budget
    survived brief blips but not the sustained multi-minute ones observed in
    practice -- 15 attempts with a 60s cap totals ~12.5 minutes, long enough
    to ride out those without hanging indefinitely on a truly dead mount."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"WARNING: transient I/O error ({e}), retrying in {delay}s "
                  f"(attempt {attempt+1}/{retries})", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # cudnn.deterministic=True was tried for tighter multi-seed reproducibility
    # but cost a ~4x throughput hit on A40 (measured: 15 vs 117 clips/s at
    # steady state) — not worth it against a hard wall-clock budget, and it
    # was only ever a "reduces, doesn't eliminate" nicety on top of the seed
    # itself, not part of the actual training recipe (steps/LR/batch/seed are
    # what's held identical across the comparison). cudnn.benchmark=True is
    # safe here since every run in a given config uses a fixed clip shape.


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp, max_batches=None):
    """Mean loss + mean collapse-diagnostic metrics over the val loader."""
    model.eval()
    totals, n = {}, 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        clip = batch["clip"].to(device, non_blocking=True).float()
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            _, metrics = model(clip)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / n for k, v in totals.items()}


def build_model(objective: str, spec: ClipSpec, cfg: dict) -> torch.nn.Module:
    encoder = build_encoder(spec, cfg.get("encoder", {}))
    obj_cfg = cfg.get("objective", {})
    if objective == "jepa":
        return JEPAModel(encoder, obj_cfg)
    if objective == "mae":
        return MAEModel(encoder, obj_cfg)
    raise ValueError(f"unknown objective {objective!r}")


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # target encoder
        if p.ndim <= 1 or name.endswith("mask_token"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.get("weight_decay", 0.05)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
    )


def lr_at(step: int, cfg: dict) -> float:
    warmup = cfg.get("warmup_steps", 2000)
    total = cfg["total_steps"]
    base = cfg["lr"]
    min_lr = cfg.get("min_lr", base * 0.01)
    if step < warmup:
        return base * step / max(warmup, 1)
    frac = (step - warmup) / max(total - warmup, 1)
    return min_lr + (base - min_lr) * (math.cos(math.pi * min(frac, 1.0)) + 1) / 2


def infinite_loader(loader):
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="overrides data.base_path")
    ap.add_argument("--steps", type=int, default=None, help="overrides total_steps")
    ap.add_argument("--lr", type=float, default=None, help="overrides optim.lr")
    ap.add_argument("--mask-ratio", type=float, default=None, help="overrides objective.mask_ratio")
    ap.add_argument("--out", default=None, help="overrides output dir")
    ap.add_argument("--seed", type=int, default=None, help="overrides seed; appends _seed{N} to run_name")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["base_path"] = args.data_root
    if args.steps:
        cfg["optim"]["total_steps"] = args.steps
    if args.lr:
        cfg["optim"]["lr"] = args.lr
    if args.mask_ratio:
        cfg["objective"]["mask_ratio"] = args.mask_ratio
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["run_name"] = f"{cfg['run_name']}_seed{args.seed}"

    run_name = cfg["run_name"]
    out_dir = Path(args.out or cfg.get("out_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    set_seed(cfg.get("seed", 0))

    dcfg = cfg["data"]
    base_path = os.path.expanduser(dcfg["base_path"])
    dataset_name = dcfg["dataset_name"]
    n_frames = dcfg.get("n_frames", 8)

    if dcfg.get("memmap", False):
        # Trajectory-disjoint pseudo-validation carved out of `train` — the
        # Well's shipped `valid` split is too small/regime-degenerate for
        # stable checkpoint selection (see train_valid_trajectory_split's
        # docstring). Used only for checkpoint selection, not for claims
        # about generalization to unseen physical regimes.
        n_traj = MemmapClipDataset(base_path, dataset_name, split="train", n_frames=n_frames).mm.shape[0]
        fit_idx, valid_idx = train_valid_trajectory_split(n_traj, dcfg.get("valid_stride", 8))
        fit_ds = MemmapClipDataset(base_path, dataset_name, split="train", n_frames=n_frames, trajectories=fit_idx)
        val_ds = MemmapClipDataset(base_path, dataset_name, split="train", n_frames=n_frames, trajectories=valid_idx)
    else:
        fit_ds = WellClipDataset(base_path, dataset_name, split="train", n_frames=n_frames)
        val_ds = WellClipDataset(base_path, dataset_name, split="valid", n_frames=n_frames)

    spec = fit_ds.spec
    print(f"dataset {dataset_name}: {len(fit_ds)} fit clips, {len(val_ds)} val clips, spec={spec}")

    loader = torch.utils.data.DataLoader(
        fit_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=dcfg.get("num_workers", 4) > 0,
    )
    batches = infinite_loader(loader)

    # A second large persistent worker pool alongside the fit loader's
    # deadlocked in practice (measured: hung indefinitely on its first
    # iteration) -- the val set is small (~26 batches at batch 64 for
    # active_matter) and isn't throughput-critical, so a couple of
    # non-persistent workers is both sufficient and avoids the contention.
    val_num_workers = min(2, dcfg.get("num_workers", 4))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )

    model = build_model(cfg["objective_name"], spec, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {cfg['objective_name']}, {n_params/1e6:.1f}M trainable params")

    ocfg = cfg["optim"]
    optimizer = make_optimizer(model, ocfg)
    total_steps = ocfg["total_steps"]

    start_step = 0
    best_val_loss = float("inf")
    best_val_step = 0
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = retry_io(torch.load, latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_step = ckpt.get("best_val_step", 0)
        print(f"resumed from {latest} at step {start_step}")

    # history.jsonl is written every log_every steps (frequent) -- every
    # training crash observed in practice traced to exactly this write
    # hitting the flaky network mount backing /workspace (never a checkpoint
    # save, which is 20x less frequent and always survived). The container's
    # own root filesystem is a separate, local, non-network disk that has
    # shown no such failures -- write the hot path there instead, and only
    # touch /workspace at the same infrequent cadence as checkpoint saves
    # (already retry_io-hardened), where a local disk loss costs at most
    # save_every steps of curve granularity, never training progress itself
    # (that's recovered from latest.pt regardless).
    local_history_dir = Path("/root/.vjepa_local_history") / run_name
    local_history_dir.mkdir(parents=True, exist_ok=True)
    local_history_path = local_history_dir / "history.jsonl"
    workspace_history_path = out_dir / "history.jsonl"
    if workspace_history_path.exists() and not local_history_path.exists():
        # Resuming on a fresh local disk (e.g. after a pod restart) -- reseed
        # the local mirror from the last durable copy so the next sync still
        # writes a complete file, not just the fragment since this restart.
        shutil.copyfile(workspace_history_path, local_history_path)
    history_f = open(local_history_path, "a")

    def _write_history_line(step: int, phase: str, metrics: dict):
        history_f.write(json.dumps({"step": step, "phase": phase, **metrics}) + "\n")
        history_f.flush()

    def log_history(step: int, phase: str, metrics: dict):
        _write_history_line(step, phase, metrics)

    def sync_history_to_workspace():
        history_f.flush()
        retry_io(shutil.copyfile, local_history_path, workspace_history_path)

    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False) and not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"].get("project", "vjepa-vmae-well"),
            name=run_name,
            config=cfg,
            resume="allow",
            id=cfg["wandb"].get("id", run_name),
        )

    log_every = cfg.get("log_every", 50)
    save_every = cfg.get("save_every", 1000)
    img_every = cfg.get("img_every", 5000)
    val_every = cfg.get("val_every", 2000)
    val_max_batches = cfg.get("val_max_batches", None)
    milestones = {int(total_steps * f): f for f in MILESTONE_FRACS}
    first_val_feat_std = {}  # baseline for the in-loop collapse warning

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        lr = lr_at(step, ocfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(batches)
        # memmaps store fp16 to halve host->device bytes; cast on-GPU
        clip = batch["clip"].to(device, non_blocking=True).float()

        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss, metrics = model(clip)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            ocfg.get("grad_clip", 1.0),
        )
        optimizer.step()
        model.post_step(step, total_steps)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        if (step + 1) % log_every == 0:
            ips = log_every * clip.size(0) / (time.time() - t0)
            t0 = time.time()
            line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step {step+1}/{total_steps} lr={lr:.2e} {line} clips/s={ips:.1f}")
            log_history(step + 1, "train", {**metrics, "lr": lr, "grad_norm": grad_norm.item()})
            if wandb_run:
                wandb_run.log(
                    {**metrics, "lr": lr, "grad_norm": grad_norm.item(),
                     "clips_per_s": ips},
                    step=step + 1,
                )

        if wandb_run and isinstance(model, MAEModel) and (step + 1) % img_every == 0:
            import wandb

            model.eval()
            orig, recon = model.reconstruction_figure(clip[:1])
            model.train()
            wandb_run.log(
                {"recon": [wandb.Image(orig, caption="original"),
                           wandb.Image(recon, caption="reconstruction")]},
                step=step + 1,
            )

        if (step + 1) % val_every == 0 or (step + 1) == total_steps:
            val_metrics = evaluate(model, val_loader, device, use_amp, val_max_batches)
            line = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"step {step+1}/{total_steps} VAL {line}")
            log_history(step + 1, "val", val_metrics)
            if wandb_run:
                wandb_run.log({f"val_{k}": v for k, v in val_metrics.items()}, step=step + 1)

            for k, v in val_metrics.items():
                if not k.endswith("_feat_std"):
                    continue
                if k not in first_val_feat_std:
                    first_val_feat_std[k] = v
                elif v < 0.1 * first_val_feat_std[k]:
                    print(
                        f"WARNING: possible representation collapse — {k}={v:.4g} "
                        f"is <10% of its first-eval value {first_val_feat_std[k]:.4g}"
                    )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_step = step + 1
                retry_io(
                    torch.save,
                    {"encoder": model.encoder.state_dict(), "config": cfg,
                     "spec": asdict(spec), "step": step + 1,
                     "val_loss": best_val_loss},
                    out_dir / "encoder_best_val.pt",
                )

        if (step + 1) % save_every == 0 or (step + 1) in milestones:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
                "spec": asdict(spec),
                "best_val_loss": best_val_loss,
                "best_val_step": best_val_step,
            }
            retry_io(torch.save, ckpt, latest)
            sync_history_to_workspace()
            if (step + 1) in milestones:
                frac = milestones[step + 1]
                retry_io(
                    torch.save,
                    {"encoder": model.encoder.state_dict(), "config": cfg,
                     "spec": asdict(spec), "step": step + 1},
                    out_dir / f"encoder_{int(frac*100):03d}pct.pt",
                )

    print(f"training complete — best val loss {best_val_loss:.4f} at step {best_val_step}")
    sync_history_to_workspace()
    history_f.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
