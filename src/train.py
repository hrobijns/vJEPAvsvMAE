"""Unified training entrypoint for current-clip and future-clip JEPA and MAE.

Usage:
    python -m src.train --config configs/active_matter_jepa.yaml
    python -m src.train --config configs/rayleigh_benard_mae.yaml --data-root ~/well_data

The encoders and data pipeline are shared; objective heads and learning rates
are configured separately.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.well import (
    ClipSpec,
    MemmapClipDataset,
    WellClipDataset,
    train_valid_trajectory_split,
)
from src.models.vit import build_encoder
from src.objectives.jepa import JEPAModel
from src.objectives.mae import MAEModel
from src.evaluation.artifacts import canonical_hash, provenance

MILESTONE_FRACS = (0.25, 0.5, 0.75, 1.0)
CONTINUATION_VERSION = 1
CONTINUATION_EXIT = 75


def training_identity(config, dataset, fit_indices, valid_indices, code_sha256=None):
    """Bind resume to data, sampling, architecture, and optimization settings."""
    return canonical_hash(
        {
            "continuation_version": CONTINUATION_VERSION,
            "code_sha256": code_sha256,
            "source": dataset.identity,
            "fit": fit_indices,
            "valid": valid_indices,
            "sampling": {
                key: config["data"].get(key)
                for key in ("n_frames", "frame_limit", "valid_stride")
            },
            "objective_name": config["objective_name"],
            "objective": config["objective"],
            "encoder": config["encoder"],
            "optim": config["optim"],
            "seed": config["seed"],
            "bf16": config.get("bf16", True),
            "validation": [config.get("val_every", 2000), config.get("val_max_batches")],
            "images": [config.get("wandb", {}).get("enabled", False), config.get("img_every", 5000)],
        }
    )


def validate_resume(checkpoint, identity):
    if checkpoint.get("training_identity") != identity:
        raise ValueError(
            "checkpoint data/configuration differs or lacks resume identity; use a new run directory"
        )


def training_exposure(step, optim, fit_clips, valid_clips):
    """Count input examples used by updates; a future pair counts once."""
    processed = step * optim["batch_size"]
    planned = optim["total_steps"] * optim["batch_size"]
    return {
        "eligible_train_clips": fit_clips,
        "eligible_validation_clips": valid_clips,
        "planned_clips": planned,
        "planned_equivalent_passes": planned / fit_clips,
        "clips_processed": processed,
        "equivalent_passes": processed / fit_clips,
    }


def retry_io(fn, *args, retries=15, delay=3.0, max_delay=60.0, **kwargs):
    """Retry transient filesystem failures with bounded exponential backoff."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(
                f"WARNING: transient I/O error ({e}), retrying in {delay}s "
                f"(attempt {attempt + 1}/{retries})",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seeds do not force deterministic CUDA kernels; retain the workshop recipe.


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("resume requires the same number of visible CUDA devices")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    """An interrupted write must leave the previous checkpoint usable."""
    path = Path(path)

    def write():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    retry_io(write)


def restore_history(path, checkpoint):
    """Discard log writes newer than the last durable training state."""
    if not path.exists() or path.stat().st_size < checkpoint["history_bytes"]:
        raise ValueError("history is missing or shorter than the saved checkpoint")
    with path.open("r+b") as handle:
        handle.truncate(checkpoint["history_bytes"])


def export_encoders(checkpoint, model, out_dir, milestones):
    """Derived exports are recoverable from latest after an interrupted save."""
    step = checkpoint["step"]
    export = {
        key: checkpoint[key]
        for key in ("config", "spec", "step", "training_identity", "data_exposure")
    }
    export["encoder"] = model.encoder.state_dict()
    if step == checkpoint["best_val_step"]:
        atomic_save({**export, "val_loss": checkpoint["best_val_loss"]}, out_dir / "encoder_best_val.pt")
    if step in milestones:
        atomic_save(export, out_dir / f"encoder_{int(milestones[step] * 100):03d}pct.pt")


def batch_to_device(batch, device):
    return {key: value.to(device, non_blocking=True).float() for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp, max_batches=None):
    """Mean loss + mean collapse-diagnostic metrics over the val loader."""
    model.eval()
    totals, n = {}, 0
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            _, metrics = model(**batch)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / n for k, v in totals.items()}


def build_model(objective: str, spec: ClipSpec, cfg: dict) -> torch.nn.Module:
    encoder = build_encoder(spec, cfg.get("encoder", {}))
    obj_cfg = cfg.get("objective", {})
    if objective in ("jepa", "jepa_future"):
        return JEPAModel(encoder, obj_cfg, future=objective == "jepa_future")
    if objective in ("mae", "mae_future"):
        return MAEModel(encoder, obj_cfg, future=objective == "mae_future")
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


def training_batches(size, batch_size, seed, start_step):
    """Resume by consumed updates, independently of worker prefetch."""
    per_pass = size // batch_size
    if per_pass == 0:
        raise ValueError("batch size exceeds the available training clips")
    pass_number, offset = divmod(start_step, per_pass)
    while True:
        # Hash both coordinates so seeds 1 and 2 do not share shifted passes.
        pass_seed = int.from_bytes(hashlib.sha256(f"{seed}:{pass_number}".encode()).digest()[:8], "little")
        order = torch.randperm(size, generator=torch.Generator().manual_seed(pass_seed)).tolist()
        for batch in range(offset, per_pass):
            yield order[batch * batch_size : (batch + 1) * batch_size]
        pass_number, offset = pass_number + 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="overrides data.base_path")
    ap.add_argument("--steps", type=int, default=None, help="overrides total_steps")
    ap.add_argument("--lr", type=float, default=None, help="overrides optim.lr")
    ap.add_argument(
        "--mask-ratio", type=float, default=None, help="overrides objective.mask_ratio"
    )
    ap.add_argument("--out", default=None, help="overrides output dir")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="overrides seed; appends _seed{N} to run_name",
    )
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
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    run_name = cfg["run_name"]
    out_dir = Path(args.out or cfg.get("out_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # The lock covers initialization and recovery as well as optimizer updates.
    with (out_dir / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another trainer is using {out_dir}") from error
        stopping = False

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True

        previous = signal.signal(signal.SIGUSR1, request_stop)
        try:
            return train(cfg, out_dir, lambda: stopping)
        finally:
            signal.signal(signal.SIGUSR1, previous)


def train(cfg, out_dir, stop_requested):
    code_sha256 = provenance()["code_sha256"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    set_seed(cfg.get("seed", 0))

    dcfg = cfg["data"]
    base_path = os.path.expanduser(dcfg["base_path"])
    dataset_name = dcfg["dataset_name"]
    n_frames = dcfg.get("n_frames", 8)

    dataset_type = MemmapClipDataset if dcfg.get("memmap", False) else WellClipDataset
    # Both backends reserve validation trajectories within official train.
    # The official valid and test splits belong exclusively to downstream probes.
    if "frame_limit" not in dcfg:
        raise ValueError("data.frame_limit must explicitly specify temporal support")
    data_kwargs = dict(
        base_path=base_path,
        dataset_name=dataset_name,
        split="train",
        n_frames=n_frames,
        frame_limit=dcfg["frame_limit"],
        future=cfg["objective_name"] in ("jepa_future", "mae_future"),
    )
    inventory = dataset_type(**data_kwargs)
    fit_idx, valid_idx = train_valid_trajectory_split(
        inventory.n_traj, dcfg.get("valid_stride", 8)
    )
    fit_ds = dataset_type(**data_kwargs, trajectories=fit_idx)
    val_ds = dataset_type(**data_kwargs, trajectories=valid_idx)
    identity = training_identity(cfg, inventory, fit_idx, valid_idx, code_sha256)

    spec = fit_ds.spec
    print(
        f"dataset {dataset_name}: {len(fit_ds)} fit clips, {len(val_ds)} val clips, spec={spec}"
    )

    if len(fit_ds) < cfg["optim"]["batch_size"]:
        raise ValueError("batch size exceeds the available training clips")

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
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    model = build_model(cfg["objective_name"], spec, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {cfg['objective_name']}, {n_params / 1e6:.1f}M trainable params")

    ocfg = cfg["optim"]
    optimizer = make_optimizer(model, ocfg)
    total_steps = ocfg["total_steps"]

    start_step = 0
    best_val_loss = float("inf")
    best_val_step = 0
    latest = out_dir / "latest.pt"
    history_path = out_dir / "history.jsonl"
    milestones = {int(total_steps * f): f for f in MILESTONE_FRACS}
    first_val_feat_std = {}
    ckpt = None
    if latest.exists():
        ckpt = retry_io(torch.load, latest, map_location="cpu", weights_only=False)
        if (ckpt.get("continuation_version") != CONTINUATION_VERSION
                or not {"rng", "history_bytes", "first_val_feat_std", "code_sha256"} <= ckpt.keys()):
            raise ValueError("checkpoint lacks complete continuation state; use a new run directory")
        validate_resume(ckpt, identity)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_step = ckpt.get("best_val_step", 0)
        first_val_feat_std = ckpt["first_val_feat_std"]
        restore_history(history_path, ckpt)
        export_encoders(ckpt, model, out_dir, milestones)
        print(f"resumed from {latest} at step {start_step}")
    elif history_path.exists() and history_path.stat().st_size:
        raise ValueError("history exists without a checkpoint; use a new run directory")

    if start_step == total_steps:
        print("training already complete; encoder exports checked")
        return 0
    if stop_requested():
        print("stop requested before any updates; no automatic continuation")
        return 1

    loader = torch.utils.data.DataLoader(
        fit_ds,
        batch_sampler=training_batches(len(fit_ds), ocfg["batch_size"], cfg["seed"], start_step),
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        persistent_workers=dcfg.get("num_workers", 4) > 0,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    batches = iter(loader)

    exposure = training_exposure(start_step, ocfg, len(fit_ds), len(val_ds))
    print(f"training exposure: {json.dumps(exposure)}")

    history_f = history_path.open("ab")

    def log_history(step: int, phase: str, metrics: dict):
        history_f.write(
            json.dumps(
                {"step": step, "phase": phase, **metrics, "data_exposure": exposure}
            ).encode() + b"\n"
        )
        history_f.flush()

    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"].get("project", "vjepa-vmae-well"),
            name=cfg["run_name"],
            config=cfg,
            resume="allow",
            id=cfg["wandb"].get("id", cfg["run_name"]),
        )

    log_every = cfg.get("log_every", 50)
    save_every = cfg.get("save_every", 1000)
    img_every = cfg.get("img_every", 5000)
    val_every = cfg.get("val_every", 2000)
    val_max_batches = cfg.get("val_max_batches", None)

    model.train()
    if ckpt is not None:
        restore_rng(ckpt["rng"])
    # Release the CPU copy of model/optimizer weights before the first update.
    ckpt = None
    t0 = time.time()
    last_log_step = start_step
    for step in range(start_step, total_steps):
        lr = lr_at(step, ocfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(batches)
        # memmaps store fp16 to halve host->device bytes; cast on-GPU
        batch = batch_to_device(batch, device)

        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss, metrics = model(**batch)
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

        exposure = training_exposure(step + 1, ocfg, len(fit_ds), len(val_ds))
        if (step + 1) % log_every == 0:
            ips = (step + 1 - last_log_step) * batch["clip"].size(0) / (time.time() - t0)
            last_log_step = step + 1
            t0 = time.time()
            line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(
                f"step {step + 1}/{total_steps} lr={lr:.2e} {line} clips/s={ips:.1f} "
                f"clips={exposure['clips_processed']} passes={exposure['equivalent_passes']:.3f}"
            )
            log_history(
                step + 1, "train", {**metrics, "lr": lr, "grad_norm": grad_norm.item()}
            )
            if wandb_run:
                wandb_run.log(
                    {
                        **metrics,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "clips_per_s": ips,
                        **{f"data/{k}": v for k, v in exposure.items()},
                    },
                    step=step + 1,
                )

        if wandb_run and isinstance(model, MAEModel) and (step + 1) % img_every == 0:
            import wandb

            model.eval()
            orig, recon = model.reconstruction_figure(
                **{key: value[:1] for key, value in batch.items()}
            )
            model.train()
            captions = (
                ("future target (normalized patches)", "future prediction (normalized patches)")
                if model.future and model.norm_pix
                else ("future target", "future prediction")
                if model.future
                else ("original", "reconstruction")
            )
            wandb_run.log(
                {
                    "future_prediction" if model.future else "recon": [
                        wandb.Image(orig, caption=captions[0]),
                        wandb.Image(recon, caption=captions[1]),
                    ]
                },
                step=step + 1,
            )

        validated = (step + 1) % val_every == 0 or (step + 1) == total_steps
        if validated:
            val_metrics = evaluate(model, val_loader, device, use_amp, val_max_batches)
            line = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"step {step + 1}/{total_steps} VAL {line}")
            log_history(step + 1, "val", val_metrics)
            if wandb_run:
                wandb_run.log(
                    {f"val_{k}": v for k, v in val_metrics.items()}, step=step + 1
                )

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

        stopping = stop_requested()
        if (step + 1) % save_every == 0 or (step + 1) in milestones or validated or stopping:
            history_f.flush()
            os.fsync(history_f.fileno())
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
                "spec": asdict(spec),
                "training_identity": identity,
                "data_exposure": exposure,
                "best_val_loss": best_val_loss,
                "best_val_step": best_val_step,
                "first_val_feat_std": first_val_feat_std,
                "continuation_version": CONTINUATION_VERSION,
                "code_sha256": code_sha256,
                "rng": rng_state(),
                "history_bytes": history_f.tell(),
            }
            atomic_save(ckpt, latest)
            export_encoders(ckpt, model, out_dir, milestones)

        if stopping and step + 1 < total_steps:
            print(f"planned stop at step {step + 1}; saved {latest}", flush=True)
            history_f.close()
            if wandb_run:
                wandb_run.finish()
            return CONTINUATION_EXIT

    print(
        f"training complete — best val loss {best_val_loss:.4f} at step {best_val_step}"
    )
    history_f.close()
    if wandb_run:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
