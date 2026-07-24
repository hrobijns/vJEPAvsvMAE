"""Unified training entrypoint for both objectives.

Usage:
    python -m src.train --config configs/active_matter_jepa.yaml
    python -m src.train --config configs/debug_mae.yaml --data-root ~/well_data

Everything except the `objective` section of the config is shared between the
JEPA and MAE runs of a pair.
"""

import argparse
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.well import ClipSpec, WellClipDataset
from src.models.vit import build_encoder
from src.objectives.jepa import JEPAModel
from src.objectives.mae import MAEModel

MILESTONE_FRACS = (0.25, 0.5, 0.75, 1.0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    ap.add_argument("--out", default=None, help="overrides output dir")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["base_path"] = args.data_root
    if args.steps:
        cfg["optim"]["total_steps"] = args.steps

    run_name = cfg["run_name"]
    out_dir = Path(args.out or cfg.get("out_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    set_seed(cfg.get("seed", 0))

    dcfg = cfg["data"]
    dataset = WellClipDataset(
        base_path=os.path.expanduser(dcfg["base_path"]),
        dataset_name=dcfg["dataset_name"],
        split="train",
        n_frames=dcfg.get("n_frames", 8),
    )
    spec = dataset.spec
    print(f"dataset {dcfg['dataset_name']}: {len(dataset)} clips, spec={spec}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 4),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=dcfg.get("num_workers", 4) > 0,
    )
    batches = infinite_loader(loader)

    model = build_model(cfg["objective_name"], spec, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {cfg['objective_name']}, {n_params/1e6:.1f}M trainable params")

    ocfg = cfg["optim"]
    optimizer = make_optimizer(model, ocfg)
    total_steps = ocfg["total_steps"]

    start_step = 0
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {latest} at step {start_step}")

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
    milestones = {int(total_steps * f): f for f in MILESTONE_FRACS}

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        lr = lr_at(step, ocfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(batches)
        clip = batch["clip"].to(device, non_blocking=True)

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

        if (step + 1) % save_every == 0 or (step + 1) in milestones:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
                "spec": asdict(spec),
            }
            torch.save(ckpt, latest)
            if (step + 1) in milestones:
                frac = milestones[step + 1]
                torch.save(
                    {"encoder": model.encoder.state_dict(), "config": cfg,
                     "spec": asdict(spec), "step": step + 1},
                    out_dir / f"encoder_{int(frac*100):03d}pct.pt",
                )

    print("training complete")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
