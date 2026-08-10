"""Train the post-hoc rollout heads (latent dynamics predictor + small pixel
decoder) on top of a FROZEN, already-pretrained encoder. See
src/objectives/rollout_heads.py for what these heads do and why.

Usage:
    uv run python scripts/train_rollout_heads.py \
        --config configs/rollout_heads_active_matter_jepa.yaml --data-root /workspace/data
"""

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_encoder import load_encoder
from src.data.well import ClipSpec
from src.objectives.rollout_heads import RolloutHeadModel
from src.train import infinite_loader, lr_at, make_optimizer, set_seed


class PairedWindowMemmapDataset(torch.utils.data.Dataset):
    """Yields consecutive, disjoint (window_a, window_b) pairs: window_b
    starts exactly where window_a ends. Same memmap layout as
    src.data.well.MemmapClipDataset (produced by scripts/preprocess_memmap.py)
    -- this class isn't added there since it's only used by this script."""

    def __init__(self, base_path: str, dataset_name: str, split: str, n_frames: int = 8):
        d = Path(base_path) / "memmap" / dataset_name
        self.path = d / f"{split}.npy"
        meta = d / f"{split}.meta.json"
        if not self.path.exists() or not meta.exists():
            raise FileNotFoundError(
                f"{self.path} (+ meta) missing — run scripts/preprocess_memmap.py first"
            )
        if '"NCTHW"' not in meta.read_text():
            raise ValueError(f"{self.path} has stale layout — re-run preprocessing")
        self.mm = np.load(self.path, mmap_mode="r")
        self.n_frames = n_frames
        n_traj, _, t, _, _ = self.mm.shape
        self.windows_per_traj = t - 2 * n_frames + 1
        if self.windows_per_traj <= 0:
            raise ValueError(
                f"trajectory has {t} frames, too short for two {n_frames}-frame windows"
            )
        self.length = n_traj * self.windows_per_traj

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        traj, off = divmod(idx, self.windows_per_traj)
        pair = np.array(self.mm[traj, :, off : off + 2 * self.n_frames])  # (C, 2*n_frames, H, W)
        pair = torch.from_numpy(pair)
        return {"window_a": pair[:, : self.n_frames], "window_b": pair[:, self.n_frames :]}

    @property
    def spec(self) -> ClipSpec:
        _, c, _, h, w = self.mm.shape
        return ClipSpec(n_channels=c, n_frames=self.n_frames, height=h, width=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None, help="overrides data.base_path")
    ap.add_argument("--steps", type=int, default=None, help="overrides optim.total_steps")
    ap.add_argument("--lr", type=float, default=None, help="overrides optim.lr")
    ap.add_argument("--out", default=None, help="overrides output dir")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg["data"]["base_path"] = args.data_root
    if args.steps:
        cfg["optim"]["total_steps"] = args.steps
    if args.lr:
        cfg["optim"]["lr"] = args.lr

    run_name = cfg["run_name"]
    out_dir = Path(args.out or cfg.get("out_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    set_seed(cfg.get("seed", 0))

    encoder, _enc_cfg, enc_spec = load_encoder(cfg["encoder_ckpt"])
    encoder = encoder.to(device)

    dcfg = cfg["data"]
    dataset = PairedWindowMemmapDataset(
        base_path=os.path.expanduser(dcfg["base_path"]),
        dataset_name=dcfg["dataset_name"],
        split="train",
        n_frames=dcfg.get("n_frames", 8),
    )
    spec = dataset.spec
    if asdict(spec) != asdict(enc_spec):
        raise ValueError(
            f"dataset spec {spec} doesn't match encoder checkpoint spec {enc_spec} "
            f"({cfg['encoder_ckpt']}) — wrong dataset/encoder pairing?"
        )
    print(f"dataset {dcfg['dataset_name']}: {len(dataset)} window pairs, spec={spec}")

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

    model = RolloutHeadModel(encoder, cfg["heads"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"rollout heads: {n_params/1e6:.1f}M trainable params (predictor+decoder)")

    ocfg = cfg["optim"]
    optimizer = make_optimizer(model, ocfg)
    total_steps = ocfg["total_steps"]

    start_step = 0
    latest = out_dir / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.predictor.load_state_dict(ckpt["predictor"])
        model.decoder.load_state_dict(ckpt["decoder"])
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
    save_every = cfg.get("save_every", 500)

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        lr = lr_at(step, ocfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(batches)
        window_a = batch["window_a"].to(device, non_blocking=True).float()
        window_b = batch["window_b"].to(device, non_blocking=True).float()

        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss, metrics = model(window_a, window_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            ocfg.get("grad_clip", 1.0),
        )
        optimizer.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        if (step + 1) % log_every == 0:
            ips = log_every * window_a.size(0) / (time.time() - t0)
            t0 = time.time()
            line = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step {step+1}/{total_steps} lr={lr:.2e} {line} pairs/s={ips:.1f}")
            if wandb_run:
                wandb_run.log(
                    {**metrics, "lr": lr, "grad_norm": grad_norm.item(), "pairs_per_s": ips},
                    step=step + 1,
                )

        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            ckpt = {
                "predictor": model.predictor.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
                "spec": asdict(spec),
                "encoder_ckpt": cfg["encoder_ckpt"],
            }
            torch.save(ckpt, latest)

    print("training complete")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
