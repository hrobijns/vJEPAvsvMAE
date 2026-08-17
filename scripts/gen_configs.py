"""Regenerate configs/. Run: uv run python scripts/gen_configs.py"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"

DATASETS = ["active_matter", "shear_flow", "rayleigh_benard"]
OBJECTIVES = {
    "jepa": {
        "lr": 1.0e-4,
        "objective": {
            "mask_ratio": 0.9,
            # Narrowed from 384 (= encoder width) to 192: I-JEPA/V-JEPA's own
            # design keeps the predictor narrower than the encoder (a
            # deliberate bottleneck -- forces the encoder to be informative
            # rather than letting a powerful predictor shortcut the
            # objective); at encoder_dim=384 that mechanism was absent since
            # predictor_dim was also 384. This also brings the predictor's
            # parameter count (~2.8M) in line with the MAE decoder's
            # (~2.2-2.9M), removing a head-capacity confound between the two
            # objectives' pretraining budgets.
            "predictor_dim": 192,
            "predictor_depth": 6,
            "predictor_heads": 6,
            "ema_start": 0.996,
            "ema_end": 1.0,
        },
    },
    "mae": {
        # 5e-5 chosen by per-channel probe R^2 on the active_matter LR sweep
        # (2026-07-25); matches VideoMAE's canonical LR scaled to batch 64.
        "lr": 5.0e-5,
        "objective": {
            "mask_ratio": 0.9,
            "decoder_dim": 192,
            "decoder_depth": 4,
            "decoder_heads": 6,
            "norm_pix": True,
        },
    },
}


def make(dataset: str, objective: str, debug: bool = False) -> dict:
    spec = OBJECTIVES[objective]
    name = f"{'debug' if debug else dataset}_{objective}"
    return {
        "run_name": name,
        "objective_name": objective,
        "out_dir": "runs",
        "seed": 0,
        "bf16": True,
        "log_every": 5 if debug else 50,
        "save_every": 25 if debug else 1000,
        "img_every": 25 if debug else 5000,
        "data": {
            "base_path": "~/well_data" if debug else "/workspace/data",
            "dataset_name": "turbulent_radiative_layer_2D" if debug else dataset,
            "n_frames": 8,
            # 16 workers was tuned for the pre-memmap WellDataset path
            # (~0.5s/clip); memmap reads are near-instant, so fewer workers
            # are plenty -- and with several concurrent training processes
            # sharing one pod, many worker pools competing for /dev/shm-backed
            # DataLoader IPC (semaphores etc, unavoidably shm-backed
            # regardless of tensor-sharing strategy) can exhaust it.
            # Measured directly: with 6 full-scale jobs concurrent (24 workers
            # total at num_workers=4), single memmap __getitem__ calls slowed
            # to 0.6-3.3s (vs ~1ms isolated) from memory-bandwidth contention;
            # 3 concurrent jobs (12 workers) ran at full speed during the LR
            # sweep. Dropped to 2/job so all 6 pairs can still run in parallel
            # without crossing that contention cliff.
            "num_workers": 0 if debug else 2,
            "memmap": not debug,
            "valid_stride": 8,
        },
        "val_every": 10 if debug else 2000,
        "encoder": {
            "patch_t": 2,
            "patch_h": 16,
            "patch_w": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
        "objective": dict(spec["objective"]),
        "optim": {
            "batch_size": 4 if debug else 64,
            "total_steps": 50 if debug else 100_000,
            "warmup_steps": 10 if debug else 5000,
            "lr": spec["lr"],
            "min_lr": 1.0e-6,
            "weight_decay": 0.05,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
        },
        "wandb": {"enabled": not debug, "project": "vjepa-vmae-well"},
    }


def main():
    CONFIG_DIR.mkdir(exist_ok=True)
    for objective in OBJECTIVES:
        for dataset in DATASETS:
            cfg = make(dataset, objective)
            path = CONFIG_DIR / f"{cfg['run_name']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(f"wrote {path}")
        cfg = make("", objective, debug=True)
        path = CONFIG_DIR / f"{cfg['run_name']}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
