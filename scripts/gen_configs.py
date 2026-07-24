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
            "predictor_dim": 384,
            "predictor_depth": 6,
            "predictor_heads": 6,
            "ema_start": 0.996,
            "ema_end": 1.0,
        },
    },
    "mae": {
        "lr": 2.0e-4,
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
            "num_workers": 0 if debug else 8,
        },
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
