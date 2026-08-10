"""Regenerate configs/rollout_heads_*.yaml.
Run: uv run python scripts/gen_rollout_heads_configs.py

Kept separate from scripts/gen_configs.py (which regenerates the
encoder-pretraining configs) so running one never risks touching the other's
files. Unlike encoder pretraining, rollout-head training always reads from
the memmap layout (scripts/preprocess_memmap.py) — there's no WellDataset
fallback, since PairedWindowMemmapDataset (scripts/train_rollout_heads.py)
only supports the memmap layout.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"

DATASETS = ["active_matter", "shear_flow", "rayleigh_benard"]
OBJECTIVES = ["jepa", "mae"]

HEADS = {
    "predictor_dim": 384, "predictor_depth": 6, "predictor_heads": 6,
    "decoder_dim": 192, "decoder_depth": 4, "decoder_heads": 6,
    "norm_pix": True,
}


def make(dataset: str, objective: str, debug: bool = False) -> dict:
    name = f"rollout_heads_{'debug' if debug else dataset}_{objective}"
    ds_name = "turbulent_radiative_layer_2D" if debug else dataset
    encoder_ckpt = (
        f"runs/debug_{objective}/encoder_100pct.pt" if debug
        else f"checkpoints/{dataset}_{objective}.pt"
    )
    return {
        "run_name": name,
        "encoder_ckpt": encoder_ckpt,
        "out_dir": "runs",
        "seed": 0,
        "bf16": True,
        "log_every": 5 if debug else 50,
        "save_every": 25 if debug else 500,
        "data": {
            "base_path": "~/well_data" if debug else "/workspace/data",
            "dataset_name": ds_name,
            "n_frames": 8,
            "num_workers": 0 if debug else 16,
        },
        "heads": dict(HEADS),
        "optim": {
            "batch_size": 4 if debug else 64,
            "total_steps": 50 if debug else 20_000,
            "warmup_steps": 5 if debug else 1000,
            "lr": 1.0e-4,
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
