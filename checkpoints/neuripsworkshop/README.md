# NeurIPS workshop checkpoints — rayleigh_benard, 3 seeds

The exact 6 checkpoints behind this repo's headline results (see repo root
README and [docs/OVERVIEW.md](../../docs/OVERVIEW.md)). Every number in
`sweep_results/rayleigh_benard_workshop_test_eval.json` (`n_encoder_seeds: 3`
throughout) comes from probing these files.

| file | objective | seed | source checkpoint |
|---|---|---|---|
| `rayleigh_benard_jepa_seed1.pt` | JEPA | 1 | `runs/rayleigh_benard_jepa_seed1/encoder_100pct.pt` |
| `rayleigh_benard_jepa_seed2.pt` | JEPA | 2 | `runs/rayleigh_benard_jepa_seed2/encoder_100pct.pt` |
| `rayleigh_benard_jepa_seed3.pt` | JEPA | 3 | `runs/rayleigh_benard_jepa_seed3/encoder_100pct.pt` |
| `rayleigh_benard_mae_seed1.pt` | MAE | 1 | `runs/rayleigh_benard_mae_seed1/encoder_100pct.pt` |
| `rayleigh_benard_mae_seed2.pt` | MAE | 2 | `runs/rayleigh_benard_mae_seed2/encoder_100pct.pt` |
| `rayleigh_benard_mae_seed3.pt` | MAE | 3 | `runs/rayleigh_benard_mae_seed3/encoder_100pct.pt` |

**Architecture**: ViT-S encoder (384d × 12 blocks, 6 heads), 2×16×16 tubelet
patches, tube masking @ 0.9, T=8 clips at native resolution (4ch, 512×128),
per-channel z-score norm, AdamW + cosine, 100k steps. JEPA: EMA target
encoder (0.996→1.0) + 6-layer predictor, **192-d** (narrower than the 384-d
encoder — this is the current architecture, post predictor-narrowing). MAE:
4-layer decoder (192d), MSE on masked patches (`norm_pix`).

**Training**: seeds 1/2/3 (seed 0 reserved as the LR-tuning pilot only, never
used as one of these three — see
[configs/tuned_lr.json](../../configs/tuned_lr.json), selected via
`scripts/run_lr_minisweep.sh` + `scripts/pick_lr.py` on held-out validation
loss). LR: JEPA 5e-05, MAE 1e-04. Checkpoint selection is `encoder_100pct.pt`
(final-step, not best-val) — see `runs/rayleigh_benard_*_seed*/history.jsonl`
in the full local archive if you want the val-loss curve or an earlier
milestone instead.

**Reproduce from scratch**: `configs/tuned_lr.json` already has the LR;
`bash scripts/run_final_training.sh` runs all (dataset, objective, seed)
combinations, including these three seeds for `rayleigh_benard`.

Each `.pt` bundles weights + exact config — self-contained, load via
`scripts.load_encoder.load_encoder(...)`, no separate config file needed.
Only encoder weights are included (not optimizer/EMA state) — these are
inference/probing-ready but not resumable for further training; the full
resumable `latest.pt` per seed lives outside git, in a local archive kept off
GitHub given its size.
