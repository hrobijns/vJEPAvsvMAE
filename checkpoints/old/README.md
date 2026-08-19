# Legacy single-seed checkpoints

One JEPA + one MAE encoder per dataset, seed 0, no held-out validation loop
during pretraining. Superseded for `rayleigh_benard` by
[`checkpoints/neuripsworkshop/`](../neuripsworkshop/) — kept here for
quick-start/demo use on `active_matter`/`shear_flow`, which don't yet have a
current-architecture 3-seed run (see repo root README's "Pretrained
encoders" section for why).

| file | architecture | notes |
|---|---|---|
| `active_matter_jepa.pt` | **384-d predictor** (pre-narrowing) | predates the 384→192-d predictor change |
| `active_matter_mae.pt` | current | MAE has no predictor, unaffected by that change |
| `shear_flow_jepa.pt` | **384-d predictor** (pre-narrowing) | same as above |
| `shear_flow_mae.pt` | current | unaffected |
| `rayleigh_benard_jepa.pt` | current (192-d predictor) | single-seed predecessor of the 3-seed set in `neuripsworkshop/` |
| `rayleigh_benard_mae.pt` | current | single-seed predecessor of the 3-seed set in `neuripsworkshop/` |

Shared architecture (except the JEPA predictor-width caveat above): ViT-S
encoder (384d × 12 blocks, 6 heads), 2×16×16 tubelet patches, tube masking @
0.9, T=8 clips at native resolution, per-channel z-score norm, AdamW +
cosine, 100k steps. Trained with the original global LR (JEPA 1e-4, MAE
5e-5) rather than the per-dataset tuned values in
[`configs/tuned_lr.json`](../../configs/tuned_lr.json). Each `.pt` bundles
weights + exact config — self-contained, no separate config file needed.

The encoder architecture itself (ViT-S, 384d×12) is identical across every
checkpoint here regardless of the predictor-width caveat — these all load
and probe fine via `scripts/load_encoder.py`.
