now it is time to do the active matter system. 


what i want:
- same plots as in here vJEPAvsvMAE/reports/rayleigh_benard_best_val_ridge_mlp but for the active matter system


## How big is the active matter dataset on the Well?

| Split | Files | Size |
|---|---:|---:|
| train | 45 | 42.5 GB |
| valid | 16 | 6.3 GB |
| test | 21 | 7.0 GB |
| **total** | 82 | **55.8 GB** |

Each trajectory is 81 timesteps of 256x256 on a periodic 10x10 domain, with 11
stored channels. The parameter grid is alpha in {-1,-2,-3,-4,-5} and zeta in
{1,3,5,7,9,11,13,15,17} (45 parameter sets, 5 trajectories each, beta fixed at
0.8). The pod currently holds only Rayleigh-Benard under
`/workspace/well/datasets`, so active matter must be downloaded first.

## What are we probing?

Regime parameters (governing-parameter table): `alpha` and `zeta`. Unlike
Rayleigh-Benard they are **not** log-transformed (`log_parameters=False` in
`src/physics/systems.py`), because alpha is negative. The table columns are
therefore plain `alpha`/`zeta`, not `log10 ...`.

Physical targets (4, versus 5 for Rayleigh-Benard):

| Target | Definition |
|---|---|
| `kinetic_energy` | `0.5 (u^2 + v^2)` |
| `enstrophy` | squared vorticity from spectral periodic derivatives |
| `nematic_order` | `sqrt(2 |Q|^2)` with `Q = D/c - I/2` |
| `nematic_gradient_energy` | summed squared gradients of `Q` |

Same probe set as Rayleigh-Benard: Ridge and MLP on frozen features at all 13
encoder outputs, pooled and token representations, plus the dumb global
(regime+time) MLP, the dumb local (regime+time+position) MLP, and persistence.

## Differences that affect the analysis

1. **Only 81 frames per trajectory.** With 8 context frames and offsets
   `[0,16,24,40]` there are 34 window starts per trajectory versus 153 for
   Rayleigh-Benard. Fewer probe samples per trajectory, partly offset by more
   trajectories.
2. **The eval protocol needed the t+16 horizon.** `configs/eval_active_matter.yaml`
   listed `[0,16,40]`; it is now `[0,16,24,40]`, matching Rayleigh-Benard so the
   depth plots have the same four columns (t+0, t+8, t+16, t+32).
3. **11 input channels, not 4.** Encoders are dataset-specific; the frozen
   active-matter `best_val` checkpoints already exist in the roster.
4. **Fully periodic domain.** All derivatives are spectral in both directions
   and pooling is a plain mean; Rayleigh-Benard uses wall-normal weighting.
5. **Same token count.** 256x256 with patch (2,16,16) gives 4x16x16 = 1024
   tokens per clip, identical to Rayleigh-Benard's 512x128, so probe cost and
   memory per encoder output are comparable.
6. **Targets are not comparable across systems.** Depth/physics figures are
   per-system; only the qualitative depth and objective ordering transfer.
7. **Attentive probes stay off**, as in the Rayleigh-Benard study, for cost.
8. **Storage.** Plan for the 56 GB download plus a Well cache and per-encoder
   feature shards; prune each encoder's features after its test scores seal.