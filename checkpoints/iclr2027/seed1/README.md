# ICLR study: seed-1 encoder candidates

All retained encoders from the twelve 100,000-step training runs: three
systems and four objectives. Each run supplies the 25%, 50%, 75%, and 100%
milestones plus its minimum-pretraining-validation-loss encoder. These are
candidates, not downstream-selected final models. `index.json` records actual
steps, checksums, input geometry, exposure, and training provenance. Identical
encoder-state and configuration hashes identify equivalent candidates.

Files preserve the original float32 checkpoint payloads byte-for-byte and are
tracked by the repository's existing `checkpoints/**/*.pt` Git LFS rule.
Training continuation states and feature/probe caches are not included.

To avoid fetching every checkpoint on clone, use `GIT_LFS_SKIP_SMUDGE=1 git
clone REPOSITORY`, then fetch only a needed system from the repository root:

```bash
git lfs pull --include="checkpoints/iclr2027/seed1/shear_flow/**/*.pt"
```

Load an encoder through the repository's normal loader:

```python
from src.models.checkpoints import load_encoder
encoder, config, metadata = load_encoder(
    "checkpoints/iclr2027/seed1/shear_flow/jepa/encoder_100pct.pt"
)
```

The encoder takes normalized, channel-first eight-frame clips. Obtain input
normalization and channel ordering from `WellSource` and `normalize` in
`src.data.source`; use the matching dataset and native resolution. Global
features average tokens; local features retain the sampled token positions.
The evaluation commands and exact target definitions are in the main README
and methods documentation. All checkpoint and probe choices use validation
data before test scoring.

The payloads total approximately 5 GiB. Lossless compression saved only 7–8%
in representative checks and is not used. Confirm the repository owner's
remaining LFS storage and download allowance before uploading. Nothing in
this preparation script pushes files or changes GitHub billing settings.
