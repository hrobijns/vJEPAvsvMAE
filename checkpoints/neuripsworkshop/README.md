# Workshop Rayleigh–Bénard encoders

Six final checkpoints: JEPA and MAE, each with independent seeds 1, 2, and 3.
The files are tracked with Git LFS; fetch their weight payloads with
`git lfs pull` when cloning.

Each payload contains `encoder`, `config`, `spec`, and `step`. The configurations
inside the checkpoints are authoritative provenance for their training:

- ViT-S: 384 dimensions, 12 blocks, 6 attention heads; patches 2×16×16.
- Inputs: four channels, eight frames, native 512×128 spatial grid.
- Training: 100,000 steps, batch 64, 90% tube masking, three independent seeds.
- JEPA: 192-dimensional, six-block predictor; LR 5e-5.
- MAE: 192-dimensional, four-block decoder; LR 1e-4.
- Training temporal support: frames 0–100 inclusive. Both targets are inside
  the context clip; neither model was trained to predict a future clip.

These historical payloads lack the newer data/resume identity. They load for
frozen-encoder analysis and regression checks, but cannot automatically resume
training under the new data contract.

```python
from src.models.checkpoints import load_encoder

encoder, config, metadata = load_encoder(
    'checkpoints/neuripsworkshop/rayleigh_benard_jepa_seed1.pt'
)
# encoder is in eval mode; metadata includes the input spec and content hash.
```

The earlier checkpoint tier was removed from the working tree after native-shape
regression checks. Its Git LFS references remain in Git history; neither Git
history nor LFS object storage was pruned.
