"""Load one of the pretrained encoders and sanity-check it with a forward pass.

Usage:
    uv run python scripts/load_encoder.py checkpoints/old/active_matter_jepa.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.well import ClipSpec
from src.models.vit import build_encoder


def load_encoder(ckpt_path: str):
    """Returns (encoder, config, spec). encoder is in eval() mode."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec = ClipSpec(**ckpt["spec"])
    encoder = build_encoder(spec, ckpt["config"]["encoder"])
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder, ckpt["config"], spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()

    encoder, cfg, spec = load_encoder(args.checkpoint)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"loaded: {args.checkpoint}")
    print(f"  {n_params/1e6:.1f}M params, objective={cfg['objective_name']}")
    print(f"  trained on {cfg['data']['dataset_name']}: "
          f"lr={cfg['optim']['lr']}, steps={cfg['optim']['total_steps']}, "
          f"mask_ratio={cfg['objective']['mask_ratio']}")
    print(f"  input spec: {spec}")

    # forward pass on random input matching the training spec, to confirm the
    # checkpoint loads and runs end-to-end
    dummy = torch.randn(1, spec.n_channels, spec.n_frames, spec.height, spec.width)
    with torch.no_grad():
        feats = encoder(dummy)
    print(f"  forward pass OK: output shape {tuple(feats.shape)} (batch, n_tokens, dim)")


if __name__ == "__main__":
    main()
