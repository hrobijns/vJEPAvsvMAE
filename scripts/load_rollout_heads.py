"""Load a scripts/train_rollout_heads.py checkpoint (predictor + decoder,
NOT the encoder — pass the already-loaded frozen encoder in). Mirrors
scripts/load_predictor.py's pattern.

Usage:
    uv run python scripts/load_rollout_heads.py runs/active_matter_jepa_rollout_heads/latest.pt \
        --encoder-ckpt checkpoints/active_matter_jepa.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.load_encoder import load_encoder
from src.models.decoder import MAEDecoder
from src.models.predictor import JEPAPredictor
from src.models.vit import VideoViT


def load_rollout_heads(ckpt_path: str, encoder: VideoViT):
    """Rebuilds predictor (grid_t=2*encoder.grid_t) + decoder (grid_t=encoder.grid_t)
    from the checkpoint's own 'heads' config, sized against the given
    (already loaded) frozen encoder. Returns (predictor, decoder, cfg), both
    eval mode."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hcfg = cfg["heads"]

    predictor = JEPAPredictor(
        encoder_dim=encoder.embed_dim,
        grid_t=2 * encoder.grid_t, grid_h=encoder.grid_h, grid_w=encoder.grid_w,
        dim=hcfg.get("predictor_dim", 384),
        depth=hcfg.get("predictor_depth", 6),
        num_heads=hcfg.get("predictor_heads", 6),
    )
    decoder = MAEDecoder(
        encoder_dim=encoder.embed_dim, patch_dim=encoder.patch_dim,
        grid_t=encoder.grid_t, grid_h=encoder.grid_h, grid_w=encoder.grid_w,
        dim=hcfg.get("decoder_dim", 192),
        depth=hcfg.get("decoder_depth", 4),
        num_heads=hcfg.get("decoder_heads", 6),
    )
    predictor.load_state_dict(ckpt["predictor"])
    decoder.load_state_dict(ckpt["decoder"])
    predictor.eval()
    decoder.eval()
    return predictor, decoder, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to a rollout-heads latest.pt")
    ap.add_argument("--encoder-ckpt", required=True, help="the frozen encoder these heads were trained against")
    args = ap.parse_args()

    encoder, _cfg, spec = load_encoder(args.encoder_ckpt)
    predictor, decoder, cfg = load_rollout_heads(args.checkpoint, encoder)
    n_pred = sum(p.numel() for p in predictor.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"loaded: {args.checkpoint}")
    print(f"  predictor: {n_pred/1e6:.1f}M params, decoder: {n_dec/1e6:.1f}M params")
    print(f"  trained at step {cfg.get('step', '?')}, encoder_ckpt={cfg.get('encoder_ckpt')}")
    print(f"  input spec: {spec}")

    n = encoder.n_tokens
    dummy_a = torch.randn(2, spec.n_channels, spec.n_frames, spec.height, spec.width)
    dummy_b = torch.randn(2, spec.n_channels, spec.n_frames, spec.height, spec.width)
    with torch.no_grad():
        feats_a = encoder(dummy_a)
        ctx_idx = torch.arange(n).unsqueeze(0).expand(2, -1)
        future_idx = torch.arange(n, 2 * n).unsqueeze(0).expand(2, -1)
        pred = predictor(feats_a, ctx_idx, future_idx)
        from src.models.decoder import decode_all
        decoded = decode_all(decoder, pred)
    print(f"  forward pass OK: pred {tuple(pred.shape)}, decoded {tuple(decoded.shape)} "
          f"(random input — uninformative, real clips needed for a meaningful check)")


if __name__ == "__main__":
    main()
