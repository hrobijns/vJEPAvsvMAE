"""Load a JEPA run's full model (encoder + predictor + EMA target encoder) for
predictor probing. Needs the `latest.pt` full-state checkpoint, not the
encoder-only `encoder_*pct.pt` milestones — those don't contain the predictor.

Usage:
    uv run python scripts/load_predictor.py local_runs/active_matter_jepa/latest.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.well import ClipSpec
from src.masking import gather_tokens, tube_mask
from src.models.decoder import MAEDecoder
from src.models.predictor import JEPAPredictor
from src.models.vit import build_encoder


def load_jepa(ckpt_path: str):
    """Returns (encoder, predictor, target_encoder, config, spec), all eval mode.

    Rebuilds the same JEPAModel submodules train.py constructs, then loads the
    full `model` state dict from a `latest.pt` checkpoint — this is the only
    checkpoint that carries predictor + target_encoder weights.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(
            f"{ckpt_path} has no 'model' key — looks like an encoder-only "
            "milestone checkpoint (encoder_*pct.pt), not a full latest.pt"
        )
    spec = ClipSpec(**ckpt["spec"])
    cfg = ckpt["config"]

    encoder = build_encoder(spec, cfg["encoder"])
    obj_cfg = cfg["objective"]
    predictor = JEPAPredictor(
        encoder_dim=encoder.embed_dim,
        grid_t=encoder.grid_t,
        grid_h=encoder.grid_h,
        grid_w=encoder.grid_w,
        dim=obj_cfg.get("predictor_dim", 384),
        depth=obj_cfg.get("predictor_depth", 6),
        num_heads=obj_cfg.get("predictor_heads", 6),
    )
    import copy

    target_encoder = copy.deepcopy(encoder)

    state = ckpt["model"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                              if k.startswith("encoder.")})
    predictor.load_state_dict({k[len("predictor."):]: v for k, v in state.items()
                                if k.startswith("predictor.")})
    target_encoder.load_state_dict({k[len("target_encoder."):]: v for k, v in state.items()
                                     if k.startswith("target_encoder.")})

    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    return encoder, predictor, target_encoder, cfg, spec


def load_mae(ckpt_path: str):
    """Returns (encoder, decoder, config, spec), both eval mode.

    Symmetric to load_jepa(): rebuilds MAEModel's submodules from a full
    `latest.pt` (needs decoder weights, unlike the encoder-only milestones).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(
            f"{ckpt_path} has no 'model' key — looks like an encoder-only "
            "milestone checkpoint (encoder_*pct.pt), not a full latest.pt"
        )
    spec = ClipSpec(**ckpt["spec"])
    cfg = ckpt["config"]

    encoder = build_encoder(spec, cfg["encoder"])
    obj_cfg = cfg["objective"]
    decoder = MAEDecoder(
        encoder_dim=encoder.embed_dim,
        patch_dim=encoder.patch_dim,
        grid_t=encoder.grid_t,
        grid_h=encoder.grid_h,
        grid_w=encoder.grid_w,
        dim=obj_cfg.get("decoder_dim", 192),
        depth=obj_cfg.get("decoder_depth", 4),
        num_heads=obj_cfg.get("decoder_heads", 6),
    )

    state = ckpt["model"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                              if k.startswith("encoder.")})
    decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items()
                              if k.startswith("decoder.")})

    encoder.eval()
    decoder.eval()
    return encoder, decoder, cfg, spec


@torch.no_grad()
def predict_masked(encoder, predictor, target_encoder, clip: torch.Tensor,
                    mask_ratio: float | None = None, generator: torch.Generator | None = None):
    """Runs the exact JEPAModel.forward masking + prediction pipeline (eval mode,
    no grad). Returns a dict with everything useful for probing:

        keep_idx, mask_idx: (B, N_visible)/(B, N_masked) token indices
        context:            (B, N_visible, D) online-encoder features (visible tokens)
        target_all:         (B, N, D) EMA target-encoder features, all tokens, layer-normed
        target:              (B, N_masked, D) target features at masked positions (what pred is trained to match)
        pred:                (B, N_masked, D) predictor's output at masked positions
        loss:                scalar smooth-L1(pred, target), same objective as training
    """
    if mask_ratio is None:
        mask_ratio = 0.9  # project-wide default (see configs/*.yaml)
    b = clip.size(0)
    keep_idx, mask_idx, _ = tube_mask(
        b, encoder.grid_t, encoder.grid_h, encoder.grid_w, mask_ratio, clip.device,
        generator=generator,
    )
    context = encoder(clip, keep_idx)
    pred = predictor(context, keep_idx, mask_idx)

    target_all = target_encoder(clip)
    target_all = F.layer_norm(target_all, (target_all.size(-1),))
    target = gather_tokens(target_all, mask_idx)

    loss = F.smooth_l1_loss(pred, target)
    return {
        "keep_idx": keep_idx, "mask_idx": mask_idx,
        "context": context, "target_all": target_all, "target": target,
        "pred": pred, "loss": loss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to a latest.pt (full state, not encoder-only)")
    ap.add_argument("--mask-ratio", type=float, default=None, help="default: training mask_ratio")
    args = ap.parse_args()

    encoder, predictor, target_encoder, cfg, spec = load_jepa(args.checkpoint)
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"loaded: {args.checkpoint}")
    print(f"  encoder: {n_enc/1e6:.1f}M params, predictor: {n_pred/1e6:.1f}M params")
    print(f"  trained on {cfg['data']['dataset_name']}: objective={cfg['objective_name']}, "
          f"mask_ratio={cfg['objective']['mask_ratio']}, "
          f"ema=[{cfg['objective'].get('ema_start')}, {cfg['objective'].get('ema_end')}]")
    print(f"  input spec: {spec}")

    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg["objective"]["mask_ratio"]
    dummy = torch.randn(2, spec.n_channels, spec.n_frames, spec.height, spec.width)
    out = predict_masked(encoder, predictor, target_encoder, dummy, mask_ratio=mask_ratio)
    print(f"  forward pass OK: pred {tuple(out['pred'].shape)}, target {tuple(out['target'].shape)}, "
          f"smooth-L1 on random input = {out['loss'].item():.4f} (uninformative — real clips needed "
          f"for a meaningful number)")


if __name__ == "__main__":
    main()
