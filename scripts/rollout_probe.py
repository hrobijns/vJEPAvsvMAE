"""Sliding-window forecast probe: how well does each objective's representation
support predicting a genuinely unseen future group of frames within a clip
window? NOT true autoregressive rollout — context is always real trajectory
pixels at every offset, never fed-back predictions.

JEPA side: predictor(context, keep_idx, mask_idx) gives predicted latents at
the masked (future) tokens. Pool them and ridge-probe against physical
quantities computed on the REAL future sub-window, reported alongside two
free baselines from the same forward passes:
  - ceiling:     probe on the REAL target-encoder features of the future
                 sub-window (upper bound — what's achievable if the model
                 actually saw it).
  - persistence: probe on context features (pooled) against the SAME future
                 target (naive "assume nothing changes" floor).
If the predictor doesn't beat persistence, that's the finding.

MAE side: decoder(context, keep_idx, mask_idx) gives predicted patch pixels
at masked positions (still norm_pix-normalized). We can't know a future
patch's own mean/var in advance (that would leak the future), so we
de-normalize using the LAST VISIBLE context group's own patch statistics as
a causal proxy (same spatial index, one group back — mirrors
src/objectives/mae.py's own denorm code, swapping the stats source). A
sanity baseline applies the same proxy denorm to the REAL future patches
(correct shape, proxy scale) to separate pure proxy-approximation error from
genuine decoder error. Reconstructed vs real derived quantities are compared
directly via correlation + R^2 — no probe needed, MAE already outputs
pixels.

Usage:
    uv run python scripts/rollout_probe.py \
        --jepa-ckpt local_runs/active_matter_jepa/latest.pt \
        --mae-ckpt local_runs/active_matter_mae/latest.pt \
        --data-root local_data --dataset active_matter --n-context-groups 3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import build_dataset, contemporaneous_targets, ridge_r2
from scripts.analyze_encoders_local import local_target_maps
from scripts.load_predictor import load_jepa, load_mae
from src.masking import causal_temporal_mask, gather_tokens
from src.models.patchify import patchify as _patchify, unpatchify as _unpatchify


def patchify(encoder, x: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, N, patch_dim), same token order as encoder.tokenize."""
    return _patchify(x, encoder.patch_size)


def unpatchify(encoder, patches: torch.Tensor) -> torch.Tensor:
    """(B, N, patch_dim) -> (B, C, T, H, W), inverse of patchify."""
    return _unpatchify(patches, encoder.grid_t, encoder.grid_h, encoder.grid_w, encoder.patch_size)


def r2_across_clips(pred: torch.Tensor, real: torch.Tensor) -> float:
    """pred, real: (N,) per-clip scalar values. Standard R^2 vs real's own mean."""
    ss_res = ((pred - real) ** 2).sum()
    ss_tot = ((real - real.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def pearson_r(pred: torch.Tensor, real: torch.Tensor) -> float:
    pred_c, real_c = pred - pred.mean(), real - real.mean()
    denom = (pred_c.norm() * real_c.norm()).clamp_min(1e-8)
    return (pred_c @ real_c / denom).item()


@torch.no_grad()
def jepa_rollout(ckpt_path: str, clips: torch.Tensor, dataset: str, n_context_groups: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, target_encoder, cfg, spec = load_jepa(ckpt_path)
    encoder, predictor, target_encoder = encoder.to(device), predictor.to(device), target_encoder.to(device)
    pt = encoder.patch_size[0]
    future_start = n_context_groups * pt

    pred_feats, ceiling_feats, persist_feats, future_targets = [], [], [], []
    batch_size = 16
    for start in range(0, clips.size(0), batch_size):
        batch = clips[start : start + batch_size].to(device)
        b = batch.size(0)
        keep_idx, mask_idx, _ = causal_temporal_mask(
            b, encoder.grid_t, encoder.grid_h, encoder.grid_w, n_context_groups, device
        )
        context = encoder(batch, keep_idx)
        pred = predictor(context, keep_idx, mask_idx)  # (B, Nm, D)
        target_all = target_encoder(batch)
        target_all = F.layer_norm(target_all, (target_all.size(-1),))
        target = gather_tokens(target_all, mask_idx)  # (B, Nm, D)

        pred_feats.append(pred.mean(dim=1).float().cpu())
        ceiling_feats.append(target.mean(dim=1).float().cpu())
        persist_feats.append(context.mean(dim=1).float().cpu())

        future_frames = batch[:, :, future_start:, :, :]
        t = contemporaneous_targets(future_frames, dataset=dataset)
        future_targets.append({k: v.cpu() for k, v in t.items()})

    pred_feats = torch.cat(pred_feats)
    ceiling_feats = torch.cat(ceiling_feats)
    persist_feats = torch.cat(persist_feats)
    targets = {k: torch.cat([ft[k] for ft in future_targets]) for k in future_targets[0]}

    results = {}
    for tname, y in targets.items():
        results[tname] = {
            "predictor": ridge_r2(pred_feats, y),
            "ceiling": ridge_r2(ceiling_feats, y),
            "persistence": ridge_r2(persist_feats, y),
        }
    return results


@torch.no_grad()
def jepa_rollout_token(ckpt_path: str, clips: torch.Tensor, dataset: str, n_context_groups: int,
                        max_clips: int):
    """Per-token variant of jepa_rollout: pred/target are already per-token
    (B, Nm, D) before pooling — this probes them directly (no pooling)
    against local_target_maps() on the real future sub-window, which follows
    the same time-major/h/w token order as tokenize()/mask_idx so no
    reordering is needed. Capped to max_clips (token tensors are ~Nm x
    larger than the pooled version)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, target_encoder, cfg, spec = load_jepa(ckpt_path)
    encoder, predictor, target_encoder = encoder.to(device), predictor.to(device), target_encoder.to(device)
    pt = encoder.patch_size[0]
    future_start = n_context_groups * pt

    sub = clips[:max_clips]
    pred_feats, ceiling_feats, persist_feats, future_targets = [], [], [], []
    batch_size = 8
    for start in range(0, sub.size(0), batch_size):
        batch = sub[start : start + batch_size].to(device)
        b = batch.size(0)
        keep_idx, mask_idx, _ = causal_temporal_mask(
            b, encoder.grid_t, encoder.grid_h, encoder.grid_w, n_context_groups, device
        )
        context = encoder(batch, keep_idx)
        pred = predictor(context, keep_idx, mask_idx)  # (B, Nm, D)
        target_all = target_encoder(batch)
        target_all = F.layer_norm(target_all, (target_all.size(-1),))
        target = gather_tokens(target_all, mask_idx)  # (B, Nm, D)

        pred_feats.append(pred.float().cpu())
        ceiling_feats.append(target.float().cpu())
        # persistence baseline: no natural per-token correspondence between
        # context and a disjoint future sub-window, so broadcast the
        # context's pooled feature to every future token position.
        persist_feats.append(context.mean(dim=1, keepdim=True).expand(-1, pred.size(1), -1).float().cpu())

        future_frames = batch[:, :, future_start:, :, :]
        t = local_target_maps(future_frames, *encoder.patch_size, dataset=dataset)  # each (B, Nm)
        future_targets.append({k: v.cpu() for k, v in t.items()})

    d = pred_feats[0].size(-1)
    pred_feats = torch.cat(pred_feats).reshape(-1, d)
    ceiling_feats = torch.cat(ceiling_feats).reshape(-1, d)
    persist_feats = torch.cat(persist_feats).reshape(-1, d)
    targets = {k: torch.cat([ft[k] for ft in future_targets]).reshape(-1) for k in future_targets[0]}

    results = {}
    for tname, y in targets.items():
        results[tname] = {
            "predictor": ridge_r2(pred_feats, y),
            "ceiling": ridge_r2(ceiling_feats, y),
            "persistence": ridge_r2(persist_feats, y),
        }
    return results


@torch.no_grad()
def mae_rollout(ckpt_path: str, clips: torch.Tensor, dataset: str, n_context_groups: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, cfg, spec = load_mae(ckpt_path)
    encoder, decoder = encoder.to(device), decoder.to(device)
    pt = encoder.patch_size[0]
    n_spatial = encoder.grid_h * encoder.grid_w
    future_start = n_context_groups * pt
    norm_pix = cfg["objective"].get("norm_pix", True)

    # last visible context group's token indices, spatially aligned with mask_idx
    # (causal_temporal_mask is deterministic/identical across the batch)
    last_ctx_idx_1d = (n_context_groups - 1) * n_spatial + torch.arange(n_spatial, device=device)

    decoded_targets, sanity_targets, real_targets = [], [], []
    batch_size = 16
    for start in range(0, clips.size(0), batch_size):
        batch = clips[start : start + batch_size].to(device)
        b = batch.size(0)
        keep_idx, mask_idx, _ = causal_temporal_mask(
            b, encoder.grid_t, encoder.grid_h, encoder.grid_w, n_context_groups, device
        )
        feats = encoder(batch, keep_idx)
        pred_norm = decoder(feats, keep_idx, mask_idx)  # (B, Nm, patch_dim), normalized

        patches = patchify(encoder, batch)
        target_patches = gather_tokens(patches, mask_idx)  # real future, raw pixels

        if norm_pix:
            last_ctx = gather_tokens(patches, last_ctx_idx_1d.unsqueeze(0).expand(b, -1))
            proxy_mean = last_ctx.mean(dim=-1, keepdim=True)
            proxy_var = last_ctx.var(dim=-1, keepdim=True)
            pred_denorm = pred_norm * (proxy_var + 1e-6).sqrt() + proxy_mean

            true_mean = target_patches.mean(dim=-1, keepdim=True)
            true_var = target_patches.var(dim=-1, keepdim=True)
            target_norm = (target_patches - true_mean) / (true_var + 1e-6).sqrt()
            sanity_denorm = target_norm * (proxy_var + 1e-6).sqrt() + proxy_mean
        else:
            pred_denorm = pred_norm
            sanity_denorm = target_patches  # norm_pix off: proxy step is a no-op

        recon = patches.clone()
        recon.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, pred_denorm.size(-1)), pred_denorm)
        recon_clip = unpatchify(encoder, recon)

        sanity_recon = patches.clone()
        sanity_recon.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, sanity_denorm.size(-1)), sanity_denorm)
        sanity_clip = unpatchify(encoder, sanity_recon)

        decoded_targets.append({k: v.cpu() for k, v in
            contemporaneous_targets(recon_clip[:, :, future_start:], dataset=dataset).items()})
        sanity_targets.append({k: v.cpu() for k, v in
            contemporaneous_targets(sanity_clip[:, :, future_start:], dataset=dataset).items()})
        real_targets.append({k: v.cpu() for k, v in
            contemporaneous_targets(batch[:, :, future_start:], dataset=dataset).items()})

    keys = decoded_targets[0].keys()
    decoded = {k: torch.cat([d[k] for d in decoded_targets]) for k in keys}
    sanity = {k: torch.cat([d[k] for d in sanity_targets]) for k in keys}
    real = {k: torch.cat([d[k] for d in real_targets]) for k in keys}

    results = {}
    for tname in keys:
        results[tname] = {
            "decoded_r2": r2_across_clips(decoded[tname], real[tname]),
            "decoded_corr": pearson_r(decoded[tname], real[tname]),
            "proxy_sanity_r2": r2_across_clips(sanity[tname], real[tname]),
            "proxy_sanity_corr": pearson_r(sanity[tname], real[tname]),
        }
    return results


@torch.no_grad()
def mae_rollout_token(ckpt_path: str, clips: torch.Tensor, dataset: str, n_context_groups: int,
                       max_clips: int):
    """Per-token variant of mae_rollout: reconstructed/real future windows are
    already full pixel fields, so this just swaps contemporaneous_targets for
    local_target_maps and flattens (B, N) -> (B*N,) instead of pooling —
    r2_across_clips/pearson_r are generic over any matched pair of 1D
    tensors. Capped to max_clips for the same reason as jepa_rollout_token."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, cfg, spec = load_mae(ckpt_path)
    encoder, decoder = encoder.to(device), decoder.to(device)
    pt = encoder.patch_size[0]
    n_spatial = encoder.grid_h * encoder.grid_w
    future_start = n_context_groups * pt
    norm_pix = cfg["objective"].get("norm_pix", True)

    last_ctx_idx_1d = (n_context_groups - 1) * n_spatial + torch.arange(n_spatial, device=device)

    sub = clips[:max_clips]
    decoded_targets, sanity_targets, real_targets = [], [], []
    batch_size = 8
    for start in range(0, sub.size(0), batch_size):
        batch = sub[start : start + batch_size].to(device)
        b = batch.size(0)
        keep_idx, mask_idx, _ = causal_temporal_mask(
            b, encoder.grid_t, encoder.grid_h, encoder.grid_w, n_context_groups, device
        )
        feats = encoder(batch, keep_idx)
        pred_norm = decoder(feats, keep_idx, mask_idx)

        patches = patchify(encoder, batch)
        target_patches = gather_tokens(patches, mask_idx)

        if norm_pix:
            last_ctx = gather_tokens(patches, last_ctx_idx_1d.unsqueeze(0).expand(b, -1))
            proxy_mean = last_ctx.mean(dim=-1, keepdim=True)
            proxy_var = last_ctx.var(dim=-1, keepdim=True)
            pred_denorm = pred_norm * (proxy_var + 1e-6).sqrt() + proxy_mean

            true_mean = target_patches.mean(dim=-1, keepdim=True)
            true_var = target_patches.var(dim=-1, keepdim=True)
            target_norm = (target_patches - true_mean) / (true_var + 1e-6).sqrt()
            sanity_denorm = target_norm * (proxy_var + 1e-6).sqrt() + proxy_mean
        else:
            pred_denorm = pred_norm
            sanity_denorm = target_patches

        recon = patches.clone()
        recon.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, pred_denorm.size(-1)), pred_denorm)
        recon_clip = unpatchify(encoder, recon)

        sanity_recon = patches.clone()
        sanity_recon.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, sanity_denorm.size(-1)), sanity_denorm)
        sanity_clip = unpatchify(encoder, sanity_recon)

        pt_, ph_, pw_ = encoder.patch_size
        decoded_targets.append({k: v.cpu() for k, v in
            local_target_maps(recon_clip[:, :, future_start:], pt_, ph_, pw_, dataset=dataset).items()})
        sanity_targets.append({k: v.cpu() for k, v in
            local_target_maps(sanity_clip[:, :, future_start:], pt_, ph_, pw_, dataset=dataset).items()})
        real_targets.append({k: v.cpu() for k, v in
            local_target_maps(batch[:, :, future_start:], pt_, ph_, pw_, dataset=dataset).items()})

    keys = decoded_targets[0].keys()
    decoded = {k: torch.cat([d[k] for d in decoded_targets]).reshape(-1) for k in keys}
    sanity = {k: torch.cat([d[k] for d in sanity_targets]).reshape(-1) for k in keys}
    real = {k: torch.cat([d[k] for d in real_targets]).reshape(-1) for k in keys}

    results = {}
    for tname in keys:
        results[tname] = {
            "decoded_r2": r2_across_clips(decoded[tname], real[tname]),
            "decoded_corr": pearson_r(decoded[tname], real[tname]),
            "proxy_sanity_r2": r2_across_clips(sanity[tname], real[tname]),
            "proxy_sanity_corr": pearson_r(sanity[tname], real[tname]),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa-ckpt", required=True, help="latest.pt (full state, has predictor)")
    ap.add_argument("--mae-ckpt", required=True, help="latest.pt (full state, has decoder)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="active_matter")
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--n-context-groups", type=int, default=3,
                     help="of grid_t=4 groups; default 3 => predict trailing 2 of 8 frames")
    ap.add_argument("--token-max-clips", type=int, default=64,
                     help="clips used for the per-token variant (token tensors are ~Nm x larger)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("building probe dataset...")
    clips, _contemp, _future = build_dataset(args.data_root, args.dataset, args.n_offsets, horizon=1)
    print(f"{clips.size(0)} clips total")

    print(f"\n=== JEPA rollout: {args.jepa_ckpt} ===")
    jepa_res = jepa_rollout(args.jepa_ckpt, clips, args.dataset, args.n_context_groups)
    for tname, r in jepa_res.items():
        print(f"  {tname:24s} predictor={r['predictor']:.3f}  ceiling={r['ceiling']:.3f}  persistence={r['persistence']:.3f}")
    jepa_tok_res = jepa_rollout_token(args.jepa_ckpt, clips, args.dataset, args.n_context_groups, args.token_max_clips)
    print("  -- per-token --")
    for tname, r in jepa_tok_res.items():
        print(f"  {tname:24s} predictor={r['predictor']:.3f}  ceiling={r['ceiling']:.3f}  persistence={r['persistence']:.3f}")

    print(f"\n=== MAE rollout: {args.mae_ckpt} ===")
    mae_res = mae_rollout(args.mae_ckpt, clips, args.dataset, args.n_context_groups)
    for tname, r in mae_res.items():
        print(f"  {tname:24s} decoded_r2={r['decoded_r2']:.3f}  decoded_corr={r['decoded_corr']:.3f}  "
              f"proxy_sanity_r2={r['proxy_sanity_r2']:.3f}")
    mae_tok_res = mae_rollout_token(args.mae_ckpt, clips, args.dataset, args.n_context_groups, args.token_max_clips)
    print("  -- per-token --")
    for tname, r in mae_tok_res.items():
        print(f"  {tname:24s} decoded_r2={r['decoded_r2']:.3f}  decoded_corr={r['decoded_corr']:.3f}  "
              f"proxy_sanity_r2={r['proxy_sanity_r2']:.3f}")

    if args.out:
        out = {
            "jepa": jepa_res, "mae": mae_res,
            "jepa_token": jepa_tok_res, "mae_token": mae_tok_res,
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
