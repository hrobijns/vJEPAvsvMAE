"""Held-out forecast CONTENT probe: does the frozen encoder's representation of
the present contain information that forecasts genuinely future physics — as
opposed to a probe that reuses the model's own pretrained predictor/decoder
to extract it.

Reusing a pretrained prediction head under a causal temporal mask it never
saw during training (only spatially-random tube masks were used) is a real
confound: a weak score could mean either "no forecast info in the
representation" or "the head doesn't generalize to this mask geometry". This
script sidesteps that entirely: no predictor, no decoder, no masking. Just
encode a context window normally (full visible forward pass, exactly like
every other probe in this project), and freshly
train a probe from those features to physics computed on a SEPARATE, later
window the encoder never saw.

Sweeps over multiple --gaps in one run to build a "forecast quality vs. how
far into the future" curve — real trajectory pixels as context at every
horizon, never the model's own prior prediction fed back in (that's a
different, harder question with its own JEPA/MAE architecture asymmetry:
only MAE has a pixel decoder to bootstrap from). The (traj, offset) sample
set is fixed across the whole sweep (sized for the largest gap requested,
so every horizon is compared on the same samples). The CONTEXT window is
identical for every gap at a given offset, so context targets/features are
computed ONCE per (noise level) and reused across every gap — only the
future window's encoding/targets repeat per gap.

Rigor dimensions, matching the rest of the probing suite (see
docs/LINEAR_PROBE.md): **depth** (every layer, not just final — already
present before this pass), **linear vs. MLP** readout (ridge_r2 and mlp_r2,
both pooled and per-token), and **noise robustness** (Gaussian noise injected
into the CONTEXT input only — same convention as analyze_noise_robustness.py:
targets are always computed on the CLEAN clip, only the encoder's input is
corrupted, so a dropping score reflects the representation losing track of
real physics under noise, not the target itself moving; the future/ceiling
side is never noised, since it's the upper bound of what's achievable).

Four numbers per target, per gap, per layer, per noise level:
  - naive persistence: true context-window target value -> true future-window
    target value (no encoder at all, always on CLEAN context). The "assume
    nothing changes" floor — linear-only, not swept over noise (it doesn't
    use the encoder, so noising the input has nothing to act on).
  - probe_linear / probe_mlp (pooled AND per-token): frozen context-window
    features -> future-window target, at whichever noise level is active.
  - ceiling_linear / ceiling_mlp: frozen FUTURE-window features (always
    clean) -> future-window target. Upper bound if you'd actually seen it.
  - skill_linear / skill_mlp: persistence-relative skill score (see
    skill_score() in analyze_encoders.py) for the linear/MLP probe against
    the naive-persistence baseline — the fair, cross-quantity-comparable
    version of the raw R^2 above.

Per-token forecasting uses the same spatial token index in the context and
future windows as its correspondence — advection can physically move a
feature to a different token by the future window, so this is a disclosed
approximation, not exact ground truth of where the physics went.

Memory note: analyze_encoders.py's build_dataset() OOM'd on shear_flow/
rayleigh_benard by materializing contemporaneous_targets() (several full-
clip-sized intermediate tensors each) on the ENTIRE stacked clips tensor at
once. This script never does that: every pass processes chunk_size samples at
a time (targets computed AND encoded per chunk, raw clips discarded before
the next chunk).

Usage:
    uv run python scripts/forecast_content_probe.py \
        --checkpoints local_runs/active_matter_jepa/encoder_100pct.pt local_runs/active_matter_mae/encoder_100pct.pt \
        --data-root local_data --dataset active_matter --gaps 0 8 16 32 64 \
        --noise-stds 0 0.5 1.0 2.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import (
    compute_layerwise_features_batched, contemporaneous_targets,
    load_checkpoint_encoder, ridge_r2, skill_score,
)
from scripts.analyze_encoders_local import layerwise_token_features, local_target_maps, mlp_r2


def _offsets(n_traj_frames: int, n_frames: int, max_gap: int, n_offsets: int) -> np.ndarray:
    """Offset set sized for the LARGEST gap in the sweep, so every horizon in
    the sweep is evaluated on the exact same (traj, offset) samples."""
    max_off = n_traj_frames - (2 * n_frames + max_gap)
    if max_off < 0:
        raise ValueError(
            f"trajectory has {n_traj_frames} frames, too short for "
            f"context({n_frames}) + max_gap({max_gap}) + future({n_frames})"
        )
    return np.linspace(0, max_off, n_offsets).astype(int)


def _load_window(mm: np.ndarray, traj: int, start: int, n_frames: int) -> torch.Tensor:
    return torch.from_numpy(np.array(mm[traj, :, start : start + n_frames])).float()


def _chunked_encode_and_target(pairs, mm, dataset, n_frames, chunk_size, device,
                                encoder, offset_fn, per_token: bool, patch_size=None):
    """Shared chunked loop: for each chunk of (traj, base_off) pairs, loads the
    window starting at offset_fn(traj, base_off), computes its physics targets
    and layerwise features, accumulates, discards the raw chunk. Used for both
    the context window (offset_fn = identity) and a given gap's future window.
    """
    n_layers = len(encoder.blocks) + 1
    feats = [[] for _ in range(n_layers)]
    targets = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        clips = torch.stack([_load_window(mm, tr, offset_fn(tr, off), n_frames) for tr, off in chunk])
        if per_token:
            pt, ph, pw = patch_size
            targets.append(local_target_maps(clips, pt, ph, pw, dataset=dataset))
            f = layerwise_token_features(encoder, clips.to(device))
        else:
            targets.append(contemporaneous_targets(clips, dataset=dataset))
            f = compute_layerwise_features_batched(encoder, clips.to(device))
        for i in range(n_layers):
            feats[i].append(f[i])
    feats = [torch.cat(fs) for fs in feats]
    targets = {k: torch.cat([d[k] for d in targets]) for k in targets[0]}
    return feats, targets


def _chunked_encode_multi_noise(pairs, mm, dataset, n_frames, chunk_size, device,
                                 encoder, noise_stds, per_token: bool, patch_size=None):
    """Context-window variant of _chunked_encode_and_target: loads each chunk
    from the memmap ONCE, computes targets on the CLEAN clip once (targets
    never move — only the encoder's input is corrupted, matching
    analyze_noise_robustness.py's convention), then encodes that same chunk
    at every noise level in `noise_stds` without re-reading from disk or
    recomputing targets. Returns (targets, feats) where feats[std] is a list
    of n_layers tensors."""
    n_layers = len(encoder.blocks) + 1
    targets_acc = []
    feats_acc = {std: [[] for _ in range(n_layers)] for std in noise_stds}
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        clips = torch.stack([_load_window(mm, tr, off, n_frames) for tr, off in chunk])
        if per_token:
            pt, ph, pw = patch_size
            targets_acc.append(local_target_maps(clips, pt, ph, pw, dataset=dataset))
        else:
            targets_acc.append(contemporaneous_targets(clips, dataset=dataset))
        for std in noise_stds:
            noisy = (clips + std * torch.randn_like(clips)) if std > 0 else clips
            if per_token:
                f = layerwise_token_features(encoder, noisy.to(device))
            else:
                f = compute_layerwise_features_batched(encoder, noisy)
            for i in range(n_layers):
                feats_acc[std][i].append(f[i])
    targets = {k: torch.cat([d[k] for d in targets_acc]) for k in targets_acc[0]}
    feats = {std: [torch.cat(fs) for fs in feats_acc[std]] for std in noise_stds}
    return targets, feats


def _score_layer(ctx_feat_layer, fut_target, naive_persistence: float) -> dict:
    """One (layer, noise level, target) cell: linear + MLP probe R^2, plus
    each one's persistence-relative skill score. Shared by pooled_sweep and
    token_sweep (token callers pass already-flattened (N, D)/(N,) tensors)."""
    lin = ridge_r2(ctx_feat_layer, fut_target)
    mlp = mlp_r2(ctx_feat_layer, fut_target)
    return {
        "probe_linear": lin,
        "probe_mlp": mlp,
        "skill_linear": skill_score(lin, naive_persistence),
        "skill_mlp": skill_score(mlp, naive_persistence),
    }


def pooled_sweep(encoder, mm: np.ndarray, dataset: str, n_offsets: int, gaps: list,
                  n_frames: int, chunk_size: int, device: torch.device, noise_stds: list,
                  max_trajectories: int | None = None):
    n_traj, _, t, _, _ = mm.shape
    offsets = _offsets(t, n_frames, max(gaps), n_offsets)
    trajectories = list(range(n_traj))
    if max_trajectories is not None and max_trajectories < n_traj:
        # evenly spaced, not a prefix, for coverage across regime params —
        # same subsampling convention as analyze_encoders.py's build_dataset().
        # Needed here because unlike build_dataset() (which OOM'd and forced
        # a memory-driven cap), this script's chunked reads never OOM — but on
        # a slow network-mounted memmap (see docs/OVERVIEW.md's MooseFS perf
        # note) many small per-(traj,offset) reads are wall-clock-slow
        # regardless of memory, and this script has no cap by default, unlike
        # every other build_dataset()-based script in the suite.
        idx = np.linspace(0, n_traj - 1, max_trajectories).astype(int)
        trajectories = sorted(set(idx.tolist()))
    pairs = [(traj, off) for traj in trajectories for off in offsets]

    # context is identical across every gap (same window), so compute once —
    # at every noise level — and reuse for each gap below.
    ctx_targets, ctx_feats = _chunked_encode_multi_noise(
        pairs, mm, dataset, n_frames, chunk_size, device, encoder, noise_stds, per_token=False,
    )
    n_layers = len(ctx_feats[noise_stds[0]])

    results = {}
    for gap in gaps:
        fut_feats, fut_targets = _chunked_encode_and_target(
            pairs, mm, dataset, n_frames, chunk_size, device, encoder,
            offset_fn=lambda tr, off, g=gap: off + n_frames + g, per_token=False,
        )
        naive_persistence = {
            tname: ridge_r2(ctx_targets[tname].unsqueeze(1), fut_targets[tname]) for tname in ctx_targets
        }
        ceiling = [
            {
                tname: {
                    "ceiling_linear": ridge_r2(fut_feats[layer_idx], fut_targets[tname]),
                    "ceiling_mlp": mlp_r2(fut_feats[layer_idx], fut_targets[tname]),
                }
                for tname in fut_targets
            }
            for layer_idx in range(n_layers)
        ]
        by_noise = {}
        for std in noise_stds:
            layers = []
            for layer_idx in range(n_layers):
                layer_res = {}
                for tname in fut_targets:
                    layer_res[tname] = _score_layer(
                        ctx_feats[std][layer_idx], fut_targets[tname], naive_persistence[tname]
                    )
                    layer_res[tname].update(ceiling[layer_idx][tname])
                layers.append(layer_res)
            by_noise[str(std)] = layers
        results[gap] = {
            "n_samples": ctx_feats[noise_stds[0]][0].size(0),
            "naive_persistence": naive_persistence,
            "n_layers": n_layers,
            "noise_stds": noise_stds,
            "by_noise": by_noise,
        }
    return results


def token_sweep(encoder, mm: np.ndarray, dataset: str, n_offsets: int, gaps: list,
                 n_frames: int, max_clips: int, device: torch.device, noise_stds: list):
    """Small, unchunked subset (max_clips pairs) — per-token tensors are much
    larger per sample, so this mirrors analyze_encoders_local.py's existing
    token-probe subsetting convention rather than chunking."""
    n_traj, _, t, _, _ = mm.shape
    offsets = _offsets(t, n_frames, max(gaps), n_offsets)
    pairs = [(traj, off) for traj in range(n_traj) for off in offsets][:max_clips]
    patch_size = encoder.patch_size

    ctx_targets, ctx_feats = _chunked_encode_multi_noise(
        pairs, mm, dataset, n_frames, max_clips, device, encoder, noise_stds,
        per_token=True, patch_size=patch_size,
    )
    n_layers = len(ctx_feats[noise_stds[0]])

    results = {}
    for gap in gaps:
        fut_feats, fut_targets = _chunked_encode_and_target(
            pairs, mm, dataset, n_frames, max_clips, device, encoder,
            offset_fn=lambda tr, off, g=gap: off + n_frames + g, per_token=True, patch_size=patch_size,
        )
        # naive persistence at token level: local context value -> local
        # future value, same spatial index (see module docstring's caveat).
        naive_persistence = {
            tname: ridge_r2(
                ctx_targets[tname].reshape(-1).unsqueeze(1), fut_targets[tname].reshape(-1)
            )
            for tname in ctx_targets
        }
        ceiling = []
        for layer_idx in range(n_layers):
            ff = fut_feats[layer_idx].reshape(-1, fut_feats[layer_idx].size(-1))
            ceiling.append({
                tname: {
                    "ceiling_linear": ridge_r2(ff, fut_targets[tname].reshape(-1)),
                    "ceiling_mlp": mlp_r2(ff, fut_targets[tname].reshape(-1)),
                }
                for tname in fut_targets
            })
        by_noise = {}
        for std in noise_stds:
            layers = []
            for layer_idx in range(n_layers):
                cf = ctx_feats[std][layer_idx].reshape(-1, ctx_feats[std][layer_idx].size(-1))
                layer_res = {}
                for tname in fut_targets:
                    y = fut_targets[tname].reshape(-1)
                    layer_res[tname] = _score_layer(cf, y, naive_persistence[tname])
                    layer_res[tname].update(ceiling[layer_idx][tname])
                layers.append(layer_res)
            by_noise[str(std)] = layers
        results[gap] = {
            "n_samples": len(pairs),
            "naive_persistence": naive_persistence,
            "n_layers": n_layers,
            "noise_stds": noise_stds,
            "by_noise": by_noise,
        }
    return results


def analyze_forecast_checkpoint(ckpt_path: str, base: str, dataset: str, n_offsets: int,
                                 gaps: list, n_frames: int, chunk_size: int, token_max_clips: int,
                                 noise_stds: list, max_traj: int | None = None, split: str = "train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _spec = load_checkpoint_encoder(ckpt_path, device)
    mm = np.load(Path(base) / "memmap" / dataset / f"{split}.npy", mmap_mode="r")

    pooled = pooled_sweep(encoder, mm, dataset, n_offsets, gaps, n_frames, chunk_size, device, noise_stds,
                           max_trajectories=max_traj)
    token = token_sweep(encoder, mm, dataset, n_offsets, gaps, n_frames, token_max_clips, device, noise_stds)
    return {
        "gaps": gaps,
        "noise_stds": noise_stds,
        "pooled": pooled,
        "token": {
            "note": "per-token forecast uses same-spatial-index correspondence between "
                    "context and future windows; advection means this is an approximation, "
                    "not ground truth of where the physics actually moved to.",
            "by_gap": token,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-offsets", type=int, default=3)
    ap.add_argument("--gaps", type=int, nargs="+", default=[0],
                     help="frames between context and future window, e.g. --gaps 0 8 16 32 64")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=128, help="pooled-pass chunk size (memory control)")
    ap.add_argument("--token-max-clips", type=int, default=64)
    ap.add_argument("--noise-stds", type=float, nargs="+", default=[0, 0.5, 1.0, 2.0],
                     help="Gaussian noise std injected into the CONTEXT input only "
                          "(targets always computed on the clean clip); matches "
                          "analyze_noise_robustness.py's convention")
    ap.add_argument("--max-traj", type=int, default=None,
                     help="cap trajectories used in the POOLED sweep (evenly subsampled) — this "
                          "script's chunked reads never OOM, but on a slow network-mounted memmap "
                          "many small per-(traj,offset) reads are wall-clock-slow regardless of "
                          "memory, and unlike every other build_dataset()-based script in the "
                          "suite this one has no default cap. Token sweep is separately capped "
                          "via --token-max-clips.")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"],
                     help="which preprocessed memmap split to probe (default train; use valid for a "
                          "held-out generalization check against the pretraining data)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        res = analyze_forecast_checkpoint(
            ckpt_path, args.data_root, args.dataset, args.n_offsets, args.gaps,
            args.n_frames, args.chunk_size, args.token_max_clips, args.noise_stds, args.max_traj,
            split=args.split,
        )
        all_results[ckpt_path] = res

        for gap in args.gaps:
            p = res["pooled"][gap]
            print(f"  -- gap={gap} (n_samples={p['n_samples']}) --")
            print("    naive persistence: " + "  ".join(f"{k}={v:.3f}" for k, v in p["naive_persistence"].items()))
            final_clean = p["by_noise"][str(args.noise_stds[0])][-1]
            print(f"    pooled final_norm, noise={args.noise_stds[0]} (lin/mlp R^2, skill_lin/skill_mlp): " +
                  "  ".join(f"{k}={v['probe_linear']:.3f}/{v['probe_mlp']:.3f} "
                            f"({v['skill_linear']:.2f}/{v['skill_mlp']:.2f})" for k, v in final_clean.items()))
            tfinal_clean = res["token"]["by_gap"][gap]["by_noise"][str(args.noise_stds[0])][-1]
            print(f"    token  final_norm, noise={args.noise_stds[0]} (lin/mlp R^2, skill_lin/skill_mlp): " +
                  "  ".join(f"{k}={v['probe_linear']:.3f}/{v['probe_mlp']:.3f} "
                            f"({v['skill_linear']:.2f}/{v['skill_mlp']:.2f})" for k, v in tfinal_clean.items()))

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
