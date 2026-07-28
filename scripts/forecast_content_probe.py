"""Held-out forecast CONTENT probe: does the frozen encoder's representation of
the present contain information that forecasts genuinely future physics — as
opposed to rollout_probe.py's question of whether the model's own pretrained
predictor/decoder can extract and use it.

rollout_probe.py reuses each model's pretrained prediction head under a new
causal temporal mask that head never saw during training (only spatially-random
tube masks were used) — a real confound: a weak rollout score could mean either
"no forecast info in the representation" or "the head doesn't generalize to
this mask geometry". This script sidesteps that entirely: no predictor, no
decoder, no masking. Just encode a context window normally (full visible
forward pass, exactly like every other probe in this project), and freshly
train a linear ridge probe from those features to physics computed on a
SEPARATE, later window the encoder never saw.

Sweeps over multiple --gaps in one run to build a "forecast quality vs. how
far into the future" curve — real trajectory pixels as context at every
horizon, never the model's own prior prediction fed back in (that's a
different, harder question with its own JEPA/MAE architecture asymmetry:
only MAE has a pixel decoder to bootstrap from). The (traj, offset) sample
set is fixed across the whole sweep (sized for the largest gap requested,
so every horizon is compared on the same samples), and the CONTEXT window is
identical for every gap at a given offset — so context features/targets are
computed once and reused across the sweep, only the future window's
encoding/targets repeat per gap.

Three comparisons per target, per gap:
  - naive persistence: true context-window target value -> true future-window
    target value (no encoder at all). The real "assume nothing changes" floor.
  - forecast probe (pooled AND per-token): frozen context-window features ->
    future-window target. This is the new capability.
  - ceiling: frozen FUTURE-window features -> future-window target (upper
    bound - what's achievable if you'd actually seen it).

Per-token forecasting uses the same spatial token index in the context and
future windows as its correspondence — advection can physically move a
feature to a different token by the future window, so this is a disclosed
approximation, not exact ground truth of where the physics went.

Memory note: analyze_encoders.py's build_dataset() OOM'd on shear_flow/
rayleigh_benard by materializing contemporaneous_targets() (several full-
clip-sized intermediate tensors each) on the ENTIRE stacked clips tensor at
once. This script never does that: both passes process chunk_size samples at
a time (targets computed AND encoded per chunk, raw clips discarded before
the next chunk), so it should need no n_offsets reduction on any dataset.

Usage:
    uv run python scripts/forecast_content_probe.py \
        --checkpoints local_runs/active_matter_jepa/encoder_100pct.pt local_runs/active_matter_mae/encoder_100pct.pt \
        --data-root local_data --dataset active_matter --gaps 0 8 16 32 64
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
    load_checkpoint_encoder, ridge_r2,
)
from scripts.analyze_encoders_local import layerwise_token_features, local_target_maps


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


def pooled_sweep(encoder, mm: np.ndarray, dataset: str, n_offsets: int, gaps: list,
                  n_frames: int, chunk_size: int, device: torch.device):
    n_traj, _, t, _, _ = mm.shape
    offsets = _offsets(t, n_frames, max(gaps), n_offsets)
    pairs = [(traj, off) for traj in range(n_traj) for off in offsets]

    ctx_feats, ctx_targets = _chunked_encode_and_target(
        pairs, mm, dataset, n_frames, chunk_size, device, encoder,
        offset_fn=lambda tr, off: off, per_token=False,
    )

    results = {}
    for gap in gaps:
        fut_feats, fut_targets = _chunked_encode_and_target(
            pairs, mm, dataset, n_frames, chunk_size, device, encoder,
            offset_fn=lambda tr, off, g=gap: off + n_frames + g, per_token=False,
        )
        naive_persistence = {
            tname: ridge_r2(ctx_targets[tname].unsqueeze(1), fut_targets[tname]) for tname in ctx_targets
        }
        layers = []
        for layer_idx in range(len(ctx_feats)):
            layer_res = {}
            for tname in fut_targets:
                layer_res[tname] = {
                    "probe": ridge_r2(ctx_feats[layer_idx], fut_targets[tname]),
                    "ceiling": ridge_r2(fut_feats[layer_idx], fut_targets[tname]),
                }
            layers.append(layer_res)
        results[gap] = {
            "n_samples": ctx_feats[0].size(0),
            "naive_persistence": naive_persistence,
            "n_layers": len(layers),
            "layers": layers,
        }
    return results


def token_sweep(encoder, mm: np.ndarray, dataset: str, n_offsets: int, gaps: list,
                 n_frames: int, max_clips: int, device: torch.device):
    """Small, unchunked subset (max_clips pairs) — per-token tensors are much
    larger per sample, so this mirrors analyze_encoders_local.py's existing
    token-probe subsetting convention rather than chunking."""
    n_traj, _, t, _, _ = mm.shape
    offsets = _offsets(t, n_frames, max(gaps), n_offsets)
    pairs = [(traj, off) for traj in range(n_traj) for off in offsets][:max_clips]
    patch_size = encoder.patch_size

    ctx_feats, ctx_targets = _chunked_encode_and_target(
        pairs, mm, dataset, n_frames, max_clips, device, encoder,
        offset_fn=lambda tr, off: off, per_token=True, patch_size=patch_size,
    )

    results = {}
    for gap in gaps:
        fut_feats, fut_targets = _chunked_encode_and_target(
            pairs, mm, dataset, n_frames, max_clips, device, encoder,
            offset_fn=lambda tr, off, g=gap: off + n_frames + g, per_token=True, patch_size=patch_size,
        )
        layers = []
        for layer_idx in range(len(ctx_feats)):
            cf = ctx_feats[layer_idx].reshape(-1, ctx_feats[layer_idx].size(-1))
            ff = fut_feats[layer_idx].reshape(-1, fut_feats[layer_idx].size(-1))
            layer_res = {}
            for tname in fut_targets:
                y = fut_targets[tname].reshape(-1)
                layer_res[tname] = {"probe": ridge_r2(cf, y), "ceiling": ridge_r2(ff, y)}
            layers.append(layer_res)
        results[gap] = {"n_samples": ctx_feats[0].size(0), "n_layers": len(layers), "layers": layers}
    return results


def analyze_forecast_checkpoint(ckpt_path: str, base: str, dataset: str, n_offsets: int,
                                 gaps: list, n_frames: int, chunk_size: int, token_max_clips: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _spec = load_checkpoint_encoder(ckpt_path, device)
    mm = np.load(Path(base) / "memmap" / dataset / "train.npy", mmap_mode="r")

    pooled = pooled_sweep(encoder, mm, dataset, n_offsets, gaps, n_frames, chunk_size, device)
    token = token_sweep(encoder, mm, dataset, n_offsets, gaps, n_frames, token_max_clips, device)
    return {
        "gaps": gaps,
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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_results = {}
    for ckpt_path in args.checkpoints:
        print(f"\n=== {ckpt_path} ===")
        res = analyze_forecast_checkpoint(
            ckpt_path, args.data_root, args.dataset, args.n_offsets, args.gaps,
            args.n_frames, args.chunk_size, args.token_max_clips,
        )
        all_results[ckpt_path] = res

        for gap in args.gaps:
            p = res["pooled"][gap]
            print(f"  -- gap={gap} (n_samples={p['n_samples']}) --")
            print("    naive persistence: " + "  ".join(f"{k}={v:.3f}" for k, v in p["naive_persistence"].items()))
            final = p["layers"][-1]
            print("    pooled final_norm (probe/ceiling): " +
                  "  ".join(f"{k}={v['probe']:.3f}/{v['ceiling']:.3f}" for k, v in final.items()))
            tfinal = res["token"]["by_gap"][gap]["layers"][-1]
            print("    token  final_norm (probe/ceiling): " +
                  "  ".join(f"{k}={v['probe']:.3f}/{v['ceiling']:.3f}" for k, v in tfinal.items()))

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
