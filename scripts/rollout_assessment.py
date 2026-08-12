"""Genuine autoregressive latent-space rollout assessment:

    real window 1 (pixels) -> frozen encoder -> latent dynamics predictor
        -> small decoder -> decoded window 2 (pixels)
        -> re-encode -> predictor -> decoder -> decoded window 3 -> ...

Run identically for the JEPA and MAE frozen encoders (using the post-hoc
rollout heads trained by scripts/train_rollout_heads.py — see
src/objectives/rollout_heads.py for why fresh heads are needed), so the only
difference between the two pipelines is which encoder sits at the top.

Two chained modes per step:
  - fed_back: the actual autoregressive test — context is the PREVIOUS
    step's decoded output (real only at step 1). This is the headline number.
  - oracle: context is the REAL window at every step, regardless of prior
    predictions — isolates one-step prediction error from compounding error
    (generalizes scripts/rollout_probe.py's single-shot logic to every step
    along the trajectory).

Plus two baselines, computed directly from real data (no predictor/decoder):
  - persistence: the step-1 real window, held fixed, compared against every
    later real window — the "assume nothing evolves" floor.
  - ceiling: real target-encoder features of the true future window,
    ridge-probed against that window's own physics — the upper bound if the
    model had actually seen it (rollout_probe.py's existing ceiling
    computation, generalized to every step).

Composition note: decode_all(decoder, pred_latent) is used to turn a
predicted (or, for ceiling, real) latent into pixels — see
src/models/decoder.py's decode_all, which bypasses MAEDecoder's
partial-visibility mask-token machinery since every position here always has
a full feature vector (predicted or real), never a "missing" one.

Usage:
    uv run python scripts/rollout_assessment.py \
        --dataset active_matter --data-root local_data \
        --n-offsets 3 --n-steps 8 --out sweep_results/active_matter_rollout_assessment.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_encoders import contemporaneous_targets, ridge_r2, skill_score
from scripts.analyze_encoders_local import local_target_maps
from scripts.load_encoder import load_encoder
from scripts.load_rollout_heads import load_rollout_heads
from scripts.rollout_probe import pearson_r, r2_across_clips
from src.models.decoder import decode_all
from src.models.patchify import patchify, unpatchify

N_FRAMES = 8


def sample_seeds(mm: np.ndarray, n_offsets: int, n_steps: int, n_frames: int = N_FRAMES):
    """Returns (seeds, total_len). seeds is a list of (traj, off); each seed
    needs total_len = n_frames * (n_steps + 1) contiguous real frames (1
    starting window + n_steps ground-truth future windows to compare
    against)."""
    n_traj, _, t, _, _ = mm.shape
    total_len = n_frames * (n_steps + 1)
    max_off = t - total_len
    if max_off < 0:
        raise ValueError(
            f"trajectory has {t} frames, too short for n_steps={n_steps} "
            f"(needs {total_len} contiguous frames) — use a smaller --n-steps"
        )
    offsets = np.linspace(0, max_off, n_offsets).astype(int)
    seeds = [(traj, off) for traj in range(n_traj) for off in offsets]
    return seeds, total_len


def load_seed_batch(mm: np.ndarray, seeds: list, total_len: int, n_frames: int = N_FRAMES) -> torch.Tensor:
    """Returns (B, n_steps+1, C, n_frames, H, W): windows[:, 0] is the real
    starting window, windows[:, s] for s>=1 is the real window s steps ahead."""
    clips = [torch.from_numpy(np.array(mm[traj, :, off : off + total_len])).float() for traj, off in seeds]
    clips = torch.stack(clips)  # (B, C, total_len, H, W)
    b, c, _, h, w = clips.shape
    n_windows = total_len // n_frames
    windows = clips.view(b, c, n_windows, n_frames, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
    return windows


@torch.no_grad()
def decode_window(encoder, predictor, decoder, ctx: torch.Tensor, ctx_idx, future_idx, norm_pix: bool):
    """One rollout step: ctx (B,C,T,H,W) -> predicted next window (B,C,T,H,W)
    pixels. de-normalizes decode_all's norm_pix output using ctx's own patch
    stats as a causal proxy (matches scripts/rollout_probe.py's existing
    denorm approach — the true future window's own stats aren't knowable
    without leaking the future)."""
    feats = encoder(ctx)
    pred_latent = predictor(feats, ctx_idx, future_idx)
    decoded_norm = decode_all(decoder, pred_latent)
    patches_ctx = patchify(ctx, encoder.patch_size)
    if norm_pix:
        mean = patches_ctx.mean(dim=-1, keepdim=True)
        var = patches_ctx.var(dim=-1, keepdim=True)
        decoded_patches = decoded_norm * (var + 1e-6).sqrt() + mean
    else:
        decoded_patches = decoded_norm
    return unpatchify(decoded_patches, encoder.grid_t, encoder.grid_h, encoder.grid_w, encoder.patch_size), pred_latent


def _new_pixel_acc() -> dict:
    """Running sums for MSE/rel-L2 — MSE = sum(sq_diff)/n and rel_l2 =
    sum(sq_diff)/sum(sq_real) are both associative across batches, so this
    reproduces the exact global metric without ever holding more than one
    batch's pixels in memory at a time."""
    return {"sq_diff": 0.0, "sq_real": 0.0, "n": 0}


def _accumulate_pixel(acc_dict: dict, pred: torch.Tensor, real: torch.Tensor):
    diff = pred - real
    acc_dict["sq_diff"] += (diff ** 2).sum().item()
    acc_dict["sq_real"] += (real ** 2).sum().item()
    acc_dict["n"] += real.numel()


def _accumulate_targets(store: dict, targets: dict):
    """targets: {name: (b,)} for one batch. Physics quantities are already
    per-clip scalars (tiny), so — unlike raw pixels — these are cheap to keep
    per-batch and concatenate at the end; r2_across_clips/pearson_r/ridge_r2
    all need the full per-clip array, not just running sums."""
    for k, v in targets.items():
        store.setdefault(k, []).append(v.cpu())


def _skill_dict(probe_r2s: dict, baseline_r2s: dict) -> dict:
    """Per-quantity persistence-relative skill score (see skill_score() in
    analyze_encoders.py) — the fair, cross-quantity-comparable companion to
    the raw R^2 dicts this script already reports. Needed here specifically
    because a raw mean-R^2-across-quantities summary (as OVERVIEW.md's
    headline tables currently use) lets one exploding/degenerate quantity
    dominate the average — see LINEAR_PROBE.md's vorticity_signed/divergence
    write-up for why some of these targets are near-degenerate to begin
    with."""
    return {k: skill_score(probe_r2s[k], baseline_r2s[k]) for k in probe_r2s}


@torch.no_grad()
def rollout_assessment(encoder_ckpt: str, heads_ckpt: str, mm: np.ndarray, seeds: list, total_len: int,
                        dataset: str, n_steps: int, batch_size: int, device: torch.device,
                        token_max_seeds: int = 16) -> dict:
    """Single pass over the data per batch: every metric is either a small
    per-clip scalar (accumulated across batches) or a running sum (MSE/rel-L2)
    — full-size decoded/real pixel tensors are never retained past the batch
    that produced them. An earlier version stored every batch's decoded and
    real pixel tensors for all steps/modes before reducing at the end, which
    for a few hundred seeds over 8 steps ballooned into hundreds of GB of
    resident memory; this version's peak memory is independent of n_seeds.

    Also accumulates a per-token variant for the first `token_max_seeds`
    samples only (per-token feature tensors are ~n_tokens x larger than the
    pooled ones per step/mode, so this stays capped independent of n_seeds).
    Every window here (context, decoded, ground-truth) is a full n_frames
    clip sharing the encoder's native (grid_t, grid_h, grid_w) token grid, so
    — unlike rollout_probe.py's disjoint context/future sub-windows —
    predicted and target tokens correspond 1:1 by index with no gathering
    needed: pred_latent's token i predicts windows[:, step]'s token i."""
    encoder, _cfg, _spec = load_encoder(encoder_ckpt)
    encoder = encoder.to(device)
    predictor, decoder, hcfg = load_rollout_heads(heads_ckpt, encoder)
    predictor, decoder = predictor.to(device), decoder.to(device)
    norm_pix = hcfg["heads"].get("norm_pix", True)
    n = encoder.n_tokens
    pt, ph, pw = encoder.patch_size

    acc = {m: {s: {"pred_pooled": [], "targets": {}, "pixel": _new_pixel_acc()} for s in range(1, n_steps + 1)}
           for m in ("fed_back", "oracle")}
    ceiling_feats = {s: [] for s in range(1, n_steps + 1)}
    real_targets = {s: {} for s in range(1, n_steps + 1)}
    persistence_pixel = {s: _new_pixel_acc() for s in range(1, n_steps + 1)}
    persistence_targets_step0: dict = {}
    persistence_feats = []

    tok_acc = {m: {s: {"pred_local": [], "targets": {}} for s in range(1, n_steps + 1)}
               for m in ("fed_back", "oracle")}
    tok_ceiling_feats = {s: [] for s in range(1, n_steps + 1)}
    tok_real_targets = {s: {} for s in range(1, n_steps + 1)}
    tok_persistence_targets_step0: dict = {}
    tok_persistence_feats = []
    n_tok_seen = 0

    for start in range(0, len(seeds), batch_size):
        windows = load_seed_batch(mm, seeds[start : start + batch_size], total_len).to(device)
        b = windows.size(0)
        use_token = n_tok_seen < token_max_seeds
        ctx_idx = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)
        future_idx = torch.arange(n, 2 * n, device=device).unsqueeze(0).expand(b, -1)

        step0_feats = encoder(windows[:, 0])
        persistence_feats.append(step0_feats.mean(dim=1).cpu())
        _accumulate_targets(persistence_targets_step0, contemporaneous_targets(windows[:, 0], dataset=dataset))
        if use_token:
            tok_persistence_feats.append(step0_feats.float().cpu())
            _accumulate_targets(tok_persistence_targets_step0,
                                 local_target_maps(windows[:, 0], pt, ph, pw, dataset=dataset))

        for mode in ("fed_back", "oracle"):
            current = windows[:, 0]
            for step in range(1, n_steps + 1):
                ctx = current if mode == "fed_back" else windows[:, step - 1]
                decoded_window, pred_latent = decode_window(encoder, predictor, decoder, ctx, ctx_idx, future_idx, norm_pix)
                acc[mode][step]["pred_pooled"].append(pred_latent.mean(dim=1).cpu())
                _accumulate_targets(acc[mode][step]["targets"], contemporaneous_targets(decoded_window, dataset=dataset))
                _accumulate_pixel(acc[mode][step]["pixel"], decoded_window, windows[:, step])
                if use_token:
                    tok_acc[mode][step]["pred_local"].append(pred_latent.float().cpu())
                    _accumulate_targets(tok_acc[mode][step]["targets"],
                                         local_target_maps(decoded_window, pt, ph, pw, dataset=dataset))
                if mode == "fed_back":
                    current = decoded_window

        for step in range(1, n_steps + 1):
            tgt_feats = F.layer_norm(encoder(windows[:, step]), (encoder.embed_dim,))
            ceiling_feats[step].append(tgt_feats.mean(dim=1).cpu())
            _accumulate_targets(real_targets[step], contemporaneous_targets(windows[:, step], dataset=dataset))
            _accumulate_pixel(persistence_pixel[step], windows[:, 0], windows[:, step])
            if use_token:
                tok_ceiling_feats[step].append(tgt_feats.float().cpu())
                _accumulate_targets(tok_real_targets[step],
                                     local_target_maps(windows[:, step], pt, ph, pw, dataset=dataset))

        if use_token:
            n_tok_seen += b

    persistence_feats_all = torch.cat(persistence_feats)
    persistence_targets_step0 = {k: torch.cat(v) for k, v in persistence_targets_step0.items()}
    d = persistence_feats_all.size(-1)
    tok_persistence_feats_all = torch.cat(tok_persistence_feats).reshape(-1, d)
    tok_persistence_targets_step0 = {k: torch.cat(v).reshape(-1) for k, v in tok_persistence_targets_step0.items()}

    results = {}
    for step in range(1, n_steps + 1):
        targets = {k: torch.cat(v) for k, v in real_targets[step].items()}
        ceiling_feats_all = torch.cat(ceiling_feats[step])
        pp = persistence_pixel[step]

        step_res = {
            "ceiling_latent_r2": {k: ridge_r2(ceiling_feats_all, v) for k, v in targets.items()},
            "persistence": {
                "latent_r2": {k: ridge_r2(persistence_feats_all, v) for k, v in targets.items()},
                "pixel_mse": pp["sq_diff"] / pp["n"],
                "pixel_rel_l2": pp["sq_diff"] / pp["sq_real"],
                "physics_r2": {k: r2_across_clips(persistence_targets_step0[k], v) for k, v in targets.items()},
                "physics_corr": {k: pearson_r(persistence_targets_step0[k], v) for k, v in targets.items()},
            },
        }
        for mode in ("fed_back", "oracle"):
            pred_pooled_all = torch.cat(acc[mode][step]["pred_pooled"])
            decoded_targets = {k: torch.cat(v) for k, v in acc[mode][step]["targets"].items()}
            px = acc[mode][step]["pixel"]
            latent_r2 = {k: ridge_r2(pred_pooled_all, v) for k, v in targets.items()}
            physics_r2 = {k: r2_across_clips(decoded_targets[k], v) for k, v in targets.items()}
            step_res[mode] = {
                "latent_r2": latent_r2,
                "pixel_mse": px["sq_diff"] / px["n"],
                "pixel_rel_l2": px["sq_diff"] / px["sq_real"],
                "physics_r2": physics_r2,
                "physics_corr": {k: pearson_r(decoded_targets[k], v) for k, v in targets.items()},
                "skill_latent": _skill_dict(latent_r2, step_res["persistence"]["latent_r2"]),
                "skill_physics": _skill_dict(physics_r2, step_res["persistence"]["physics_r2"]),
            }

        # per-token variant: same structure, flattened (n_tok_samples*N,) tensors
        tok_targets = {k: torch.cat(v).reshape(-1) for k, v in tok_real_targets[step].items()}
        tok_ceiling_feats_all = torch.cat(tok_ceiling_feats[step]).reshape(-1, d)
        token_res = {
            "n_token_samples": tok_ceiling_feats_all.size(0),
            "ceiling_latent_r2": {k: ridge_r2(tok_ceiling_feats_all, v) for k, v in tok_targets.items()},
            "persistence": {
                "latent_r2": {k: ridge_r2(tok_persistence_feats_all, v) for k, v in tok_targets.items()},
                "physics_r2": {k: r2_across_clips(tok_persistence_targets_step0[k], v) for k, v in tok_targets.items()},
                "physics_corr": {k: pearson_r(tok_persistence_targets_step0[k], v) for k, v in tok_targets.items()},
            },
        }
        for mode in ("fed_back", "oracle"):
            tok_pred_all = torch.cat(tok_acc[mode][step]["pred_local"]).reshape(-1, d)
            tok_decoded_targets = {k: torch.cat(v).reshape(-1) for k, v in tok_acc[mode][step]["targets"].items()}
            tok_latent_r2 = {k: ridge_r2(tok_pred_all, v) for k, v in tok_targets.items()}
            tok_physics_r2 = {k: r2_across_clips(tok_decoded_targets[k], v) for k, v in tok_targets.items()}
            token_res[mode] = {
                "latent_r2": tok_latent_r2,
                "physics_r2": tok_physics_r2,
                "physics_corr": {k: pearson_r(tok_decoded_targets[k], v) for k, v in tok_targets.items()},
                "skill_latent": _skill_dict(tok_latent_r2, token_res["persistence"]["latent_r2"]),
                "skill_physics": _skill_dict(tok_physics_r2, token_res["persistence"]["physics_r2"]),
            }
        step_res["token"] = token_res

        results[str(step)] = step_res

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-offsets", type=int, default=3, help="starting windows per trajectory")
    ap.add_argument("--n-steps", type=int, default=8, help="rollout steps (8 frames/step)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--token-max-seeds", type=int, default=16,
                     help="seeds used for the per-token variant (token tensors are ~n_tokens x larger)")
    ap.add_argument("--jepa-encoder-ckpt", default=None)
    ap.add_argument("--mae-encoder-ckpt", default=None)
    ap.add_argument("--jepa-heads-ckpt", default=None)
    ap.add_argument("--mae-heads-ckpt", default=None)
    ap.add_argument("--heads-dir", default="runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jepa_encoder_ckpt = args.jepa_encoder_ckpt or f"checkpoints/{args.dataset}_jepa.pt"
    mae_encoder_ckpt = args.mae_encoder_ckpt or f"checkpoints/{args.dataset}_mae.pt"
    jepa_heads_ckpt = args.jepa_heads_ckpt or f"{args.heads_dir}/rollout_heads_{args.dataset}_jepa/latest.pt"
    mae_heads_ckpt = args.mae_heads_ckpt or f"{args.heads_dir}/rollout_heads_{args.dataset}_mae/latest.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm = np.load(Path(args.data_root) / "memmap" / args.dataset / "train.npy", mmap_mode="r")
    seeds, total_len = sample_seeds(mm, args.n_offsets, args.n_steps)
    print(f"{len(seeds)} seeds, {total_len} frames/seed ({args.n_steps} rollout steps)")

    out = {"dataset": args.dataset, "n_steps": args.n_steps, "n_seeds": len(seeds)}
    for objective, encoder_ckpt, heads_ckpt in (
        ("jepa", jepa_encoder_ckpt, jepa_heads_ckpt),
        ("mae", mae_encoder_ckpt, mae_heads_ckpt),
    ):
        print(f"\n=== {objective}: {heads_ckpt} ===")
        res = rollout_assessment(encoder_ckpt, heads_ckpt, mm, seeds, total_len, args.dataset,
                                  args.n_steps, args.batch_size, device, args.token_max_seeds)
        out[objective] = res
        for step in range(1, args.n_steps + 1):
            r = res[str(step)]
            print(f"  step {step}: fed_back_mse={r['fed_back']['pixel_mse']:.4f}  "
                  f"oracle_mse={r['oracle']['pixel_mse']:.4f}  persistence_mse={r['persistence']['pixel_mse']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
