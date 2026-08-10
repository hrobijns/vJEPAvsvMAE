"""Patchify/unpatchify: (B, C, T, H, W) clip <-> (B, N, patch_dim) tokens,
time-major token order (matches VideoViT.tokenize's patch_embed conv
ordering). Shared by MAEModel's training-time patchify and any post-hoc code
that needs the same conversion (previously duplicated in src/objectives/mae.py
and scripts/rollout_probe.py)."""

import torch
from einops import rearrange


def patchify(x: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, N, patch_dim)."""
    pt, ph, pw = patch_size
    return rearrange(
        x, "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)", pt=pt, ph=ph, pw=pw
    )


def unpatchify(
    patches: torch.Tensor,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """(B, N, patch_dim) -> (B, C, T, H, W), inverse of patchify."""
    pt, ph, pw = patch_size
    return rearrange(
        patches,
        "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
        t=grid_t, h=grid_h, w=grid_w, pt=pt, ph=ph, pw=pw,
    )
