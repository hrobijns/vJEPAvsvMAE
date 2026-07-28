"""Tube masking shared by both objectives.

A single spatial mask is sampled per clip and repeated across all temporal
token slices ("tubes"), following VideoMAE. Returns index tensors compatible
with torch.gather over the token dimension.
"""

import torch


def tube_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
):
    """Sample per-clip tube masks.

    Returns:
        keep_idx: (B, N_visible) token indices of visible tokens
        mask_idx: (B, N_masked) token indices of masked tokens
        mask: (B, N) bool, True where masked
    Token layout is time-major: index = t * (grid_h * grid_w) + spatial.
    """
    n_spatial = grid_h * grid_w
    n_masked_spatial = int(round(n_spatial * mask_ratio))
    if not 0 < n_masked_spatial < n_spatial:
        raise ValueError(f"mask_ratio {mask_ratio} leaves no visible or no masked tokens")

    noise = torch.rand(batch_size, n_spatial, device=device, generator=generator)
    order = noise.argsort(dim=1)
    spatial_keep = order[:, n_masked_spatial:]  # (B, n_vis_spatial)
    spatial_mask = order[:, :n_masked_spatial]  # (B, n_masked_spatial)

    t_offsets = (torch.arange(grid_t, device=device) * n_spatial).view(1, grid_t, 1)
    keep_idx = (spatial_keep.unsqueeze(1) + t_offsets).flatten(1)  # (B, T*n_vis)
    mask_idx = (spatial_mask.unsqueeze(1) + t_offsets).flatten(1)  # (B, T*n_masked)

    n_tokens = grid_t * n_spatial
    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, mask_idx, True)
    return keep_idx, mask_idx, mask


def causal_temporal_mask(
    batch_size: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    n_context_groups: int,
    device: torch.device,
):
    """Deterministic contiguous split: the first `n_context_groups` temporal
    groups are fully visible, the rest fully masked — for sliding-window
    forecast probing (no training uses this mask; out-of-distribution
    relative to tube_mask's spatially-random pattern).

    Same (keep_idx, mask_idx, mask) convention/shapes as tube_mask, but
    identical across the batch (no randomness).
    """
    if not 0 < n_context_groups < grid_t:
        raise ValueError(f"n_context_groups {n_context_groups} must leave both visible and masked groups (grid_t={grid_t})")

    n_spatial = grid_h * grid_w
    n_tokens = grid_t * n_spatial
    spatial = torch.arange(n_spatial, device=device)

    keep_t = torch.arange(n_context_groups, device=device)
    mask_t = torch.arange(n_context_groups, grid_t, device=device)
    keep_idx_1d = (keep_t.unsqueeze(1) * n_spatial + spatial.unsqueeze(0)).flatten()
    mask_idx_1d = (mask_t.unsqueeze(1) * n_spatial + spatial.unsqueeze(0)).flatten()

    keep_idx = keep_idx_1d.unsqueeze(0).expand(batch_size, -1)
    mask_idx = mask_idx_1d.unsqueeze(0).expand(batch_size, -1)

    mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
    mask[:, mask_idx_1d] = True
    return keep_idx, mask_idx, mask


def gather_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along token dim. tokens: (B, N, D), idx: (B, K) -> (B, K, D)."""
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
