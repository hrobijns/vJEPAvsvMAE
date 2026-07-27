"""MAE decoder: reconstructs masked patch pixels from visible-token features."""

import torch
import torch.nn as nn

from src.masking import gather_tokens
from src.models.vit import Block, sincos_3d


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        patch_dim: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        pos = sincos_3d(dim, grid_t, grid_h, grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        visible_feats: torch.Tensor,  # (B, Nv, De)
        keep_idx: torch.Tensor,  # (B, Nv)
        mask_idx: torch.Tensor,  # (B, Nm)
    ) -> torch.Tensor:
        """Returns predicted patch pixels at masked positions: (B, Nm, patch_dim)."""
        b = visible_feats.size(0)
        n_tokens = self.pos_embed.size(1)
        vis = self.proj(visible_feats)  # may be bf16 under autocast
        x = self.mask_token.to(vis.dtype).expand(b, n_tokens, -1).clone()
        x.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)), vis)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.head(self.norm(x))
        return gather_tokens(x, mask_idx)
