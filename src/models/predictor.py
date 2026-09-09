"""JEPA predictor for masked current patches or all patches of the next clip."""

import torch
import torch.nn as nn

from src.masking import gather_tokens
from src.models.vit import Block, sincos_3d


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        future: bool = False,
    ):
        super().__init__()
        self.proj_in = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.future = future
        self.n_tokens = grid_t * grid_h * grid_w
        pos = sincos_3d(dim, grid_t * (2 if future else 1), grid_h, grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, encoder_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_feats: torch.Tensor,  # (B, Nv, De)
        keep_idx: torch.Tensor,  # (B, Nv)
        mask_idx: torch.Tensor,  # (B, Nm)
    ) -> torch.Tensor:
        """Return features for hidden current patches or every future patch.

        Context tokens and mask tokens are processed jointly (positions from
        pos_embed); only target tokens produce predictions.
        """
        if self.future:
            mask_idx = torch.arange(
                self.n_tokens, 2 * self.n_tokens, device=keep_idx.device
            ).expand(context_feats.size(0), -1)
        ctx = self.proj_in(context_feats) + gather_tokens(
            self.pos_embed.expand(context_feats.size(0), -1, -1), keep_idx
        )
        tgt = self.mask_token + gather_tokens(
            self.pos_embed.expand(context_feats.size(0), -1, -1), mask_idx
        )
        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, ctx.size(1) :])
        return self.proj_out(x)
