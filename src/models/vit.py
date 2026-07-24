"""Shared ViT encoder used identically by both objectives.

3D tubelet patch embedding over (C, T, H, W) clips, fixed 3D sin-cos
positional embeddings, pre-norm transformer blocks. The encoder can run on a
visible subset of tokens (context/MAE-style) or on all tokens (JEPA target
encoder).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masking import gather_tokens


def sincos_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    """(len(positions), dim) sin-cos embedding; dim must be even."""
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim // 2)
    omega = 1.0 / (10000.0**omega)
    out = positions.double().flatten()[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def sincos_3d(embed_dim: int, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """(grid_t*grid_h*grid_w, embed_dim), time-major token order.

    Temporal axis gets ~1/4 of the dim, spatial axes split the rest, matching
    common video-ViT practice.
    """
    dim_t = embed_dim // 4
    dim_h = (embed_dim - dim_t) // 2
    dim_w = embed_dim - dim_t - dim_h
    assert dim_t % 2 == 0 and dim_h % 2 == 0 and dim_w % 2 == 0, embed_dim

    t = torch.arange(grid_t)
    h = torch.arange(grid_h)
    w = torch.arange(grid_w)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
    emb = torch.cat(
        [sincos_1d(dim_t, tt), sincos_1d(dim_h, hh), sincos_1d(dim_w, ww)], dim=1
    )
    return emb  # (N, D)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, N, dh)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VideoViT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_frames: int,
        height: int,
        width: int,
        patch_t: int = 2,
        patch_h: int = 16,
        patch_w: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if n_frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError(
                f"clip ({n_frames},{height},{width}) not divisible by "
                f"patch ({patch_t},{patch_h},{patch_w})"
            )
        self.grid_t = n_frames // patch_t
        self.grid_h = height // patch_h
        self.grid_w = width // patch_w
        self.n_tokens = self.grid_t * self.grid_h * self.grid_w
        self.embed_dim = embed_dim
        self.patch_dim = n_channels * patch_t * patch_h * patch_w
        self.patch_size = (patch_t, patch_h, patch_w)

        self.patch_embed = nn.Conv3d(
            n_channels,
            embed_dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        pos = sincos_3d(embed_dim, self.grid_t, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, N, D) with positional embeddings added."""
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return tokens + self.pos_embed

    def forward(self, x: torch.Tensor, keep_idx: torch.Tensor | None = None):
        """Encode a clip; if keep_idx given, only those tokens are processed."""
        tokens = self.tokenize(x)
        if keep_idx is not None:
            tokens = gather_tokens(tokens, keep_idx)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


def build_encoder(spec, cfg: dict) -> VideoViT:
    return VideoViT(
        n_channels=spec.n_channels,
        n_frames=spec.n_frames,
        height=spec.height,
        width=spec.width,
        patch_t=cfg.get("patch_t", 2),
        patch_h=cfg.get("patch_h", 16),
        patch_w=cfg.get("patch_w", 16),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    )
