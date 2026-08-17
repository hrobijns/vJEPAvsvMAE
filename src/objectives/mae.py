"""VideoMAE-style objective: masked patch reconstruction in pixel space."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masking import gather_tokens, tube_mask
from src.models.decoder import MAEDecoder
from src.models.patchify import patchify as _patchify, unpatchify as _unpatchify
from src.models.vit import VideoViT


class MAEModel(nn.Module):
    def __init__(self, encoder: VideoViT, cfg: dict):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = cfg.get("mask_ratio", 0.9)
        self.norm_pix = cfg.get("norm_pix", True)
        self.decoder = MAEDecoder(
            encoder_dim=encoder.embed_dim,
            patch_dim=encoder.patch_dim,
            grid_t=encoder.grid_t,
            grid_h=encoder.grid_h,
            grid_w=encoder.grid_w,
            dim=cfg.get("decoder_dim", 192),
            depth=cfg.get("decoder_depth", 4),
            num_heads=cfg.get("decoder_heads", 6),
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, N, patch_dim), same token order as encoder."""
        return _patchify(x, self.encoder.patch_size)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b = clip.size(0)
        keep_idx, mask_idx, _ = tube_mask(
            b,
            self.encoder.grid_t,
            self.encoder.grid_h,
            self.encoder.grid_w,
            self.mask_ratio,
            clip.device,
        )
        feats = self.encoder(clip, keep_idx)
        pred = self.decoder(feats, keep_idx, mask_idx)

        target = gather_tokens(self.patchify(clip), mask_idx)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = F.mse_loss(pred, target)

        with torch.no_grad():
            # Collapse diagnostic: std of encoder output across batch+tokens
            # per dim, same convention as JEPAModel's context_feat_std so the
            # two objectives are directly comparable on this axis.
            ctx_std = feats.reshape(-1, feats.size(-1)).std(dim=0).mean()
        return loss, {"loss": loss.item(), "context_feat_std": ctx_std.item()}

    def post_step(self, step: int, total_steps: int):
        pass  # no EMA; hook kept for API parity with JEPA

    @torch.no_grad()
    def reconstruction_figure(self, clip: torch.Tensor):
        """Full-field reconstruction of channel 0, frame 0 for W&B logging.

        Returns (original, reconstructed) numpy arrays (H, W) for one sample.
        Masked patches are filled with predictions (de-normalized per patch if
        norm_pix), visible patches with ground truth.
        """
        b = clip.size(0)
        keep_idx, mask_idx, _ = tube_mask(
            b,
            self.encoder.grid_t,
            self.encoder.grid_h,
            self.encoder.grid_w,
            self.mask_ratio,
            clip.device,
        )
        feats = self.encoder(clip, keep_idx)
        pred = self.decoder(feats, keep_idx, mask_idx)
        patches = self.patchify(clip)
        if self.norm_pix:
            tgt = gather_tokens(patches, mask_idx)
            mean = tgt.mean(dim=-1, keepdim=True)
            var = tgt.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6).sqrt() + mean
        recon = patches.clone()
        recon.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, pred.size(-1)), pred)

        unpatch = lambda p: _unpatchify(
            p, self.encoder.grid_t, self.encoder.grid_h, self.encoder.grid_w,
            self.encoder.patch_size,
        )
        recon_clip = unpatch(recon)
        return (
            clip[0, 0, 0].float().cpu().numpy(),
            recon_clip[0, 0, 0].float().cpu().numpy(),
        )
