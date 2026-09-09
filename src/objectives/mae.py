"""Pixel prediction for masked current patches or every patch of the next clip."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masking import gather_tokens, tube_mask
from src.models.decoder import MAEDecoder
from src.models.patchify import patchify as _patchify, unpatchify as _unpatchify
from src.models.vit import VideoViT


class MAEModel(nn.Module):
    def __init__(self, encoder: VideoViT, cfg: dict, future: bool = False):
        super().__init__()
        self.encoder = encoder
        self.future = future
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
            future=future,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, N, patch_dim), same token order as encoder."""
        return _patchify(x, self.encoder.patch_size)

    def _target_patches(self, clip, mask_idx):
        target = self.patchify(clip)
        if not self.future:
            target = gather_tokens(target, mask_idx)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        return target

    def forward(
        self, clip: torch.Tensor, target_clip: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        if (target_clip is not None) != self.future:
            raise ValueError("target_clip is required only for future prediction")
        if self.future and target_clip.shape != clip.shape:
            raise ValueError("context and future clips must have the same shape")
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

        target = self._target_patches(target_clip if self.future else clip, mask_idx)
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
    def reconstruction_figure(
        self, clip: torch.Tensor, target_clip: torch.Tensor | None = None
    ):
        """Full-field reconstruction of channel 0, frame 0 for W&B logging.

        Returns (original, reconstructed) numpy arrays (H, W) for one sample.
        Masked patches are filled with predictions (de-normalized per patch if
        norm_pix), visible patches with ground truth.
        Future predictions and targets stay in normalized patch units; no
        target-derived means/variances are used to rescale predictions.
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
        if self.future:
            target = self._target_patches(target_clip, mask_idx)
            clips = [
                _unpatchify(
                    patches, self.encoder.grid_t, self.encoder.grid_h,
                    self.encoder.grid_w, self.encoder.patch_size,
                )
                for patches in (target, pred)
            ]
            return tuple(x[0, 0, 0].float().cpu().numpy() for x in clips)
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
