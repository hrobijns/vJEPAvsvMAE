"""V-JEPA-style objective: masked feature prediction in latent space.

Targets come from an EMA "target encoder" run on the full (unmasked) clip;
targets are layer-normalized (no affine) and gradients are stopped. The online
encoder sees only visible tokens; a narrow predictor fills in masked positions.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masking import gather_tokens, tube_mask
from src.models.predictor import JEPAPredictor
from src.models.vit import VideoViT


class JEPAModel(nn.Module):
    def __init__(self, encoder: VideoViT, cfg: dict):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = cfg.get("mask_ratio", 0.9)
        self.ema_start = cfg.get("ema_start", 0.996)
        self.ema_end = cfg.get("ema_end", 1.0)
        self.predictor = JEPAPredictor(
            encoder_dim=encoder.embed_dim,
            grid_t=encoder.grid_t,
            grid_h=encoder.grid_h,
            grid_w=encoder.grid_w,
            dim=cfg.get("predictor_dim", 384),
            depth=cfg.get("predictor_depth", 6),
            num_heads=cfg.get("predictor_heads", 6),
        )
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

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
        context = self.encoder(clip, keep_idx)
        pred = self.predictor(context, keep_idx, mask_idx)

        with torch.no_grad():
            target_all = self.target_encoder(clip)  # full clip, all tokens
            target_all = F.layer_norm(target_all, (target_all.size(-1),))
            target = gather_tokens(target_all, mask_idx)

        loss = F.smooth_l1_loss(pred, target)

        with torch.no_grad():
            # Collapse diagnostics: std of features across batch+tokens per dim.
            tgt_std = target_all.reshape(-1, target_all.size(-1)).std(dim=0).mean()
            ctx_std = context.reshape(-1, context.size(-1)).std(dim=0).mean()
            pred_std = pred.reshape(-1, pred.size(-1)).std(dim=0).mean()
        return loss, {
            "loss": loss.item(),
            "target_feat_std": tgt_std.item(),
            "context_feat_std": ctx_std.item(),
            "pred_feat_std": pred_std.item(),
        }

    def _momentum(self, step: int, total_steps: int) -> float:
        # Cosine ramp ema_start -> ema_end over training.
        frac = min(step / max(total_steps, 1), 1.0)
        return self.ema_end - (self.ema_end - self.ema_start) * (
            math.cos(math.pi * frac) + 1
        ) / 2

    @torch.no_grad()
    def post_step(self, step: int, total_steps: int):
        m = self._momentum(step, total_steps)
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.lerp_(p_online, 1.0 - m)
