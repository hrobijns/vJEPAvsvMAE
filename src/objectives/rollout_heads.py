"""Post-hoc latent-dynamics predictor + pixel decoder trained on top of a
FROZEN pretrained encoder (either objective's), for genuine autoregressive
latent-space rollout assessment (see scripts/rollout_assessment.py):

    window i (pixels) -> frozen encoder -> 1024x384 tokens
        -> predictor -> predicted 1024x384 tokens (window i+1)
        -> decoder -> window i+1 (pixels) -> feed back as the next window i

Both heads reuse JEPAPredictor/MAEDecoder as-is (same architecture, same
training recipe) regardless of which objective's encoder is frozen
underneath, so the only difference between the two pipelines is which
encoder sits at the top.

Every position in both windows is always fully known (context = all of
window i, target = all of window i+1) -- there is no partial visibility to
model, so this is deliberately mask-free: "context" and "future" are just two
fixed, disjoint index ranges (not a sampled tube_mask/causal_temporal_mask),
and the decoder is called through decode_all(), bypassing MAEDecoder's
mask-token-fill machinery entirely (see src/models/decoder.py).

The decoder is trained only on the frozen encoder's REAL features (window
i's own features -> window i's own pixels), never on the predictor's
predicted features -- so at rollout-assessment time, any mismatch between
predicted and real latents shows up honestly as compounding error, instead of
being absorbed by a decoder that learned to compensate for the predictor's
specific error patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import MAEDecoder, decode_all
from src.models.patchify import patchify
from src.models.predictor import JEPAPredictor
from src.models.vit import VideoViT


class RolloutHeadModel(nn.Module):
    def __init__(self, encoder: VideoViT, cfg: dict):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.norm_pix = cfg.get("norm_pix", True)
        n = encoder.n_tokens

        self.predictor = JEPAPredictor(
            encoder_dim=encoder.embed_dim,
            grid_t=2 * encoder.grid_t, grid_h=encoder.grid_h, grid_w=encoder.grid_w,
            dim=cfg.get("predictor_dim", 384),
            depth=cfg.get("predictor_depth", 6),
            num_heads=cfg.get("predictor_heads", 6),
        )
        self.decoder = MAEDecoder(
            encoder_dim=encoder.embed_dim, patch_dim=encoder.patch_dim,
            grid_t=encoder.grid_t, grid_h=encoder.grid_h, grid_w=encoder.grid_w,
            dim=cfg.get("decoder_dim", 192),
            depth=cfg.get("decoder_depth", 4),
            num_heads=cfg.get("decoder_heads", 6),
        )
        self.register_buffer("ctx_idx", torch.arange(n), persistent=False)
        self.register_buffer("future_idx", torch.arange(n, 2 * n), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # frozen encoder never leaves eval mode
        return self

    def forward(self, window_a: torch.Tensor, window_b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b = window_a.size(0)
        with torch.no_grad():
            feats_a = self.encoder(window_a)  # (B, N, D), all tokens visible
            feats_b = F.layer_norm(self.encoder(window_b), (self.encoder.embed_dim,))

        ctx_idx = self.ctx_idx.unsqueeze(0).expand(b, -1)
        future_idx = self.future_idx.unsqueeze(0).expand(b, -1)
        pred = self.predictor(feats_a, ctx_idx, future_idx)  # (B, N, D)
        p_loss = F.smooth_l1_loss(pred, feats_b)

        with torch.no_grad():
            patches_a = patchify(window_a, self.encoder.patch_size)  # (B, N, patch_dim)
            if self.norm_pix:
                mean = patches_a.mean(dim=-1, keepdim=True)
                var = patches_a.var(dim=-1, keepdim=True)
                target_patches = (patches_a - mean) / (var + 1e-6).sqrt()
            else:
                target_patches = patches_a
        decoded = decode_all(self.decoder, feats_a)  # (B, N, patch_dim)
        d_loss = F.mse_loss(decoded, target_patches)

        loss = p_loss + d_loss
        return loss, {
            "loss": loss.item(), "predictor_loss": p_loss.item(), "decoder_loss": d_loss.item(),
        }

    def post_step(self, step: int, total_steps: int):
        pass  # no EMA; kept for API parity with JEPAModel/MAEModel
