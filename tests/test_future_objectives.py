"""Check the information available to future predictors and their loss targets."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from src.data.well import ClipSpec
from src.masking import gather_tokens, tube_mask
from src.models.patchify import patchify, unpatchify
from src.train import build_model


class FutureObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.cfg = dict(
            encoder=dict(patch_t=2, patch_h=2, patch_w=2, embed_dim=16, depth=1, num_heads=2),
            objective=dict(
                mask_ratio=0.9, predictor_dim=16, predictor_depth=1, predictor_heads=2,
                decoder_dim=16, decoder_depth=1, decoder_heads=2, norm_pix=True,
            ),
        )
        self.spec = ClipSpec(4, 8, 8, 8)
        self.clip = torch.randn(2, 4, 8, 8, 8)
        self.target = torch.randn_like(self.clip)
        self.mask = tube_mask(2, 4, 4, 4, 0.9, "cpu")

    def test_future_targets_losses_and_information_boundary(self):
        for family in ("jepa", "mae"):
            with self.subTest(family=family), patch(
                f"src.objectives.{family}.tube_mask", return_value=self.mask
            ):
                model = build_model(family + "_future", self.spec, self.cfg)
                head = model.predictor if family == "jepa" else model.decoder
                predictions, encoder_inputs, teacher_inputs = [], [], []
                head.register_forward_hook(lambda m, a, out: predictions.append(out))
                model.encoder.register_forward_pre_hook(lambda m, a: encoder_inputs.append(a))
                if family == "jepa":
                    model.target_encoder.register_forward_pre_hook(
                        lambda m, a: teacher_inputs.append(a[0])
                    )
                loss, _ = model(self.clip, self.target)
                pred = predictions[-1]
                self.assertEqual(pred.shape[:2], (2, 64))
                torch.testing.assert_close(encoder_inputs[0][0], self.clip)
                torch.testing.assert_close(encoder_inputs[0][1], self.mask[0])
                if family == "jepa":
                    self.assertEqual(len(teacher_inputs), 1)
                    torch.testing.assert_close(teacher_inputs[0], self.target)
                    with torch.no_grad():
                        target = F.layer_norm(model.target_encoder(self.target), (16,))
                    expected = F.smooth_l1_loss(pred, target)
                else:
                    target = patchify(self.target, (2, 2, 2))
                    target = (target - target.mean(-1, keepdim=True)) / (
                        target.var(-1, keepdim=True) + 1e-6
                    ).sqrt()
                    expected = F.mse_loss(pred, target)
                torch.testing.assert_close(loss, expected)

                changed_loss, _ = model(self.clip, torch.randn_like(self.target))
                torch.testing.assert_close(predictions[-1], pred, rtol=0, atol=0)
                self.assertNotEqual(loss.item(), changed_loss.item())

                # Alter every hidden context pixel: neither predictions nor
                # the future-only loss may use these values.
                hidden = patchify(self.clip, (2, 2, 2)).clone()
                hidden.scatter_(1, self.mask[1].unsqueeze(-1).expand(-1, -1, 32), 100.0)
                changed_clip = unpatchify(hidden, 4, 4, 4, (2, 2, 2))
                hidden_loss, _ = model(changed_clip, self.target)
                torch.testing.assert_close(predictions[-1], pred, rtol=0, atol=0)
                torch.testing.assert_close(hidden_loss, loss, rtol=0, atol=0)

                # Corresponding past/future positions retain the same spatial
                # coordinates but have distinct temporal coordinates.
                past, future = head.pos_embed[:, :64], head.pos_embed[:, 64:]
                torch.testing.assert_close(past[..., 4:], future[..., 4:])
                self.assertFalse(torch.equal(past[..., :4], future[..., :4]))
                with self.assertRaisesRegex(ValueError, "target_clip"):
                    model(self.clip)

    def test_future_gradients_and_teacher_update(self):
        for family in ("jepa", "mae"):
            with self.subTest(family=family):
                model = build_model(family + "_future", self.spec, self.cfg)
                target = self.target.clone().requires_grad_()
                loss, _ = model(self.clip, target)
                loss.backward()
                head = model.predictor if family == "jepa" else model.decoder
                for module in (model.encoder, head):
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
                if family == "mae":
                    self.assertFalse(hasattr(model, "target_encoder"))
                    continue
                self.assertIsNone(target.grad)
                before = [p.detach().clone() for p in model.target_encoder.parameters()]
                self.assertTrue(all(p.grad is None for p in model.target_encoder.parameters()))
                torch.optim.SGD(model.parameters(), lr=0.1).step()
                model.post_step(3, 10)
                momentum = model._momentum(3, 10)
                for old, online, teacher in zip(
                    before, model.encoder.parameters(), model.target_encoder.parameters()
                ):
                    torch.testing.assert_close(teacher, old * momentum + online * (1 - momentum))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
                    before, model.target_encoder.parameters()
                )))

    def test_current_objectives_keep_masked_targets_and_weight_layout(self):
        for family in ("jepa", "mae"):
            with self.subTest(family=family), patch(
                f"src.objectives.{family}.tube_mask", return_value=self.mask
            ):
                model = build_model(family, self.spec, self.cfg)
                future = build_model(family + "_future", self.spec, self.cfg)
                # Positions add no learned weights; existing state dictionaries
                # retain their names and shapes. Resume identity separates tasks.
                future.load_state_dict(model.state_dict(), strict=True)
                head = model.predictor if family == "jepa" else model.decoder
                predictions = []
                head.register_forward_hook(lambda m, a, out: predictions.append(out))
                loss, _ = model(self.clip)
                if family == "jepa":
                    with torch.no_grad():
                        target = F.layer_norm(model.target_encoder(self.clip), (16,))
                    expected = F.smooth_l1_loss(predictions[-1], gather_tokens(target, self.mask[1]))
                else:
                    target = gather_tokens(patchify(self.clip, (2, 2, 2)), self.mask[1])
                    target = (target - target.mean(-1, keepdim=True)) / (
                        target.var(-1, keepdim=True) + 1e-6
                    ).sqrt()
                    expected = F.mse_loss(predictions[-1], target)
                self.assertEqual(predictions[-1].size(1), self.mask[1].size(1))
                torch.testing.assert_close(loss, expected)
                with self.assertRaisesRegex(ValueError, "target_clip"):
                    model(self.clip, self.target)

    def test_future_pixel_figures_do_not_rescale_predictions_with_target_statistics(self):
        with patch("src.objectives.mae.tube_mask", return_value=self.mask):
            model = build_model("mae_future", self.spec, self.cfg)
            target, pred = model.reconstruction_figure(self.clip, self.target)
            _, changed = model.reconstruction_figure(self.clip, 10 * self.target + 20)
            np.testing.assert_array_equal(pred, changed)
            expected = model._target_patches(self.target, None)
            expected = unpatchify(expected, 4, 4, 4, (2, 2, 2))[0, 0, 0].numpy()
            np.testing.assert_array_equal(target, expected)
