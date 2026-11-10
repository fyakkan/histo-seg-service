"""Model definition and the masked Dice + BCE loss."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn


def build_model(encoder: str, encoder_weights: str | None) -> nn.Module:
    """U-Net with a pretrained encoder and one output channel (target logit per pixel)."""
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)


def masked_dice_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Dice loss + binary cross-entropy, both computed only where valid == 1.

    BCE gives stable per-pixel gradients; Dice counters the class imbalance of a small
    target, where predicting "background everywhere" would otherwise look good.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)

    probs = torch.sigmoid(logits) * valid
    target = target * valid
    dice = (2 * (probs * target).sum() + eps) / (probs.sum() + target.sum() + eps)
    return bce + (1 - dice)
