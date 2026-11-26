"""Hàm mất mát BCE + Dice mềm tính theo ảnh (mục 4.3)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.eps) / (p.sum(1) + t.sum(1) + self.eps)
        return bce + (1 - dice).mean()
