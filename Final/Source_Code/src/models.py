"""Ba kiến trúc (mục 4): U-Net/ResNet-34, FPN/MiT-B2, DeepLabV3+/ResNet-50.

Chuẩn hóa ImageNet nằm trong mô hình: đầu vào là ảnh [0, 1].
E2 dùng đúng kiến trúc E3 với encoder_weights=None nên số tham số bằng nhau.
"""
from __future__ import annotations

import os

import segmentation_models_pytorch as smp
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PolypNet(nn.Module):
    def __init__(self, arch: str, encoder: str, pretrained: bool):
        super().__init__()
        if os.environ.get("POLYP_NO_PRETRAINED") == "1":
            pretrained = False
        weights = "imagenet" if pretrained else None
        self.net = getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights,
                                      in_channels=3, classes=1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def build_model(exp_cfg: dict) -> PolypNet:
    return PolypNet(exp_cfg["arch"], exp_cfg["encoder"], bool(exp_cfg["pretrained"]))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from .utils import exp_cfg

    os.environ.setdefault("POLYP_NO_PRETRAINED", "1")
    for e in ["E1", "E2", "E3", "E8"]:
        m = build_model(exp_cfg(e))
        print(e, exp_cfg(e)["arch"], exp_cfg(e)["encoder"], f"{count_params(m) / 1e6:.2f}M tham số")
