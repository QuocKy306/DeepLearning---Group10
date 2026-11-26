"""Dataset và tăng cường dữ liệu (mục 5).

Ảnh trả về là tensor float trong [0, 1]; chuẩn hóa ImageNet nằm TRONG mô hình (src/models.py),
để RandConv (E7) và demo làm việc trên ảnh gốc.

Ảnh và mặt nạ luôn đi qua CÙNG một phép biến đổi hình học (build_transforms).
Không tự viết phép biến đổi riêng cho mặt nạ.
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .color import fda_swap, lab_stats, reinhard_transfer
from .utils import P, read_image, read_mask


def build_transforms(image_size: int, train: bool, appearance: str = "none",
                     app_params: dict | None = None) -> A.Compose:
    ops = [A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.75, 1.25), rotate=(-30, 30), p=0.5),
        ]
        if appearance == "jitter":
            q = app_params or {}
            ops.append(A.ColorJitter(brightness=q.get("brightness", 0.2), contrast=q.get("contrast", 0.2),
                                     saturation=q.get("saturation", 0.2), hue=q.get("hue", 0.05),
                                     p=q.get("p", 0.8)))
    return A.Compose(ops)


class PolypDataset(Dataset):
    """rows: DataFrame có image_path, mask_path (có thể rỗng), id.

    appearance: none | jitter | lab_exchange | fda | randconv
      lab_exchange và fda lấy ảnh tham chiếu ngẫu nhiên từ ref_rows (chính tập huấn luyện nguồn).
      randconv được áp theo lô trên GPU trong src/train.py (RandConv).
      jitter nằm trong build_transforms.
    color_norm_stats: nếu có, mọi ảnh được chuyển màu Lab về thống kê này (thí nghiệm phản thực tế 6.4).
    """

    def __init__(self, rows, image_size: int, train: bool, appearance: str = "none",
                 app_params: dict | None = None, ref_rows=None, color_norm_stats=None):
        self.rows = rows.reset_index(drop=True)
        self.train = train
        self.appearance = appearance if train else "none"
        self.p = app_params or {}
        self.tf = build_transforms(image_size, train, self.appearance, self.p)
        self.color_norm_stats = color_norm_stats
        self.ref_paths = None
        self.ref_stats = None
        if self.appearance in ("lab_exchange", "fda"):
            ref = (ref_rows if ref_rows is not None else self.rows).reset_index(drop=True)
            self.ref_paths = [P(p) for p in ref.image_path]
            if self.appearance == "lab_exchange":
                self.ref_stats = np.stack([lab_stats(read_image(p)) for p in self.ref_paths])

    def __len__(self):
        return len(self.rows)

    def _appearance(self, img: np.ndarray) -> np.ndarray:
        rng = np.random
        if self.appearance == "lab_exchange" and rng.rand() < self.p.get("p", 0.5):
            t = self.ref_stats[rng.randint(len(self.ref_stats))].copy()
            lo, hi = self.p.get("mean_scale", [0.9, 1.1])
            t[:3] *= rng.uniform(lo, hi, 3)
            lo, hi = self.p.get("std_scale", [0.8, 1.2])
            t[3:] *= rng.uniform(lo, hi, 3)
            img = reinhard_transfer(img, t)
        elif self.appearance == "fda" and rng.rand() < self.p.get("p", 0.5):
            ref = read_image(self.ref_paths[rng.randint(len(self.ref_paths))])
            lo, hi = self.p.get("beta", [0.01, 0.05])
            img = fda_swap(img, ref, rng.uniform(lo, hi))
        return img

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = read_image(P(r.image_path))
        h, w = img.shape[:2]
        has_mask = isinstance(r.mask_path, str) and len(r.mask_path) > 0
        mask = read_mask(P(r.mask_path)) if has_mask else np.zeros((h, w), np.uint8)
        if self.color_norm_stats is not None:
            img = reinhard_transfer(img, self.color_norm_stats)
        if self.train:
            img = self._appearance(img)
        out = self.tf(image=img, mask=mask)
        x = torch.from_numpy(out["image"].astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(out["mask"].astype(np.float32))[None]
        return {"image": x, "mask": y, "idx": i}


class RandConv(torch.nn.Module):
    """RandConv (Xu và cộng sự, 2021) áp theo lô trên ảnh [0,1].

    Mỗi lô: chọn kích thước nhân k, trọng số ~ N(0, 1/(3k^2)), trộn với ảnh gốc theo alpha~U(0,1),
    rồi chuẩn hóa lại trung bình/độ lệch chuẩn từng kênh về của ảnh gốc để giữ thang giá trị.
    """

    def __init__(self, kernel_sizes=(1, 3, 5, 7), p: float = 0.5):
        super().__init__()
        self.ks = list(kernel_sizes)
        self.p = p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        k = self.ks[torch.randint(len(self.ks), (1,)).item()]
        w = torch.randn(3, 3, k, k, device=x.device, dtype=x.dtype) / np.sqrt(3 * k * k)
        y = torch.nn.functional.conv2d(x, w, padding=k // 2)
        mu_x, sd_x = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-6
        mu_y, sd_y = y.mean((2, 3), keepdim=True), y.std((2, 3), keepdim=True) + 1e-6
        y = (y - mu_y) / sd_y * sd_x + mu_x
        a = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype)
        return (a * x + (1 - a) * y).clamp(0, 1)


def worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
