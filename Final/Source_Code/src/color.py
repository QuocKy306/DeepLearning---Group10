"""Các phép toán màu dùng chung cho tăng cường (E5, E6), EDA và phân tích cơ chế màu (mục 5, 6.4, 8.5)."""
from __future__ import annotations

import cv2
import numpy as np


def to_lab(img_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> Lab float32 (L 0..100, a/b xấp xỉ -127..127)."""
    return cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def lab_stats(img_u8: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Vector 6 chiều [mu_L, mu_a, mu_b, sd_L, sd_a, sd_b]. Bỏ viền đen (L<5) thường gặp ở ảnh nội soi."""
    lab = to_lab(img_u8).reshape(-1, 3)
    keep = lab[:, 0] > 5
    if mask is not None:
        keep &= mask.reshape(-1).astype(bool)
    if keep.sum() < 50:
        keep = np.ones(len(lab), bool)
    v = lab[keep]
    return np.concatenate([v.mean(0), v.std(0) + 1e-6]).astype(np.float32)


def reinhard_transfer(img_u8: np.ndarray, target_stats: np.ndarray) -> np.ndarray:
    """Chuyển trung bình/độ lệch chuẩn từng kênh Lab của ảnh sang target_stats (Reinhard 2001)."""
    lab = to_lab(img_u8)
    src = lab_stats(img_u8)
    mu_s, sd_s = src[:3], src[3:]
    mu_t, sd_t = target_stats[:3], target_stats[3:]
    out = (lab - mu_s) / sd_s * sd_t + mu_t
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    out[..., 1:] = np.clip(out[..., 1:], -127, 127)
    # Giữ viền đen là đen để không tạo cạnh giả.
    black = lab[..., 0] <= 5
    out[black] = lab[black]
    return from_lab(out)


def fda_swap(img_u8: np.ndarray, ref_u8: np.ndarray, beta: float) -> np.ndarray:
    """Thay biên độ tần thấp của img bằng của ref, giữ pha (Yang & Soatto 2020)."""
    h, w = img_u8.shape[:2]
    ref = cv2.resize(ref_u8, (w, h), interpolation=cv2.INTER_AREA)
    src = img_u8.astype(np.float32)
    trg = ref.astype(np.float32)
    fs = np.fft.fftshift(np.fft.fft2(src, axes=(0, 1)), axes=(0, 1))
    ft = np.fft.fftshift(np.fft.fft2(trg, axes=(0, 1)), axes=(0, 1))
    amp_s, pha_s = np.abs(fs), np.angle(fs)
    amp_t = np.abs(ft)
    b = max(1, int(np.floor(min(h, w) * beta)))
    ch, cw = h // 2, w // 2
    amp_s[ch - b:ch + b + 1, cw - b:cw + b + 1] = amp_t[ch - b:ch + b + 1, cw - b:cw + b + 1]
    f_new = np.fft.ifftshift(amp_s * np.exp(1j * pha_s), axes=(0, 1))
    out = np.real(np.fft.ifft2(f_new, axes=(0, 1)))
    return np.clip(out, 0, 255).astype(np.uint8)


def quality_features(img_u8: np.ndarray) -> dict:
    """Biến kiểm soát cho hồi quy H3a: độ tương phản (sd của L) và độ nét (phương sai Laplacian)."""
    lab = to_lab(img_u8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return {"contrast": float(lab[..., 0].std()),
            "sharpness": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))}


def mahalanobis(X: np.ndarray, ref: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Khoảng cách Mahalanobis của từng hàng X tới phân bố ref, hiệp phương sai co rút về đường chéo."""
    mu = ref.mean(0)
    C = np.cov(ref, rowvar=False)
    C = (1 - shrink) * C + shrink * np.diag(np.diag(C))
    Ci = np.linalg.pinv(C)
    D = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, Ci, D))
