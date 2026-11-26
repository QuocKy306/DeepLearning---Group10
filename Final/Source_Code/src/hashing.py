"""Băm cảm nhận (dHash, pHash) và gom cụm ảnh gần trùng bằng hợp-tìm (mục 3.2).

Hai ảnh được nối nếu khoảng cách Hamming của CẢ dHash và pHash <= tau.
Cụm = thành phần liên thông.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    """dHash: so sánh độ sáng hai điểm kề nhau theo hàng. Trả về vector bit bool (size*size)."""
    g = cv2.resize(_gray(img), (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g[:, 1:] > g[:, :-1]).ravel()


def phash(img: np.ndarray, size: int = 8, factor: int = 4) -> np.ndarray:
    """pHash: dấu của hệ số DCT tần thấp so với trung vị."""
    n = size * factor
    g = cv2.resize(_gray(img), (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:size, :size].ravel()
    med = np.median(d[1:])
    return d > med


def center_crop(img: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Giữ vùng trung tâm. Ảnh nội soi có viền bát giác, ô đen ở góc và chữ chèn (ngày giờ, thông số)
    giống nhau giữa các ca; băm toàn ảnh bị các vùng này chi phối và nối nhầm các polyp khác nhau
    (kiểm tra bằng mắt trên Kvasir-SEG ở khoảng cách 8-10)."""
    if frac >= 1:
        return img
    h, w = img.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return img[dh:h - dh, dw:w - dw]


def hash_pair(img: np.ndarray, size: int = 8, crop: float = 0.6) -> np.ndarray:
    c = center_crop(img, crop)
    return np.concatenate([dhash(c, size), phash(c, size)])


def hamming_matrix(bits_a: np.ndarray, bits_b: np.ndarray | None = None) -> np.ndarray:
    """Khoảng cách Hamming giữa mọi cặp hàng. bits: (n, k) bool."""
    a = bits_a.astype(np.int16)
    b = a if bits_b is None else bits_b.astype(np.int16)
    # hamming = k - (a·b + (1-a)·(1-b))
    same = a @ b.T + (1 - a) @ (1 - b).T
    return a.shape[1] - same


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def cluster_hashes(hashes: np.ndarray, tau: int, size: int = 8,
                   return_pairs: bool = False):
    """hashes: (n, 2*size*size) = [dHash | pHash]. Trả về nhãn cụm 0..m-1 (và các cặp bị nối)."""
    k = size * size
    dd = hamming_matrix(hashes[:, :k])
    dp = hamming_matrix(hashes[:, k:])
    link = (dd <= tau) & (dp <= tau)
    np.fill_diagonal(link, False)
    n = len(hashes)
    uf = UnionFind(n)
    ii, jj = np.where(np.triu(link, 1))
    for i, j in zip(ii, jj):
        uf.union(int(i), int(j))
    roots = [uf.find(i) for i in range(n)]
    _, labels = np.unique(roots, return_inverse=True)
    if return_pairs:
        dist = np.maximum(dd, dp)[ii, jj]
        return labels, list(zip(ii.tolist(), jj.tolist(), dist.tolist()))
    return labels


def compute_hashes(paths, read_fn, size: int = 8) -> np.ndarray:
    return np.stack([hash_pair(read_fn(p), size) for p in paths])
