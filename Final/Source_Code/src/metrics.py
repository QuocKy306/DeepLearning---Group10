"""Độ đo đánh giá (mục 7). Mọi độ đo tính ở độ phân giải gốc của mặt nạ.

per_image_metrics trả về:
  dice, iou, precision, recall
  hd95            NaN nếu dự đoán hoặc nhãn rỗng (báo cáo riêng trên ảnh không rỗng)
  hd95_diag       hd95 nhưng gán đường chéo ảnh khi dự đoán rỗng (chỉ để đối chiếu văn liệu)
  empty_pred      1 nếu dự đoán rỗng
  pred_area_frac, gt_area_frac
  n_polyps, n_detected, n_fp_components   (mức polyp, IoU >= ngưỡng)
  fp_frame        chỉ có ý nghĩa với khung âm tính: dự đoán > negative_fp_area
và danh sách bản ghi từng polyp (diện tích, đã phát hiện hay chưa) để phân tầng theo kích thước.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def _boundary(m: np.ndarray) -> np.ndarray:
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    return m & ~er


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Phân vị 95 của khoảng cách biên đối xứng (điểm ảnh). NaN nếu một bên rỗng."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    bp, bg = _boundary(pred), _boundary(gt)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    d = np.concatenate([dt_g[bp], dt_p[bg]])
    return float(np.percentile(d, 95))


def nsd(pred: np.ndarray, gt: np.ndarray, tol: float) -> float:
    """Normalized Surface Dice (Nikolov và cộng sự 2021): tỉ lệ điểm biên của hai bên nằm trong dung sai tol.

    Ít nhạy với khác biệt quy ước vẽ biên giữa các trung tâm hơn Dice. NaN nếu một bên rỗng.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    bp, bg = _boundary(pred), _boundary(gt)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    ok = (dt_g[bp] <= tol).sum() + (dt_p[bg] <= tol).sum()
    return float(ok / (bp.sum() + bg.sum()))


def overlap(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    if tp + fp + fn == 0:  # cả hai rỗng
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    return {"dice": 2 * tp / (2 * tp + fp + fn),
            "iou": tp / (tp + fp + fn),
            "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
            "recall": tp / (tp + fn) if tp + fn > 0 else 1.0}


def polyp_level(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5):
    """Mỗi thành phần liên thông của nhãn là một polyp. Phát hiện nếu IoU(polyp, hợp các thành phần dự đoán chạm nó) >= ngưỡng."""
    n_g, lab_g = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    n_p, lab_p = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    area = gt.size
    records, touched_pred = [], set()
    for g in range(1, n_g):
        gm = lab_g == g
        ids = np.unique(lab_p[gm])
        ids = ids[ids > 0]
        touched_pred.update(ids.tolist())
        pm = np.isin(lab_p, ids) if len(ids) else np.zeros_like(gm)
        inter = (gm & pm).sum()
        union = (gm | pm).sum()
        iou = inter / union if union else 0.0
        records.append({"area_frac": float(gm.sum() / area), "iou": float(iou),
                        "detected": int(iou >= iou_thr)})
    n_fp = len(set(range(1, n_p)) - touched_pred)
    return records, n_fp


def per_image_metrics(pred: np.ndarray, gt: np.ndarray | None, iou_thr: float = 0.5,
                      negative_fp_area: float = 0.001):
    pred = pred.astype(bool)
    h, w = pred.shape
    diag = float(np.hypot(h, w))
    out = {"pred_area_frac": float(pred.mean()), "empty_pred": int(not pred.any())}
    if gt is None:  # khung âm tính
        n_p = cv2.connectedComponents(pred.astype(np.uint8))[0] - 1
        out.update({"fp_frame": int(pred.mean() > negative_fp_area), "n_fp_components": int(n_p)})
        return out, []
    gt = gt.astype(bool)
    out.update(overlap(pred, gt))
    hd = hd95(pred, gt)
    out["hd95"] = hd
    out["hd95_diag"] = diag if (np.isnan(hd) and gt.any()) else hd
    # dung sai = 1% đường chéo ảnh, để cùng ý nghĩa ở mọi độ phân giải
    nv = nsd(pred, gt, tol=0.01 * diag)
    out["nsd"] = 0.0 if (np.isnan(nv) and gt.any()) else nv
    out["gt_area_frac"] = float(gt.mean())
    recs, n_fp = polyp_level(pred, gt, iou_thr)
    out.update({"n_polyps": len(recs), "n_detected": sum(r["detected"] for r in recs),
                "n_fp_components": n_fp})
    return out, recs


def size_bucket(area_frac: float, bins=(0.01, 0.05)) -> str:
    if area_frac < bins[0]:
        return "nho"
    if area_frac <= bins[1]:
        return "vua"
    return "lon"


def batch_dice(logits, target, threshold: float = 0.5):
    """Dice theo ảnh ở độ phân giải huấn luyện, dùng để theo dõi trên kvasir_val khi huấn luyện."""
    import torch

    p = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1), torch.ones_like(denom))
    return d


if __name__ == "__main__":
    # Tự kiểm tra trên mặt nạ vẽ tay (ngày 2 của kế hoạch).
    g = np.zeros((100, 100), np.uint8)
    cv2.circle(g, (50, 50), 20, 1, -1)
    m, _ = per_image_metrics(g, g)
    assert abs(m["dice"] - 1) < 1e-9 and m["hd95"] == 0 and m["n_detected"] == 1, m
    s = np.roll(g, 2, axis=1)
    m, _ = per_image_metrics(s, g)
    assert 0.9 < m["dice"] < 1 and 1.0 <= m["hd95"] <= 2.0, m
    m, _ = per_image_metrics(np.zeros_like(g), g)
    assert m["dice"] == 0 and np.isnan(m["hd95"]) and m["empty_pred"] == 1 and m["n_detected"] == 0, m
    assert abs(m["hd95_diag"] - np.hypot(100, 100)) < 1e-9
    m, _ = per_image_metrics(np.zeros_like(g), np.zeros_like(g))
    assert m["dice"] == 1.0
    fp = g.copy()
    cv2.circle(fp, (10, 10), 3, 1, -1)
    m, _ = per_image_metrics(fp, g)
    assert m["n_fp_components"] == 1 and m["n_detected"] == 1, m
    m, _ = per_image_metrics(g, None)
    assert m["fp_frame"] == 1
    m, _ = per_image_metrics(np.roll(g, 1, axis=1), g)
    assert m["nsd"] == 1.0 and m["dice"] < 1, m  # lệch 1 px < dung sai 1,41 px: NSD bỏ qua, Dice phạt
    print("metrics: mọi kiểm tra ĐẠT")
