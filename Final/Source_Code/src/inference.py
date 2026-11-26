"""Suy luận dùng chung cho đánh giá, E9 và demo.

Dự đoán ở image_size được nội suy song tuyến về kích thước gốc trước khi lấy ngưỡng (mục 7.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import PolypDataset
from .metrics import per_image_metrics, size_bucket
from .models import build_model
from .utils import P, base_cfg, read_mask


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, exp_cfg: dict) -> torch.nn.Module:
    model = build_model(exp_cfg)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device()).eval()


@torch.no_grad()
def predict_probs(model, rows: pd.DataFrame, image_size: int, batch_size: int = 8,
                  num_workers: int = 2, color_norm_stats=None):
    """Sinh (chỉ số dòng, bản đồ xác suất ở kích thước gốc)."""
    ds = PolypDataset(rows, image_size, train=False, color_norm_stats=color_norm_stats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dev = device()
    for b in dl:
        x = b["image"].to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            logits = model(x)
        prob = torch.sigmoid(logits.float())
        for k, i in enumerate(b["idx"].tolist()):
            h, w = int(rows.height.iloc[i]), int(rows.width.iloc[i])
            pk = F.interpolate(prob[k:k + 1], size=(h, w), mode="bilinear", align_corners=False)
            yield i, pk[0, 0].cpu().numpy()


def evaluate_rows(model, rows: pd.DataFrame, set_name: str, color_norm_stats=None,
                  save_pred_dir=None):
    """Trả về (bảng độ đo theo ảnh, bảng theo polyp)."""
    cfg = base_cfg()
    thr = cfg["threshold"]
    mc = cfg["metrics"]
    img_recs, pol_recs = [], []
    for i, prob in predict_probs(model, rows, cfg["image_size"], num_workers=cfg["num_workers"],
                                 color_norm_stats=color_norm_stats):
        r = rows.iloc[i]
        pred = prob >= thr
        has_mask = isinstance(r.mask_path, str) and len(r.mask_path) > 0
        gt = read_mask(P(r.mask_path)) if has_mask else None
        m, recs = per_image_metrics(pred, gt, mc["polyp_iou_threshold"], mc["negative_fp_area"])
        img_recs.append({"set": set_name, "id": r.id, "cluster": r.cluster, **m})
        for k, pr in enumerate(recs):
            pol_recs.append({"set": set_name, "id": r.id, "cluster": r.cluster, "polyp": k,
                             "size": size_bucket(pr["area_frac"], mc["size_bins"]), **pr})
        if save_pred_dir is not None:
            import cv2
            cv2.imwrite(str(P(save_pred_dir) / f"{r.id}.png"), (pred * 255).astype(np.uint8))
    return pd.DataFrame(img_recs), pd.DataFrame(pol_recs)
