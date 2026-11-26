"""Độ bền dưới dịch chuyển ngoại quan có kiểm soát (thay tạm khi chưa có tập ngoài miền).

Áp từng phép biến đổi với nhiều mức lên Kvasir-test, đánh giá lại cùng checkpoint:
  lab_a, lab_b   dịch trung bình kênh a hoặc b (đơn vị Lab) — đổi sắc độ, như khác thiết bị
  lab_L          dịch kênh độ sáng L
  contrast       nhân độ lệch chuẩn kênh L quanh trung bình
  blur           làm mờ Gauss (sigma, px ở độ phân giải gốc)
  jpeg           nén JPEG chất lượng thấp
Điểm 0 của mọi trục là ảnh gốc. Viền đen (L<=5) giữ nguyên.

Đầu ra: reports/robustness/shift_curves.csv (mDice theo exp, phép, mức) và per_image.csv
Chạy: python -m src.analysis.shift_robustness --exps P2 P3 P5 --seed 42
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..color import from_lab, to_lab
from ..inference import device, load_checkpoint
from ..metrics import per_image_metrics
from ..utils import P, base_cfg, ensure_dir, exp_cfg, read_image, read_mask, run_dir, set_rows

SHIFTS = {
    "lab_a": [-20, -10, 0, 10, 20],
    "lab_b": [-20, -10, 0, 10, 20],
    "lab_L": [-20, -10, 0, 10, 20],
    "contrast": [0.5, 0.75, 1.0, 1.25, 1.5],
    "blur": [0, 1.5, 3, 5],
    "jpeg": [100, 30, 15, 8],
}


def apply(img: np.ndarray, kind: str, v) -> np.ndarray:
    if kind == "blur":
        return img if v == 0 else cv2.GaussianBlur(img, (0, 0), v)
    if kind == "jpeg":
        if v >= 100:
            return img
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    lab = to_lab(img)
    black = lab[..., 0] <= 5
    if kind == "lab_a":
        lab[..., 1] += v
    elif kind == "lab_b":
        lab[..., 2] += v
    elif kind == "lab_L":
        lab[..., 0] += v
    elif kind == "contrast":
        mu = lab[..., 0][~black].mean()
        lab[..., 0] = mu + (lab[..., 0] - mu) * v
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    lab[..., 1:] = np.clip(lab[..., 1:], -127, 127)
    out = from_lab(lab)
    out[black] = img[black]
    return out


@torch.no_grad()
def predict(model, img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(x.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device())
    p = torch.sigmoid(model(x).float())
    p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["P2", "P3", "P5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", default="kvasir_test")
    ap.add_argument("--kinds", nargs="+", default=list(SHIFTS))
    args = ap.parse_args(argv)
    cfg = base_cfg()
    rows = set_rows(args.set)
    imgs = [read_image(P(p)) for p in rows.image_path]
    gts = [read_mask(P(p)) for p in rows.mask_path]
    out = ensure_dir("reports/robustness")
    recs = []
    for e in args.exps:
        model = load_checkpoint(run_dir(e, args.seed) / "best.pt", exp_cfg(e))
        for kind in args.kinds:
            for v in SHIFTS[kind]:
                for i, (img, gt) in enumerate(zip(imgs, gts)):
                    pr = predict(model, apply(img, kind, v), cfg["image_size"]) >= cfg["threshold"]
                    m, _ = per_image_metrics(pr, gt)
                    recs.append({"exp": e, "kind": kind, "level": v, "id": rows.id[i],
                                 "cluster": rows.cluster[i], "dice": m["dice"], "empty_pred": m["empty_pred"]})
            print(e, kind, "xong", flush=True)
    for e, g in pd.DataFrame(recs).groupby("exp"):
        g.to_csv(out / f"per_image_{e}.csv", index=False)
    df = pd.concat([pd.read_csv(f) for f in sorted(out.glob("per_image_*.csv"))])
    curves = df.groupby(["exp", "kind", "level"]).agg(mdice=("dice", "mean"), empty=("empty_pred", "mean")).reset_index()
    curves.to_csv(out / "shift_curves.csv", index=False)
    print(curves.pivot_table(index=["kind", "level"], columns="exp", values="mdice").round(3).to_string())


if __name__ == "__main__":
    main()
