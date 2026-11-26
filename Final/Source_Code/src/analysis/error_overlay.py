"""Ảnh lỗi tệ nhất để gán nhãn nguyên nhân thủ công.

Viền xanh lá = nhãn, viền đỏ = dự đoán. Ghi reports/errors/<exp>_seed<seed>/worst_<set>_<k>.png
và bảng gan_nhan_nguyen_nhan_loi.csv (cột nguyen_nhan để trống, điền tay).
Gợi ý nhãn nguyên nhân: polyp_nho, polyp_phang, mau_khac_biet, anh_mo, phan_chieu, dung_cu,
  quy_uoc_bien_khac, nhieu_polyp, nham_nep_niem_mac, khac.
Chạy: python -m src.analysis.error_overlay --exp E3 --seed 42 --n 12
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from ..inference import load_checkpoint, predict_probs
from ..utils import EXTERNAL_SETS, P, base_cfg, ensure_dir, exp_cfg, read_image, read_mask, run_dir, set_rows


def overlay(img, gt, pred):
    out = img.copy()
    for m, col in [(gt, (0, 255, 0)), (pred, (255, 0, 0))]:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, max(2, img.shape[0] // 200))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sets", nargs="+", default=["kvasir_test"] + EXTERNAL_SETS)
    args = ap.parse_args(argv)
    cfg = base_cfg()
    rd = run_dir(args.exp, args.seed)
    out = ensure_dir(f"reports/errors/{args.exp}_seed{args.seed}")
    model = load_checkpoint(rd / "best.pt", exp_cfg(args.exp))
    table = []
    for s in args.sets:
        f = rd / f"eval_{s}.csv"
        if not f.exists():
            continue
        ev = pd.read_csv(f, dtype={"id": str}).nsmallest(args.n, "dice")
        rows = set_rows(s)
        rows = rows[rows.id.isin(ev.id)].reset_index(drop=True)
        for i, prob in predict_probs(model, rows, cfg["image_size"], num_workers=0):
            r = rows.iloc[i]
            img, gt = read_image(P(r.image_path)), read_mask(P(r.mask_path))
            pred = prob >= cfg["threshold"]
            d = float(ev.loc[ev.id == r.id, "dice"].iloc[0])
            k = int(ev.reset_index(drop=True).index[ev.id.values == r.id][0])
            name = f"worst_{s}_{k:02d}_{r.id}.png"
            cv2.imwrite(str(out / name), cv2.cvtColor(overlay(img, gt, pred), cv2.COLOR_RGB2BGR))
            table.append({"set": s, "id": r.id, "cluster": r.cluster, "dice": round(d, 4),
                          "gt_area_frac": round(float(gt.mean()), 5), "file": name, "nguyen_nhan": ""})
    path = P("reports/errors/gan_nhan_nguyen_nhan_loi.csv")
    pd.DataFrame(table).sort_values(["set", "dice"]).to_csv(path, index=False)
    print(f"Đã ghi {len(table)} ảnh vào {out} và {path}")


if __name__ == "__main__":
    main()
