"""E9: đo mức thổi phồng do rò rỉ theo chuỗi video trên CVC-ClinicDB (mục 6.5).

  folds    tạo splits/e9_folds.csv: fold_random (KFold xáo trộn theo khung) và
           fold_sequence (GroupKFold theo chuỗi), cùng 612 ảnh
  summary  gộp runs/E9/<scheme>/fold*/oof.csv -> reports/e9_leakage.csv, reports/e9_leakage_summary.json
           Mức thổi phồng = Dice(random) - Dice(sequence), so cặp trên cùng ảnh,
           khoảng tin cậy bootstrap theo chuỗi, p hoán dấu theo chuỗi.
Chạy: python -m src.e9_leakage folds ; ... ; python -m src.e9_leakage summary
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from .stats import cluster_bootstrap_ci, signflip_p
from .utils import P, base_cfg, save_json, set_rows, write_with_hash


def make_folds(cfg):
    rows = set_rows("clinicdb")
    if rows.cluster.str.startswith("img_").any():
        raise SystemExit("Chưa có splits/clusters_clinicdb.csv. Chạy python -m src.clinicdb_sequences trước.")
    k = cfg["e9"]["folds"]
    fr = np.zeros(len(rows), int)
    for f, (_, te) in enumerate(KFold(k, shuffle=True, random_state=cfg["e9"]["seed"]).split(rows)):
        fr[te] = f
    fs = np.zeros(len(rows), int)
    for f, (_, te) in enumerate(GroupKFold(k).split(rows, groups=rows.cluster)):
        fs[te] = f
    df = pd.DataFrame({"id": rows.id, "cluster": rows.cluster, "fold_random": fr, "fold_sequence": fs})
    # kiểm tra: ở sơ đồ theo chuỗi, mỗi chuỗi chỉ thuộc một phần
    assert (df.groupby("cluster").fold_sequence.nunique() == 1).all()
    write_with_hash(df, "splits/e9_folds.csv")
    leak = df.groupby("cluster").fold_random.nunique()
    print(f"Đã ghi splits/e9_folds.csv. Sơ đồ ngẫu nhiên: {int((leak > 1).sum())}/{len(leak)} chuỗi "
          "bị chia qua nhiều phần (đây chính là rò rỉ cần đo).")


def summary(cfg):
    parts = {}
    for scheme in ["random", "sequence"]:
        fs = sorted(P(f"runs/E9/{scheme}").glob("fold*/oof.csv"))
        if not fs:
            raise SystemExit(f"Thiếu kết quả runs/E9/{scheme}/fold*/oof.csv")
        parts[scheme] = pd.concat([pd.read_csv(f, dtype={"id": str, "cluster": str}) for f in fs])
    per_fold = pd.concat(parts.values()).groupby(["scheme", "fold"]).dice.mean().reset_index()
    a = parts["random"][["id", "cluster", "dice"]].rename(columns={"dice": "dice_random"})
    b = parts["sequence"][["id", "dice"]].rename(columns={"dice": "dice_sequence"})
    m = a.merge(b, on="id")
    n_total = len(set_rows("clinicdb"))
    complete = len(m) == n_total
    diff = (m.dice_random - m.dice_sequence).values[None, :]  # 1 "seed"
    clusters = m.cluster.values
    est = float(diff.mean())
    lo, hi = cluster_bootstrap_ci(diff, clusters, cfg["stats"]["bootstrap"], seed=0)
    p = signflip_p(diff, clusters, cfg["stats"]["permutations"], cfg["stats"]["exact_max_clusters"])
    res = {"n_images_paired": len(m), "complete": complete, "n_sequences": int(m.cluster.nunique()),
           "dice_random": float(m.dice_random.mean()), "dice_sequence": float(m.dice_sequence.mean()),
           "inflation": est, "ci95": [lo, hi], "p_signflip": p,
           "per_sequence_dice_sequence": m.groupby("cluster").dice_sequence.mean().round(4).to_dict()}
    per_fold.to_csv(P("reports/e9_leakage.csv"), index=False)
    m.to_csv(P("reports/e9_leakage_per_image.csv"), index=False)
    save_json(res, "reports/e9_leakage_summary.json")
    print(per_fold.to_string(index=False))
    print({k: v for k, v in res.items() if k != "per_sequence_dice_sequence"})
    if not complete:
        print(f"[CẢNH BÁO] Mới có {len(m)}/{n_total} ảnh có dự đoán ở cả hai sơ đồ.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["folds", "summary"])
    args = ap.parse_args(argv)
    cfg = base_cfg()
    make_folds(cfg) if args.cmd == "folds" else summary(cfg)


if __name__ == "__main__":
    main()
