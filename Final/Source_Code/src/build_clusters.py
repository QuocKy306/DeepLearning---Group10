"""Dựng cụm cho các tập ngoài không có bảng chuỗi chính thức (ColonDB, ETIS, khung âm tính) - mục 3.2.

Ưu tiên bảng chính thức nếu datasets.yaml có clusters_csv (cột id,cluster).
Nếu không, tái dựng bằng băm cảm nhận với ngưỡng hashing.tau (hiệu chỉnh trên ClinicDB).
--sensitivity dùng hashing.tau_sensitivity và ghi splits/clusters_<set>_sens.csv cho phân tích độ nhạy.

Chạy: python -m src.build_clusters [--sets colondb etis negatives] [--sensitivity]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .hashing import cluster_hashes
from .split_kvasir import load_hashes
from .utils import P, base_cfg, datasets_cfg, load_index, save_json, write_with_hash


def main(argv=None):
    cfg = base_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["colondb", "etis", "negatives"])
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args(argv)
    tau = cfg["hashing"]["tau_sensitivity" if args.sensitivity else "tau"]
    suffix = "_sens" if args.sensitivity else ""
    idx = load_index()
    dcfg = datasets_cfg()
    summary = {}
    for s in args.sets:
        rows = idx[idx.dataset == s].reset_index(drop=True)
        if rows.empty:
            print(f"Bỏ qua {s}: không có trong index")
            continue
        official = dcfg.get(s, {}).get("clusters_csv")
        if official and not args.sensitivity:
            tab = pd.read_csv(P(official), dtype={"id": str, "cluster": str})
            df = rows[["id"]].merge(tab, on="id", how="left")
            if df.cluster.isna().any():
                raise SystemExit(f"{s}: bảng cụm chính thức thiếu {df.cluster.isna().sum()} ảnh")
            source = "bảng chính thức"
        else:
            H = load_hashes(s, rows.id.tolist())
            lab = cluster_hashes(H, tau, cfg["hashing"]["size"])
            df = pd.DataFrame({"id": rows.id, "cluster": [f"{s[:2]}{c:04d}" for c in lab]})
            source = f"băm cảm nhận, tau={tau}"
        write_with_hash(df, f"splits/clusters_{s}{suffix}.csv")
        sizes = df.groupby("cluster").size()
        summary[s] = {"source": source, "n_images": len(df), "n_clusters": int(sizes.size),
                      "mean_size": float(sizes.mean()), "max_size": int(sizes.max())}
        print(s, summary[s])
    save_json(summary, f"reports/clusters_summary{suffix}.json")


if __name__ == "__main__":
    main()
