"""Chia Kvasir-SEG 80/10/10 theo cụm ảnh gần trùng (mục 3.3).

Đầu ra:
  splits/kvasir_split.csv (+ .sha256)   id,cluster,split
  reports/near_duplicates/*.png         các cặp bị nối, để kiểm tra bằng mắt
  reports/split_summary.json

Nếu thấy hai polyp KHÁC nhau bị nối: giảm --tau rồi chạy lại.
Chạy: python -m src.split_kvasir [--tau 10]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from .hashing import cluster_hashes
from .utils import P, base_cfg, ensure_dir, load_index, read_image, save_json, write_with_hash


def load_hashes(dataset: str, ids: list[str]) -> np.ndarray:
    z = np.load(P("data/hashes.npz"))
    pos = {k: i for i, k in enumerate(z["keys"].tolist())}
    return z["hashes"][[pos[f"{dataset}/{i}"] for i in ids]]


def group_split(clusters: np.ndarray, ratios, seed: int) -> np.ndarray:
    """Gán cả cụm vào train/val/test sao cho số ẢNH gần với tỉ lệ yêu cầu."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(clusters, return_counts=True)
    order = rng.permutation(len(uniq))
    n = len(clusters)
    targets = {"test": ratios[2] * n, "val": ratios[1] * n}
    assign, filled = {}, {"test": 0, "val": 0}
    for j in order:
        c, k = uniq[j], counts[j]
        for part in ("test", "val"):
            if filled[part] + k <= targets[part] + 0.5:
                assign[c] = part
                filled[part] += k
                break
        else:
            assign[c] = "train"
    return np.array([assign[c] for c in clusters])


def save_pairs(rows: pd.DataFrame, pairs, out_dir, limit: int = 40):
    ensure_dir(out_dir)
    for f in P(out_dir).glob("*.png"):
        f.unlink()
    for n, (i, j, d) in enumerate(sorted(pairs, key=lambda t: -t[2])[:limit]):
        a = read_image(P(rows.image_path.iloc[i]))
        b = read_image(P(rows.image_path.iloc[j]))
        h = 256
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        canvas = np.concatenate([a, np.full((h, 8, 3), 255, np.uint8), b], axis=1)
        name = f"{n:02d}_d{d}_{rows.id.iloc[i]}__{rows.id.iloc[j]}.png"
        cv2.imwrite(str(P(out_dir) / name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main(argv=None):
    cfg = base_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, default=cfg["hashing"]["tau"])
    args = ap.parse_args(argv)

    rows = load_index().query("dataset == 'kvasir'").reset_index(drop=True)
    H = load_hashes("kvasir", rows.id.tolist())
    labels, pairs = cluster_hashes(H, args.tau, cfg["hashing"]["size"], return_pairs=True)
    clusters = np.array([f"k{c:04d}" for c in labels])
    split = group_split(clusters, cfg["split"]["ratios"], cfg["split"]["seed"])
    df = pd.DataFrame({"id": rows.id, "cluster": clusters, "split": split})

    # Kiểm tra bắt buộc: không cụm nào nằm ở hai phần.
    per_cluster = df.groupby("cluster").split.nunique()
    assert (per_cluster == 1).all(), "Cụm bị chia qua nhiều phần!"
    write_with_hash(df, "splits/kvasir_split.csv")
    save_pairs(rows, pairs, "reports/near_duplicates")

    sizes = df.groupby("cluster").size()
    summary = {"tau": args.tau, "n_images": len(df), "n_clusters": int(sizes.size),
               "n_multi_clusters": int((sizes > 1).sum()), "largest_cluster": int(sizes.max()),
               "n_linked_pairs": len(pairs),
               "counts": df.split.value_counts().to_dict(),
               "cluster_disjoint_check": "ĐẠT"}
    save_json(summary, "reports/split_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
