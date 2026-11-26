"""Khảo sát dữ liệu: phân bố diện tích polyp và màu Lab theo bộ dữ liệu (động lực của RQ3).

Đầu ra: reports/eda/lab_stats.csv, reports/eda/area_hist.png, reports/eda/lab_hist.png, reports/eda/summary.csv
Chạy: python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .color import lab_stats, quality_features
from .utils import P, ensure_dir, load_index, read_image

COLS = ["mu_L", "mu_a", "mu_b", "sd_L", "sd_a", "sd_b"]


def compute_lab_table(idx: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r in tqdm(idx.itertuples(), total=len(idx), desc="Lab"):
        img = read_image(P(r.image_path))
        recs.append({"dataset": r.dataset, "id": r.id,
                     **dict(zip(COLS, lab_stats(img).tolist())), **quality_features(img)})
    return pd.DataFrame(recs)


def main():
    out = ensure_dir("reports/eda")
    idx = load_index()
    lab = compute_lab_table(idx)
    lab.to_csv(out / "lab_stats.csv", index=False)
    sets = list(idx.dataset.unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in sets:
        a = idx.loc[(idx.dataset == d) & idx.mask_area_frac.notna(), "mask_area_frac"]
        if len(a):
            ax.hist(np.log10(a.clip(lower=1e-4)), bins=30, alpha=0.5, label=d, density=True)
    ax.set_xlabel("log10(diện tích polyp / diện tích ảnh)")
    ax.set_ylabel("mật độ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "area_hist.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, c in zip(axes, ["mu_L", "mu_a", "mu_b"]):
        for d in sets:
            ax.hist(lab.loc[lab.dataset == d, c], bins=30, alpha=0.5, label=d, density=True)
        ax.set_xlabel(c)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "lab_hist.png", dpi=150)
    plt.close(fig)

    summ = lab.groupby("dataset")[COLS + ["contrast", "sharpness"]].agg(["mean", "std"])
    area = idx.groupby("dataset").mask_area_frac.describe()
    summ.to_csv(out / "summary.csv")
    area.to_csv(out / "area_summary.csv")
    print(summ.round(2))
    print(area.round(4))


if __name__ == "__main__":
    main()
