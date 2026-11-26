"""Phân tích cơ chế màu, quan sát (H3a, mục 8.5).

Với mỗi ảnh đích: khoảng cách Mahalanobis của vector Lab 6 chiều tới phân bố Kvasir-train;
Dice trung bình qua seed được hồi quy theo khoảng cách màu, log diện tích polyp, độ tương phản, độ nét,
hiệu ứng cố định theo bộ, sai số chuẩn vững theo cụm. H3a dự đoán hệ số của dist âm.

Đầu ra: reports/color/spearman.csv, reports/color/regression.csv, reports/color/per_image.csv,
        reports/color/lab_means.png
Chạy: python -m src.analysis.color_analysis [--exps E3 E5]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from ..color import mahalanobis
from ..eda import COLS, compute_lab_table
from ..stats import load_set
from ..utils import EXTERNAL_SETS, P, all_seeds, base_cfg, ensure_dir, load_index, set_rows


def lab_table() -> pd.DataFrame:
    f = P("reports/eda/lab_stats.csv")
    if f.exists():
        return pd.read_csv(f, dtype={"id": str})
    lab = compute_lab_table(load_index())
    ensure_dir(f.parent)
    lab.to_csv(f, index=False)
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["E3", "E5"])
    args = ap.parse_args(argv)
    cfg = base_cfg()
    out = ensure_dir("reports/color")
    lab = lab_table()
    idx = load_index()[["dataset", "id", "mask_area_frac"]]
    train_ids = set(set_rows("kvasir_train").id)
    ref = lab[(lab.dataset == "kvasir") & lab.id.isin(train_ids)][COLS].values
    lab["dist"] = mahalanobis(lab[COLS].values, ref)
    lab = lab.merge(idx, on=["dataset", "id"])

    # Hình: trung bình a*, b* theo bộ
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for d, g in lab.groupby("dataset"):
        ax.scatter(g.mu_a, g.mu_b, s=6, alpha=0.4, label=d)
    ax.set_xlabel("trung bình a*")
    ax.set_ylabel("trung bình b*")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "lab_means.png", dpi=150)
    plt.close(fig)

    sp_rows, reg_rows, per_img = [], [], []
    for exp in args.exps:
        seeds = all_seeds(cfg, exp, include_extra=True)
        parts = []
        for s in EXTERNAL_SETS + ["kvasir_test"]:
            d = load_set(exp, s, seeds)
            if not d:
                continue
            mean = sum(df.dice for df in d.values()) / len(d)
            base = next(iter(d.values()))
            ds = "kvasir" if s == "kvasir_test" else s
            parts.append(pd.DataFrame({"set": s, "dataset": ds, "id": base.id.values,
                                       "cluster": base.cluster.values, "dice": mean.values}))
        if not parts:
            print(f"[bỏ qua] {exp}: chưa có kết quả tập kiểm tra")
            continue
        df = pd.concat(parts).merge(lab, on=["dataset", "id"])
        df["log_area"] = np.log(df.mask_area_frac.clip(lower=1e-4))
        df["exp"] = exp
        per_img.append(df)
        for s, g in list(df.groupby("set")) + [("external_pooled", df[df.set != "kvasir_test"])]:
            if len(g) > 5:
                r, p = spearmanr(g.dist, g.dice)
                sp_rows.append({"exp": exp, "set": s, "rho": r, "p_naive": p, "n": len(g),
                                "note": "mô tả; p không tính cụm"})
        ext = df[df.set != "kvasir_test"].copy()
        if ext.set.nunique() >= 1 and len(ext) > 10:
            formula = "dice ~ dist + log_area + contrast + sharpness"
            if ext.set.nunique() > 1:
                formula += " + C(set)"
            groups = pd.factorize(ext.set + "|" + ext.cluster)[0]
            fit = smf.ols(formula, data=ext).fit(cov_type="cluster", cov_kwds={"groups": groups})
            ci = fit.conf_int()
            for term in fit.params.index:
                reg_rows.append({"exp": exp, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                                 "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                                 "p": fit.pvalues[term], "n": int(fit.nobs),
                                 "n_clusters": int(groups.max() + 1)})
    pd.DataFrame(sp_rows).to_csv(out / "spearman.csv", index=False)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out / "regression.csv", index=False)
    if per_img:
        pd.concat(per_img).to_csv(out / "per_image.csv", index=False)
    if len(reg):
        print(reg[reg.term.isin(["dist", "log_area"])].round(4).to_string(index=False))
    print("Đã ghi reports/color/")


if __name__ == "__main__":
    main()
