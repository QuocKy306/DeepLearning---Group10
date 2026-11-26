"""Thí nghiệm phản thực tế về màu (mục 6.4). Chạy SAU đánh giá chính trên tập kiểm tra.

Thuận: chuẩn hóa màu ảnh đích về Kvasir-train, đánh giá lại cùng checkpoint ->
       R_k = (Dice_norm - Dice_orig) / (Dice_source - Dice_orig)  = tỉ lệ khoảng cách được khôi phục.
Ngược: chuyển màu Kvasir-test sang thống kê của từng bộ đích -> Dice giảm bao nhiêu.

Đầu ra: reports/color/counterfactual.csv
Chạy: python -m src.analysis.color_counterfactual --exp E3
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import evaluate
from ..stats import cluster_bootstrap_ci
from ..utils import EXTERNAL_SETS, P, all_seeds, base_cfg, ensure_dir, run_dir


def _load(exp, seeds, name):
    out = {}
    for s in seeds:
        f = run_dir(exp, s) / f"eval_{name}.csv"
        if f.exists():
            out[s] = pd.read_csv(f, dtype={"id": str, "cluster": str}).set_index("id")
    return out


def _mat(d, ids):
    return np.stack([d[s].loc[ids, "dice"].values for s in sorted(d)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E3")
    ap.add_argument("--skip_eval", action="store_true", help="chỉ tổng hợp từ file đã có")
    args = ap.parse_args(argv)
    cfg = base_cfg()
    B = cfg["stats"]["bootstrap"]
    seeds = [s for s in all_seeds(cfg, args.exp) if (run_dir(args.exp, s) / "eval_kvasir_test.csv").exists()]
    if not seeds:
        raise SystemExit("Chưa đánh giá kvasir_test cho thí nghiệm này.")
    sets = [s for s in EXTERNAL_SETS if (run_dir(args.exp, seeds[0]) / f"eval_{s}.csv").exists()]
    if not args.skip_eval:
        for seed in seeds:
            evaluate.main(["--exp", args.exp, "--seed", str(seed), "--sets", *sets, "--color_norm", "to_kvasir"])
            for s in sets:
                evaluate.main(["--exp", args.exp, "--seed", str(seed), "--sets", "kvasir_test",
                               "--color_norm", f"to_{s}"])
    src = _load(args.exp, seeds, "kvasir_test")
    src_ids = src[seeds[0]].index
    S = _mat(src, src_ids)
    _, _, bS = cluster_bootstrap_ci(S, src[seeds[0]].cluster.values, B, seed=0, return_samples=True)
    rows = []
    for s in sets:
        orig = _load(args.exp, seeds, s)
        norm = _load(args.exp, seeds, f"{s}_cf-to_kvasir")
        ids = orig[seeds[0]].index
        O, N = _mat(orig, ids), _mat(norm, ids)
        cl = orig[seeds[0]].cluster.values
        _, _, bO = cluster_bootstrap_ci(O, cl, B, seed=1, return_samples=True)
        _, _, bD = cluster_bootstrap_ci(N - O, cl, B, seed=1, return_samples=True)  # cùng seed RNG -> cùng mẫu cụm
        gap = S.mean() - O.mean()
        R = (N.mean() - O.mean()) / gap if abs(gap) > 1e-9 else np.nan
        bR = bD / (bS - bO)
        rev = _load(args.exp, seeds, f"kvasir_test_cf-to_{s}")
        Rv = _mat(rev, src_ids)
        lo_r, hi_r = cluster_bootstrap_ci(Rv - S, src[seeds[0]].cluster.values, B, seed=2)
        rows.append({"exp": args.exp, "target": s, "dice_source": S.mean(), "dice_target_orig": O.mean(),
                     "dice_target_norm": N.mean(), "gap": gap, "recovered": N.mean() - O.mean(),
                     "R": R, "R_ci_low": float(np.nanquantile(bR, 0.025)),
                     "R_ci_high": float(np.nanquantile(bR, 0.975)),
                     "reverse_drop": float((Rv - S).mean()), "reverse_ci_low": lo_r, "reverse_ci_high": hi_r,
                     "n_seeds": len(seeds)})
    df = pd.DataFrame(rows)
    ensure_dir("reports/color")
    df.to_csv(P("reports/color/counterfactual.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
