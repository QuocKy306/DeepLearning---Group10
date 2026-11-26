"""Năm so sánh khai báo trước ở QUY MÔ THU NHỎ (P1..P8 thay cho E1..E8; 160 px, 12 epoch, 1 seed).

Ánh xạ: C1 = P3-P1, C2 = P3-P8, C3 = P3-P2, C4 = P5-P3, C5 = P5-P4; mô tả: P6-P3, P7-P3, P5-P6, P5-P7.
Mỗi tập (Kvasir-test, ClinicDB, ColonDB, ETIS, gộp ngoài): hiệu mDice ghép cặp, CI bootstrap theo cụm (gộp: phân tầng
theo bộ), p hoán dấu theo cụm, Holm cho C1..C5 trong mỗi tập; kèm CI/p coi mỗi ảnh độc lập để đối chiếu.
Thêm: mDice và Gap từng cấu hình, tỉ lệ báo động giả trên khung âm tính.

Đầu ra: reports/reduced_scale/{planned.csv, means.csv, false_alarm.csv}
Chạy: python -m src.analysis.reduced_scale
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import cluster_bootstrap_ci, holm, signflip_p
from ..utils import EXTERNAL_SETS, P, ensure_dir, run_dir

EXPS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
PLANNED = [("C1", "RQ2/H1", "P3", "P1"), ("C2", "RQ2/H1", "P3", "P8"), ("C3", "RQ2/H2", "P3", "P2"),
           ("C4", "RQ3/H3b", "P5", "P3"), ("C5", "RQ3/H3b", "P5", "P4")]
SECOND = [("S1", "mô tả", "P6", "P3"), ("S2", "mô tả", "P7", "P3"), ("S3", "mô tả", "P5", "P6"), ("S4", "mô tả", "P5", "P7")]
SETS = ["kvasir_test"] + EXTERNAL_SETS + ["pooled"]


def load(e, s):
    sets = EXTERNAL_SETS if s == "pooled" else [s]
    d = pd.concat([pd.read_csv(run_dir(e, 42) / f"eval_{x}.csv", dtype={"id": str, "cluster": str}) for x in sets])
    d["key"] = d["set"] + "/" + d["id"]
    return d.set_index("key")


def compare(a, b, s, B=10000):
    A, Bm = load(a, s), load(b, s)
    Bm = Bm.loc[A.index]
    x = (A.dice - Bm.dice).values[None, :]
    cl, st = A.cluster.values, (A["set"].values if s == "pooled" else None)
    lo, hi = cluster_bootstrap_ci(x, cl, B, strata=st)
    lon, hin = cluster_bootstrap_ci(x, np.arange(x.shape[1]), B, strata=st)
    return dict(set=s, a=a, b=b, mean_a=float(A.dice.mean()), mean_b=float(Bm.dice.mean()), diff=float(x.mean()),
                ci_low=lo, ci_high=hi, p=signflip_p(x, cl, 20000, 15, strata=st),
                ci_low_naive=lon, ci_high_naive=hin, p_naive=signflip_p(x, np.arange(x.shape[1]), 20000, 0, strata=st),
                n=int(x.shape[1]), m=int(len(set(zip(st, cl)) if st is not None else set(cl))))


def main():
    out = ensure_dir("reports/reduced_scale")
    rows = []
    for s in SETS:
        blk = [dict(id=c, rq=rq, **compare(a, b, s)) for c, rq, a, b in PLANNED]
        for r, ph in zip(blk, holm([r["p"] for r in blk])):
            r["p_holm"] = float(ph)
        rows += blk + [dict(id=c, rq=rq, p_holm=np.nan, **compare(a, b, s, 4000)) for c, rq, a, b in SECOND]
    pd.DataFrame(rows).to_csv(out / "planned.csv", index=False)
    means = []
    for e in EXPS:
        kt = load(e, "kvasir_test")
        _, _, bk = cluster_bootstrap_ci(kt.dice.values[None, :], kt.cluster.values, 10000, seed=0, return_samples=True)
        for s in SETS:
            d = load(e, s)
            st = d["set"].values if s == "pooled" else None
            lo, hi, bs = cluster_bootstrap_ci(d.dice.values[None, :], d.cluster.values, 10000, seed=1, strata=st, return_samples=True)
            g = bk - bs
            means.append(dict(exp=e, set=s, mdice=float(d.dice.mean()), ci_low=lo, ci_high=hi, nsd=float(d.nsd.mean()),
                              gap=float(kt.dice.mean() - d.dice.mean()) if s != "kvasir_test" else np.nan,
                              gap_low=float(np.quantile(g, .025)) if s != "kvasir_test" else np.nan,
                              gap_high=float(np.quantile(g, .975)) if s != "kvasir_test" else np.nan))
    pd.DataFrame(means).to_csv(out / "means.csv", index=False)
    fa = []
    for e in EXPS:
        d = pd.read_csv(run_dir(e, 42) / "eval_negatives.csv", dtype={"cluster": str})
        x = d.fp_frame.values[None, :].astype(float)
        lo, hi = cluster_bootstrap_ci(x, d.cluster.values, 10000)
        fa.append(dict(exp=e, fp=float(x.mean()), ci_low=lo, ci_high=hi))
    pd.DataFrame(fa).to_csv(out / "false_alarm.csv", index=False)
    pl = pd.DataFrame(rows)
    show = pl[pl.id.str.startswith("C")][["set", "id", "a", "b", "diff", "ci_low", "ci_high", "p_holm", "ci_low_naive", "ci_high_naive", "m"]]
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(fa).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
