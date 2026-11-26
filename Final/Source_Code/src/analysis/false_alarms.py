"""Báo động giả trên khung âm tính (RQ5, mục 4.5.1): khung bị tính báo động giả khi diện tích dự đoán > 0,1% ảnh.

Với mỗi mô hình: tỉ lệ báo động giả (CI bootstrap theo cụm và theo ảnh, CI theo cụm của phân tích độ nhạy),
tỉ lệ theo lớp khung, so sánh ghép cặp giữa các mô hình (hoán dấu theo cụm; KHÔNG hiệu chỉnh bội, khám phá),
AUC của độ tin cậy tối đa giữa Kvasir-test (dương) và khung âm tính, và đường đánh đổi theo ngưỡng nhị phân.
Lưu ý: AUC gộp hai loại cảnh khác nhau (khung có polyp so với khung mốc giải phẫu) nên bị nhiễu bởi loại cảnh.

Đầu ra: reports/negatives_results.json, reports/negatives_threshold_tradeoff.csv
Chạy:  python -m src.analysis.false_alarms --exps P2 P3 P5 --seed 42
(cần: đã đánh giá, đã dựng splits/clusters_negatives(.csv|_sens.csv) bằng src.build_clusters)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..inference import load_checkpoint, predict_probs
from ..metrics import per_image_metrics
from ..stats import cluster_bootstrap_ci, signflip_p
from ..utils import P, base_cfg, exp_cfg, read_mask, run_dir, save_json, set_rows

AMIN = 0.001


def fp(pm, t):
    return np.array([(p >= t).mean() > AMIN for p in pm])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["P2", "P3", "P5"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    cfg = base_cfg()
    neg, kt = set_rows("negatives"), set_rows("kvasir_test")
    neg["cls"] = neg.id.str.split("__").str[0]
    sens = pd.read_csv(P("splits/clusters_negatives_sens.csv"), dtype={"id": str, "cluster": str}).set_index("id").cluster
    gts = [read_mask(P(r.mask_path)) for r in kt.itertuples()]
    probs, R, rows = {}, {}, []
    for e in args.exps:
        m = load_checkpoint(run_dir(e, args.seed) / "best.pt", exp_cfg(e))
        probs[e] = {}
        for name, d in [("neg", neg), ("kt", kt)]:
            o = sorted(predict_probs(m, d, cfg["image_size"], num_workers=0), key=lambda t: t[0])
            probs[e][name] = [p for _, p in o]
        f = fp(probs[e]["neg"], 0.5)
        x = f[None, :].astype(float)
        area = np.array([(p >= 0.5).mean() for p in probs[e]["neg"]])
        R[e] = dict(fp=float(f.mean()), ci=list(cluster_bootstrap_ci(x, neg.cluster.values, 10000)),
                    ci_naive=list(cluster_bootstrap_ci(x, np.arange(len(f)), 10000)),
                    ci_sens=list(cluster_bootstrap_ci(x, sens.loc[neg.id].values, 5000)),
                    by_class={c: float(f[(neg.cls == c).values].mean()) for c in sorted(neg.cls.unique())},
                    area_median_fp=float(np.median(area[f])) if f.any() else None,
                    n_clusters=int(neg.cluster.nunique()))
        s_neg = np.array([p.max() for p in probs[e]["neg"]]); s_pos = np.array([p.max() for p in probs[e]["kt"]])
        a_neg = np.array([(p >= 0.5).mean() for p in probs[e]["neg"]]); a_pos = np.array([(p >= 0.5).mean() for p in probs[e]["kt"]])
        yy = lambda a, b: np.r_[np.ones(len(a)), np.zeros(len(b))]
        R[e]["auc_maxprob"] = float(roc_auc_score(yy(s_pos, s_neg), np.r_[s_pos, s_neg]))
        R[e]["auc_area"] = float(roc_auc_score(yy(a_pos, a_neg), np.r_[a_pos, a_neg]))
        R[e]["median_maxprob"] = dict(pos=float(np.median(s_pos)), neg=float(np.median(s_neg)))
        for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
            dice = np.mean([per_image_metrics(p >= t, g)[0]["dice"] for p, g in zip(probs[e]["kt"], gts)])
            rows.append(dict(exp=e, thr=t, fp=float(fp(probs[e]["neg"], t).mean()), dice=float(dice),
                             detect=float(np.mean(fp(probs[e]["kt"], t)))))
    for a, b in [("P5", "P3"), ("P3", "P2")]:
        if a in probs and b in probs:
            d = fp(probs[a]["neg"], .5).astype(float) - fp(probs[b]["neg"], .5).astype(float)
            lo, hi = cluster_bootstrap_ci(d[None, :], neg.cluster.values, 10000)
            R[f"{a}-{b}"] = dict(diff=float(d.mean()), ci=[lo, hi], p=signflip_p(d[None, :], neg.cluster.values, 20000, 15))
    pd.DataFrame(rows).to_csv(P("reports/negatives_threshold_tradeoff.csv"), index=False)
    R["tradeoff"] = rows
    save_json(R, "reports/negatives_results.json")
    for e in args.exps:
        print(e, "báo động giả %.3f" % R[e]["fp"], [round(v, 3) for v in R[e]["ci"]], R[e]["by_class"], "AUC %.3f" % R[e]["auc_maxprob"])
    for k in ("P5-P3", "P3-P2"):
        if k in R:
            print(k, R[k])


if __name__ == "__main__":
    main()
