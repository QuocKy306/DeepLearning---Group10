"""Thống kê (mục 8): bootstrap phân tầng theo seed và cụm, kiểm định hoán dấu theo cụm, Holm.

Lệnh:
  planned --sets kvasir_val                    -> reports/stats/planned_kvasir_val.csv
  planned --sets kvasir_test clinicdb colondb etis
          (tự thêm tập gộp 'external_pooled' khi có đủ ba bộ ngoài)
  table   --sets ...                           -> reports/stats/table_<sets>.csv (trung bình ± sd giữa seed)
  rq3                                          -> reports/stats/rq3_domain_gap.csv, rq3_color.csv

Cột RQ: RQ1/RQ2/RQ3 = năm so sánh Dice khai báo trước, đã hiệu chỉnh Holm trong mỗi tập.
        'phu' = độ đo phụ hoặc so sánh mô tả, không hiệu chỉnh, không dùng để kết luận.
Cột unit: đơn vị lấy mẫu lại (chuỗi video, cụm...).
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from .utils import (EXTERNAL_SETS, P, all_seeds, base_cfg, datasets_cfg, ensure_dir,
                    experiments_cfg, run_dir)

SECONDARY_METRICS = ["iou", "precision", "recall", "hd95", "nsd"]
POOLED = "external_pooled"


# ---------------------------------------------------------------- lõi thống kê
def _codes(clusters, strata=None):
    clusters = np.asarray(clusters).astype(str)
    if strata is not None:
        clusters = np.char.add(np.char.add(np.asarray(strata).astype(str), "|"), clusters)
    uniq, code = np.unique(clusters, return_inverse=True)
    cl_strata = None
    if strata is not None:
        cl_strata = np.array([u.split("|", 1)[0] for u in uniq])
    return code, len(uniq), cl_strata


def cluster_bootstrap_ci(diff: np.ndarray, clusters, B: int = 10000, seed: int = 0,
                         strata=None, alpha: float = 0.05, return_samples: bool = False):
    """Bootstrap phân tầng: lấy mẫu lại seed (hàng của diff) rồi lấy mẫu lại cụm (trong từng tầng nếu có).

    diff: (S, n) hiệu theo seed và ảnh (đã ghép cặp). Đại lượng: trung bình theo ảnh của trung bình qua seed.
    """
    diff = np.atleast_2d(np.asarray(diff, float))
    S, n = diff.shape
    code, m, cl_strata = _codes(clusters, strata)
    csum = np.zeros((S, m))
    for s in range(S):
        csum[s] = np.bincount(code, weights=diff[s], minlength=m)
    ncl = np.bincount(code, minlength=m).astype(float)
    rng = np.random.default_rng(seed)
    groups = [np.arange(m)] if cl_strata is None else [np.where(cl_strata == g)[0] for g in np.unique(cl_strata)]
    out = []
    for start in range(0, B, 1000):
        b = min(1000, B - start)
        V = rng.multinomial(S, np.full(S, 1 / S), size=b) / S          # (b, S)
        W = np.zeros((b, m))
        for g in groups:
            W[:, g] = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=b)
        num = ((V @ csum) * W).sum(1)
        den = W @ ncl
        out.append(num / den)
    samples = np.concatenate(out)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    if return_samples:
        return float(lo), float(hi), samples
    return float(lo), float(hi)


def signflip_p(diff: np.ndarray, clusters, R: int = 20000, exact_max: int = 15,
               strata=None, seed: int = 1) -> float:
    """Kiểm định hoán dấu hai phía theo cụm trên tổng hiệu (trung bình qua seed) của từng cụm."""
    diff = np.atleast_2d(np.asarray(diff, float)).mean(0)
    code, m, _ = _codes(clusters, strata)
    D = np.bincount(code, weights=diff, minlength=m)
    n = len(diff)
    T = abs(D.sum() / n)
    if m <= exact_max:
        signs = np.array(list(itertools.product([-1, 1], repeat=m)))
        Tr = np.abs(signs @ D / n)
        return float((Tr >= T - 1e-12).mean())
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, R, 5000):
        r = min(5000, R - start)
        E = rng.choice([-1.0, 1.0], size=(r, m))
        count += int((np.abs(E @ D / n) >= T - 1e-12).sum())
    return float((count + 1) / (R + 1))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- nạp kết quả
def load_eval(exp: str, seed: int, set_name: str) -> pd.DataFrame | None:
    f = run_dir(exp, seed) / f"eval_{set_name}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, dtype={"id": str, "cluster": str})


def _override_clusters(df: pd.DataFrame, set_name: str, suffix: str) -> pd.DataFrame:
    """Phân tích độ nhạy: thay cột cluster bằng splits/clusters_<set><suffix>.csv (không cần đánh giá lại)."""
    f = P(f"splits/clusters_{set_name}{suffix}.csv")
    if not suffix or not f.exists():
        return df
    cl = pd.read_csv(f, dtype={"id": str, "cluster": str})
    return df.drop(columns="cluster").merge(cl[["id", "cluster"]], on="id", how="left")


def load_set(exp: str, set_name: str, seeds, suffix: str = "") -> dict[int, pd.DataFrame]:
    sets = EXTERNAL_SETS if set_name == POOLED else [set_name]
    out = {}
    for s in seeds:
        parts = [load_eval(exp, s, x) for x in sets]
        parts = [None if p is None else _override_clusters(p, x, suffix) for p, x in zip(parts, sets)]
        if all(p is not None for p in parts):
            df = pd.concat(parts, ignore_index=True)
            df["key"] = df["set"] + "/" + df["id"]
            out[s] = df.set_index("key")
    return out


def paired_matrix(a: dict, b: dict, metric: str):
    """Ghép cặp theo seed (cùng giá trị seed) và theo ảnh. Trả về diff (S, n), cluster, strata, seeds."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    keys = a[seeds[0]].index
    for s in seeds:
        keys = keys.intersection(a[s].index).intersection(b[s].index)
    A = np.stack([a[s].loc[keys, metric].values for s in seeds]).astype(float)
    Bm = np.stack([b[s].loc[keys, metric].values for s in seeds]).astype(float)
    ok = ~(np.isnan(A).any(0) | np.isnan(Bm).any(0))  # hd95 có thể NaN
    ref = a[seeds[0]].loc[keys]
    return (A - Bm)[:, ok], ref.cluster.values[ok], ref["set"].values[ok], seeds, A[:, ok], Bm[:, ok]


def unit_of(set_name: str) -> str:
    if set_name.startswith("kvasir"):
        return "cụm ảnh gần trùng"
    if set_name == POOLED:
        return "cụm, phân tầng theo bộ"
    return datasets_cfg().get(set_name, {}).get("unit", "cụm")


# ---------------------------------------------------------------- lệnh
def compare(a_exp, b_exp, set_name, metric, cfg, include_extra=True, suffix=""):
    sa = all_seeds(cfg, a_exp, include_extra)
    sb = all_seeds(cfg, b_exp, include_extra)
    a = load_set(a_exp, set_name, sa, suffix)
    b = load_set(b_exp, set_name, sb, suffix)
    res = paired_matrix(a, b, metric)
    if res is None:
        return None
    diff, cl, st, seeds, A, Bm = res
    if diff.shape[1] < 2:  # vd. HD95 rỗng ở mọi ảnh
        return None
    strata = st if set_name == POOLED else None
    sc = cfg["stats"]
    lo, hi = cluster_bootstrap_ci(diff, cl, sc["bootstrap"], seed=0, strata=strata)
    p = signflip_p(diff, cl, sc["permutations"], sc["exact_max_clusters"], strata=strata)
    # Đối chứng phương pháp (mục 5.1b): cùng phép tính nhưng coi mỗi ảnh là một cụm (giả định độc lập sai).
    naive = np.arange(diff.shape[1])
    lo_n, hi_n = cluster_bootstrap_ci(diff, naive, sc["bootstrap"], seed=0, strata=strata)
    p_n = signflip_p(diff, naive, sc["permutations"], 0, strata=strata)
    return {"ci_low_naive": lo_n, "ci_high_naive": hi_n, "p_naive": p_n,
            "ci_width_ratio": (hi - lo) / (hi_n - lo_n) if hi_n > lo_n else float("nan"),"set": set_name, "metric": metric, "mean_a": float(A.mean()), "mean_b": float(Bm.mean()),
            "estimate": float(diff.mean()), "ci_low": lo, "ci_high": hi, "p_value": p,
            "n_images": int(diff.shape[1]), "n_clusters": int(len(np.unique(cl))),
            "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)), "unit": unit_of(set_name)}


def cmd_planned(sets, cfg, suffix=""):
    ex = experiments_cfg()
    work = list(sets)
    if all(s in sets for s in EXTERNAL_SETS):
        work.append(POOLED)
    rows = []
    for s in work:
        block = []
        for c in ex["planned"]:
            r = compare(c["a"], c["b"], s, "dice", cfg, suffix=suffix)
            if r is None:
                print(f"[bỏ qua] {c['id']} trên {s}: thiếu kết quả")
                continue
            block.append({"RQ": c["rq"], "id": c["id"], "comparison": f"{c['a']} - {c['b']}", **r})
        if block:
            adj = holm([b["p_value"] for b in block])
            for b, pa in zip(block, adj):
                b["p_holm"] = float(pa)
                b["reject_holm"] = bool(pa < cfg["stats"]["alpha"])
            rows += block
        for c in ex["planned"]:
            for met in SECONDARY_METRICS:
                r = compare(c["a"], c["b"], s, met, cfg, suffix=suffix)
                if r:
                    rows.append({"RQ": "phu", "id": c["id"], "comparison": f"{c['a']} - {c['b']}", **r})
        for c in ex["secondary"]:
            r = compare(c["a"], c["b"], s, "dice", cfg, suffix=suffix)
            if r:
                rows.append({"RQ": "phu", "id": c["id"], "comparison": f"{c['a']} - {c['b']}", **r})
    df = pd.DataFrame(rows)
    out = ensure_dir("reports/stats") / f"planned_{'_'.join(sets)}{suffix}.csv"
    df.to_csv(out, index=False)
    show = df[df.RQ != "phu"] if len(df) else df
    if len(show):
        print(show[["set", "RQ", "comparison", "estimate", "ci_low", "ci_high", "p_value", "p_holm",
                    "reject_holm", "n_clusters", "unit"]].round(4).to_string(index=False))
    print(f"Đã ghi {out}")
    return df


def cmd_table(sets, cfg):
    ex = experiments_cfg()["experiments"]
    recs = []
    for exp in ex:
        if exp == "E9":
            continue
        for s in sets:
            per_seed = []
            for seed in all_seeds(cfg, exp, include_extra=True):
                df = load_eval(exp, seed, s)
                if df is None:
                    continue
                pol = run_dir(exp, seed) / f"eval_{s}_polyps.csv"
                pol = pd.read_csv(pol) if pol.exists() and pol.stat().st_size > 1 else pd.DataFrame()
                if "dice" in df:
                    rec = {k: df[k].mean() for k in ["dice", "iou", "precision", "recall"]}
                    rec["hd95_nonempty"] = df.hd95.mean(skipna=True)
                    if "nsd" in df:
                        rec["nsd"] = df.nsd.mean()
                    rec["empty_pred_rate"] = df.empty_pred.mean()
                    rec["polyp_sensitivity"] = pol.detected.mean() if len(pol) else np.nan
                    for sz in ["nho", "vua", "lon"]:
                        q = pol[pol["size"] == sz] if len(pol) else pol
                        rec[f"sens_{sz}"] = q.detected.mean() if len(q) else np.nan
                    rec["fp_comp_per_img"] = df.n_fp_components.mean()
                else:
                    rec = {"fp_frame_rate": df.fp_frame.mean(), "fp_comp_per_img": df.n_fp_components.mean()}
                per_seed.append(rec)
            if not per_seed:
                continue
            d = pd.DataFrame(per_seed)
            row = {"exp": exp, "set": s, "n_seeds": len(d)}
            for k in d.columns:
                row[f"{k}_mean"] = d[k].mean()
                row[f"{k}_sd"] = d[k].std(ddof=1) if len(d) > 1 else np.nan
            recs.append(row)
    df = pd.DataFrame(recs)
    out = ensure_dir("reports/stats") / f"table_{'_'.join(sets)}.csv"
    df.to_csv(out, index=False)
    cols = [c for c in ["exp", "set", "n_seeds", "dice_mean", "dice_sd", "iou_mean", "hd95_nonempty_mean",
                        "polyp_sensitivity_mean", "fp_frame_rate_mean"] if c in df]
    if len(df):
        print(df[cols].round(4).to_string(index=False))
    print(f"Đã ghi {out}")


def cmd_rq3(cfg):
    ex = experiments_cfg()["experiments"]
    sc = cfg["stats"]
    gap_rows = []
    for exp in [e for e in ex if e != "E9"]:
        seeds = all_seeds(cfg, exp, include_extra=True)
        src = load_set(exp, "kvasir_test", seeds)
        for s in EXTERNAL_SETS:
            tgt = load_set(exp, s, seeds)
            common = sorted(set(src) & set(tgt))
            if not common:
                continue
            Ds = np.stack([src[x].dice.values for x in common])
            Dt = np.stack([tgt[x].dice.values for x in common])
            cs, ct = src[common[0]].cluster.values, tgt[common[0]].cluster.values
            _, _, bs = cluster_bootstrap_ci(Ds, cs, sc["bootstrap"], seed=0, return_samples=True)
            _, _, bt = cluster_bootstrap_ci(Dt, ct, sc["bootstrap"], seed=1, return_samples=True)
            g = bs - bt  # hai tập độc lập
            gap_rows.append({"exp": exp, "target": s, "dice_source": float(Ds.mean()),
                             "dice_target": float(Dt.mean()), "gap": float(Ds.mean() - Dt.mean()),
                             "ci_low": float(np.quantile(g, 0.025)), "ci_high": float(np.quantile(g, 0.975)),
                             "n_seeds": len(common)})
    gap = pd.DataFrame(gap_rows)
    ensure_dir("reports/stats")
    gap.to_csv(P("reports/stats/rq3_domain_gap.csv"), index=False)
    if len(gap):
        print(gap.round(4).to_string(index=False))

    color_rows = []
    for a, b in [("E5", "E3"), ("E5", "E4"), ("E4", "E3"), ("E6", "E3"), ("E7", "E3")]:
        for s in EXTERNAL_SETS + [POOLED]:
            r = compare(a, b, s, "dice", cfg)
            if r:
                color_rows.append({"comparison": f"{a} - {b}",
                                   "role": "H3b (xem planned cho p Holm)" if a == "E5" and b in ("E3", "E4")
                                   else "mô tả", **r})
    col = pd.DataFrame(color_rows)
    col.to_csv(P("reports/stats/rq3_color.csv"), index=False)
    if len(col):
        print(col[["comparison", "set", "estimate", "ci_low", "ci_high", "p_value"]].round(4).to_string(index=False))
    print("Đã ghi reports/stats/rq3_domain_gap.csv, reports/stats/rq3_color.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["planned", "table", "rq3"])
    ap.add_argument("--sets", nargs="+", default=["kvasir_val"])
    ap.add_argument("--suffix", default="", help="vd. _sens để dùng cụm của phân tích độ nhạy")
    args = ap.parse_args(argv)
    cfg = base_cfg()
    if args.cmd == "planned":
        cmd_planned(args.sets, cfg, args.suffix)
    elif args.cmd == "table":
        cmd_table(args.sets, cfg)
    else:
        cmd_rq3(cfg)


if __name__ == "__main__":
    main()
