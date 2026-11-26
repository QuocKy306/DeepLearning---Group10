"""Tổng hợp A/B/C, vẽ hình, chọn mô hình theo validation và đánh giá test lần cuối."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from .data import gaussian_2d
from .metrics import regression_metrics
from .model import MLP
from .plots import COLORS, LABELS, grid
from .train import predict

SPLITS = ("train", "val", "test")


def _err_plot(ax, summ, metric="mse", logx=False):
    for s in SPLITS:
        ax.errorbar(summ["value"], summ[f"{s}_{metric}_mean"], yerr=summ[f"{s}_{metric}_std"], marker="o",
                    ms=4, capsize=3, color=COLORS[s], label=LABELS[s])
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_ylabel(f"{metric.upper()} (trung bình ± độ lệch chuẩn, 3 seed)")
    ax.legend(fontsize=8)


def _curves(ax, logs: Path, tags_labels, col="val_mse"):
    cmap = plt.get_cmap("viridis")
    for i, (tag, lab) in enumerate(tags_labels):
        h = pd.read_csv(logs / f"{tag}.csv")
        ax.plot(h["epoch"], h[col], lw=1.2, color=cmap(i / max(1, len(tags_labels) - 1)), label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8, ncol=2)


def table(summ, value_name, extra=()):
    cols = {"value": value_name}
    t = pd.DataFrame({value_name: summ["value"].astype(int)})
    t["Số tham số"] = summ["n_params_mean"].astype(int)
    for s in SPLITS:
        t[f"{LABELS[s]} MSE"] = [f"{m:.2e} ± {d:.1e}" for m, d in zip(summ[f"{s}_mse_mean"], summ[f"{s}_mse_std"])]
    t["Test RMSE"] = [f"{m:.2e}" for m in summ["test_rmse_mean"]]
    t["Test MAE"] = [f"{m:.2e}" for m in summ["test_mae_mean"]]
    for e in extra:
        if e == "time":
            t["Thời gian (s)"] = [f"{m:.1f} ± {d:.1f}" for m, d in zip(summ["train_time_s_mean"], summ["train_time_s_std"])]
            t["ms/epoch"] = [f"{m:.1f}" for m in summ["time_per_epoch_ms_mean"]]
            t["Epoch tốt nhất"] = [f"{m:.0f}" for m in summ["best_epoch_mean"]]
    return t


def analyze(splits, results: Path):
    fig_dir, tab_dir, logs = results / "figures", results / "tables", results / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(results / "summary.csv")
    runs = pd.read_csv(results / "runs.csv")
    S = {g: summ[summ.group == g].sort_values("value") for g in summ.group.unique()}

    # ---------------- Nhóm A ----------------
    if "A" in S:
        a = S["A"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        _err_plot(axes[0], a); axes[0].set_xlabel("Số mẫu huấn luyện A")
        axes[0].set_title("A – MSE theo số mẫu huấn luyện")
        _err_plot(axes[1], a, "mae"); axes[1].set_xlabel("Số mẫu huấn luyện A")
        axes[1].set_title("A – MAE theo số mẫu huấn luyện")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_expA_error.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        _curves(ax, logs, [(f"A_N{n}_s11", f"A = {n}") for n in (200, 600, 1000, 1400, 2000)])
        ax.set_ylabel("Validation MSE"); ax.set_title("A – Đường hội tụ validation (seed 11)")
        fig.savefig(fig_dir / "fig_expA_curves.png"); plt.close(fig)
        table(a, "A (số mẫu)").to_csv(tab_dir / "table_A.csv", index=False)
        slope, icpt = np.polyfit(np.log10(a["value"]), np.log10(a["test_mse_mean"]), 1)
        fits = {"A_fixed_epochs": {"slope": float(slope), "intercept": float(icpt)}}
        if "A_steps" in S:
            ast = S["A_steps"]
            table(ast, "A (số mẫu)").to_csv(tab_dir / "table_A_steps.csv", index=False)
            fig, ax = plt.subplots(figsize=(7.5, 4.3))
            for (df, lab, ls) in ((a, "300 epoch cố định (nhóm A)", "-"),
                                  (ast, "cùng số bước ≈ 9 600 (A_steps)", "--")):
                for sp in ("train", "test"):
                    ax.errorbar(df["value"], df[f"{sp}_mse_mean"], yerr=df[f"{sp}_mse_std"], ls=ls, marker="o",
                                ms=4, capsize=3, color=COLORS[sp], label=f"{LABELS[sp]} – {lab}")
            ax.set_yscale("log"); ax.set(xlabel="Số mẫu huấn luyện A", ylabel="MSE (trung bình ± độ lệch chuẩn)",
                                         title="Tách hiệu ứng số mẫu khỏi số bước cập nhật")
            ax.legend(fontsize=7.5)
            fig.savefig(fig_dir / "fig_expA_steps.png"); plt.close(fig)
            s2, i2 = np.polyfit(np.log10(ast["value"]), np.log10(ast["test_mse_mean"]), 1)
            fits["A_fixed_steps"] = {"slope": float(s2), "intercept": float(i2)}
        (tab_dir / "powerlaw_fit.json").write_text(json.dumps(fits, indent=2))

    # ---------------- Nhóm B ----------------
    if "B" in S:
        b = S["B"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        _err_plot(axes[0], b, logx=True); axes[0].set_xlabel("Số nơ-ron tầng ẩn B (thang log₂)")
        axes[0].set_title("B – MSE theo độ rộng (1 tầng ẩn, A = 2000)")
        gap = b["test_mse_mean"] / b["train_mse_mean"]
        axes[1].plot(b["value"], gap, "o-", color="#9467bd")
        axes[1].axhline(1, ls="--", color="gray", lw=1)
        axes[1].set_xscale("log", base=2)
        axes[1].set(xlabel="Số nơ-ron tầng ẩn B (thang log₂)", ylabel="Test MSE / Train MSE",
                    title="B – Tỷ lệ sai số test/train")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_expB_error.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        _curves(ax, logs, [(f"B_W{w}_s11", f"B = {w}") for w in (1, 2, 4, 8, 10, 32, 64)])
        ax.set_ylabel("Validation MSE"); ax.set_title("B – Đường hội tụ validation (seed 11)")
        fig.savefig(fig_dir / "fig_expB_curves.png"); plt.close(fig)
        table(b, "B (nơ-ron)").to_csv(tab_dir / "table_B.csv", index=False)
    if "B_small" in S:
        bs = S["B_small"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        _err_plot(axes[0], bs, logx=True); axes[0].set_xlabel("Số nơ-ron tầng ẩn B (thang log₂)")
        axes[0].set_title(f"B bổ sung – ít dữ liệu (A = {cfg.B_SMALL_N})")
        gap = bs["test_mse_mean"] / bs["train_mse_mean"]
        axes[1].plot(bs["value"], gap, "o-", color="#9467bd")
        axes[1].axhline(1, ls="--", color="gray", lw=1)
        axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
        axes[1].set(xlabel="Số nơ-ron tầng ẩn B (thang log₂)", ylabel="Test MSE / Train MSE",
                    title="Khoảng cách tổng quát hóa khi ít dữ liệu")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_expB_small.png"); plt.close(fig)
        table(bs, "B (nơ-ron)").to_csv(tab_dir / "table_B_small.csv", index=False)

    # ---------------- Nhóm C ----------------
    if "C" in S:
        c = S["C"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        _err_plot(axes[0], c); axes[0].set_xlabel("Số tầng ẩn C (32 nơ-ron/tầng)")
        axes[0].set_title("C – MSE theo độ sâu (A = 2000)")
        ax = axes[1]
        ax.errorbar(c["value"], c["train_time_s_mean"], yerr=c["train_time_s_std"], marker="o", capsize=3,
                    color="#d62728", label="Tổng thời gian huấn luyện (s)")
        ax.set(xlabel="Số tầng ẩn C", ylabel="Thời gian huấn luyện (s)", title="C – Chi phí huấn luyện")
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(c["value"], c["time_per_epoch_ms_mean"], "s--", color="#8c564b", label="ms / epoch")
        ax2.set_ylabel("Thời gian mỗi epoch (ms)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_expC_error_time.png"); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        tl = [(f"C_D{d}_s11", f"C = {d}") for d in cfg.C_VALUES]
        _curves(axes[0], logs, tl); axes[0].set_ylabel("Validation MSE")
        axes[0].set_title("C – Đường hội tụ validation (seed 11)")
        _curves(axes[1], logs, tl, col="grad_norm_first_layer")
        axes[1].set_ylabel("‖∂L/∂W₁‖ trung bình mỗi epoch")
        axes[1].set_title("C – Chuẩn gradient tầng đầu tiên")
        fig.tight_layout(); fig.savefig(fig_dir / "fig_expC_curves.png"); plt.close(fig)
        table(c, "C (tầng ẩn)", extra=("time",)).to_csv(tab_dir / "table_C.csv", index=False)
        if "C_full" in S:
            cf = S["C_full"]
            table(cf, "C (tầng ẩn)", extra=("time",)).to_csv(tab_dir / "table_C_full.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            for df, lab, ls in ((c, "early stopping (C)", "-"), (cf, "không early stopping (C_full)", "--")):
                axes[0].errorbar(df["value"], df["val_mse_mean"], yerr=df["val_mse_std"], ls=ls, marker="o",
                                 capsize=3, label=f"Validation – {lab}")
                axes[0].errorbar(df["value"], df["test_mse_mean"], yerr=df["test_mse_std"], ls=ls, marker="^",
                                 capsize=3, label=f"Test – {lab}")
            axes[0].set_yscale("log")
            axes[0].set(xlabel="Số tầng ẩn C", ylabel="MSE (trung bình ± độ lệch chuẩn)",
                        title="C – Ảnh hưởng của early stopping")
            axes[0].legend(fontsize=7.5)
            rr = runs[runs.group.isin(["C", "C_full"])]
            for g, mk, col in (("C", "o", "#1f77b4"), ("C_full", "^", "#d62728")):
                q = rr[rr.group == g]
                axes[1].scatter(q["value"] + (0.12 if g == "C_full" else -0.12), q["test_mse"], marker=mk,
                                color=col, label="early stopping" if g == "C" else "không early stopping")
            axes[1].set_yscale("log")
            axes[1].set(xlabel="Số tầng ẩn C", ylabel="Test MSE từng seed",
                        title="C – Kết quả từng seed (3 seed/cấu hình)")
            axes[1].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(fig_dir / "fig_expC_full.png"); plt.close(fig)

    # ---------------- Chọn mô hình theo VALIDATION ----------------
    cand = runs[(runs.n_train == 2000) & runs.group.isin(["A", "B", "C", "C_full"])].copy()
    cand["early_stopping"] = cand["patience"] <= cand["max_epochs"]
    cand["hidden"] = cand["hidden"] + np.where(cand["early_stopping"], "", " noES")
    cand = cand.drop_duplicates(subset=["hidden", "seed"])                  # A2000 ≡ C2 ≡ baseline
    agg = cand.groupby("hidden").agg(val_mse_mean=("val_mse", "mean"), val_mse_std=("val_mse", "std"),
                                     n_params=("n_params", "first"), n_seeds=("seed", "count"))
    agg = agg.sort_values("val_mse_mean")
    agg.to_csv(tab_dir / "selection_ranking.csv")
    best_hidden = agg.index[0]
    best_run = cand[cand.hidden == best_hidden].sort_values("val_mse").iloc[0]
    ck = torch.load(results / "checkpoints" / f"{best_run.tag}.pt", map_location="cpu", weights_only=False)
    model = MLP(hidden=tuple(ck["config"]["hidden"]), activation=ck["config"]["activation"])
    model.load_state_dict(ck["state_dict"])
    (results / "models").mkdir(exist_ok=True)
    shutil.copy(results / "checkpoints" / f"{best_run.tag}.pt", results / "models" / "best_model.pt")

    final = {}
    for s in SPLITS:
        x, y = splits[s]
        final[s] = regression_metrics(y, predict(model, x))
    a_, b_, g = grid()
    zt = gaussian_2d(g)
    zp = predict(model, g)
    final["dense_grid_121x241"] = regression_metrics(zt, zp)
    sel = {"selection_rule": "MSE validation trung bình 3 seed nhỏ nhất (test không tham gia)",
           "best_hidden": json.loads(best_hidden.replace(" noES", "")),
           "early_stopping": not best_hidden.endswith("noES"), "best_tag": best_run.tag, "seed": int(best_run.seed),
           "n_params": int(best_run.n_params), "best_epoch": int(ck["best_epoch"]),
           "val_mse_mean_3seeds": float(agg.iloc[0].val_mse_mean),
           "val_mse_std_3seeds": float(agg.iloc[0].val_mse_std), "metrics": final,
           "ranking_top5": [{"hidden": h, "val_mse_mean": float(r.val_mse_mean),
                             "val_mse_std": float(r.val_mse_std), "n_params": int(r.n_params)}
                            for h, r in agg.head(5).iterrows()]}
    (results / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    xt, yt = splits["test"]
    pt = predict(model, xt)
    pd.DataFrame({"x1": xt[:, 0], "x2": xt[:, 1], "y_true": yt, "y_pred": pt, "abs_err": np.abs(pt - yt)}) \
        .to_csv(results / "test_predictions.csv", index=False)

    # Hình mô hình cuối
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for s, mk in (("val", "o"), ("test", "^")):
        x, y = splits[s]
        axes[0].scatter(y, predict(model, x), s=14 if s == "test" else 6, marker=mk, alpha=.7,
                        color=COLORS[s], label=f"{LABELS[s]} (MSE = {final[s]['mse']:.2e})")
    lim = [0, max(splits["val"][1].max(), splits["test"][1].max()) * 1.05]
    axes[0].plot(lim, lim, "k--", lw=1, label="ŷ = y")
    axes[0].set(xlabel="y thật", ylabel="ŷ dự đoán", title="y thật – ŷ dự đoán (mô hình được chọn)")
    axes[0].legend(fontsize=8)
    axes[1].hist(pt - yt, bins=25, color=COLORS["test"], alpha=.8)
    axes[1].set(xlabel="Phần dư ŷ − y trên tập test", ylabel="Số mẫu", title="Phân bố phần dư (test)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_final_pred_vs_true.png"); plt.close(fig)

    fig = plt.figure(figsize=(16, 4.6))
    for i, (zz, title) in enumerate(((zt, "Hàm thật f(x)"), (zp, "Bề mặt xấp xỉ ŷ(x)"))):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(a_, b_, zz.reshape(a_.shape), cmap="viridis", linewidth=0)
        ax.set(xlabel="x1", ylabel="x2", zlabel="y", title=title); ax.view_init(25, -60)
    ax = fig.add_subplot(1, 3, 3)
    cs = ax.contourf(a_, b_, np.abs(zp - zt).reshape(a_.shape), levels=20, cmap="magma")
    fig.colorbar(cs, ax=ax, label="|ŷ − f|", format="%.1e")
    ax.set(xlabel="x1", ylabel="x2", title=f"Sai số tuyệt đối trên lưới 121×241 (max = {final['dense_grid_121x241']['maxae']:.1e})")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_final_surface.png"); plt.close(fig)

    h = pd.read_csv(logs / f"{best_run.tag}.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(h.epoch, h.train_mse, label="Train MSE", color=COLORS["train"])
    ax.plot(h.epoch, h.val_mse, label="Validation MSE", color=COLORS["val"])
    ax.axvline(ck["best_epoch"], ls="--", color="gray", lw=1, label=f"checkpoint (epoch {ck['best_epoch']})")
    ax.set_yscale("log"); ax.set(xlabel="Epoch", ylabel="MSE", title=f"Đường học của mô hình được chọn ({best_run.tag})")
    ax2 = ax.twinx(); ax2.grid(False); ax2.plot(h.epoch, h.lr, color="gray", alpha=.5, lw=1)
    ax2.set_ylabel("learning rate (đường xám)")
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "fig_final_loss.png"); plt.close(fig)
    return sel
