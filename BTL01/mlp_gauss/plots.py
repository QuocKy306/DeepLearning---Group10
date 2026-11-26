"""Hình khảo sát dữ liệu và sơ đồ kiến trúc."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from . import config as cfg
from .data import gaussian_2d

COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3, "savefig.dpi": 160,
                     "savefig.bbox": "tight"})


def grid(n1=121, n2=241):
    a, b = np.meshgrid(np.linspace(*cfg.X1_RANGE, n1), np.linspace(*cfg.X2_RANGE, n2))
    return a, b, np.column_stack([a.ravel(), b.ravel()])


def data_figures(splits, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # 1) scatter 3D của 3 tập
    fig = plt.figure(figsize=(14, 4.4))
    for i, k in enumerate(("train", "val", "test")):
        x, y = splits[k]
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], y, s=3 if k == "train" else 8, c=COLORS[k], alpha=.7)
        ax.set(xlabel="x1", ylabel="x2", zlabel="y = f(x)", title=f"{LABELS[k]} (n = {len(y)})")
        ax.view_init(25, -60)
    fig.savefig(out / "fig_data_scatter3d.png"); plt.close(fig)
    # 2) bề mặt hàm thật + đường đồng mức với vị trí mẫu
    a, b, g = grid()
    z = gaussian_2d(g).reshape(a.shape)
    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(a, b, z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set(xlabel="x1", ylabel="x2", zlabel="f(x)", title="Bề mặt hàm Gaussian thật")
    ax.view_init(25, -60)
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(a, b, z, levels=20, cmap="viridis")
    x, _ = splits["train"]
    ax2.scatter(x[:, 0], x[:, 1], s=2, c="white", alpha=.5, label="mẫu train")
    xt, _ = splits["test"]
    ax2.scatter(xt[:, 0], xt[:, 1], s=12, c="red", marker="x", label="mẫu test")
    fig.colorbar(cs, ax=ax2, label="f(x)")
    ax2.set(xlabel="x1", ylabel="x2", title="Đường đồng mức và vị trí mẫu")
    ax2.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "fig_true_surface.png"); plt.close(fig)
    # 3) phân bố x1, x2, y
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for j, (name, f) in enumerate((("x1", lambda s: s[0][:, 0]), ("x2", lambda s: s[0][:, 1]),
                                   ("y", lambda s: s[1]))):
        for k in ("train", "val", "test"):
            axes[j].hist(f(splits[k]), bins=30, density=True, histtype="step", lw=1.6, color=COLORS[k],
                         label=LABELS[k])
        axes[j].set(xlabel=name, ylabel="mật độ", title=f"Phân bố {name}")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_data_hist.png"); plt.close(fig)
    architecture_figure(out / "fig_mlp_architecture.png")


def architecture_figure(path: Path, widths=(32, 32)):
    """Sơ đồ MLP: 2 đầu vào -> các tầng ẩn tanh -> 1 đầu ra tuyến tính (vẽ 5 nút đại diện mỗi tầng ẩn)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_aspect("equal"); ax.axis("off")
    xs = [1.0] + [3.6 + 2.6 * i for i in range(len(widths))] + [3.6 + 2.6 * len(widths)]
    cy, r = 3.2, 0.26

    def ys(n, gap=0.85, c=cy):
        return [c + gap * ((n - 1) / 2 - k) for k in range(n)]
    hid = ys(4) + [cy - 0.85 * 3.2]
    mid = (hid[0] + hid[-1]) / 2                       # căn giữa đầu vào/đầu ra theo tầng ẩn
    nodes = [ys(2, 1.2, mid)] + [hid for _ in widths] + [ys(1, c=mid)]
    # ys(4)+1 nút cuối tách xa để chèn dấu "⋮" (đại diện cho 32 nút)
    for li in range(len(nodes) - 1):
        for y1 in nodes[li]:
            for y2 in nodes[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [y1, y2], color="#aab4c0", lw=.5, zorder=1)
    cols = ["#4c78a8"] + ["#f58518"] * len(widths) + ["#54a24b"]
    for li, ylist in enumerate(nodes):
        for y in ylist:
            ax.add_patch(Circle((xs[li], y), r, color=cols[li], zorder=3))
        if 0 < li < len(nodes) - 1:
            ax.text(xs[li], (ylist[3] + ylist[4]) / 2, "⋮", ha="center", va="center", fontsize=16, zorder=4)
    ax.text(xs[0] - 0.45, nodes[0][0], "x1", ha="right", va="center", fontsize=11)
    ax.text(xs[0] - 0.45, nodes[0][1], "x2", ha="right", va="center", fontsize=11)
    ax.text(xs[-1] + 0.45, mid, "ŷ", ha="left", va="center", fontsize=12)
    lo = min(min(n) for n in nodes) - 0.9
    ax.text(xs[0], lo, "Đầu vào (2)\nz-score", ha="center", va="top", fontsize=9)
    for i, w in enumerate(widths):
        ax.text(xs[i + 1], lo, f"Tầng ẩn {i + 1}\n{w} nơ-ron\nLinear + tanh", ha="center", va="top", fontsize=9)
    ax.text(xs[-1], lo, "Đầu ra (1)\nLinear", ha="center", va="top", fontsize=9)
    ax.set_xlim(0, xs[-1] + 1.0); ax.set_ylim(lo - 1.3, cy + 2.0)
    n_par = sum((a + 1) * b for a, b in zip((2, *widths), (*widths, 1)))
    ax.set_title(f"Kiến trúc MLP baseline: 2 → {' → '.join(map(str, widths))} → 1 ({n_par:,} tham số)".replace(",", " "),
                 fontsize=11)
    fig.savefig(path); plt.close(fig)
