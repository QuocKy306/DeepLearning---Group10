"""Điểm vào duy nhất của BTL01.

Ví dụ:
    python main.py data                         # sinh dữ liệu, thống kê, hình dữ liệu
    python main.py run --groups A B C B_small    # chạy các nhóm thực nghiệm (3 seed mặc định)
    python main.py analyze                       # bảng tổng hợp, hình, chọn mô hình, đánh giá test
    python main.py all                           # toàn bộ quy trình
    python main.py all --quick                   # chạy thử nhanh (1 seed, 30 epoch) để kiểm tra
"""
import argparse
from pathlib import Path

import torch

from mlp_gauss import config as cfg
from mlp_gauss.data import ensure_data, describe
from mlp_gauss.experiments import run_group, summarize

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="BTL01 – MLP xấp xỉ Gaussian 2D (PyTorch)")
    p.add_argument("command", choices=["data", "run", "analyze", "all"])
    p.add_argument("--groups", nargs="+", default=["A", "B", "C", "A_steps", "B_small", "C_full"])
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--quick", action="store_true", help="1 seed, 30 epoch – chỉ để kiểm tra code")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    data_dir, results = Path(a.data_dir), Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    seeds, max_epochs = (a.seeds[:1], 30) if a.quick else (a.seeds, None)

    splits = ensure_data(data_dir)
    if a.command in ("data", "all"):
        from mlp_gauss import plots
        describe(splits).to_csv(results / "data_stats.csv", index=False)
        plots.data_figures(splits, results / "figures")
        print("Đã sinh dữ liệu:", data_dir)
    if a.command in ("run", "all"):
        for g in a.groups:
            print(f"===== Nhóm {g} =====", flush=True)
            run_group(g, splits, results, seeds=seeds, max_epochs=max_epochs)
    if a.command in ("analyze", "all"):
        from mlp_gauss import analysis
        summarize(results)
        analysis.analyze(splits, results)
        print("Đã ghi kết quả vào:", results)


if __name__ == "__main__":
    main()
