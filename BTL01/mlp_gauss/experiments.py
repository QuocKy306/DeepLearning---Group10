"""Định nghĩa và chạy các nhóm thực nghiệm A, B, C (mỗi nhóm chỉ thay đổi một yếu tố)."""
from pathlib import Path
import json
import pandas as pd

from . import config as cfg
from .config import TrainConfig
from .train import train_model

BASE = TrainConfig()   # baseline: 2 tầng ẩn x 32, tanh, Adam 5e-3, batch 64


def steps_factor(n: int) -> float:
    """Tỷ lệ số epoch cần thêm để n mẫu có cùng số bước cập nhật với REF_STEPS_N mẫu (batch cố định)."""
    import math
    return math.ceil(cfg.REF_STEPS_N / BASE.batch_size) / math.ceil(n / BASE.batch_size)


def configs_for(group: str, seeds):
    """Sinh danh sách (tag, value, TrainConfig) cho một nhóm thực nghiệm."""
    out = []
    for s in seeds:
        if group == "A":        # thay đổi số mẫu train, giữ nguyên kiến trúc baseline
            for n in cfg.A_VALUES:
                out.append((f"A_N{n}_s{s}", n, TrainConfig(hidden=BASE.hidden, n_train=n, seed=s)))
        elif group == "B":      # MLP 1 tầng ẩn, thay đổi số nơ-ron, N = 2000
            for b in cfg.B_VALUES:
                out.append((f"B_W{b}_s{s}", b, TrainConfig(hidden=(b,), n_train=2000, seed=s)))
        elif group == "A_steps":  # bổ sung: như A nhưng tăng số epoch để tổng số bước cập nhật ≈ A = 2000
            for n in cfg.A_STEPS_VALUES:
                k = steps_factor(n)
                out.append((f"As_N{n}_s{s}", n, TrainConfig(hidden=BASE.hidden, n_train=n, seed=s,
                                                             max_epochs=round(BASE.max_epochs * k),
                                                             patience=round(BASE.patience * k))))
        elif group == "B_small":  # bổ sung: 1 tầng ẩn, chỉ 200 mẫu, đủ số bước cập nhật -> quan sát quá khớp
            k = steps_factor(cfg.B_SMALL_N)
            for b in cfg.B_SMALL_VALUES:
                out.append((f"Bs_W{b}_s{s}", b, TrainConfig(hidden=(b,), n_train=cfg.B_SMALL_N, seed=s,
                                                            max_epochs=round(BASE.max_epochs * k),
                                                            patience=round(BASE.patience * k))))
        elif group == "C_full":   # bổ sung: như C nhưng tắt early stopping (vẫn checkpoint theo val)
            for d in cfg.C_VALUES:
                out.append((f"Cf_D{d}_s{s}", d, TrainConfig(hidden=(cfg.C_WIDTH,) * d, n_train=2000, seed=s,
                                                             patience=BASE.max_epochs + 1)))
        elif group == "C":      # thay đổi số tầng ẩn (32 nơ-ron/tầng), N = 2000
            for d in cfg.C_VALUES:
                out.append((f"C_D{d}_s{s}", d, TrainConfig(hidden=(cfg.C_WIDTH,) * d, n_train=2000, seed=s)))
        else:
            raise ValueError(group)
    return out


def run_group(group: str, splits: dict, results: Path, seeds=cfg.SEEDS, max_epochs=None):
    """Chạy tất cả cấu hình của một nhóm; ghi từng lần chạy (CSV) – có thể chạy tiếp nếu bị dừng giữa chừng."""
    runs_csv = results / "runs.csv"
    done = set(pd.read_csv(runs_csv)["tag"]) if runs_csv.exists() else set()
    for tag, value, c in configs_for(group, seeds):
        if tag in done:
            continue
        if max_epochs:
            c.max_epochs = max_epochs
        _, rec, hist = train_model(c, splits, ckpt_path=results / "checkpoints" / f"{tag}.pt")
        rec = {"tag": tag, "group": group, "value": value, **rec}
        rec["hidden"] = json.dumps(rec["hidden"])
        (results / "logs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist).to_csv(results / "logs" / f"{tag}.csv", index=False)
        pd.DataFrame([rec]).to_csv(runs_csv, mode="a", header=not runs_csv.exists(), index=False)
        print(f"{tag:16s} val_mse={rec['val_mse']:.3e} test_mse={rec['test_mse']:.3e} "
              f"epochs={rec['epochs_run']:4d} t={rec['train_time_s']:.1f}s", flush=True)


def summarize(results: Path) -> pd.DataFrame:
    """Gộp theo (nhóm, giá trị): trung bình ± độ lệch chuẩn qua các seed."""
    df = pd.read_csv(results / "runs.csv")
    metrics = [f"{s}_{m}" for s in ("train", "val", "test") for m in ("mse", "rmse", "mae", "maxae", "r2")]
    metrics += ["best_epoch", "epochs_run", "train_time_s", "time_per_epoch_ms", "n_params"]
    g = df.groupby(["group", "value"])[metrics]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    summ = summ.reset_index().sort_values(["group", "value"])
    summ.to_csv(results / "summary.csv", index=False)
    return summ
