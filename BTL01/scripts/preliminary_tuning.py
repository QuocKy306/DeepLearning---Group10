"""Khảo sát sơ bộ (chỉ dùng VALIDATION) để chốt cấu hình baseline trước khi chạy A/B/C.

So sánh: khởi tạo trọng số (Xavier-tanh vs mặc định PyTorch) × chuẩn hóa đầu vào (min-max về [-1,1] vs z-score),
và learning rate ban đầu của Adam + cosine. Kiến trúc 2-32-32-1, 200 epoch, 3 seed.
Chạy:  python scripts/preliminary_tuning.py   -> results/tables/preliminary_tuning.csv
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from mlp_gauss import config as cfg
from mlp_gauss.config import TrainConfig
from mlp_gauss.data import ensure_data
from mlp_gauss import train as T

torch.set_num_threads(1)
splits = ensure_data(ROOT / "data")
orig_set_stats = T.MLP.set_input_stats


def run(init, norm, lr, seed):
    def set_stats(self, x):
        if norm == "minmax":      # [-1, 1] theo biên miền
            self.in_mean.copy_(torch.tensor([0.0, 0.0]))
            self.in_std.copy_(torch.tensor([2.0, 4.0]))
        else:
            orig_set_stats(self, x)
        if init == "xavier_tanh":
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                    nn.init.zeros_(m.bias)
    T.MLP.set_input_stats = set_stats
    _, rec, _ = T.train_model(TrainConfig(lr=lr, seed=seed, max_epochs=200, patience=201), splits)
    T.MLP.set_input_stats = orig_set_stats
    return rec["val_mse"]


rows = []
grid = [("xavier_tanh", "minmax", 1e-2), ("xavier_tanh", "zscore", 1e-2), ("default", "minmax", 1e-2),
        ("default", "zscore", 1e-2), ("default", "zscore", 5e-3), ("default", "zscore", 3e-3)]
for init, norm, lr in grid:
    vals = [run(init, norm, lr, s) for s in cfg.SEEDS]
    rows.append({"init": init, "input_norm": norm, "lr": lr, "val_mse_mean": sum(vals) / 3,
                 "val_mse_std": pd.Series(vals).std(), **{f"val_mse_s{s}": v for s, v in zip(cfg.SEEDS, vals)}})
    print(rows[-1], flush=True)
out = ROOT / "results" / "tables"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "preliminary_tuning.csv", index=False)
