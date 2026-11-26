"""Sinh, lưu, nạp và thống kê dữ liệu Gaussian 2D."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from . import config as cfg


def gaussian_2d(x: np.ndarray, mu: np.ndarray = cfg.MU, cov: np.ndarray = cfg.COV) -> np.ndarray:
    """f(x) = 1 / (2π sqrt(det C)) * exp(-1/2 (x-μ)^T C^{-1} (x-μ)), tính vector hóa cho n điểm."""
    d = np.asarray(x, dtype=np.float64) - mu
    inv = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", d, inv, d)          # (x-μ)^T C^{-1} (x-μ) cho từng hàng
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))


def sample_uniform(n: int, rng: np.random.Generator):
    """Lấy mẫu đều x1 ∈ [-2, 2], x2 ∈ [-4, 4] và tính nhãn y = f(x)."""
    lo = np.array([cfg.X1_RANGE[0], cfg.X2_RANGE[0]])
    hi = np.array([cfg.X1_RANGE[1], cfg.X2_RANGE[1]])
    x = rng.uniform(lo, hi, size=(n, 2))
    return x, gaussian_2d(x)


def generate_splits(seed: int = cfg.DATA_SEED):
    """Ba tập độc lập sinh từ một bộ sinh số ngẫu nhiên có seed cố định (thứ tự: train, val, test)."""
    rng = np.random.default_rng(seed)
    return {
        "train": sample_uniform(cfg.N_TRAIN_POOL, rng),
        "val": sample_uniform(cfg.N_VAL, rng),
        "test": sample_uniform(cfg.N_TEST, rng),
    }


def save_splits(splits, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in splits.items():
        pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}).to_csv(out_dir / f"{name}.csv", index=False)


def load_splits(data_dir: Path):
    out = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(data_dir / f"{name}.csv")
        out[name] = (df[["x1", "x2"]].to_numpy(), df["y"].to_numpy())
    return out


def ensure_data(data_dir: Path):
    """Sinh dữ liệu nếu chưa có, ngược lại nạp lại từ CSV để mọi cấu hình dùng cùng một dữ liệu."""
    if not (data_dir / "train.csv").exists():
        save_splits(generate_splits(), data_dir)
    return load_splits(data_dir)


def to_tensors(x: np.ndarray, y: np.ndarray):
    """Chuyển numpy -> tensor float32; y có dạng (n, 1) để khớp với đầu ra của mạng."""
    return (torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1))


def make_loader(x, y, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    xt, yt = to_tensors(x, y)
    gen = torch.Generator().manual_seed(seed)          # cố định thứ tự mini-batch theo seed
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle, generator=gen)


def describe(splits) -> pd.DataFrame:
    rows = []
    for name, (x, y) in splits.items():
        for col, v in (("x1", x[:, 0]), ("x2", x[:, 1]), ("y", y)):
            rows.append({"split": name, "variable": col, "n": len(v), "mean": v.mean(), "std": v.std(),
                         "min": v.min(), "q25": np.quantile(v, .25), "median": np.median(v),
                         "q75": np.quantile(v, .75), "max": v.max()})
    return pd.DataFrame(rows)
