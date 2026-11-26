"""Tiện ích dùng chung: đường dẫn, cấu hình, seed, đọc ảnh, mã băm file.

Biến môi trường:
  POLYP_WORKDIR    thư mục chứa data/, splits/, runs/, reports/ (mặc định: gốc kho mã)
  POLYP_DATASETS   đường dẫn datasets.yaml thay thế
  POLYP_OVERRIDES  file yaml ghi đè một phần base.yaml (dùng cho kiểm thử nhanh)
  POLYP_NO_PRETRAINED=1  không tải trọng số ImageNet (kiểm thử ngoại tuyến)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("POLYP_WORKDIR", REPO)).resolve()

EXTERNAL_SETS = ["clinicdb", "colondb", "etis"]
KVASIR_SETS = ["kvasir_train", "kvasir_val", "kvasir_test"]
TEST_LOG = "runs/TEST_EVALUATED.txt"


def P(rel: str | Path) -> Path:
    """Đường dẫn tuyệt đối trong thư mục làm việc."""
    p = Path(rel)
    return p if p.is_absolute() else WORK / p


def ensure_dir(p: str | Path) -> Path:
    p = P(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def base_cfg() -> dict:
    cfg = load_yaml(REPO / "configs" / "base.yaml")
    ov = os.environ.get("POLYP_OVERRIDES")
    if ov:
        _deep_update(cfg, load_yaml(ov))
    return cfg


def datasets_cfg() -> dict:
    path = os.environ.get("POLYP_DATASETS", REPO / "configs" / "datasets.yaml")
    return load_yaml(path)["datasets"]


def experiments_cfg() -> dict:
    return load_yaml(REPO / "configs" / "experiments.yaml")


def exp_cfg(exp: str) -> dict:
    ex = experiments_cfg()
    if exp not in ex["experiments"]:
        raise KeyError(f"Không có thí nghiệm {exp} trong configs/experiments.yaml")
    cfg = dict(ex["experiments"][exp])
    cfg["name"] = exp
    cfg["appearance_params"] = ex["appearance_params"].get(cfg["appearance"], {})
    return cfg


def all_seeds(cfg: dict, exp: str, include_extra: bool = False) -> list[int]:
    seeds = list(cfg["seeds"])
    if include_extra:
        seeds += list(cfg.get("extra_seeds", {}).get(exp, []))
    return seeds


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_image(path: str | Path) -> np.ndarray:
    """Đọc ảnh RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Đọc mặt nạ, nhị phân hóa ở ngưỡng 128 (mặt nạ JPEG có nhiễu nén)."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m >= 128).astype(np.uint8)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_with_hash(df: pd.DataFrame, rel: str) -> Path:
    """Ghi CSV và file .sha256 kèm theo để phát hiện sửa đổi về sau."""
    path = P(rel)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    Path(str(path) + ".sha256").write_text(sha256_file(path))
    return path


def check_hash(rel: str) -> str:
    """Trả về mã băm hiện tại; báo lỗi nếu khác mã băm đã lưu."""
    path = P(rel)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {rel}")
    cur = sha256_file(path)
    saved = Path(str(path) + ".sha256")
    if saved.exists() and saved.read_text().strip() != cur:
        raise RuntimeError(f"{rel} đã bị sửa sau khi tạo (mã băm không khớp). Dừng lại.")
    return cur


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> dict:
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "git": git_hash()}
    try:
        import segmentation_models_pytorch as smp
        info["smp"] = smp.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_json(obj, rel: str | Path) -> Path:
    path = P(rel)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path


def load_json(rel: str | Path):
    with open(P(rel), encoding="utf-8") as f:
        return json.load(f)


def load_index() -> pd.DataFrame:
    path = P("data/index.csv")
    if not path.exists():
        raise FileNotFoundError("Chưa có data/index.csv. Chạy: python -m src.prepare_data")
    return pd.read_csv(path, dtype={"id": str})


def set_rows(set_name: str, cluster_suffix: str = "") -> pd.DataFrame:
    """Các dòng index của một tập: kvasir_train/val/test hoặc tên bộ ngoài, kèm cột cluster."""
    idx = load_index()
    if set_name.startswith("kvasir_"):
        check_hash("splits/kvasir_split.csv")
        split = pd.read_csv(P("splits/kvasir_split.csv"), dtype={"id": str})
        part = set_name.split("_", 1)[1]
        ids = split[split.split == part][["id", "cluster"]]
        rows = idx[idx.dataset == "kvasir"].merge(ids, on="id")
    else:
        rows = idx[idx.dataset == set_name].copy()
        cpath = P(f"splits/clusters_{set_name}{cluster_suffix}.csv")
        if cpath.exists():
            check_hash(f"splits/clusters_{set_name}{cluster_suffix}.csv")
            cl = pd.read_csv(cpath, dtype={"id": str, "cluster": str})
            rows = rows.merge(cl[["id", "cluster"]], on="id", how="left")
        else:
            rows["cluster"] = np.nan
        if rows["cluster"].isna().any():
            print(f"[CẢNH BÁO] {set_name}: thiếu cụm cho {rows.cluster.isna().sum()} ảnh; "
                  "dùng mỗi ảnh một cụm (khoảng tin cậy sẽ quá hẹp).")
            rows.loc[rows.cluster.isna(), "cluster"] = "img_" + rows.loc[rows.cluster.isna(), "id"]
    rows["cluster"] = rows["cluster"].astype(str)
    return rows.reset_index(drop=True)


def run_dir(exp: str, seed: int) -> Path:
    return P(f"runs/{exp}/seed{seed}")
