"""Quét các bộ dữ liệu, kiểm tra chất lượng và lập data/index.csv (mục 3.4).

Đầu ra:
  data/index.csv                 dataset,id,image_path,mask_path,height,width,mask_area_frac,n_components,md5
  data/hashes.npz                mã băm cảm nhận của mọi ảnh (dùng lại khi chia tập, dựng cụm)
  reports/data_check.json        thiếu mặt nạ, lệch kích thước, mặt nạ rỗng, mặt nạ không nhị phân, ảnh trùng MD5
  reports/cross_dataset_duplicates.csv   ảnh gần trùng GIỮA các bộ (vd. PolypGen/HyperKvasir với Kvasir)

Chạy: python -m src.prepare_data
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .hashing import hamming_matrix, hash_pair
from .utils import P, base_cfg, datasets_cfg, ensure_dir, md5_file, read_image, save_json

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_files(folder: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMG_EXT and not p.name.startswith("."):
            out[p.stem] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    cfg = base_cfg()
    size = cfg["hashing"]["size"]
    rows, report = [], {}
    hashes, hash_keys = [], []

    for name, d in datasets_cfg().items():
        img_dir = P(d["images"])
        if not img_dir.exists():
            print(f"Bỏ qua {name}: chưa có thư mục {img_dir}")
            continue
        imgs = list_files(img_dir)
        masks = list_files(P(d["masks"])) if d.get("masks") else {}
        rep = defaultdict(list)
        md5_seen = {}
        for stem, ip in tqdm(imgs.items(), desc=name):
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if img is None:
                rep["unreadable"].append(stem)
                continue
            h, w = img.shape[:2]
            md5 = md5_file(ip)
            if md5 in md5_seen:
                rep["exact_duplicates"].append([md5_seen[md5], stem])
            md5_seen.setdefault(md5, stem)
            mp, area, ncomp = None, np.nan, 0
            if d.get("masks"):
                mp = masks.get(stem)
                if mp is None:
                    rep["missing_mask"].append(stem)
                    continue
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m.shape != (h, w):
                    rep["size_mismatch"].append([stem, [h, w], list(m.shape)])
                    continue
                vals = np.unique(m)
                if len(vals) > 2:
                    rep["non_binary_mask"].append(stem)  # vẫn dùng, nhị phân hóa ở 128
                mb = (m >= 128).astype(np.uint8)
                area = float(mb.mean())
                ncomp = int(cv2.connectedComponents(mb)[0] - 1)
                if area == 0:
                    rep["empty_mask"].append(stem)
            rows.append({"dataset": name, "id": stem, "image_path": str(ip.relative_to(P("."))
                         if ip.is_relative_to(P(".")) else ip),
                         "mask_path": "" if mp is None else str(mp.relative_to(P("."))
                         if mp.is_relative_to(P(".")) else mp),
                         "height": h, "width": w, "mask_area_frac": area,
                         "n_components": ncomp, "md5": md5})
            hashes.append(hash_pair(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size, cfg["hashing"].get("center_crop", 0.6)))
            hash_keys.append(f"{name}/{stem}")
        orphan = sorted(set(masks) - set(imgs))
        if orphan:
            rep["mask_without_image"] = orphan
        report[name] = {"n_images": len(imgs), "n_kept": sum(r["dataset"] == name for r in rows),
                        **{k: v for k, v in rep.items()}}
        print(f"{name}: {report[name]['n_kept']}/{len(imgs)} ảnh hợp lệ; "
              + ", ".join(f"{k}={len(v)}" for k, v in rep.items()))

    if not rows:
        raise SystemExit("Không tìm thấy bộ dữ liệu nào. Kiểm tra configs/datasets.yaml.")
    df = pd.DataFrame(rows)
    ensure_dir("data")
    df.to_csv(P("data/index.csv"), index=False)
    H = np.stack(hashes)
    np.savez_compressed(P("data/hashes.npz"), keys=np.array(hash_keys), hashes=H)
    save_json(report, "reports/data_check.json")

    # Ảnh gần trùng giữa các bộ khác nhau: rò rỉ giữa nguồn và đích.
    k = size * size
    ds = np.array([key.split("/")[0] for key in hash_keys])
    dd, dp = hamming_matrix(H[:, :k]), hamming_matrix(H[:, k:])
    tau = cfg["hashing"]["tau"]
    close = (dd <= tau) & (dp <= tau) & (ds[:, None] != ds[None, :])
    ii, jj = np.where(np.triu(close, 1))
    cross = pd.DataFrame({"a": [hash_keys[i] for i in ii], "b": [hash_keys[j] for j in jj],
                          "hamming_max": np.maximum(dd, dp)[ii, jj]})
    cross.to_csv(P("reports/cross_dataset_duplicates.csv"), index=False)
    if len(cross):
        print(f"[CẢNH BÁO] {len(cross)} cặp ảnh gần trùng GIỮA các bộ dữ liệu. "
              "Xem reports/cross_dataset_duplicates.csv trước khi tiếp tục.")
    print(f"Đã ghi data/index.csv ({len(df)} ảnh) và reports/data_check.json")


if __name__ == "__main__":
    main()
