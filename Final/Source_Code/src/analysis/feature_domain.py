"""Mức tách miền của biểu diễn (khám phá, mục 4.7): đặc trưng nút cổ chai của bộ mã hóa (gộp trung bình toàn cục)
trên Kvasir-test và ba bộ ngoài; bộ phân loại tuyến tính dự đoán bộ dữ liệu, kiểm định chéo theo NHÓM CỤM
(tránh rò rỉ giữa khung cùng video); độ chính xác cân bằng càng gần 0,25 càng bất biến miền. Kèm t-SNE và tốc độ suy luận CPU.
Chạy: python -m src.analysis.feature_domain --exps P1 P2 P3 P5
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..inference import load_checkpoint
from ..utils import EXTERNAL_SETS, P, base_cfg, ensure_dir, exp_cfg, read_image, run_dir, save_json, set_rows

SETS = ["kvasir_test"] + EXTERNAL_SETS


@torch.no_grad()
def features(model, rows, size):
    out = []
    for p in rows.image_path:
        x = cv2.resize(read_image(P(p)), (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(x.astype(np.float32) / 255).permute(2, 0, 1)[None]
        f = model.net.encoder((x - model.mean) / model.std)[-1]
        out.append(f.mean((2, 3))[0].numpy())
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["P1", "P2", "P3", "P5"])
    args = ap.parse_args(argv)
    cfg = base_cfg()
    torch.set_num_threads(1)
    rows = pd.concat([set_rows(s).assign(ds=s) for s in SETS], ignore_index=True)
    y = rows.ds.map({s: i for i, s in enumerate(SETS)}).values
    groups = (rows.ds + "|" + rows.cluster.astype(str)).values
    res, emb = {}, {}
    for e in args.exps:
        m = load_checkpoint(run_dir(e, 42) / "best.pt", exp_cfg(e)).cpu()
        X = features(m, rows, cfg["image_size"])
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
        x1 = torch.rand(1, 3, cfg["image_size"], cfg["image_size"])
        for _ in range(3):
            m(x1)
        t = time.time()
        for _ in range(20):
            m(x1)
        fps = 20 / (time.time() - t)
        s = pd.read_json(run_dir(e, 42) / "summary.json", typ="series")
        res[e] = dict(domain_bal_acc=float(balanced_accuracy_score(y, pred)), feat_dim=int(X.shape[1]),
                      cpu_fps_1thread=float(fps), train_minutes_cpu=float(s["seconds_per_epoch"] * s["epochs_run"] / 60))
        emb[e] = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(StandardScaler().fit_transform(X))
        print(e, res[e], flush=True)
    out = ensure_dir("reports/feature_domain")
    save_json(res, out / "summary.json")
    np.savez(out / "tsne.npz", ds=rows.ds.values, **emb)


if __name__ == "__main__":
    main()
