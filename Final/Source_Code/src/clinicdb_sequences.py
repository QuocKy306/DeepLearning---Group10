"""Ánh xạ khung -> chuỗi video của CVC-ClinicDB (mục 2.2 kế hoạch, 3.2 lý thuyết).

Mặc định: đọc configs/clinicdb_sequences.csv (điền tay theo Bảng 1 trang chính thức),
kiểm tra phủ đúng một lần mọi khung, ghi splits/clusters_clinicdb.csv.

  --suggest    gợi ý ranh giới chuỗi từ độ khác biệt giữa hai khung liên tiếp
               -> configs/clinicdb_sequences_SUGGESTED.csv (CHỈ để đối chiếu, không dùng cho kết quả)
  --calibrate  chọn ngưỡng băm tau khớp nhất với chuỗi thật (chỉ số Rand hiệu chỉnh),
               -> reports/cluster_calibration.csv; dùng tau này cho ColonDB/ETIS
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .hashing import cluster_hashes, hamming_matrix
from .split_kvasir import load_hashes
import cv2

from .utils import P, REPO, WORK, base_cfg, ensure_dir, load_index, read_image, save_json, write_with_hash


def _table_path():
    w = P("configs/clinicdb_sequences.csv")
    return w if w.exists() else REPO / "configs" / "clinicdb_sequences.csv"


def clinic_rows() -> pd.DataFrame:
    rows = load_index().query("dataset == 'clinicdb'").copy()
    if rows.empty:
        raise SystemExit("Chưa có CVC-ClinicDB trong data/index.csv")
    rows["frame"] = pd.to_numeric(rows.id, errors="coerce")
    if rows.frame.isna().any():
        raise SystemExit("Tên khung ClinicDB phải là số (1.tif, 2.tif, ...)")
    return rows.sort_values("frame").reset_index(drop=True)


def frame_to_sequence(rows: pd.DataFrame) -> pd.Series:
    tab = pd.read_csv(_table_path())
    if tab.empty:
        raise SystemExit("configs/clinicdb_sequences.csv còn trống. Điền theo Bảng 1 trang CVC-ClinicDB.")
    seq = pd.Series(np.nan, index=rows.index)
    cover = np.zeros(len(rows), int)
    for t in tab.itertuples():
        m = (rows.frame >= t.start_frame) & (rows.frame <= t.end_frame)
        cover += m.values.astype(int)
        seq[m] = t.sequence
    if (cover != 1).any():
        bad = rows.frame[cover != 1].astype(int).tolist()[:20]
        raise SystemExit(f"Bảng chuỗi chưa phủ đúng một lần mọi khung. Khung lỗi (tối đa 20): {bad}")
    return seq.astype(int)


def build(rows):
    seq = frame_to_sequence(rows)
    df = pd.DataFrame({"id": rows.id, "cluster": ["s%02d" % s for s in seq]})
    write_with_hash(df, "splits/clusters_clinicdb.csv")
    print(f"Đã ghi splits/clusters_clinicdb.csv: {df.cluster.nunique()} chuỗi, {len(df)} khung")


def suggest(rows, cfg, n_seq: int = 29, min_len: int = 5):
    """Gợi ý ranh giới chuỗi từ khác biệt điểm ảnh giữa hai khung liên tiếp (ảnh xám 32x24).

    Lấy n_seq-1 đỉnh khác biệt lớn nhất, với ức chế lân cận để mỗi chuỗi dài ít nhất min_len khung.
    (Phiên bản đầu dùng Hamming của mã băm với ngưỡng median+3IQR: trên ClinicDB thật nó gộp cả
    612 khung thành 1 chuỗi, nên đã bị thay.) Kết quả CHỈ để đối chiếu với Bảng 1 chính thức.
    """
    small = [cv2.resize(cv2.cvtColor(read_image(P(p)), cv2.COLOR_RGB2GRAY), (32, 24),
                        interpolation=cv2.INTER_AREA).astype(np.float32) for p in rows.image_path]
    d = np.array([np.abs(small[i + 1] - small[i]).mean() for i in range(len(small) - 1)])
    cuts = []
    for j in np.argsort(d)[::-1]:
        if len(cuts) == n_seq - 1:
            break
        if all(abs(j - c) >= min_len for c in cuts) and min_len - 1 <= j <= len(d) - min_len:
            cuts.append(int(j))
    cuts = sorted(cuts)
    starts = [0] + [c + 1 for c in cuts]
    ends = cuts + [len(rows) - 1]
    sug = pd.DataFrame({"sequence": range(1, len(starts) + 1),
                        "start_frame": rows.frame.iloc[starts].astype(int).values,
                        "end_frame": rows.frame.iloc[ends].astype(int).values})
    path = P("configs/clinicdb_sequences_SUGGESTED.csv")
    ensure_dir(path.parent)
    sug.to_csv(path, index=False)
    # độ tách: ranh giới yếu nhất so với khác biệt trung vị trong chuỗi
    inside = np.delete(d, cuts)
    print(f"Gợi ý {len(sug)} chuỗi. Khác biệt tại ranh giới yếu nhất {d[cuts].min():.1f}, "
          f"trung vị trong chuỗi {np.median(inside):.1f}. Ghi {path}. CHỈ để đối chiếu với Bảng 1.")
    return sug, d, cuts


def calibrate(rows, cfg):
    seq = frame_to_sequence(rows)
    H = load_hashes("clinicdb", rows.id.tolist())
    recs = []
    for tau in range(2, 33, 2):
        lab = cluster_hashes(H, tau, cfg["hashing"]["size"])
        recs.append({"tau": tau, "n_clusters": int(lab.max() + 1),
                     "ari": adjusted_rand_score(seq, lab)})
    df = pd.DataFrame(recs)
    df.to_csv(P("reports/cluster_calibration.csv"), index=False)
    best = df.loc[df.ari.idxmax()]
    save_json({"best_tau": int(best.tau), "ari": float(best.ari), "true_sequences": int(seq.nunique())},
              "reports/cluster_calibration.json")
    print(df.to_string(index=False))
    print(f"tau tốt nhất = {int(best.tau)} (ARI {best.ari:.3f}). "
          "Ghi giá trị này vào hashing.tau trong configs/base.yaml trước khi chạy src.build_clusters.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args(argv)
    cfg = base_cfg()
    rows = clinic_rows()
    if args.suggest:
        suggest(rows, cfg)
    elif args.calibrate:
        calibrate(rows, cfg)
    else:
        build(rows)


if __name__ == "__main__":
    main()
