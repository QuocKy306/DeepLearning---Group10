"""Đánh giá checkpoint best.pt trên các tập (mục 7, 9.2).

Mọi tập khác kvasir_val là tập kiểm tra: cần --confirm_test, chỉ được đánh giá MỘT lần cho mỗi
(thí nghiệm, seed, tập); mỗi lần được ghi vào runs/TEST_EVALUATED.txt.

--color_norm (thí nghiệm phản thực tế 6.4, chỉ chạy SAU đánh giá chính):
   to_kvasir      chuyển màu ảnh về thống kê Lab trung bình của kvasir_train
   to_<tập>       chuyển màu ảnh về thống kê Lab trung bình của <tập> (dùng cho kvasir_test)
   Kết quả ghi file riêng *_cf-<mode>.csv, không ghi vào TEST_EVALUATED.

Đầu ra trong runs/<exp>/seed<seed>/: eval_<set>.csv (theo ảnh), eval_<set>_polyps.csv, eval_<set>.json
Chạy: python -m src.evaluate --exp E3 --seed 42 --sets kvasir_val
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from .color import lab_stats
from .inference import evaluate_rows, load_checkpoint
from .utils import (P, TEST_LOG, check_hash, ensure_dir, exp_cfg, load_json, read_image, run_dir,
                    save_json, set_rows, sha256_file)


def already_evaluated(exp, seed, s) -> bool:
    path = P(TEST_LOG)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == exp and parts[1] == str(seed) and parts[2] == s:
            return True
    return False


def log_test(exp, seed, s, ckpt_hash):
    ensure_dir(P(TEST_LOG).parent)
    with open(P(TEST_LOG), "a", encoding="utf-8") as f:
        f.write(f"{exp}\t{seed}\t{s}\t{dt.datetime.now().isoformat(timespec='seconds')}\t{ckpt_hash[:16]}\n")


def mean_lab_stats(set_name: str) -> np.ndarray:
    cache = P(f"reports/color/mean_lab_{set_name}.npy")
    if cache.exists():
        return np.load(cache)
    rows = set_rows(set_name)
    st = np.stack([lab_stats(read_image(P(p))) for p in rows.image_path]).mean(0)
    ensure_dir(cache.parent)
    np.save(cache, st)
    return st


def summarize(img: pd.DataFrame, pol: pd.DataFrame) -> dict:
    out = {"n_images": len(img)}
    if "dice" in img:
        for k in ["dice", "iou", "precision", "recall"]:
            out[f"m{k}"] = float(img[k].mean())
        out["hd95_nonempty"] = float(img.hd95.mean(skipna=True))
        out["mnsd"] = float(img.nsd.mean())
        out["hd95_diag"] = float(img.hd95_diag.mean(skipna=True))
        out["empty_pred_rate"] = float(img.empty_pred.mean())
        if len(pol):
            out["polyp_sensitivity"] = float(pol.detected.mean())
            out["polyp_sensitivity_by_size"] = pol.groupby("size").detected.mean().to_dict()
        out["fp_components_per_image"] = float(img.n_fp_components.mean())
    else:
        out["fp_frame_rate"] = float(img.fp_frame.mean())
        out["fp_components_per_image"] = float(img.n_fp_components.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sets", nargs="+", default=["kvasir_val"])
    ap.add_argument("--confirm_test", action="store_true")
    ap.add_argument("--color_norm", default="none")
    ap.add_argument("--skip_done", action="store_true",
                    help="bỏ qua (không lỗi) các tập kiểm tra đã đánh giá; dùng khi chạy tiếp sau gián đoạn")
    args = ap.parse_args(argv)

    rd = run_dir(args.exp, args.seed)
    ckpt = rd / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Thiếu {ckpt}")
    summ = load_json(rd / "summary.json")
    if summ["split_sha256"] != check_hash("splits/kvasir_split.csv"):
        raise SystemExit("splits/kvasir_split.csv khác với lúc huấn luyện. Dừng lại.")
    ckpt_hash = sha256_file(ckpt)
    model = load_checkpoint(ckpt, exp_cfg(args.exp))

    for s in args.sets:
        is_test = s != "kvasir_val"
        cf = args.color_norm != "none"
        if is_test and not cf:
            if not args.confirm_test:
                raise SystemExit(f"{s} là tập kiểm tra: cần --confirm_test (chỉ sau họp chốt ngày 14).")
            if already_evaluated(args.exp, args.seed, s):
                if args.skip_done:
                    print(f"bỏ qua {args.exp}/seed{args.seed}/{s}: đã đánh giá")
                    continue
                raise SystemExit(f"{args.exp}/seed{args.seed}/{s} đã được đánh giá. Không đánh giá lại.")
        if cf and is_test and not already_evaluated(args.exp, args.seed, s):
            raise SystemExit("Phân tích phản thực tế chỉ chạy sau đánh giá chính trên tập này.")
        stats = None
        if cf:
            target = "kvasir_train" if args.color_norm == "to_kvasir" else args.color_norm[3:]
            stats = mean_lab_stats(target)
        rows = set_rows(s)
        img, pol = evaluate_rows(model, rows, s, color_norm_stats=stats)
        tag = s + (f"_cf-{args.color_norm}" if cf else "")
        img.to_csv(rd / f"eval_{tag}.csv", index=False)
        pol.to_csv(rd / f"eval_{tag}_polyps.csv", index=False)
        res = summarize(img, pol)
        res.update({"exp": args.exp, "seed": args.seed, "set": s, "color_norm": args.color_norm,
                    "checkpoint_sha256": ckpt_hash})
        save_json(res, rd / f"eval_{tag}.json")
        if is_test and not cf:
            log_test(args.exp, args.seed, s, ckpt_hash)
        print(f"{args.exp} seed{args.seed} {tag}: " +
              ", ".join(f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
