"""Chọn tốc độ học cho từng cấu hình từ các lần chạy lrsearch (seed 42, chỉ dùng kvasir_val).

Đọc runs/lrsearch/<exp>/lr*/summary.json -> configs/lr_selected.json và reports/lr_search.csv.
Hòa nhau (chênh < 1e-4) thì chọn tốc độ học nhỏ hơn.
Chạy: python -m src.select_lr
"""
from __future__ import annotations

import pandas as pd

from .utils import P, ensure_dir, load_json, save_json


def main():
    recs = []
    for f in sorted(P("runs/lrsearch").glob("*/lr*/summary.json")):
        s = load_json(f)
        recs.append({"exp": s["exp"], "lr": s["lr"], "best_val_dice": s["best_val_dice"],
                     "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"]})
    if not recs:
        raise SystemExit("Chưa có lần chạy lrsearch nào trong runs/lrsearch/")
    df = pd.DataFrame(recs).sort_values(["exp", "lr"])
    ensure_dir("reports")
    df.to_csv(P("reports/lr_search.csv"), index=False)
    sel = {}
    for exp, g in df.groupby("exp"):
        top = g.best_val_dice.max()
        cand = g[g.best_val_dice >= top - 1e-4].sort_values("lr")
        r = cand.iloc[0]
        sel[exp] = {"lr": float(r.lr), "val_dice": float(r.best_val_dice), "n_tried": len(g)}
    save_json(sel, "configs/lr_selected.json")
    print(df.to_string(index=False))
    print(sel)


if __name__ == "__main__":
    main()
