"""Huấn luyện một thí nghiệm (mục 4.4).

Chế độ:
  main      E1..E8 trên kvasir_train, dừng sớm theo kvasir_val      -> runs/<exp>/seed<seed>/
  lrsearch  như main nhưng với --lr chỉ định                         -> runs/lrsearch/<exp>/lr<lr>/
  e9        E9 trên CVC-ClinicDB, một phần của kiểm định chéo,
            số epoch cố định, không dừng sớm, dự đoán ngoài phần    -> runs/E9/<scheme>/fold<k>/

Ví dụ:
  python -m src.train --exp E3 --seed 42
  python -m src.train --exp E3 --mode lrsearch --lr 2e-4
  python -m src.train --exp E9 --mode e9 --scheme sequence --fold 0
  (thêm --resume để chạy tiếp sau khi phiên Kaggle bị ngắt)

Mỗi thư mục chạy có: best.pt, last.pt, log.csv, summary.json (cấu hình, mã băm split, môi trường).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PolypDataset, RandConv, worker_init_fn
from .losses import BCEDiceLoss
from .metrics import batch_dice
from .models import build_model, count_params
from .utils import (P, base_cfg, check_hash, ensure_dir, env_info, exp_cfg, load_json, run_dir,
                    save_json, set_rows, set_seed)


def lr_factor(epoch: int, total: int, warmup: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t))


def selected_lr(exp: str) -> float:
    path = P("configs/lr_selected.json")
    if not path.exists():
        raise SystemExit("Chưa có configs/lr_selected.json. Chạy lrsearch rồi python -m src.select_lr")
    sel = load_json(path)
    if exp not in sel:
        raise SystemExit(f"lr_selected.json không có {exp}")
    return float(sel[exp]["lr"])


def make_loaders(cfg, ecfg, train_rows, val_rows, seed):
    ds_tr = PolypDataset(train_rows, cfg["image_size"], train=True, appearance=ecfg["appearance"],
                         app_params=ecfg["appearance_params"], ref_rows=train_rows)
    g = torch.Generator()
    g.manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                       num_workers=cfg["num_workers"], worker_init_fn=worker_init_fn, generator=g,
                       pin_memory=torch.cuda.is_available(), persistent_workers=cfg["num_workers"] > 0)
    dl_va = None
    if val_rows is not None:
        ds_va = PolypDataset(val_rows, cfg["image_size"], train=False)
        dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                           num_workers=cfg["num_workers"])
    return dl_tr, dl_va


@torch.no_grad()
def validate(model, dl, dev, thr):
    model.eval()
    scores = []
    for b in dl:
        x, y = b["image"].to(dev), b["mask"].to(dev)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            logits = model(x)
        scores.append(batch_dice(logits.float(), y, thr).cpu())
    return float(torch.cat(scores).mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["main", "lrsearch", "e9"], default="main")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max_epochs", type=int, help="chỉ dùng cho chạy thử (vd. đo thời gian ngày 4)")
    ap.add_argument("--scheme", choices=["random", "sequence"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep_last", action="store_true", help="giữ last.pt sau khi xong (mặc định xóa, ~200 MB)")
    ap.add_argument("--stop_after", type=int, help="chỉ để kiểm thử --resume: dừng sau N epoch, không ghi summary")
    args = ap.parse_args(argv)

    cfg = base_cfg()
    ecfg = exp_cfg(args.exp)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["amp"]) and dev.type == "cuda"

    if args.mode == "e9":
        if args.scheme is None or args.fold is None:
            raise SystemExit("Chế độ e9 cần --scheme và --fold")
        seed = cfg["e9"]["seed"]
        lr = args.lr or selected_lr(cfg["e9"]["lr_from"])
        out = ensure_dir(f"runs/E9/{args.scheme}/fold{args.fold}")
        check_hash("splits/e9_folds.csv")
        folds = pd.read_csv(P("splits/e9_folds.csv"), dtype={"id": str})
        rows = set_rows("clinicdb").merge(folds[["id", f"fold_{args.scheme}"]], on="id")
        col = f"fold_{args.scheme}"
        train_rows = rows[rows[col] != args.fold].reset_index(drop=True)
        held_rows = rows[rows[col] == args.fold].reset_index(drop=True)
        val_rows = None
        max_epochs = args.max_epochs or cfg["e9"]["epochs"]
        split_hash = check_hash("splits/e9_folds.csv")
    else:
        seed = args.seed
        if args.mode == "lrsearch":
            if args.lr is None:
                raise SystemExit("lrsearch cần --lr")
            lr = args.lr
            out = ensure_dir(f"runs/lrsearch/{args.exp}/lr{lr:g}")
        else:
            lr = args.lr or selected_lr(args.exp)
            out = ensure_dir(run_dir(args.exp, seed))
        train_rows = set_rows("kvasir_train")
        val_rows = set_rows("kvasir_val")
        max_epochs = args.max_epochs or cfg["max_epochs"]
        split_hash = check_hash("splits/kvasir_split.csv")

    set_seed(seed)
    model = build_model(ecfg).to(dev)
    randconv = RandConv(**ecfg["appearance_params"]) if ecfg["appearance"] == "randconv" else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_factor(e, max_epochs, cfg["warmup_epochs"]))
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    loss_fn = BCEDiceLoss()
    dl_tr, dl_va = make_loaders(cfg, ecfg, train_rows, val_rows, seed)

    if (out / "summary.json").exists():
        print(f"{out} đã hoàn tất (có summary.json). Bỏ qua. Xóa thư mục nếu thật sự muốn chạy lại.")
        return
    start, best, best_epoch, bad, log = 0, -1.0, -1, 0, []
    if args.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best, best_epoch, bad = st["epoch"] + 1, st["best"], st["best_epoch"], st["bad"]
        torch.set_rng_state(st["rng_torch"])
        np.random.set_state(st["rng_np"])
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        log = pd.read_csv(out / "log.csv").to_dict("records") if (out / "log.csv").exists() else []
        log = [r for r in log if r["epoch"] < start]
        print(f"Tiếp tục từ epoch {start}")
    elif (out / "last.pt").exists():
        raise SystemExit(f"{out} có lần chạy dở dang. Dùng --resume để chạy tiếp.")

    patience = cfg["patience"] if args.mode != "e9" else None
    for epoch in range(start, max_epochs):
        model.train()
        t0, losses = time.time(), []
        for b in dl_tr:
            x, y = b["image"].to(dev, non_blocking=True), b["mask"].to(dev, non_blocking=True)
            if randconv is not None:
                x = randconv(x)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": float(np.mean(losses)),
               "seconds": time.time() - t0}
        if dl_va is not None:
            rec["val_dice"] = validate(model, dl_va, dev, cfg["threshold"])
            if rec["val_dice"] > best:
                best, best_epoch, bad = rec["val_dice"], epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best.pt")
            else:
                bad += 1
        log.append(rec)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch, "bad": bad, "rng_torch": torch.get_rng_state(),
                    "rng_np": np.random.get_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   out / "last.pt")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
              flush=True)
        if args.stop_after is not None and epoch + 1 - start >= args.stop_after:
            print(f"Dừng thử sau epoch {epoch} (--stop_after). Chạy lại với --resume.")
            return
        if patience is not None and bad >= patience:
            print(f"Dừng sớm ở epoch {epoch}; tốt nhất epoch {best_epoch} (Dice val {best:.4f})")
            break

    summary = {"exp": args.exp, "seed": seed, "mode": args.mode, "lr": lr, "epochs_run": len(log),
               "best_epoch": best_epoch, "best_val_dice": best,
               "seconds_per_epoch": float(np.mean([r["seconds"] for r in log])) if log else None,
               "n_params": count_params(model), "split_sha256": split_hash,
               "config": cfg, "exp_config": ecfg, "env": env_info()}

    if args.mode == "e9":
        from .inference import evaluate_rows
        model.eval()
        img, _ = evaluate_rows(model, held_rows, "clinicdb")
        img["fold"] = args.fold
        img["scheme"] = args.scheme
        img.to_csv(out / "oof.csv", index=False)
        summary["oof_mean_dice"] = float(img.dice.mean())
        summary["scheme"], summary["fold"] = args.scheme, args.fold

    save_json(summary, out / "summary.json")
    # Dọn checkpoint để tiết kiệm đĩa: lrsearch chỉ cần summary; E9 chỉ cần oof.csv.
    if not args.keep_last:
        (out / "last.pt").unlink(missing_ok=True)
    if args.mode in ("lrsearch", "e9"):
        (out / "best.pt").unlink(missing_ok=True)
    print(f"Xong: {out}")


if __name__ == "__main__":
    main()
