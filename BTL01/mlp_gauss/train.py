"""Vòng lặp huấn luyện / validation / checkpoint / đánh giá – viết tay bằng PyTorch."""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import TrainConfig
from .data import make_loader, to_tensors
from .metrics import regression_metrics
from .model import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CPU mặc định, GPU nếu có


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, x_np: np.ndarray) -> np.ndarray:
    """Dự đoán ở chế độ eval, không tính gradient."""
    model.eval()
    xt, _ = to_tensors(x_np, np.zeros(len(x_np)))
    return model(xt.to(DEVICE)).cpu().numpy().ravel()


def build_optimizer(model, c: TrainConfig):
    if c.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    if c.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=c.lr, momentum=0.9, weight_decay=c.weight_decay)
    raise ValueError(c.optimizer)


def train_model(c: TrainConfig, splits: dict, ckpt_path: Path | None = None):
    """Huấn luyện một cấu hình với một seed. Trả về (model tốt nhất, bản ghi kết quả, lịch sử theo epoch)."""
    set_seed(c.seed)
    x_tr, y_tr = splits["train"][0][: c.n_train], splits["train"][1][: c.n_train]
    x_va, y_va = splits["val"]
    x_te, y_te = splits["test"]

    loader = make_loader(x_tr, y_tr, c.batch_size, seed=c.seed, shuffle=True)
    xtr_t, ytr_t = (t.to(DEVICE) for t in to_tensors(x_tr, y_tr))
    xva_t, yva_t = (t.to(DEVICE) for t in to_tensors(x_va, y_va))

    model = MLP(hidden=c.hidden, activation=c.activation).to(DEVICE)
    model.set_input_stats(xtr_t)                   # z-score theo thống kê của tập train
    loss_fn = nn.MSELoss()
    opt = build_optimizer(model, c)
    # Cosine annealing: lr giảm mượt từ c.lr về c.min_lr trong c.max_epochs epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.max_epochs, eta_min=c.min_lr)

    first_linear = model.net[0]
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, c.max_epochs + 1):
        # ---------- train ----------
        model.train()
        running, grad_first = 0.0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)                       # forward
            loss = loss_fn(pred, yb)               # MSE
            opt.zero_grad()                        # xóa gradient cũ
            loss.backward()                        # backpropagation
            grad_first += first_linear.weight.grad.norm().item()
            opt.step()                             # cập nhật trọng số
            running += loss.item() * len(xb)
        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            train_mse = loss_fn(model(xtr_t), ytr_t).item()
            val_mse = loss_fn(model(xva_t), yva_t).item()
        sched.step()
        history.append({"epoch": epoch, "train_batch_loss": running / len(x_tr), "train_mse": train_mse,
                        "val_mse": val_mse, "lr": opt.param_groups[0]["lr"],
                        "grad_norm_first_layer": grad_first / len(loader)})
        # ---------- checkpoint theo val ----------
        if val_mse < best_val * (1 - 1e-4):
            best_val, best_epoch, wait = val_mse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= c.patience:                 # early stopping
                break
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)              # khôi phục checkpoint tốt nhất
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": c.to_dict(), "best_epoch": best_epoch,
                    "best_val_mse": best_val}, ckpt_path)

    record = {**c.to_dict(), "n_params": model.n_params(), "best_epoch": best_epoch,
              "epochs_run": len(history), "train_time_s": train_time,
              "time_per_epoch_ms": 1000 * train_time / len(history)}
    for split, (x, y) in (("train", (x_tr, y_tr)), ("val", (x_va, y_va)), ("test", (x_te, y_te))):
        for k, v in regression_metrics(y, predict(model, x)).items():
            record[f"{split}_{k}"] = v
    return model, record, history
