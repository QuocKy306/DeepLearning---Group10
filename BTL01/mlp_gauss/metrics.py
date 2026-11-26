"""Các độ đo hồi quy."""
import numpy as np


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "maxae": float(np.max(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }
