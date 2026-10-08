"""Error metrics and day-block bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def mae(f, y):
    return float(np.mean(np.abs(f - y)))


def rmse(f, y):
    return float(np.sqrt(np.mean((f - y) ** 2)))


def daily_abs_err(f, y):
    return np.mean(np.abs(f - y), axis=1)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, block: int = 7, seed: int = 0, level: float = 0.95):
    """Moving-block bootstrap CI for the mean of a daily series (keeps weekly autocorrelation)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def coverage(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi)))


def pinball(q_pred, y, q):
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))
