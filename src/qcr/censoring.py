"""Flag days where zero sales most likely mean "out of stock", not "no demand".

Retail sales data records what was sold, not what customers wanted. When a shelf is empty,
recorded sales drop to zero and a model trained on them learns too little demand.
M5 has no stock-status field, so we use a run test: a run of k zero-sale days on a SKU whose
trailing 28-day mean is mu has probability exp(-mu * k) under Poisson demand. Runs with
probability below `pvalue` are flagged as probable stockouts.
"""
import numpy as np


def flag_probable_stockouts(sales: np.ndarray, min_run: int = 3, pvalue: float = 1e-3,
                            window: int = 28) -> np.ndarray:
    n, t = sales.shape
    flags = np.zeros_like(sales, dtype=bool)
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(sales, axis=1)], axis=1)
    for i in range(n):
        row = sales[i]
        j = 0
        while j < t:
            if row[j] != 0:
                j += 1
                continue
            start = j
            while j < t and row[j] == 0:
                j += 1
            run = j - start
            if run < min_run or start < window:
                continue
            mu = (csum[i, start] - csum[i, start - window]) / window
            if np.exp(-mu * run) < pvalue:
                flags[i, start:j] = True
    return flags


def impute_censored(sales: np.ndarray, flags: np.ndarray, window: int = 28) -> np.ndarray:
    """Replace flagged days with the trailing mean of unflagged days (used only as simulated demand)."""
    out = sales.copy()
    n, t = sales.shape
    for i in np.where(flags.any(axis=1))[0]:
        for j in np.where(flags[i])[0]:
            lo = max(0, j - window)
            ok = ~flags[i, lo:j]
            out[i, j] = sales[i, lo:j][ok].mean() if ok.any() else 0.0
    return out
