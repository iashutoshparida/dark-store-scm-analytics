"""ABC (revenue share) x XYZ (demand variability) segmentation, computed on training data only."""
import numpy as np
import pandas as pd


def abc_xyz(sales: np.ndarray, price: np.ndarray, store_ids: np.ndarray,
            a_cut: float = 0.80, b_cut: float = 0.95,
            x_cut: float = 0.5, y_cut: float = 1.0) -> pd.DataFrame:
    """ABC within each store by cumulative revenue; XYZ by coefficient of variation of weekly units."""
    revenue = (sales * price).sum(axis=1)
    abc = np.empty(len(revenue), dtype=object)
    for s in np.unique(store_ids):
        m = store_ids == s
        order = np.argsort(-revenue[m])
        share = np.cumsum(revenue[m][order]) / revenue[m].sum()
        cls = np.where(share <= a_cut, "A", np.where(share <= b_cut, "B", "C"))
        tmp = np.empty(m.sum(), dtype=object)
        tmp[order] = cls
        abc[m] = tmp

    weeks = sales.shape[1] // 7
    weekly = sales[:, -weeks * 7:].reshape(len(sales), weeks, 7).sum(axis=2)
    cv = weekly.std(axis=1) / np.maximum(weekly.mean(axis=1), 1e-9)
    xyz = np.where(cv <= x_cut, "X", np.where(cv <= y_cut, "Y", "Z"))
    return pd.DataFrame({"abc": abc, "xyz": xyz, "segment": abc.astype(str) + xyz, "weekly_cv": cv,
                         "revenue": revenue})
