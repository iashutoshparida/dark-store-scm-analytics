"""Day-ahead demand forecasts with a rolling-origin backtest.

Every forecast for day t+h uses only data up to day t (the night the replenishment order is placed).
Models:
  snaive  - same weekday last week
  ma28    - 28-day moving average
  dow_ma  - average of the same weekday over the last 4 weeks
  lgbm    - one global LightGBM model (Tweedie loss) across all SKU-stores, trained per fold on an
            expanding window that ends before the fold starts
Probable-stockout days are hidden from features and never used as training targets.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CFG, Config

MODELS = ("snaive", "ma28", "dow_ma", "lgbm")


@dataclass
class Folds:
    starts: list      # first day index of each fold (calibration folds first)
    length: int
    n_calib: int

    def bounds(self, k):
        return self.starts[k], self.starts[k] + self.length

    @property
    def test_start(self):
        return self.starts[self.n_calib]


def make_folds(n_days: int, cfg: Config = CFG) -> Folds:
    n = cfg.n_calib_folds + cfg.n_test_folds
    first = n_days - n * cfg.fold_len
    return Folds([first + k * cfg.fold_len for k in range(n)], cfg.fold_len, cfg.n_calib_folds)


def _rolling(y_nan: np.ndarray, w: int, fn: str) -> np.ndarray:
    df = pd.DataFrame(y_nan.T)
    r = df.rolling(w, min_periods=max(3, w // 4))
    return (r.mean() if fn == "mean" else r.std()).to_numpy().T


class FeatureBuilder:
    def __init__(self, sales, flags, price, calendar, meta):
        self.y = sales
        self.y_nan = np.where(flags, np.nan, sales)
        self.flags = flags
        self.r7 = _rolling(self.y_nan, 7, "mean")
        self.r28 = _rolling(self.y_nan, 28, "mean")
        self.s28 = _rolling(self.y_nan, 28, "std")
        pmean = pd.DataFrame(price.T).rolling(28, min_periods=1).mean().to_numpy().T
        self.price = price
        self.price_rel = price / pmean
        self.dow = calendar.dow.to_numpy()
        self.snap = calendar.snap.to_numpy()
        self.event = calendar.event.to_numpy()
        self.dept = meta.dept_id.astype("category").cat.codes.to_numpy()
        self.store = meta.store_id.astype("category").cat.codes.to_numpy()
        self.n, self.t = sales.shape
        self.long_mean = np.nanmean(self.y_nan, axis=1)

    def rows(self, origins: np.ndarray, h: int):
        """Feature matrix for all series x origins at horizon h. Returns X, target, target_censored, (i, t) index."""
        ii, tt = np.meshgrid(np.arange(self.n), origins, indexing="ij")
        ii, tt = ii.ravel(), tt.ravel()
        tgt = tt + h
        valid = tgt < self.t
        ii, tt, tgt = ii[valid], tt[valid], tgt[valid]
        f = {f"lag_{k}": self.y_nan[ii, tt - k] for k in range(7)}
        for k in (7, 14, 21, 28):
            f[f"same_dow_{k}"] = self.y_nan[ii, tgt - k]
        f.update(
            r7=self.r7[ii, tt], r28=self.r28[ii, tt], s28=self.s28[ii, tt],
            dow=self.dow[tgt], snap=self.snap[tgt], event=self.event[tgt],
            price=self.price[ii, tgt], price_rel=self.price_rel[ii, tgt],
            dept=self.dept[ii], store=self.store[ii], h=np.full(len(ii), h),
        )
        X = pd.DataFrame(f)
        return X, self.y[ii, tgt], self.flags[ii, tgt], ii, tt


def baseline(fb: FeatureBuilder, name: str, ii, tt, h):
    tgt = tt + h
    if name == "snaive":
        return np.nan_to_num(fb.y_nan[ii, tgt - 7], nan=fb.r28[ii, tt])
    if name == "ma28":
        return fb.r28[ii, tt]
    if name == "dow_ma":
        vals = np.stack([fb.y_nan[ii, tgt - k] for k in (7, 14, 21, 28)])
        with np.errstate(all="ignore"):
            cnt = (~np.isnan(vals)).sum(axis=0)
            out = np.where(cnt > 0, np.nansum(vals, axis=0) / np.maximum(cnt, 1), np.nan)
        return np.where(np.isnan(out), fb.r28[ii, tt], out)
    raise ValueError(name)


LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)


def backtest(fb: FeatureBuilder, folds: Folds, cfg: Config = CFG, n_rounds: int = 300, log=print):
    """Return preds[model] arrays of shape (N, T, H+1); index [i, t, h] = forecast made at origin t for day t+h."""
    H = cfg.max_horizon
    preds = {m: np.full((fb.n, fb.t, H + 1), np.nan, dtype=np.float32) for m in MODELS}
    importance = None
    for k in range(len(folds.starts)):
        lo, hi = folds.bounds(k)
        test_origins = np.arange(lo - 1, hi - 1)
        # training origins: targets strictly before the fold starts
        train_origins = np.arange(max(28, lo - 1 - cfg.train_window), lo - 1)
        Xs, ys = [], []
        for h in range(1, H + 1):
            X, y, cens, _, tt = fb.rows(train_origins, h)
            keep = (~cens) & (tt + h < lo)
            Xs.append(X[keep]); ys.append(y[keep])
        X_train = pd.concat(Xs, ignore_index=True); y_train = np.concatenate(ys)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, y_train, categorical_feature=["dept", "store", "dow"]),
                          num_boost_round=n_rounds)
        imp = pd.Series(model.feature_importance("gain"), index=X_train.columns)
        importance = imp if importance is None else importance + imp
        for h in range(1, H + 1):
            X, _, _, ii, tt = fb.rows(test_origins, h)
            preds["lgbm"][ii, tt, h] = np.maximum(model.predict(X), 0)
            for m in MODELS[:-1]:
                b_ = baseline(fb, m, ii, tt, h)
                b_ = np.where(np.isnan(b_), fb.r28[ii, tt], b_)
                b_ = np.where(np.isnan(b_), fb.long_mean[ii], b_)
                preds[m][ii, tt, h] = np.maximum(b_, 0)
        log(f"  fold {k + 1}/{len(folds.starts)}: trained on {len(y_train):,} rows")
    return preds, (importance / importance.sum()).sort_values(ascending=False)


def accuracy_table(preds, sales, flags, folds: Folds, seg: pd.Series, h: int = 1) -> pd.DataFrame:
    """WAPE and bias at horizon h over the test folds, on days not flagged as probable stockouts."""
    lo, hi = folds.test_start, folds.starts[-1] + folds.length
    origins = np.arange(lo - 1, hi - 1)
    origins = origins[origins + h < sales.shape[1]]
    tgt = origins + h
    y = sales[:, tgt]
    ok = ~flags[:, tgt]
    out = []
    for m, p in preds.items():
        f = p[:, origins, h]
        for name, mask in [("All", np.ones(len(seg), bool))] + [(s, (seg == s).to_numpy()) for s in sorted(seg.unique())]:
            yy, ff, oo = y[mask], f[mask], ok[mask]
            err = np.abs(ff - yy)[oo].sum()
            out.append(dict(model=m, segment=name, wape=err / yy[oo].sum(),
                            bias=(ff - yy)[oo].sum() / yy[oo].sum(), n_series=int(mask.sum())))
    return pd.DataFrame(out)
