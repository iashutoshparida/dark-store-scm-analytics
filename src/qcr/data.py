"""Load the public M5 files and build dense SKU-store x day matrices.

Output is a `Panel`: arrays shaped (n_series, n_days) plus per-series and per-day metadata.
Dense arrays keep the forecasting and simulation code simple and fast.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG, RAW, Config


@dataclass
class Panel:
    sales: np.ndarray          # (N, T) units sold per day
    price: np.ndarray          # (N, T) shelf price, forward/back filled
    meta: pd.DataFrame         # one row per series: item_id, dept_id, store_id
    calendar: pd.DataFrame     # one row per day: date, dow, snap, event

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]


def load_panel(cfg: Config = CFG, raw_dir=RAW) -> Panel:
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    keep = sales[(sales.cat_id == cfg.category) & (sales.store_id.isin(cfg.stores))]
    last_year = keep[day_cols[-365:]].mean(axis=1)
    keep = keep[last_year >= cfg.min_mean_daily_units].reset_index(drop=True)

    cols = day_cols[-cfg.history_days:]
    y = keep[cols].to_numpy(dtype=np.float64)

    cal = pd.read_csv(raw_dir / "calendar.csv")
    cal["d"] = [f"d_{i + 1}" for i in range(len(cal))]
    cal = cal.set_index("d").loc[cols].reset_index()
    state = cfg.stores[0].split("_")[0]
    calendar = pd.DataFrame({
        "date": pd.to_datetime(cal["date"]),
        "dow": pd.to_datetime(cal["date"]).dt.dayofweek.to_numpy(),
        "snap": cal[f"snap_{state}"].to_numpy(),
        "event": cal["event_name_1"].notna().astype(int).to_numpy(),
        "wm_yr_wk": cal["wm_yr_wk"].to_numpy(),
    })

    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    prices = prices[prices.store_id.isin(cfg.stores)]
    wide = prices.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    idx = pd.MultiIndex.from_frame(keep[["store_id", "item_id"]])
    wide = wide.reindex(idx)
    weekly = wide.reindex(columns=sorted(set(calendar.wm_yr_wk))).ffill(axis=1).bfill(axis=1)
    price = weekly[calendar.wm_yr_wk.to_numpy()].to_numpy(dtype=np.float64)

    meta = keep[["item_id", "dept_id", "store_id"]].copy()
    return Panel(sales=y, price=price, meta=meta, calendar=calendar)
