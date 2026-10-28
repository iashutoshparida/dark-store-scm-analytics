"""Project-wide settings. Every business assumption lives here so it can be read and changed in one place.

Values marked ASSUMPTION are not taken from any company. They are stated, round, conservative
numbers chosen to make the simulation concrete, and the scenario runs vary the important ones.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class Config:
    # ---- data scope (public M5 data, Walmart US, 2011-2016) ----
    stores: tuple = ("CA_1", "CA_2", "CA_3")        # three stores treated as three dark stores
    category: str = "FOODS"
    min_mean_daily_units: float = 1.0               # keep SKU-stores selling >= 1 unit/day in the final year
    history_days: int = 730                         # last two years of daily history

    # ---- backtest design ----
    fold_len: int = 28
    n_test_folds: int = 4                           # 4 x 28 = 112 test days
    n_calib_folds: int = 1                          # one earlier fold, used only to tune policies
    max_horizon: int = 3                            # forecast 1..3 days ahead
    train_window: int = 365                         # forecast origins used for training per fold

    # ---- stockout (censored demand) detection ----
    censor_min_run: int = 3                         # consecutive zero-sale days
    censor_pvalue: float = 1e-3                     # P(run of zeros | Poisson at trailing mean)

    # ---- economics (ASSUMPTIONS) ----
    gross_margin: float = 0.30                      # unit cost = 70% of shelf price
    holding_rate_daily: float = 0.18 / 365          # 18% annual carrying cost on unit cost

    # ---- operations (ASSUMPTIONS) ----
    base_lead_time: int = 1                         # order at night, delivered next morning from DC
    shelf_life_days: int = 5                        # headline perishable scenario
    shelf_life_grid: tuple = (3, 5, 7, 14)
    stress_lead_times: tuple = (1, 2, 3)
    stress_lead_probs: tuple = (0.6, 0.3, 0.1)      # late DC deliveries in the stress test
    warmup_days: int = 7
    seed: int = 7

    # ---- policy grid searched on the calibration fold ----
    csl_grid: tuple = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
    fill_floor: tuple = (("A", 0.97), ("B", 0.95), ("C", 0.90))   # ASSUMPTION: minimum fill rate by ABC class
    uniform_csl: float = 0.95
    minmax_days: tuple = (2.0, 4.0)                 # rule-of-thumb min / max in days of cover


CFG = Config()
