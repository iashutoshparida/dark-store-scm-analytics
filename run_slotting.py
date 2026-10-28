"""Project 2: dark-store slotting study. Run: python run_slotting.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from qcr.config import RAW, REPORTS  # noqa: E402
from qcr.slotting import Layout, assign, evaluate, make_orders  # noqa: E402

STORE = "CA_1"
MEAN_LINES = 6            # ASSUMPTION: average lines per order
WALK_MPS = 1.0            # ASSUMPTION: walking speed with a pick trolley in narrow aisles
PICK_S = 12.0             # per-line pick time, low end of the 12-15 s cited for quick-commerce stores
EVAL_DAYS = 28
TRAIN_DAYS = 84           # velocity window used for slotting
STALE_GAP = 182           # "stale" slotting uses velocity from ~6 months earlier
SEEDS = (1, 2, 3)


def main():
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    df = sales[(sales.store_id == STORE) & (sales.cat_id == "FOODS")].reset_index(drop=True)
    y = df[day_cols].to_numpy()
    lay = Layout()
    # keep the SKUs that sold in the last year, ranked by recent velocity, capped at store capacity
    active = y[:, -365:].sum(axis=1) > 0
    df, y = df[active].reset_index(drop=True), y[active]
    recent = y[:, -(EVAL_DAYS + TRAIN_DAYS):-EVAL_DAYS].sum(axis=1)
    keep = np.argsort(-recent)[:lay.n_slots]
    df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    n = len(df)
    dept = df.dept_id.astype("category").cat.codes.to_numpy()

    test = y[:, -EVAL_DAYS:]
    vel_fresh = y[:, -(EVAL_DAYS + TRAIN_DAYS):-EVAL_DAYS].sum(axis=1)
    vel_stale = y[:, -(EVAL_DAYS + TRAIN_DAYS + STALE_GAP):-(EVAL_DAYS + STALE_GAP)].sum(axis=1)

    # demand concentration (pick-frequency Pareto) on the slotting window
    share = np.sort(vel_fresh)[::-1].cumsum() / vel_fresh.sum()
    top20_share = float(share[int(0.2 * n) - 1])

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        orders = make_orders(test, MEAN_LINES, rng)
        configs = [
            ("Random slotting", "random", vel_fresh),
            ("Category blocks (dept, then A-Z)", "category", vel_fresh),
            ("Velocity: nearest aisles first", "across_aisle", vel_fresh),
            ("Velocity: front band across aisles", "front_band", vel_fresh),
            ("Velocity: by distance to pack station", "distance", vel_fresh),
            ("Velocity by distance, 6-month-old data", "distance", vel_stale),
        ]
        for name, strat, vel in configs:
            ranked = np.argsort(-vel, kind="stable")
            a, yy = assign(ranked, lay, strat, dept=dept, rng=np.random.default_rng(seed + 100))
            for routing in ("s_shape", "best_of_two"):
                rows.append(dict(seed=seed, policy=name, routing=routing,
                                 **evaluate(orders, a, yy, lay, WALK_MPS, PICK_S, routing)))
    res = pd.DataFrame(rows)
    agg = res.groupby(["routing", "policy"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    base = agg.loc[(agg.policy == "Category blocks (dept, then A-Z)") & (agg.routing == "s_shape")].iloc[0]
    agg["walk_vs_category"] = agg.avg_walk_m / base.avg_walk_m - 1
    agg["pick_time_vs_category"] = agg.avg_pick_time_s / base.avg_pick_time_s - 1
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / "slotting_runs.csv", index=False)
    agg.to_csv(REPORTS / "slotting_results.csv", index=False)
    meta = {"store": STORE, "n_skus_slotted": int(n), "slots": lay.n_slots,
            "eval_window_days": EVAL_DAYS, "velocity_window_days": TRAIN_DAYS,
            "orders_per_seed": int(res.groupby("seed").orders.first().mean()),
            "top20pct_skus_share_of_units": top20_share,
            "assumptions": {"mean_lines_per_order": MEAN_LINES, "walk_speed_mps": WALK_MPS,
                            "pick_seconds_per_line": PICK_S,
                            "layout": "8 aisles x 12 m, 2.4 m apart, 800 slots"}}
    (REPORTS / "slotting_summary.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps(meta, indent=2))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
