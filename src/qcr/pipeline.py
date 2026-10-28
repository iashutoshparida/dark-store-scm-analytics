"""End-to-end run: data -> stockout flags -> segments -> forecasts -> policy tuning -> simulation -> results."""
import json
import time

import numpy as np
import pandas as pd

from . import censoring, forecast, segmentation
from .config import CFG, PROCESSED, REPORTS, Config
from .data import load_panel
from .simulate import Policy, SimInputs, simulate, summarise


def forecast_error_sigma(preds, demand, folds, n_days, max_p=3, model="lgbm", fallback_std=None):
    """sigma[P][i, t]: std of P-day-sum forecast errors from all earlier out-of-sample folds.

    For the first fold there are no earlier out-of-sample errors, so a scaled rolling std is used.
    """
    N = demand.shape[0]
    sig = {}
    for P in range(1, max_p + 1):
        s = np.full((N, n_days), np.nan)
        errs = []
        for k in range(len(folds.starts)):
            lo, hi = folds.bounds(k)
            origins = np.arange(lo - 1, hi - 1)
            if errs:
                e = np.concatenate(errs, axis=1)
                s[:, lo - 1:hi - 1] = np.nanstd(e, axis=1, ddof=1)[:, None]
            else:
                s[:, lo - 1:hi - 1] = fallback_std[:, lo - 1:hi - 1] * np.sqrt(P)
            valid = origins + P < n_days
            o = origins[valid]
            f = np.nansum(np.stack([preds[model][:, o, h] for h in range(1, P + 1)]), axis=0)
            a = np.stack([demand[:, o + h] for h in range(1, P + 1)]).sum(axis=0)
            errs.append(a - f)
        # floor: never below a Poisson-like std of the forecast itself
        fP = np.nansum(np.stack([preds[model][:, :, h] for h in range(1, P + 1)]), axis=0)
        sig[P] = np.fmax(s, np.sqrt(np.maximum(fP, 0)) * 0.5)
    return sig


def run(cfg: Config = CFG, log=print):
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    log("1/7 loading M5 FOODS data")
    panel = load_panel(cfg)
    N, T = panel.sales.shape
    folds = forecast.make_folds(T, cfg)

    log("2/7 flagging probable stockout days")
    flags = censoring.flag_probable_stockouts(panel.sales, cfg.censor_min_run, cfg.censor_pvalue)
    demand = censoring.impute_censored(panel.sales, flags)

    log("3/7 ABC-XYZ segmentation on pre-test data only")
    train_end = folds.starts[0]
    seg = segmentation.abc_xyz(panel.sales[:, :train_end], panel.price[:, :train_end],
                               panel.meta.store_id.to_numpy())
    seg = pd.concat([panel.meta.reset_index(drop=True), seg], axis=1)

    log("4/7 rolling-origin forecast backtest")
    fb = forecast.FeatureBuilder(panel.sales, flags, panel.price, panel.calendar, panel.meta)
    cache = PROCESSED / "backtest_preds.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        preds = {m: z[m] for m in forecast.MODELS}
        importance = pd.Series(z["imp_vals"], index=z["imp_names"])
        log("  loaded cached backtest forecasts")
    else:
        preds, importance = forecast.backtest(fb, folds, cfg, log=log)
        np.savez_compressed(cache, **preds, imp_vals=importance.values, imp_names=importance.index.values)
    acc = pd.concat([forecast.accuracy_table(preds, panel.sales, flags, folds, seg.segment, h=h).assign(horizon=h)
                     for h in (1, 2, 3)])

    log("5/7 tuning per-segment service levels on the calibration fold")
    mean_daily = fb.r28.copy()
    mean_daily = np.where(np.isnan(mean_daily), np.nanmean(panel.sales, axis=1, keepdims=True), mean_daily)
    fallback = np.where(np.isnan(fb.s28), np.sqrt(mean_daily), fb.s28)
    sigmas = {m: forecast_error_sigma(preds, demand, folds, T, model=m, fallback_std=fallback)
              for m in ("lgbm", "ma28")}

    def inputs(model):
        return SimInputs(demand=demand, price=panel.price, preds=preds, sigma=sigmas[model],
                         mean_daily=mean_daily, gross_margin=cfg.gross_margin,
                         holding_rate_daily=cfg.holding_rate_daily)

    c_lo, c_hi = folds.bounds(0)
    test_lo, test_hi = folds.test_start, folds.starts[-1] + folds.length
    segments = seg.segment.to_numpy()

    def tune(shelf_life, lead_times=(cfg.base_lead_time,), lead_probs=(1.0,), lt_aware=False):
        contrib, sales_, dem_ = {}, {}, {}
        for c in cfg.csl_grid:
            pol = Policy("probe", "order_up_to", "lgbm", c, expiry_cap=True, lt_aware=lt_aware)
            r = simulate(pol, inputs("lgbm"), c_lo, c_hi, c_lo + cfg.warmup_days, shelf_life,
                         lead_times, lead_probs, cfg.seed)
            contrib[c] = r.per_series["contribution"]
            sales_[c] = r.per_series["sales"]; dem_[c] = r.per_series["demand"]
        floors = dict(cfg.fill_floor)
        best = {}
        for s in np.unique(segments):
            m = segments == s
            fill = {c: sales_[c][m].sum() / dem_[c][m].sum() for c in cfg.csl_grid}
            ok = [c for c in cfg.csl_grid if fill[c] >= floors[s[0]]]
            best[s] = max(ok, key=lambda c: contrib[c][m].sum()) if ok else max(cfg.csl_grid, key=lambda c: fill[c])
        return np.array([best[s] for s in segments]), best

    def policies(csl_vec, lt_aware=False):
        return [
            Policy("Min/max rule (2-4 days cover)", "minmax", minmax_days=cfg.minmax_days),
            Policy("Order-up-to, MA-28 forecast, 95% CSL", "order_up_to", "ma28", cfg.uniform_csl, lt_aware=lt_aware),
            Policy("Order-up-to, LightGBM forecast, 95% CSL", "order_up_to", "lgbm", cfg.uniform_csl, lt_aware=lt_aware),
            Policy("Proposed: LightGBM + segment CSL + expiry cap", "order_up_to", "lgbm", csl_vec,
                   expiry_cap=True, lt_aware=lt_aware),
        ]

    def evaluate(shelf_life, lead_times=(cfg.base_lead_time,), lead_probs=(1.0,), lt_aware=False, tag="base"):
        csl_vec, best = tune(shelf_life, lead_times, lead_probs, lt_aware)
        rows, results = [], {}
        for pol in policies(csl_vec, lt_aware):
            model = "ma28" if pol.forecast == "ma28" else "lgbm"
            r = simulate(pol, inputs(model), test_lo - cfg.warmup_days, test_hi, test_lo, shelf_life,
                         lead_times, lead_probs, cfg.seed)
            results[pol.name] = r
            rows.append(dict(scenario=tag, shelf_life=shelf_life, policy=pol.name, **summarise(r)))
        return pd.DataFrame(rows), results, best

    log("6/7 simulating policies on the 112-day test window")
    base_df, base_res, best_csl = evaluate(cfg.shelf_life_days, tag="base")
    seg_rows = []
    for pname, r in base_res.items():
        for s in sorted(np.unique(segments)):
            seg_rows.append(dict(policy=pname, segment=s, **summarise(r, segments == s)))
    seg_df = pd.DataFrame(seg_rows)

    frontier = []
    for c in cfg.csl_grid:
        pol = Policy(f"order-up-to {c:.2f}", "order_up_to", "lgbm", c, expiry_cap=True)
        r = simulate(pol, inputs("lgbm"), test_lo - cfg.warmup_days, test_hi, test_lo, cfg.shelf_life_days,
                     (cfg.base_lead_time,), (1.0,), cfg.seed)
        frontier.append(dict(family="LightGBM order-up-to + expiry cap", setting=c, **summarise(r)))
    for lo_, hi_ in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5)]:
        pol = Policy(f"min/max {lo_}-{hi_}", "minmax", minmax_days=(lo_, hi_))
        r = simulate(pol, inputs("lgbm"), test_lo - cfg.warmup_days, test_hi, test_lo, cfg.shelf_life_days,
                     (cfg.base_lead_time,), (1.0,), cfg.seed)
        frontier.append(dict(family="Min/max days of cover", setting=f"{lo_}-{hi_}", **summarise(r)))
    pd.DataFrame(frontier).to_csv(REPORTS / "tradeoff_frontier.csv", index=False)

    log("7/7 scenarios: shelf life and late DC deliveries")
    scen = [base_df]
    for sl in cfg.shelf_life_grid:
        if sl != cfg.shelf_life_days:
            scen.append(evaluate(sl, tag=f"shelf_life_{sl}")[0])
    stress_df, _, _ = evaluate(cfg.shelf_life_days, cfg.stress_lead_times, cfg.stress_lead_probs,
                               lt_aware=True, tag="late_deliveries_lt_aware")
    stress_naive, _, _ = evaluate(cfg.shelf_life_days, cfg.stress_lead_times, cfg.stress_lead_probs,
                                  lt_aware=False, tag="late_deliveries_lt_ignored")
    scen_df = pd.concat(scen + [stress_df, stress_naive], ignore_index=True)

    # ---- save ----
    acc.to_csv(REPORTS / "forecast_accuracy.csv", index=False)
    importance.to_csv(REPORTS / "feature_importance.csv", header=["share_of_gain"])
    seg.to_csv(REPORTS / "segments.csv", index=False)
    base_df.to_csv(REPORTS / "policy_results.csv", index=False)
    seg_df.to_csv(REPORTS / "policy_results_by_segment.csv", index=False)
    scen_df.to_csv(REPORTS / "scenario_results.csv", index=False)

    drill_idx = np.argsort(-seg.revenue.to_numpy())[:12]
    drill = {
        "series": [f"{panel.meta.store_id.iloc[i]} / {panel.meta.item_id.iloc[i]}" for i in drill_idx],
        "dates": panel.calendar.date.iloc[test_lo:test_hi].dt.strftime("%Y-%m-%d").tolist(),
        "demand": demand[drill_idx, test_lo:test_hi].round(1).tolist(),
        "forecast": np.round(preds["lgbm"][drill_idx, test_lo - 1:test_hi - 1, 1], 1).tolist(),
        "onhand": {p: r.daily_onhand[drill_idx].round(1).tolist() for p, r in base_res.items()},
    }
    summary = {
        "data": {"source": "M5 Forecasting (Walmart), public", "stores": list(cfg.stores),
                 "category": cfg.category, "n_series": int(N), "n_items": int(panel.meta.item_id.nunique()),
                 "history_days": int(T), "first_day": str(panel.calendar.date.iloc[0].date()),
                 "last_day": str(panel.calendar.date.iloc[-1].date()),
                 "test_days": int(test_hi - test_lo),
                 "test_window": [str(panel.calendar.date.iloc[test_lo].date()),
                                 str(panel.calendar.date.iloc[test_hi - 1].date())],
                 "flagged_stockout_day_share": float(flags.mean()),
                 "series_with_flags_share": float(flags.any(axis=1).mean())},
        "segment_counts": seg.segment.value_counts().to_dict(),
        "tuned_csl": {k: float(v) for k, v in best_csl.items()},
        "assumptions": {"gross_margin": cfg.gross_margin, "holding_rate_annual": cfg.holding_rate_daily * 365,
                        "base_lead_time_days": cfg.base_lead_time, "shelf_life_days": cfg.shelf_life_days,
                        "stress_lead_times": list(cfg.stress_lead_times),
                        "stress_lead_probs": list(cfg.stress_lead_probs)},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    (REPORTS / "drilldown.json").write_text(json.dumps(drill))
    log(f"done in {summary['runtime_sec']} s")
    return summary
