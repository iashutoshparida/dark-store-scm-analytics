"""Store-level inventory simulation for perishable SKUs with daily DC-to-store replenishment.

Daily sequence for every SKU-store, all series vectorised:
  1. morning: DC delivery arrives (fresh stock, full shelf life)
  2. day: customers buy; stock is issued oldest-first (FIFO); unmet demand is lost
  3. night: stock at the end of its shelf life is written off; holding cost is charged on what is left
  4. night: the store raises its replenishment order using the chosen policy

Demand fed to the simulation is observed sales, with probable-stockout days replaced by the
trailing mean (see censoring.py). Results are simulated outcomes on historical demand, not
measured results from any live store.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class Policy:
    name: str
    kind: str                          # "minmax" or "order_up_to"
    forecast: str = "lgbm"             # which forecast feeds the order-up-to level
    csl: object = 0.95                 # scalar or per-series array of cycle service levels
    expiry_cap: bool = False           # cap the order-up-to level by sellable demand within shelf life
    lt_aware: bool = False             # add lead-time variance to safety stock
    minmax_days: tuple = (2.0, 4.0)


@dataclass
class SimInputs:
    demand: np.ndarray                 # (N, T) demand used by the simulator
    price: np.ndarray                  # (N, T)
    preds: dict                        # model -> (N, T, H+1)
    sigma: dict                        # P -> (N, T) std of P-day forecast error known at origin t
    mean_daily: np.ndarray             # (N, T) trailing 28-day mean known at origin t
    gross_margin: float
    holding_rate_daily: float


@dataclass
class SimResult:
    per_series: dict = field(default_factory=dict)   # metric -> (N,) totals over the scored window
    daily_onhand: np.ndarray = None                   # (N, days) end-of-day stock for drill-down


def _order_up_to(pol: Policy, inp: SimInputs, t: int, P: int, var_l: float, shelf_life: int, mean_l: float):
    f = inp.preds[pol.forecast]
    fP = np.nansum(f[:, t, 1:P + 1], axis=1)
    sig = inp.sigma[P][:, t]
    if pol.lt_aware and var_l > 0:
        sig = np.sqrt(sig ** 2 + (inp.mean_daily[:, t] ** 2) * var_l)
    z = norm.ppf(np.asarray(pol.csl))
    S = fP + z * sig
    if pol.expiry_cap:
        cap = inp.mean_daily[:, t] * (mean_l + shelf_life - 1)
        S = np.minimum(S, np.maximum(cap, fP))
    return np.maximum(S, 0)


def simulate(pol: Policy, inp: SimInputs, start: int, end: int, score_from: int, shelf_life: int,
             lead_times=(1,), lead_probs=(1.0,), seed: int = 7) -> SimResult:
    rng = np.random.default_rng(seed)
    N = inp.demand.shape[0]
    lead_times = np.asarray(lead_times); lead_probs = np.asarray(lead_probs)
    mean_l = float((lead_times * lead_probs).sum())
    var_l = float(((lead_times - mean_l) ** 2 * lead_probs).sum())
    P = int(np.clip(np.ceil(mean_l), 1, 3)) if pol.lt_aware else int(lead_times.min())
    max_l = int(lead_times.max())

    inv = np.zeros((N, shelf_life))            # inv[:, a] = units with a+1 days of life left
    pipe = np.zeros((N, end - start + max_l + 2))
    # opening stock: one target level of fresh stock
    if pol.kind == "minmax":
        inv[:, -1] = np.ceil(inp.mean_daily[:, start - 1] * pol.minmax_days[1])
    else:
        inv[:, -1] = np.ceil(_order_up_to(pol, inp, start - 1, P, var_l, shelf_life, mean_l))

    keys = ("demand", "sales", "lost", "waste", "received", "onhand_sum", "holding", "margin",
            "waste_cost", "stockout_days", "demand_days", "orders")
    tot = {k: np.zeros(N) for k in keys}
    onhand_track = []
    for t in range(start, end):
        k = t - start
        inv[:, -1] += pipe[:, k]
        d = inp.demand[:, t]
        need = d.copy()
        for a in range(shelf_life):
            take = np.minimum(inv[:, a], need)
            inv[:, a] -= take
            need -= take
        sold = d - need
        waste = inv[:, 0].copy()
        inv[:, :-1] = inv[:, 1:]; inv[:, -1] = 0
        onhand = inv.sum(axis=1)
        price = inp.price[:, t]
        cost = price * (1 - inp.gross_margin)

        # ---- order at night ----
        ip = onhand + pipe[:, k + 1:].sum(axis=1)
        if pol.kind == "minmax":
            lo, hi = pol.minmax_days
            s_, S_ = inp.mean_daily[:, t] * lo, inp.mean_daily[:, t] * hi
            q = np.where(ip <= s_, np.ceil(S_ - ip), 0.0)
        else:
            S = _order_up_to(pol, inp, t, P, var_l, shelf_life, mean_l)
            q = np.ceil(np.maximum(S - ip, 0))
        L = rng.choice(lead_times, size=N, p=lead_probs) if len(lead_times) > 1 else np.full(N, lead_times[0])
        np.add.at(pipe, (np.arange(N), k + L), q)

        if t >= score_from:
            tot["demand"] += d; tot["sales"] += sold; tot["lost"] += need; tot["waste"] += waste
            tot["received"] += pipe[:, k]
            tot["onhand_sum"] += onhand
            tot["holding"] += onhand * cost * inp.holding_rate_daily
            tot["margin"] += sold * price * inp.gross_margin
            tot["waste_cost"] += waste * cost
            tot["stockout_days"] += (need > 1e-9) & (d > 0)
            tot["demand_days"] += d > 0
            tot["orders"] += q > 0
            onhand_track.append(onhand)
    tot["contribution"] = tot["margin"] - tot["waste_cost"] - tot["holding"]
    tot["days"] = np.full(N, end - score_from)
    return SimResult(per_series=tot, daily_onhand=np.stack(onhand_track, axis=1))


def summarise(res: SimResult, mask=None) -> dict:
    s = res.per_series
    m = np.ones_like(s["demand"], bool) if mask is None else mask
    g = {k: v[m].sum() for k, v in s.items()}
    days = s["days"][0]
    avg_onhand = g["onhand_sum"] / days
    return {
        "fill_rate": g["sales"] / g["demand"],
        "stockout_day_rate": g["stockout_days"] / g["demand_days"],
        "waste_pct_of_sales": g["waste"] / g["sales"],
        "days_of_cover": avg_onhand / (g["demand"] / days),
        "annual_turns": (g["sales"] / days * 365) / max(avg_onhand, 1e-9),
        "contribution": g["contribution"],
        "lost_units": g["lost"],
        "waste_units": g["waste"],
        "avg_onhand_units": avg_onhand,
    }
