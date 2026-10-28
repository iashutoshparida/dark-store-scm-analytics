import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qcr.censoring import flag_probable_stockouts  # noqa: E402
from qcr.forecast import make_folds  # noqa: E402
from qcr.config import CFG  # noqa: E402
from qcr.simulate import Policy, SimInputs, simulate  # noqa: E402
from qcr.slotting import Layout, return_distance, s_shape_distance  # noqa: E402


def _inputs(demand, forecast):
    N, T = demand.shape
    preds = {"lgbm": np.zeros((N, T, 4), np.float32)}
    preds["lgbm"][:, :, 1:] = forecast
    return SimInputs(demand=demand, price=np.ones((N, T)), preds=preds,
                     sigma={p: np.zeros((N, T)) for p in (1, 2, 3)},
                     mean_daily=np.full((N, T), forecast), gross_margin=0.3, holding_rate_daily=0.0)


def test_stockout_run_flagged_only_for_fast_movers():
    fast = np.r_[np.full(40, 5.0), np.zeros(4), np.full(10, 5.0)]
    slow = np.r_[np.full(40, 0.05), np.zeros(4), np.full(10, 0.05)]
    flags = flag_probable_stockouts(np.vstack([fast, slow]))
    assert flags[0, 40:44].all() and not flags[0, :40].any()
    assert not flags[1].any()


def test_perfect_forecast_gives_full_fill_and_no_waste():
    demand = np.full((2, 40), 3.0)
    res = simulate(Policy("p", "order_up_to", csl=0.5), _inputs(demand, 3.0), 1, 40, 5, shelf_life=5)
    s = res.per_series
    assert np.allclose(s["lost"], 0) and np.allclose(s["waste"], 0)


def test_overstock_expires_fifo():
    demand = np.zeros((1, 30)); demand[0, :] = 1.0
    # forecast 4x demand -> stock piles up and must expire, never go negative
    res = simulate(Policy("p", "order_up_to", csl=0.5), _inputs(demand, 4.0), 1, 30, 1, shelf_life=3)
    assert res.per_series["waste"][0] > 0
    assert (res.daily_onhand >= 0).all()


def test_folds_are_ordered_and_test_is_last():
    folds = make_folds(730, CFG)
    assert folds.starts == sorted(folds.starts)
    assert folds.starts[-1] + folds.length == 730
    assert folds.test_start == folds.starts[CFG.n_calib_folds]


def test_routing_distances():
    lay = Layout()
    a = np.array([0, 0]); y = np.array([1.0, 3.0])
    assert return_distance(a, y, lay) == 6.0                 # one aisle, 3 m deep, out and back
    assert s_shape_distance(a, y, lay) == 6.0                # odd count -> same as return for one aisle
    a2 = np.array([0, 1]); y2 = np.array([1.0, 1.0])
    assert s_shape_distance(a2, y2, lay) == 2 * lay.aisle_len + 2 * lay.aisle_gap_m
