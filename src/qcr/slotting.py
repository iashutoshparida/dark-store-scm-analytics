"""Dark-store slotting: where each SKU sits decides how far a picker walks per order.

Layout (ASSUMPTION, sized like a 2,500-3,000 sq ft dark store):
  8 parallel aisles, 2.4 m apart, each 12 m long, 10 bays of 1.2 m per side, 5 SKU slots per bay side
  -> 800 slots. The pack station sits at the front-left corner (x=0, y=0); a cross-aisle runs along the front.
Routing: S-shape (traversal) - the picker walks every aisle that holds a pick end to end, alternating
direction; if the number of visited aisles is odd, the last one is entered and left from the front.
Orders: built from real M5 daily SKU sales for one store; each day's units are split into baskets
(ASSUMPTION: mean 6 lines per order). Velocity used for slotting comes only from weeks before the
evaluation window, so the test is out of sample.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    n_aisles: int = 8
    aisle_gap_m: float = 2.4
    bays_per_side: int = 10
    bay_len_m: float = 1.2
    slots_per_bay_side: int = 5

    @property
    def aisle_len(self):
        return self.bays_per_side * self.bay_len_m

    def slots(self):
        """Return (aisle, y) for every slot; y is the distance into the aisle to the bay centre."""
        a, y = [], []
        for ai in range(self.n_aisles):
            for b in range(self.bays_per_side):
                for _ in range(2 * self.slots_per_bay_side):
                    a.append(ai); y.append((b + 0.5) * self.bay_len_m)
        return np.array(a), np.array(y)

    @property
    def n_slots(self):
        return self.n_aisles * self.bays_per_side * 2 * self.slots_per_bay_side


def s_shape_distance(aisles: np.ndarray, ys: np.ndarray, lay: Layout) -> float:
    """Walking distance for one order from the pack station and back, S-shape routing."""
    visited = np.unique(aisles)
    k = len(visited)
    x_far = visited.max() * lay.aisle_gap_m
    if k % 2 == 0:
        return k * lay.aisle_len + 2 * x_far
    last = visited.max()
    y_last = ys[aisles == last].max()
    return (k - 1) * lay.aisle_len + 2 * y_last + 2 * x_far


def return_distance(aisles: np.ndarray, ys: np.ndarray, lay: Layout) -> float:
    """Return routing: enter each visited aisle from the front, walk to the deepest pick, come back."""
    visited = np.unique(aisles)
    deep = sum(2 * ys[aisles == v].max() for v in visited)
    return deep + 2 * visited.max() * lay.aisle_gap_m


def best_route_distance(aisles, ys, lay):
    """Picker takes the shorter of S-shape and return routing for each order."""
    return min(s_shape_distance(aisles, ys, lay), return_distance(aisles, ys, lay))


ROUTERS = {"s_shape": s_shape_distance, "return": return_distance, "best_of_two": best_route_distance}


def slot_order(lay: Layout, strategy: str, rng=None) -> np.ndarray:
    """Return slot indices ordered from 'best' to 'worst' for velocity-ranked assignment."""
    a, y = lay.slots()
    if strategy == "across_aisle":          # fill the aisles nearest the pack station first
        return np.lexsort((y, a))
    if strategy == "front_band":            # fill the bays nearest the front cross-aisle first, across aisles
        return np.lexsort((a, y))
    if strategy == "distance":              # rank by straight walking distance from the pack station
        return np.argsort(a * lay.aisle_gap_m + y, kind="stable")
    raise ValueError(strategy)


def assign(skus_ranked: np.ndarray, lay: Layout, strategy: str, dept=None, rng=None):
    """Map each SKU (index) to a slot. Returns aisle and y arrays indexed by SKU."""
    a, y = lay.slots()
    n = len(skus_ranked)
    sku_aisle, sku_y = np.empty(n, int), np.empty(n)
    if strategy == "random":
        slots = rng.permutation(lay.n_slots)[:n]
        sku_aisle[skus_ranked] = a[slots]; sku_y[skus_ranked] = y[slots]
    elif strategy == "category":            # department blocks, alphabetical within department
        order = np.lexsort((np.arange(n), dept))
        slots = np.lexsort((y, a))[:n]
        sku_aisle[order] = a[slots]; sku_y[order] = y[slots]
    else:
        slots = slot_order(lay, strategy)[:n]
        sku_aisle[skus_ranked] = a[slots]; sku_y[skus_ranked] = y[slots]
    return sku_aisle, sku_y


def make_orders(daily_units: np.ndarray, mean_lines: float, rng) -> list:
    """Split each day's SKU units into orders. daily_units: (n_sku, n_days) integers."""
    orders = []
    for d in range(daily_units.shape[1]):
        pool = np.repeat(np.arange(daily_units.shape[0]), daily_units[:, d].astype(int))
        rng.shuffle(pool)
        i = 0
        while i < len(pool):
            size = max(1, rng.poisson(mean_lines))
            lines = np.unique(pool[i:i + size])
            orders.append(lines)
            i += size
    return orders


def evaluate(orders, sku_aisle, sku_y, lay: Layout, walk_mps: float, pick_s_per_line: float,
             routing: str = "s_shape"):
    route = ROUTERS[routing]
    dist = np.array([route(sku_aisle[o], sku_y[o], lay) for o in orders])
    lines = np.array([len(o) for o in orders])
    walk_s = dist / walk_mps
    total_s = walk_s + lines * pick_s_per_line
    return {"orders": len(orders), "avg_lines": lines.mean(), "avg_walk_m": dist.mean(),
            "avg_walk_s": walk_s.mean(), "avg_pick_time_s": total_s.mean(),
            "walk_share_of_time": walk_s.sum() / total_s.sum(),
            "avg_aisles_visited": np.mean([len(np.unique(sku_aisle[o])) for o in orders])}
