# Dark-Store Supply Chain Analytics: Replenishment and Slotting

Two connected studies of how a quick-commerce grocery store should be stocked and laid out. Both run on public
data (the M5 Walmart dataset) with every business assumption stated in one config file. They answer two
questions a dark-store operations team faces every week:

1. **Replenishment.** How much should each store order from the DC each night so shelves stay full without
   fresh stock expiring?
2. **Slotting.** Where should each SKU sit so pickers walk less per order?

Nothing here uses any company's internal data. All results are **simulated outcomes on historical public demand**,
not measured results from a live store.

---

## Data

| Item | Detail |
|---|---|
| Source | M5 Forecasting dataset (Walmart, public), files `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` |
| Download used | `https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip` (SHA-256 `cc704ba1...e450eed774`) |
| Scope, project 1 | FOODS category, stores CA_1, CA_2, CA_3; SKU-stores selling at least 1 unit/day in the final year: **2,369 series, 992 items** |
| History | 730 days, 2014-05-24 to 2016-05-22 |
| Scope, project 2 | FOODS, store CA_1, top 800 SKUs by recent velocity (store capacity) |

Place the unzipped CSVs in `data/raw/`.

---

## Project 1: Perishable replenishment engine (DC to store)

### Problem
A dark store re-orders nightly from its DC. A simple rule like "keep 2 to 4 days of cover" is easy to run but
over-stocks slow movers, and for fresh food the extra stock expires. The aim is a policy that keeps availability
high while cutting expiry waste and stock on hand.

### Method
1. **Censored demand.** Recorded sales hit zero when a shelf is empty, so sales understate demand. A Poisson run
   test flags zero-sale runs that are too unlikely for a SKU's recent rate (p < 0.001). 11.0% of SKU-store-days
   were flagged. These days are hidden from model features and never used as training targets.
2. **ABC/XYZ segmentation.** ABC by revenue within each store (80/95% cut-offs), XYZ by the coefficient of
   variation of weekly units. Computed on pre-test data only.
3. **Forecasting.** One global LightGBM model (Tweedie loss) using lags, rolling means, weekday, price and events,
   against three baselines: same weekday last week, 28-day moving average, and the average of the same weekday
   over 4 weeks. Rolling-origin backtest with 5 folds of 28 days. Fold 1 is used only for calibration and
   folds 2 to 5 (112 days) are the test. Every forecast for day t+h uses data up to day t only.
4. **Policies compared** in a daily FIFO inventory simulation with lost sales and expiry:
   - Min/max rule: reorder up to 4 days of cover when stock falls below 2 days
   - Order-up-to with a 95% cycle service level, using the 28-day average forecast
   - Order-up-to with a 95% cycle service level, using the LightGBM forecast
   - **Proposed:** LightGBM, with the service level for each ABC/XYZ segment tuned on the calibration fold to
     maximise contribution subject to fill-rate floors (A 97%, B 95%, C 90%), plus an expiry cap. The cap stops
     the store holding more than it can sell within shelf life.
   Safety stock uses the standard deviation of each series' own out-of-sample forecast errors from earlier folds.
5. **Scenarios.** Shelf life of 3, 5, 7 and 14 days. Late DC deliveries, where 40% of orders arrive 1 to 2 days
   late, with and without lead-time variance in the safety stock.

### Results (112-day out-of-sample test)

**Forecast accuracy, 1 day ahead, on non-stockout days**

| Model | WAPE | Bias |
|---|---|---|
| Seasonal naive (same weekday last week) | 66.7% | -0.1% |
| Same weekday, 4-week average | 56.2% | -0.7% |
| 28-day moving average | 54.1% | -0.5% |
| **LightGBM (global)** | **50.6%** | **-0.2%** |

LightGBM has the lowest error in 8 of the 9 ABC/XYZ segments and ties the 28-day average in CX (77.3%). For A items its WAPE is 39.5 to 44.0%,
and the hardest segment, CY, reaches 80%. High WAPE is normal for daily SKU-store grocery data.

**Inventory policies, 5-day shelf life, next-morning DC delivery**

| Policy | Fill rate | Waste (% of units sold) | Days of cover | Simulated contribution vs min/max |
|---|---|---|---|---|
| Min/max 2-4 days | 98.1% | 10.9% | 2.36 | baseline |
| Order-up-to, MA-28, 95% | 96.3% | 4.0% | 1.33 | +20.3% |
| Order-up-to, LightGBM, 95% | 96.8% | 4.1% | 1.28 | +20.9% |
| Proposed (segment CSL + expiry cap) | 96.4% | 3.5% | 1.21 | +21.8% |

**Matched-service comparison** from the trade-off sweep (`reports/tradeoff_frontier.csv`): at essentially the same fill
rate (98.2% vs 98.1%), the LightGBM order-up-to policy at a 99% service level, compared with the 2-4 day
min/max rule:
- cuts expiry waste from **10.9% to 5.8% of units sold** (-47%)
- cuts stock cover from **2.36 to 1.66 days** (-29%)
- raises simulated contribution by **17%**

**Late DC deliveries.** When 40% of orders arrive 1 to 2 days late, safety stock that ignores lead-time variance
drops the fill rate of the LightGBM 95% policy from 96.8% to **84.3%**. Adding lead-time variance restores it to
**98.0%**.

**Shelf life matters most.** At a 3-day shelf life the min/max rule wastes 48% of units sold and simulated
contribution turns negative. The proposed policy wastes 13.1%.

**What the tuning found.** With a 30% gross margin, the cost of an expired unit (70% of price) is larger than the
margin lost on a missed sale. The cost-optimal service level is therefore below the usual 95% for most segments.
The fill-rate floors are what keep A items available.

---

## Project 2: Dark-store slotting and pick-path study

### Problem
In a dark store, the time to pick an order is walking plus picking. Placing fast-moving SKUs well cuts walking,
but the best placement depends on how pickers route through the aisles.

### Method
- **Layout (assumption):** 8 aisles, 12 m long and 2.4 m apart, with 800 slots and the pack station at the
  front corner.
- **Orders:** 28 days of real CA_1 SKU sales, split into baskets averaging 6 lines. This gives 12,712 orders per
  run, averaged over 3 random seeds.
- **Velocity:** slotting uses the 84 days before the test window, so the test is out of sample. The top 20% of
  SKUs carry 54.5% of units.
- **Slotting strategies:** random, category blocks (department then A-Z), and velocity-based in three geometries:
  nearest aisles first, a front band across aisles, and by walking distance to the pack station. The last
  strategy is also run with 6-month-old velocity data.
- **Routing:** S-shape (walk every visited aisle end to end) and best-of-two, where the picker takes the shorter of
  S-shape and return routing for each order.
- **Timing (assumptions):** walking at 1.0 m/s, and 12 s per line, the low end of the 12-15 s per pick reported
  for Indian quick-commerce stores.

### Results (vs category blocks with S-shape routing)

| Slotting | Routing | Walk per order | Change | Pick time per order |
|---|---|---|---|---|
| Category blocks | S-shape | 71.7 m | baseline | 142.8 s |
| Random | S-shape | 79.8 m | +11.3% | 150.9 s |
| Velocity, nearest aisles first | S-shape | 64.4 m | -10.2% | 135.5 s |
| Velocity, front band | S-shape | 75.9 m | +5.8% | 147.0 s |
| Velocity, front band | best-of-two | 60.3 m | -15.9% | 131.4 s |
| **Velocity, by distance** | **best-of-two** | **59.6 m** | **-16.9%** | **130.7 s (-8.5%)** |
| Velocity by distance, 6-month-old data | best-of-two | 63.3 m | -11.8% | 134.4 s |

Takeaways:
- **Slotting and routing have to be chosen together.** A front-band layout is worse than category blocks under
  S-shape routing, but nearly the best once pickers can turn back mid-aisle.
- **Velocity goes stale.** Six-month-old data loses about 30% of the walking saving, so re-slot at least quarterly.
- **Walking is about half of pick time,** so a 17% cut in walking saves roughly 12 s per order.

---

## Reproduce

```bash
pip install pandas numpy scipy lightgbm pytest
python run_pipeline.py      # project 1, about 4 minutes, writes reports/*.csv and summary.json
python run_slotting.py      # project 2, about 10 seconds
python -m pytest -q tests   # 5 tests: stockout flags, FIFO expiry, fold order, routing distances
```

## Repository layout

```
src/qcr/config.py        all settings and stated assumptions
src/qcr/data.py          M5 loading into SKU-store x day matrices
src/qcr/censoring.py     probable-stockout detection and imputation
src/qcr/segmentation.py  ABC/XYZ
src/qcr/forecast.py      baselines, LightGBM, rolling-origin backtest, accuracy tables
src/qcr/simulate.py      FIFO perishable inventory simulation and policies
src/qcr/pipeline.py      project 1 end to end
src/qcr/slotting.py      layout, slotting strategies, S-shape and return routing
run_pipeline.py, run_slotting.py
reports/                 all result tables
tests/
```

## Assumptions and limitations

- **Public data, not FirstClub data.** M5 is US Walmart data from 2014-16, not Indian quick commerce.
  Store behaviour, basket sizes and SKU ranges differ.
- **Shelf life is assumed.** M5 has no shelf-life field, so the same shelf life is applied to every item and
  varied as a scenario.
- **Stockout detection is a statistical guess.** M5 has no stock-status field. A dataset with real stock flags,
  such as FreshRetailNet-50K, would allow proper latent-demand recovery.
- **Costs are assumed:** 30% margin, 18% annual holding cost, and no penalty beyond lost margin for a stockout.
  Contribution figures are for comparing policies, not forecasts of profit.
- **The layout and order baskets in project 2 are modelled,** not measured, and only one pick per line is
  assumed.
- **Next steps:** run both projects on real pick logs and stock data; add a DC capacity constraint and vendor
  minimum order quantities; add batch picking and test re-slotting frequency directly.
