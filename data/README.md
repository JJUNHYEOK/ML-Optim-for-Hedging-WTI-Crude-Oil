# Dataset Description (as of 2026-07)

## 1. Preprocessing

- **Source**: WTI daily data
- **Final date range**: 2007-05-14 ~ 2026-02-28
  - Start date: OVX (volatility index) is a placeholder constant (27.09) before this point, so we start at the first normal trading day once the real index begins
  - End date: from 2026-03 onward, daily volatility is ~3x normal, OVX stays at COVID-crisis levels (70-95) for months, and the last few days of the file have prices completely frozen — data quality breaks down, so we cut it off before that
- **Holiday removal**: dropped 200 rows where spot and futures prices are both unchanged from the previous day (market closed)
- **Monthly-indicator lag**: Supply Chain Pressure updates only once a month, unlike the other variables, and in the raw data the new value appears on the last day of the month it describes — using it as-is would leak information that hasn't actually been published yet. Fixed by delaying the finalized month-end value by one month (1,530 rows adjusted; the first 13 rows are missing since there's no prior month to reference yet)
- **File**: `WTI_daily_preprocessed.csv`

## 2. Feature Engineering

**Principle**: since the target is "return 21 trading days (~1 month) ahead," features are also aggregated over 5-day (1 week) / 21-day (1 month) / 63-day (1 quarter) windows. Single-day snapshot values were tested beforehand and had almost no predictive power, so they're not used.

**26 features** total, built from the 6 raw variables + the oil price itself + seasonality:

| feature | what it is | why it's included |
|---|---|---|
| `Spot_mom_5/21/63d` | oil price's own past return | the most basic self-momentum |
| `OVX_mean_5/21/63d` | average volatility level | how anxious the market currently is. Highest correlation with the target (0.20-0.21) |
| `OVX_chg_5/21/63d` | change in volatility | whether volatility is rising |
| `DXY_chg_5/21/63d` | change in the dollar index | oil is dollar-denominated → dollar strength and oil weakness tend to move inversely |
| `Basis_chg_5/21/63d` | change in the (spot-futures) spread | contango/backwardation shifts signal supply-demand imbalance |
| `Yield_Curve_chg_5/21/63d` | change in the yield spread | recession signal → indirectly tied to oil demand outlook |
| `GPR_mean_5/21/63d` | average geopolitical risk level | reflects supply-shock risk |
| `SCP_mean_5/21/63d` | average supply-chain pressure level | reflects supply bottleneck severity |
| `month_sin`, `month_cos` | month encoded on a circle | captures seasonality (e.g. driving season) |

**Note**: the 5/21/63-day windows of the same variable are highly correlated with each other (SCP 0.93-0.99, OVX 0.75-0.94, GPR 0.83-0.84) — they're not fully independent information, and Lasso-family models are expected to prune this redundancy automatically.

**File**: `WTI_daily_features.csv`

## 3. Train / Val / Test

- **Ratio**: 65 / 15 / 20
- **63-trading-day gap at each split boundary** — features look back up to 63 days and the target looks forward 21 days, so cutting without a gap would leak information across the boundary in both directions

| split | period | rows | share of crisis-regime rows |
|---|---|---|---|
| train | 2007-08 ~ 2019-08 | 3,012 (65%) | 10.1% |
| val | 2019-11 ~ 2022-08 | 695 (15%) | 25.0% |
| test | 2022-11 ~ 2026-01 | 801 (17%) | 1.9% |

**Note**: the test period has almost no high-volatility (crisis) cases (1.9%) — crisis-regime performance is hard to judge from test alone, so check val too.

**Metrics**: direction accuracy + RMSE. Both must **always be compared against the random-walk baseline** (numbers on their own don't mean anything).

**File**: `WTI_daily_split.csv` (the final version — the two files above plus a `split` column)

For how this dataset was built, see `src/feature_engineering_daily.py` and `src/split_daily.py`. The script that did the raw preprocessing itself (holiday removal, SCP lag) only exists in the original development repo — it's not included here since the source file it needs to re-run isn't present in this repo.
