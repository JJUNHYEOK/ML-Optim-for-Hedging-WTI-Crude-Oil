# ML-Optim for Hedging WTI Crude Oil

Decision-focused learning for WTI crude oil hedge ratios — learning the hedge
ratio directly from realized procurement cost, instead of predicting prices
and optimizing afterward.

## Overview

Companies exposed to WTI crude oil price risk decide, every period, what
fraction `q` of next period's exposure to lock in via futures. The
conventional approach is two-stage: forecast the price, then optimize the
hedge given the forecast. This project instead learns `q(x)` by directly
minimizing the realized hedging cost — a **newsvendor-style loss** — so the
model is trained on the objective that actually matters, not on forecast
accuracy as a proxy.

```
C(q; D) = b * max(D - q, 0) + h * max(q - D, 0)
```

| Symbol | Meaning |
|---|---|
| `q` | hedge ratio locked in via futures (model output) |
| `D` | ex-post optimal hedge ratio (label, derived from realized prices) |
| `b` | underage cost — spot rallies, unhedged portion gets bought at a premium |
| `h` | overage cost — spot falls, capital tied up in futures + transaction cost |

Target: `min_{q(.)} E_D[ C(q(x); D) | x ]`, approximated by empirical risk
over historical `(x, D)` pairs.

## Background

Purdue summer research project (student team across 8 Korean universities),
run independently without a faculty advisor. Target venues: INFORMS
workshops, NeurIPS OR workshop, applied OR journals.

## Repository Structure

```
code/
├── data/
│   ├── WTI_preprocessed.csv          # main dataset, monthly, 230 rows (2007-05 ~ 2026-06)
│   ├── WTI_daily_preprocessed.csv    # daily dataset, 5606 rows
│   └── WTI_features.csv              # engineered features + targets (generated)
├── eda/                              # EDA figures (time series, distributions,
│                                      # stationarity, correlation, regime analysis)
├── src/
│   └── feature_engineering.py        # builds x (features) and D (targets)
├── models/                           # (planned) baseline + ERM + SPO+ implementations
├── CLAUDE.md                         # Claude Code project instructions
└── PROJECT_CONTEXT.md                # full project design context
```

## Data

Monthly WTI data, 2007-05 to 2026-06 (230 rows), assembled from spot/futures
prices and macro/risk indicators:

| Column | Meaning |
|---|---|
| `WTI_Spot` | WTI spot price (USD/bbl) |
| `WTI_Futures` | WTI front-month futures price (USD/bbl) |
| `Basis` | `WTI_Spot - WTI_Futures` |
| `OVX` | crude oil volatility index |
| `DXY` | US Dollar Index |
| `Yield_Curve` | 10Y-2Y treasury spread |
| `GPR` | geopolitical risk index |
| `SCP` | supply chain pressure index |

Preprocessing: weekends dropped, the 2020-04-20 negative-price outlier
removed, pre-OVX-launch history (before 2007-05) excluded, early fixed
futures values interpolated, aggregated to monthly.

### EDA highlights

- Stationary at level: `WTI_Spot`, `OVX`, `SCP`. Non-stationary (need
  differencing): `Basis`, `GPR`, `Yield_Curve`, `DXY`.
- Raw-level `DXY`/`Yield_Curve` are highly collinear (Pearson -0.80); this
  drops to -0.01 after differencing — another reason to model on
  differenced series only.
- `Basis` has a heavy right tail (skew 4.21, kurtosis 26.82); the largest
  backwardation spike in the series sits in the most recent (test-side)
  months — watch for train/test leakage of this outlier.
- `SCP` shows no significant regime effect and is excluded; `GPR` is weak
  and kept only as an ablation candidate.

Full detail in `PROJECT_CONTEXT.md`.

## Feature Engineering (`src/feature_engineering.py`)

- Zero-crossing variables (`Basis`, `Yield_Curve`, `GPR`) → first difference.
- Strictly positive variables (`OVX`, `DXY`) → log difference.
- 1–3 month lags of each differenced signal.
- `WTI_Spot` own momentum: 1/3/6-month log returns, 3/6-month realized vol.
- Seasonality via `sin`/`cos` of month (2 dims instead of 11 dummies, given
  n ≈ 230).
- Targets: `D_binary = 1[S_{t+1} > F_t]` (corner solution of the linear
  hedging cost, used as the main label) and `D_continuous` (trailing-vol
  normalized price gap, used as a robustness check).

Run with:

```bash
python3 src/feature_engineering.py
```

## Modeling Plan

| Stage | Method | Loss |
|---|---|---|
| (a) Baseline | LightGBM point forecast → derive `q` | MSE |
| (b) Baseline | LightGBM quantile regression at `b/(b+h)` | pinball |
| (c) Main | ERM (LP), Ban & Rudin (2019) via `cvxpy` | newsvendor cost, direct |
| (d) Extension | SPO+ via `PyEPO` | SPO+ loss |

Expected result: (a) < (b) < (c) in realized hedging cost.

### Cost parameters

```python
h = 0.0067   # monthly capital cost (8% annual / 12)
b = 4 * h    # underage costs more than overage
# critical fractile b/(b+h) = 0.8
```
Sensitivity analysis planned over `b/h` in [1, 10].

## Setup

```bash
pip install cvxpy lightgbm pyepo scikit-learn pandas numpy matplotlib seaborn statsmodels
```

## Status

- [x] EDA (stationarity, correlation, distribution, regime analysis)
- [x] Feature engineering (`src/feature_engineering.py`)
- [ ] Train/val/test split (time-ordered)
- [ ] Baseline (a): LightGBM point forecast
- [ ] Baseline (b): LightGBM quantile regression
- [ ] Main model (c): ERM (LP) via cvxpy
- [ ] Extension (d): SPO+ via PyEPO
- [ ] Cost comparison across (a)/(b)/(c)/(d)
- [ ] b/h sensitivity analysis
- [ ] D definition robustness check (binary vs. continuous)

## References

- Ban, G.-Y., & Rudin, C. (2019). *The Big Data Newsvendor: Practical Insights from Machine Learning*. Operations Research.
- Elmachtoub, A. N., & Grigas, P. (2022). *Smart "Predict, then Optimize"*. Management Science.
- Conformal prediction for the newsvendor problem, arXiv:2412.13159.
- World Bank Commodity Markets Outlook — forecasting limitations for commodity prices.

*Developed by **Pervengers** during the Purdue Academy for Global Engineering(PAGE) 2026*
