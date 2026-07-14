# Deep Learning Forecasting Experiments (2026-07, `WTI_forecasting_experiments.ipynb`)

## What this is

A separate, exploratory experiment from the official ML track defined in
`TASKS.md` (은지: RW+Ridge / 혜원: Lasso+ElasticNet / 준혁: LightGBM — all on
21-trading-day `fwd_ret`, purge-gap split, evaluated against a random-walk
baseline via direction accuracy). This notebook instead trains LSTM /
Transformer / PatchTST to forecast WTI spot price **30 days ahead, directly
(multi-step)**. Data, target, and validation setup differ from the official
track (see "Differences from the official track" below). Results below were
produced by a single run on GPU (`cuda`, torch 2.3.1) and are saved in the
notebook's cell outputs.

## Experimental design

2×2 (target transform × variable set) × 3 models = **12 configurations**,
each trained with 10-fold walk-forward CV.

|              | Multivariate (target + 6 exogenous vars) | Univariate (target only) |
|--------------|-----|-----|
| **Raw price**    | A_raw_multi      | B_raw_univ      |
| **Log-return**   | C_logret_multi   | D_logret_univ   |

- **Data**: `WTI_data.csv`, 9,678 rows, 2000-01-01 to 2026-06-30 (a different source, date range, and row count than the official track's `WTI_daily_preprocessed.csv` / `data/v2/`)
- **Target**: raw experiments use the price level directly; log-return experiments use `r_t = log(p_t+50) - log(p_{t-1}+50)` (the +50 shift guards against the one negative-price day, WTI at -$36.98 on 2020-04-20)
- **6 exogenous variables**: Supply_Chain_Pressure, Geopolitical_Risk_GPR, Oil_Volatility_OVX, Yield_Curve_10Y2Y, WTI_Futures, US_Dollar_Index
- **Horizon/lookback**: 30-day forecast, 96-day lookback input sequence
- **Validation**: 10-fold walk-forward CV (30 days per fold), final model retrained using the mean of each fold's best epoch, then evaluated once on the last 30 days (2026-06-01 to 2026-06-30) as a single held-out test window
- **Models**: LSTM (2-layer), Transformer encoder, PatchTST (patch_len=16, with RevIN)
- **Metrics**: RMSE/MAE/MAPE on the original price scale ($) — CV loss is reported separately as scaled MSE

## Results

### Test RMSE (final 30-day window, $, lower is better)

| model | A_raw_multi | B_raw_univ | C_logret_multi | D_logret_univ |
|---|---|---|---|---|
| lstm | 16.16 | **8.37** | 11.68 | 8.89 |
| transformer | 20.31 | 10.04 | **8.78** | 9.17 |
| patchtst | 15.94 | 15.21 | 13.57 | 13.77 |

### CV mean price RMSE (10-fold average, $ — a stability indicator for each configuration)

| model | A_raw_multi | B_raw_univ | C_logret_multi | D_logret_univ |
|---|---|---|---|---|
| lstm | 5.37 ± 7.31 | 5.00 ± 6.48 | 5.63 ± 6.94 | 5.57 ± 6.60 |
| transformer | 6.03 ± 7.88 | 4.83 ± 6.57 | 5.16 ± 6.59 | 5.35 ± 6.61 |
| patchtst | 5.19 ± 7.00 | 5.19 ± 7.08 | 7.16 ± 7.46 | 7.21 ± 7.53 |

The notebook also writes the full per-fold table (fold RMSE, best epoch,
MAE/MAPE) to `experiments_summary.csv` / `experiments_meta.json` when run —
those files aren't checked into the repo currently.

## Observations

1. **Univariate beats multivariate almost across the board** (LSTM: 8.37 vs 16.16, Transformer: 10.04 vs 20.31, PatchTST: 15.21 vs 15.94) — adding the 6 exogenous variables hurt rather than helped in this 30-day direct-forecast setup. With a single scalar target and a 96-step lookback, the sequence models likely didn't have enough signal-per-parameter and overfit on the added channels.
2. **The effect of the log-return transform is model-dependent**: Transformer improves dramatically going from raw+multivariate (A, 20.31) to log-return+multivariate (C, 8.78) — consistent with log-return mitigating the non-stationarity of the raw price series. LSTM shows the opposite within univariate: raw (B, 8.37) slightly beats log-return (D, 8.89).
3. **The two best configurations are close, not a single clear winner**: B_raw_univ+LSTM (8.37) and C_logret_multi+Transformer (8.78) — suggesting two viable paths ("univariate + LSTM" or "log-return + multivariate + Transformer") rather than one dominant setup.
4. **PatchTST is the weakest of the three models** — worst or tied-worst test RMSE in all 4 experiments. The lookback=96 / patch_len=16 configuration may simply be over-provisioned for this data size.
5. **Large gap between CV RMSE (5-7) and test RMSE (8.4-20.3)** — all configurations look similarly good under 10-fold CV, but diverge sharply on the single final 30-day test window (June 2026). Since test is effectively a sample size of one, good CV performance doesn't guarantee good test performance here.
6. **High fold-to-fold variance** (standard deviation is often comparable to or larger than the mean, e.g. LSTM A_raw_multi 5.37±7.31) — best epoch also swings from 1 to 38 across folds, with early stopping frequently triggering within just a few epochs, suggesting unstable training.

## Limitations / unfinished pieces

- **Optuna hyperparameter tuning (notebook §10, cells 31-35) is coded but not executed** — no outputs recorded. It's set up to tune 20 trials for one chosen `TARGET_EXPERIMENT`, but hasn't been run yet.
- **No direct comparison against a random-walk baseline** — unlike the official track (`models/baseline_randomwalk.py`), this notebook doesn't compute the RW+drift improvement. Raw $ RMSE alone doesn't establish whether these results represent a real signal.
- **No direction-accuracy (dir_acc) metric** — the official track's primary evaluation metric isn't computed here; only RMSE/MAE/MAPE.
- **Single seed, single test window** — no reproducibility/luck check (different seeds, different test periods).

## Differences from the official track (caution)

| | Official track (`TASKS.md`, `models/`) | This notebook (`ml/`) |
|---|---|---|
| Data | `data/v2/` (2007-05-14 to 2026-02-28, holiday-filtered, outliers removed) | `WTI_data.csv` (2000-01-01 to 2026-06-30) |
| Target | `fwd_ret` = 21-trading-day-ahead log-return | raw price or log-return, 30-day direct horizon |
| Split | 63-trading-day purge gap train/val/test | 10-fold walk-forward CV + single final 30-day test |
| Baseline | Random walk, compared via dir_acc/RMSE | none (only $ RMSE) |
| Models | RW/Ridge/Lasso/ElasticNet/LightGBM | LSTM/Transformer/PatchTST |

**These results should not be directly compared to or substituted for the
"main results" in CLAUDE.md** (e.g. Ridge dir_acc 60.5% vs RW 54.4%). Treat
this as a separate, exploratory deep-learning experiment — folding it into
the official conclusions would require, at minimum, adding an RW baseline
comparison and a dir_acc metric.
