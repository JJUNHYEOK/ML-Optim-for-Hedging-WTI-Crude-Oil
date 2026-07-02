# ML-Optim for Hedging WTI Crude Oil

Connecting machine learning to WTI crude oil hedging decisions. Started as
Decision-Focused Learning (learning the hedge ratio directly from realized
cost), pivoted to price forecasting after confirming no hedge-direction
signal existed in that framing, and is now connecting the forecasting
signal back into the hedge decision.

## Current Status

**Phase 1 — Decision-Focused Learning (concluded, kept as background).**
Tried learning the hedge ratio `q` directly from a newsvendor cost function
(Ban & Rudin 2019) across six different approaches. All converged to the
same conclusion: no exploitable hedge-direction signal in this feature set.
Code kept, not deleted — it's the evidence base for why the project
pivoted.

**Phase 2 — Price forecasting (current main result, confirmed).**
Reframed the same features as a standard forecasting problem: predict WTI
spot log-return 21 trading days ahead. Regularized linear models
(Ridge/Lasso/ElasticNet) beat the random-walk baseline on direction
accuracy (61-64%), holding up across 5-fold walk-forward validation.
LightGBM overfits (good on train/val, degrades on test) — the simpler
linear models generalize better.

Dataset details: [data/README.md](data/README.md).
Current task split across the team: [TASKS.md](TASKS.md).

**Phase 3 — Connecting the signal to the hedge decision (confirmed result).**
Feeding the Phase 2 signal back into the original newsvendor framing failed
across five different methods — the asymmetric cost structure demands a
confidence level the signal never reaches (proven, not just observed).
Switching the objective to the standard finance hedging framework (minimum-
variance hedge, Ederington 1979) and adding the forecast as an
**independent overlay position** (instead of adjusting the hedge ratio
itself) gets a real, statistically significant reduction in residual
variance (basis risk).

## What's Next

- [ ] Re-validate the Phase 3 overlay strategy with walk-forward splits (currently a single split)
- [ ] Evaluate the overlay strategy on Sharpe ratio / realized P&L, not just variance reduction
- [ ] Strengthen the significance test for the direction-accuracy edge (HAC / Diebold-Mariano)
- [ ] Write up the final report

## Repository Structure

```
data/
├── README.md                    # Phase 2/3 dataset details (preprocessing, features, split)
├── WTI_daily_preprocessed.csv   # cleaned daily raw data (2007-05-14 ~ 2026-02-28)
├── WTI_daily_features.csv       # feature-engineered (26 features)
└── WTI_daily_split.csv          # final version with train/val/test labels
src/
├── feature_engineering_daily.py # builds the 26 features + target (fwd_ret)
└── split_daily.py               # purge-gap (63 trading days) train/val/test split
TASKS.md                          # current (Phase 2/3) task assignments
```

## Setup

```bash
pip install cvxpy lightgbm scikit-learn pandas numpy scipy matplotlib seaborn statsmodels
```

## Background

Purdue summer research project (student team across 8 Korean universities),
run independently without a faculty advisor. Target venues: INFORMS
workshops, NeurIPS OR workshop, applied OR journals.

## References

- Ban, G.-Y., & Rudin, C. (2019). *The Big Data Newsvendor: Practical Insights from Machine Learning*. Operations Research.
- Ederington, L. H. (1979). *The Hedging Performance of the New Futures Markets*. Journal of Finance.
- Elmachtoub, A. N., & Grigas, P. (2022). *Smart "Predict, then Optimize"*. Management Science.

*Developed by **Pervengers** during the Purdue Academy for Global Engineering (PAGE) 2026*
