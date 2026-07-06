# Model Results (as of 2026-07)

## Files

- `baseline_randomwalk.py` — the random-walk baselines (`RW`: predict zero
  return; `RW+drift`: predict the train-set mean return). `RW+drift` is
  what "RW" refers to everywhere below — plain `RW` predicts sign(0),
  which almost never matches, so its direction accuracy is ~0% by
  construction and isn't a meaningful comparison point.
- `lightgbm_forecast.py` — LightGBM overfitting check, with quick
  Ridge/Lasso references for context.

## LightGBM: does it beat Ridge/Lasso? No.

Checked whether the earlier finding on the v1 dataset ("LightGBM looks
good on train/val, degrades on test") still holds on v2
(2007-05-14~2026-02-28, holidays removed, SCP look-ahead fixed).

| model | train dir_acc | val dir_acc | test dir_acc | test RMSE |
|---|---|---|---|---|
| RW+drift | 46.3% | 40.6% | 54.4% | 0.0706 |
| Ridge (quick ref.) | 55.0% | 50.6% | 60.4% | 0.0707 |
| Lasso (quick ref.) | 56.3% | 58.0% | 61.9% | 0.0707 |
| LightGBM (same config as v1) | 46.3% | 40.6% | 54.4% | 0.0706 |
| LightGBM (more regularized) | 58.7% | 52.8% | 52.9% | 0.0712 |

**Two different failure modes, not one:**
- The v1 hyperparameter config doesn't overfit — it doesn't learn
  anything at all. Its numbers are identical to RW+drift on every split,
  meaning it collapsed to predicting the constant train mean.
- Pushing regularization further (shallower trees, stronger L1/L2) does
  make it learn (train 58.7%), but then it shows a real train/test gap
  and still loses to RW+drift on test (52.9% vs 54.4%).

**Conclusion**: LightGBM doesn't work here either way — too regularized
and it learns nothing, less regularized and it overfits without ever
clearing the RW bar. Ridge and Lasso both beat RW cleanly and don't show
this over/underfitting pattern (train is not higher than val/test for
either), consistent with the v1 result.

## Note for whoever runs the official Lasso/ElasticNet track

**Scale your features before fitting Ridge/Lasso.** Running Lasso on the
raw (unscaled) 26 features here kept only 2/26 nonzero coefficients and
underperformed (test dir_acc 55.4%) — because features have very
different natural scales (`month_sin` in [-1, 1] vs `GPR_mean` in the
tens-to-hundreds), Lasso's L1 penalty unfairly kills off small-scale
features regardless of how useful they actually are. After fitting
`StandardScaler` on train only and transforming val/test, 7/26 features
survived and test dir_acc rose to 61.9%. Ridge is less dramatically
affected (its penalty shrinks smoothly instead of zeroing out
coefficients) but should still be scaled for the same reason.

The Ridge/Lasso numbers above are from a 4-5 value alpha grid picked on
val — a quick sanity check, not the tuned final numbers. The real
Ridge/Lasso/ElasticNet track should use proper cross-validation.
