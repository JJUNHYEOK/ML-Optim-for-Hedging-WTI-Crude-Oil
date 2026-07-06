"""담당 3 (LightGBM / 비선형-과적합 검증), v2 데이터 기준.

v1에서 나온 결론("LightGBM은 train/val에서만 좋고 test에서 과적합")이
v2(2007-05-14~2026-02-28, 공휴일 제거, SCP lag 수정)에서도 재현되는지 확인.
RW baseline과 metrics 계산은 models/baseline_randomwalk.py를 그대로 재사용
(SPLIT_PATH만 v2로 교체) -- v1과 동일한 방식으로 비교해야 의미가 있어서 로직을
새로 안 만들고 그대로 가져다 씀.

두 configs 비교:
- v1_config: models/lgbm_forecast.py와 완전히 같은 하이퍼파라미터 (이미 어느 정도
  정규화가 걸려있는 상태) -- v1 결론이 재현되는지 먼저 확인
- more_regularized: 그래도 과적합이면 depth/leaves를 더 줄이고 reg를 더 세게 준
  버전으로 완화되는지 확인

Ridge는 "복잡한 모델이 실제로 나은지" 비교할 최소한의 참고용 (담당1이 제대로
튜닝할 정식 Ridge 트랙과는 별개, 여기선 val로 alpha만 간단히 선택).
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baseline_randomwalk import evaluate as rw_evaluate  # noqa: E402
from models.baseline_randomwalk import metrics  # noqa: E402

SPLIT_PATH = "data/WTI_daily_split.csv"
NON_FEATURE_COLS = {"Date", "fwd_ret", "Spot_now", "Spot_fwd", "OVX_now", "is_crisis", "split"}

LGBM_CONFIGS = {
    "v1_config": dict(
        objective="regression", n_estimators=2000, learning_rate=0.01, max_depth=4,
        num_leaves=15, min_child_samples=50, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=0, verbosity=-1,
    ),
    "more_regularized": dict(
        objective="regression", n_estimators=2000, learning_rate=0.01, max_depth=2,
        num_leaves=4, min_child_samples=150, subsample=0.6, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=1.0, random_state=0, verbosity=-1,
    ),
}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def load_split(path=SPLIT_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df, feature_cols


def train_lgbm(params, train, val, test, feature_cols):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[feature_cols], train["fwd_ret"],
        eval_set=[(val[feature_cols], val["fwd_ret"])],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    results = {}
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        pred = model.predict(sub[feature_cols])
        results[name] = {"n": len(sub), **metrics(pred, sub["fwd_ret"])}
    return model, results


def scale_features(train, val, test, feature_cols):
    """Ridge/Lasso alpha grids only make sense in comparable units -- fit
    the scaler on train only (no leakage), apply to val/test."""
    scaler = StandardScaler().fit(train[feature_cols])
    return (
        scaler.transform(train[feature_cols]),
        scaler.transform(val[feature_cols]),
        scaler.transform(test[feature_cols]),
    )


def train_ridge_quick(train, val, test, feature_cols):
    X_train, X_val, X_test = scale_features(train, val, test, feature_cols)
    best_alpha, best_val_rmse = None, np.inf
    for alpha in RIDGE_ALPHAS:
        m = Ridge(alpha=alpha).fit(X_train, train["fwd_ret"])
        val_rmse = metrics(m.predict(X_val), val["fwd_ret"])["rmse"]
        if val_rmse < best_val_rmse:
            best_alpha, best_val_rmse = alpha, val_rmse

    model = Ridge(alpha=best_alpha).fit(X_train, train["fwd_ret"])
    results = {}
    for name, X, sub in [("train", X_train, train), ("val", X_val, val), ("test", X_test, test)]:
        pred = model.predict(X)
        results[name] = {"n": len(sub), **metrics(pred, sub["fwd_ret"])}
    return model, results, best_alpha


def train_lasso_quick(train, val, test, feature_cols):
    X_train, X_val, X_test = scale_features(train, val, test, feature_cols)
    best_alpha, best_val_rmse = None, np.inf
    for alpha in LASSO_ALPHAS:
        m = Lasso(alpha=alpha, max_iter=20000).fit(X_train, train["fwd_ret"])
        val_rmse = metrics(m.predict(X_val), val["fwd_ret"])["rmse"]
        if val_rmse < best_val_rmse:
            best_alpha, best_val_rmse = alpha, val_rmse

    model = Lasso(alpha=best_alpha, max_iter=20000).fit(X_train, train["fwd_ret"])
    n_nonzero = int(np.sum(model.coef_ != 0))
    results = {}
    for name, X, sub in [("train", X_train, train), ("val", X_val, val), ("test", X_test, test)]:
        pred = model.predict(X)
        results[name] = {"n": len(sub), **metrics(pred, sub["fwd_ret"])}
    return model, results, best_alpha, n_nonzero


def print_row(label, m, rw):
    print(
        f"  {label:<10} RMSE={m['rmse']:.4f} dir_acc={m['directional_acc']:.3f}"
        f"   |  RW+drift: RMSE={rw['rmse']:.4f} dir_acc={rw['directional_acc']:.3f}"
    )


if __name__ == "__main__":
    df, feature_cols = load_split()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"features used: {len(feature_cols)} | train={len(train)} val={len(val)} test={len(test)}")

    rw_results, drift = rw_evaluate(split_path=SPLIT_PATH)

    print("\n########## Ridge (quick reference, alpha selected on val) ##########")
    ridge_model, ridge_results, best_alpha = train_ridge_quick(train, val, test, feature_cols)
    print(f"  selected alpha={best_alpha}")
    for split_name in ["train", "val", "test"]:
        rw = rw_results[split_name]["all"]["random_walk_drift"]
        print_row(split_name, ridge_results[split_name], rw)

    print("\n########## Lasso (quick reference, alpha selected on val) ##########")
    lasso_model, lasso_results, lasso_alpha, n_nonzero = train_lasso_quick(train, val, test, feature_cols)
    print(f"  selected alpha={lasso_alpha}  |  nonzero coefs: {n_nonzero}/{len(feature_cols)}")
    for split_name in ["train", "val", "test"]:
        rw = rw_results[split_name]["all"]["random_walk_drift"]
        print_row(split_name, lasso_results[split_name], rw)

    last_model = None
    for config_name, params in LGBM_CONFIGS.items():
        print(f"\n########## LightGBM [{config_name}] ##########")
        model, results = train_lgbm(params, train, val, test, feature_cols)
        last_model = model
        for split_name in ["train", "val", "test"]:
            rw = rw_results[split_name]["all"]["random_walk_drift"]
            print_row(split_name, results[split_name], rw)
        gap = results["train"]["directional_acc"] - results["test"]["directional_acc"]
        verdict = "overfit suspected" if gap > 0.05 else "no strong overfit signal"
        print(f"  train-test dir_acc gap: {gap:+.3f}  ({verdict})")
        print(f"  vs Ridge test dir_acc: LightGBM={results['test']['directional_acc']:.3f}  Ridge={ridge_results['test']['directional_acc']:.3f}")

    print("\ntop 10 feature importances (last LightGBM config trained):")
    imp = pd.Series(last_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(imp.head(10).to_string())
