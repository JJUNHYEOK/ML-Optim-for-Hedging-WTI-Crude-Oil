"""
하이퍼파라미터 고정 버전 (튜닝 없음).
- 튜닝된 버전과 비교용 sanity check.
- Ridge/RF는 결측치 못 다루므로 SimpleImputer(mean) 파이프라인에 포함.
- XGB/LGBM/CatBoost는 NaN을 native로 처리하므로 imputer 없음.
"""
from __future__ import annotations

from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Random Walk + drift 베이스라인 (하이퍼파라미터 없음).
# 예측 fwd_ret = train set 평균 fwd_ret. 모든 ML 모델은 최소한 이걸 넘어야
# "쓸모 있음"이라고 말할 수 있음.
# --------------------------------------------------------------------------- #
def build_rw_drift():
    """RW + drift: 예측 fwd_ret = train set 평균 fwd_ret."""
    return DummyRegressor(strategy="mean")


def build_ridge():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def build_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.7,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def build_xgb():
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def build_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost():
    return CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )


FIXED_MODELS = {
    "rw_drift":       build_rw_drift,    # 예측 = train 평균 fwd_ret (베이스라인)
    "ridge_fixed":    build_ridge,
    "rf_fixed":       build_rf,
    "xgb_fixed":      build_xgb,
    "lgbm_fixed":     build_lgbm,
    "catboost_fixed": build_catboost,
}