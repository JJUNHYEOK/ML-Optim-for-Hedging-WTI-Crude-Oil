"""
Optuna 튜닝 버전.

절차:
  1) train으로 fit, val RMSE 최소화하도록 하이퍼파라미터 탐색.
  2) best params 찾으면 train + val 합쳐서 최종 모델 refit.
  3) test에서 평가.

시계열이므로 shuffle-CV는 사용하지 않습니다.
(train → val → test의 시간 순서를 지키는 hold-out 방식.)
"""
from __future__ import annotations

import numpy as np
import optuna
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _val_rmse(model, X_val, y_val) -> float:
    pred = model.predict(X_val)
    return float(np.sqrt(np.mean((pred - y_val) ** 2)))


def _make_study(random_state: int = RANDOM_STATE) -> optuna.Study:
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )


def _combine(X_tr, y_tr, X_val, y_val):
    return np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val])


# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #
def tune_ridge(X_tr, y_tr, X_val, y_val, n_trials: int = 50):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        m = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
        ])
        m.fit(X_tr, y_tr)
        return _val_rmse(m, X_val, y_val)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    final = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=best["alpha"], random_state=RANDOM_STATE)),
    ])
    X_all, y_all = _combine(X_tr, y_tr, X_val, y_val)
    final.fit(X_all, y_all)
    return final, best


# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
def tune_rf(X_tr, y_tr, X_val, y_val, n_trials: int = 30):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
        }
        m = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", RandomForestRegressor(
                **params, n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ])
        m.fit(X_tr, y_tr)
        return _val_rmse(m, X_val, y_val)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    final = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            **best, n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])
    X_all, y_all = _combine(X_tr, y_tr, X_val, y_val)
    final.fit(X_all, y_all)
    return final, best


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def tune_xgb(X_tr, y_tr, X_val, y_val, n_trials: int = 50):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        }
        m = xgb.XGBRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", verbosity=0,
        )
        m.fit(X_tr, y_tr)
        return _val_rmse(m, X_val, y_val)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    final = xgb.XGBRegressor(
        **best, random_state=RANDOM_STATE, n_jobs=-1,
        tree_method="hist", verbosity=0,
    )
    X_all, y_all = _combine(X_tr, y_tr, X_val, y_val)
    final.fit(X_all, y_all)
    return final, best


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #
def tune_lgbm(X_tr, y_tr, X_val, y_val, n_trials: int = 50):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500, step=100),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        m = lgb.LGBMRegressor(
            **params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        return _val_rmse(m, X_val, y_val)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    final = lgb.LGBMRegressor(
        **best, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    X_all, y_all = _combine(X_tr, y_tr, X_val, y_val)
    final.fit(X_all, y_all)
    return final, best


# --------------------------------------------------------------------------- #
# CatBoost
# --------------------------------------------------------------------------- #
def tune_catboost(X_tr, y_tr, X_val, y_val, n_trials: int = 30):
    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 200, 1500, step=100),
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength":     trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
        }
        m = CatBoostRegressor(
            **params, random_seed=RANDOM_STATE,
            verbose=False, allow_writing_files=False,
        )
        m.fit(X_tr, y_tr)
        return _val_rmse(m, X_val, y_val)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    final = CatBoostRegressor(
        **best, random_seed=RANDOM_STATE,
        verbose=False, allow_writing_files=False,
    )
    X_all, y_all = _combine(X_tr, y_tr, X_val, y_val)
    final.fit(X_all, y_all)
    return final, best


TUNERS = {
    "ridge_optuna":    tune_ridge,
    "rf_optuna":       tune_rf,
    "xgb_optuna":      tune_xgb,
    "lgbm_optuna":     tune_lgbm,
    "catboost_optuna": tune_catboost,
}
