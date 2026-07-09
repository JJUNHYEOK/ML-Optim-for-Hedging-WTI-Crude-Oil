"""
ML 베이스라인 실행 스크립트.

구조 가정:
    project/
    ├── data/WTI_daily_split.csv
    ├── model/  (이 스크립트가 여기 있음)
    │   ├── ml_common.py
    │   ├── ml_fixed.py
    │   ├── ml_optuna.py
    │   └── run_ml_baselines.py
    └── results/ml_baselines/  (자동 생성)

실행 예시 (project 루트에서):
    python model/run_ml_baselines.py --n-trials 50           # 전체
    python model/run_ml_baselines.py --models xgb_optuna     # 특정 모델
    python model/run_ml_baselines.py --skip-fixed            # optuna만

model/ 안에서 실행해도 경로 자동 해결됨:
    cd model && python run_ml_baselines.py --n-trials 50

출력:
    results/ml_baselines/summary.csv       — 모델 x split x segment long-format
    results/ml_baselines/best_params.json  — Optuna 최적 파라미터

타겟:
    fwd_ret = HORIZON일(≈1개월) 후의 forward return (스칼라 1개, 방향 정보 포함).
    RW 베이스라인과 동일하게 CSV의 `split` 컬럼으로 train/val/test 구분.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml_common import (
    SPLIT_PATH, PROJECT_ROOT, NON_FEATURE_COLS,
    load_split, get_feature_cols, get_data_splits,
    evaluate_predictions, predict_by_split,
    print_results, results_to_dataframe,
    check_no_leakage,
)
from ml_fixed import FIXED_MODELS
from ml_optuna import TUNERS

DEFAULT_OUT_DIR = str(PROJECT_ROOT / "results" / "ml_baselines")


def run_fixed_model(name, builder, df, feature_cols):
    print(f"\n>>> Fitting {name} ...")
    X_tr, y_tr, *_ = get_data_splits(df, feature_cols)
    model = builder()
    model.fit(X_tr, y_tr)   # fixed: train만 사용
    preds = predict_by_split(model, df, feature_cols)
    return model, evaluate_predictions(df, preds)


def run_optuna_model(name, tuner, df, feature_cols, n_trials):
    print(f"\n>>> Tuning {name} with Optuna (n_trials={n_trials}) ...")
    X_tr, y_tr, X_va, y_va, *_ = get_data_splits(df, feature_cols)
    model, best_params = tuner(X_tr, y_tr, X_va, y_va, n_trials=n_trials)
    print(f"    best params: {best_params}")
    preds = predict_by_split(model, df, feature_cols)
    return model, evaluate_predictions(df, preds), best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=SPLIT_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="실행할 모델 이름 필터 (예: xgb_fixed xgb_optuna)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --models 이름 검증 (오타로 조용히 아무것도 안 돌아가는 것 방지)
    known = set(FIXED_MODELS) | set(TUNERS)
    if args.models:
        unknown = [m for m in args.models if m not in known]
        if unknown:
            parser.error(
                f"unknown model name(s): {unknown}\n"
                f"available: {sorted(known)}"
            )

    df = load_split(args.data)
    feature_cols = get_feature_cols(df)

    print(f"Loaded {len(df):,} rows")
    print(f"  split counts: {df['split'].value_counts().to_dict()}")
    print(f"  # features used: {len(feature_cols)}")
    print(f"  features: {feature_cols}")
    excluded = sorted(set(df.columns) - set(feature_cols))
    print(f"  # excluded: {len(excluded)}")
    print(f"  excluded: {excluded}")

    # --- 자동 누수 진단 (train 상관관계 기반) ---
    print("\n[leakage check on train]")
    check_no_leakage(df, feature_cols)
    print()

    all_results: list[pd.DataFrame] = []
    best_params_log: dict = {}

    # ---------------- Fixed ----------------
    if not args.skip_fixed:
        for name, builder in FIXED_MODELS.items():
            if args.models and name not in args.models:
                continue
            _, results = run_fixed_model(name, builder, df, feature_cols)
            print_results(name, results)
            all_results.append(results_to_dataframe(name, results))

    # ---------------- Optuna ---------------
    if not args.skip_optuna:
        for name, tuner in TUNERS.items():
            if args.models and name not in args.models:
                continue
            _, results, best_params = run_optuna_model(
                name, tuner, df, feature_cols, args.n_trials,
            )
            best_params_log[name] = best_params
            print_results(name, results)
            all_results.append(results_to_dataframe(name, results))

    # ---------------- Save & summarize ----
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary → {summary_path}")

        # test / all 리더보드
        test_all = (
            summary[(summary["split"] == "test") & (summary["segment"] == "all")]
            .sort_values("rmse")
            .reset_index(drop=True)
        )
        print("\n=== Test/all leaderboard (sorted by RMSE) ===")
        print(
            test_all[["model", "n", "directional_acc", "rmse", "nrmse_range_pct"]]
            .to_string(index=False)
        )

        # crisis segment 리더보드도 하나
        test_crisis = (
            summary[(summary["split"] == "test") & (summary["segment"] == "crisis")]
            .sort_values("rmse")
            .reset_index(drop=True)
        )
        if len(test_crisis) > 0:
            print("\n=== Test/crisis leaderboard (sorted by RMSE) ===")
            print(
                test_crisis[["model", "n", "directional_acc", "rmse", "nrmse_range_pct"]]
                .to_string(index=False)
            )

    if best_params_log:
        params_path = out_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params_log, f, indent=2)
        print(f"Saved best params → {params_path}")


if __name__ == "__main__":
    main()