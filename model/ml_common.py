"""
Shared utilities for ML baselines on WTI fwd_ret forecasting.

- Data loading & auto-feature detection
- Metrics: directional_acc, RMSE, NRMSE by range (%)
- Evaluation harness (all / normal / crisis segments per split)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# 이 파일(ml_common.py)이 <project>/model/ 아래에 있고,
# 데이터는 <project>/data/WTI_daily_split.csv 라고 가정.
_MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODEL_DIR.parent
SPLIT_PATH = str(PROJECT_ROOT / "data" / "WTI_daily_split.csv")

TARGET_COL = "fwd_ret"
# 피처로 절대 쓰면 안 되는 컬럼들. (팀 공용 정의 — 담당3 코드와 일치)
# - Date, split, is_crisis, fwd_ret : 메타/타겟
# - Spot_fwd : t+HORIZON 시점 spot price. fwd_ret 정의식의 미래 항 (완전 누출)
# - Spot_now : 오늘의 raw spot price. leak 는 아니나 train(2007-2019) vs
#              test(2022-2026) 가격 분포가 크게 달라 트리 모델 extrapolation 위험.
#              가격 정보는 이미 Spot_mom_5d/21d/63d 로 상대값으로 들어있음.
# - OVX_now  : 오늘의 raw OVX 수준. is_crisis 정의(OVX_now > 51.56)에 쓰여서
#              crisis segment 평가가 모델이 학습한 regime 판별과 얽힘. 정보는
#              OVX_chg_*, OVX_mean_* 로 이미 포함됨.
NON_FEATURE_COLS = {
    "Date", "split", "is_crisis", TARGET_COL,
    "Spot_fwd", "Spot_now", "OVX_now",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(path: str = SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


# 새 피처가 CSV에 추가됐을 때 미래 정보가 섞였는지 자동 감지용 패턴.
# get_feature_cols가 이 패턴 중 하나를 이름에 포함한 피처를 탐지하면 경고 출력.
_SUSPICIOUS_PATTERNS = ("fwd", "future", "next_", "target", "_lead", "ahead", "_t+", "shifted_neg")


def get_feature_cols(
    df: pd.DataFrame,
    exclude: Iterable[str] = NON_FEATURE_COLS,
    warn_suspicious: bool = True,
) -> list[str]:
    """Numeric-only 컬럼 중에서 exclude를 뺀 나머지를 피처로 사용.

    새 피처가 실수로 미래 정보를 담고 있을 때 감지하기 위해,
    이름에 fwd/future/next/target/_lead 등이 들어간 컬럼이 exclude되지 않고
    피처로 편입되면 경고를 띄웁니다. (완벽하진 않지만 안전망 역할)
    """
    exclude = set(exclude)
    features = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if warn_suspicious:
        flagged = [
            c for c in features
            if any(p in c.lower() for p in _SUSPICIOUS_PATTERNS)
        ]
        if flagged:
            print(
                "⚠️  누수 의심 피처 감지 (이름 기반):\n"
                f"    {flagged}\n"
                "    → 미래 시점 정보라면 NON_FEATURE_COLS 에 반드시 추가하세요.\n"
                "    → 이름만 그렇고 실제로는 과거 정보라면 이 경고는 무시 가능."
            )

    return features


def check_no_leakage(df: pd.DataFrame, feature_cols: list[str], target: str = TARGET_COL) -> None:
    """train 구간에서 각 피처와 타겟의 |상관|을 계산. 위험 수준이면 경고.

    - |corr| > 0.5  → 강한 leak 의심 (실전 금융 예측에선 나올 수 없는 수치)
    - |corr| > 0.3  → 확인 필요
    - |corr| < 0.2  → 정상 범위
    """
    train = df[df["split"] == "train"]
    if len(train) == 0:
        return
    corr = (
        train[feature_cols + [target]].corr()[target]
        .drop(target).abs().sort_values(ascending=False)
    )
    strong = corr[corr > 0.5]
    moderate = corr[(corr > 0.3) & (corr <= 0.5)]
    if len(strong) > 0:
        print(f"❌ |corr with {target}| > 0.5 (강한 leak 의심):")
        print(strong.round(3).to_string())
    if len(moderate) > 0:
        print(f"⚠️  |corr with {target}| 0.3~0.5 (확인 필요):")
        print(moderate.round(3).to_string())
    if len(strong) == 0 and len(moderate) == 0:
        top3 = corr.head(3)
        print(f"✓  no strong leakage signal (top-3 |corr|: "
              f"{', '.join(f'{k}={v:.3f}' for k, v in top3.items())})")


def split_xy(df: pd.DataFrame, feature_cols: list[str], target: str = TARGET_COL):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(np.float64)
    return X, y


def get_data_splits(df: pd.DataFrame, feature_cols: list[str]):
    """(X_train, y_train, X_val, y_val, X_test, y_test) 반환."""
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "val"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    X_tr, y_tr = split_xy(tr, feature_cols)
    X_va, y_va = split_xy(va, feature_cols)
    X_te, y_te = split_xy(te, feature_cols)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def predict_by_split(model, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    preds = {}
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name].reset_index(drop=True)
        if len(sub) == 0:
            continue
        X, _ = split_xy(sub, feature_cols)
        preds[split_name] = model.predict(X)
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(pred_ret, actual_ret) -> dict:
    """
    회귀/방향 지표.
    - directional_acc : sign(pred) == sign(actual) 의 평균 (부호 정확도)
    - rmse            : sqrt(MSE), 수익률과 동일 단위
    - nrmse_range_pct : RMSE / (y_max - y_min) * 100
                        (segment 내부 range 기준. 다른 정규화 원하면 여기 수정)
    """
    pred_ret = np.asarray(pred_ret, dtype=float)
    actual_ret = np.asarray(actual_ret, dtype=float)

    rmse = float(np.sqrt(np.mean((pred_ret - actual_ret) ** 2)))
    y_range = float(actual_ret.max() - actual_ret.min())
    nrmse_range_pct = float(rmse / y_range * 100) if y_range > 0 else float("nan")
    directional_acc = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))

    return {
        "directional_acc": directional_acc,
        "rmse": rmse,
        "nrmse_range_pct": nrmse_range_pct,
    }


# --------------------------------------------------------------------------- #
# Evaluation harness
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    df_split: pd.DataFrame,
    predictions: dict,
    segment_by_crisis: bool = True,
) -> dict:
    """
    predictions: {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}
    반환: results[split_name][segment_name] = {n, crisis_share, metrics}
    """
    results = {}
    for split_name in ["train", "val", "test"]:
        sub = df_split[df_split["split"] == split_name].reset_index(drop=True)
        if split_name not in predictions or len(sub) == 0:
            continue
        pred_all = np.asarray(predictions[split_name])
        assert len(pred_all) == len(sub), (
            f"prediction length mismatch on {split_name}: "
            f"{len(pred_all)} vs {len(sub)}"
        )

        segments = {"all": (sub, pred_all)}
        if segment_by_crisis:
            mask_normal = (sub["is_crisis"] == 0).values
            mask_crisis = (sub["is_crisis"] == 1).values
            segments["normal"] = (sub[mask_normal], pred_all[mask_normal])
            segments["crisis"] = (sub[mask_crisis], pred_all[mask_crisis])

        results[split_name] = {}
        for seg_name, (seg_df, seg_pred) in segments.items():
            if len(seg_df) == 0:
                continue
            actual = seg_df[TARGET_COL].values
            results[split_name][seg_name] = {
                "n": int(len(seg_df)),
                "crisis_share": (
                    float(seg_df["is_crisis"].mean()) if seg_name == "all" else None
                ),
                "metrics": metrics(seg_pred, actual),
            }
    return results


def print_results(model_name: str, results: dict) -> None:
    print(f"\n=== {model_name} ===")
    for split_name, segments in results.items():
        print(f"  [{split_name}]")
        for seg_name, seg_res in segments.items():
            n = seg_res["n"]
            share = (
                f" crisis_share={seg_res['crisis_share']:.2%}"
                if seg_res["crisis_share"] is not None else ""
            )
            m = seg_res["metrics"]
            print(
                f"    {seg_name:6s} n={n:5d}{share}  "
                f"dir_acc={m['directional_acc']:.3f}  "
                f"RMSE={m['rmse']:.5f}  "
                f"NRMSE(range)={m['nrmse_range_pct']:.2f}%"
            )


def results_to_dataframe(model_name: str, results: dict) -> pd.DataFrame:
    """중첩 dict를 long-format DataFrame으로 flatten."""
    rows = []
    for split_name, segments in results.items():
        for seg_name, seg_res in segments.items():
            m = seg_res["metrics"]
            rows.append({
                "model": model_name,
                "split": split_name,
                "segment": seg_name,
                "n": seg_res["n"],
                "crisis_share": seg_res["crisis_share"],
                "directional_acc": m["directional_acc"],
                "rmse": m["rmse"],
                "nrmse_range_pct": m["nrmse_range_pct"],
            })
    return pd.DataFrame(rows)