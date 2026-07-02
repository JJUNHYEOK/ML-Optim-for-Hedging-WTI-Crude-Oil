"""Feature engineering for the daily price-forecasting pipeline.

Target: fwd_ret = log(Spot[t+HORIZON]) - log(Spot[t]), ~1 calendar month
ahead counting only trading days (21).

Features are rolling/cumulative over windows matched to the horizon
(5/21/63 trading days), not single-day snapshots -- see eda_7 vs eda_8
in PROJECT_CONTEXT.md for why single-day features showed ~0 correlation
while window-matched ones reach |corr| ~ 0.15-0.18.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/WTI_daily_preprocessed.csv"
OUT_PATH = "data/WTI_daily_features.csv"

HORIZON = 21
WINDOWS = [5, 21, 63]

RENAME = {
    "Supply_Chain_Pressure": "SCP",
    "Geopolitical_Risk_GPR": "GPR",
    "Oil_Volatility_OVX": "OVX",
    "Yield_Curve_10Y2Y": "Yield_Curve",
    "US_Dollar_Index": "DXY",
}

CHANGE_COLS = {"OVX": "log", "DXY": "log", "Basis": "level", "Yield_Curve": "level"}
LEVEL_MEAN_COLS = ["GPR", "SCP", "OVX"]


def load_raw(path=RAW_PATH, start="2007-05-01"):
    """Weekends are already absent from the source file, but US market
    holidays are not -- they show up as rows where the previous trading
    day's price was silently carried forward (WTI_Spot and WTI_Futures
    both unchanged). These aren't real "no movement" days; they're
    non-trading days and would inject fake zero-return information into
    cumulative-change features and the forward-return target. Dropped by
    detecting simultaneous frozen Spot+Futures (200/5606 rows; 87% land on
    US federal holidays, the rest on NYMEX-only closures like Good Friday
    and the day after Thanksgiving -- both checked and confirmed, not
    assumed)."""
    df = pd.read_csv(path, parse_dates=["Date"]).rename(columns=RENAME).sort_values("Date").reset_index(drop=True)
    frozen = (df["WTI_Spot"].diff() == 0) & (df["WTI_Futures"].diff() == 0)
    df = df[~frozen].reset_index(drop=True)
    return df[df["Date"] >= start].reset_index(drop=True)


def build_features(df, windows=WINDOWS):
    out = pd.DataFrame({"Date": df["Date"]})
    log_spot = np.log(df["WTI_Spot"])

    for col, mode in CHANGE_COLS.items():
        series = np.log(df[col]) if mode == "log" else df[col]
        for w in windows:
            out[f"{col}_chg_{w}d"] = series.diff(w)

    for col in LEVEL_MEAN_COLS:
        for w in windows:
            out[f"{col}_mean_{w}d"] = df[col].rolling(w).mean()

    for w in windows:
        out[f"Spot_mom_{w}d"] = log_spot.diff(w)

    month = df["Date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    return out


CRISIS_OVX_THRESHOLD = 51.56  # 90th percentile of OVX computed on the train split only (2007-07~2019-10)


def build_targets(df, horizon=HORIZON):
    log_spot = np.log(df["WTI_Spot"])
    fwd_ret = log_spot.shift(-horizon) - log_spot
    return pd.DataFrame(
        {
            "Date": df["Date"],
            "fwd_ret": fwd_ret,
            "Spot_now": df["WTI_Spot"],
            "Spot_fwd": df["WTI_Spot"].shift(-horizon),
            "OVX_now": df["OVX"],
            "is_crisis": (df["OVX"] > CRISIS_OVX_THRESHOLD).astype(int),
        }
    )


def build_dataset(raw_path=RAW_PATH, windows=WINDOWS, horizon=HORIZON):
    df = load_raw(raw_path)
    features = build_features(df, windows=windows)
    targets = build_targets(df, horizon=horizon)
    return features.merge(targets, on="Date")


def run(raw_path=RAW_PATH, out_path=OUT_PATH, **kwargs):
    dataset = build_dataset(raw_path=raw_path, **kwargs)
    n_total = len(dataset)
    clean = dataset.dropna().reset_index(drop=True)
    print(f"rows: {n_total} total -> {len(clean)} after dropping NaN warm-up/tail rows")
    print(f"date range used: {clean['Date'].min().date()} ~ {clean['Date'].max().date()}")
    feature_cols = [c for c in dataset.columns if c not in ("Date", "fwd_ret", "Spot_now", "Spot_fwd", "OVX_now", "is_crisis")]
    print(f"feature columns ({len(feature_cols)}): {feature_cols}")
    clean.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")
    return clean


if __name__ == "__main__":
    run()
