"""Time-ordered train/val/test split for the daily price-forecasting
pipeline, with a purge gap between splits.

Both features (rolling windows up to 63 trading days) and the target
(21-day forward return) span multiple days, so adjacent rows share
information. Without a gap, the last training rows' targets reach into
the test period (forward leakage) and the first test rows' features
reach back into the training period (backward leakage). A gap of
max(target_horizon, longest_feature_window) = 63 trading days on each
split boundary blocks both directions. See PROJECT_CONTEXT.md for the
worked-through example.
"""

import pandas as pd

FEATURES_PATH = "data/WTI_daily_features.csv"
OUT_PATH = "data/WTI_daily_split.csv"

TRAIN_FRAC = 0.65
VAL_FRAC = 0.15
PURGE_GAP = 63  # trading days: max(target horizon=21, longest feature window=63)


def build_split(features_path=FEATURES_PATH, purge_gap=PURGE_GAP):
    df = pd.read_csv(features_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(df)

    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))

    train_end = n_train
    val_start = train_end + purge_gap
    val_end = val_start + n_val
    test_start = val_end + purge_gap

    split = pd.Series(["drop"] * n, index=df.index)
    split.iloc[:train_end] = "train"
    split.iloc[val_start:val_end] = "val"
    split.iloc[test_start:] = "test"
    df["split"] = split
    return df[df["split"] != "drop"].reset_index(drop=True)


if __name__ == "__main__":
    df = build_split()
    for name in ["train", "val", "test"]:
        sub = df[df["split"] == name]
        print(f"{name}: n={len(sub)}, {sub['Date'].min().date()} ~ {sub['Date'].max().date()}")
    df.to_csv(OUT_PATH, index=False)
    print(f"saved -> {OUT_PATH}")
