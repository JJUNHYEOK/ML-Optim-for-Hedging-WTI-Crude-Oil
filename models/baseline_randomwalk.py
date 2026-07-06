"""Baseline 0: random walk (no model) -- the standard, hard-to-beat
benchmark in commodity price forecasting. Every subsequent model must
clear this bar to be worth using.

- RW: predicted fwd_ret = 0  (price in HORIZON days = price today)
- RW + drift: predicted fwd_ret = train-set mean fwd_ret
"""

import numpy as np
import pandas as pd

SPLIT_PATH = "data/WTI_daily_split.csv"


def load_split(path=SPLIT_PATH):
    return pd.read_csv(path, parse_dates=["Date"])


def metrics(pred_ret, actual_ret):
    pred_ret = np.asarray(pred_ret, dtype=float)
    actual_ret = np.asarray(actual_ret, dtype=float)
    rmse = float(np.sqrt(np.mean((pred_ret - actual_ret) ** 2)))
    mae = float(np.mean(np.abs(pred_ret - actual_ret)))
    directional_acc = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)))
    return {"rmse": rmse, "mae": mae, "directional_acc": directional_acc}


def evaluate(split_path=SPLIT_PATH, segment_by_crisis=True):
    df = load_split(split_path)
    train = df[df["split"] == "train"]
    drift = train["fwd_ret"].mean()

    results = {}
    for name in ["train", "val", "test"]:
        sub = df[df["split"] == name]
        segments = {"all": sub}
        if segment_by_crisis:
            segments["normal"] = sub[sub["is_crisis"] == 0]
            segments["crisis"] = sub[sub["is_crisis"] == 1]
        results[name] = {}
        for seg_name, seg in segments.items():
            if len(seg) == 0:
                continue
            results[name][seg_name] = {
                "n": len(seg),
                "crisis_share": float(seg["is_crisis"].mean()) if seg_name == "all" else None,
                "random_walk": metrics(np.zeros(len(seg)), seg["fwd_ret"]),
                "random_walk_drift": metrics(np.full(len(seg), drift), seg["fwd_ret"]),
            }
    return results, drift


if __name__ == "__main__":
    results, drift = evaluate()
    print(f"train drift (mean fwd_ret): {drift:.5f}  (crisis = OVX > {51.56:g}, train 90th pct)")
    for split_name, segments in results.items():
        print(f"=== {split_name} ===")
        for seg_name, seg_results in segments.items():
            n = seg_results["n"]
            share = f" (crisis_share={seg_results['crisis_share']:.2%})" if seg_results["crisis_share"] is not None else ""
            print(f"  [{seg_name}] n={n}{share}")
            for method_name in ["random_walk", "random_walk_drift"]:
                m = seg_results[method_name]
                print(f"    {method_name}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  dir_acc={m['directional_acc']:.3f}")
