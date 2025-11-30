#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_split(parquet_path: str, sample: int | None):
    df = pd.read_parquet(parquet_path) if parquet_path.endswith(".parquet") else pd.read_csv(parquet_path)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    targ_cols = [c for c in df.columns if c.startswith("t")]
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[targ_cols].to_numpy(dtype=np.float32) if targ_cols else None
    rows = df["row"].to_numpy(dtype=np.int32) if "row" in df.columns else None
    cols = df["col"].to_numpy(dtype=np.int32) if "col" in df.columns else None
    return X, y, feat_cols, targ_cols, rows, cols, df


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # y_* shape: (N,)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    # R2
    var = float(np.var(y_true)) if y_true.size else 0.0
    r2 = float(1.0 - (np.mean(diff ** 2) / var)) if var > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_scatter(y_true, y_pred, out_png, title):
    plt.figure(figsize=(4, 4), dpi=160)
    plt.scatter(y_true, y_pred, s=2, alpha=0.3)
    mmin = float(min(np.min(y_true), np.min(y_pred)))
    mmax = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mmin, mmax], [mmin, mmax], "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_residuals(y_true, y_pred, out_png, title):
    res = y_pred - y_true
    plt.figure(figsize=(5, 3), dpi=160)
    plt.hist(res, bins=60, alpha=0.85)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser("LightGBM band-wise test")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--base", default="/Volumes/WD_BLACK/FortMyers/processed/lgbm", help="Base directory with parquet and models")
    parser.add_argument("--sample", type=int, default=None, help="Optional number of rows to sample")
    parser.add_argument("--out", default="/Volumes/WD_BLACK/FortMyers/processed/lgbm_eval", help="Output directory for metrics and plots")
    args = parser.parse_args()

    parquet_path = os.path.join(args.base, f"{args.split}.parquet")
    models_dir = args.base
    ensure_dir(args.out)

    X, y, feat_cols, targ_cols, rows, cols, df = load_split(parquet_path, args.sample)
    if y is None or len(targ_cols) != 6:
        raise RuntimeError("Expected targets t0..t5 in the split parquet.")

    band_names = ["B02_Blue", "B03_Green", "B04_Red", "B8A_NNIR", "B11_SWIR1", "B12_SWIR2"]

    metrics_all = {}
    preds_all = np.zeros_like(y, dtype=np.float32)

    for i, tcol in enumerate(targ_cols):
        model_path = os.path.join(models_dir, f"model_t{i}.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        booster = lgb.Booster(model_file=model_path)
        pred = booster.predict(X)
        preds_all[:, i] = pred.astype(np.float32)
        m = compute_metrics(y[:, i], pred)
        metrics_all[tcol] = {"band": band_names[i], **m}

        # Plots
        band_tag = band_names[i]
        plot_scatter(y[:, i], pred, os.path.join(args.out, f"{args.split}_{tcol}_{band_tag}_scatter.png"), f"{args.split} {tcol} {band_tag}")
        plot_residuals(y[:, i], pred, os.path.join(args.out, f"{args.split}_{tcol}_{band_tag}_residuals.png"), f"{args.split} {tcol} {band_tag}")

    # Save metrics
    with open(os.path.join(args.out, f"{args.split}_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    # Save a small summary CSV
    rows_csv = []
    for tcol in targ_cols:
        entry = metrics_all[tcol]
        rows_csv.append({"target": tcol, "band": entry["band"], "mae": entry["mae"], "rmse": entry["rmse"], "r2": entry["r2"]})
    pd.DataFrame(rows_csv).to_csv(os.path.join(args.out, f"{args.split}_metrics.csv"), index=False)

    print({"split": args.split, "num_rows": int(len(df)), "out": args.out})


if __name__ == "__main__":
    main()


