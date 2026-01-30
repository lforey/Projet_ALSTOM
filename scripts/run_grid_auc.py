import argparse
import os
import json
import itertools
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import DEFAULT_ANOMALIES
from src.utils import ensure_dir, cleanup
from src.data_io import find_csv, load_raw_csv
from src.preprocessing import resample_mean, add_ground_truth, impute_sensors, fit_transform_scaler
from src.windowing import build_window_starts, pure_normal_starts_fast
from src.model import LSTMAutoencoder
from src.training import train_with_early_stopping
from src.scoring import score_time_series_from_windows
from src.evaluation import compute_auc_metrics
from src.plotting import plot_score_and_roc


class StartsWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, starts: np.ndarray, window: int):
        self.X = X.astype(np.float32, copy=False)
        self.starts = starts
        self.window = window

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return torch.from_numpy(self.X[s:s + self.window])


def parse_args():
    p = argparse.ArgumentParser(description="Grid search (AUC) for LSTM autoencoder anomaly detection.")
    p.add_argument("--data_dir", required=True, help="Directory containing CSV file(s).")
    p.add_argument("--csv_path", default=None, help="Optional direct path to CSV.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--timestamp_col", default="timestamp", help="Timestamp column name.")
    p.add_argument("--resample", default="20S", help="Resample rule (e.g., 1T, 20S).")
    return p.parse_args()


def build_param_grid(resample_rule: str) -> dict:
    """
    A practical grid for long-duration anomalies (start small, refine later).
    This is intentionally not gigantic; otherwise you'll end up with multi-day runs.
    """
    return {
        "RESAMPLE_RULE": [resample_rule],
        "WINDOW_SIZE": [720, 1080, 1440],  # 4h / 6h / 8h if resample=20S
        "STRIDE": [30, 60, 90],            # 10/20/30 min if resample=20S
        "HIDDEN_SIZE": [64, 96],
        "LATENT_SIZE": [2, 4, 6],
        "BATCH_SIZE": [32, 64],
        "EPOCHS": [20],
        "LR": [1e-3],
    }


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    torch.set_num_threads(1)
    device = "cpu"

    csv_path = args.csv_path or find_csv(args.data_dir)
    print(f"Using CSV: {csv_path}")

    # Load once (raw) to avoid re-reading the CSV for each run.
    df_raw = load_raw_csv(csv_path, timestamp_col=args.timestamp_col)

    anomalies = [(a.name, a.start, a.end) for a in DEFAULT_ANOMALIES]

    grid = build_param_grid(args.resample)
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"grid_auc_{run_stamp}")
    ensure_dir(run_dir)

    # Save grid for reproducibility
    with open(os.path.join(run_dir, "param_grid.json"), "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)

    out_csv = os.path.join(run_dir, "grid_results_partial.csv")

    # Prepare combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Total runs: {len(combos)}")
    print(f"Outputs: {run_dir}")

    # Resume logic: skip run_name already present in CSV
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        if "run_name" in prev.columns:
            done = set(prev["run_name"].dropna().astype(str).tolist())
        print(f"Resume enabled: {len(done)} runs already done -> will be skipped.")

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))

        RESAMPLE_RULE = params["RESAMPLE_RULE"]
        WINDOW_SIZE = int(params["WINDOW_SIZE"])
        STRIDE = int(params["STRIDE"])
        HIDDEN_SIZE = int(params["HIDDEN_SIZE"])
        LATENT_SIZE = int(params["LATENT_SIZE"])
        BATCH_SIZE = int(params["BATCH_SIZE"])
        EPOCHS = int(params["EPOCHS"])
        LR = float(params["LR"])

        run_name = (
            f"run{idx:04d}_res{RESAMPLE_RULE}_W{WINDOW_SIZE}_S{STRIDE}_"
            f"H{HIDDEN_SIZE}_Z{LATENT_SIZE}_B{BATCH_SIZE}_E{EPOCHS}_LR{LR}"
        )

        out_png = os.path.join(run_dir, run_name + ".png")

        if run_name in done and os.path.exists(out_png):
            print(f"[{idx}/{len(combos)}] SKIP {run_name}")
            continue

        print(f"\n[{idx}/{len(combos)}] {run_name}")
        t0 = time.time()

        # --- Prepare dataset for this run
        df = resample_mean(df_raw, RESAMPLE_RULE)
        df = add_ground_truth(df, anomalies, label_col="ground_truth_anomaly")

        feature_cols = [c for c in df.columns if c != "ground_truth_anomaly"]
        gt = df["ground_truth_anomaly"].astype(int).values
        normal_mask = (gt == 0)

        df = impute_sensors(df, feature_cols, normal_mask)
        X_all = df[feature_cols].values.astype(np.float32, copy=False)

        if np.isnan(X_all).any() or np.isinf(X_all).any():
            raise ValueError("NaN/Inf still present after imputation (unexpected).")

        T = len(df)
        if T < WINDOW_SIZE + 1:
            raise ValueError("Time series too short for chosen window size.")

        X, _ = fit_transform_scaler(X_all, normal_mask)

        all_starts = build_window_starts(T, WINDOW_SIZE, STRIDE)
        trainval_starts = pure_normal_starts_fast(gt, WINDOW_SIZE, STRIDE)
        if len(trainval_starts) < 200:
            raise ValueError("Not enough pure-normal windows for train/val split.")

        # Chronological split
        n_w = len(trainval_starts)
        w_train_end = int(n_w * 0.7)
        w_val_end = int(n_w * 0.85)
        train_starts = trainval_starts[:w_train_end]
        val_starts = trainval_starts[w_train_end:w_val_end]

        train_loader = DataLoader(
            StartsWindowDataset(X, train_starts, WINDOW_SIZE),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
        )
        val_loader = DataLoader(
            StartsWindowDataset(X, val_starts, WINDOW_SIZE),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False
        )

        # --- Train
        model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE).to(device)

        summary = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=EPOCHS,
            lr=LR,
            patience=7,
            min_delta=1e-4,
            use_early_stopping=True,
        )

        # --- Score & AUC
        all_loader = DataLoader(
            StartsWindowDataset(X, all_starts, WINDOW_SIZE),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False
        )

        score = score_time_series_from_windows(
            df_index=df.index,
            T=T,
            all_starts=all_starts,
            window_size=WINDOW_SIZE,
            model=model,
            all_loader=all_loader,
            device=device,
        )

        auc = compute_auc_metrics(gt, score)

        info_box = (
            f"RESAMPLE={RESAMPLE_RULE}\n"
            f"W={WINDOW_SIZE}  STRIDE={STRIDE}\n"
            f"H={HIDDEN_SIZE}  Z={LATENT_SIZE}\n"
            f"B={BATCH_SIZE}  E(max)={EPOCHS}\n"
            f"LR={LR}\n"
            f"epochs_ran={summary['epochs_ran']}\n"
            f"ROC_AUC={auc['roc_auc']:.3f}\n"
            f"PR_AUC={auc['pr_auc']:.3f}\n"
        )

        plot_score_and_roc(
            df_index=df.index,
            score=score,
            anomalies=anomalies,
            roc_data=auc,
            title=run_name,
            out_path=out_png,
            info_box=info_box,
        )

        # --- Append results immediately (safe stop/resume)
        row = {
            **params,
            "run_name": run_name,
            "png_path": out_png,
            "roc_auc": auc["roc_auc"],
            "pr_auc": auc["pr_auc"],
            "val_mse_best": summary["best_val_mse"],
            "epochs_ran": summary["epochs_ran"],
            "runtime_s": round(time.time() - t0, 2),
        }
        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(out_csv)
        df_row.to_csv(out_csv, mode="a", header=write_header, index=False)

        print(f"OK | ROC_AUC={auc['roc_auc']:.3f} PR_AUC={auc['pr_auc']:.3f} | {row['runtime_s']}s")

        cleanup(df, X_all, X, model, train_loader, val_loader, all_loader, score)

    print("\nDone.")
    print(f"Results: {out_csv}")
    print(f"Images:  {run_dir}")
    cleanup(df_raw)


if __name__ == "__main__":
    main()
