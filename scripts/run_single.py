import argparse
import os
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
    p = argparse.ArgumentParser(description="Single run: LSTM autoencoder anomaly detection (AUC evaluation).")
    p.add_argument("--data_dir", required=True, help="Directory containing the CSV file(s).")
    p.add_argument("--csv_path", default=None, help="Optional direct path to a CSV. Overrides data_dir search.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--timestamp_col", default="timestamp", help="Timestamp column name.")
    p.add_argument("--resample", default="1T", help="Pandas resample rule (e.g., 1T, 20S).")

    p.add_argument("--window", type=int, default=360, help="Window size (in resampled points).")
    p.add_argument("--stride", type=int, default=20, help="Stride (in resampled points).")

    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden size.")
    p.add_argument("--latent", type=int, default=4, help="Latent size.")
    p.add_argument("--batch", type=int, default=128, help="Batch size.")
    p.add_argument("--epochs", type=int, default=20, help="Max epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    p.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Early stopping min delta.")
    p.add_argument("--no_early_stop", action="store_true", help="Disable early stopping.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    torch.set_num_threads(1)
    device = "cpu"

    csv_path = args.csv_path or find_csv(args.data_dir)
    print(f"Using CSV: {csv_path}")

    df_raw = load_raw_csv(csv_path, timestamp_col=args.timestamp_col)
    df = resample_mean(df_raw, args.resample)

    anomalies = [(a.name, a.start, a.end) for a in DEFAULT_ANOMALIES]
    df = add_ground_truth(df, anomalies, label_col="ground_truth_anomaly")

    feature_cols = [c for c in df.columns if c != "ground_truth_anomaly"]
    gt = df["ground_truth_anomaly"].astype(int).values
    normal_mask = (gt == 0)

    df = impute_sensors(df, feature_cols, normal_mask)
    X_all = df[feature_cols].values.astype(np.float32, copy=False)

    if np.isnan(X_all).any() or np.isinf(X_all).any():
        raise ValueError("NaN/Inf still present after imputation (unexpected).")

    T = len(df)
    if T < args.window + 1:
        raise ValueError("Time series too short for the chosen window size.")

    X, _ = fit_transform_scaler(X_all, normal_mask)

    all_starts = build_window_starts(T, args.window, args.stride)
    trainval_starts = pure_normal_starts_fast(gt, args.window, args.stride)

    if len(trainval_starts) < 200:
        raise ValueError("Not enough pure-normal windows for train/val split.")

    # Chronological split
    n_w = len(trainval_starts)
    w_train_end = int(n_w * 0.7)
    w_val_end = int(n_w * 0.85)

    train_starts = trainval_starts[:w_train_end]
    val_starts = trainval_starts[w_train_end:w_val_end]

    train_loader = DataLoader(StartsWindowDataset(X, train_starts, args.window),
                              batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(StartsWindowDataset(X, val_starts, args.window),
                            batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=args.hidden, latent_size=args.latent).to(device)

    summary = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        min_delta=args.min_delta,
        use_early_stopping=(not args.no_early_stop),
    )

    # Score for AUC
    all_loader = DataLoader(StartsWindowDataset(X, all_starts, args.window),
                            batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)

    score = score_time_series_from_windows(
        df_index=df.index,
        T=T,
        all_starts=all_starts,
        window_size=args.window,
        model=model,
        all_loader=all_loader,
        device=device,
    )

    auc = compute_auc_metrics(gt, score)

    out_png = os.path.join(args.out_dir, "single_run_score_auc.png")

    info_box = (
        f"RESAMPLE={args.resample}\n"
        f"WINDOW={args.window}  STRIDE={args.stride}\n"
        f"HIDDEN={args.hidden}  LATENT={args.latent}\n"
        f"BATCH={args.batch}  EPOCHS(max)={args.epochs}\n"
        f"LR={args.lr}\n"
        f"epochs_ran={summary['epochs_ran']}\n"
        f"ROC_AUC={auc['roc_auc']:.3f}\n"
        f"PR_AUC={auc['pr_auc']:.3f}"
    )

    plot_score_and_roc(
        df_index=df.index,
        score=score,
        anomalies=anomalies,
        roc_data=auc,
        title="Single run - score & ROC",
        out_path=out_png,
        info_box=info_box,
    )

    print(f"Saved: {out_png}")
    cleanup(df_raw, df, X_all, X, model, train_loader, val_loader, all_loader)


if __name__ == "__main__":
    main()
