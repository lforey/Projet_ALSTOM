import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_io import find_csv, load_raw_csv
from src.config import DEFAULT_ANOMALIES

from src.forecast_dataset import (
    build_forecast_starts,
    pure_normal_forecast_starts,
    ForecastWindowDataset,
)
from src.forecast_model import LSTMForecaster
from src.forecast_metrics import compute_auc_metrics_limited_thresholds

def normalize_anomaly_periods(anomaly_periods):
    normalized = []
    for a in anomaly_periods:
        if hasattr(a, "name") and hasattr(a, "start") and hasattr(a, "end"):
            normalized.append((a.name, pd.Timestamp(a.start), pd.Timestamp(a.end)))
        else:
            name, start, end = a
            normalized.append((name, pd.Timestamp(start), pd.Timestamp(end)))
    return normalized


def add_ground_truth(df: pd.DataFrame, anomaly_periods) -> pd.DataFrame:
    df = df.copy()
    df["ground_truth_anomaly"] = 0
    for _, start, end in anomaly_periods:
        df.loc[(df.index >= start) & (df.index <= end), "ground_truth_anomaly"] = 1
    return df


def impute_ffill_bfill_median(df: pd.DataFrame, feature_cols: list, gt01: np.ndarray) -> pd.DataFrame:
    df_features = df[feature_cols].copy()
    df_features = df_features.ffill().bfill()

    normal_mask = (gt01 == 0)
    med = df_features.loc[df.index[normal_mask]].median(numeric_only=True)
    df_features = df_features.fillna(med).fillna(0.0)

    df.loc[:, feature_cols] = df_features.values.astype(np.float32, copy=False)
    return df


def forecast_errors(model, loader, pred_horizon: int, device: str) -> np.ndarray:
    model.eval()
    errs = []
    with torch.no_grad():
        for x_past, x_fut in loader:
            x_past = x_past.to(device)
            x_fut = x_fut.to(device)
            y_pred = model(x_past, pred_horizon=pred_horizon)
            mse = torch.mean((y_pred - x_fut) ** 2, dim=(1, 2))  # [B]
            errs.append(mse.detach().cpu().numpy())
    return np.concatenate(errs, axis=0) if errs else np.array([], dtype=np.float32)


def run_single_forecast_experiment(
    data_dir: str,
    out_dir: str,
    resample_rule: str = "1T",
    input_window: int = 180,
    pred_horizon: int = 30,
    stride: int = 20,
    hidden_size: int = 64,
    latent_size: int = 4,
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    early_stopping: bool = True,
    patience: int = 7,
    min_delta: float = 1e-4,
    auc_threshold_points: int = 500,
    device: str = "cpu",
    anomaly_periods=DEFAULT_ANOMALIES,
    csv_path: str = None,
    timestamp_col: str = "timestamp",
):
    os.makedirs(out_dir, exist_ok=True)

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)

    anomaly_periods = normalize_anomaly_periods(anomaly_periods)

    csv_path = csv_path or find_csv(data_dir)
    df_raw = load_raw_csv(csv_path, timestamp_col=timestamp_col)

    # Resample + GT
    df = df_raw.resample(resample_rule).mean()
    df = add_ground_truth(df, anomaly_periods)

    feature_cols = [c for c in df.columns if c != "ground_truth_anomaly"]
    gt = df["ground_truth_anomaly"].astype(int).values

    # Impute
    df = impute_ffill_bfill_median(df, feature_cols, gt)

    X_all = df[feature_cols].values.astype(np.float32, copy=False)
    if np.isnan(X_all).any() or np.isinf(X_all).any():
        raise ValueError("NaN/Inf in features after imputation.")

    T = len(df)
    if T < (input_window + pred_horizon + 1):
        raise ValueError("Time series too short for input_window + pred_horizon.")

    normal_mask = (gt == 0)
    if normal_mask.sum() < 1000:
        raise ValueError("Not enough normal points to fit scaler.")

    # Scale on normal only
    scaler = StandardScaler()
    scaler.fit(X_all[normal_mask])
    X = scaler.transform(X_all).astype(np.float32)

    # Window starts
    all_starts = build_forecast_starts(T, input_window, pred_horizon, stride)
    trainval_starts = pure_normal_forecast_starts(gt, input_window, pred_horizon, stride)

    if len(trainval_starts) < 200:
        raise ValueError("Not enough normal windows for training/validation.")

        # Chrono split with temporal gap to avoid overlap leakage
    n_w = len(trainval_starts)
    w_train_end = int(n_w * 0.7)
    w_val_end = int(n_w * 0.85)

    raw_train_starts = trainval_starts[:w_train_end]
    raw_val_starts = trainval_starts[w_train_end:w_val_end]

    if len(raw_train_starts) == 0 or len(raw_val_starts) == 0:
        raise ValueError("Chronological split produced an empty train or validation set.")

    gap = input_window + pred_horizon
    last_train_end = raw_train_starts[-1] + gap
    val_starts = raw_val_starts[raw_val_starts >= last_train_end]

    if len(val_starts) == 0:
        raise ValueError("Validation set became empty after enforcing a temporal gap.")

    train_starts = raw_train_starts

    train_loader = DataLoader(
        ForecastWindowDataset(X, train_starts, input_window, pred_horizon),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        ForecastWindowDataset(X, val_starts, input_window, pred_horizon),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # Train
    model = LSTMForecaster(len(feature_cols), hidden_size, latent_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    epochs_ran = 0

    for ep in range(1, epochs + 1):
        epochs_ran = ep
        model.train()
        losses = []

        for x_past, x_fut in train_loader:
            x_past = x_past.to(device)
            x_fut = x_fut.to(device)

            y_pred = model(x_past, pred_horizon=pred_horizon)
            loss = loss_fn(y_pred, x_fut)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss during training.")

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        val_errs = forecast_errors(model, val_loader, pred_horizon, device)
        val_mse = float(np.mean(val_errs)) if len(val_errs) else float("inf")
        train_loss = float(np.mean(losses)) if len(losses) else float("inf")

        print(f"Epoch {ep:02d} | train_loss={train_loss:.6f} | val_mse={val_mse:.6f}")

        improvement = best_val - val_mse
        if improvement > 0:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping:
            if improvement > min_delta:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {patience} epochs without improvement > {min_delta}.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val_mse retained: {best_val:.6f} (epochs_ran={epochs_ran})")

    # Score all windows
    all_loader = DataLoader(
        ForecastWindowDataset(X, all_starts, input_window, pred_horizon),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    win_errs = forecast_errors(model, all_loader, pred_horizon, device)

    # Place score at the START of predicted future (t = s + input_window)
    anchor = all_starts + input_window
    score = np.full(T, np.nan, dtype=np.float32)
    score[anchor] = win_errs

    score_interp = pd.Series(score, index=df.index).interpolate(limit_direction="both").values.astype(np.float32)

    # AUC metrics
    auc = compute_auc_metrics_limited_thresholds(gt, score_interp, n_thresholds=auc_threshold_points)

    # Plot
    fig_path = os.path.join(out_dir, "forecast_score.png")
    plt.figure(figsize=(16, 5))
    plt.plot(df.index, score_interp, linewidth=0.8, label="Forecast error score")

    # Known anomalies shading
    for i, (name, start, end) in enumerate(anomaly_periods):
        plt.axvspan(start, end, alpha=0.2, label="Known anomalies" if i == 0 else None)

    plt.title("Forecast-based anomaly score (LSTM forecaster)")
    plt.xlabel("Time")
    plt.ylabel("Prediction error (MSE)")
    plt.grid(True, linewidth=0.3)

    # Info box
    info = (
        f"RESAMPLE={resample_rule}\n"
        f"IW={input_window}  PH={pred_horizon}  STRIDE={stride}\n"
        f"H={hidden_size}  Z={latent_size}  B={batch_size}\n"
        f"E(max)={epochs}  LR={lr}\n"
        f"ROC_AUC~={auc['roc_auc']:.3f}\n"
        f"PR_AUC={auc['pr_auc']:.3f}\n"
        f"ROC_pts={auc_threshold_points}"
    )
    ax = plt.gca()
    ax.text(
        0.995, 0.995, info,
        transform=ax.transAxes,
        fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close("all")

    # Save metrics
    metrics_path = os.path.join(out_dir, "forecast_metrics.json")
    import json
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "roc_auc_approx": auc["roc_auc"],
                "pr_auc": auc["pr_auc"],
                "epochs_ran": epochs_ran,
                "best_val_mse": best_val,
                "params": {
                    "resample_rule": resample_rule,
                    "input_window": input_window,
                    "pred_horizon": pred_horizon,
                    "stride": stride,
                    "hidden_size": hidden_size,
                    "latent_size": latent_size,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "auc_threshold_points": auc_threshold_points,
                },
            },
            f,
            indent=2
        )

    # Cleanup
    del df_raw, df, X_all, X, scaler, model, optim
    gc.collect()

    return {"fig_path": fig_path, "metrics_path": metrics_path}