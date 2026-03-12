import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import ensure_dir, cleanup
from src.dataset_specs import load_detection_source, materialize_detection_frame
from src.preprocessing import impute_sensors, fit_transform_scaler
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
    p = argparse.ArgumentParser(description="Single run: reconstruction-based anomaly scoring on configured datasets.")
    p.add_argument("--dataset", choices=["metropt3", "arc_mm_braking_5"], default="metropt3")
    p.add_argument("--data_dir", required=True, help="MetroPT3 data folder or extracted root folder for the arc dataset.")
    p.add_argument("--csv_path", default=None, help="Optional direct path to a CSV. Used only for metropt3.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--timestamp_col", default="timestamp", help="Timestamp column name for metropt3.")
    p.add_argument("--resample", default="1T", help="Pandas resample rule or 'raw' to keep native sampling.")

    p.add_argument("--window", type=int, default=360, help="Window size in points after resampling.")
    p.add_argument("--stride", type=int, default=20, help="Stride in points after resampling.")
    p.add_argument("--min_train_windows", type=int, default=50, help="Minimum number of pure-normal windows required.")

    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden size.")
    p.add_argument("--latent", type=int, default=4, help="Latent size.")
    p.add_argument("--batch", type=int, default=128, help="Batch size.")
    p.add_argument("--epochs", type=int, default=20, help="Max epochs.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    p.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Early stopping min delta.")
    p.add_argument("--threshold_points", type=int, default=1000, help="Number of score thresholds explored for ROC/F1.")
    p.add_argument("--no_early_stop", action="store_true", help="Disable early stopping.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    device = "cpu"

    source = load_detection_source(
        dataset=args.dataset,
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        timestamp_col=args.timestamp_col,
    )
    prepared = materialize_detection_frame(source, args.resample)

    df = prepared["df"]
    feature_cols = prepared["feature_cols"]
    normal_mask = prepared["normal_mask"]
    window_gt01 = prepared["window_gt01"]
    eval_label = prepared["eval_label"]
    segment_ids = prepared["segment_ids"]

    df = impute_sensors(df, feature_cols, normal_mask)
    X_all = df[feature_cols].values.astype(np.float32, copy=False)

    if np.isnan(X_all).any() or np.isinf(X_all).any():
        raise ValueError("NaN/Inf still present after imputation (unexpected).")

    T = len(df)
    if T < args.window + 1:
        raise ValueError("Time series too short for the chosen window size.")

    X, _ = fit_transform_scaler(X_all, normal_mask)

    all_starts = build_window_starts(T, args.window, args.stride, segment_ids=segment_ids)
    trainval_starts = pure_normal_starts_fast(
        window_gt01,
        args.window,
        args.stride,
        segment_ids=segment_ids,
    )

    if len(trainval_starts) < args.min_train_windows:
        raise ValueError(f"Not enough pure-normal windows for train/val split (need >= {args.min_train_windows}).")

    n_w = len(trainval_starts)
    w_train_end = int(n_w * 0.7)
    w_val_end = int(n_w * 0.85)

    raw_train_starts = trainval_starts[:w_train_end]
    raw_val_starts = trainval_starts[w_train_end:w_val_end]

    if len(raw_train_starts) == 0 or len(raw_val_starts) == 0:
        raise ValueError("Chronological split produced an empty train or validation set.")

    gap = args.window
    last_train_end = raw_train_starts[-1] + gap
    val_starts = raw_val_starts[raw_val_starts >= last_train_end]
    train_starts = raw_train_starts

    if len(val_starts) == 0:
        raise ValueError("Validation set became empty after enforcing a temporal gap.")

    train_loader = DataLoader(
        StartsWindowDataset(X, train_starts, args.window),
        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        StartsWindowDataset(X, val_starts, args.window),
        batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False,
    )

    model = LSTMAutoencoder(
        n_features=len(feature_cols),
        hidden_size=args.hidden,
        latent_size=args.latent,
    ).to(device)

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

    all_loader = DataLoader(
        StartsWindowDataset(X, all_starts, args.window),
        batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False,
    )

    score = score_time_series_from_windows(
        df_index=df.index,
        T=T,
        all_starts=all_starts,
        window_size=args.window,
        model=model,
        all_loader=all_loader,
        device=device,
    )

    auc = compute_auc_metrics(eval_label, score, n_thresholds=args.threshold_points)

    out_png = os.path.join(args.out_dir, "single_run_score_auc.png")
    metrics_json = os.path.join(args.out_dir, "single_run_metrics.json")

    if prepared["eval_kind"] == "full_labels":
        info_box = (
            f"DATASET={args.dataset}\n"
            f"RESAMPLE={args.resample}\n"
            f"WINDOW={args.window}  STRIDE={args.stride}\n"
            f"HIDDEN={args.hidden}  LATENT={args.latent}\n"
            f"BATCH={args.batch}  EPOCHS(max)={args.epochs}\n"
            f"LR={args.lr}\n"
            f"epochs_ran={summary['epochs_ran']}\n"
            f"ROC_AUC={auc['roc_auc']:.3f}\n"
            f"PR_AUC={auc['pr_auc']:.3f}\n"
            f"best_thr={auc['best_threshold']:.6f}\n"
            f"F1={auc['f1']:.3f}  P={auc['precision']:.3f}  R={auc['recall']:.3f}\n"
            f"TN={auc['TN']} FP={auc['FP']} FN={auc['FN']} TP={auc['TP']}"
        )
        roc_title = "ROC curve"
    else:
        info_box = (
            f"DATASET={args.dataset}\n"
            f"RESAMPLE={args.resample}\n"
            f"WINDOW={args.window}  STRIDE={args.stride}\n"
            f"HIDDEN={args.hidden}  LATENT={args.latent}\n"
            f"BATCH={args.batch}  EPOCHS(max)={args.epochs}\n"
            f"LR={args.lr}\n"
            f"epochs_ran={summary['epochs_ran']}\n"
            f"PROXY_ROC_AUC={auc['roc_auc']:.3f}\n"
            f"PROXY_PR_AUC={auc['pr_auc']:.3f}\n"
            f"best_thr={auc['best_threshold']:.6f}\n"
            f"F1={auc['f1']:.3f}  P={auc['precision']:.3f}  R={auc['recall']:.3f}\n"
            f"TN={auc['TN']} FP={auc['FP']} FN={auc['FN']} TP={auc['TP']}"
        )
        roc_title = "Proxy ROC (known healthy region vs rest)"

    plot_score_and_roc(
        df_index=df.index,
        score=score,
        title=f"Single run - {args.dataset}",
        out_path=out_png,
        info_box=info_box,
        anomalies=prepared["plot_anomalies"],
        known_normal_spans=prepared["plot_known_normal_spans"],
        roc_data=auc,
        score_label="Score (reconstruction error)",
        roc_title=roc_title,
    )

    metrics = {
        "dataset": args.dataset,
        "eval_kind": prepared["eval_kind"],
        "roc_auc": float(auc["roc_auc"]),
        "pr_auc": float(auc["pr_auc"]),
        "best_threshold": float(auc["best_threshold"]),
        "precision": float(auc["precision"]),
        "recall": float(auc["recall"]),
        "f1": float(auc["f1"]),
        "TN": int(auc["TN"]),
        "FP": int(auc["FP"]),
        "FN": int(auc["FN"]),
        "TP": int(auc["TP"]),
        "epochs_ran": int(summary["epochs_ran"]),
        "best_val_mse": float(summary["best_val_mse"]),
        "params": {
            "resample": args.resample,
            "window": args.window,
            "stride": args.stride,
            "hidden": args.hidden,
            "latent": args.latent,
            "batch": args.batch,
            "epochs": args.epochs,
            "lr": args.lr,
            "threshold_points": args.threshold_points,
        },
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved figure: {out_png}")
    print(f"Saved metrics: {metrics_json}")
    print(
        "Best operating point | "
        f"threshold={auc['best_threshold']:.6f} | "
        f"F1={auc['f1']:.3f} | "
        f"P={auc['precision']:.3f} | "
        f"R={auc['recall']:.3f} | "
        f"TN={auc['TN']} FP={auc['FP']} FN={auc['FN']} TP={auc['TP']}"
    )
    print(f"Result folder: {os.path.abspath(args.out_dir)}")

    cleanup(df, X_all, X, model, train_loader, val_loader, all_loader, score)


if __name__ == "__main__":
    main()