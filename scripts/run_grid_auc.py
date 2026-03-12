import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
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
    p = argparse.ArgumentParser(description="Grid search for reconstruction-based anomaly scoring on configured datasets.")
    p.add_argument("--dataset", choices=["metropt3", "arc_mm_braking_5"], default="metropt3")
    p.add_argument("--data_dir", required=True, help="MetroPT3 data folder or extracted root folder for the arc dataset.")
    p.add_argument("--csv_path", default=None, help="Optional direct path to CSV. Used only for metropt3.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--run_dir", default=None, help="Existing run directory to resume. If omitted, create a new timestamped folder.")
    p.add_argument("--timestamp_col", default="timestamp", help="Timestamp column name for metropt3.")
    p.add_argument("--resample", default="1T", help="Pandas resample rule or 'raw' to keep native sampling.")
    p.add_argument("--min_train_windows", type=int, default=50, help="Minimum number of pure-normal windows required.")
    p.add_argument("--threshold_points", type=int, default=1000, help="Number of score thresholds explored for ROC/F1.")
    return p.parse_args()


def build_param_grid(dataset: str, resample_rule: str) -> dict:
    if dataset == "metropt3":
        return {
            "RESAMPLE_RULE": [resample_rule],
            "WINDOW_SIZE": [180, 195, 210, 225, 240],
            "STRIDE": [15, 20, 30],
            "HIDDEN_SIZE": [48, 64],
            "LATENT_SIZE": [2, 3, 4],
            "BATCH_SIZE": [64],
            "EPOCHS": [20],
            "LR": [1e-3],
        }

    return {
        "RESAMPLE_RULE": [resample_rule],
        "WINDOW_SIZE": [256, 512, 1024, 2048],
        "STRIDE": [32, 64, 128],
        "HIDDEN_SIZE": [32, 48, 64],
        "LATENT_SIZE": [2, 4, 8],
        "BATCH_SIZE": [64],
        "EPOCHS": [20],
        "LR": [1e-3],
    }


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

    grid = build_param_grid(args.dataset, args.resample)

    if args.run_dir is not None:
        run_dir = args.run_dir
        ensure_dir(run_dir)
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.out_dir, f"grid_auc_{args.dataset}_{stamp}")
        ensure_dir(run_dir)

    param_grid_path = os.path.join(run_dir, "param_grid.json")
    if not os.path.exists(param_grid_path):
        with open(param_grid_path, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2)

    out_csv = os.path.join(run_dir, "grid_results_partial.csv")

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Total runs: {len(combos)}")
    print(f"Outputs: {run_dir}")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        if "run_name" in prev.columns:
            done = set(prev["run_name"].dropna().astype(str).tolist())
        print(f"Resume enabled: {len(done)} runs already done -> will be skipped.")

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        resample_rule = params["RESAMPLE_RULE"]
        window_size = int(params["WINDOW_SIZE"])
        stride = int(params["STRIDE"])
        hidden_size = int(params["HIDDEN_SIZE"])
        latent_size = int(params["LATENT_SIZE"])
        batch_size = int(params["BATCH_SIZE"])
        epochs = int(params["EPOCHS"])
        lr = float(params["LR"])

        run_name = (
            f"run{idx:04d}_{args.dataset}_res{resample_rule}_W{window_size}_S{stride}_"
            f"H{hidden_size}_Z{latent_size}_B{batch_size}_E{epochs}_LR{lr}"
        )
        out_png = os.path.join(run_dir, run_name + ".png")

        if run_name in done and os.path.exists(out_png):
            print(f"[{idx}/{len(combos)}] SKIP {run_name}")
            continue

        print(f"\n[{idx}/{len(combos)}] {run_name}")
        t0 = time.time()

        try:
            prepared = materialize_detection_frame(source, resample_rule)
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
            if T < window_size + 1:
                raise ValueError("Time series too short for chosen window size.")

            X, _ = fit_transform_scaler(X_all, normal_mask)

            all_starts = build_window_starts(T, window_size, stride, segment_ids=segment_ids)
            trainval_starts = pure_normal_starts_fast(
                window_gt01,
                window_size,
                stride,
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

            gap = window_size
            last_train_end = raw_train_starts[-1] + gap
            val_starts = raw_val_starts[raw_val_starts >= last_train_end]
            train_starts = raw_train_starts

            if len(val_starts) == 0:
                raise ValueError("Validation set became empty after enforcing a temporal gap.")

            train_loader = DataLoader(
                StartsWindowDataset(X, train_starts, window_size),
                batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
            )
            val_loader = DataLoader(
                StartsWindowDataset(X, val_starts, window_size),
                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
            )

            model = LSTMAutoencoder(
                n_features=len(feature_cols),
                hidden_size=hidden_size,
                latent_size=latent_size,
            ).to(device)

            summary = train_with_early_stopping(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                patience=7,
                min_delta=1e-4,
                use_early_stopping=True,
            )

            all_loader = DataLoader(
                StartsWindowDataset(X, all_starts, window_size),
                batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
            )

            score = score_time_series_from_windows(
                df_index=df.index,
                T=T,
                all_starts=all_starts,
                window_size=window_size,
                model=model,
                all_loader=all_loader,
                device=device,
            )

            auc = compute_auc_metrics(eval_label, score, n_thresholds=args.threshold_points)

            if prepared["eval_kind"] == "full_labels":
                info_box = (
                    f"DATASET={args.dataset}\n"
                    f"RESAMPLE={resample_rule}\n"
                    f"W={window_size}  STRIDE={stride}\n"
                    f"H={hidden_size}  Z={latent_size}\n"
                    f"B={batch_size}  E(max)={epochs}\n"
                    f"LR={lr}\n"
                    f"epochs_ran={summary['epochs_ran']}\n"
                    f"ROC_AUC={auc['roc_auc']:.3f}\n"
                    f"PR_AUC={auc['pr_auc']:.3f}\n"
                    f"best_thr={auc['best_threshold']:.6f}\n"
                    f"F1={auc['f1']:.3f}  P={auc['precision']:.3f}  R={auc['recall']:.3f}"
                )
                roc_title = "ROC curve"
            else:
                info_box = (
                    f"DATASET={args.dataset}\n"
                    f"RESAMPLE={resample_rule}\n"
                    f"W={window_size}  STRIDE={stride}\n"
                    f"H={hidden_size}  Z={latent_size}\n"
                    f"B={batch_size}  E(max)={epochs}\n"
                    f"LR={lr}\n"
                    f"epochs_ran={summary['epochs_ran']}\n"
                    f"PROXY_ROC_AUC={auc['roc_auc']:.3f}\n"
                    f"PROXY_PR_AUC={auc['pr_auc']:.3f}\n"
                    f"best_thr={auc['best_threshold']:.6f}\n"
                    f"F1={auc['f1']:.3f}  P={auc['precision']:.3f}  R={auc['recall']:.3f}"
                )
                roc_title = "Proxy ROC (known healthy region vs rest)"

            plot_score_and_roc(
                df_index=df.index,
                score=score,
                title=run_name,
                out_path=out_png,
                info_box=info_box,
                anomalies=prepared["plot_anomalies"],
                known_normal_spans=prepared["plot_known_normal_spans"],
                roc_data=auc,
                score_label="Score (reconstruction error)",
                roc_title=roc_title,
            )

            row = {
                **params,
                "dataset": args.dataset,
                "eval_kind": prepared["eval_kind"],
                "run_name": run_name,
                "png_path": out_png,
                "best_threshold": auc["best_threshold"],
                "precision": auc["precision"],
                "recall": auc["recall"],
                "f1": auc["f1"],
                "TN": auc["TN"],
                "FP": auc["FP"],
                "FN": auc["FN"],
                "TP": auc["TP"],
                "val_mse_best": summary["best_val_mse"],
                "epochs_ran": summary["epochs_ran"],
                "runtime_s": round(time.time() - t0, 2),
            }

            if prepared["eval_kind"] == "full_labels":
                row["roc_auc"] = auc["roc_auc"]
                row["pr_auc"] = auc["pr_auc"]
                print(
                    f"OK | ROC_AUC={auc['roc_auc']:.3f} PR_AUC={auc['pr_auc']:.3f} "
                    f"F1={auc['f1']:.3f} | thr={auc['best_threshold']:.6f} | "
                    f"TN={auc['TN']} FP={auc['FP']} FN={auc['FN']} TP={auc['TP']} | "
                    f"{row['runtime_s']}s"
                )
            else:
                row["proxy_roc_auc"] = auc["roc_auc"]
                row["proxy_pr_auc"] = auc["pr_auc"]
                print(
                    f"OK | PROXY_ROC_AUC={auc['roc_auc']:.3f} PROXY_PR_AUC={auc['pr_auc']:.3f} "
                    f"F1={auc['f1']:.3f} | thr={auc['best_threshold']:.6f} | "
                    f"TN={auc['TN']} FP={auc['FP']} FN={auc['FN']} TP={auc['TP']} | "
                    f"{row['runtime_s']}s"
                )

        except Exception as e:
            row = {
                **params,
                "dataset": args.dataset,
                "run_name": run_name,
                "png_path": out_png,
                "error": repr(e),
                "runtime_s": round(time.time() - t0, 2),
            }
            print(f"ERROR | {run_name} | {e!r}")

        write_header = not os.path.exists(out_csv)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=write_header, index=False)
        cleanup()

    print("\nDone.")
    print(f"Results CSV: {out_csv}")
    print(f"Images folder: {run_dir}")
    print(f"Result folder: {os.path.abspath(run_dir)}")
    cleanup()


if __name__ == "__main__":
    main()