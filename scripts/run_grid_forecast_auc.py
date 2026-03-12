import os
import gc
import itertools
import json
import pandas as pd

from src.forecast_pipeline import run_single_forecast_experiment


def build_forecast_grid(resample_rule: str) -> dict:
    return {
        "RESAMPLE_RULE": [resample_rule],
        "INPUT_WINDOW": [120, 180, 240],
        "PRED_HORIZON": [10, 30, 60],
        "STRIDE": [20, 30],
        "HIDDEN": [48, 64],
        "LATENT": [2, 4],
        "BATCH_SIZE": [64],
        "EPOCHS": [20],
        "LR": [1e-3],
        "AUC_POINTS": [500],
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Grid search for forecasting-based anomaly scoring.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--csv_path", default=None, help="Optional direct path to a CSV. Overrides data_dir search.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_dir", default=None, help="Existing run directory to resume. If omitted, create a new timestamped folder.")
    p.add_argument("--timestamp_col", default="timestamp", help="Timestamp column name.")
    p.add_argument("--resample", default="1T")
    args = p.parse_args()

    grid = build_forecast_grid(args.resample)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.run_dir is not None:
        run_dir = args.run_dir
        os.makedirs(run_dir, exist_ok=True)
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.out_dir, f"forecast_grid_{stamp}")
        os.makedirs(run_dir, exist_ok=True)

    param_grid_path = os.path.join(run_dir, "param_grid.json")
    if not os.path.exists(param_grid_path):
        with open(param_grid_path, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2)

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    out_csv = os.path.join(run_dir, "grid_results_partial.csv")

    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        if "run_name" in prev.columns:
            done = set(prev["run_name"].dropna().astype(str).tolist())
        print(f"Resume enabled: {len(done)} runs already done -> will be skipped.")

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        run_name = (
            f"run{i:04d}_res{params['RESAMPLE_RULE']}_"
            f"IW{params['INPUT_WINDOW']}_PH{params['PRED_HORIZON']}_S{params['STRIDE']}_"
            f"H{params['HIDDEN']}_Z{params['LATENT']}_B{params['BATCH_SIZE']}_"
            f"E{params['EPOCHS']}_LR{params['LR']}_AUC{params['AUC_POINTS']}"
        )
        out_run = os.path.join(run_dir, run_name)

        metrics_path = os.path.join(out_run, "forecast_metrics.json")
        if run_name in done and os.path.exists(metrics_path):
            print(f"[{i}/{len(combos)}] SKIP {run_name}")
            continue

        print(f"\n[{i}/{len(combos)}] {run_name}")

        try:
            ret = run_single_forecast_experiment(
                data_dir=args.data_dir,
                csv_path=args.csv_path,
                out_dir=out_run,
                timestamp_col=args.timestamp_col,
                resample_rule=params["RESAMPLE_RULE"],
                input_window=int(params["INPUT_WINDOW"]),
                pred_horizon=int(params["PRED_HORIZON"]),
                stride=int(params["STRIDE"]),
                hidden_size=int(params["HIDDEN"]),
                latent_size=int(params["LATENT"]),
                batch_size=int(params["BATCH_SIZE"]),
                epochs=int(params["EPOCHS"]),
                lr=float(params["LR"]),
                auc_threshold_points=int(params["AUC_POINTS"]),
                device="cpu",
            )

            m = ret["metrics"]

            row = {
                **params,
                "run_name": run_name,
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "best_threshold": m["best_threshold"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TP": m["TP"],
                "epochs_ran": m["epochs_ran"],
                "best_val_mse": m["best_val_mse"],
                "fig_path": ret["fig_path"],
            }

            print(
                f"OK | ROC_AUC={m['roc_auc']:.3f} PR_AUC={m['pr_auc']:.3f} "
                f"F1={m['f1']:.3f} | thr={m['best_threshold']:.6f} | "
                f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}"
            )

        except Exception as e:
            row = {**params, "run_name": run_name, "error": repr(e)}
            print(f"ERROR | {run_name} | {e!r}")

        write_header = not os.path.exists(out_csv)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=write_header, index=False)
        gc.collect()

    print(f"\nDone. Results: {out_csv}")
    print(f"Runs folder: {run_dir}")
    print(f"Result folder: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()