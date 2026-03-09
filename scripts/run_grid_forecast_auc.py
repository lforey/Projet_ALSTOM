import os, gc, itertools, json
import numpy as np
import pandas as pd

from src.forecast_pipeline import run_single_forecast_experiment


def build_forecast_grid(resample_rule: str) -> dict:
    return {
        "RESAMPLE_RULE": [resample_rule],
        "INPUT_WINDOW": [120, 180, 240],   # minutes if 1T
        "PRED_HORIZON": [10, 30, 60],      # 10/30/60 min ahead
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
    p = argparse.ArgumentParser(description="Grid search for forecasting-based anomaly scoring (AUC).")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resample", default="1T")
    args = p.parse_args()

    grid = build_forecast_grid(args.resample)
    os.makedirs(args.out_dir, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"forecast_grid_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "param_grid.json"), "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    results = []
    out_csv = os.path.join(run_dir, "grid_results_partial.csv")

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        run_name = (
            f"run{i:04d}_res{params['RESAMPLE_RULE']}_"
            f"IW{params['INPUT_WINDOW']}_PH{params['PRED_HORIZON']}_S{params['STRIDE']}_"
            f"H{params['HIDDEN']}_Z{params['LATENT']}_B{params['BATCH_SIZE']}_"
            f"E{params['EPOCHS']}_LR{params['LR']}_AUC{params['AUC_POINTS']}"
        )
        out_run = os.path.join(run_dir, run_name)

        print(f"\n[{i}/{len(combos)}] {run_name}")

        try:
            ret = run_single_forecast_experiment(
                data_dir=args.data_dir,
                out_dir=out_run,
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

            # Read metrics.json produced by the run
            import json as _json
            with open(os.path.join(out_run, "forecast_metrics.json"), "r", encoding="utf-8") as f:
                m = _json.load(f)

            results.append({
                **params,
                "run_name": run_name,
                "roc_auc_approx": m["roc_auc_approx"],
                "pr_auc": m["pr_auc"],
                "epochs_ran": m["epochs_ran"],
                "best_val_mse": m["best_val_mse"],
                "fig_path": ret["fig_path"],
            })

        except Exception as e:
            results.append({**params, "run_name": run_name, "error": repr(e)})

        # Always checkpoint partial CSV (so you can stop/resume safely)
        pd.DataFrame(results).to_csv(out_csv, index=False)

        # Cleanup between runs
        gc.collect()

    print(f"\nDone. Results: {out_csv}")
    print(f"Runs folder: {run_dir}")


if __name__ == "__main__":
    main()