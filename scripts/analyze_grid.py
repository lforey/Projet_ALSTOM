import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze grid results.")
    p.add_argument("--results_csv", required=True, help="Path to grid_results_partial.csv")
    p.add_argument("--top", type=int, default=20, help="How many top runs to display.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    if {"proxy_roc_auc", "proxy_pr_auc"}.issubset(df.columns):
        cols = [
            "run_name", "dataset", "eval_kind", "proxy_pr_auc", "proxy_roc_auc",
            "f1", "precision", "recall", "TN", "FP", "FN", "TP", "best_threshold",
            "WINDOW_SIZE", "STRIDE", "HIDDEN_SIZE", "LATENT_SIZE", "BATCH_SIZE",
            "epochs_ran", "val_mse_best", "runtime_s", "png_path",
        ]
        cols = [c for c in cols if c in df.columns]

        print("\nTop runs by proxy PR-AUC (known healthy region vs rest):")
        print(df.sort_values("proxy_pr_auc", ascending=False)[cols].head(args.top).to_string(index=False))

        print("\nTop runs by proxy ROC-AUC (known healthy region vs rest):")
        print(df.sort_values("proxy_roc_auc", ascending=False)[cols].head(args.top).to_string(index=False))
        return

    if {"roc_auc", "pr_auc"}.issubset(df.columns):
        cols = [
            "run_name", "dataset", "eval_kind", "pr_auc", "roc_auc",
            "f1", "precision", "recall", "TN", "FP", "FN", "TP", "best_threshold",
            "WINDOW_SIZE", "STRIDE", "HIDDEN_SIZE", "LATENT_SIZE", "BATCH_SIZE",
            "INPUT_WINDOW", "PRED_HORIZON", "HIDDEN", "LATENT",
            "epochs_ran", "best_val_mse", "val_mse_best", "runtime_s", "png_path", "fig_path",
        ]
        cols = [c for c in cols if c in df.columns]

        print("\nTop runs by PR-AUC:")
        print(df.sort_values("pr_auc", ascending=False)[cols].head(args.top).to_string(index=False))

        print("\nTop runs by ROC-AUC:")
        print(df.sort_values("roc_auc", ascending=False)[cols].head(args.top).to_string(index=False))

        if "f1" in df.columns:
            print("\nTop runs by F1:")
            print(df.sort_values("f1", ascending=False)[cols].head(args.top).to_string(index=False))
        return

    raise ValueError(
        "Unsupported CSV schema. Expected either detection/forecast columns "
        "(roc_auc, pr_auc) or weak-label proxy columns (proxy_roc_auc, proxy_pr_auc)."
    )


if __name__ == "__main__":
    main()