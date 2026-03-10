import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze grid AUC results.")
    p.add_argument("--results_csv", required=True, help="Path to a grid results CSV.")
    p.add_argument("--top", type=int, default=20, help="How many top runs to display.")
    return p.parse_args()


def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    if {"roc_auc", "pr_auc"}.issubset(df.columns):
        score_col = "roc_auc"
        cols = existing_cols(
            df,
            [
                "run_name", "pr_auc", "roc_auc",
                "WINDOW_SIZE", "STRIDE", "HIDDEN_SIZE", "LATENT_SIZE", "BATCH_SIZE",
                "epochs_ran", "val_mse_best", "runtime_s", "png_path",
            ],
        )
    elif {"roc_auc_approx", "pr_auc"}.issubset(df.columns):
        score_col = "roc_auc_approx"
        cols = existing_cols(
            df,
            [
                "run_name", "pr_auc", "roc_auc_approx",
                "INPUT_WINDOW", "PRED_HORIZON", "STRIDE", "HIDDEN", "LATENT", "BATCH_SIZE",
                "epochs_ran", "best_val_mse", "fig_path",
            ],
        )
    else:
        raise ValueError(
            "Unsupported CSV schema. Expected detection columns "
            "(roc_auc, pr_auc, WINDOW_SIZE, ...) or forecasting columns "
            "(roc_auc_approx, pr_auc, INPUT_WINDOW, ...)."
        )

    print("\nTop runs by PR-AUC:")
    print(df.sort_values("pr_auc", ascending=False)[cols].head(args.top).to_string(index=False))

    print(f"\nTop runs by {score_col}:")
    print(df.sort_values(score_col, ascending=False)[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()