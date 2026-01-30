import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze grid AUC results.")
    p.add_argument("--results_csv", required=True, help="Path to grid_results_partial.csv")
    p.add_argument("--top", type=int, default=20, help="How many top runs to display.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results_csv)

    if "roc_auc" not in df.columns or "pr_auc" not in df.columns:
        raise ValueError("CSV must contain roc_auc and pr_auc columns.")

    print("\nTop runs by PR-AUC (often best for imbalanced anomaly detection):")
    cols = ["run_name", "pr_auc", "roc_auc", "WINDOW_SIZE", "STRIDE", "HIDDEN_SIZE", "LATENT_SIZE", "BATCH_SIZE", "runtime_s", "png_path"]
    print(df.sort_values("pr_auc", ascending=False)[cols].head(args.top).to_string(index=False))

    print("\nTop runs by ROC-AUC:")
    print(df.sort_values("roc_auc", ascending=False)[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
