import argparse
from src.forecast_pipeline import run_single_forecast_experiment


def main():
    p = argparse.ArgumentParser(description="Run a single LSTM forecasting-based anomaly experiment.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--resample", default="1T")
    p.add_argument("--input_window", type=int, default=180)
    p.add_argument("--pred_horizon", type=int, default=30)
    p.add_argument("--stride", type=int, default=20)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=4)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--auc_points", type=int, default=500)
    args = p.parse_args()

    run_single_forecast_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        resample_rule=args.resample,
        input_window=args.input_window,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        hidden_size=args.hidden,
        latent_size=args.latent,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        auc_threshold_points=args.auc_points,
        device="cpu",
    )


if __name__ == "__main__":
    main()