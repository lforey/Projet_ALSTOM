# Projet_ALSTOM

# Alstom Time Series Anomaly Detection & Forecasting

This project implements two complementary pipelines for multivariate time series:

1. **Anomaly Detection (Reconstruction-based)**  
   Temporal LSTM autoencoder that detects abnormal behavior using reconstruction error.

2. **Anomaly Prediction (Forecasting-based)**  
   LSTM encoder–decoder that predicts future behavior and scores anomalies using prediction error.

The goal is to detect long and subtle changes in system behavior that may not be
obvious when looking at individual signals.

---

## Project structure

- src/        : reusable Python modules (data, model, training, evaluation)
- scripts/    : executable scripts (single run, grid search)
- data/       : local data directory (not versioned)
- outputs/    : results (plots, JSON metrics, CSV summaries)

Detection and forecasting pipelines are independent.
You can switch between them without modifying existing files.

---

## Data

Data is not included in this repository.

Place your CSV file in the `data/` directory.

The CSV must contain:
- a timestamp column (default: `timestamp`)
- numerical sensor columns

The pipeline automatically:
- resamples time
- imputes missing values
- scales using only normal periods
- builds sliding windows

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You might have to use:

```bash
set PYTHONPATH=.
```

instead of:

```bash
PYTHONPATH=.
```

depending on your system.

---

# Reconstruction-Based Anomaly Detection

This is the original LSTM autoencoder pipeline.

The model reconstructs past windows and uses reconstruction error as anomaly score.

## Run a single detection experiment

```bash
PYTHONPATH=. python scripts/run_single.py \
  --data_dir ./data \
  --out_dir ./outputs/single_run \
  --resample 1T \
  --window 360 \
  --stride 20 \
  --hidden 64 \
  --latent 4
```

---

## Run an AUC grid search (Detection)

```bash
PYTHONPATH=. python scripts/run_grid_auc.py \
  --data_dir ./data \
  --out_dir ./outputs/grid_auc \
  --resample 1T
```

Grid search exports:

- One figure per run (score timeline + AUC information)
- A CSV (`grid_results_partial.csv`) appended after each run
- To resume an interrupted grid, relaunch the script with `--run_dir <existing_run_folder>`

---

# Forecast-Based Anomaly Prediction

This pipeline predicts future behavior instead of reconstructing the past.

Instead of learning:

> "What does normal look like?"

It learns:

> "Given the past, what should the future look like?"

Anomaly score = prediction error.

This enables early-warning analysis.

---

## Run a single forecasting experiment

```bash
PYTHONPATH=. python scripts/run_single_forecast.py \
  --data_dir ./data \
  --out_dir ./outputs/single_forecast \
  --resample 1T \
  --input_window 180 \
  --pred_horizon 30 \
  --stride 20 \
  --hidden 64 \
  --latent 4 \
  --auc_points 500
```

Parameters:

- `input_window` : size of past window (minutes if 1T)
- `pred_horizon` : future horizon to predict
- `stride` : window shift
- `hidden` / `latent` : model capacity
- `auc_points` : number of thresholds used for ROC approximation

---

## Run a forecasting AUC grid search

```bash
PYTHONPATH=. python scripts/run_grid_forecast_auc.py \
  --data_dir ./data \
  --out_dir ./outputs/grid_forecast_auc \
  --resample 1T
```

Outputs:

- One folder per run
- A score plot per run
- A `forecast_metrics.json` file
- A `grid_results_partial.csv` file
- To resume an interrupted forecasting grid, relaunch the script with `--run_dir <existing_run_folder>`

---

## Analyze grid results

Detection:

```bash
python scripts/analyze_grid.py --results_csv ./outputs/grid_auc/<RUN_FOLDER>/grid_results_partial.csv
```

Forecasting:

```bash
python scripts/analyze_grid.py --results_csv ./outputs/grid_forecast_auc/<RUN_FOLDER>/grid_results_partial.csv
```

---

## Notes

For anomaly detection with extreme class imbalance:

- PR-AUC is often more informative than ROC-AUC.
- ROC-AUC is approximated using a limited number of thresholds for efficiency.
- Threshold-free evaluation is preferred over fixed F1 selection.

Reconstruction-based detection identifies abnormal segments.

Forecast-based prediction evaluates whether future behavior is predictable from past dynamics.

Both approaches can be compared under identical preprocessing conditions.