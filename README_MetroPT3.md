# Projet_ALSTOM

# Alstom Anomaly Detection

This project implements an anomaly detection pipeline for multivariate time series
using a temporal LSTM autoencoder.

The goal is to detect long and subtle changes in system behavior that may not be
obvious when looking at individual signals.

## Project structure

- src/        : reusable Python modules (data, model, training, evaluation)
- scripts/    : executable scripts (single run, grid search)
- data/       : local data directory (not versioned)
- outputs/    : results (plots, CSV summaries)

## Data

Data is not included in this repository.

Place your CSV file in the `data/` directory.
The CSV must contain:
- a timestamp column (default: `timestamp`)
- numerical sensor columns

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Run a single experiment

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

you might have to use 'set PYTHONPATH=.' instead of 'PYTHONPATH=.' 

## Run an AUC grid search

```bash
PYTHONPATH=. python scripts/run_grid_auc.py \
  --data_dir ./data \
  --out_dir ./outputs/grid_auc \
  --resample 20S
```


Grid search exports:

One figure per run (score timeline + ROC curve)

A CSV (grid_results_partial.csv) that is appended after each run (safe to stop/resume)

## Analyze grid results

```bash
python scripts/analyze_grid.py --results_csv ./outputs/grid_auc/<RUN_FOLDER>/grid_results_partial.csv
```

## Notes

For anomaly detection with extreme class imbalance, PR-AUC is often more informative than ROC-AUC.

All thresholds are handled by AUC; you can still inspect a fixed threshold for sanity checks.



