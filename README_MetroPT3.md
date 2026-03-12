# Projet_ALSTOM

# Alstom Time Series Anomaly Detection & Forecasting

This repository currently supports **two explicit dataset protocols** for the reconstruction-based pipeline:

1. **metropt3**  
   Multivariate compressor dataset with anomaly periods explicitly defined in `src/config.py`.

2. **arc_mm_braking_5**  
   Pantograph electric arc dataset handled explicitly as the concatenation of:
   `MM_B_1`, `MM_B_2`, `MM_B_3`, `MM_B_4`, `MM_B_5`.

   For this second dataset:
   - the first 20% of each event is treated as **known healthy**
   - the remaining 80% is treated as **unknown**
   - grid-search metrics are therefore **proxy metrics**:
     separation between known healthy region and the rest

Forecasting remains documented and usable for **MetroPT3 only** in this repository version.

---

## Important convention

All window parameters are expressed in **number of points after optional resampling**.

Examples:
- `--window 360` means **360 resampled points**
- `--stride 20` means **20 resampled points**

Use:
- `--resample 1T` for 1-minute aggregation
- `--resample 20S` for 20-second aggregation
- `--resample raw` to keep the native sampling unchanged

No automatic conversion from "minutes" to another scale is performed.

---

## Project structure

- `src/`      : reusable Python modules
- `scripts/`  : executable scripts
- `data/`     : local data directory
- `outputs/`  : plots and CSV summaries

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Depending on your system, you may need:

```bash
set PYTHONPATH=.
```

instead of:

```bash
PYTHONPATH=.
```

Reconstruction-Based Detection
Supported datasets
A) MetroPT3

Expected input:

one CSV file

one timestamp column (default: timestamp)

numerical sensor columns

Known anomaly periods are defined in src/config.py.

B) Arc pantograph dataset (arc_mm_braking_5)

Expected input:

the extracted archive root folder

the repository will look explicitly for:

MM_B_1

MM_B_2

MM_B_3

MM_B_4

MM_B_5

For each event, the code expects the text files:

_x.txt

_Vp.txt

_Vf.txt

_Ip.txt

_IR.txt

The five events are concatenated internally.
Windows are forced to stay inside each event.

Single run — MetroPT3

```bash
PYTHONPATH=. python scripts/run_single.py \
  --dataset metropt3 \
  --data_dir ./data \
  --out_dir ./outputs/single_run_metro \
  --resample 1T \
  --window 360 \
  --stride 20 \
  --hidden 64 \
  --latent 4
```

Grid search — MetroPT3

```bash
PYTHONPATH=. python scripts/run_grid_auc.py \
  --dataset metropt3 \
  --data_dir ./data \
  --out_dir ./outputs/grid_auc_metro \
  --resample 1T
```

Outputs:

one figure per run

one CSV file: grid_results_partial.csv

Single run — Arc dataset

```bash
PYTHONPATH=. python scripts/run_single.py \
  --dataset arc_mm_braking_5 \
  --data_dir ./data/Dataset_Arc_Events_v3_extracted \
  --out_dir ./outputs/single_run_arc \
  --resample raw \
  --window 1024 \
  --stride 64 \
  --hidden 48 \
  --latent 4
```

Notes:

raw keeps the native event sampling

the evaluation is a proxy evaluation

the shaded green regions correspond to the known healthy prefixes

Grid search — Arc dataset

```bash
PYTHONPATH=. python scripts/run_grid_auc.py \
  --dataset arc_mm_braking_5 \
  --data_dir ./data/Dataset_Arc_Events_v3_extracted \
  --out_dir ./outputs/grid_auc_arc \
  --resample raw
```

Outputs:

one figure per run

one CSV file: grid_results_partial.csv

proxy metrics:

proxy_roc_auc

proxy_pr_auc

These are not true anomaly AUC metrics.
They measure the separation between the known healthy prefix and the remaining unknown region.

Analyze grid results
MetroPT3 / standard detection
python scripts/analyze_grid.py --results_csv ./outputs/grid_auc_metro/<RUN_FOLDER>/grid_results_partial.csv
Arc dataset / weak-label proxy evaluation
python scripts/analyze_grid.py --results_csv ./outputs/grid_auc_arc/<RUN_FOLDER>/grid_results_partial.csv
Forecasting
python scripts/analyze_grid.py --results_csv ./outputs/grid_forecast_auc/<RUN_FOLDER>/grid_results_partial.csv
Forecast-Based Anomaly Prediction

Forecasting scripts are kept in the repository and remain usable for the original MetroPT3 workflow.

At this stage, the pantograph arc dataset is not integrated into the forecasting pipeline in this repository version.

Single forecasting run

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

Forecasting grid

```bash
PYTHONPATH=. python scripts/run_grid_forecast_auc.py \
  --data_dir ./data \
  --out_dir ./outputs/grid_forecast_auc \
  --resample 1T
```