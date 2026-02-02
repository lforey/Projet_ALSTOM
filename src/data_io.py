import os
import glob
import pandas as pd


# Columns to exclude from the feature set (redundant sensors, etc.)
EXCLUDED_COLUMNS = [
    "Reservoirs",  # perfectly correlated (corr=1) with TP3
]


def find_csv(data_dir: str, pattern: str = "*MetroPT*.csv") -> str:
    """
    Find a CSV file under data_dir recursively.

    Parameters
    ----------
    data_dir : str
        Root directory where CSV files are located.
    pattern : str
        Glob pattern to filter likely files.

    Returns
    -------
    str
        Path to the selected CSV.

    Raises
    ------
    FileNotFoundError
        If no CSV is found.
    """
    candidates = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No CSV found under: {data_dir}")

    # Prefer MetroPT3 naming if present (tiny convenience)
    preferred = [p for p in candidates if "MetroPT3" in os.path.basename(p)]
    return preferred[0] if preferred else candidates[0]


def load_raw_csv(csv_path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Load a raw CSV file and return a dataframe indexed by timestamp.

    Notes
    -----
    - Keeps only numeric columns (float32).
    - Drops rows with missing timestamps.
    - Removes user-defined excluded columns.
    - Sorts by time.

    Parameters
    ----------
    csv_path : str
        Path to the CSV.
    timestamp_col : str
        Name of timestamp column.

    Returns
    -------
    pd.DataFrame
        Time-indexed dataframe with numeric columns only.
    """
    header = pd.read_csv(csv_path, nrows=0)
    cols = header.columns.tolist()

    if timestamp_col not in cols:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found. Available: {cols}"
        )

    usecols = [c for c in cols if not str(c).startswith("Unnamed")]

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        parse_dates=[timestamp_col],
        low_memory=False,
    )

    df = (
        df
        .dropna(subset=[timestamp_col])
        .sort_values(timestamp_col)
        .set_index(timestamp_col)
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the CSV.")

    # Remove excluded columns (e.g. perfectly correlated sensors)
    numeric_cols = [
        c for c in numeric_cols
        if c not in EXCLUDED_COLUMNS
    ]

    if len(numeric_cols) == 0:
        raise ValueError(
            "All numeric columns were excluded. Check EXCLUDED_COLUMNS."
        )

    df = df[numeric_cols].astype("float32")

    return df
