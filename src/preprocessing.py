import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List


def add_ground_truth(
    df: pd.DataFrame,
    anomalies: List[tuple],
    label_col: str = "ground_truth_anomaly",
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = 0
    for _, start, end in anomalies:
        out.loc[(out.index >= start) & (out.index <= end), label_col] = 1
    return out


def resample_mean(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample time series using mean aggregation.

    Special values:
    - raw / none / native / keep : return the dataframe unchanged
    """
    if rule is None:
        return df.copy()

    rule_str = str(rule).strip().lower()
    if rule_str in {"raw", "none", "native", "keep"}:
        return df.copy()

    return df.resample(rule).mean()


def impute_sensors(df: pd.DataFrame, feature_cols: List[str], normal_mask: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    features = out[feature_cols].copy()

    features = features.ffill().bfill()

    if normal_mask.sum() > 0:
        normal_index = out.index[normal_mask]
        medians = features.loc[normal_index].median(numeric_only=True)
        features = features.fillna(medians)

    features = features.fillna(0.0)
    out.loc[:, feature_cols] = features.values.astype("float32", copy=False)
    return out


def fit_transform_scaler(X_all: np.ndarray, normal_mask: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    if normal_mask.sum() < 1000:
        raise ValueError("Not enough normal points to fit a scaler reliably (need >= 1000).")

    scaler = StandardScaler()
    scaler.fit(X_all[normal_mask])
    X_scaled = scaler.transform(X_all).astype("float32")

    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        raise ValueError("NaN/Inf detected after scaling (check data variance/outliers).")

    return X_scaled, scaler