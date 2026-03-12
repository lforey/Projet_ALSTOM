import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_ANOMALIES,
    ARC_MM_BRAKING_5_EVENTS,
    ARC_KNOWN_NORMAL_FRACTION,
)
from src.data_io import find_csv, load_raw_csv
from src.preprocessing import add_ground_truth, resample_mean

LABEL_COL = "ground_truth_anomaly"
NORMAL_COL = "known_normal_region"
PROXY_COL = "proxy_eval_label"
SEGMENT_COL = "segment_id"
META_COLS = {LABEL_COL, NORMAL_COL, PROXY_COL, SEGMENT_COL}


def _read_numeric_txt_series(txt_path: str) -> np.ndarray:
    raw = pd.read_csv(
        txt_path,
        sep=r"\s+|,|;",
        engine="python",
        header=None,
        comment="#",
    )
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    counts = numeric.notna().sum(axis=0)
    if counts.max() == 0:
        raise ValueError(f"No numeric data found in: {txt_path}")
    best_col = counts.idxmax()
    series = numeric.iloc[:, best_col].dropna().reset_index(drop=True)
    return series.to_numpy(dtype=np.float64, copy=False)


def _find_arc_event_files(data_dir: str, event_id: str) -> Dict[str, str]:
    txt_candidates = glob.glob(os.path.join(data_dir, "**", f"{event_id}_*.txt"), recursive=True)

    files = {"x": None, "vp": None, "vf": None, "ip": None, "ir": None}
    for path in txt_candidates:
        low = os.path.basename(path).lower()
        if low.endswith("_x.txt"):
            files["x"] = path
        elif low.endswith("_vp.txt"):
            files["vp"] = path
        elif low.endswith("_vf.txt"):
            files["vf"] = path
        elif low.endswith("_ip.txt"):
            files["ip"] = path
        elif low.endswith("_ir.txt"):
            files["ir"] = path

    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"Event {event_id} is incomplete under {data_dir}. Missing files for: {missing}"
        )

    return files


def _load_one_arc_event_raw(data_dir: str, event_id: str) -> pd.DataFrame:
    files = _find_arc_event_files(data_dir, event_id)

    x = _read_numeric_txt_series(files["x"])
    vp = _read_numeric_txt_series(files["vp"])
    vf = _read_numeric_txt_series(files["vf"])
    ip = _read_numeric_txt_series(files["ip"])
    ir = _read_numeric_txt_series(files["ir"])

    n = min(len(x), len(vp), len(vf), len(ip), len(ir))
    if n < 10:
        raise ValueError(f"Event {event_id} is too short after loading.")

    idx = pd.to_timedelta(x[:n], unit="s")
    df = pd.DataFrame(
        {
            "vp": vp[:n].astype("float32", copy=False),
            "ip": ip[:n].astype("float32", copy=False),
            "vf": vf[:n].astype("float32", copy=False),
            "ir": ir[:n].astype("float32", copy=False),
        },
        index=idx,
    )

    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_detection_source(
    dataset: str,
    data_dir: str,
    csv_path: str | None = None,
    timestamp_col: str = "timestamp",
) -> dict:
    if dataset == "metropt3":
        selected_csv = csv_path or find_csv(data_dir)
        df_raw = load_raw_csv(selected_csv, timestamp_col=timestamp_col)
        anomalies = [(a.name, a.start, a.end) for a in DEFAULT_ANOMALIES]
        return {
            "dataset": "metropt3",
            "df_raw": df_raw,
            "anomalies": anomalies,
        }

    if dataset == "arc_mm_braking_5":
        events = []
        for event_id in ARC_MM_BRAKING_5_EVENTS:
            events.append({
                "event_id": event_id,
                "df_raw": _load_one_arc_event_raw(data_dir, event_id),
            })
        return {
            "dataset": "arc_mm_braking_5",
            "events": events,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def _infer_step(index: pd.Index) -> pd.Timedelta:
    if len(index) >= 2:
        return index[1] - index[0]
    return pd.to_timedelta(1, unit="ms")


def materialize_detection_frame(source: dict, resample_rule: str) -> dict:
    if source["dataset"] == "metropt3":
        df = resample_mean(source["df_raw"], resample_rule)
        df = add_ground_truth(df, source["anomalies"], label_col=LABEL_COL)
        df[NORMAL_COL] = (df[LABEL_COL] == 0).astype(int)
        df[PROXY_COL] = df[LABEL_COL].astype(int)
        df[SEGMENT_COL] = 0

        feature_cols = [c for c in df.columns if c not in META_COLS]
        normal_mask = df[NORMAL_COL].astype(bool).values
        eval_label = df[LABEL_COL].astype(int).values

        return {
            "df": df,
            "feature_cols": feature_cols,
            "normal_mask": normal_mask,
            "window_gt01": (~normal_mask).astype(int),
            "eval_label": eval_label,
            "eval_kind": "full_labels",
            "plot_anomalies": source["anomalies"],
            "plot_known_normal_spans": [],
            "segment_ids": df[SEGMENT_COL].astype(int).values,
        }

    if source["dataset"] == "arc_mm_braking_5":
        event_frames: List[pd.DataFrame] = []
        known_spans: List[tuple] = []
        current_start = pd.Timestamp("2000-01-01 00:00:00")

        for seg_id, event in enumerate(source["events"]):
            df_evt = resample_mean(event["df_raw"], resample_rule)
            df_evt = df_evt.dropna(how="all").copy()
            if len(df_evt) < 10:
                raise ValueError(
                    f"Event {event['event_id']} became too short after resampling with rule={resample_rule}."
                )

            step = _infer_step(df_evt.index)
            new_index = pd.date_range(start=current_start, periods=len(df_evt), freq=step)
            df_evt.index = new_index

            n = len(df_evt)
            n_normal = max(1, int(np.floor(ARC_KNOWN_NORMAL_FRACTION * n)))
            df_evt[NORMAL_COL] = 0
            df_evt.iloc[:n_normal, df_evt.columns.get_loc(NORMAL_COL)] = 1
            df_evt[PROXY_COL] = (df_evt[NORMAL_COL] == 0).astype(int)
            df_evt[LABEL_COL] = np.nan
            df_evt[SEGMENT_COL] = seg_id

            known_spans.append(
                (f"{event['event_id']}_healthy", df_evt.index[0], df_evt.index[n_normal - 1])
            )
            event_frames.append(df_evt)

            current_start = new_index[-1] + (10 * step)

        df = pd.concat(event_frames, axis=0)
        feature_cols = [c for c in df.columns if c not in META_COLS]
        normal_mask = df[NORMAL_COL].astype(bool).values
        eval_label = df[PROXY_COL].astype(int).values

        return {
            "df": df,
            "feature_cols": feature_cols,
            "normal_mask": normal_mask,
            "window_gt01": (~normal_mask).astype(int),
            "eval_label": eval_label,
            "eval_kind": "weak_proxy_known_normal_vs_rest",
            "plot_anomalies": [],
            "plot_known_normal_spans": known_spans,
            "segment_ids": df[SEGMENT_COL].astype(int).values,
        }

    raise ValueError(f"Unsupported dataset: {source['dataset']}")