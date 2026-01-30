from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class KnownAnomaly:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


DEFAULT_ANOMALIES = [
    KnownAnomaly("A1", pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:59")),
    KnownAnomaly("A2", pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 05:00:00")),
    KnownAnomaly("A3", pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 23:59:59")),
    KnownAnomaly("A4", pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),
]