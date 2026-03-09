import numpy as np
import torch
from torch.utils.data import Dataset


def build_forecast_starts(T: int, input_window: int, pred_horizon: int, stride: int) -> np.ndarray:
    """
    Start indices s such that:
      past  = X[s : s+input_window]
      future= X[s+input_window : s+input_window+pred_horizon]
    is valid within [0, T).
    """
    max_start = T - (input_window + pred_horizon)
    if max_start < 0:
        return np.array([], dtype=np.int64)
    return np.arange(0, max_start + 1, stride, dtype=np.int64)


def pure_normal_forecast_starts(gt01: np.ndarray, input_window: int, pred_horizon: int, stride: int) -> np.ndarray:
    """
    Keep starts where the FULL interval (past+future) contains no anomaly.
    Uses cumulative sum for O(T) selection.
    """
    T = len(gt01)
    starts = build_forecast_starts(T, input_window, pred_horizon, stride)
    if len(starts) == 0:
        return starts

    span = input_window + pred_horizon
    cs = np.zeros(T + 1, dtype=np.int64)
    cs[1:] = np.cumsum(gt01.astype(np.int64), dtype=np.int64)

    win_sums = cs[starts + span] - cs[starts]
    return starts[win_sums == 0]


class ForecastWindowDataset(Dataset):
    """
    Returns (x_past, x_future)
    - x_past:  [IW, F]
    - x_future:[PH, F]
    """
    def __init__(self, X: np.ndarray, starts: np.ndarray, input_window: int, pred_horizon: int):
        self.X = X.astype(np.float32, copy=False)
        self.starts = starts.astype(np.int64, copy=False)
        self.iw = int(input_window)
        self.ph = int(pred_horizon)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x_past = self.X[s : s + self.iw]
        x_fut  = self.X[s + self.iw : s + self.iw + self.ph]
        return torch.from_numpy(x_past), torch.from_numpy(x_fut)