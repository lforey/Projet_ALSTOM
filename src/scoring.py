import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .windowing import centers_from_starts
from .training import reconstruction_errors


def score_time_series_from_windows(
    df_index: pd.Index,
    T: int,
    all_starts: np.ndarray,
    window_size: int,
    model,
    all_loader: DataLoader,
    device: str,
) -> np.ndarray:
    """
    Produce a continuous score for each timestamp.

    Approach:
    - Compute window reconstruction error for each window.
    - Place each window error at the window center timestamp.
    - Interpolate missing points to get a per-timestep score.

    Returns
    -------
    np.ndarray
        Score array of shape [T], float32.
    """
    win_errs = reconstruction_errors(model, all_loader, device)
    centers = centers_from_starts(all_starts, window_size)

    score = np.full(T, np.nan, dtype=np.float32)
    score[centers] = win_errs.astype(np.float32, copy=False)

    score_interp = (
        pd.Series(score, index=df_index)
        .interpolate(limit_direction="both")
        .values.astype(np.float32, copy=False)
    )
    return score_interp
