import numpy as np


def build_window_starts(T: int, window: int, stride: int) -> np.ndarray:
    """
    Compute window start indices.

    Parameters
    ----------
    T : int
        Length of the time series.
    window : int
        Window length.
    stride : int
        Step between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of start indices.
    """
    if T < window:
        return np.array([], dtype=np.int64)
    return np.arange(0, T - window + 1, stride, dtype=np.int64)


def pure_normal_starts_fast(gt01: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Return window starts such that the window contains no anomaly labels.

    This runs in O(T) using prefix sums and avoids slow Python loops.

    Parameters
    ----------
    gt01 : np.ndarray
        Ground truth labels (0/1) of length T.
    window : int
        Window length.
    stride : int
        Window stride.

    Returns
    -------
    np.ndarray
        Start indices for "pure normal" windows.
    """
    T = len(gt01)
    starts = build_window_starts(T, window, stride)
    if len(starts) == 0:
        return starts

    cs = np.zeros(T + 1, dtype=np.int64)
    cs[1:] = np.cumsum(gt01, dtype=np.int64)
    win_sums = cs[starts + window] - cs[starts]
    return starts[win_sums == 0]


def centers_from_starts(starts: np.ndarray, window: int) -> np.ndarray:
    """Compute center indices from window starts."""
    return starts + (window // 2)
