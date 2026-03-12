import numpy as np


def build_window_starts(T: int, window: int, stride: int, segment_ids: np.ndarray | None = None) -> np.ndarray:
    """
    Compute window start indices.

    If segment_ids is provided, windows are generated independently
    inside each segment so that no window crosses a segment boundary.
    """
    if T < window:
        return np.array([], dtype=np.int64)

    if segment_ids is None:
        return np.arange(0, T - window + 1, stride, dtype=np.int64)

    segment_ids = np.asarray(segment_ids)
    if len(segment_ids) != T:
        raise ValueError("segment_ids must have length T.")

    cuts = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [T]))

    starts_list = []
    for seg_start, seg_end in zip(bounds[:-1], bounds[1:]):
        seg_len = seg_end - seg_start
        if seg_len >= window:
            local = np.arange(0, seg_len - window + 1, stride, dtype=np.int64)
            starts_list.append(seg_start + local)

    if not starts_list:
        return np.array([], dtype=np.int64)

    return np.concatenate(starts_list).astype(np.int64, copy=False)


def pure_normal_starts_fast(
    gt01: np.ndarray,
    window: int,
    stride: int,
    segment_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return window starts such that the window contains no anomaly labels.

    If segment_ids is provided, windows are restricted to remain inside
    each segment.
    """
    T = len(gt01)
    starts = build_window_starts(T, window, stride, segment_ids=segment_ids)
    if len(starts) == 0:
        return starts

    cs = np.zeros(T + 1, dtype=np.int64)
    cs[1:] = np.cumsum(gt01, dtype=np.int64)
    win_sums = cs[starts + window] - cs[starts]
    return starts[win_sums == 0]


def centers_from_starts(starts: np.ndarray, window: int) -> np.ndarray:
    return starts + (window // 2)