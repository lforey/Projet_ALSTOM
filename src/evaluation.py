import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_auc_metrics(
    gt: np.ndarray,
    score: np.ndarray,
    n_thresholds: int = 500
) -> dict:
    """
    Compute exact ROC AUC and exact PR AUC.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels (0/1).
    score : np.ndarray
        Continuous anomaly score (higher = more anomalous).
    n_thresholds : int, optional
        Kept only for backward compatibility. Not used anymore.

    Returns
    -------
    dict
        - roc_auc   : float
        - pr_auc    : float
        - fpr       : np.ndarray
        - tpr       : np.ndarray
        - thresholds: np.ndarray
    """
    gt = gt.astype(int)

    mask = ~np.isnan(score)
    gt = gt[mask]
    score = score[mask]

    if len(np.unique(gt)) < 2:
        raise ValueError(
            "Ground truth must contain both classes (0 and 1) to compute AUC."
        )

    pr_auc = float(average_precision_score(gt, score))
    roc_auc = float(roc_auc_score(gt, score))
    fpr, tpr, thresholds = roc_curve(gt, score)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }