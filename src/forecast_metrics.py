import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_auc_metrics_limited_thresholds(
    gt: np.ndarray,
    score: np.ndarray,
    n_thresholds: int = 500
) -> dict:
    """
    Exact ROC-AUC and exact PR-AUC.
    n_thresholds is kept only for backward compatibility.
    """
    gt = gt.astype(int)

    mask = ~np.isnan(score)
    gt = gt[mask]
    score = score[mask]

    if len(np.unique(gt)) < 2:
        raise ValueError("Ground truth must contain both classes (0 and 1) to compute AUC.")

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