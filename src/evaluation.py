import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


def compute_auc_metrics(gt: np.ndarray, score: np.ndarray) -> dict:
    """
    Compute AUC metrics based on a continuous anomaly score.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels (0/1).
    score : np.ndarray
        Continuous anomaly score (higher = more anomalous).

    Returns
    -------
    dict
        - roc_auc
        - pr_auc (average precision)
        - fpr, tpr (for ROC plotting)
    """
    if len(np.unique(gt)) < 2:
        raise ValueError("Ground truth must contain both classes (0 and 1) to compute AUC.")

    roc_auc = float(roc_auc_score(gt, score))
    pr_auc = float(average_precision_score(gt, score))
    fpr, tpr, _ = roc_curve(gt, score)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "fpr": fpr, "tpr": tpr}
