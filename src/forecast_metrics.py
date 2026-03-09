import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def compute_auc_metrics_limited_thresholds(
    gt: np.ndarray,
    score: np.ndarray,
    n_thresholds: int = 500
) -> dict:
    """
    PR-AUC exact (fast), ROC-AUC approximated with quantile thresholds.
    Returns fpr/tpr for plotting.
    """
    gt = gt.astype(int)

    mask = ~np.isnan(score)
    gt = gt[mask]
    score = score[mask]

    if len(np.unique(gt)) < 2:
        raise ValueError("Ground truth must contain both classes (0 and 1) to compute AUC.")

    pr_auc = float(average_precision_score(gt, score))

    qs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.quantile(score, qs)

    P = np.sum(gt == 1)
    N = np.sum(gt == 0)

    tpr = np.zeros(len(thresholds), dtype=np.float64)
    fpr = np.zeros(len(thresholds), dtype=np.float64)

    for i, thr in enumerate(thresholds):
        pred = (score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
        tpr[i] = tp / P if P > 0 else 0.0
        fpr[i] = fp / N if N > 0 else 0.0

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    thresholds = thresholds[order]

    roc_auc = float(np.trapz(tpr, fpr))
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}