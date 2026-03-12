import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def _candidate_thresholds(score: np.ndarray, n_thresholds: int) -> np.ndarray:
    unique_scores = np.unique(score)
    if len(unique_scores) <= n_thresholds:
        thresholds = unique_scores
    else:
        qs = np.linspace(0.0, 1.0, n_thresholds)
        thresholds = np.quantile(score, qs)
        thresholds = np.unique(thresholds)

    if len(thresholds) == 0:
        raise ValueError("Could not build threshold candidates from score.")

    return thresholds.astype(np.float64, copy=False)


def compute_auc_metrics_limited_thresholds(
    gt: np.ndarray,
    score: np.ndarray,
    n_thresholds: int = 500
) -> dict:
    """
    Forecasting counterpart of detection metrics:
    approximate ROC AUC, exact PR AUC, and best operating point by F1.
    """
    gt = gt.astype(int)

    mask = ~np.isnan(score)
    gt = gt[mask]
    score = score[mask]

    if len(np.unique(gt)) < 2:
        raise ValueError("Ground truth must contain both classes (0 and 1) to compute AUC.")

    pr_auc = float(average_precision_score(gt, score))
    thresholds = _candidate_thresholds(score, n_thresholds=n_thresholds)

    P = int(np.sum(gt == 1))
    N = int(np.sum(gt == 0))

    tpr = np.zeros(len(thresholds), dtype=np.float64)
    fpr = np.zeros(len(thresholds), dtype=np.float64)

    best = None

    for i, thr in enumerate(thresholds):
        pred = (score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        tpr[i] = tp / P if P > 0 else 0.0
        fpr[i] = fp / N if N > 0 else 0.0

        row = {
            "best_threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

        if best is None:
            best = row
        else:
            is_better = row["f1"] > best["f1"]
            same_f1 = np.isclose(row["f1"], best["f1"])
            fewer_fp = row["FP"] < best["FP"]
            higher_recall = row["recall"] > best["recall"]
            if is_better or (same_f1 and fewer_fp) or (same_f1 and row["FP"] == best["FP"] and higher_recall):
                best = row

    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    thresholds = thresholds[order]

    roc_auc = float(np.trapz(tpr, fpr))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        **best,
    }