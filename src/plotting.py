import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def plot_score_and_roc(
    df_index: pd.Index,
    score: np.ndarray,
    anomalies: List[Tuple[str, pd.Timestamp, pd.Timestamp]],
    roc_data: Dict[str, np.ndarray],
    title: str,
    out_path: str,
    info_box: str,
) -> None:
    """
    Save a figure with:
    - score timeline with known anomaly spans
    - ROC curve

    This is intentionally kept simple: plots should be easy to read at a glance.
    """
    plt.figure(figsize=(18, 6))

    # Left: score timeline
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(df_index, score, linewidth=0.8, label="Score (reconstruction error)")

    for i, (name, start, end) in enumerate(anomalies):
        ax1.axvspan(start, end, alpha=0.2, label="Known anomalies" if i == 0 else None)

    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Score")
    ax1.grid(True, linewidth=0.3)
    ax1.legend(loc="upper left")

    ax1.text(
        0.995, 0.995, info_box,
        transform=ax1.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    # Right: ROC
    ax2 = plt.subplot(1, 2, 2)
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    auc = roc_data["roc_auc"]

    ax2.plot(fpr, tpr, linewidth=1.5, label=f"ROC (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC curve (all thresholds)")
    ax2.grid(True, linewidth=0.3)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close("all")
