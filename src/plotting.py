import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def plot_score_and_roc(
    df_index: pd.Index,
    score: np.ndarray,
    title: str,
    out_path: str,
    info_box: str,
    anomalies: List[Tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
    known_normal_spans: List[Tuple[str, pd.Timestamp, pd.Timestamp]] | None = None,
    roc_data: Dict[str, np.ndarray] | None = None,
    score_label: str = "Score (reconstruction error)",
    roc_title: str = "ROC curve",
) -> None:
    anomalies = anomalies or []
    known_normal_spans = known_normal_spans or []
    has_roc = roc_data is not None and all(k in roc_data for k in ["fpr", "tpr", "roc_auc"])

    if has_roc:
        plt.figure(figsize=(18, 6))
        ax1 = plt.subplot(1, 2, 1)
    else:
        plt.figure(figsize=(14, 5))
        ax1 = plt.subplot(1, 1, 1)

    ax1.plot(df_index, score, linewidth=0.8, label=score_label)

    for i, (_, start, end) in enumerate(anomalies):
        ax1.axvspan(start, end, alpha=0.20, label="Known anomalies" if i == 0 else None)

    for i, (_, start, end) in enumerate(known_normal_spans):
        ax1.axvspan(start, end, alpha=0.15, label="Known healthy region" if i == 0 else None)

    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Score")
    ax1.grid(True, linewidth=0.3)
    ax1.legend(loc="upper left")

    ax1.text(
        0.995, 0.995, info_box,
        transform=ax1.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    if has_roc:
        ax2 = plt.subplot(1, 2, 2)
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data["roc_auc"]

        ax2.plot(fpr, tpr, linewidth=1.5, label=f"ROC (AUC={auc:.3f})")
        ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Random")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title(roc_title)
        ax2.grid(True, linewidth=0.3)
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close("all")