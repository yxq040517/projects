"""Count-based multiclass metrics shared by reference and deployed-pipeline evaluation."""

from __future__ import annotations

import numpy as np


def classification_metrics(confusion: np.ndarray, classes: list[str]) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or np.any(matrix < 0):
        raise ValueError(
            "Confusion matrix must have one nonnegative row/column per declared class."
        )
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = matrix.diagonal().astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "total": total,
        "per_class": [
            {
                "label": label,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, label in enumerate(classes)
        ],
        "confusion_matrix": matrix.tolist(),
    }
