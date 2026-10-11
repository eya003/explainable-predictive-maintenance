"""Calcul des métriques binaires et données de courbes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metrics_at_threshold(
    y_true: Any, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    predictions = (np.asarray(scores) >= threshold).astype(int)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, scores)
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_true, scores)
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "fpr": false_positive_rate,
        "tpr": true_positive_rate,
        "roc_thresholds": roc_thresholds,
        "pr_precision": precision_curve,
        "pr_recall": recall_curve,
        "pr_thresholds": pr_thresholds,
        "threshold": threshold,
    }


def evaluate_pipeline(pipeline: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    scores = pipeline.predict_proba(x_test)[:, 1]
    evaluation = metrics_at_threshold(y_test, scores, 0.5)
    evaluation["scores"] = scores
    return evaluation
