"""Evaluation metrics required by the charter."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {}
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["f1"] = float(f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0))
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(task_type: str, y_true, y_pred) -> dict:
    return classification_metrics(y_true, y_pred) if task_type == "clf" else regression_metrics(y_true, y_pred)
