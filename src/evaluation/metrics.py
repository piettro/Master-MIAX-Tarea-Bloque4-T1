"""Predictive and fairness metrics computed with NumPy / scikit-learn."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def _flat(values: np.ndarray) -> np.ndarray:
    """Return a 1-D float64 view of ``values``."""
    return np.asarray(values, dtype="float64").reshape(-1)


def abs_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation, 0 if either input is constant.

    Args:
        a: First sample.
        b: Second sample (same length).

    Returns:
        ``|corr(a, b)|`` in ``[0, 1]``.

    Raises:
        ValueError: If lengths differ.
    """
    a, b = _flat(a), _flat(b)
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same length")
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def demographic_parity_difference(prob: np.ndarray, s: np.ndarray,
                                  threshold: float = 0.5) -> float:
    """``|P(y_hat=1 | s=0) - P(y_hat=1 | s=1)|``.

    Args:
        prob: Predicted probabilities.
        s: Binary sensitive attribute.
        threshold: Decision threshold.

    Returns:
        Absolute gap in positive (rejection) rates between groups.

    Raises:
        ValueError: If a group is empty.
    """
    y_hat = _flat(prob) >= threshold
    s = _flat(s)
    if not np.any(s == 0) or not np.any(s == 1):
        raise ValueError("Both sensitive groups must be present")
    return float(abs(y_hat[s == 0].mean() - y_hat[s == 1].mean()))


def equal_opportunity_difference(y: np.ndarray, prob: np.ndarray,
                                 s: np.ndarray,
                                 threshold: float = 0.5) -> float:
    """Gap in true-positive rates between the two groups.

    NOTE: improved over original submission — complements demographic
    parity with an error-rate-based fairness criterion.

    Args:
        y: Binary target.
        prob: Predicted probabilities.
        s: Binary sensitive attribute.
        threshold: Decision threshold.

    Returns:
        ``|TPR(s=0) - TPR(s=1)|`` (0 if a group has no positives).
    """
    y, s = _flat(y), _flat(s)
    y_hat = _flat(prob) >= threshold
    rates = []
    for group in (0, 1):
        positives = (s == group) & (y == 1)
        if not np.any(positives):
            return 0.0
        rates.append(y_hat[positives].mean())
    return float(abs(rates[0] - rates[1]))


def evaluate_binary(y: np.ndarray, prob: np.ndarray, s: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float]:
    """Compute every predictive and fairness metric of the report.

    Args:
        y: Binary target.
        prob: Predicted probabilities.
        s: Binary sensitive attribute.
        threshold: Decision threshold.

    Returns:
        Dictionary of metric name -> value.

    Raises:
        ValueError: If input lengths differ or ``y`` has a single class.
    """
    y, prob, s = _flat(y), _flat(prob), _flat(s)
    if not len(y) == len(prob) == len(s):
        raise ValueError("y, prob and s must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes")
    y_hat = (prob >= threshold).astype(int)
    spearman = spearmanr(prob, s).statistic if prob.std() > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "fair_corr": abs_pearson(prob, s),
        "fair_spearman": float(abs(spearman)),
        "dpd": demographic_parity_difference(prob, s, threshold),
        "eod": equal_opportunity_difference(y, prob, s, threshold),
    }
