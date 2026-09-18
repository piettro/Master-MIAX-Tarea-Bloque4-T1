"""Pareto front and best-FAIR-model selection (validation data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_mask(fairness: np.ndarray, performance: np.ndarray
                ) -> np.ndarray:
    """Flag the non-dominated points (low dependence, high performance).

    A point is dominated if another point has lower-or-equal dependence
    and higher-or-equal performance, with at least one strict inequality.

    Args:
        fairness: Dependence measure per candidate (lower is better).
        performance: Performance per candidate (higher is better).

    Returns:
        Boolean array, True for Pareto-optimal candidates.

    Raises:
        ValueError: If lengths differ.
    """
    x = np.asarray(fairness, dtype="float64")
    y = np.asarray(performance, dtype="float64")
    if x.shape != y.shape:
        raise ValueError("fairness and performance lengths differ")
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        dominated = ((x <= x[i]) & (y >= y[i])
                     & ((x < x[i]) | (y > y[i])))
        mask[i] = not dominated.any()
    return mask


def select_best_fair(candidates: pd.DataFrame, corr_threshold: float,
                     corr_col: str = "val_fair_corr",
                     score_col: str = "val_roc_auc") -> pd.Series:
    """Pick the best FAIR candidate using validation metrics only.

    Rule: highest validation ROC-AUC among fair candidates
    (``lambda > 0``) whose validation ``|corr|`` is below the threshold;
    if none qualifies, the candidate with the lowest ``|corr|``.

    NOTE: added — the original submission chose the "best FAIR" model by
    looking at *test* metrics, leaking the test set into model selection.

    Args:
        candidates: One row per trained candidate; must contain
            ``lambda_fair``, ``corr_col`` and ``score_col``.
        corr_threshold: Max accepted dependence.
        corr_col: Column with the validation dependence.
        score_col: Column with the validation performance.

    Returns:
        The selected row.

    Raises:
        ValueError: If there is no candidate with ``lambda_fair > 0``.
    """
    fair = candidates[candidates["lambda_fair"] > 0]
    if fair.empty:
        raise ValueError("No FAIR candidate (lambda_fair > 0) available")
    eligible = fair[fair[corr_col] <= corr_threshold]
    if eligible.empty:
        return fair.loc[fair[corr_col].idxmin()]
    return eligible.loc[eligible[score_col].idxmax()]
