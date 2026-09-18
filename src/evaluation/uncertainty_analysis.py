"""Aggregations answering the brief's uncertainty questions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

UNCERTAINTY_COLUMNS = ("expected_abs_error", "mc_variance")


def count_missing_ext_sources(X: np.ndarray,
                              feature_names: Sequence[str],
                              flag_columns: Sequence[str]) -> np.ndarray:
    """Number of originally-missing (imputed) EXT_SOURCE scores per row.

    Args:
        X: Feature matrix containing the missing-value flags.
        feature_names: Column names of ``X``.
        flag_columns: Names of the 0/1 missing flags.

    Returns:
        Integer array with values in ``[0, len(flag_columns)]``.

    Raises:
        KeyError: If a flag column is absent.
    """
    names = list(feature_names)
    missing = [col for col in flag_columns if col not in names]
    if missing:
        raise KeyError(f"Flag columns not found: {missing}")
    idx = [names.index(col) for col in flag_columns]
    return np.rint(X[:, idx].sum(axis=1)).astype(int)


def summarize_uncertainty(predictions: pd.DataFrame, group_col: str,
                          value_cols: Sequence[str] = UNCERTAINTY_COLUMNS
                          ) -> pd.DataFrame:
    """Count, mean and median of the uncertainty measures per group.

    Args:
        predictions: Output of ``UncertaintyEstimator.predict`` plus any
            grouping column.
        group_col: Column to group by.
        value_cols: Uncertainty columns to summarise.

    Returns:
        DataFrame indexed by group with flattened ``<col>_<stat>`` names.

    Raises:
        KeyError: If a column is missing.
    """
    needed = [group_col, *value_cols]
    absent = [col for col in needed if col not in predictions.columns]
    if absent:
        raise KeyError(f"Columns not found: {absent}")
    summary = (predictions.groupby(group_col)[list(value_cols)]
               .agg(["count", "mean", "median"]))
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    count_cols = [f"{col}_count" for col in value_cols[1:]]
    summary = summary.drop(columns=count_cols)
    return summary.rename(columns={f"{value_cols[0]}_count": "count"})
