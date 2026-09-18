"""Tests for evaluation metrics, model selection and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    abs_pearson,
    demographic_parity_difference,
    equal_opportunity_difference,
    evaluate_binary,
)
from src.evaluation.reporting import (
    comparison_table,
    to_markdown_highlighted,
)
from src.evaluation.selection import pareto_mask, select_best_fair
from src.evaluation.uncertainty_analysis import (
    count_missing_ext_sources,
    summarize_uncertainty,
)


def test_demographic_parity_difference():
    prob = np.array([0.9, 0.8, 0.1, 0.2])
    s = np.array([0, 0, 1, 1])
    assert demographic_parity_difference(prob, s) == 1.0
    with pytest.raises(ValueError):
        demographic_parity_difference(prob, np.zeros(4))


def test_equal_opportunity_difference():
    y = np.array([1, 1, 1, 1])
    prob = np.array([0.9, 0.9, 0.9, 0.1])
    s = np.array([0, 0, 1, 1])
    assert equal_opportunity_difference(y, prob, s) == pytest.approx(0.5)


def test_abs_pearson_handles_constant_input():
    assert abs_pearson(np.ones(5), np.arange(5)) == 0.0
    assert abs_pearson(np.arange(5), -np.arange(5)) == pytest.approx(1.0)


def test_evaluate_binary_keys_and_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1)
    s = rng.integers(0, 2, 200)
    metrics = evaluate_binary(y, prob, s)
    assert set(metrics) == {"accuracy", "balanced_accuracy", "roc_auc",
                            "pr_auc", "fair_corr", "fair_spearman", "dpd",
                            "eod"}
    assert metrics["roc_auc"] > 0.9
    with pytest.raises(ValueError):
        evaluate_binary(np.zeros(3), np.zeros(3), np.zeros(3))


def test_pareto_mask():
    fairness = np.array([0.30, 0.10, 0.05, 0.20, 0.01])
    auc = np.array([0.75, 0.74, 0.72, 0.70, 0.60])
    np.testing.assert_array_equal(pareto_mask(fairness, auc),
                                  [True, True, True, False, True])


def test_select_best_fair_uses_threshold_then_fallback():
    candidates = pd.DataFrame({
        "lambda_fair": [0.0, 1.0, 5.0, 10.0],
        "val_fair_corr": [0.25, 0.08, 0.04, 0.01],
        "val_roc_auc": [0.75, 0.74, 0.73, 0.70],
    })
    assert select_best_fair(candidates, 0.05)["lambda_fair"] == 5.0
    assert select_best_fair(candidates, 0.001)["lambda_fair"] == 10.0
    with pytest.raises(ValueError):
        select_best_fair(candidates.iloc[:1], 0.05)


def test_markdown_highlights_best_values():
    table = comparison_table({
        "Base": {"roc_auc": 0.75, "fair_corr": 0.25},
        "Fair": {"roc_auc": 0.73, "fair_corr": 0.02},
    })
    markdown = to_markdown_highlighted(table)
    assert "**0.7500**" in markdown and "**0.0200**" in markdown
    assert "| **Fair** |" in markdown
    with pytest.raises(ValueError):
        comparison_table({})


def test_uncertainty_summaries():
    X = np.array([[0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype="float32")
    names = ("x", "A_MISSING", "B_MISSING")
    counts = count_missing_ext_sources(X, names, ("A_MISSING", "B_MISSING"))
    np.testing.assert_array_equal(counts, [2, 0, 2])
    with pytest.raises(KeyError):
        count_missing_ext_sources(X, names, ("C_MISSING",))
    frame = pd.DataFrame({"group": counts,
                          "expected_abs_error": [0.4, 0.2, 0.6],
                          "mc_variance": [0.01, 0.02, 0.03]})
    summary = summarize_uncertainty(frame, "group")
    assert summary.loc[2, "count"] == 2
    assert summary.loc[2, "expected_abs_error_mean"] == pytest.approx(0.5)
