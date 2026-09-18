"""Figures required by the brief (saved as PNG, headless backend)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.selection import pareto_mask  # noqa: E402
from src.utils.config import CLASS_LABELS  # noqa: E402

logger = logging.getLogger(__name__)

# Validated categorical palette (first three slots pass all-pairs CVD).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NEUTRAL = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FIGURE_DPI = 140
LINE_WIDTH = 2.0
MARKER_SIZE = 64

CLASS_COLORS = {CLASS_LABELS[0]: BLUE, CLASS_LABELS[1]: ORANGE}
UNCERTAINTY_LABELS = {
    "expected_abs_error": "Expected |y - p| (auxiliary error model)",
    "mc_variance": "Predictive variance (MC dropout)",
}


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared, recessive axis styling."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def plot_loss_curves(histories: Mapping[str, Mapping[str, list]],
                     path: Path) -> Path:
    """Train vs validation loss for every final model (convergence).

    Args:
        histories: ``{model title: keras history dict}``.
        path: Output PNG path.

    Returns:
        The saved path.

    Raises:
        ValueError: If ``histories`` is empty.
    """
    if not histories:
        raise ValueError("No history to plot")
    fig, axes = plt.subplots(1, len(histories),
                             figsize=(5.2 * len(histories), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (title, history) in zip(axes, histories.items()):
        epochs = np.arange(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"], color=BLUE, lw=LINE_WIDTH,
                label="Train")
        ax.plot(epochs, history["val_loss"], color=ORANGE, lw=LINE_WIDTH,
                label="Validation")
        best = int(np.argmin(history["val_loss"]))
        ax.scatter(epochs[best], history["val_loss"][best], s=MARKER_SIZE,
                   color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="Restored (best val)")
        _style(ax, title, "Epoch", "Loss")
        ax.legend(frameon=False)
    return _save(fig, path)


def plot_pareto(candidates: pd.DataFrame, best: pd.Series,
                corr_threshold: float, path: Path) -> Path:
    """Performance vs FAIR dependence trade-off (validation set).

    Left panel: ROC-AUC; right panel: accuracy (the brief's "precision").
    Sweep points are joined and annotated with lambda; Keras Tuner trials
    are drawn as crosses; the dashed line is the empirical Pareto front.

    Args:
        candidates: Rows with ``source``, ``lambda_fair``,
            ``val_fair_corr``, ``val_roc_auc`` and ``val_accuracy``.
        best: Selected best FAIR candidate.
        corr_threshold: Fairness threshold used for the selection.
        path: Output PNG path.

    Returns:
        The saved path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panels = (("val_roc_auc", "ROC-AUC"), ("val_accuracy", "Accuracy"))
    for ax, (metric, label) in zip(axes, panels):
        front = candidates[pareto_mask(candidates["val_fair_corr"],
                                       candidates[metric])]
        front = front.sort_values("val_fair_corr")
        ax.step(front["val_fair_corr"], front[metric], where="post",
                color=NEUTRAL, lw=1.2, ls="--", label="Pareto front")
        sweep = candidates[candidates["source"] == "sweep"]
        sweep = sweep.sort_values("lambda_fair")
        ax.plot(sweep["val_fair_corr"], sweep[metric], "-o", color=BLUE,
                lw=LINE_WIDTH, ms=6, label="Lambda sweep")
        for _, row in sweep.iterrows():
            ax.annotate(f"λ={row['lambda_fair']:g}",
                        (row["val_fair_corr"], row[metric]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color=TEXT_SECONDARY)
        tuner = candidates[candidates["source"] == "tuner"]
        ax.scatter(tuner["val_fair_corr"], tuner[metric], marker="x",
                   s=MARKER_SIZE, color=ORANGE, linewidth=2,
                   label="Keras Tuner trials")
        ax.scatter(best["val_fair_corr"], best[metric], s=220,
                   facecolor="none", edgecolor=AQUA, linewidth=2.5,
                   zorder=4, label="Selected best FAIR")
        ax.axvline(corr_threshold, color=NEUTRAL, lw=1, ls=":")
        _style(ax, f"{label} vs dependence (validation)",
               "FAIR dependence |corr(ŷ, gender)|  (← fairer)",
               f"{label} (↑ better)")
    axes[0].legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_uncertainty_by_class(predictions: pd.DataFrame,
                              path: Path) -> Path:
    """Uncertainty distribution for predicted good vs bad payers.

    Args:
        predictions: Output of ``UncertaintyEstimator.predict``.
        path: Output PNG path.

    Returns:
        The saved path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (col, label) in zip(axes, UNCERTAINTY_LABELS.items()):
        values = predictions[col]
        bins = np.linspace(values.min(), values.max() + 1e-9, 40)
        for name, color in CLASS_COLORS.items():
            group = values[predictions["predicted_label"] == name]
            if group.empty:
                continue
            ax.hist(group, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.35, color=color, edgecolor=color, lw=1.5,
                    label=f"{name} (n={len(group):,}, "
                          f"mean={group.mean():.3g})")
        _style(ax, label, label.split(" (")[0], "Density")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Test-set uncertainty by predicted class",
                 color=TEXT_PRIMARY)
    return _save(fig, path)


def plot_uncertainty_by_missing_sources(predictions: pd.DataFrame,
                                        missing_col: str,
                                        path: Path) -> Path:
    """Uncertainty vs number of imputed EXT_SOURCE scores.

    Args:
        predictions: Predictions including ``missing_col``.
        missing_col: Column counting missing EXT_SOURCE values.
        path: Output PNG path.

    Returns:
        The saved path.
    """
    levels = sorted(predictions[missing_col].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (col, label) in zip(axes, UNCERTAINTY_LABELS.items()):
        data = [predictions.loc[predictions[missing_col] == lvl, col]
                for lvl in levels]
        box = ax.boxplot(data, showfliers=False, patch_artist=True,
                         widths=0.55, medianprops={"color": TEXT_PRIMARY})
        for patch in box["boxes"]:
            patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
        means = [series.mean() for series in data]
        ax.plot(range(1, len(levels) + 1), means, "-o", color=ORANGE,
                lw=LINE_WIDTH, label="Mean")
        ax.set_xticks(range(1, len(levels) + 1),
                      [f"{lvl}\n(n={len(d):,})"
                       for lvl, d in zip(levels, data)])
        _style(ax, label, "Number of missing (imputed) EXT_SOURCE scores",
               label.split(" (")[0])
        ax.legend(frameon=False)
    fig.suptitle("Is the model less certain when external scores are "
                 "missing?", color=TEXT_PRIMARY)
    return _save(fig, path)


def plot_error_calibration(predicted_error: np.ndarray,
                           actual_error: np.ndarray, path: Path,
                           n_bins: int = 10) -> Path:
    """Binned predicted vs actual absolute error of the classifier.

    Args:
        predicted_error: Output of the auxiliary error model.
        actual_error: Realised ``|y - p_hat|`` on the test set.
        path: Output PNG path.
        n_bins: Number of quantile bins.

    Returns:
        The saved path.
    """
    frame = pd.DataFrame({"pred": np.asarray(predicted_error).ravel(),
                          "actual": np.asarray(actual_error).ravel()})
    frame["bin"] = pd.qcut(frame["pred"], q=n_bins, duplicates="drop")
    binned = frame.groupby("bin", observed=True).mean()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    upper = float(max(binned.max().max(), 0.05)) * 1.05
    ax.plot([0, upper], [0, upper], color=NEUTRAL, ls="--", lw=1.2,
            label="Perfect calibration")
    ax.plot(binned["pred"], binned["actual"], "-o", color=BLUE,
            lw=LINE_WIDTH, label="Quantile bins")
    corr = np.corrcoef(frame["pred"], frame["actual"])[0, 1]
    _style(ax, f"Error model on test (corr = {corr:.3f})",
           "Predicted |y - p|", "Actual |y - p|")
    ax.legend(frameon=False)
    return _save(fig, path)
