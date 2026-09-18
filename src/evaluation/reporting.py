"""Result tables (CSV + Markdown with the best test values in bold)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

HIGHER_IS_BETTER = ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc")
LOWER_IS_BETTER = ("fair_corr", "fair_spearman", "dpd", "eod")
METRIC_TITLES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced acc.",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "fair_corr": "|Pearson(ŷ,s)|",
    "fair_spearman": "|Spearman(ŷ,s)|",
    "dpd": "Demographic parity diff.",
    "eod": "Equal opportunity diff.",
}


def comparison_table(results: Mapping[str, Mapping[str, float]]
                     ) -> pd.DataFrame:
    """Stack per-model metric dictionaries into one table.

    Args:
        results: ``{model name: metric dict}``.

    Returns:
        DataFrame indexed by model name, one column per metric.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("No results to tabulate")
    table = pd.DataFrame.from_dict(results, orient="index")
    table.index.name = "model"
    return table


def to_markdown_highlighted(table: pd.DataFrame, decimals: int = 4) -> str:
    """Render a Markdown table with the best value per column in bold.

    Args:
        table: Output of :func:`comparison_table`.
        decimals: Rounding.

    Returns:
        Markdown string.
    """
    header = ["Model"] + [METRIC_TITLES.get(c, c) for c in table.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    best = {}
    for col in table.columns:
        if col in HIGHER_IS_BETTER:
            best[col] = table[col].max()
        elif col in LOWER_IS_BETTER:
            best[col] = table[col].min()
    for name, row in table.iterrows():
        cells = [f"**{name}**" if name == table.index[-1] else str(name)]
        for col in table.columns:
            text = f"{row[col]:.{decimals}f}"
            cells.append(f"**{text}**" if best.get(col) == row[col]
                         else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(table: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV, creating parent folders.

    Args:
        table: Table to save.
        path: Output CSV path.

    Returns:
        The saved path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_csv(path)
    except OSError as exc:
        raise OSError(f"Could not write '{path}': {exc}") from exc
    logger.info("Saved table %s", path)
    return path


def save_text(text: str, path: Path) -> Path:
    """Write UTF-8 text to ``path``.

    Args:
        text: Content.
        path: Output path.

    Returns:
        The saved path.

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write '{path}': {exc}") from exc
    logger.info("Saved %s", path)
    return path
