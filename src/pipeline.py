"""End-to-end experiment: data -> Base/FAIR -> AutoML -> uncertainty.

Model selection uses the validation split only; the test split is
evaluated exactly once, for the final Base and best-FAIR models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import PreparedData, prepare_data
from src.evaluation.reporting import (
    comparison_table,
    save_table,
    save_text,
    to_markdown_highlighted,
)
from src.evaluation.selection import select_best_fair
from src.evaluation.uncertainty_analysis import (
    count_missing_ext_sources,
    summarize_uncertainty,
)
from src.models.uncertainty import UncertaintyEstimator
from src.training.trainer import (
    evaluate_model,
    train_classifier,
    train_error_model,
)
from src.training.tuning import run_topology_search
from src.utils.config import EXT_MISSING_COLUMNS, PipelineConfig, Topology
from src.utils.reproducibility import set_global_seed
from src.visualization.plots import (
    plot_error_calibration,
    plot_loss_curves,
    plot_pareto,
    plot_uncertainty_by_class,
    plot_uncertainty_by_missing_sources,
)

logger = logging.getLogger(__name__)

MISSING_COUNT_COLUMN = "n_missing_ext_sources"
BASE_NAME = "Base (lambda=0)"


@dataclass
class PipelineResults:
    """Key outputs of a pipeline run.

    Attributes:
        test_table: Base vs best-FAIR test metrics.
        candidates: Every sweep/tuner candidate with validation metrics.
        best_candidate: Selected best FAIR configuration.
        uncertainty_by_class: Uncertainty summary per predicted class.
        uncertainty_by_missing: Uncertainty summary per missing count.
        artifacts: Paths of every file written.
    """

    test_table: pd.DataFrame
    candidates: pd.DataFrame
    best_candidate: pd.Series
    uncertainty_by_class: pd.DataFrame
    uncertainty_by_missing: pd.DataFrame
    artifacts: list[Path] = field(default_factory=list)


def _val_metrics_row(metrics: dict[str, float]) -> dict[str, float]:
    """Prefix validation metrics with ``val_``."""
    keys = ("roc_auc", "accuracy", "fair_corr", "dpd")
    return {f"val_{key}": metrics[key] for key in keys}


def run_lambda_sweep(data: PreparedData,
                     config: PipelineConfig) -> pd.DataFrame:
    """Train the default topology for every lambda of the sweep.

    Args:
        data: Prepared splits.
        config: Pipeline configuration.

    Returns:
        One row per lambda with validation metrics.
    """
    rows = []
    topology = config.topology
    for lambda_fair in config.fairness.lambdas:
        result = train_classifier(
            data, topology, lambda_fair, config.fairness, config.training,
            config.debt_ratio, config.seed,
            name=f"sweep_lambda_{lambda_fair:g}")
        metrics = evaluate_model(result.model, data.val,
                                 config.fairness.decision_threshold)
        rows.append({
            "source": "sweep",
            "trial_id": f"lambda_{lambda_fair:g}",
            "lambda_fair": float(lambda_fair),
            "hidden_units": topology.hidden_units,
            "dropout_rates": topology.dropout_rates,
            "activation": topology.activation,
            "learning_rate": topology.learning_rate,
            **_val_metrics_row(metrics),
        })
        logger.info("Sweep lambda=%g | val AUC=%.4f |corr|=%.4f",
                    lambda_fair, metrics["roc_auc"], metrics["fair_corr"])
    return pd.DataFrame(rows)


def topology_from_row(row: pd.Series) -> Topology:
    """Rebuild a :class:`Topology` from a candidate row.

    Args:
        row: Candidate with topology columns.

    Returns:
        The topology.
    """
    return Topology(
        hidden_units=tuple(int(u) for u in row["hidden_units"]),
        dropout_rates=tuple(float(d) for d in row["dropout_rates"]),
        activation=str(row["activation"]),
        learning_rate=float(row["learning_rate"]),
    )


def _to_jsonable(value):
    """Convert NumPy / Path / tuple values for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Execute the full experiment and write every deliverable.

    Args:
        config: Pipeline configuration.

    Returns:
        Summary of the results and written artifacts.
    """
    set_global_seed(config.seed, config.deterministic_ops)
    out = Path(config.output_dir)
    figures, tables, models = out / "figures", out / "tables", out / "models"
    artifacts: list[Path] = []
    threshold = config.fairness.decision_threshold

    # 1. Data --------------------------------------------------------------
    data = prepare_data(config.data, config.seed)

    # 2. FAIR loss: manual lambda sweep on the default topology -----------
    sweep = run_lambda_sweep(data, config)

    # 3. AutoML: Keras Tuner over topology + lambda -----------------------
    tuner_trials = run_topology_search(data, config)
    candidates = pd.concat([sweep, tuner_trials], ignore_index=True)
    artifacts.append(save_table(candidates, tables / "candidates_val.csv"))

    # 4. Model selection on validation only --------------------------------
    best = select_best_fair(candidates, config.fairness.corr_threshold)
    best_topology = topology_from_row(best)
    best_lambda = float(best["lambda_fair"])
    logger.info("Best FAIR: %s %s lambda=%g | val AUC=%.4f |corr|=%.4f",
                best["source"], best_topology, best_lambda,
                best["val_roc_auc"], best["val_fair_corr"])
    artifacts.append(plot_pareto(candidates, best,
                                 config.fairness.corr_threshold,
                                 figures / "pareto_fairness.png"))

    # 5. Final Base vs FAIR with the SAME topology (only lambda differs) ---
    fair_name = f"Best FAIR (lambda={best_lambda:.3g})"
    base = train_classifier(data, best_topology, 0.0, config.fairness,
                            config.training, config.debt_ratio,
                            config.seed, name="base_model")
    fair = train_classifier(data, best_topology, best_lambda,
                            config.fairness, config.training,
                            config.debt_ratio, config.seed,
                            name="fair_model")
    test_table = comparison_table({
        BASE_NAME: evaluate_model(base.model, data.test, threshold),
        fair_name: evaluate_model(fair.model, data.test, threshold),
    })
    artifacts.append(save_table(test_table,
                                tables / "base_vs_fair_test.csv"))
    artifacts.append(save_text(to_markdown_highlighted(test_table),
                               tables / "base_vs_fair_test.md"))

    # 6. Uncertainty ---------------------------------------------------------
    error = train_error_model(fair.model, data, config.uncertainty,
                              config.training, config.seed)
    estimator = UncertaintyEstimator(
        classifier=fair.model, error_model=error.model,
        threshold=threshold, mc_samples=config.uncertainty.mc_samples,
        use_prediction_feature=config.uncertainty.use_prediction_feature)
    predictions = estimator.predict(data.test.X)
    predictions["y_true"] = data.test.y.astype(int)
    predictions["actual_abs_error"] = np.abs(
        predictions["y_true"] - predictions["probability"])
    predictions[MISSING_COUNT_COLUMN] = count_missing_ext_sources(
        data.test.X, data.feature_names, EXT_MISSING_COLUMNS)
    by_class = summarize_uncertainty(predictions, "predicted_label")
    by_missing = summarize_uncertainty(predictions, MISSING_COUNT_COLUMN)
    artifacts += [
        save_table(predictions, tables / "test_predictions_uncertainty.csv"),
        save_table(by_class, tables / "uncertainty_by_class.csv"),
        save_table(by_missing, tables / "uncertainty_by_missing_ext.csv"),
        plot_uncertainty_by_class(predictions,
                                  figures / "uncertainty_by_class.png"),
        plot_uncertainty_by_missing_sources(
            predictions, MISSING_COUNT_COLUMN,
            figures / "uncertainty_by_missing_ext_sources.png"),
        plot_error_calibration(predictions["expected_abs_error"],
                               predictions["actual_abs_error"],
                               figures / "error_model_calibration.png"),
        plot_loss_curves({BASE_NAME: base.history,
                          fair_name: fair.history,
                          "Auxiliary error model": error.history},
                         figures / "loss_curves.png"),
    ]

    # 7. Models + run summary ---------------------------------------------
    for name, result in (("base_model", base), ("fair_model", fair),
                         ("error_model", error)):
        path = models / f"{name}.keras"
        path.parent.mkdir(parents=True, exist_ok=True)
        result.model.save(path)
        artifacts.append(path)
    summary = {
        "config": asdict(config),
        "best_fair": {"source": best["source"], "lambda": best_lambda,
                      "topology": asdict(best_topology),
                      "val_roc_auc": best["val_roc_auc"],
                      "val_fair_corr": best["val_fair_corr"]},
        "epochs": {"base": base.epochs_trained,
                   "fair": fair.epochs_trained,
                   "error_model": error.epochs_trained},
        "test_metrics": test_table.to_dict(orient="index"),
        "uncertainty_by_class": by_class.to_dict(orient="index"),
        "uncertainty_by_missing": by_missing.to_dict(orient="index"),
    }
    artifacts.append(save_text(
        json.dumps(_to_jsonable(summary), indent=2),
        out / "run_summary.json"))

    logger.info("Test results:\n%s", test_table.round(4).to_string())
    logger.info("Uncertainty by predicted class:\n%s",
                by_class.round(4).to_string())
    logger.info("Uncertainty by missing EXT_SOURCE count:\n%s",
                by_missing.round(4).to_string())
    return PipelineResults(test_table, candidates, best, by_class,
                           by_missing, artifacts)
