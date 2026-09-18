"""AutoML with Keras Tuner (task 3): topology + lambda search."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Mapping

import keras_tuner as kt
import pandas as pd

from src.data.preprocessing import PreparedData
from src.training.trainer import evaluate_model, train_classifier
from src.utils.config import PipelineConfig, Topology, TunerConfig

logger = logging.getLogger(__name__)

OBJECTIVE_NAME = "val_score"
TRIAL_METRICS = ("val_score", "val_roc_auc", "val_accuracy",
                 "val_fair_corr", "val_dpd")


def build_search_space(config: TunerConfig) -> kt.HyperParameters:
    """Declare the full search space up front.

    Args:
        config: Tuner configuration.

    Returns:
        The hyper-parameter container shared by every trial.
    """
    hp = kt.HyperParameters()
    hp.Int("n_hidden", 1, config.max_hidden_layers)
    low, high, step = config.dropout_range
    for i in range(1, config.max_hidden_layers + 1):
        hp.Choice(f"units_{i}", list(config.units_choices))
        hp.Float(f"dropout_{i}", low, high, step=step)
    hp.Choice("activation", list(config.activations))
    hp.Choice("learning_rate", list(config.learning_rates))
    hp.Float("lambda_fair", *config.lambda_range, sampling="log")
    return hp


def topology_from_values(values: Mapping) -> Topology:
    """Convert sampled hyper-parameter values into a :class:`Topology`.

    Args:
        values: Mapping produced by ``HyperParameters.values``.

    Returns:
        The corresponding topology (unused deeper layers are ignored).
    """
    n_hidden = int(values["n_hidden"])
    layers = range(1, n_hidden + 1)
    return Topology(
        hidden_units=tuple(int(values[f"units_{i}"]) for i in layers),
        dropout_rates=tuple(round(float(values[f"dropout_{i}"]), 4)
                            for i in layers),
        activation=str(values["activation"]),
        learning_rate=float(values["learning_rate"]),
    )


class FairTopologyTuner(kt.RandomSearch):
    """Random search whose objective trades off AUC and fairness.

    ``run_trial`` is overridden (black-box mode) so each trial reuses the
    project's training loop and returns *exact* full-validation metrics.
    The scalarised objective is::

        val_score = val_roc_auc - fairness_weight * val_fair_corr

    NOTE: improved over original submission — optimising ``val_auc`` alone
    pushes the tuner towards lambda -> 0 (it selected lambda=0.01, i.e. an
    unfair model) and the final lambda had to be overridden by hand.
    """

    def __init__(self, data: PreparedData, config: PipelineConfig,
                 **kwargs) -> None:
        """Create the tuner.

        Args:
            data: Prepared splits (the test split is never used).
            config: Full pipeline configuration.
            **kwargs: Extra ``kt.RandomSearch`` arguments.
        """
        tuner_cfg = config.tuner
        super().__init__(
            hypermodel=None,
            objective=kt.Objective(OBJECTIVE_NAME, direction="max"),
            max_trials=tuner_cfg.max_trials,
            seed=config.seed,
            hyperparameters=build_search_space(tuner_cfg),
            tune_new_entries=False,
            allow_new_entries=False,
            directory=str(tuner_cfg.directory),
            project_name="fair_topology_search",
            overwrite=True,
            **kwargs,
        )
        self._data = data
        self._config = config
        self._trial_training = replace(
            config.training, epochs=tuner_cfg.epochs,
            patience=tuner_cfg.patience)

    def run_trial(self, trial, *args, **kwargs) -> dict[str, float]:
        """Train one sampled configuration and return its metrics.

        Args:
            trial: Keras Tuner trial holding the sampled values.
            *args: Unused.
            **kwargs: Unused.

        Returns:
            Dictionary with the objective and diagnostic metrics.
        """
        del args, kwargs
        values = trial.hyperparameters.values
        topology = topology_from_values(values)
        lambda_fair = float(values["lambda_fair"])
        cfg = self._config
        result = train_classifier(
            self._data, topology, lambda_fair, cfg.fairness,
            self._trial_training, cfg.debt_ratio, cfg.seed,
            name=f"trial_{trial.trial_id}")
        metrics = evaluate_model(result.model, self._data.val,
                                 cfg.fairness.decision_threshold)
        score = (metrics["roc_auc"]
                 - cfg.tuner.fairness_weight * metrics["fair_corr"])
        logger.info("Trial %s | %s | lambda=%.3g | val AUC=%.4f "
                    "|corr|=%.4f score=%.4f", trial.trial_id,
                    topology.hidden_units, lambda_fair, metrics["roc_auc"],
                    metrics["fair_corr"], score)
        return {
            "val_score": score,
            "val_roc_auc": metrics["roc_auc"],
            "val_accuracy": metrics["accuracy"],
            "val_fair_corr": metrics["fair_corr"],
            "val_dpd": metrics["dpd"],
        }


def trials_to_frame(tuner: kt.Tuner) -> pd.DataFrame:
    """Collect every completed trial into a DataFrame.

    Args:
        tuner: A tuner after ``search``.

    Returns:
        One row per trial with its topology, lambda and val metrics.
    """
    rows = []
    for trial in tuner.oracle.trials.values():
        if trial.status != kt.engine.trial.TrialStatus.COMPLETED:
            continue
        values = trial.hyperparameters.values
        topology = topology_from_values(values)
        row = {
            "source": "tuner",
            "trial_id": trial.trial_id,
            "lambda_fair": float(values["lambda_fair"]),
            "hidden_units": topology.hidden_units,
            "dropout_rates": topology.dropout_rates,
            "activation": topology.activation,
            "learning_rate": topology.learning_rate,
        }
        for name in TRIAL_METRICS:
            row[name] = trial.metrics.get_last_value(name)
        rows.append(row)
    return pd.DataFrame(rows)


def run_topology_search(data: PreparedData,
                        config: PipelineConfig) -> pd.DataFrame:
    """Run the Keras Tuner search.

    Args:
        data: Prepared splits.
        config: Pipeline configuration.

    Returns:
        DataFrame of completed trials sorted by objective.

    Raises:
        RuntimeError: If no trial completed successfully.
    """
    tuner = FairTopologyTuner(data, config)
    logger.info("Starting Keras Tuner search (%d trials)",
                config.tuner.max_trials)
    tuner.search(verbose=0)
    trials = trials_to_frame(tuner)
    if trials.empty:
        raise RuntimeError("Keras Tuner produced no completed trial")
    return trials.sort_values(OBJECTIVE_NAME, ascending=False,
                              ignore_index=True)
