"""Training loops for the classifier and the auxiliary error model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import keras
import numpy as np

from src.data.preprocessing import DataSplit, PreparedData
from src.evaluation.metrics import evaluate_binary
from src.models.classifier import build_classifier, debt_ratio_layer_kwargs
from src.models.losses import (
    FairLoss,
    compute_balanced_class_weights,
    stack_target_sensitive,
)
from src.models.metrics import FairCorrelation, TargetAUC
from src.models.uncertainty import augment_with_prediction, build_error_model
from src.utils.config import (
    PREDICT_BATCH_SIZE,
    DebtRatioConfig,
    FairnessConfig,
    Topology,
    TrainingConfig,
    UncertaintyConfig,
)
from src.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """A trained model and its per-epoch history.

    Attributes:
        model: Trained Keras model (best weights restored).
        history: ``History.history`` dictionary.
    """

    model: keras.Model
    history: dict[str, list[float]]

    @property
    def epochs_trained(self) -> int:
        """Number of epochs actually run."""
        return len(self.history.get("loss", []))


def _early_stopping(patience: int) -> keras.callbacks.EarlyStopping:
    """Early stopping on ``val_loss`` restoring the best weights.

    NOTE: improved over original submission — it monitored ``val_auc``,
    which ignores the fairness term being optimised and could restore an
    early, still-unfair epoch. ``val_loss`` tracks the actual objective.
    """
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True)


def predict_proba(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Predict probabilities as a flat array.

    Args:
        model: Trained classifier.
        X: Feature matrix.

    Returns:
        Probabilities of shape ``(N,)``.
    """
    return model.predict(X, batch_size=PREDICT_BATCH_SIZE,
                         verbose=0).reshape(-1)


def evaluate_model(model: keras.Model, split: DataSplit,
                   threshold: float) -> dict[str, float]:
    """Evaluate a classifier on one split.

    Args:
        model: Trained classifier.
        split: Data split.
        threshold: Decision threshold.

    Returns:
        Metric dictionary (see :func:`evaluate_binary`).
    """
    return evaluate_binary(split.y, predict_proba(model, split.X), split.s,
                           threshold)


def train_classifier(data: PreparedData, topology: Topology,
                     lambda_fair: float, fairness: FairnessConfig,
                     training: TrainingConfig, debt_ratio: DebtRatioConfig,
                     seed: int, name: str = "credit_classifier"
                     ) -> TrainingResult:
    """Train the classifier with the FAIR loss (lambda = 0 -> Base).

    Args:
        data: Prepared splits.
        topology: Architecture and learning rate.
        lambda_fair: Weight of the fairness penalty.
        fairness: FAIR loss settings.
        training: Epochs, batch size, patience.
        debt_ratio: Custom layer settings.
        seed: Random seed (re-applied so every run is reproducible).
        name: Model name.

    Returns:
        The trained model and its history.
    """
    set_global_seed(seed)
    negative_weight, positive_weight = compute_balanced_class_weights(
        data.train.y)
    model = build_classifier(
        data.train.X.shape[1], topology,
        debt_ratio_layer_kwargs(data, debt_ratio), name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(topology.learning_rate),
        loss=FairLoss(lambda_fair=lambda_fair, penalty=fairness.penalty,
                      negative_weight=negative_weight,
                      positive_weight=positive_weight),
        metrics=[TargetAUC(name="auc"), FairCorrelation(name="fair_corr")],
    )
    history = model.fit(
        data.train.X, stack_target_sensitive(data.train.y, data.train.s),
        validation_data=(data.val.X,
                         stack_target_sensitive(data.val.y, data.val.s)),
        epochs=training.epochs, batch_size=training.batch_size,
        callbacks=[_early_stopping(training.patience)], verbose=0,
    )
    result = TrainingResult(model=model, history=history.history)
    logger.info("Trained %s (lambda=%g) for %d epochs | best val_loss=%.4f",
                name, lambda_fair, result.epochs_trained,
                min(result.history["val_loss"]))
    return result


def train_error_model(classifier: keras.Model, data: PreparedData,
                      uncertainty: UncertaintyConfig,
                      training: TrainingConfig, seed: int
                      ) -> TrainingResult:
    """Train the auxiliary model that predicts ``|y - p_hat|``.

    Targets are the classifier's absolute errors on train (fit) and
    validation (early stopping), exactly as in the class example; the test
    split is never touched.

    Args:
        classifier: Trained classifier whose errors are modelled.
        data: Prepared splits.
        uncertainty: Error-model settings.
        training: Epochs, batch size, patience.
        seed: Random seed.

    Returns:
        The trained error model and its history.
    """
    set_global_seed(seed)
    inputs, targets = [], []
    for split in (data.train, data.val):
        prob = predict_proba(classifier, split.X)
        inputs.append(augment_with_prediction(split.X, prob)
                      if uncertainty.use_prediction_feature else split.X)
        targets.append(np.abs(split.y - prob).reshape(-1, 1))
    model = build_error_model(inputs[0].shape[1], uncertainty.hidden_units,
                              uncertainty.learning_rate)
    history = model.fit(
        inputs[0], targets[0], validation_data=(inputs[1], targets[1]),
        epochs=training.epochs, batch_size=training.batch_size,
        callbacks=[_early_stopping(training.patience)], verbose=0,
    )
    result = TrainingResult(model=model, history=history.history)
    logger.info("Trained error model for %d epochs | best val_mae=%.4f",
                result.epochs_trained, min(result.history["val_mae"]))
    return result
