"""Uncertainty estimation (task 4).

Two complementary estimators are provided:

1. **Auxiliary error model** (professor's approach): a second network
   learns ``|y - p_hat|``, i.e. how wrong the classifier usually is for a
   given profile. For a calibrated classifier E|y - p| = 2 p (1 - p), twice
   the Bernoulli variance, so it is an estimate of *aleatoric* spread.
2. **MC-dropout variance** — NOTE: improved over professor's solution: the
   brief explicitly asks for the *variance* of the prediction; keeping
   dropout active at inference and sampling T forward passes yields the
   predictive variance of the model itself (*epistemic* uncertainty).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd

from src.utils.config import CLASS_LABELS, PREDICT_BATCH_SIZE

logger = logging.getLogger(__name__)


def augment_with_prediction(X: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Append the classifier probability as an extra input column.

    Args:
        X: Feature matrix ``(N, d)``.
        prob: Probabilities ``(N,)`` or ``(N, 1)``.

    Returns:
        float32 matrix ``(N, d + 1)``.

    Raises:
        ValueError: If row counts differ.
    """
    prob = np.asarray(prob, dtype="float32").reshape(-1, 1)
    if len(prob) != len(X):
        raise ValueError("X and prob must have the same number of rows")
    return np.hstack([np.asarray(X, dtype="float32"), prob])


def build_error_model(input_dim: int, hidden_units: tuple[int, ...],
                      learning_rate: float) -> keras.Model:
    """Build and compile the auxiliary absolute-error regressor.

    Args:
        input_dim: Number of input features.
        hidden_units: Hidden-layer widths.
        learning_rate: Adam learning rate.

    Returns:
        Compiled model trained with MSE (MAE reported), as in class.

    Raises:
        ValueError: If arguments are invalid.
    """
    if input_dim <= 0 or not hidden_units:
        raise ValueError("input_dim and hidden_units must be non-empty")
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units, start=1):
        x = keras.layers.Dense(units, activation="relu",
                               name=f"dense_{i}")(x)
    # NOTE: improved over professor's solution — the class example used a
    # ReLU output; |y - p| lies in [0, 1], so a sigmoid bounds it exactly.
    outputs = keras.layers.Dense(1, activation="sigmoid",
                                 name="expected_abs_error")(x)
    model = keras.Model(inputs, outputs, name="error_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="mse", metrics=[keras.metrics.MeanAbsoluteError(
                      name="mae")])
    return model


def mc_dropout_predict(model: keras.Model, X: np.ndarray, n_samples: int,
                       batch_size: int = PREDICT_BATCH_SIZE
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo dropout: mean and variance of T stochastic passes.

    Args:
        model: Classifier containing Dropout layers.
        X: Feature matrix.
        n_samples: Number of stochastic forward passes (T >= 2).
        batch_size: Rows per forward pass.

    Returns:
        ``(mean_probability, variance)``, each of shape ``(N,)``.

    Raises:
        ValueError: If ``n_samples < 2``.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    X = np.asarray(X, dtype="float32")
    total = np.zeros(len(X), dtype="float64")
    total_sq = np.zeros(len(X), dtype="float64")
    for _ in range(n_samples):
        for start in range(0, len(X), batch_size):
            chunk = X[start:start + batch_size]
            pred = keras.ops.convert_to_numpy(model(chunk, training=True))
            pred = pred.reshape(-1).astype("float64")
            total[start:start + len(chunk)] += pred
            total_sq[start:start + len(chunk)] += pred ** 2
    mean = total / n_samples
    variance = np.maximum(total_sq / n_samples - mean ** 2, 0.0)
    return mean.astype("float32"), variance.astype("float32")


@dataclass
class UncertaintyEstimator:
    """Classifier wrapper returning class, probability and uncertainty.

    Attributes:
        classifier: Trained (FAIR) classifier.
        error_model: Trained auxiliary error model.
        threshold: Decision threshold on the probability.
        mc_samples: MC-dropout forward passes.
        use_prediction_feature: Whether the error model takes p_hat.
    """

    classifier: keras.Model
    error_model: keras.Model
    threshold: float = 0.5
    mc_samples: int = 50
    use_prediction_feature: bool = True

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predict the class together with its uncertainty.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with columns ``probability``, ``predicted_class``,
            ``predicted_label``, ``expected_abs_error`` (auxiliary model),
            ``mc_mean`` and ``mc_variance`` (MC dropout).
        """
        prob = self.classifier.predict(
            X, batch_size=PREDICT_BATCH_SIZE, verbose=0).reshape(-1)
        error_input = (augment_with_prediction(X, prob)
                       if self.use_prediction_feature else X)
        expected_error = self.error_model.predict(
            error_input, batch_size=PREDICT_BATCH_SIZE,
            verbose=0).reshape(-1)
        mc_mean, mc_variance = mc_dropout_predict(
            self.classifier, X, self.mc_samples)
        predicted_class = (prob >= self.threshold).astype(int)
        return pd.DataFrame({
            "probability": prob,
            "predicted_class": predicted_class,
            "predicted_label": pd.Series(predicted_class).map(CLASS_LABELS),
            "expected_abs_error": expected_error,
            "mc_mean": mc_mean,
            "mc_variance": mc_variance,
        })
