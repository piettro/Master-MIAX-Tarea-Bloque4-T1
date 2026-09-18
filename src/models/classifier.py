"""Credit-default classifier architecture (shared by Base and FAIR)."""

from __future__ import annotations

import keras

from src.data.preprocessing import PreparedData
from src.models.layers import DebtRatioLayer
from src.utils.config import (
    AMOUNT_COLUMNS,
    ANNUITY_COLUMN,
    CREDIT_COLUMN,
    INCOME_COLUMN,
    DebtRatioConfig,
    Topology,
)


def debt_ratio_layer_kwargs(data: PreparedData,
                            config: DebtRatioConfig) -> dict:
    """Build the constructor arguments of :class:`DebtRatioLayer`.

    Args:
        data: Prepared data (feature order and train log-amount stats).
        config: Debt-ratio layer settings.

    Returns:
        Keyword arguments for ``DebtRatioLayer``.
    """
    stats = [data.amount_log_stats[col] for col in AMOUNT_COLUMNS]
    return {
        "income_index": data.feature_index(INCOME_COLUMN),
        "credit_index": data.feature_index(CREDIT_COLUMN),
        "annuity_index": data.feature_index(ANNUITY_COLUMN),
        "log_means": [mean for mean, _ in stats],
        "log_stds": [std for _, std in stats],
        "ratio_caps": config.ratio_caps,
        "exponent_bounds": config.exponent_bounds,
        "log_ratio_clip": config.log_ratio_clip,
    }


def validate_topology(topology: Topology) -> None:
    """Check that a topology is well formed.

    Args:
        topology: Topology to check.

    Raises:
        ValueError: If the topology is inconsistent.
    """
    if not topology.hidden_units:
        raise ValueError("At least one hidden layer is required")
    if len(topology.dropout_rates) != len(topology.hidden_units):
        raise ValueError("dropout_rates and hidden_units lengths differ")
    if any(units <= 0 for units in topology.hidden_units):
        raise ValueError("hidden_units must be positive")
    if any(not 0 <= rate < 1 for rate in topology.dropout_rates):
        raise ValueError("dropout rates must be in [0, 1)")
    if topology.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def build_classifier(input_dim: int, topology: Topology,
                     debt_ratio_kwargs: dict | None = None,
                     name: str = "credit_classifier") -> keras.Model:
    """Build an (uncompiled) MLP with the custom debt-ratio layer.

    Architecture::

        Input -> DebtRatioLayer -> [Dense -> Dropout] x L -> Dense(1, sigmoid)

    Args:
        input_dim: Number of input features.
        topology: Hidden layers, dropout and activation.
        debt_ratio_kwargs: Arguments of :class:`DebtRatioLayer`; ``None``
            skips the custom layer.
        name: Model name.

    Returns:
        The Keras model.

    Raises:
        ValueError: If ``input_dim`` or the topology is invalid.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    validate_topology(topology)

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    if debt_ratio_kwargs is not None:
        x = DebtRatioLayer(name="debt_ratio", **debt_ratio_kwargs)(x)
    layers = zip(topology.hidden_units, topology.dropout_rates)
    for i, (units, rate) in enumerate(layers, start=1):
        x = keras.layers.Dense(units, activation=topology.activation,
                               name=f"dense_{i}")(x)
        x = keras.layers.Dropout(rate, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid",
                                 name="default_probability")(x)
    return keras.Model(inputs, outputs, name=name)
