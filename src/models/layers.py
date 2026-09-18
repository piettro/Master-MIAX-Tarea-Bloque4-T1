"""Custom Keras layers (task 1: customised architecture).

Only ``keras.ops`` is used so the layers are backend-agnostic
(TensorFlow / JAX / PyTorch), as recommended in class.
"""

from __future__ import annotations

from typing import Sequence

import keras
from keras import ops

PACKAGE = "b4t1"


@keras.saving.register_keras_serializable(package=PACKAGE)
class ClipConstraint(keras.constraints.Constraint):
    """Clip weights into ``[min_value, max_value]`` after each update.

    Same idea as the professor's ``CustomClipValue``.
    """

    def __init__(self, min_value: float, max_value: float) -> None:
        """Create the constraint.

        Args:
            min_value: Lower bound.
            max_value: Upper bound.

        Raises:
            ValueError: If ``min_value >= max_value``.
        """
        if min_value >= max_value:
            raise ValueError("min_value must be smaller than max_value")
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def __call__(self, w):
        """Clip ``w`` into the allowed range."""
        return ops.clip(w, self.min_value, self.max_value)

    def get_config(self) -> dict:
        """Return the serialisable configuration."""
        return {"min_value": self.min_value, "max_value": self.max_value}


@keras.saving.register_keras_serializable(package=PACKAGE)
class DebtRatioLayer(keras.layers.Layer):
    """Compute saturated debt ratios and append them to the features.

    The layer receives the standardised feature vector, undoes the
    standardisation of the log-amounts (using train statistics stored in
    its config) and computes two genuine monetary ratios::

        debt_to_income   = (1 + annuity) / (1 + income)
        credit_to_income = (1 + credit)  / (1 + income)

    Mathematical restriction (saturation): each ratio ``r`` is reshaped by
    a trainable exponent ``a`` (professor's ``ExponentLayer`` idea,
    initialised to 1 and clipped to ``[0.1, 3]``) and bounded with
    ``tanh``::

        g(r) = tanh(r ** a / cap)      ->  g in [0, 1)

    so extremely leveraged applicants cannot dominate the dense layers.
    Output shape: ``(batch, n_features + 2)``.

    NOTE: improved over original submission — the previous layer applied
    ``tanh(z_credit + z_annuity - z_income)`` to z-scores, i.e. a linear
    combination rather than a ratio, with no trainable parameters.
    """

    def __init__(self, income_index: int, credit_index: int,
                 annuity_index: int, log_means: Sequence[float],
                 log_stds: Sequence[float],
                 ratio_caps: Sequence[float] = (0.5, 10.0),
                 exponent_bounds: Sequence[float] = (0.1, 3.0),
                 log_ratio_clip: float = 20.0, **kwargs) -> None:
        """Create the layer.

        Args:
            income_index: Column of the standardised log income.
            credit_index: Column of the standardised log credit.
            annuity_index: Column of the standardised log annuity.
            log_means: Train means of log1p(income, credit, annuity).
            log_stds: Train stds of log1p(income, credit, annuity).
            ratio_caps: Saturation scale of (debt/income, credit/income).
            exponent_bounds: (min, max) of the trainable exponents.
            log_ratio_clip: Clip applied to log-ratios for stability.
            **kwargs: Standard ``keras.layers.Layer`` arguments.

        Raises:
            ValueError: If the statistics or caps are malformed.
        """
        super().__init__(**kwargs)
        if len(log_means) != 3 or len(log_stds) != 3:
            raise ValueError("log_means and log_stds need 3 values each")
        if min(log_stds) <= 0:
            raise ValueError("log_stds must be strictly positive")
        if len(ratio_caps) != 2 or min(ratio_caps) <= 0:
            raise ValueError("ratio_caps needs 2 positive values")
        self.income_index = int(income_index)
        self.credit_index = int(credit_index)
        self.annuity_index = int(annuity_index)
        self.log_means = [float(v) for v in log_means]
        self.log_stds = [float(v) for v in log_stds]
        self.ratio_caps = [float(v) for v in ratio_caps]
        self.exponent_bounds = [float(v) for v in exponent_bounds]
        self.log_ratio_clip = float(log_ratio_clip)
        self.exponents = None

    def build(self, input_shape) -> None:
        """Create the two trainable exponents."""
        n_features = input_shape[-1]
        for index in (self.income_index, self.credit_index,
                      self.annuity_index):
            if n_features is not None and not 0 <= index < n_features:
                raise ValueError(f"Feature index {index} out of range")
        self.exponents = self.add_weight(
            name="exponents", shape=(2,), initializer="ones",
            constraint=ClipConstraint(*self.exponent_bounds),
            trainable=True,
        )

    def _log_amount(self, inputs, index: int, stat: int):
        """Undo the standardisation of one log-amount column."""
        return (inputs[:, index] * self.log_stds[stat]
                + self.log_means[stat])

    def call(self, inputs):
        """Append the two saturated ratios to ``inputs``."""
        log_income = self._log_amount(inputs, self.income_index, 0)
        log_credit = self._log_amount(inputs, self.credit_index, 1)
        log_annuity = self._log_amount(inputs, self.annuity_index, 2)
        log_ratios = ops.stack([log_annuity - log_income,
                                log_credit - log_income], axis=-1)
        log_ratios = ops.clip(log_ratios, -self.log_ratio_clip,
                              self.log_ratio_clip)
        # r ** a computed as exp(a * log r): r > 0 is guaranteed.
        shaped = ops.exp(log_ratios * self.exponents)
        caps = ops.convert_to_tensor(self.ratio_caps, dtype=inputs.dtype)
        saturated = ops.tanh(shaped / caps)
        return ops.concatenate([inputs, saturated], axis=-1)

    def compute_output_shape(self, input_shape):
        """Return ``(batch, n_features + 2)``."""
        return (*input_shape[:-1], input_shape[-1] + 2)

    def get_config(self) -> dict:
        """Return the serialisable configuration."""
        config = super().get_config()
        config.update({
            "income_index": self.income_index,
            "credit_index": self.credit_index,
            "annuity_index": self.annuity_index,
            "log_means": self.log_means,
            "log_stds": self.log_stds,
            "ratio_caps": self.ratio_caps,
            "exponent_bounds": self.exponent_bounds,
            "log_ratio_clip": self.log_ratio_clip,
        })
        return config
