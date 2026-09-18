"""FAIR loss (task 2: fair learning).

    L = weighted_BCE(y, y_hat) + lambda * penalty(corr(y_hat, s))

The dependence measure is the Pearson correlation between the predicted
probability and the sensitive attribute, computed on each mini-batch with
differentiable tensor ops only.
"""

from __future__ import annotations

import numpy as np
import keras
from keras import ops

from src.models.layers import PACKAGE
from src.utils.config import EPSILON, PROBABILITY_CLIP

PENALTIES = ("squared", "abs")


def stack_target_sensitive(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Pack target and sensitive attribute into one ``(N, 2)`` label.

    Keras passes only ``y_true`` and ``y_pred`` to a loss, so the
    sensitive attribute travels as the second label column (class trick).

    Args:
        y: Binary target, shape ``(N,)`` or ``(N, 1)``.
        s: Sensitive attribute, same length as ``y``.

    Returns:
        float32 array of shape ``(N, 2)``: ``[target, sensitive]``.

    Raises:
        ValueError: If lengths differ.
    """
    y = np.asarray(y, dtype="float32").reshape(-1)
    s = np.asarray(s, dtype="float32").reshape(-1)
    if y.shape != s.shape:
        raise ValueError(f"y and s lengths differ: {y.shape} vs {s.shape}")
    return np.column_stack([y, s])


def pearson_correlation(x, y, eps: float = EPSILON):
    """Differentiable (signed) Pearson correlation of two tensors.

    Both inputs are flattened first. This avoids the silent
    ``(N,)`` vs ``(N, 1)`` broadcasting bug discussed in the correction
    lecture (which turns an element-wise product into an ``N x N`` one).

    Args:
        x: Tensor with N elements.
        y: Tensor with N elements.
        eps: Stabiliser added to the denominator.

    Returns:
        Scalar tensor in ``[-1, 1]``.
    """
    x = ops.reshape(ops.cast(x, "float32"), (-1,))
    y = ops.reshape(ops.cast(y, "float32"), (-1,))
    xc = x - ops.mean(x)
    yc = y - ops.mean(y)
    numerator = ops.sum(xc * yc)
    denominator = ops.sqrt(ops.sum(xc * xc) * ops.sum(yc * yc)) + eps
    return numerator / denominator


def compute_balanced_class_weights(y: np.ndarray) -> tuple[float, float]:
    """Balanced class weights ``n / (2 * n_class)``.

    Args:
        y: Binary target.

    Returns:
        ``(negative_weight, positive_weight)``.

    Raises:
        ValueError: If ``y`` does not contain both classes.
    """
    y = np.asarray(y).reshape(-1)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required to compute weights")
    n = n_pos + n_neg
    return n / (2.0 * n_neg), n / (2.0 * n_pos)


@keras.saving.register_keras_serializable(package=PACKAGE)
class FairLoss(keras.losses.Loss):
    """Class-weighted BCE plus a correlation-based fairness penalty.

    ``y_true`` must have shape ``(batch, 2)`` = ``[target, sensitive]``
    (see :func:`stack_target_sensitive`). The argument order is always
    ``(y_true, y_pred)``: the loss is *not* symmetric, a pitfall stressed in
    the correction lecture.

    With ``lambda_fair = 0`` this is exactly the Base model's loss, so Base
    and FAIR models are compared apples-to-apples.
    """

    def __init__(self, lambda_fair: float = 0.0, penalty: str = "squared",
                 negative_weight: float = 1.0, positive_weight: float = 1.0,
                 eps: float = EPSILON, name: str = "fair_loss",
                 **kwargs) -> None:
        """Create the loss.

        Args:
            lambda_fair: Weight of the fairness penalty (>= 0).
            penalty: ``"squared"`` -> corr^2 (professor's choice, smooth
                at 0) or ``"abs"`` -> |corr|.
            negative_weight: Weight of class 0 samples.
            positive_weight: Weight of class 1 samples.
            eps: Numerical stabiliser.
            name: Loss name.
            **kwargs: Standard ``keras.losses.Loss`` arguments.

        Raises:
            ValueError: If an argument is out of range.
        """
        super().__init__(name=name, **kwargs)
        if lambda_fair < 0:
            raise ValueError("lambda_fair must be non-negative")
        if penalty not in PENALTIES:
            raise ValueError(f"penalty must be one of {PENALTIES}")
        if negative_weight <= 0 or positive_weight <= 0:
            raise ValueError("class weights must be positive")
        self.lambda_fair = float(lambda_fair)
        self.penalty = penalty
        self.negative_weight = float(negative_weight)
        self.positive_weight = float(positive_weight)
        self.eps = float(eps)

    def call(self, y_true, y_pred):
        """Compute the batch loss.

        Args:
            y_true: ``(batch, 2)`` tensor ``[target, sensitive]``.
            y_pred: ``(batch, 1)`` predicted probabilities.

        Returns:
            Scalar loss tensor.
        """
        y_true = ops.cast(y_true, "float32")
        target = y_true[:, 0]
        sensitive = y_true[:, 1]
        prob = ops.reshape(ops.cast(y_pred, "float32"), (-1,))
        prob = ops.clip(prob, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)

        bce = -(target * ops.log(prob)
                + (1.0 - target) * ops.log(1.0 - prob))
        weights = (target * self.positive_weight
                   + (1.0 - target) * self.negative_weight)
        base = ops.sum(weights * bce) / (ops.sum(weights) + self.eps)

        corr = pearson_correlation(prob, sensitive, self.eps)
        if self.penalty == "squared":
            fairness = ops.square(corr)
        else:
            fairness = ops.abs(corr)
        return base + self.lambda_fair * fairness

    def get_config(self) -> dict:
        """Return the serialisable configuration."""
        config = super().get_config()
        config.update({
            "lambda_fair": self.lambda_fair,
            "penalty": self.penalty,
            "negative_weight": self.negative_weight,
            "positive_weight": self.positive_weight,
            "eps": self.eps,
        })
        return config
