"""Keras metrics compatible with the ``(N, 2)`` FAIR labels."""

from __future__ import annotations

import keras
from keras import ops

from src.models.layers import PACKAGE
from src.utils.config import EPSILON


@keras.saving.register_keras_serializable(package=PACKAGE)
class FairCorrelation(keras.metrics.Metric):
    """Exact epoch-level ``|corr(y_hat, s)|`` via streaming sums.

    Unlike averaging per-batch correlations, the running sums give the
    exact Pearson correlation over every sample seen in the epoch.
    """

    _SUMS = ("n", "sum_p", "sum_s", "sum_pp", "sum_ss", "sum_ps")

    def __init__(self, name: str = "fair_corr", eps: float = EPSILON,
                 **kwargs) -> None:
        """Create the metric.

        Args:
            name: Metric name.
            eps: Numerical stabiliser.
            **kwargs: Standard ``keras.metrics.Metric`` arguments.
        """
        super().__init__(name=name, **kwargs)
        self.eps = float(eps)
        for sum_name in self._SUMS:
            setattr(self, sum_name,
                    self.add_weight(name=sum_name, initializer="zeros"))

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        """Accumulate the sufficient statistics of one batch.

        Args:
            y_true: ``(batch, 2)`` labels ``[target, sensitive]``.
            y_pred: ``(batch, 1)`` probabilities.
            sample_weight: Ignored.
        """
        del sample_weight
        p = ops.reshape(ops.cast(y_pred, "float32"), (-1,))
        s = ops.reshape(ops.cast(y_true, "float32")[:, 1], (-1,))
        self.n.assign_add(ops.cast(ops.size(p), "float32"))
        self.sum_p.assign_add(ops.sum(p))
        self.sum_s.assign_add(ops.sum(s))
        self.sum_pp.assign_add(ops.sum(p * p))
        self.sum_ss.assign_add(ops.sum(s * s))
        self.sum_ps.assign_add(ops.sum(p * s))

    def result(self):
        """Return ``|corr(y_hat, s)|`` over the accumulated samples."""
        cov = self.n * self.sum_ps - self.sum_p * self.sum_s
        var_p = self.n * self.sum_pp - self.sum_p ** 2
        var_s = self.n * self.sum_ss - self.sum_s ** 2
        denom = ops.sqrt(ops.maximum(var_p * var_s, 0.0)) + self.eps
        return ops.abs(cov / denom)

    def reset_state(self) -> None:
        """Zero every running sum."""
        for variable in self.variables:
            variable.assign(0.0)

    def get_config(self) -> dict:
        """Return the serialisable configuration."""
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


@keras.saving.register_keras_serializable(package=PACKAGE)
class TargetAUC(keras.metrics.AUC):
    """ROC-AUC computed on the target column of ``(N, 2)`` labels."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the AUC with the target column only."""
        return super().update_state(y_true[:, 0:1], y_pred, sample_weight)
