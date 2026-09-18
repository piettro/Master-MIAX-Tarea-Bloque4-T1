"""Tests for the custom layer, FAIR loss, metrics and model builders."""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops

from src.models.classifier import build_classifier, debt_ratio_layer_kwargs
from src.models.layers import ClipConstraint, DebtRatioLayer
from src.models.losses import (
    FairLoss,
    compute_balanced_class_weights,
    pearson_correlation,
    stack_target_sensitive,
)
from src.models.metrics import FairCorrelation, TargetAUC
from src.models.uncertainty import (
    augment_with_prediction,
    build_error_model,
    mc_dropout_predict,
)
from src.utils.config import DebtRatioConfig, Topology

LOG_MEANS = [11.0, 13.0, 10.0]
LOG_STDS = [0.5, 0.7, 0.5]


def _standardised_row(income, credit, annuity):
    """Encode raw amounts as the layer expects (standardised log1p)."""
    raw = np.log1p([income, credit, annuity])
    return ((raw - LOG_MEANS) / LOG_STDS).astype("float32")[None, :]


def _layer(**kwargs):
    return DebtRatioLayer(0, 1, 2, LOG_MEANS, LOG_STDS, **kwargs)


# --- DebtRatioLayer ----------------------------------------------------------
def test_debt_ratio_layer_computes_saturated_ratios():
    layer = _layer(ratio_caps=(0.5, 10.0))
    out = ops.convert_to_numpy(layer(_standardised_row(1000, 4000, 200)))
    assert out.shape == (1, 5)
    expected = np.tanh(np.array([201 / 1001, 4001 / 1001]) / [0.5, 10.0])
    np.testing.assert_allclose(out[0, 3:], expected, rtol=1e-4)
    assert np.all((out[0, 3:] >= 0) & (out[0, 3:] < 1))


def test_debt_ratio_layer_saturates_extreme_leverage():
    out = ops.convert_to_numpy(_layer()(_standardised_row(10, 1e7, 1e6)))
    assert np.all(out[0, 3:] > 0.999)


def test_debt_ratio_layer_config_round_trip():
    layer = _layer(ratio_caps=(0.4, 8.0))
    clone = DebtRatioLayer.from_config(layer.get_config())
    x = _standardised_row(1000, 5000, 300)
    np.testing.assert_allclose(ops.convert_to_numpy(layer(x)),
                               ops.convert_to_numpy(clone(x)))


def test_debt_ratio_layer_validates_arguments():
    with pytest.raises(ValueError):
        DebtRatioLayer(0, 1, 2, [1.0], [1.0])
    with pytest.raises(ValueError):
        DebtRatioLayer(0, 1, 2, LOG_MEANS, [1.0, 0.0, 1.0])


def test_clip_constraint():
    clipped = ClipConstraint(0.1, 3.0)(ops.convert_to_tensor([-1.0, 5.0]))
    np.testing.assert_allclose(ops.convert_to_numpy(clipped), [0.1, 3.0])
    with pytest.raises(ValueError):
        ClipConstraint(2.0, 1.0)


# --- FAIR loss ---------------------------------------------------------------
def test_pearson_matches_numpy_and_is_shape_invariant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200).astype("float32")
    b = (0.5 * a + rng.normal(size=200)).astype("float32")
    expected = np.corrcoef(a, b)[0, 1]
    flat = float(pearson_correlation(a, b))
    # (N,) vs (N, 1) must not broadcast into an N x N product.
    column = float(pearson_correlation(a[:, None], b))
    assert flat == pytest.approx(expected, abs=1e-5)
    assert column == pytest.approx(expected, abs=1e-5)


def test_fair_loss_without_lambda_is_weighted_bce():
    y = np.array([0, 1, 1, 0], dtype="float32")
    s = np.array([0, 1, 0, 1], dtype="float32")
    p = np.array([[0.2], [0.7], [0.6], [0.1]], dtype="float32")
    loss = FairLoss(lambda_fair=0.0, negative_weight=1.0,
                    positive_weight=3.0)
    value = float(loss(stack_target_sensitive(y, s), p))
    bce = -(y * np.log(p[:, 0]) + (1 - y) * np.log(1 - p[:, 0]))
    w = np.where(y == 1, 3.0, 1.0)
    assert value == pytest.approx(np.sum(w * bce) / w.sum(), rel=1e-5)


@pytest.mark.parametrize("penalty", ["squared", "abs"])
def test_fair_loss_penalises_dependence(penalty):
    y = np.array([0, 1, 0, 1, 0, 1], dtype="float32")
    s = np.array([0, 0, 0, 1, 1, 1], dtype="float32")
    labels = stack_target_sensitive(y, s)
    independent = np.array([[.3], [.6], [.3], [.3], [.6], [.6]], "float32")
    dependent = np.array([[.3], [.3], [.3], [.6], [.6], [.6]], "float32")
    base = FairLoss(lambda_fair=0.0)
    fair = FairLoss(lambda_fair=5.0, penalty=penalty)
    extra_indep = float(fair(labels, independent)
                        - base(labels, independent))
    extra_dep = float(fair(labels, dependent) - base(labels, dependent))
    assert extra_dep > extra_indep >= 0


def test_fair_loss_gradients_are_finite():
    import tensorflow as tf

    labels = stack_target_sensitive([0, 1, 0, 1], [1, 1, 0, 0])
    p = tf.Variable([[0.5], [0.5], [0.5], [0.5]])  # constant -> corr 0/0
    with tf.GradientTape() as tape:
        value = FairLoss(lambda_fair=2.0)(labels, p)
    assert np.all(np.isfinite(tape.gradient(value, p).numpy()))


def test_fair_loss_validation_and_config():
    with pytest.raises(ValueError):
        FairLoss(lambda_fair=-1)
    with pytest.raises(ValueError):
        FairLoss(penalty="cubic")
    loss = FairLoss(lambda_fair=2.0, penalty="abs", positive_weight=4.0)
    clone = FairLoss.from_config(loss.get_config())
    assert (clone.lambda_fair, clone.penalty, clone.positive_weight) == (
        2.0, "abs", 4.0)


def test_stack_and_class_weights():
    assert stack_target_sensitive([1, 0], [0, 1]).shape == (2, 2)
    with pytest.raises(ValueError):
        stack_target_sensitive([1, 0], [0])
    neg, pos = compute_balanced_class_weights(np.array([0, 0, 0, 1]))
    assert (neg, pos) == pytest.approx((4 / 6, 2.0))
    with pytest.raises(ValueError):
        compute_balanced_class_weights(np.zeros(5))


# --- Keras metrics -----------------------------------------------------------
def test_fair_correlation_metric_is_exact_over_batches():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=300).astype("float32")
    s = (rng.uniform(size=300) < p).astype("float32")
    labels = stack_target_sensitive(np.zeros(300), s)
    metric = FairCorrelation()
    for start in range(0, 300, 64):
        metric.update_state(labels[start:start + 64],
                            p[start:start + 64, None])
    assert float(metric.result()) == pytest.approx(
        abs(np.corrcoef(p, s)[0, 1]), abs=1e-4)
    metric.reset_state()
    assert float(metric.n) == 0.0


def test_target_auc_uses_target_column():
    labels = stack_target_sensitive([0, 0, 1, 1], [1, 1, 0, 0])
    metric = TargetAUC()
    metric.update_state(labels, np.array([[.1], [.2], [.8], [.9]]))
    assert float(metric.result()) == pytest.approx(1.0)


# --- Builders ----------------------------------------------------------------
def test_build_classifier_shapes(prepared_data):
    kwargs = debt_ratio_layer_kwargs(prepared_data, DebtRatioConfig())
    model = build_classifier(prepared_data.train.X.shape[1],
                             Topology((8, 4), (0.1, 0.1)), kwargs)
    out = model.predict(prepared_data.val.X, verbose=0)
    assert out.shape == (len(prepared_data.val), 1)
    assert np.all((out > 0) & (out < 1))
    with pytest.raises(ValueError):
        build_classifier(5, Topology((8,), (0.1, 0.2)))


def test_mc_dropout_variance():
    x = np.random.default_rng(0).normal(size=(50, 4)).astype("float32")
    with_dropout = build_classifier(4, Topology((16,), (0.5,)))
    _, var = mc_dropout_predict(with_dropout, x, n_samples=10)
    assert var.shape == (50,) and np.all(var > 0)
    no_dropout = build_classifier(4, Topology((16,), (0.0,)))
    _, var = mc_dropout_predict(no_dropout, x, n_samples=5)
    np.testing.assert_allclose(var, 0.0, atol=1e-10)
    with pytest.raises(ValueError):
        mc_dropout_predict(no_dropout, x, n_samples=1)


def test_error_model_output_is_bounded():
    model = build_error_model(5, (4,), 1e-3)
    x = augment_with_prediction(np.ones((3, 4)), np.array([0.1, 0.5, 0.9]))
    out = model.predict(x, verbose=0)
    assert x.shape == (3, 5) and np.all((out >= 0) & (out <= 1))
