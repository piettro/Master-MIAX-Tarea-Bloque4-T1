"""Integration test: the whole pipeline on a small synthetic dataset."""

from __future__ import annotations

import json
from dataclasses import replace

import keras
import numpy as np

from src.models.losses import stack_target_sensitive
from src.pipeline import run_pipeline
from src.training.tuning import build_search_space, topology_from_values
from src.utils.config import PipelineConfig, TunerConfig

EXPECTED_FILES = (
    "figures/pareto_fairness.png",
    "figures/loss_curves.png",
    "figures/uncertainty_by_class.png",
    "figures/uncertainty_by_missing_ext_sources.png",
    "figures/error_model_calibration.png",
    "tables/base_vs_fair_test.csv",
    "tables/base_vs_fair_test.md",
    "tables/candidates_val.csv",
    "tables/uncertainty_by_class.csv",
    "tables/uncertainty_by_missing_ext.csv",
    "tables/test_predictions_uncertainty.csv",
    "models/fair_model.keras",
    "run_summary.json",
)


def test_search_space_to_topology():
    hp = build_search_space(TunerConfig())
    values = dict(hp.values, n_hidden=2)
    topology = topology_from_values(values)
    assert len(topology.hidden_units) == 2
    assert len(topology.dropout_rates) == 2


def test_end_to_end_pipeline(tmp_path, raw_frame):
    csv_path = tmp_path / "application_train.csv"
    raw_frame.to_csv(csv_path, index=False)
    base = PipelineConfig().quick()
    config = replace(
        base,
        output_dir=tmp_path / "outputs",
        deterministic_ops=False,
        data=replace(base.data, csv_path=csv_path, sample_size=None),
        tuner=replace(base.tuner, directory=tmp_path / "tuner"),
    )
    results = run_pipeline(config)

    for relative in EXPECTED_FILES:
        assert (config.output_dir / relative).is_file(), relative
    assert list(results.test_table.index) == [
        "Base (lambda=0)", results.test_table.index[1]]
    assert results.best_candidate["lambda_fair"] > 0
    summary = json.loads(
        (config.output_dir / "run_summary.json").read_text("utf-8"))
    assert "test_metrics" in summary

    # The saved FAIR model reloads with its custom objects.
    model = keras.models.load_model(
        config.output_dir / "models" / "fair_model.keras")
    x = np.zeros((2, model.input_shape[-1]), dtype="float32")
    assert model.predict(x, verbose=0).shape == (2, 1)
    loss = model.evaluate(x, stack_target_sensitive([0, 1], [0, 1]),
                          verbose=0)
    assert np.all(np.isfinite(loss))
