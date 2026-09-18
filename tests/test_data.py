"""Tests for data loading and preprocessing."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_raw_data
from src.data.preprocessing import (
    FeaturePreprocessor,
    clean_raw_data,
    split_data,
)
from src.utils.config import (
    AGE_COLUMN,
    EXT_MISSING_COLUMNS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    DataConfig,
)


def test_load_raw_data_reads_csv_and_zip(tmp_path, raw_frame):
    csv_path = tmp_path / "application_train.csv"
    raw_frame.to_csv(csv_path, index=False)
    from_csv = load_raw_data(csv_path, tmp_path / "missing.zip")
    zip_path = tmp_path / "application_train.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="application_train.csv")
    from_zip = load_raw_data(tmp_path / "missing.csv", zip_path)
    assert list(from_csv.columns) == list(RAW_COLUMNS)
    pd.testing.assert_frame_equal(from_csv, from_zip)


def test_load_raw_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_raw_data(tmp_path / "a.csv", tmp_path / "a.zip")


def test_load_raw_data_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"TARGET": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Could not read columns"):
        load_raw_data(path, tmp_path / "none.zip")


def test_clean_raw_data(raw_frame):
    cleaned = clean_raw_data(raw_frame)
    assert len(cleaned) == len(raw_frame) - 3  # three "XNA" rows dropped
    assert set(cleaned["CODE_GENDER"].unique()) <= {0, 1}
    assert cleaned[AGE_COLUMN].between(19, 66).all()
    expected_flag = raw_frame.loc[
        raw_frame["CODE_GENDER"] != "XNA", "EXT_SOURCE_1"].isna()
    np.testing.assert_array_equal(cleaned["EXT_SOURCE_1_MISSING"],
                                  expected_flag.astype(int))


def test_clean_raw_data_requires_columns(raw_frame):
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_raw_data(raw_frame.drop(columns=["AMT_CREDIT"]))


def test_split_is_stratified_and_disjoint(raw_frame):
    cleaned = clean_raw_data(raw_frame)
    train, val, test = split_data(cleaned, DataConfig(), seed=0)
    assert len(train) + len(val) + len(test) == len(cleaned)
    assert not set(train.index) & set(test.index)
    rate = cleaned["TARGET"].mean()
    for part in (train, val, test):
        assert abs(part["TARGET"].mean() - rate) < 0.03


def test_split_rejects_bad_fractions(raw_frame):
    with pytest.raises(ValueError):
        split_data(clean_raw_data(raw_frame),
                   DataConfig(val_size=0.6, test_size=0.5), seed=0)


def test_preprocessor_uses_train_statistics_only(raw_frame):
    cleaned = clean_raw_data(raw_frame)
    train, _, test = split_data(cleaned, DataConfig(), seed=0)
    prep = FeaturePreprocessor().fit(train)
    # A shifted test set must not change the fitted statistics (no leak).
    shifted = test.copy()
    shifted["EXT_SOURCE_2"] += 100
    before = prep.medians_.copy()
    prep.transform(shifted)
    pd.testing.assert_series_equal(before, prep.medians_)
    X_train = prep.transform(train)
    assert X_train.shape == (len(train), len(FEATURE_COLUMNS))
    assert not np.isnan(X_train).any()
    ext_idx = FEATURE_COLUMNS.index("EXT_SOURCE_2")
    assert abs(X_train[:, ext_idx].mean()) < 1e-5


def test_binary_features_are_not_scaled(prepared_data):
    for name in ("CODE_GENDER", *EXT_MISSING_COLUMNS):
        column = prepared_data.train.X[:, prepared_data.feature_index(name)]
        assert set(np.unique(column)) <= {0.0, 1.0}


def test_transform_before_fit_raises(raw_frame):
    with pytest.raises(RuntimeError):
        FeaturePreprocessor().transform(clean_raw_data(raw_frame))
