"""Shared fixtures: a synthetic Home-Credit-like dataset."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.data.preprocessing import (  # noqa: E402
    PreparedData,
    prepare_data_from_frame,
)
from src.utils.config import DataConfig  # noqa: E402

N_ROWS = 1500


def make_raw_frame(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Build a raw frame with the real schema and a gender effect.

    Args:
        n_rows: Number of rows.
        seed: Random seed.

    Returns:
        DataFrame with ``RAW_COLUMNS`` (including missing values and a few
        unknown genders).
    """
    rng = np.random.default_rng(seed)
    gender = rng.choice(["M", "F"], size=n_rows, p=[0.35, 0.65])
    income = rng.lognormal(12, 0.5, n_rows)
    credit = income * rng.uniform(1, 6, n_rows)
    annuity = credit * rng.uniform(0.03, 0.08, n_rows)
    ext = rng.uniform(0, 1, size=(n_rows, 3))
    logit = (-2.5 + 1.2 * (gender == "M") - 2.0 * (ext.mean(axis=1) - 0.5)
             + 0.5 * np.log(annuity / income + 1e-9) + 1.5)
    target = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({
        "TARGET": target.astype(int),
        "CODE_GENDER": gender,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -rng.integers(20 * 365, 65 * 365, n_rows),
        "EXT_SOURCE_1": ext[:, 0],
        "EXT_SOURCE_2": ext[:, 1],
        "EXT_SOURCE_3": ext[:, 2],
    })
    df.loc[rng.uniform(size=n_rows) < 0.5, "EXT_SOURCE_1"] = np.nan
    df.loc[rng.uniform(size=n_rows) < 0.2, "EXT_SOURCE_3"] = np.nan
    df.loc[:2, "CODE_GENDER"] = "XNA"
    df.loc[5, "AMT_ANNUITY"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Synthetic raw dataset."""
    return make_raw_frame()


@pytest.fixture(scope="session")
def prepared_data(raw_frame: pd.DataFrame) -> PreparedData:
    """Synthetic dataset after cleaning, splitting and scaling."""
    return prepare_data_from_frame(raw_frame, DataConfig(), seed=0)
