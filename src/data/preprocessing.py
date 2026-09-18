"""Cleaning, leakage-free splitting, imputation and scaling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data
from src.utils.config import (
    AGE_COLUMN,
    AMOUNT_COLUMNS,
    BINARY_COLUMNS,
    BIRTH_COLUMN,
    CONTINUOUS_COLUMNS,
    DAYS_PER_YEAR,
    EXT_MISSING_COLUMNS,
    EXT_SOURCE_COLUMNS,
    FEATURE_COLUMNS,
    GENDER_ENCODING,
    RAW_COLUMNS,
    SENSITIVE_COLUMN,
    TARGET_COLUMN,
    DataConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """One partition of the dataset.

    Attributes:
        X: Feature matrix (float32, ``FEATURE_COLUMNS`` order).
        y: Binary target (1 = payment difficulties).
        s: Sensitive attribute (gender, M=0 / F=1).
    """

    X: np.ndarray
    y: np.ndarray
    s: np.ndarray

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.y)


@dataclass(frozen=True)
class PreparedData:
    """Train / validation / test splits plus preprocessing metadata.

    Attributes:
        train: Training split.
        val: Validation split (early stopping and model selection).
        test: Test split, used only once at the very end.
        feature_names: Column names of ``X``.
        amount_log_stats: ``{column: (mean, std)}`` of the log1p amounts
            fitted on train; the debt-ratio layer uses it to undo the
            standardisation and recover monetary ratios.
    """

    train: DataSplit
    val: DataSplit
    test: DataSplit
    feature_names: tuple[str, ...]
    amount_log_stats: dict[str, tuple[float, float]]

    def feature_index(self, name: str) -> int:
        """Return the column index of a feature.

        Args:
            name: Feature name.

        Returns:
            Position of ``name`` in ``feature_names``.

        Raises:
            KeyError: If the feature does not exist.
        """
        try:
            return self.feature_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown feature: {name!r}") from exc


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-wise transformations that need no fitted statistics.

    Steps: drop unknown genders, encode gender (M=0, F=1), convert age to
    positive years, log-transform heavy-tailed amounts and add one
    missing-value flag per EXT_SOURCE column (before imputation, so the
    flags keep the "data quality" information used in the uncertainty
    analysis).

    Args:
        df: Raw data containing ``RAW_COLUMNS``.

    Returns:
        A new cleaned DataFrame.

    Raises:
        ValueError: If required columns are missing or nothing is left.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[df[SENSITIVE_COLUMN].isin(GENDER_ENCODING)].copy()
    dropped = len(df) - len(out)
    if dropped:
        # NOTE: improved over professor's solution — the class notebook
        # mapped unknown genders ("XNA") to 0 (male) via fillna(0).
        logger.info("Dropped %d rows with unknown gender", dropped)
    if out.empty:
        raise ValueError("No rows left after removing unknown genders")

    out[SENSITIVE_COLUMN] = out[SENSITIVE_COLUMN].map(GENDER_ENCODING)
    out[AGE_COLUMN] = out.pop(BIRTH_COLUMN).abs() / DAYS_PER_YEAR
    for col in AMOUNT_COLUMNS:
        # NOTE: improved over original submission — incomes span several
        # orders of magnitude; log1p tames the tail before scaling.
        out[col] = np.log1p(out[col].clip(lower=0))
    for col, flag in zip(EXT_SOURCE_COLUMNS, EXT_MISSING_COLUMNS):
        out[flag] = out[col].isna().astype("int8")
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, config: DataConfig,
               seed: int) -> tuple[pd.DataFrame, pd.DataFrame,
                                   pd.DataFrame]:
    """Stratified train / validation / test split.

    Args:
        df: Cleaned data.
        config: Split fractions and optional subsample size.
        seed: Random seed.

    Returns:
        ``(train_df, val_df, test_df)``.

    Raises:
        ValueError: If the split fractions are invalid.
    """
    if not 0 < config.val_size < 1 or not 0 < config.test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if config.val_size + config.test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    if config.sample_size is not None and config.sample_size < len(df):
        df, _ = train_test_split(df, train_size=config.sample_size,
                                 stratify=df[TARGET_COLUMN],
                                 random_state=seed)

    holdout = config.val_size + config.test_size
    train_df, rest = train_test_split(df, test_size=holdout,
                                      stratify=df[TARGET_COLUMN],
                                      random_state=seed)
    val_df, test_df = train_test_split(
        rest, test_size=config.test_size / holdout,
        stratify=rest[TARGET_COLUMN], random_state=seed,
    )
    return train_df, val_df, test_df


class FeaturePreprocessor:
    """Median imputation + standardisation fitted on the training split.

    NOTE: improved over original submission and professor's notebook —
    both imputed medians on the full dataset before splitting (test-set
    leakage). Every statistic here is learnt from the training split only.
    """

    def __init__(self) -> None:
        """Create an unfitted preprocessor."""
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> FeaturePreprocessor:
        """Learn medians, means and standard deviations on train.

        Args:
            train_df: Cleaned training data.

        Returns:
            ``self``.
        """
        continuous = train_df[list(CONTINUOUS_COLUMNS)]
        self.medians_ = continuous.median()
        imputed = continuous.fillna(self.medians_)
        self.means_ = imputed.mean()
        # Guard against constant columns (division by zero).
        self.stds_ = imputed.std(ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Impute, standardise and order the features.

        Args:
            df: Cleaned data (any split).

        Returns:
            float32 matrix with columns in ``FEATURE_COLUMNS`` order.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.medians_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first")
        continuous = df[list(CONTINUOUS_COLUMNS)].fillna(self.medians_)
        scaled = (continuous - self.means_) / self.stds_
        features = pd.concat([scaled, df[list(BINARY_COLUMNS)]], axis=1)
        return features[list(FEATURE_COLUMNS)].to_numpy(dtype="float32")

    def amount_log_stats(self) -> dict[str, tuple[float, float]]:
        """Return the (mean, std) of every log-amount column.

        Returns:
            Mapping ``{column: (mean, std)}``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.means_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first")
        return {col: (float(self.means_[col]), float(self.stds_[col]))
                for col in AMOUNT_COLUMNS}


def _to_split(df: pd.DataFrame, preprocessor: FeaturePreprocessor
              ) -> DataSplit:
    """Convert a cleaned DataFrame into a :class:`DataSplit`."""
    return DataSplit(
        X=preprocessor.transform(df),
        y=df[TARGET_COLUMN].to_numpy(dtype="float32"),
        s=df[SENSITIVE_COLUMN].to_numpy(dtype="float32"),
    )


def prepare_data_from_frame(df: pd.DataFrame, config: DataConfig,
                            seed: int) -> PreparedData:
    """Run cleaning, splitting and preprocessing on a raw DataFrame.

    Args:
        df: Raw data with ``RAW_COLUMNS``.
        config: Data configuration.
        seed: Random seed.

    Returns:
        The prepared splits.
    """
    cleaned = clean_raw_data(df)
    train_df, val_df, test_df = split_data(cleaned, config, seed)
    preprocessor = FeaturePreprocessor().fit(train_df)
    data = PreparedData(
        train=_to_split(train_df, preprocessor),
        val=_to_split(val_df, preprocessor),
        test=_to_split(test_df, preprocessor),
        feature_names=FEATURE_COLUMNS,
        amount_log_stats=preprocessor.amount_log_stats(),
    )
    logger.info("Split sizes: train=%d, val=%d, test=%d | default rate "
                "(train)=%.3f | female share (train)=%.3f",
                len(data.train), len(data.val), len(data.test),
                data.train.y.mean(), data.train.s.mean())
    return data


def prepare_data(config: DataConfig, seed: int) -> PreparedData:
    """Load the dataset from disk and prepare the splits.

    Args:
        config: Data configuration (paths, split sizes).
        seed: Random seed.

    Returns:
        The prepared splits.
    """
    raw = load_raw_data(config.csv_path, config.zip_path)
    return prepare_data_from_frame(raw, config, seed)
