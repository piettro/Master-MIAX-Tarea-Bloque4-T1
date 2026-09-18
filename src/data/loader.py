"""Raw data loading for the Home Credit Default Risk dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.utils.config import DATASET_URL, RAW_COLUMNS

logger = logging.getLogger(__name__)


def load_raw_data(csv_path: Path, zip_path: Path,
                  columns: Sequence[str] = RAW_COLUMNS) -> pd.DataFrame:
    """Load the selected columns of ``application_train.csv``.

    The extracted CSV is preferred; if it does not exist the zipped copy
    shipped with the repository is read directly (no manual unzip needed).

    Args:
        csv_path: Path of the extracted CSV file.
        zip_path: Path of the zip archive containing the CSV.
        columns: Columns to read.

    Returns:
        DataFrame with exactly ``columns``.

    Raises:
        FileNotFoundError: If neither the CSV nor the zip file exists.
        ValueError: If the file lacks some of the requested columns or
            cannot be parsed.
    """
    csv_path, zip_path = Path(csv_path), Path(zip_path)
    if csv_path.is_file():
        source, compression = csv_path, None
    elif zip_path.is_file():
        source, compression = zip_path, "zip"
    else:
        raise FileNotFoundError(
            f"Dataset not found. Expected '{csv_path}' or '{zip_path}'. "
            f"Download application_train.csv from {DATASET_URL} and place "
            f"it (or its zip) inside '{csv_path.parent}'."
        )

    logger.info("Reading %s", source)
    try:
        df = pd.read_csv(source, usecols=list(columns),
                         compression=compression)
    except ValueError as exc:
        raise ValueError(
            f"Could not read columns {list(columns)} from '{source}': {exc}"
        ) from exc
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse '{source}': {exc}") from exc

    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])
    return df
