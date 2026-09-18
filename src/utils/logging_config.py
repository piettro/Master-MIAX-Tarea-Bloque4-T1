"""Root logger configuration (called once from ``main.py``)."""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
NOISY_LOGGERS = ("tensorflow", "absl", "h5py", "matplotlib", "PIL")


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger and silence noisy third-party loggers.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT,
                        datefmt=DATE_FORMAT, force=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
