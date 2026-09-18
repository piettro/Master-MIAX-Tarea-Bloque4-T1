"""Global seeding helpers."""

from __future__ import annotations

import logging
import os
import random

import keras
import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and Keras/TensorFlow random generators.

    Args:
        seed: Non-negative integer seed.
        deterministic_ops: If True, request deterministic TensorFlow
            kernels (slightly slower, bit-reproducible on CPU).

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic_ops and keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
    logger.debug("Global seed set to %d", seed)
