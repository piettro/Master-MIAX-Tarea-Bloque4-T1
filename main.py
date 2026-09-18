"""Command-line entry point of the B4-T1 trustworthy-networks project.

Usage:
    python main.py                 # full experiment (~20-30 min on CPU)
    python main.py --quick         # smoke test on a 6k-row sample
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fair + uncertainty-aware credit default classifier.")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny data sample and budgets (smoke test).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Folder for figures, tables and models.")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Path to application_train.csv (or .zip).")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Override the Keras Tuner trial budget.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the global random seed.")
    parser.add_argument("--log-level", default="INFO",
                        help="DEBUG, INFO, WARNING...")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the configuration from CLI flags and run the pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging(args.log_level)
    # Imported after logging setup so TensorFlow honours the log level.
    from src.pipeline import run_pipeline
    from src.utils.config import PipelineConfig

    config = PipelineConfig()
    if args.quick:
        config = config.quick()
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.max_trials is not None:
        config = replace(config, tuner=replace(
            config.tuner, max_trials=args.max_trials))
    if args.data_path is not None:
        path = args.data_path
        field = "zip_path" if path.suffix == ".zip" else "csv_path"
        config = replace(config, data=replace(config.data, **{field: path}))

    start = time.perf_counter()
    try:
        results = run_pipeline(config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Finished in %.1f min; %d artifacts written to %s",
                (time.perf_counter() - start) / 60, len(results.artifacts),
                config.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
