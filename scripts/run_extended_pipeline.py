# scripts/run_extended_pipeline.py
"""
Run the extended offline feature pipeline for multiple tickers.

Usage:
    python scripts/run_extended_pipeline.py --tickers AAPL MSFT GOOGL
or:
    python scripts/run_extended_pipeline.py            # uses config/tickers.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd  # not strictly needed but harmless

# Ensure project root on path so we can import scripts & src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.build_features_extended import (
    build_features_for_ticker,
    save_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_tickers_from_config(config_path: Path) -> List[str]:
    """
    Load all tickers from config/tickers.json.
    Returns [] if file missing or invalid.
    """
    try:
        if not config_path.exists():
            logger.warning(f"[PIPELINE] No config file at {config_path}")
            return []

        with open(config_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning(f"[PIPELINE] tickers.json is not a dict.")
            return []

        return sorted(list(data.keys()))

    except Exception as e:
        logger.error(f"[PIPELINE] Failed reading {config_path}: {e}")
        return []


def run_pipeline(
    tickers: List[str],
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Run build_features_extended for each ticker."""
    success = []
    failed = []

    for t in tickers:
        t = t.upper()
        try:
            logger.info("\n" + "=" * 60)
            logger.info(f"[PIPELINE] Processing {t}")
            logger.info("=" * 60)

            feats = build_features_for_ticker(t, data_dir)
            save_features(feats, t, output_dir)
            success.append(t)

        except Exception as e:
            logger.error(f"[PIPELINE] Failed for {t}: {e}")
            failed.append(t)

    logger.info("\n" + "=" * 60)
    logger.info("[PIPELINE] Summary")
    logger.info("=" * 60)
    logger.info(f"Success: {success}")
    if failed:
        logger.warning(f"Failed: {failed}")
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run extended offline feature pipeline for multiple tickers"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Tickers to process (e.g. AAPL MSFT GOOGL). If omitted, uses config/tickers.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Output directory for features (default: data/features)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "tickers.json",
        help="Tickers config JSON (default: config/tickers.json)",
    )

    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_tickers_from_config(args.config)

    if not tickers:
        logger.error("[PIPELINE] No tickers to process.")
        sys.exit(1)

    run_pipeline(tickers, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
