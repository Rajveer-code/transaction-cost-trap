# scripts/build_features_extended.py
"""
Build extended offline features for a single ticker.

- Loads OHLCV from data/prices/{ticker}_extended.csv
- Loads news from data/news/{ticker}_merged.csv (optional)
- Computes technical indicators (from src.feature_engineering.technical_indicators)
- Computes simple daily sentiment features (from src.feature_engineering.nlp_pipeline)
- Saves to data/features/{ticker}_features_extended.csv
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature_engineering.technical_indicators import calculate_all_technical_indicators
from src.feature_engineering.nlp_pipeline import process_news_data, aggregate_daily_sentiment

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------
def load_ticker_metadata(ticker: str, config_path: Path) -> Dict:
    """
    Load metadata for a ticker from config/tickers.json.
    Safe: if file is missing or broken, returns {}.
    """
    try:
        if not config_path.exists():
            logger.warning(f"[CONFIG] No tickers.json at {config_path}")
            return {}

        with open(config_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.warning(f"[CONFIG] tickers.json is not a dict. Ignoring.")
            return {}

        return data.get(ticker, {})

    except Exception as e:
        logger.error(f"[CONFIG] Failed reading {config_path}: {e}")
        return {}


# ---------------------------------------------------------------------
# Price loader
# ---------------------------------------------------------------------
def load_price_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """
    Load OHLCV prices from data/prices/{ticker}_extended.csv
    and return a DataFrame indexed by date.
    """
    price_file = data_dir / "prices" / f"{ticker}_extended.csv"

    if not price_file.exists():
        available = [f.name for f in (data_dir / "prices").glob("*.csv")]
        raise FileNotFoundError(
            f"[PRICE] Missing file: {price_file}\nAvailable: {available}"
        )

    try:
        df = pd.read_csv(price_file)
        logger.info(f"[PRICE] Loaded {price_file}")

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"[PRICE] Missing columns {missing} in {price_file}. "
                f"Found: {list(df.columns)}"
            )

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        logger.info(f"[PRICE] Shape after cleaning: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"[PRICE] Failed for {ticker}: {e}")
        raise


# ---------------------------------------------------------------------
# News loader
# ---------------------------------------------------------------------
def load_news_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """
    Load news from data/news/{ticker}_merged.csv.
    Accepts files where:
    - date is missing but datetime exists
    - or time_published exists
    - headline may be named title
    """
    news_file = data_dir / "news" / f"{ticker}_merged.csv"

    if not news_file.exists():
        logger.warning(f"[NEWS] No news file for {ticker}: {news_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(news_file)

        # ------------------------------------------------------
        # FIX MISSING DATE COLUMN
        # ------------------------------------------------------
        if "date" not in df.columns:
            if "datetime" in df.columns:
                # Example: 2025-01-01 02:43:00
                df["date"] = pd.to_datetime(df["datetime"], errors="coerce")

            elif "time_published" in df.columns:
                # Example: 20250101T024300
                df["date"] = pd.to_datetime(
                    df["time_published"].astype(str),
                    format="%Y%m%dT%H%M%S",
                    errors="coerce",
                )

            else:
                logger.warning(f"[NEWS] No usable date column in {news_file}")
                return pd.DataFrame()

        # ------------------------------------------------------
        # FIX MISSING HEADLINE COLUMN
        # ------------------------------------------------------
        if "headline" not in df.columns:
            if "title" in df.columns:
                df["headline"] = df["title"]
            else:
                logger.warning(f"[NEWS] No headline column in {news_file}")
                return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        return df

    except Exception as e:
        logger.error(f"[NEWS] Failed for {ticker}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------
# Build features for a single ticker
# ---------------------------------------------------------------------
def build_features_for_ticker(ticker: str, data_dir: Path) -> pd.DataFrame:
    """
    Orchestrates:
    - price loading
    - technical indicators
    - news sentiment aggregation
    - merging them into one feature table.
    """
    ticker = ticker.upper()
    logger.info("\n" + "=" * 60)
    logger.info(f"Processing {ticker}")
    logger.info("=" * 60)

    # Load metadata (not strictly needed, but kept for future use)
    config_path = project_root / "config" / "tickers.json"
    _ = load_ticker_metadata(ticker, config_path)

    # 1. Prices + technical indicators
    price_df = load_price_data(ticker, data_dir)
    if price_df.empty:
        raise ValueError(f"No price data for {ticker}")

    tech_df = calculate_all_technical_indicators(price_df)

    # 2. News + sentiment
    news_df = load_news_data(ticker, data_dir)

    if not news_df.empty:
        per_article = process_news_data(news_df)
        daily_sentiment = aggregate_daily_sentiment(per_article)
        if not daily_sentiment.empty:
            daily_sentiment = daily_sentiment.set_index("date")
        else:
            daily_sentiment = pd.DataFrame()
            logger.warning(f"[SENTIMENT] No aggregated sentiment for {ticker}")
    else:
        daily_sentiment = pd.DataFrame()
        logger.warning(f"[SENTIMENT] No news for {ticker}")

    # 3. Merge
    if not daily_sentiment.empty:
        features = tech_df.merge(
            daily_sentiment,
            left_index=True,
            right_index=True,
            how="left",
        )
        # forward-fill sentiment across days
        sentiment_cols = [
            c for c in daily_sentiment.columns if c in features.columns
        ]
        if sentiment_cols:
            features[sentiment_cols] = features[sentiment_cols].ffill()
    else:
        features = tech_df.copy()

    # 4. Add ticker column
    features["ticker"] = ticker

    # 5. Final clean
    features = features.ffill().bfill()

    logger.info(f"[FEATURES] Final shape for {ticker}: {features.shape}")
    return features.reset_index()  # date back as a column


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
def save_features(features_df: pd.DataFrame, ticker: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}_features_extended.csv"
    features_df.to_csv(out_path, index=False)
    logger.info(f"[SAVE] Wrote {out_path}")


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build extended features for a ticker")
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL)")
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
        help="Where to save features (default: data/features)",
    )

    args = parser.parse_args()
    ticker = args.ticker.upper()

    try:
        feats = build_features_for_ticker(ticker, args.data_dir)
        save_features(feats, ticker, args.output_dir)
        logger.info(f"[DONE] Successfully processed {ticker}")
    except Exception as e:
        logger.error(f"[FAIL] {ticker}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
