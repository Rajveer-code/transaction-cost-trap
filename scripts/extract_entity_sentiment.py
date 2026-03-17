#!/usr/bin/env python3
"""
scripts/extract_entity_sentiment.py

Extract entity-level sentiment features from news headlines.

Novel Contribution: Entity-specific sentiment (CEO, Competitor) as separate features.
This captures insider signals (CEO concerns) vs general sentiment.

Input:
    data/news/raw_headlines.parquet (from download_news_historical.py)
    config/tickers.json (ticker metadata)

Output:
    data/news/entity_sentiment_features.parquet

Schema:
    - ticker (str)
    - date (datetime64[ns])
    - avg_ceo_sentiment (float): Mean sentiment of headlines mentioning CEO
    - avg_competitor_sentiment (float): Mean sentiment of headlines mentioning competitor
    - avg_headline_sentiment (float): Mean sentiment of all headlines
    - ceo_mention_count (int): Count of CEO mentions
    - competitor_mention_count (int): Count of competitor mentions
    - total_headlines (int): Total headlines on that date
    - entity_sentiment_gap (float): max(ceo, competitor) - min(ceo, competitor)
    - entity_density (float): (ceo_mentions + competitor_mentions) / total_headlines
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_entity_sentiment")

# Defaults
INPUT_FILE = PROJECT_ROOT / "data" / "news" / "raw_headlines.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "news" / "entity_sentiment_features.parquet"
TICKERS_FILE = PROJECT_ROOT / "config" / "tickers.json"

# ============================================================
# ENTITY MAPPINGS (Novel Research Data)
# ============================================================

ENTITY_MAPPINGS = {
    "AAPL": {
        "company_name": "Apple",
        "ceo": "Tim Cook",
        "ceo_aliases": ["tim cook", "cook"],
        "competitors": ["Samsung", "Google", "Microsoft"],
        "competitor_aliases": ["samsung", "alphabet", "google", "microsoft"],
    },
    "MSFT": {
        "company_name": "Microsoft",
        "ceo": "Satya Nadella",
        "ceo_aliases": ["satya nadella", "nadella"],
        "competitors": ["Apple", "Google", "Amazon"],
        "competitor_aliases": ["apple", "alphabet", "google", "amazon"],
    },
    "GOOGL": {
        "company_name": "Alphabet / Google",
        "ceo": "Sundar Pichai",
        "ceo_aliases": ["sundar pichai", "pichai"],
        "competitors": ["Microsoft", "Apple", "Amazon"],
        "competitor_aliases": ["microsoft", "apple", "amazon"],
    },
    "AMZN": {
        "company_name": "Amazon",
        "ceo": "Andy Jassy",
        "ceo_aliases": ["andy jassy", "jassy"],
        "competitors": ["Walmart", "Microsoft", "Google"],
        "competitor_aliases": ["walmart", "microsoft", "google"],
    },
    "META": {
        "company_name": "Meta Platforms",
        "ceo": "Mark Zuckerberg",
        "ceo_aliases": ["mark zuckerberg", "zuckerberg"],
        "competitors": ["TikTok", "Snapchat", "YouTube"],
        "competitor_aliases": ["tiktok", "snapchat", "youtube"],
    },
    "NVDA": {
        "company_name": "NVIDIA",
        "ceo": "Jensen Huang",
        "ceo_aliases": ["jensen huang", "huang"],
        "competitors": ["AMD", "Intel", "Qualcomm"],
        "competitor_aliases": ["amd", "intel", "qualcomm"],
    },
    "TSLA": {
        "company_name": "Tesla",
        "ceo": "Elon Musk",
        "ceo_aliases": ["elon musk", "musk"],
        "competitors": ["BYD", "Volkswagen", "General Motors"],
        "competitor_aliases": ["byd", "volkswagen", "gm", "general motors"],
    },
}


# ============================================================
# SENTIMENT ANALYSIS FUNCTIONS
# ============================================================

def initialize_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def get_sentiment_score(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        scores = analyzer.polarity_scores(text)
        return float(scores["compound"])
    except Exception:
        return 0.0


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def check_entity_mention(text: str, aliases: List[str]) -> bool:
    normalized = normalize_text(text)
    for alias in aliases:
        alias_lower = alias.lower()
        if f" {alias_lower} " in f" {normalized} " or \
           normalized.startswith(alias_lower + " ") or \
           normalized.endswith(" " + alias_lower) or \
           normalized == alias_lower:
            return True
    return False


def extract_entity_sentiment(headline: str, ticker: str, analyzer: SentimentIntensityAnalyzer) -> Dict:
    if not isinstance(headline, str) or not headline.strip():
        return {
            "headline_sentiment": 0.0,
            "ceo_sentiment": 0.0,
            "competitor_sentiment": 0.0,
            "has_ceo": 0,
            "has_competitor": 0,
        }
    
    headline_sentiment = get_sentiment_score(headline, analyzer)
    mapping = ENTITY_MAPPINGS.get(ticker, {})
    has_ceo = check_entity_mention(headline, mapping.get("ceo_aliases", []))
    has_competitor = check_entity_mention(headline, mapping.get("competitor_aliases", []))
    
    ceo_sentiment = headline_sentiment if has_ceo else 0.0
    competitor_sentiment = headline_sentiment if has_competitor else 0.0
    
    return {
        "headline_sentiment": headline_sentiment,
        "ceo_sentiment": ceo_sentiment,
        "competitor_sentiment": competitor_sentiment,
        "has_ceo": int(has_ceo),
        "has_competitor": int(has_competitor),
    }


def process_ticker_headlines(df: pd.DataFrame, ticker: str, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    ticker_df = df[df["ticker"] == ticker].copy()
    
    if ticker_df.empty:
        logger.warning(f"No headlines found for {ticker}")
        return pd.DataFrame()
    
    sentiment_data = []
    for idx, row in ticker_df.iterrows():
        headline = row.get("title") or row.get("headline") or ""
        sentiment = extract_entity_sentiment(headline, ticker, analyzer)
        sentiment_data.append(sentiment)
    
    sentiment_df = pd.DataFrame(sentiment_data)
    ticker_df = pd.concat([ticker_df, sentiment_df], axis=1)
    
    if "published_at" in ticker_df.columns:
        ticker_df["date"] = pd.to_datetime(ticker_df["published_at"]).dt.normalize()
    elif "date" in ticker_df.columns:
        ticker_df["date"] = pd.to_datetime(ticker_df["date"]).dt.normalize()
    else:
        logger.error(f"No date column found for {ticker}")
        return pd.DataFrame()
    
    return ticker_df


def aggregate_daily_entity_sentiment(ticker_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker_df.empty:
        return pd.DataFrame()
    
    daily_agg = ticker_df.groupby("date").agg({
        "headline_sentiment": ["mean", "std", "count"],
        "ceo_sentiment": ["mean", "sum"],
        "competitor_sentiment": ["mean", "sum"],
        "has_ceo": ["sum"],
        "has_competitor": ["sum"],
    }).reset_index()
    
    daily_agg.columns = [
        "date",
        "avg_headline_sentiment",
        "std_headline_sentiment",
        "total_headlines",
        "avg_ceo_sentiment",
        "ceo_sentiment_sum",
        "avg_competitor_sentiment",
        "competitor_sentiment_sum",
        "ceo_mention_count",
        "competitor_mention_count",
    ]
    
    daily_agg["ticker"] = ticker
    
    # Handle zero divisions
    daily_agg["avg_ceo_sentiment"] = daily_agg.apply(
        lambda r: r["avg_ceo_sentiment"] if r["ceo_mention_count"] > 0 else np.nan,
        axis=1
    )
    daily_agg["avg_competitor_sentiment"] = daily_agg.apply(
        lambda r: r["avg_competitor_sentiment"] if r["competitor_mention_count"] > 0 else np.nan,
        axis=1
    )
    
    # Entity gap and density
    daily_agg["entity_sentiment_gap"] = daily_agg.apply(
        lambda r: abs(
            (r["avg_ceo_sentiment"] if pd.notna(r["avg_ceo_sentiment"]) else 0) -
            (r["avg_competitor_sentiment"] if pd.notna(r["avg_competitor_sentiment"]) else 0)
        ),
        axis=1
    )
    
    daily_agg["entity_density"] = (
        (daily_agg["ceo_mention_count"] + daily_agg["competitor_mention_count"]) /
        daily_agg["total_headlines"].fillna(1)
    )
    
    # Reorder columns
    daily_agg = daily_agg[[
        "ticker",
        "date",
        "avg_headline_sentiment",
        "avg_ceo_sentiment",
        "avg_competitor_sentiment",
        "ceo_mention_count",
        "competitor_mention_count",
        "total_headlines",
        "entity_sentiment_gap",
        "entity_density",
        "std_headline_sentiment",
    ]]
    
    return daily_agg


def run(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load headlines
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    logger.info(f"Loading headlines from {input_file}")
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        logger.error(f"Failed to read parquet: {e}")
        return
    
    logger.info(f"Loaded {len(df)} headlines")
    
    # Initialize VADER
    analyzer = initialize_vader()
    logger.info("VADER sentiment analyzer initialized")
    
    # Process each ticker
    all_daily_features = []
    tickers = df["ticker"].unique()
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        ticker_df = process_ticker_headlines(df, ticker, analyzer)
        if not ticker_df.empty:
            daily_features = aggregate_daily_entity_sentiment(ticker_df, ticker)
            all_daily_features.append(daily_features)
            logger.info(f"  Aggregated {len(daily_features)} days for {ticker}")
    
    if not all_daily_features:
        logger.error("No daily features generated")
        return
    
    # Combine all
    final_df = pd.concat(all_daily_features, ignore_index=True)
    final_df = final_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    logger.info(f"Final dataset shape: {final_df.shape}")
    
    # Save
    try:
        final_df.to_parquet(output_file, index=False, engine="pyarrow")
        logger.info(f"Saved {len(final_df)} daily entity-sentiment features to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save parquet: {e}")


if __name__ == "__main__":
    run()
