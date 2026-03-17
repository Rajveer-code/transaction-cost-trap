#!/usr/bin/env python3
"""Ensemble sentiment analysis combining FinBERT, VADER, and TextBlob.

Implements:
- VADER sentiment (vaderSentiment)
- TextBlob sentiment (textblob)
- Ensemble aggregation (simple mean, weighted, confidence-weighted)
- Temporal aggregation (1d/3d/7d rolling windows per ticker)
- Cross-ticker sentiment tracking

Input: data/news/headlines_with_sentiment.parquet
Output: data/news/headlines_with_features.parquet

Usage (PowerShell):
  python .\scripts\ensemble_sentiment.py --input .\data\news\headlines_with_sentiment.parquet --output .\data\news\headlines_with_features.parquet

Requirements: Python 3.10+, pandas, vaderSentiment, textblob, pyarrow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# json already imported above



def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'ensemble_errors.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_vader_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Get VADER compound sentiment score."""
    try:
        scores = analyzer.polarity_scores(text)
        return float(scores.get('compound', 0.0))
    except Exception as e:
        logging.warning(f"VADER error for '{text[:50]}...': {e}")
        return 0.0


def get_textblob_sentiment(text: str) -> float:
    """Get TextBlob polarity sentiment score."""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        return float(polarity)
    except Exception as e:
        logging.warning(f"TextBlob error for '{text[:50]}...': {e}")
        return 0.0


def compute_ensemble_sentiments(row: Dict) -> Dict:
    """Compute 3 ensemble sentiment aggregations."""
    finbert_score = row.get('sentiment_score', 0.0)
    finbert_conf = row.get('sentiment_confidence', 0.0)
    vader_score = row.get('vader_compound', 0.0)
    textblob_score = row.get('textblob_polarity', 0.0)

    # 1. Simple mean
    sentiment_mean = (finbert_score + vader_score + textblob_score) / 3.0

    # 2. Weighted mean (FinBERT 50%, VADER 30%, TextBlob 20%)
    sentiment_weighted = (0.5 * finbert_score) + (0.3 * vader_score) + (0.2 * textblob_score)

    # 3. Confidence-weighted (weight FinBERT by its confidence)
    # numerator: FinBERT weighted by confidence + equal weight VADER + TextBlob
    # denominator: confidence factor + 1 + 1
    numerator = (finbert_score * finbert_conf) + vader_score + textblob_score
    denominator = finbert_conf + 2.0
    sentiment_confidence_weighted = numerator / denominator if denominator > 0 else 0.0

    return {
        'sentiment_mean': float(sentiment_mean),
        'sentiment_weighted': float(sentiment_weighted),
        'sentiment_confidence_weighted': float(sentiment_confidence_weighted),
    }


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal aggregation features per ticker (1d/3d/7d)."""
    # Ensure published_at is datetime
    df['published_at'] = pd.to_datetime(df['published_at'])

    # Sort by ticker and published_at
    df = df.sort_values(['ticker', 'published_at']).reset_index(drop=True)

    # Initialize temporal columns
    temporal_cols = ['sentiment_1d', 'sentiment_3d', 'sentiment_7d',
                     'sentiment_count_1d', 'sentiment_count_3d', 'sentiment_count_7d']
    for col in temporal_cols:
        df[col] = 0.0

    # Process each row
    for idx, row in df.iterrows():
        ticker = row['ticker']
        published_at = row['published_at']

        # Filter same-ticker headlines
        same_ticker = df[df['ticker'] == ticker].copy()

        # 1-day window
        window_1d = same_ticker[
            (same_ticker['published_at'] >= published_at - timedelta(days=1)) &
            (same_ticker['published_at'] <= published_at)
        ]
        df.at[idx, 'sentiment_1d'] = float(window_1d['sentiment_weighted'].mean()) if len(window_1d) > 0 else 0.0
        df.at[idx, 'sentiment_count_1d'] = int(len(window_1d))

        # 3-day window
        window_3d = same_ticker[
            (same_ticker['published_at'] >= published_at - timedelta(days=3)) &
            (same_ticker['published_at'] <= published_at)
        ]
        df.at[idx, 'sentiment_3d'] = float(window_3d['sentiment_weighted'].mean()) if len(window_3d) > 0 else 0.0
        df.at[idx, 'sentiment_count_3d'] = int(len(window_3d))

        # 7-day window
        window_7d = same_ticker[
            (same_ticker['published_at'] >= published_at - timedelta(days=7)) &
            (same_ticker['published_at'] <= published_at)
        ]
        df.at[idx, 'sentiment_7d'] = float(window_7d['sentiment_weighted'].mean()) if len(window_7d) > 0 else 0.0
        df.at[idx, 'sentiment_count_7d'] = int(len(window_7d))

    return df


def compute_cross_ticker_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """For multi-entity headlines, extract sentiment for other tickers mentioned."""
    df['other_ticker_sentiment'] = df.apply(
        lambda row: _get_other_ticker_sentiment(row),
        axis=1
    )
    return df


def _get_other_ticker_sentiment(row: Dict) -> Dict[str, float]:
    """Extract sentiment for non-primary tickers in a headline."""
    ticker = row.get('ticker')
    matched_tickers = row.get('matched_tickers', [])
    entity_sentiment_map = row.get('entity_sentiment_map', {})

    # Safely handle lists, None values, and empty arrays
    if not isinstance(matched_tickers, list):
        matched_tickers = []
    if len(matched_tickers) == 0:
        return {}

    # All tickers except the primary one
    other_tickers = [t for t in matched_tickers if t != ticker]
    other_sentiment = {}
    for t in other_tickers:
        if t in entity_sentiment_map:
            other_sentiment[t] = float(entity_sentiment_map[t])

    return other_sentiment


def main_process(input_path: str, output_path: str):
    """Main processing pipeline."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    # Load VADER analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Load input data
    logging.info(f"Loading headlines from {input_path}")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logging.error(f"Error loading parquet file: {e}")
        print(f"Error: Could not load {input_path}: {e}")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} headlines")

    # Add VADER sentiment
    logging.info("Computing VADER sentiment...")
    df['vader_compound'] = df['title'].apply(lambda x: get_vader_sentiment(x, analyzer))

    # Add TextBlob sentiment
    logging.info("Computing TextBlob sentiment...")
    df['textblob_polarity'] = df['title'].apply(lambda x: get_textblob_sentiment(x))

    # Compute ensemble sentiments
    logging.info("Computing ensemble sentiments...")
    ensemble_features = df.apply(compute_ensemble_sentiments, axis=1)
    df['sentiment_mean'] = ensemble_features.apply(lambda x: x['sentiment_mean'])
    df['sentiment_weighted'] = ensemble_features.apply(lambda x: x['sentiment_weighted'])
    df['sentiment_confidence_weighted'] = ensemble_features.apply(lambda x: x['sentiment_confidence_weighted'])

    # Compute temporal features
    logging.info("Computing temporal aggregation features (1d/3d/7d)...")
    df = compute_temporal_features(df)

    # Compute cross-ticker sentiment
    logging.info("Computing cross-ticker sentiment...")
    df = compute_cross_ticker_sentiment(df)

    # Ensure datetime columns
    for col in ['published_at', 'collected_at']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Convert dict columns to JSON strings for Parquet compatibility
    if 'other_ticker_sentiment' in df.columns:
        df['other_ticker_sentiment'] = df['other_ticker_sentiment'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else '{}'
        )

    if 'entity_sentiment_map' in df.columns:
        df['entity_sentiment_map'] = df['entity_sentiment_map'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else '{}'
        )

    # Save output
    logging.info(f"Saving results to {output_path}")
    df.to_parquet(output_path, index=False)
    logging.info("Note: Dict columns converted to JSON strings for Parquet compatibility")

    # Compute statistics
    stats = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'input_file': input_path,
        'output_file': output_path,
        'total_headlines': len(df),
        'new_features_added': [
            'vader_compound',
            'textblob_polarity',
            'sentiment_mean',
            'sentiment_weighted',
            'sentiment_confidence_weighted',
            'sentiment_1d',
            'sentiment_3d',
            'sentiment_7d',
            'sentiment_count_1d',
            'sentiment_count_3d',
            'sentiment_count_7d',
            'other_ticker_sentiment',
        ],
        'ensemble_statistics': {
            'sentiment_mean_avg': float(df['sentiment_mean'].mean()),
            'sentiment_mean_std': float(df['sentiment_mean'].std()),
            'sentiment_weighted_avg': float(df['sentiment_weighted'].mean()),
            'sentiment_weighted_std': float(df['sentiment_weighted'].std()),
            'sentiment_confidence_weighted_avg': float(df['sentiment_confidence_weighted'].mean()),
            'sentiment_confidence_weighted_std': float(df['sentiment_confidence_weighted'].std()),
        },
        'vader_statistics': {
            'vader_mean': float(df['vader_compound'].mean()),
            'vader_std': float(df['vader_compound'].std()),
            'vader_min': float(df['vader_compound'].min()),
            'vader_max': float(df['vader_compound'].max()),
        },
        'textblob_statistics': {
            'textblob_mean': float(df['textblob_polarity'].mean()),
            'textblob_std': float(df['textblob_polarity'].std()),
            'textblob_min': float(df['textblob_polarity'].min()),
            'textblob_max': float(df['textblob_polarity'].max()),
        },
        'temporal_statistics': {
            'sentiment_1d_mean': float(df['sentiment_1d'].mean()),
            'sentiment_3d_mean': float(df['sentiment_3d'].mean()),
            'sentiment_7d_mean': float(df['sentiment_7d'].mean()),
            'sentiment_count_1d_mean': float(df['sentiment_count_1d'].mean()),
            'sentiment_count_3d_mean': float(df['sentiment_count_3d'].mean()),
            'sentiment_count_7d_mean': float(df['sentiment_count_7d'].mean()),
        },
        'cross_ticker_statistics': {
            'headlines_with_other_tickers': int(df['other_ticker_sentiment'].apply(lambda x: len(x) > 0).sum()),
            'percentage': float(df['other_ticker_sentiment'].apply(lambda x: len(x) > 0).mean() * 100),
        },
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, 'ensemble_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print('\n' + '='*70)
    print('ENSEMBLE SENTIMENT & FEATURE ENGINEERING SUMMARY')
    print('='*70)
    print(f"Total headlines processed: {stats['total_headlines']}")
    print(f"\nNew Features Added ({len(stats['new_features_added'])} total):")
    for feat in stats['new_features_added']:
        print(f"  - {feat}")
    print(f"\nEnsemble Sentiment Statistics:")
    print(f"  sentiment_mean:                    {stats['ensemble_statistics']['sentiment_mean_avg']:7.4f} ± {stats['ensemble_statistics']['sentiment_mean_std']:.4f}")
    print(f"  sentiment_weighted:                {stats['ensemble_statistics']['sentiment_weighted_avg']:7.4f} ± {stats['ensemble_statistics']['sentiment_weighted_std']:.4f}")
    print(f"  sentiment_confidence_weighted:     {stats['ensemble_statistics']['sentiment_confidence_weighted_avg']:7.4f} ± {stats['ensemble_statistics']['sentiment_confidence_weighted_std']:.4f}")
    print(f"\nVADER Statistics:")
    print(f"  Mean:     {stats['vader_statistics']['vader_mean']:7.4f}")
    print(f"  Std:      {stats['vader_statistics']['vader_std']:7.4f}")
    print(f"  Range:    [{stats['vader_statistics']['vader_min']:.4f}, {stats['vader_statistics']['vader_max']:.4f}]")
    print(f"\nTextBlob Statistics:")
    print(f"  Mean:     {stats['textblob_statistics']['textblob_mean']:7.4f}")
    print(f"  Std:      {stats['textblob_statistics']['textblob_std']:7.4f}")
    print(f"  Range:    [{stats['textblob_statistics']['textblob_min']:.4f}, {stats['textblob_statistics']['textblob_max']:.4f}]")
    print(f"\nTemporal Aggregation (per ticker):")
    print(f"  Avg sentiment 1d:  {stats['temporal_statistics']['sentiment_1d_mean']:7.4f}")
    print(f"  Avg sentiment 3d:  {stats['temporal_statistics']['sentiment_3d_mean']:7.4f}")
    print(f"  Avg sentiment 7d:  {stats['temporal_statistics']['sentiment_7d_mean']:7.4f}")
    print(f"  Avg count 1d:      {stats['temporal_statistics']['sentiment_count_1d_mean']:7.2f}")
    print(f"  Avg count 3d:      {stats['temporal_statistics']['sentiment_count_3d_mean']:7.2f}")
    print(f"  Avg count 7d:      {stats['temporal_statistics']['sentiment_count_7d_mean']:7.2f}")
    print(f"\nCross-Ticker Sentiment:")
    print(f"  Headlines with other tickers: {stats['cross_ticker_statistics']['headlines_with_other_tickers']} ({stats['cross_ticker_statistics']['percentage']:.1f}%)")
    print(f"\nMetadata saved to: {metadata_path}")
    print('='*70 + '\n')

    logging.info("Ensemble sentiment and feature engineering complete")


def parse_args():
    p = argparse.ArgumentParser(description='Ensemble sentiment analysis and feature engineering')
    p.add_argument('--input', type=str, default=os.path.join('data', 'news', 'headlines_with_sentiment.parquet'), help='Input parquet file')
    p.add_argument('--output', type=str, default=os.path.join('data', 'news', 'headlines_with_features.parquet'), help='Output parquet file')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        main_process(args.input, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
