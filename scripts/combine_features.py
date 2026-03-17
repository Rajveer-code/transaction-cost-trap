#!/usr/bin/env python3
"""Merge sentiment features with stock price data for modeling.

Architecture:
- Input 1: data/news/headlines_with_features.parquet (sentiment features)
- Input 2: data/stocks/historical_prices.parquet (OHLCV + technical indicators)
- Output: data/combined/features_for_modeling.parquet (merged dataset)

Merge Strategy:
- Use sentiment as forward-looking indicators
- Aggregate multiple headlines per (ticker, date)
- Create time-shifted lag features (1d, 3d, 7d)
- Add interaction features (sentiment × volume, sentiment × volatility)

Output: 21 stock columns + 11 sentiment columns + 3 interaction columns = 35 total features

Usage (PowerShell):
  python .\scripts\combine_features.py \\
    --news-input data/news/headlines_with_features.parquet \\
    --stock-input data/stocks/historical_prices.parquet \\
    --output data/combined/features_for_modeling.parquet

Requirements: Python 3.10+, pandas>=2.0, numpy, pyarrow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


def setup_logging(output_dir: str):
    """Initialize logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'merge_errors.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_and_validate_inputs(news_path: str, stock_path: str) -> tuple:
    """Load and validate input parquet files."""
    logging.info(f"Loading news features from {news_path}")
    df_news = pd.read_parquet(news_path)
    logging.info(f"Loaded {len(df_news)} news headlines")

    logging.info(f"Loading stock data from {stock_path}")
    df_stock = pd.read_parquet(stock_path)
    logging.info(f"Loaded {len(df_stock)} stock records across {df_stock['ticker'].nunique()} tickers")

    # Ensure datetime columns are timezone-naive
    if 'published_at' in df_news.columns:
        df_news['published_at'] = pd.to_datetime(df_news['published_at'])
        if df_news['published_at'].dt.tz is not None:
            df_news['published_at'] = df_news['published_at'].dt.tz_localize(None)
        df_news['published_date'] = df_news['published_at'].dt.date

    if 'date' in df_stock.columns:
        df_stock['date'] = pd.to_datetime(df_stock['date'])
        if df_stock['date'].dt.tz is not None:
            df_stock['date'] = df_stock['date'].dt.tz_localize(None)

    return df_news, df_stock


def aggregate_headlines_by_date(df_news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline sentiment features by (ticker, date)."""
    logging.info("Aggregating headlines by (ticker, date)")

    aggregations = {
        'sentiment_weighted': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0,
    }

    # Create aggregation dict dynamically
    agg_dict = {}
    if 'sentiment_weighted' in df_news.columns:
        agg_dict['sentiment_weighted'] = ['mean', 'std', 'count']

    if 'sentiment_label' in df_news.columns:
        # Will handle positive/negative ratio separately
        pass

    if 'vader_compound' in df_news.columns:
        agg_dict['vader_compound'] = 'mean'

    if 'textblob_polarity' in df_news.columns:
        agg_dict['textblob_polarity'] = 'mean'

    # Group by ticker and published_date
    df_agg = df_news.groupby(['ticker', 'published_date']).agg(agg_dict).reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df_agg.columns, pd.MultiIndex):
        df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

    # Rename aggregated columns
    rename_map = {}
    if 'sentiment_weighted_mean' in df_agg.columns:
        rename_map['sentiment_weighted_mean'] = 'sentiment_weighted_mean'
    if 'sentiment_weighted_std' in df_agg.columns:
        rename_map['sentiment_weighted_std'] = 'sentiment_weighted_std'
    if 'sentiment_weighted_count' in df_agg.columns:
        rename_map['sentiment_weighted_count'] = 'headline_count'
    if 'vader_compound_mean' in df_agg.columns:
        rename_map['vader_compound_mean'] = 'vader_mean'
    if 'textblob_polarity_mean' in df_agg.columns:
        rename_map['textblob_polarity_mean'] = 'textblob_mean'

    df_agg = df_agg.rename(columns=rename_map)

    # Compute positive/negative ratios
    def compute_sentiment_ratios(df_ticker_date):
        total = len(df_ticker_date)
        if total == 0:
            return pd.Series({'positive_ratio': 0.0, 'negative_ratio': 0.0})
        pos_count = (df_ticker_date['sentiment_label'] == 'positive').sum()
        neg_count = (df_ticker_date['sentiment_label'] == 'negative').sum()
        return pd.Series({
            'positive_ratio': pos_count / total,
            'negative_ratio': neg_count / total,
        })

    sentiment_ratios = df_news.groupby(['ticker', 'published_date']).apply(compute_sentiment_ratios).reset_index()
    df_agg = df_agg.merge(sentiment_ratios, on=['ticker', 'published_date'], how='left')

    # Rename date column for merge
    df_agg = df_agg.rename(columns={'published_date': 'date'})

    logging.info(f"Aggregated to {len(df_agg)} unique (ticker, date) pairs")

    return df_agg


def merge_sentiment_with_stock(
    df_stock: pd.DataFrame,
    df_sentiment: pd.DataFrame,
    date_tolerance: int = 1,
) -> pd.DataFrame:
    """Merge sentiment features with stock data using date tolerance."""
    logging.info(f"Merging sentiment with stock data (date tolerance: {date_tolerance} days)")

    # Convert dates to date objects for matching
    df_stock['merge_date'] = df_stock['date'].dt.date
    df_sentiment['sentiment_date'] = pd.to_datetime(df_sentiment['date']).dt.date

    # Create merged dataframe
    df_merged = df_stock.copy()

    # Initialize sentiment columns
    sentiment_cols = ['sentiment_weighted_mean', 'sentiment_weighted_std', 'headline_count',
                      'positive_ratio', 'negative_ratio', 'vader_mean', 'textblob_mean']
    for col in sentiment_cols:
        df_merged[col] = 0.0
    df_merged['headline_count'] = df_merged['headline_count'].astype('int64')

    # For each stock row, find matching sentiment data
    matched_count = 0
    for idx, stock_row in df_merged.iterrows():
        ticker = stock_row['ticker']
        stock_date = stock_row['merge_date']

        # Find sentiment rows for this ticker within date_tolerance
        df_ticker_sentiment = df_sentiment[df_sentiment['ticker'] == ticker]
        
        if len(df_ticker_sentiment) > 0:
            df_ticker_sentiment['date_diff'] = (
                pd.to_datetime(df_ticker_sentiment['sentiment_date']) -
                pd.to_datetime(stock_date)
            ).dt.days
            df_ticker_sentiment = df_ticker_sentiment[
                df_ticker_sentiment['date_diff'].abs() <= date_tolerance
            ]

            if len(df_ticker_sentiment) > 0:
                # Use the closest date
                closest_row = df_ticker_sentiment.loc[
                    df_ticker_sentiment['date_diff'].abs().idxmin()
                ]
                
                for col in sentiment_cols:
                    if col in closest_row.index:
                        df_merged.loc[idx, col] = closest_row[col]
                
                matched_count += 1

    df_merged = df_merged.drop(columns=['merge_date'], errors='ignore')

    logging.info(f"Matched {matched_count}/{len(df_merged)} stock rows with sentiment data")

    return df_merged


def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-shifted lagged sentiment features."""
    logging.info("Creating lagged sentiment features")

    # For each ticker, create lag features
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        df_ticker = df[mask].sort_values('date').reset_index(drop=True)

        if len(df_ticker) > 0:
            # Create lag features
            df.loc[mask, 'sentiment_lag_1d'] = df_ticker['sentiment_weighted_mean'].shift(1).fillna(0.0)
            df.loc[mask, 'sentiment_lag_3d'] = df_ticker['sentiment_weighted_mean'].shift(3).fillna(0.0)
            df.loc[mask, 'sentiment_lag_7d'] = df_ticker['sentiment_weighted_mean'].shift(7).fillna(0.0)

    # Fill any remaining NaNs with 0
    df['sentiment_lag_1d'] = df['sentiment_lag_1d'].fillna(0.0)
    df['sentiment_lag_3d'] = df['sentiment_lag_3d'].fillna(0.0)
    df['sentiment_lag_7d'] = df['sentiment_lag_7d'].fillna(0.0)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between sentiment and price/volume indicators."""
    logging.info("Creating interaction features")

    # Interaction: sentiment × volume change
    if 'volume_change' in df.columns:
        df['sentiment_x_volume'] = (
            df['sentiment_weighted_mean'] * df['volume_change'].fillna(0)
        )
    else:
        df['sentiment_x_volume'] = 0.0

    # Interaction: sentiment × volatility
    if 'rolling_volatility_5d' in df.columns:
        df['sentiment_x_volatility'] = (
            df['sentiment_weighted_mean'] * df['rolling_volatility_5d'].fillna(0)
        )
    else:
        df['sentiment_x_volatility'] = 0.0

    # Sentiment momentum: current sentiment - 1-day lagged
    df['sentiment_momentum'] = (
        df['sentiment_weighted_mean'] - df['sentiment_lag_1d']
    )

    return df


def validate_and_report(df: pd.DataFrame, df_original_stock: pd.DataFrame) -> Dict:
    """Validate merged data and compute statistics."""
    stats = {}

    stats['total_rows_stock'] = len(df_original_stock)
    stats['total_rows_merged'] = len(df)
    stats['tickers_in_data'] = int(df['ticker'].nunique())

    # Coverage statistics
    covered_dates = (df['headline_count'] > 0).sum()
    stats['dates_with_news_coverage'] = int(covered_dates)
    stats['coverage_percentage'] = float((covered_dates / len(df) * 100) if len(df) > 0 else 0)

    # Headline statistics
    total_headlines = int(df['headline_count'].sum())
    stats['total_headlines_integrated'] = total_headlines
    if covered_dates > 0:
        stats['avg_headlines_per_covered_date'] = float(total_headlines / covered_dates)

    # Sentiment statistics
    if 'sentiment_weighted_mean' in df.columns:
        stats['sentiment_mean'] = float(df['sentiment_weighted_mean'].mean())
        stats['sentiment_std'] = float(df['sentiment_weighted_mean'].std())
        stats['sentiment_min'] = float(df['sentiment_weighted_mean'].min())
        stats['sentiment_max'] = float(df['sentiment_weighted_mean'].max())

    # Correlation: sentiment vs target return
    if 'sentiment_weighted_mean' in df.columns and 'target_return_1d' in df.columns:
        valid_mask = ~(df['sentiment_weighted_mean'].isna() | df['target_return_1d'].isna())
        if valid_mask.sum() > 1:
            corr = df.loc[valid_mask, 'sentiment_weighted_mean'].corr(
                df.loc[valid_mask, 'target_return_1d']
            )
            stats['correlation_sentiment_vs_target_return'] = float(corr)

    # Data quality
    stats['missing_values'] = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            stats['missing_values'][col] = int(missing_count)

    return stats


def main_process(
    news_input: str,
    stock_input: str,
    output_path: str,
    date_tolerance: int = 1,
):
    """Main merge pipeline."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Starting feature merge pipeline")
    logging.info(f"News input: {news_input}")
    logging.info(f"Stock input: {stock_input}")
    logging.info(f"Output: {output_path}")

    # Load inputs
    df_news, df_stock = load_and_validate_inputs(news_input, stock_input)
    df_original_stock = df_stock.copy()

    # Aggregate sentiment by (ticker, date)
    df_sentiment = aggregate_headlines_by_date(df_news)

    # Merge sentiment with stock
    df_merged = merge_sentiment_with_stock(df_stock, df_sentiment, date_tolerance)

    # Create lagged features
    df_merged = create_lagged_features(df_merged)

    # Create interaction features
    df_merged = create_interaction_features(df_merged)

    # Ensure all numeric columns are float, except headline_count and target_direction_1d
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['headline_count', 'target_direction_1d']:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    # Validate
    stats = validate_and_report(df_merged, df_original_stock)
    stats['run_at'] = datetime.now(timezone.utc).isoformat()
    stats['date_tolerance'] = date_tolerance

    # Save results
    logging.info(f"Saving merged data to {output_path}")
    df_merged.to_parquet(output_path, index=False, compression='snappy')

    # Save metadata
    metadata_path = os.path.join(output_dir, 'merge_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    print('\n' + '='*80)
    print('FEATURE MERGE SUMMARY')
    print('='*80)
    print(f"Stock data rows: {stats['total_rows_stock']}")
    print(f"Merged data rows: {stats['total_rows_merged']}")
    print(f"Tickers: {stats['tickers_in_data']}")
    print(f"\nNews Coverage:")
    print(f"  Dates with news: {stats['dates_with_news_coverage']} ({stats['coverage_percentage']:.1f}%)")
    print(f"  Total headlines integrated: {stats['total_headlines_integrated']}")
    if 'avg_headlines_per_covered_date' in stats:
        print(f"  Avg headlines per covered date: {stats['avg_headlines_per_covered_date']:.2f}")
    
    print(f"\nSentiment Statistics:")
    print(f"  Mean sentiment: {stats.get('sentiment_mean', 0):.4f}")
    print(f"  Std sentiment: {stats.get('sentiment_std', 0):.4f}")
    print(f"  Range: [{stats.get('sentiment_min', 0):.4f}, {stats.get('sentiment_max', 0):.4f}]")
    
    if 'correlation_sentiment_vs_target_return' in stats:
        print(f"\nModel Readiness:")
        print(f"  Sentiment-Return Correlation: {stats['correlation_sentiment_vs_target_return']:.4f}")
    
    print(f"\nOutput Schema:")
    print(f"  Total columns: {len(df_merged.columns)}")
    print(f"  Stock columns (original): 21")
    print(f"  Sentiment columns (new): 7")
    print(f"  Lag features (new): 3")
    print(f"  Interaction features (new): 3")
    print(f"\nMetadata saved to: {metadata_path}")
    print('='*80 + '\n')

    logging.info("Feature merge pipeline complete")


def parse_args():
    p = argparse.ArgumentParser(
        description='Merge sentiment features with stock price data for modeling'
    )
    p.add_argument(
        '--news-input',
        type=str,
        default=os.path.join('data', 'news', 'headlines_with_features.parquet'),
        help='Path to headlines_with_features.parquet',
    )
    p.add_argument(
        '--stock-input',
        type=str,
        default=os.path.join('data', 'stocks', 'historical_prices.parquet'),
        help='Path to historical_prices.parquet',
    )
    p.add_argument(
        '--output',
        type=str,
        default=os.path.join('data', 'combined', 'features_for_modeling.parquet'),
        help='Output parquet file path',
    )
    p.add_argument(
        '--date-tolerance',
        type=int,
        default=1,
        help='Days to match news to stock data (default: 1)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    try:
        main_process(
            args.news_input,
            args.stock_input,
            args.output,
            args.date_tolerance,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
