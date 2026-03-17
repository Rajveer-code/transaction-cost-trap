#!/usr/bin/env python3
"""Download historical stock prices and engineer technical indicators.

Implements:
- yfinance download for 7 tickers
- OHLCV data collection
- Derived price/volatility features
- Technical indicators (RSI, MACD, Bollinger Bands)
- Target variable engineering (next-day return/direction)

Input: data/news/headlines_with_features.parquet (for ticker discovery)
Output: data/stocks/historical_prices.parquet

Usage (PowerShell):
  python .\scripts\collect_stock_data.py --start-date 2025-12-02 --end-date 2025-12-11

Requirements: Python 3.10+, yfinance>=0.2.32, pandas>=2.0, pyarrow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)


TICKERS_DEFAULT = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'stock_errors.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def download_ticker_data(ticker: str, start_date: str, end_date: str, retries: int = 3, base_delay: float = 2.0) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a ticker with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                logging.warning(f"No data returned for {ticker}")
                return None
            df['ticker'] = ticker
            return df
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1))
            if attempt < retries:
                logging.warning(f"{ticker} download failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logging.error(f"{ticker} download failed after {retries} retries: {e}")
                return None


def compute_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price change features."""
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    df['daily_return'] = df[price_col].pct_change()
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    df['open_to_close'] = (df['close'] - df['open']) / df['open']
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility indicators."""
    df['rolling_volatility_5d'] = df['daily_return'].rolling(window=5).std()
    df['rolling_volatility_20d'] = df['daily_return'].rolling(window=20).std()
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features."""
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD and signal line."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    """Compute Bollinger Bands."""
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper, lower


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators."""
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    df['rsi_14d'] = compute_rsi(df[price_col], period=14)
    macd, macd_signal = compute_macd(df[price_col])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    bollinger_upper, bollinger_lower = compute_bollinger_bands(df[price_col], period=20)
    df['bollinger_upper'] = bollinger_upper
    df['bollinger_lower'] = bollinger_lower
    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target variables (next-day return and direction)."""
    df['target_return_1d'] = df['daily_return'].shift(-1)
    df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)
    return df


def process_ticker(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Download and process all features for a single ticker."""
    # Download data
    df = download_ticker_data(ticker, start_date, end_date)
    if df is None or df.empty:
        logging.error(f"Skipping {ticker} due to download failure")
        return None

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to make date a column
    df = df.reset_index()

    # Rename columns to lowercase standard names
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Adj Close': 'adj_close',
        'Volume': 'volume',
    })

    # Ensure all column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Forward-fill missing prices (handle weekends/holidays)
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')

    # Compute features
    df = compute_price_changes(df)
    df = compute_volatility(df)
    df = compute_volume_features(df)
    df = compute_technical_indicators(df)
    df = compute_targets(df)

    # Keep only needed columns
    output_cols = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
        'daily_return', 'intraday_range', 'open_to_close',
        'rolling_volatility_5d', 'rolling_volatility_20d',
        'volume_change', 'volume_ma_20d',
        'rsi_14d', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
        'target_return_1d', 'target_direction_1d'
    ]
    df = df[[col for col in output_cols if col in df.columns]]

    return df


def validate_data(df: pd.DataFrame) -> Dict:
    """Validate downloaded data and compute statistics."""
    stats = {}
    per_ticker = df.groupby('ticker').agg({
        'date': ['min', 'max', 'count'],
        'daily_return': ['mean', 'std'],
        'rolling_volatility_20d': 'mean',
        'volume': 'mean',
    }).reset_index()

    ticker_stats = []
    for ticker in df['ticker'].unique():
        df_ticker = df[df['ticker'] == ticker]
        date_min = df_ticker['date'].min()
        date_max = df_ticker['date'].max()
        date_count = len(df_ticker)

        # Calculate expected trading days (rough estimate: 252 per year)
        date_range = (date_max - date_min).days
        expected_days = max(1, (date_range / 365) * 252)
        missing_days = max(0, int(expected_days - date_count))

        ticker_stats.append({
            'ticker': ticker,
            'date_min': date_min.isoformat(),
            'date_max': date_max.isoformat(),
            'data_points': int(date_count),
            'missing_days': missing_days,
            'mean_daily_return': float(df_ticker['daily_return'].mean()),
            'volatility_20d_mean': float(df_ticker['rolling_volatility_20d'].mean()),
            'mean_volume': float(df_ticker['volume'].mean()),
        })

    stats['per_ticker'] = ticker_stats

    # Correlation matrix
    try:
        pivot_returns = df.pivot_table(index='date', columns='ticker', values='daily_return')
        corr_matrix = pivot_returns.corr().round(4)
        stats['correlation_matrix'] = corr_matrix.to_dict()
    except Exception as e:
        logging.warning(f"Correlation matrix computation failed: {e}")
        stats['correlation_matrix'] = {}

    return stats


def main_process(start_date: str, end_date: str, output_path: str, tickers: Optional[List[str]] = None):
    """Main processing pipeline."""
    if tickers is None:
        tickers = TICKERS_DEFAULT

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    logging.info(f"Starting stock data collection for {len(tickers)} tickers")
    logging.info(f"Date range: {start_date} to {end_date}")

    # Download and process each ticker
    all_dfs = []
    for ticker in tqdm(tickers, desc='Downloading'):
        df = process_ticker(ticker, start_date, end_date)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        logging.error("No data collected for any ticker")
        print("Warning: Could not download data for any ticker (market may be closed)")
        # Don't exit with error - just skip this run
        return

    # Combine all tickers
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined.sort_values(['ticker', 'date']).reset_index(drop=True)

    logging.info(f"Combined data: {len(df_combined)} rows across {df_combined['ticker'].nunique()} tickers")

    # Validate data
    stats = validate_data(df_combined)
    stats['run_at'] = datetime.now(timezone.utc).isoformat()
    stats['start_date'] = start_date
    stats['end_date'] = end_date
    stats['total_rows'] = len(df_combined)
    stats['total_tickers'] = df_combined['ticker'].nunique()

    # Save results
    logging.info(f"Saving stock data to {output_path}")
    df_combined.to_parquet(output_path, index=False)

    # Save metadata
    metadata_path = os.path.join(output_dir, 'stock_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    # Print summary
    print('\n' + '='*70)
    print('STOCK DATA COLLECTION SUMMARY')
    print('='*70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Total tickers: {stats['total_tickers']}")
    print(f"\nPer-Ticker Statistics:")
    for ticker_stat in stats['per_ticker']:
        ticker = ticker_stat['ticker']
        points = ticker_stat['data_points']
        missing = ticker_stat['missing_days']
        ret = ticker_stat['mean_daily_return']
        vol = ticker_stat['volatility_20d_mean']
        print(f"  {ticker:6} {points:3} points, {missing:2} missing days | "
              f"Ret: {ret:7.4f} Vol: {vol:.4f}")
    
    if stats['correlation_matrix']:
        print(f"\nReturn Correlation Matrix:")
        import json as json_module
        corr = stats['correlation_matrix']
        print(json_module.dumps(corr, indent=2)[:500])  # Print first 500 chars

    print(f"\nMetadata saved to: {metadata_path}")
    print('='*70 + '\n')

    logging.info("Stock data collection complete")


def parse_args():
    p = argparse.ArgumentParser(description='Download historical stock prices and engineer features')
    p.add_argument('--start-date', type=str, default='2025-12-02', help='Start date (YYYY-MM-DD)')
    p.add_argument('--end-date', type=str, default='2025-12-11', help='End date (YYYY-MM-DD)')
    p.add_argument('--tickers', nargs='+', default=TICKERS_DEFAULT, help='Tickers to download')
    p.add_argument('--output', type=str, default=os.path.join('data', 'stocks', 'historical_prices.parquet'), help='Output parquet file')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        main_process(args.start_date, args.end_date, args.output, args.tickers)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
