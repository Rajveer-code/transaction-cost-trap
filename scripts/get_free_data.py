"""
get_free_data.py

Provides `get_free_data` which downloads OHLCV history for a list of tickers
using yfinance with `auto_adjust=True` and a polite pause between requests.

Run as a script to perform a quick download and print row counts.
"""
from typing import List, Optional
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

DEFAULT_TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']


def get_free_data(tickers: List[str] = DEFAULT_TICKERS,
                  start: str = '2015-01-01',
                  end: str = '2025-04-22',
                  pause: float = 1.0,
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """Download OHLCV history for the given tickers using yfinance.

    - Uses `auto_adjust=True` to adjust for splits/dividends.
    - Sleeps `pause` seconds between downloads to be polite.

    Returns a single DataFrame with a MultiIndex (Date, Ticker). If you prefer
    a flat DataFrame with a `Ticker` column, call `.reset_index()` on the result.
    """
    # Download all tickers in a single call so yfinance returns a MultiIndex columns
    print(f'Downloading tickers: {tickers} ...')
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        print(f'  Error downloading tickers: {e}')
        return pd.DataFrame()

    if raw is None or raw.empty:
        print('No data returned from yfinance.')
        return pd.DataFrame()

    # If columns are a MultiIndex like (PriceField, Ticker), stack the Ticker level into the index
    df_long: pd.DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            stacked = raw.stack(level='Ticker', future_stack=True)
        except Exception:
            # fallback: stack second level
            stacked = raw.stack(level=1)

        df_long = stacked.reset_index()
    else:
        # Single-level columns (e.g., single ticker). Reset index and add Ticker column.
        df_long = raw.reset_index()
        if isinstance(tickers, list) and len(tickers) == 1:
            df_long['Ticker'] = tickers[0]
        else:
            # Try to infer a Ticker column if present
            tcol = next((c for c in df_long.columns if str(c).lower() == 'ticker' or 'ticker' in str(c).lower()), None)
            if tcol is None:
                # If we can't infer, return the DataFrame as-is
                print('Could not detect Ticker column in single-level download. Returning raw frame.')
                return df_long

    # Normalize column names to strings
    df_long.columns = [str(c) for c in df_long.columns]

    # Ensure 'Date' and 'Ticker' columns exist and are named exactly
    date_col = next((c for c in df_long.columns if c.lower() == 'date' or 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError(f'Could not find date column after stacking. Columns: {list(df_long.columns)}')
    if date_col != 'Date':
        df_long = df_long.rename(columns={date_col: 'Date'})

    tcol = next((c for c in df_long.columns if c.lower() == 'ticker' or 'ticker' in c.lower()), None)
    if tcol is None:
        raise RuntimeError(f'Could not find Ticker column after stacking. Columns: {list(df_long.columns)}')
    if tcol != 'Ticker':
        df_long = df_long.rename(columns={tcol: 'Ticker'})

    # Desired final columns
    desired = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']

    # There may be an 'Adj Close' column from yfinance; drop it if present
    if 'Adj Close' in df_long.columns:
        df_long = df_long.drop(columns=['Adj Close'])

    # Ensure all desired value columns exist; if missing, create with NaN
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df_long.columns:
            df_long[col] = pd.NA

    # Keep only desired columns in the specified order (create Date/Ticker first)
    df_long = df_long[[c for c in desired if c in df_long.columns or c in ('Date', 'Ticker')] + [c for c in desired if c in df_long.columns and c not in ('Date', 'Ticker')]]
    # Reindex to exact desired order (adds any missing columns filled with NaN)
    for c in desired:
        if c not in df_long.columns:
            df_long[c] = pd.NA
    df_long = df_long[desired]

    # Collapse duplicate column names if present (stacking can sometimes create duplicates)
    cols = list(df_long.columns)
    if len(cols) != len(set(cols)):
        result = df_long[['Date', 'Ticker']].copy()
        for name in ['Open', 'High', 'Low', 'Close', 'Volume']:
            # find first column position matching the name and take that series
            idxs = [i for i, c in enumerate(df_long.columns) if c == name]
            if idxs:
                result[name] = df_long.iloc[:, idxs[0]]
            else:
                result[name] = pd.NA
        df_long = result

    # Convert Date to datetime
    df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')

    # Drop rows that contain all NaNs across data columns (Open,High,Low,Close,Volume)
    df_long = df_long.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='all')

    # Ensure proper dtypes for numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        try:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
        except Exception:
            pass

    # Save to parquet if requested
    if save_path is not None:
        outpath = Path(save_path)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_parquet(outpath)
        print(f'Saved parquet -> {outpath}')

    print('Download complete. Shape:', df_long.shape)
    return df_long


if __name__ == '__main__':
    df = get_free_data()
    print(df.head())
    print('\nTotal rows (MultiIndex):', len(df))
    print('Shape:', df.shape)
