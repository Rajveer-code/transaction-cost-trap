"""
engineer_trend_features.py

Load OHLCV parquet and compute trend-focused features:
- SMA 50/200 with ratio indicators
- Momentum (1m, 3m)
- Volatility (20d rolling)
- 5-day forward target

Grouped by ticker to avoid leakage.

Usage:
    python scripts/engineer_trend_features.py

Outputs:
    data/extended/features_trend_5d.parquet

"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_FILE = ROOT / 'data' / 'extended' / 'features_trend_5d.parquet'


def sma(series, period):
    """Simple Moving Average."""
    return series.rolling(period).mean()


def compute_trend_features(group):
    """Compute trend-focused features for 5-day horizon."""
    g = group.sort_index()
    out = pd.DataFrame(index=g.index)
    out.index.name = 'Date'
    
    # OHLCV
    out['Open'] = g['Open']
    out['High'] = g['High']
    out['Low'] = g['Low']
    out['Close'] = g['Close']
    out['Volume'] = g['Volume']

    # Moving Averages
    sma_50 = sma(g['Close'], 50)
    sma_200 = sma(g['Close'], 200)
    
    out['sma_50'] = sma_50
    out['sma_200'] = sma_200

    # Trend Ratios (Key Features)
    out['ratio_sma50'] = g['Close'] / sma_50
    out['ratio_sma200'] = g['Close'] / sma_200
    out['golden_cross'] = sma_50 / sma_200

    # Volatility: 20-day rolling std dev of daily returns
    daily_returns = g['Close'].pct_change()
    out['volatility_20d'] = daily_returns.rolling(20).std()

    # Momentum
    out['mom_1m'] = g['Close'] / g['Close'].shift(20)
    out['mom_3m'] = g['Close'] / g['Close'].shift(60)

    # Target: 5-day forward return
    out['target_return_5d'] = g['Close'].shift(-5) / g['Close'] - 1
    out['binary_label'] = (out['target_return_5d'] > 0).astype(int)

    return out


def main():
    if not IN_FILE.exists():
        print('Input parquet not found:', IN_FILE)
        return
    
    print('Loading parquet:', IN_FILE)
    df = pd.read_parquet(IN_FILE)

    # Normalize to DataFrame with Date and Ticker columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()

    col_names = [c for c in df.columns]
    date_col = next((c for c in col_names if 'date' == c.lower()), None)
    if date_col is None:
        date_col = next((c for c in col_names if 'date' in c.lower()), None)
    if date_col is None:
        raise RuntimeError('Date column not found in input parquet')
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    ticker_col = next((c for c in col_names if 'ticker' == c.lower()), None)
    if ticker_col is None:
        ticker_col = next((c for c in col_names if 'ticker' in c.lower()), None)
    if ticker_col is None:
        raise RuntimeError('Ticker column not found in input parquet')

    # Set date as index for per-group operations
    df = df.set_index(date_col)
    
    # Ensure sorted by Ticker, Date
    df = df.sort_values([ticker_col, df.index.name or 'Date'], key=lambda x: x if isinstance(x, (pd.RangeIndex, pd.DatetimeIndex)) else x)

    out_frames = []
    for ticker, g in df.groupby(ticker_col, sort=True):
        # Ensure required columns exist and are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in g.columns:
                raise RuntimeError(f'Missing column {col} for ticker {ticker}')
            g[col] = pd.to_numeric(g[col], errors='coerce')

        feats = compute_trend_features(g)
        feats = feats.reset_index()
        feats[ticker_col] = ticker
        out_frames.append(feats)

    df_out = pd.concat(out_frames, axis=0, ignore_index=True)

    # Drop first 200 rows per ticker (SMA_200 warmup + 5-day target horizon)
    df_clean = df_out.groupby(ticker_col, group_keys=False).apply(lambda g: g.iloc[200:]).reset_index(drop=True)

    # Final cleanup: drop rows with any NaNs in critical columns
    critical = ['Open', 'High', 'Low', 'Close', 'Volume', 'target_return_5d', 'binary_label']
    df_clean = df_clean.dropna(subset=critical, how='any')

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUT_FILE, index=False)
    print('Saved features to', OUT_FILE)
    print('Final shape:', df_clean.shape)
    print('Class balance for binary_label:')
    print(df_clean['binary_label'].value_counts(dropna=False))

if __name__ == '__main__':
    main()
