"""
Script: generate_extended_dataset.py
Creates a 5-year extended dataset with technical features and targets.

Usage:
    python scripts/generate_extended_dataset.py --run-full

By default it performs a quick local sanity check (one ticker, short date range).

Dependencies: yfinance, pandas, numpy

"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'extended'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'all_features_5year.parquet'

TICKERS = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA']
START = '2020-01-01'
END = '2025-04-22'

# Feature helpers

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period-1), adjust=False).mean()
    ma_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series, period=20, n_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def atr(df, period=14):
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(df):
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    vol = pd.Series(np.asarray(df['Volume']).ravel(), index=df.index)
    sign = np.sign(close.diff()).fillna(0)
    return (sign * vol).cumsum()


def cmf(df, period=20):
    # Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    vol = pd.Series(np.asarray(df['Volume']).ravel(), index=df.index)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * vol
    return mfv.rolling(period).sum() / vol.rolling(period).sum()


def vwap(df):
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    vol = pd.Series(np.asarray(df['Volume']).ravel(), index=df.index)
    tp = (high + low + close) / 3
    return (tp * vol).cumsum() / vol.cumsum()


def adx(df, period=14):
    # Average Directional Index implementation
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / atr_)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / atr_)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    adx = dx.rolling(period).mean()
    return adx


def stochastic(df, k_period=14, d_period=3, smooth_k=3):
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * ((close - low_min) / (high_max - low_min))
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(d_period).mean()
    return k_smooth, d


def williams_r(df, period=14):
    high = pd.Series(np.asarray(df['High']).ravel(), index=df.index)
    low = pd.Series(np.asarray(df['Low']).ravel(), index=df.index)
    close = pd.Series(np.asarray(df['Close']).ravel(), index=df.index)
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    return -100 * ((high_max - close) / (high_max - low_min))


def compute_features_for_ticker(df):
    # expects df with columns: DateTime index or 'Date', 'Open','High','Low','Close','Volume'
    df = df.copy()
    # Normalize column names to Title case safely
    new_cols = {}
    for c in df.columns:
        try:
            new_cols[c] = str(c).title()
        except Exception:
            new_cols[c] = c
    df = df.rename(columns=new_cols)
    # Ensure price/volume columns are 1-D Series (handle cases where columns are DataFrames or multi-index slices)
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    # Coerce price/volume columns to numeric 1-D series (flatten if needed)
    def flatten_to_series(df, col):
        if col not in df.columns:
            return None
        v = df[col]
        # If it's a DataFrame (multi-column), take first column
        if isinstance(v, pd.DataFrame):
            s = v.iloc[:, 0]
        else:
            # try to coerce to 1-D
            arr = np.asarray(v)
            if arr.ndim == 1:
                s = pd.Series(arr, index=df.index)
            else:
                # if 2-D but has same number of rows, take first column
                try:
                    s = pd.Series(arr[:, 0], index=df.index)
                except Exception:
                    s = pd.Series(arr.ravel()[:len(df.index)], index=df.index)
        return pd.to_numeric(s, errors='coerce')

    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = flatten_to_series(df, col)
    df = df.sort_index()
    # Price/Volume base
    df['volume_change'] = df['Volume'].pct_change()
    # RSI
    df['rsi_14'] = rsi(df['Close'], 14)
    # MACD
    macd_line, macd_signal, macd_hist = macd(df['Close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    # Bollinger
    bb_u, bb_m, bb_l = bollinger(df['Close'])
    df['bb_upper'] = bb_u
    df['bb_mid'] = bb_m
    df['bb_lower'] = bb_l
    # ATR
    df['atr_14'] = atr(df, 14)
    # OBV
    df['obv'] = obv(df)
    # CMF
    df['cmf_20'] = cmf(df, 20)
    # VWAP
    df['vwap'] = vwap(df)
    # EMAs
    df['ema_12'] = ema(df['Close'], 12)
    df['ema_26'] = ema(df['Close'], 26)
    # SMAs
    df['sma_20'] = sma(df['Close'], 20)
    df['sma_50'] = sma(df['Close'], 50)
    # ADX
    df['adx_14'] = adx(df, 14)
    # Stochastic
    st_k, st_d = stochastic(df, 14, 3, 3)
    df['stoch_k'] = st_k
    df['stoch_d'] = st_d
    # Williams %R
    df['williams_r_14'] = williams_r(df, 14)
    # Rolling returns
    for p in [1,2,3,5,10,20,30,40,50]:
        df[f'return_{p}d'] = df['Close'].pct_change(periods=p)

    return df


def build_dataset(tickers=TICKERS, start=START, end=END, run_full=False):
    frames = []
    for t in tickers:
        print('Downloading', t)
        if run_full:
            raw = yf.download(t, start=start, end=end, progress=False)
        else:
            # quick sample for sanity check: 2020-01-01 -> 2020-03-01
            raw = yf.download(t, start='2020-01-01', end='2020-03-01', progress=False)
        if raw.empty:
            print('Warning: empty data for', t)
            continue
        raw.index = pd.to_datetime(raw.index)
        # forward-fill price gaps up to 5 days (only on columns that exist)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in raw.columns:
                raw[col] = raw[col].ffill(limit=5)
        # compute features
        feat = compute_features_for_ticker(raw)
        # ensure column names are strings and ticker column exists
        feat.columns = [str(c) for c in feat.columns]
        feat['ticker'] = t
        # compute next-day return per ticker before concatenation
        if 'Close' in feat.columns:
            try:
                feat['next_day_return'] = feat['Close'].shift(-1) / feat['Close'] - 1
            except Exception:
                # fallback: coerce Close to numeric then compute
                close_s = pd.to_numeric(pd.Series(np.asarray(feat['Close']).ravel(), index=feat.index), errors='coerce')
                feat['next_day_return'] = close_s.shift(-1) / close_s - 1
        feat = feat.reset_index().rename(columns={'index':'date'})
        frames.append(feat)

    if not frames:
        raise RuntimeError('No data downloaded')
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    # Normalize column names to strings to avoid tuple/multiindex labels from yfinance
    df_all.columns = [str(c) for c in df_all.columns]

    # Normalize and prepare date/ticker columns robustly
    cols = list(df_all.columns)
    ticker_col = next((c for c in cols if 'ticker' == str(c).lower()), None)
    # find any column that looks like a date column
    date_col = next((c for c in cols if 'date' in str(c).lower()), None)
    if date_col is None:
        # try fallback
        date_col = 'date'
    # rename to standard names
    if ticker_col is not None and ticker_col != 'ticker':
        df_all = df_all.rename(columns={ticker_col: 'ticker'})
    if date_col is not None and date_col != 'date':
        try:
            df_all = df_all.rename(columns={date_col: 'date'})
        except Exception:
            pass
    # ensure date column exists
    if 'date' in df_all.columns:
        df_all['date'] = pd.to_datetime(df_all['date']).dt.date
    # sort if ticker is present
    if 'ticker' in df_all.columns:
        if 'date' in df_all.columns:
            df_all = df_all.sort_values(['ticker','date'])
        else:
            df_all = df_all.sort_values(['ticker'])
    else:
        if 'date' in df_all.columns:
            df_all = df_all.sort_values(['date'])

    # If next_day_return wasn't set per-frame (environment edge-cases), compute robustly per-ticker
    if 'next_day_return' not in df_all.columns:
        def scalar_val(v):
            try:
                a = np.asarray(v)
                if getattr(a, 'size', None) is None:
                    return float(v)
                if a.size == 0:
                    return np.nan
                return a.ravel()[0]
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return np.nan

        df_all['next_day_return'] = np.nan
        # robustly find ticker column
        ticker_col = next((c for c in df_all.columns if 'ticker' in str(c).lower()), None)
        if ticker_col is None:
            raise RuntimeError(f'No ticker column found. Columns: {list(df_all.columns)}')
        for t, g in df_all.groupby(ticker_col):
            idx = g.index
            # find close-like column name
            close_col = next((c for c in g.columns if 'close' in str(c).lower()), None)
            if close_col is None:
                raise RuntimeError(f'No Close-like column found for ticker {t}; columns: {list(g.columns)}')
            # extract Close scalars for each row
            close_vals = [scalar_val(v) for v in g[close_col]]
            close_s = pd.to_numeric(pd.Series(close_vals, index=idx), errors='coerce')
            df_all.loc[idx, 'next_day_return'] = close_s.shift(-1) / close_s - 1

    # Compute cross-sectional median per date and binary label.
    median_by_date = df_all.groupby('date')['next_day_return'].transform('median')
    df_all['binary_label'] = (df_all['next_day_return'] > median_by_date).astype(int)

    # Handle missing data
    feature_cols = [c for c in df_all.columns if c.lower() not in ('date','ticker','open','high','low','close','adj close','volume','next_day_return','binary_label')]
    # drop rows with >10% missing among features
    max_missing = int(np.floor(len(feature_cols) * 0.1))
    df_all['n_missing'] = df_all[feature_cols].isna().sum(axis=1)
    df_all = df_all[df_all['n_missing'] <= max_missing].copy()
    df_all = df_all.drop(columns=['n_missing'])
    df_all = df_all.dropna()

    # final column selection: date,ticker + 34 features + binary_label
    # ensure we have ~34 features: try to pick from computed set
    desired_features = []
    # price/volume
    desired_features += ['Open','High','Low','Close','Volume','volume_change']
    # technical
    desired_features += ['rsi_14','macd','macd_signal','macd_hist','bb_upper','bb_mid','bb_lower','atr_14','obv','cmf_20','vwap','ema_12','ema_26','sma_20','sma_50','adx_14','stoch_k','stoch_d','williams_r_14']
    # returns
    desired_features += [f'return_{p}d' for p in [1,2,3,5,10,20,30,40,50]]

    # keep only existing columns in that order
    cols_keep = ['date','ticker'] + [c for c in desired_features if c in df_all.columns] + ['binary_label']
    df_final = df_all[cols_keep].copy()

    return df_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-full', action='store_true', help='Download full 5-year data (may take ~10 minutes)')
    args = parser.parse_args()
    df_out = build_dataset(run_full=args.run_full)
    if args.run_full:
        df_out.to_parquet(OUT_FILE, index=False)
        print('Saved full dataset ->', OUT_FILE)
    else:
        # quick sanity output
        print('Sanity check dataset shape:', df_out.shape)
        print(df_out.head().to_markdown())
