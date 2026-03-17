"""
engineer_features.py

Load a saved OHLCV parquet and compute technical indicators grouped by ticker.

Usage:
    python scripts/engineer_features.py

Outputs:
    data/extended/features_final_10y.parquet

"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_FILE = ROOT / 'data' / 'extended' / 'features_final_10y.parquet'

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
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
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def obv(df):
    sign = np.sign(df['Close'].diff()).fillna(0)
    return (sign * df['Volume']).cumsum()

def cmf(df, period=20):
    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * vol
    return mfv.rolling(period).sum() / vol.rolling(period).sum()

def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

def adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
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
    return dx.rolling(period).mean()

def stochastic(df, k_period=14, d_period=3, smooth_k=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    k_smooth = k.rolling(smooth_k).mean()
    d = k_smooth.rolling(d_period).mean()
    return k_smooth, d

def williams_r(df, period=14):
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    return -100 * ((high_max - df['Close']) / (high_max - low_min))

def compute_features(group):
    g = group.sort_index()
    out = pd.DataFrame(index=g.index)
    out.index.name = 'Date'
    out['Open'] = g['Open']
    out['High'] = g['High']
    out['Low'] = g['Low']
    out['Close'] = g['Close']
    out['Volume'] = g['Volume']
    # volume change
    out['volume_change'] = g['Volume'].pct_change()

    # Momentum
    out['rsi_14'] = rsi(g['Close'], 14)
    out['macd'], out['macd_signal'], out['macd_hist'] = macd(g['Close'])
    st_k, st_d = stochastic(g, 14, 3, 3)
    out['stoch_k'] = st_k
    out['stoch_d'] = st_d
    out['williams_r_14'] = williams_r(g, 14)

    # Volatility
    out['bb_upper'], out['bb_mid'], out['bb_lower'] = bollinger(g['Close'], 20, 2)
    out['atr_14'] = atr(g, 14)

    # Trend
    out['ema_12'] = ema(g['Close'], 12)
    out['ema_26'] = ema(g['Close'], 26)
    out['sma_20'] = sma(g['Close'], 20)
    out['sma_50'] = sma(g['Close'], 50)
    out['adx_14'] = adx(g, 14)

    # Volume indicators
    out['obv'] = obv(g)
    out['cmf_20'] = cmf(g, 20)
    out['vwap'] = vwap(g)

    # Rolling returns
    for p in [1,2,3,5,10,20,50]:
        out[f'return_{p}d'] = g['Close'].pct_change(periods=p)

    # Targets: next day return and binary label (shift -1)
    out['next_day_return'] = g['Close'].shift(-1) / g['Close'] - 1
    out['binary_label'] = (out['next_day_return'] > 0).astype(int)

    return out

def main():
    if not IN_FILE.exists():
        print('Input parquet not found:', IN_FILE)
        return
    print('Loading parquet:', IN_FILE)
    df = pd.read_parquet(IN_FILE)

    # Expect df indexed by (Date, Ticker) or having columns 'Date' and 'Ticker'
    # Normalize to DataFrame with DatetimeIndex and columns including 'Ticker'
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()

    # ensure column names
    col_names = [c for c in df.columns]
    date_col = next((c for c in col_names if 'date' == str(c).lower()), None)
    if date_col is None:
        date_col = next((c for c in col_names if 'date' in str(c).lower()), None)
    if date_col is None:
        raise RuntimeError('Date column not found in input parquet')
    df[date_col] = pd.to_datetime(df[date_col])
    # find ticker column
    ticker_col = next((c for c in col_names if 'ticker' == str(c).lower()), None)
    if ticker_col is None:
        ticker_col = next((c for c in col_names if 'ticker' in str(c).lower()), None)
    if ticker_col is None:
        raise RuntimeError('Ticker column not found in input parquet')

    df = df.set_index(date_col)

    # Ensure data sorted by Ticker, Date for rolling calculations
    df = df.sort_values([ticker_col, date_col])

    # group by ticker and compute features
    out_frames = []
    for ticker, g in df.groupby(ticker_col, sort=True):
        # ensure required columns exist and are numeric
        for col in ['Open','High','Low','Close','Volume']:
            if col not in g.columns:
                raise RuntimeError(f'Missing column {col} for ticker {ticker}')
            g[col] = pd.to_numeric(g[col], errors='coerce')

        feats = compute_features(g)
        # reset_index preserves the Date column name because we set index.name above
        feats = feats.reset_index()
        feats[ticker_col] = ticker
        out_frames.append(feats)

    df_out = pd.concat(out_frames, axis=0, ignore_index=True)

    # Drop the first 50 rows and the last row of each ticker
    # (first rows contain NaNs from rolling windows; last row's target is NaN)
    df_clean = df_out.groupby(ticker_col, group_keys=False).apply(lambda g: g.iloc[50:-1]).reset_index(drop=True)

    # Final cleanup: drop rows with any remaining NaNs in critical columns
    critical = ['Open','High','Low','Close','Volume','next_day_return','binary_label']
    df_clean = df_clean.dropna(subset=critical, how='any')

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUT_FILE, index=False)
    print('Saved features to', OUT_FILE)
    print('Final shape:', df_clean.shape)
    print('Class balance for binary_label:\n', df_clean['binary_label'].value_counts(dropna=False))

if __name__ == '__main__':
    main()
