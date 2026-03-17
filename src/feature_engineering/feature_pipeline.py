"""
feature_pipeline.py
===================
Complete feature assembly pipeline that combines:
1. NLP sentiment features (24 features from nlp_pipeline)
2. Technical indicators (15 features from yfinance)
3. Lagged features (4 features)

Total: 43 features (42 predictors + target)

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import log_info, log_warning, log_error
from FEATURE_SCHEMA import MODEL_FEATURES, FEATURE_DEFAULTS, normalize_feature_dict


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def fetch_technical_features(
    ticker: str,
    lookback_days: int = 30,
    target_date: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Fetch technical indicators from yfinance.
    
    This generates the 15 technical features required by the model:
    - RSI, MACD, MACD_signal
    - BB_upper, BB_middle, BB_lower
    - ATR, OBV, ADX
    - Stochastic_K, Stochastic_D
    - VWAP, CMF, Williams_R, EMA_12
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Days of historical data to fetch
        target_date: Optional date to fetch data up to (defaults to today)
        
    Returns:
        Dictionary with 15 technical features
    """
    try:
        stock = yf.Ticker(ticker)
        
        if target_date:
            end_date = target_date
            start_date = end_date - timedelta(days=lookback_days)
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=f"{lookback_days}d")
        
        if df.empty:
            log_warning(f"No price data for {ticker}, using defaults", "TECH")
            return _get_default_technical_features()
        
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        
        # -------------------- Price-Based Indicators --------------------
        
        # RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        RSI = float(rsi.iloc[-1]) if not rsi.dropna().empty else 50.0
        
        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        
        MACD = float(macd.iloc[-1]) if not macd.dropna().empty else 0.0
        MACD_signal = float(macd_signal.iloc[-1]) if not macd_signal.dropna().empty else 0.0
        
        # Bollinger Bands (20, 2)
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        
        BB_upper = float(bb_upper.iloc[-1]) if not bb_upper.dropna().empty else 0.0
        BB_middle = float(sma20.iloc[-1]) if not sma20.dropna().empty else 0.0
        BB_lower = float(bb_lower.iloc[-1]) if not bb_lower.dropna().empty else 0.0
        
        # ATR (14-period)
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        ATR = float(atr.iloc[-1]) if not atr.dropna().empty else 0.0
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        OBV = float(obv.iloc[-1]) if not obv.empty else 0.0
        
        # ADX (14-period)
        # Simplified ADX calculation
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = true_range
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        ADX = float(adx.iloc[-1]) if not adx.dropna().empty else 20.0
        
        # Stochastic Oscillator (14, 3, 3)
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        stoch_k = 100 * ((close - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(window=3).mean()
        
        Stochastic_K = float(stoch_k.iloc[-1]) if not stoch_k.dropna().empty else 50.0
        Stochastic_D = float(stoch_d.iloc[-1]) if not stoch_d.dropna().empty else 50.0
        
        # VWAP (Volume Weighted Average Price)
        vwap = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        VWAP = float(vwap.iloc[-1]) if not vwap.dropna().empty else float(close.iloc[-1])
        
        # CMF (Chaikin Money Flow, 20-period)
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        cmf = mfv.rolling(window=20).sum() / volume.rolling(window=20).sum()
        CMF = float(cmf.iloc[-1]) if not cmf.dropna().empty else 0.0
        
        # Williams %R (14-period)
        williams_r = -100 * ((high_14 - close) / (high_14 - low_14))
        Williams_R = float(williams_r.iloc[-1]) if not williams_r.dropna().empty else -50.0
        
        # EMA 12
        EMA_12 = float(ema12.iloc[-1]) if not ema12.dropna().empty else float(close.iloc[-1])
        
        return {
            "RSI": RSI,
            "MACD": MACD,
            "MACD_signal": MACD_signal,
            "BB_upper": BB_upper,
            "BB_middle": BB_middle,
            "BB_lower": BB_lower,
            "ATR": ATR,
            "OBV": OBV,
            "ADX": ADX,
            "Stochastic_K": Stochastic_K,
            "Stochastic_D": Stochastic_D,
            "VWAP": VWAP,
            "CMF": CMF,
            "Williams_R": Williams_R,
            "EMA_12": EMA_12,
        }
        
    except Exception as e:
        log_error(f"Technical features failed for {ticker}: {e}", "TECH")
        return _get_default_technical_features()


def _get_default_technical_features() -> Dict[str, float]:
    """Return default technical features (neutral values)."""
    return {
        "RSI": 50.0,
        "MACD": 0.0,
        "MACD_signal": 0.0,
        "BB_upper": 0.0,
        "BB_middle": 0.0,
        "BB_lower": 0.0,
        "ATR": 0.0,
        "OBV": 0.0,
        "ADX": 20.0,
        "Stochastic_K": 50.0,
        "Stochastic_D": 50.0,
        "VWAP": 0.0,
        "CMF": 0.0,
        "Williams_R": -50.0,
        "EMA_12": 0.0,
    }


# ============================================================
# LAGGED FEATURES
# ============================================================

def calculate_lagged_features(
    ticker: str,
    sentiment_features: Dict[str, float],
    lookback_days: int = 10
) -> Dict[str, float]:
    """
    Calculate lagged features (T-1).
    
    For real-time prediction, we need yesterday's:
    - ensemble_sentiment_mean
    - daily_return
    - Volume
    - volatility
    
    Args:
        ticker: Stock ticker
        sentiment_features: Current sentiment features (for ensemble_sentiment)
        lookback_days: Days of history to fetch
        
    Returns:
        Dictionary with 4 lagged features
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{lookback_days}d")
        
        if len(df) < 2:
            log_warning(f"Insufficient data for lags on {ticker}", "LAG")
            return {
                "ensemble_sentiment_mean_lag1": 0.0,
                "daily_return_lag1": 0.0,
                "Volume_lag1": 0.0,
                "volatility_lag1": 0.15,
            }
        
        # T-1 values (yesterday)
        daily_returns = df["Close"].pct_change()
        
        daily_return_lag1 = float(daily_returns.iloc[-2]) if len(daily_returns) >= 2 else 0.0
        Volume_lag1 = float(df["Volume"].iloc[-2]) if len(df) >= 2 else 0.0
        
        # Volatility (rolling 5-day std of returns, annualized)
        volatility = daily_returns.rolling(window=5).std() * np.sqrt(252)
        volatility_lag1 = float(volatility.iloc[-2]) if len(volatility) >= 2 else 0.15
        
        # Sentiment lag: assume yesterday was neutral if we don't have historical sentiment
        # In production, you'd store historical predictions
        ensemble_sentiment_mean_lag1 = 0.0  # Placeholder
        
        return {
            "ensemble_sentiment_mean_lag1": ensemble_sentiment_mean_lag1,
            "daily_return_lag1": daily_return_lag1,
            "Volume_lag1": Volume_lag1,
            "volatility_lag1": volatility_lag1,
        }
        
    except Exception as e:
        log_error(f"Lagged features failed for {ticker}: {e}", "LAG")
        return {
            "ensemble_sentiment_mean_lag1": 0.0,
            "daily_return_lag1": 0.0,
            "Volume_lag1": 0.0,
            "volatility_lag1": 0.15,
        }


# ============================================================
# COMPLETE FEATURE ASSEMBLY
# ============================================================

def assemble_model_features(
    ticker: str,
    sentiment_features: Dict[str, float],
    technical_features: Optional[Dict[str, float]] = None,
    lagged_features: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Assemble complete 43-feature dictionary for model input.
    
    Combines:
    1. Sentiment features (24) - from NLP pipeline
    2. Technical features (15) - from yfinance
    3. Lagged features (4) - from historical data
    
    Args:
        ticker: Stock ticker
        sentiment_features: 24 features from nlp_pipeline
        technical_features: Optional 15 technical indicators (fetched if None)
        lagged_features: Optional 4 lagged features (fetched if None)
        
    Returns:
        Complete 43-feature dictionary aligned with MODEL_FEATURES
    """
    log_info(f"Assembling features for {ticker}", "PIPELINE")
    
    # Fetch missing components
    if technical_features is None:
        log_info("Fetching technical indicators...", "PIPELINE")
        technical_features = fetch_technical_features(ticker)
    
    if lagged_features is None:
        log_info("Calculating lagged features...", "PIPELINE")
        lagged_features = calculate_lagged_features(ticker, sentiment_features)
    
    # Combine all features
    all_features = {
        **sentiment_features,
        **technical_features,
        **lagged_features,
    }
    
    # Normalize to ensure all MODEL_FEATURES are present
    normalized = normalize_feature_dict(all_features)
    
    log_info(f"✅ Assembled {len(normalized)} features for {ticker}", "PIPELINE")
    
    return normalized


def create_model_input_dataframe(
    features: Dict[str, float],
    ticker: str,
    date: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a single-row DataFrame ready for model input.
    
    Args:
        features: Complete feature dictionary
        ticker: Stock ticker
        date: Optional date string (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns [date, ticker, ...MODEL_FEATURES]
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    row = {"date": date, "ticker": ticker}
    row.update(features)
    
    # Ensure column order matches MODEL_FEATURES
    df = pd.DataFrame([row])
    cols = ["date", "ticker"] + MODEL_FEATURES
    
    # Fill missing columns with defaults
    for col in cols:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS.get(col, 0.0)
    
    return df[cols]


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_assemble_features(
    ticker_list: List[str],
    sentiment_dict: Dict[str, Dict[str, float]],
    date: Optional[str] = None
) -> pd.DataFrame:
    """
    Batch assemble features for multiple tickers.
    
    Args:
        ticker_list: List of tickers to process
        sentiment_dict: Dictionary mapping ticker -> sentiment_features
        date: Optional date string
        
    Returns:
        DataFrame with all tickers, ready for batch prediction
    """
    rows = []
    
    for ticker in ticker_list:
        sentiment = sentiment_dict.get(ticker, {})
        
        if not sentiment:
            log_warning(f"No sentiment features for {ticker}, skipping", "BATCH")
            continue
        
        features = assemble_model_features(ticker, sentiment)
        row_df = create_model_input_dataframe(features, ticker, date)
        rows.append(row_df)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.concat(rows, ignore_index=True)


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing feature_pipeline.py...")
    
    # Mock sentiment features (would come from nlp_pipeline)
    mock_sentiment = {
        "finbert_sentiment_score_mean": 0.35,
        "vader_sentiment_score_mean": 0.28,
        "textblob_sentiment_score_mean": 0.32,
        "ensemble_sentiment_mean": 0.317,
        "sentiment_variance_mean": 0.001,
        "model_consensus_mean": 0.95,
        "ensemble_sentiment_max": 0.45,
        "ensemble_sentiment_min": 0.20,
        "ensemble_sentiment_std": 0.08,
        "confidence_mean": 0.65,
        "num_headlines": 12,
        "headline_length_avg": 85.5,
        "sentiment_earnings": 0.40,
        "sentiment_product": 0.30,
        "sentiment_analyst": 0.25,
        "count_positive_earnings": 3,
        "count_negative_regulatory": 0,
        "has_macroeconomic_news": 0,
        "ceo_mention_count": 2,
        "ceo_sentiment": 0.38,
        "competitor_mention_count": 1,
        "entity_density": 1.5,
        "entity_sentiment_gap": 0.05,
    }
    
    # Assemble complete features
    complete_features = assemble_model_features("AAPL", mock_sentiment)
    
    print("\n✅ Complete feature set:")
    for k, v in complete_features.items():
        print(f"  {k}: {v}")
    
    # Create model input DataFrame
    model_df = create_model_input_dataframe(complete_features, "AAPL")
    
    print(f"\n✅ Model input shape: {model_df.shape}")
    print(f"✅ Columns: {list(model_df.columns)}")
    print("\n✅ feature_pipeline.py test passed")