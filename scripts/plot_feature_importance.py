import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
FEATS_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PLOT = ROOT / 'results' / 'feature_importance.png'

def main():
    print("Loading raw price data...")
    raw = pd.read_parquet(FEATS_FILE)
    
    print("Computing features for importance analysis...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Compute technical features
    # SMA indicators
    raw['sma_50'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    raw['sma_200'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    raw['ratio_sma50'] = raw['Close'] / raw['sma_50']
    raw['ratio_sma200'] = raw['Close'] / raw['sma_200']
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    raw['rsi_14'] = raw.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, 14))
    raw['rsi_7'] = raw.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, 7))
    
    # EMA for momentum
    raw['ema_12'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    raw['ema_26'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    raw['macd'] = raw['ema_12'] - raw['ema_26']
    raw['macd_signal'] = raw.groupby('Ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    raw['macd_hist'] = raw['macd'] - raw['macd_signal']
    
    # Bollinger Bands (simplified)
    raw['bb_mid'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    raw['bb_std'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20, min_periods=1).std())
    raw['bb_upper'] = raw['bb_mid'] + (raw['bb_std'] * 2)
    raw['bb_lower'] = raw['bb_mid'] - (raw['bb_std'] * 2)
    raw['bb_width'] = raw['bb_upper'] - raw['bb_lower']
    raw['bb_position'] = (raw['Close'] - raw['bb_lower']) / (raw['bb_width'] + 1e-9)
    
    # Volatility
    raw['volatility_20'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(20, min_periods=1).std() * 100)
    
    # Momentum
    raw['momentum_1m'] = (raw['Close'] - raw.groupby('Ticker')['Close'].shift(20)) / raw.groupby('Ticker')['Close'].shift(20) * 100
    raw['momentum_3m'] = (raw['Close'] - raw.groupby('Ticker')['Close'].shift(63)) / raw.groupby('Ticker')['Close'].shift(63) * 100
    
    # ATR (simplified)
    raw['high_low'] = raw['High'] - raw['Low']
    raw['atr_14'] = raw.groupby('Ticker')['high_low'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    
    # Volume ratio
    raw['volume_sma'] = raw.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    raw['volume_ratio'] = raw['Volume'] / (raw['volume_sma'] + 1e-9)
    
    # 5-day returns
    raw['return_5d'] = raw.groupby('Ticker')['Close'].pct_change(5) * 100
    raw['close_5d_ahead'] = raw.groupby('Ticker')['Close'].shift(-5)
    raw['target_return_5d'] = (raw['close_5d_ahead'] - raw['Close']) / raw['Close'] * 100
    
    # Create binary target (1 if positive return, 0 otherwise)
    raw['binary_label'] = (raw['target_return_5d'] > 0).astype(int)
    
    # --- PREPARE DATA FOR MODEL ---
    print("Preparing data for feature importance analysis...")
    
    # Select feature columns (exclude non-features)
    non_feature_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
                        'close_5d_ahead', 'target_return_5d', 'binary_label', 'sma_50', 'sma_200', 
                        'high_low', 'volume_sma']
    feature_cols = [col for col in raw.columns if col not in non_feature_cols]
    
    X = raw[feature_cols].copy()
    y = raw['binary_label'].copy()
    
    # Handle NaNs
    X = X.fillna(0)
    
    # Remove rows where target is NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {feature_cols}")
    
    # --- TRAIN RANDOM FOREST ---
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, verbose=1)
    rf.fit(X, y)
    
    # --- EXTRACT FEATURE IMPORTANCES ---
    print("Extracting feature importances...")
    importances = rf.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # --- PRINT TOP 10 ---
    print("\n" + "="*70)
    print("TOP 10 PREDICTIVE FEATURES")
    print("="*70)
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"{row['Feature']:20s} {row['Importance']:.6f}")
    print("="*70)
    
    # --- PLOT TOP 15 ---
    print("Creating visualization...")
    top_15 = feature_importance_df.head(15)
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(top_15['Feature'], top_15['Importance'], color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Top 15 Predictive Features (10-Year Ensemble)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUT_PLOT}")
    
    # --- SAVE FULL FEATURE IMPORTANCE ---
    feature_importance_df.to_csv(ROOT / 'results' / 'feature_importance_full.csv', index=False)
    print(f"Saved full feature importance to {ROOT / 'results' / 'feature_importance_full.csv'}")

if __name__ == '__main__':
    main()
