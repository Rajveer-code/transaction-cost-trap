import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
FEATS_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PLOT = ROOT / 'results' / 'learning_curves.png'

def main():
    print("Loading raw price data...")
    raw = pd.read_parquet(FEATS_FILE)
    
    print("Computing features...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Compute technical features
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
    
    # Create binary target
    raw['binary_label'] = (raw['target_return_5d'] > 0).astype(int)
    
    # --- PREPARE DATA ---
    print("Preparing data for learning curves...")
    
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
    
    # --- SETUP MODEL AND CV ---
    print("Setting up model and cross-validation...")
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    cv = TimeSeriesSplit(n_splits=5)
    
    # --- COMPUTE LEARNING CURVE ---
    print("Computing learning curve (this may take a minute)...")
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, 
        cv=cv,
        train_sizes=train_sizes,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print("\n" + "="*70)
    print("LEARNING CURVE RESULTS")
    print("="*70)
    print(f"Training sizes: {train_sizes_abs}")
    print(f"Train scores (mean): {train_mean}")
    print(f"Val scores (mean): {val_mean}")
    print("="*70)
    
    # --- VISUALIZATION ---
    print("Creating visualization...")
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training curve with std shading
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', label='Training Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='#1f77b4')
    
    # Plot validation curve with std shading
    ax.plot(train_sizes_abs, val_mean, 'o-', color='#ff7f0e', label='Validation Score', linewidth=2, markersize=6)
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='#ff7f0e')
    
    ax.set_xlabel('Training Samples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curve: Generalization over 10 Years', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.45, 0.65])
    
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUT_PLOT}")
    
    # --- ANALYSIS ---
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    gap_initial = train_mean[0] - val_mean[0]
    gap_final = train_mean[-1] - val_mean[-1]
    
    print(f"Initial Generalization Gap (10% data): {gap_initial:.4f}")
    print(f"Final Generalization Gap (100% data): {gap_final:.4f}")
    print(f"Gap Reduction: {gap_initial - gap_final:.4f}")
    
    if gap_final > 0.02:
        print("\n-> Model shows signs of OVERFITTING (gap > 2%)")
    elif gap_final < 0.01:
        print("\n-> Model shows good GENERALIZATION (gap < 1%)")
    else:
        print("\n-> Model shows MODERATE GENERALIZATION (gap ~1-2%)")
    
    print("="*70)

if __name__ == '__main__':
    main()
