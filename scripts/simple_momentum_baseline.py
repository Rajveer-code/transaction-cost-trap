import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'

def main():
    print("Loading raw price data...")
    raw = pd.read_parquet(RAW_DATA_FILE)
    
    # --- COMPUTE FEATURES ---
    print("Computing 5-day returns...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Past 5-day return (momentum signal)
    raw['return_5d'] = raw.groupby('Ticker')['Close'].pct_change(5) * 100
    
    # Future 5-day return (target)
    raw['close_5d_ahead'] = raw.groupby('Ticker')['Close'].shift(-5)
    raw['target_return_5d'] = (raw['close_5d_ahead'] - raw['Close']) / raw['Close'] * 100
    
    # Drop rows with NaN returns (first 5 rows per ticker and last 5 rows per ticker)
    raw = raw.dropna(subset=['return_5d', 'target_return_5d'])
    
    print(f"Data shape after dropping NaNs: {raw.shape}")
    
    # --- SIMPLE MOMENTUM STRATEGY ---
    print("\n" + "="*70)
    print("SIMPLE MOMENTUM BASELINE")
    print("="*70)
    
    # Strategy: If past 5-day return > 0, Signal = 1 (Buy). Else 0 (Sell).
    raw['momentum_signal'] = (raw['return_5d'] > 0).astype(int)
    
    # Calculate win rate for momentum signals (where signal == 1)
    momentum_buy_signals = raw[raw['momentum_signal'] == 1]
    if len(momentum_buy_signals) > 0:
        # Binary target: 1 if target_return_5d > 0, else 0
        momentum_target = (momentum_buy_signals['target_return_5d'] > 0).astype(int)
        momentum_win_rate = momentum_target.mean() * 100
        momentum_n_signals = len(momentum_buy_signals)
    else:
        momentum_win_rate = 0.0
        momentum_n_signals = 0
    
    print(f"Simple Momentum Signals (return_5d > 0): {momentum_n_signals}")
    print(f"Simple Momentum Win Rate: {momentum_win_rate:.2f}%")
    
    # --- BUY & HOLD BASELINE ---
    print("\n" + "="*70)
    print("BUY & HOLD BASELINE")
    print("="*70)
    
    # Baseline: what % of all 5-day periods have positive returns?
    all_days = raw.shape[0]
    positive_returns = (raw['target_return_5d'] > 0).sum()
    buy_hold_win_rate = (positive_returns / all_days) * 100
    
    print(f"Buy & Hold: {positive_returns} out of {all_days} days had positive 5-day returns")
    print(f"Buy & Hold Baseline Win Rate: {buy_hold_win_rate:.2f}%")
    
    # --- ML ENSEMBLE COMPARISON ---
    print("\n" + "="*70)
    print("ML ENSEMBLE VS BASELINES")
    print("="*70)
    
    ml_ensemble_win_rate = 57.40  # From predictions_ensemble.csv precision
    
    print(f"ML Ensemble Win Rate: {ml_ensemble_win_rate:.2f}%")
    print(f"Simple Momentum Win Rate: {momentum_win_rate:.2f}%")
    print(f"Buy & Hold Baseline Win Rate: {buy_hold_win_rate:.2f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    ml_vs_momentum = ml_ensemble_win_rate - momentum_win_rate
    ml_vs_buyhold = ml_ensemble_win_rate - buy_hold_win_rate
    
    if ml_vs_momentum > 0:
        print(f"[+] ML Ensemble BEATS Simple Momentum by {ml_vs_momentum:.2f} percentage points")
    else:
        print(f"[-] ML Ensemble UNDERPERFORMS Simple Momentum by {abs(ml_vs_momentum):.2f} percentage points")
    
    if ml_vs_buyhold > 0:
        print(f"[+] ML Ensemble BEATS Buy & Hold Baseline by {ml_vs_buyhold:.2f} percentage points")
    else:
        print(f"[-] ML Ensemble UNDERPERFORMS Buy & Hold Baseline by {abs(ml_vs_buyhold):.2f} percentage points")
    
    print("="*70)

if __name__ == '__main__':
    main()
