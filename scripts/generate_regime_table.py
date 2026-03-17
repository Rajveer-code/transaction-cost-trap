import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PREDS_FILE = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_DATA_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_CSV = ROOT / 'results' / 'regime_stats.csv'

def main():
    print("Loading predictions...")
    preds = pd.read_csv(PREDS_FILE)
    
    print("Loading raw price data...")
    raw = pd.read_parquet(RAW_DATA_FILE)
    
    # --- COMPUTE FEATURES WE NEED ---
    print("Computing 5-day returns and RSI...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # 5-day forward return
    raw['close_5d_ahead'] = raw.groupby('Ticker')['Close'].shift(-5)
    raw['target_return_5d'] = (raw['close_5d_ahead'] - raw['Close']) / raw['Close'] * 100
    
    # SMA 50 and 200 (simplified for speed - using rolling mean)
    raw['sma_50'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
    raw['sma_200'] = raw.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    raw['ratio_sma200'] = raw['Close'] / raw['sma_200']
    
    # RSI calculation (14-period)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    raw['rsi_14'] = raw.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, 14))
    
    # Keep only necessary columns
    raw = raw[['Date', 'Ticker', 'target_return_5d', 'ratio_sma200', 'rsi_14']]
    
    # --- MERGE WITH PREDICTIONS ---
    print("Merging predictions with computed features...")
    preds['Date'] = pd.to_datetime(preds['Date'])
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()
    
    merged = pd.merge(preds, raw, on=['Date', 'Ticker'], how='inner')
    print(f"Merged shape: {merged.shape}")
    print(f"Merged columns: {merged.columns.tolist()}")
    
    if len(merged) == 0:
        print("ERROR: Merge resulted in 0 rows.")
        return
    
    # --- HANDLE NaNs ---
    print("Handling NaN values and consolidating columns...")
    
    # Use ratio_sma200_y (from raw data, more freshly computed) if available, else ratio_sma200_x
    if 'ratio_sma200_y' in merged.columns:
        merged['ratio_sma200'] = merged['ratio_sma200_y'].fillna(merged['ratio_sma200_x'].fillna(1.0))
    else:
        merged['ratio_sma200'] = merged['ratio_sma200'].fillna(1.0)
    
    merged['rsi_14'] = merged['rsi_14'].fillna(50.0)  # Default to neutral RSI
    
    # --- DEFINE REGIMES ---
    print("Defining regimes...")
    merged['Bull'] = merged['ratio_sma200'] > 1.0
    merged['Bear'] = merged['ratio_sma200'] <= 1.0
    merged['Oversold'] = merged['rsi_14'] < 30
    merged['Overbought'] = merged['rsi_14'] > 70
    
    # --- CALCULATE METRICS PER REGIME ---
    print("Calculating metrics per regime...")
    
    regimes_to_analyze = ['Bull', 'Bear']
    results = []
    
    for regime_name in regimes_to_analyze:
        regime_data = merged[merged[regime_name]]
        
        if len(regime_data) == 0:
            continue
        
        n_days = len(regime_data)
        
        # Baseline: win rate of y_true (actual outcomes)
        baseline_win_pct = regime_data['y_true'].mean() * 100
        
        # Model signals: where y_pred == 1
        model_signals = regime_data[regime_data['y_pred'] == 1]
        
        if len(model_signals) > 0:
            model_win_pct = model_signals['y_true'].mean() * 100
        else:
            model_win_pct = 0.0
        
        improvement = model_win_pct - baseline_win_pct
        
        results.append({
            'Regime': regime_name,
            'N_Days': n_days,
            'Baseline_Win%': f"{baseline_win_pct:.2f}",
            'Model_Win%': f"{model_win_pct:.2f}",
            'Improvement': f"{improvement:.2f}"
        })
    
    # Overall regime (all data)
    n_days_all = len(merged)
    baseline_win_all = merged['y_true'].mean() * 100
    model_signals_all = merged[merged['y_pred'] == 1]
    if len(model_signals_all) > 0:
        model_win_all = model_signals_all['y_true'].mean() * 100
    else:
        model_win_all = 0.0
    improvement_all = model_win_all - baseline_win_all
    
    results.append({
        'Regime': 'Overall',
        'N_Days': n_days_all,
        'Baseline_Win%': f"{baseline_win_all:.2f}",
        'Model_Win%': f"{model_win_all:.2f}",
        'Improvement': f"{improvement_all:.2f}"
    })
    
    # --- OUTPUT ---
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("REGIME ANALYSIS TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Save to CSV
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")

if __name__ == '__main__':
    main()
