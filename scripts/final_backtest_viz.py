import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PREDS_FILE = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_DATA_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PLOT = ROOT / 'results' / 'final_performance_chart.png'

def main():
    print(f"Loading predictions from {PREDS_FILE}...")
    preds = pd.read_csv(PREDS_FILE)
    
    print(f"Loading raw price data from {RAW_DATA_FILE}...")
    raw = pd.read_parquet(RAW_DATA_FILE)
    
    # --- COMPUTE 5-DAY FORWARD RETURN ---
    print("Computing 5-day forward returns...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Calculate 5-day forward return: (price_5d_ahead - price_today) / price_today
    raw['close_5d_ahead'] = raw.groupby('Ticker')['Close'].shift(-5)
    raw['target_return_5d'] = (raw['close_5d_ahead'] - raw['Close']) / raw['Close'] * 100
    raw = raw[['Date', 'Ticker', 'target_return_5d']]
    
    # --- CRITICAL FIX: FORCE DATE ALIGNMENT ---
    # Convert dates to datetime objects to ensure they match
    preds['Date'] = pd.to_datetime(preds['Date'])
    raw['Date'] = pd.to_datetime(raw['Date'])
    
    # Strip whitespace from tickers just in case
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()
    
    print("Merging predictions with 5-day returns...")
    # Merge predictions with 5-day returns
    merged = pd.merge(preds, raw, on=['Date', 'Ticker'], how='inner')
    print(f"Merged shape: {merged.shape}")
    
    if len(merged) == 0:
        print("ERROR: Merge resulted in 0 rows. Check your Date formats again.")
        return

    # Handle NaNs in the SMA ratio (assume 0 if missing)
    merged['ratio_sma200'] = merged['ratio_sma200'].fillna(0)

    # --- STRATEGY LOGIC ---
    # 1. Benchmark: Just holding the stock for the 5-day period
    merged['benchmark_return'] = merged['target_return_5d']
    
    # 2. Strategy: Only buy if Probability > 50% AND Price > SMA_200 (Regime Filter)
    # We use prob_ens (ensemble probability) with 0.50 threshold
    merged['signal'] = np.where(
        (merged['prob_ens'] > 0.50) & (merged['ratio_sma200'] > 1.0), 
        1, 0
    )
    
    # Strategy Return = Signal * Benchmark
    merged['strategy_return'] = merged['signal'] * merged['benchmark_return']
    
    total_trades = merged['signal'].sum()
    print(f"Total Trades Executed: {total_trades}")

    # --- PERFORMANCE METRICS ---
    # Group by Date to get the "Portfolio" daily return (average of all tickers active that day)
    # This simulates an equal-weight portfolio of the 7 stocks
    portfolio = merged.groupby('Date')[['benchmark_return', 'strategy_return']].mean()
    
    # Calculate Cumulative Returns
    portfolio['Benchmark_Cum'] = (1 + portfolio['benchmark_return']).cumprod()
    portfolio['Strategy_Cum'] = (1 + portfolio['strategy_return']).cumprod()
    
    # Calculate Sharpe Ratio (Annualized)
    # Assuming 5-day returns, there are ~52 periods per year
    periods_per_year = 52
    
    bench_mean = portfolio['benchmark_return'].mean()
    bench_std = portfolio['benchmark_return'].std()
    bench_sharpe = (bench_mean / bench_std) * np.sqrt(periods_per_year)
    
    strat_mean = portfolio['strategy_return'].mean()
    strat_std = portfolio['strategy_return'].std()
    strat_sharpe = (strat_mean / strat_std) * np.sqrt(periods_per_year)
    
    print("\n" + "="*40)
    print("FINAL BACKTEST RESULTS")
    print("="*40)
    print(f"Benchmark Sharpe Ratio: {bench_sharpe:.4f}")
    print(f"Strategy Sharpe Ratio:  {strat_sharpe:.4f}")
    print(f"Total Trades:           {total_trades}")
    print("="*40)

    # --- PLOTTING ---
    # Use 'ggplot' style which is available in all matplotlib versions
    plt.style.use('ggplot')
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['Benchmark_Cum'], label=f'Benchmark (Sharpe: {bench_sharpe:.2f})', alpha=0.6, linewidth=2)
    plt.plot(portfolio.index, portfolio['Strategy_Cum'], label=f'Regime-Ensemble (Sharpe: {strat_sharpe:.2f})', linewidth=2.5)
    
    plt.title('Cumulative Performance: Regime-Filtered Ensemble vs Buy & Hold (7 Tech Stocks)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return ($1 invested)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"Saved performance chart to {OUT_PLOT}")

if __name__ == '__main__':
    main()
