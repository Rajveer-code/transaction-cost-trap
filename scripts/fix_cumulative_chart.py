import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
PREDS_FILE = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_DATA_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PLOT = ROOT / 'results' / 'fixed_cumulative_chart.png'

def main():
    print("Loading predictions...")
    preds = pd.read_csv(PREDS_FILE)
    
    print("Loading raw price data...")
    raw = pd.read_parquet(RAW_DATA_FILE)
    
    # --- COMPUTE DAILY RETURNS (NOT 5-DAY FORWARD) ---
    print("Computing daily returns...")
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Calculate daily return: today's return vs previous day
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw = raw[['Date', 'Ticker', 'daily_return']]
    
    # --- MERGE WITH PREDICTIONS ---
    print("Merging predictions with returns...")
    preds['Date'] = pd.to_datetime(preds['Date'])
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()
    
    merged = pd.merge(preds, raw, on=['Date', 'Ticker'], how='inner')
    print(f"Merged shape: {merged.shape}")
    
    if len(merged) == 0:
        print("ERROR: Merge resulted in 0 rows.")
        return
    
    # --- RECALCULATE STRATEGY LOGIC ---
    print("Recalculating strategy logic...")
    
    # Benchmark return = daily return
    merged['Benchmark_Return'] = merged['daily_return']
    
    # Strategy: Signal = 1 where (y_proba > 0.5) & (ratio_sma200 > 1.0)
    merged['ratio_sma200'] = merged['ratio_sma200'].fillna(0)
    merged['Signal'] = ((merged['prob_ens'] > 0.50) & (merged['ratio_sma200'] > 1.0)).astype(int)
    
    # Strategy return = daily return * signal (0 if no signal, daily return if signal)
    merged['Strategy_Return'] = merged['Signal'] * merged['Benchmark_Return']
    
    print(f"Total signals: {merged['Signal'].sum()}")
    
    # --- CREATE PORTFOLIO (DAILY AGGREGATION) ---
    print("Creating daily portfolio...")
    
    # Aggregate returns by date (across all 7 tickers)
    portfolio = merged.groupby('Date').agg({
        'Benchmark_Return': 'mean',      # Average daily return across tickers
        'Strategy_Return': 'mean',       # Average daily return when in signal
        'Signal': 'sum'
    }).reset_index()
    
    portfolio = portfolio.sort_values('Date').reset_index(drop=True)
    
    # Now compound wealth from daily returns
    portfolio['Benchmark_Wealth'] = (1 + portfolio['Benchmark_Return']).cumprod()
    portfolio['Strategy_Wealth'] = (1 + portfolio['Strategy_Return']).cumprod()
    
    print(f"Portfolio shape: {portfolio.shape}")
    print(f"Date range: {portfolio['Date'].min()} to {portfolio['Date'].max()}")
    
    # --- DIAGNOSTIC CHECK: MIN/MAX OF DAILY RETURNS ---
    print("\n" + "="*70)
    print("DIAGNOSTIC CHECK: Daily Wealth Values")
    print("="*70)
    print(f"Benchmark Wealth - Min: {portfolio['Benchmark_Wealth'].min():.6f}, Max: {portfolio['Benchmark_Wealth'].max():.6f}")
    print(f"Strategy Wealth - Min: {portfolio['Strategy_Wealth'].min():.6f}, Max: {portfolio['Strategy_Wealth'].max():.6f}")
    print(f"Signal Count - Min: {portfolio['Signal'].min():.0f}, Max: {portfolio['Signal'].max():.0f}")
    print("="*70)
    
    # --- CRITICAL FIX: COMPOUND WEALTH CALCULATION ---
    print("\nComputing portfolio wealth (already compounded at ticker level)...")
    
    # Portfolio wealth is already calculated above as mean of ticker wealths
    # Final values
    final_bench_wealth = portfolio['Benchmark_Wealth'].iloc[-1]
    final_strat_wealth = portfolio['Strategy_Wealth'].iloc[-1]
    
    print(f"\n" + "="*70)
    print("FINAL PORTFOLIO VALUES ($1 Invested)")
    print("="*70)
    print(f"Benchmark Final Wealth: ${final_bench_wealth:.4f}")
    print(f"Strategy Final Wealth: ${final_strat_wealth:.4f}")
    print(f"Difference: ${final_strat_wealth - final_bench_wealth:.4f}")
    
    if final_strat_wealth > final_bench_wealth:
        outperformance = ((final_strat_wealth / final_bench_wealth) - 1) * 100
        print(f"Strategy OUTPERFORMS by {outperformance:.2f}%")
    else:
        underperformance = ((final_bench_wealth / final_strat_wealth) - 1) * 100
        print(f"Strategy UNDERPERFORMS by {underperformance:.2f}%")
    
    print("="*70)
    
    # --- VISUALIZATION ---
    print("\nCreating visualization...")

    # Defensive check: ensure we're plotting cumulative wealth (not raw returns)
    # If values look like returns (e.g., many values between -1 and 1), reconstruct cumprod.
    bench = portfolio['Benchmark_Wealth'].copy()
    strat = portfolio['Strategy_Wealth'].copy()

    def looks_like_returns(s):
        # returns typically have min < 1 and max <= 1 (for raw pct returns between -1 and 1)
        return (s.max() <= 1.01) and (s.min() >= -1.0)

    if looks_like_returns(bench) or looks_like_returns(strat):
        print("Detected series that look like raw returns — reconstructing cumulative wealth from daily returns.")
        portfolio['Benchmark_Wealth'] = (1 + portfolio['Benchmark_Return']).cumprod()
        portfolio['Strategy_Wealth'] = (1 + portfolio['Strategy_Return']).cumprod()
        bench = portfolio['Benchmark_Wealth']
        strat = portfolio['Strategy_Wealth']

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot cumulative wealth
    ax.plot(portfolio['Date'], bench, label='Buy & Hold Benchmark', linewidth=2.5, color='#1f77b4', alpha=0.9)
    ax.plot(portfolio['Date'], strat, label='Regime-Filtered Ensemble', linewidth=2.5, color='#ff7f0e', alpha=0.9)
    
    # Annotate end values
    last_date = portfolio['Date'].iloc[-1]
    bench_end = portfolio['Benchmark_Wealth'].iloc[-1]
    strat_end = portfolio['Strategy_Wealth'].iloc[-1]
    
    ax.annotate(f'${bench_end:.2f}', 
                xy=(last_date, bench_end), 
                xytext=(-60, -20), 
                textcoords='offset points',
                ha='right',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f77b4', alpha=0.7, edgecolor='black'),
                color='white',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#1f77b4', lw=1.5))
    
    ax.annotate(f'${strat_end:.2f}', 
                xy=(last_date, strat_end), 
                xytext=(-60, 20), 
                textcoords='offset points',
                ha='right',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ff7f0e', alpha=0.7, edgecolor='black'),
                color='white',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#ff7f0e', lw=1.5))
    
    # Add vertical line for COVID Crash (2020-03)
    covid_date = pd.to_datetime('2020-03-15')
    if portfolio['Date'].min() <= covid_date <= portfolio['Date'].max():
        ax.axvline(covid_date, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='COVID Crash (Mar 2020)')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Wealth ($1 Invested)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Performance: Regime-Filtered Ensemble vs Buy & Hold (2015-2025)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))

    # Set sensible y-limits (ensure positive axis for wealth)
    min_y = min(bench.min(), strat.min())
    max_y = max(bench.max(), strat.max())
    # If minimum is <= 0 (shouldn't happen for compounded wealth), set small positive floor
    if min_y <= 0:
        min_y = min(0.9, max_y * 0.001)
    pad = (max_y - min_y) * 0.05 if max_y > min_y else max_y * 0.05
    ax.set_ylim(max(min_y - pad, 0.0), max_y + pad)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUT_PLOT}")
    
    # --- SUMMARY STATISTICS ---
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Annualized returns
    years = (portfolio['Date'].iloc[-1] - portfolio['Date'].iloc[0]).days / 365.25
    bench_annualized = (final_bench_wealth ** (1/years) - 1) * 100
    strat_annualized = (final_strat_wealth ** (1/years) - 1) * 100
    
    print(f"Time Period: {years:.2f} years")
    print(f"Benchmark Final Wealth: ${final_bench_wealth:.4f}")
    print(f"Strategy Final Wealth: ${final_strat_wealth:.4f}")
    print(f"Benchmark Annualized Return: {bench_annualized:.2f}%")
    print(f"Strategy Annualized Return: {strat_annualized:.2f}%")
    
    # Calculate daily returns from wealth for volatility
    portfolio['Bench_Daily_Return'] = portfolio['Benchmark_Wealth'].pct_change()
    portfolio['Strat_Daily_Return'] = portfolio['Strategy_Wealth'].pct_change()
    
    # Volatility
    bench_vol = portfolio['Bench_Daily_Return'].std() * np.sqrt(252) * 100
    strat_vol = portfolio['Strat_Daily_Return'].std() * np.sqrt(252) * 100
    
    print(f"Benchmark Annualized Volatility: {bench_vol:.2f}%")
    print(f"Strategy Annualized Volatility: {strat_vol:.2f}%")
    
    # Sharpe Ratio
    bench_sharpe = bench_annualized / bench_vol if bench_vol > 0 else 0
    strat_sharpe = strat_annualized / strat_vol if strat_vol > 0 else 0
    
    print(f"Benchmark Sharpe Ratio: {bench_sharpe:.4f}")
    print(f"Strategy Sharpe Ratio: {strat_sharpe:.4f}")
    
    print("="*70)

if __name__ == '__main__':
    main()
