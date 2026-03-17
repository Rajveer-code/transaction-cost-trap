"""
Transaction cost sensitivity analysis.

Creates a simple table showing how per-trade transaction costs (in bps)
impact the strategy's annual return and Sharpe, assuming a fixed
annual trade count and constant volatility.

Usage: python scripts/transaction_cost_sensitivity.py
"""
import pandas as pd
from pathlib import Path
import math


ROOT = Path(__file__).resolve().parents[1]
PREDS_FILE = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_DATA_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'


def compute_final_wealth_from_predictions():
    """Recompute portfolio final wealth using daily returns and signals.

    This mirrors the approach used elsewhere in the repo: compute daily
    returns from raw OHLCV, merge with predictions, form signals and
    compound the average daily portfolio return.
    Returns the final strategy wealth (1 -> final multiplier).
    """
    preds = pd.read_csv(PREDS_FILE)
    raw = pd.read_parquet(RAW_DATA_FILE)

    # prepare
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw = raw[['Date', 'Ticker', 'daily_return']]

    preds['Date'] = pd.to_datetime(preds['Date'])
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()

    merged = pd.merge(preds, raw, on=['Date', 'Ticker'], how='inner')

    # Strategy logic: same filter used for main analysis
    merged['ratio_sma200'] = merged.get('ratio_sma200', 0).fillna(0)
    merged['Signal'] = ((merged['prob_ens'] > 0.50) & (merged['ratio_sma200'] > 1.0)).astype(int)

    # Benchmark and strategy returns (daily)
    merged['Benchmark_Return'] = merged['daily_return']
    merged['Strategy_Return'] = merged['Signal'] * merged['Benchmark_Return']

    # aggregate daily across tickers (equal-weight)
    portfolio = merged.groupby('Date').agg({
        'Benchmark_Return': 'mean',
        'Strategy_Return': 'mean',
    }).reset_index()

    # compound
    portfolio['Benchmark_Wealth'] = (1 + portfolio['Benchmark_Return']).cumprod()
    portfolio['Strategy_Wealth'] = (1 + portfolio['Strategy_Return']).cumprod()

    final_strategy_wealth = portfolio['Strategy_Wealth'].iloc[-1]
    final_benchmark_wealth = portfolio['Benchmark_Wealth'].iloc[-1]

    return final_strategy_wealth, final_benchmark_wealth


def run_sensitivity():
    # Settings
    TOTAL_TRADES = 4711
    YEARS = 10
    ANNUAL_TRADES = 471  # as requested

    print('Computing final wealth from predictions...')
    final_strategy_wealth, final_benchmark_wealth = compute_final_wealth_from_predictions()

    # Base annual return from final wealth
    base_annual_return = final_strategy_wealth ** (1.0 / YEARS) - 1.0

    # Cost parameters (bps per half-turn)
    costs_bps = [0, 5, 10, 15, 20, 25]

    vol = 0.15  # constant volatility (15%) as requested

    rows = []
    for c in costs_bps:
        annual_cost_pct = (ANNUAL_TRADES * 2 * c) / 10000.0  # convert bps to decimal
        net_return = base_annual_return - annual_cost_pct
        sharpe = net_return / vol if vol > 0 else float('nan')
        rows.append((c, annual_cost_pct, net_return, sharpe))

    # print markdown table
    print('\n| Cost (bps) | Annual Cost % | Net Return % | Sharpe |')
    print('|---:|---:|---:|---:|')
    for c, annual_cost_pct, net_return, sharpe in rows:
        print(f'| {c} | {annual_cost_pct*100:.4f}% | {net_return*100:.4f}% | {sharpe:.4f} |')

    # Breakeven: first cost where Sharpe < benchmark (1.22)
    benchmark_sharpe = 1.22
    breakeven = None
    for c, _, _, sharpe in rows:
        if sharpe < benchmark_sharpe:
            breakeven = c
            break

    print('\nSummary:')
    print(f' - Final strategy wealth (1 -> x): {final_strategy_wealth:.6f}')
    print(f' - Base annual return: {base_annual_return*100:.4f}%')
    if breakeven is not None:
        print(f' - Breakeven cost where Sharpe < {benchmark_sharpe}: {breakeven} bps (per half-turn)')
    else:
        print(f' - No breakeven cost found up to {costs_bps[-1]} bps (Sharpe stays >= {benchmark_sharpe})')


if __name__ == '__main__':
    run_sensitivity()
