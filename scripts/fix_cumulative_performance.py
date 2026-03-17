"""
Generate corrected cumulative performance chart per the CRITICAL FIX requirements.

If `results/predictions_ensemble.csv` already contains `strategy_return` and
`benchmark_return` those will be used. Otherwise the script will compute daily
returns from raw OHLCV (`data/extended/all_free_data_2015_2025.parquet`), merge
with predictions, apply the same regime-filter signal, and form equal-weight
daily portfolio returns.

Saves figure to `results/fixed_cumulative_performance.png`.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PRED_FILE = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PNG = ROOT / 'results' / 'fixed_cumulative_performance.png'


def load_returns_from_preds(df_preds: pd.DataFrame):
    """Return DataFrame with Date, strategy_return, benchmark_return if present."""
    cols = df_preds.columns
    if 'strategy_return' in cols and 'benchmark_return' in cols:
        out = df_preds[['Date', 'Ticker', 'strategy_return', 'benchmark_return']].copy()
        out['Date'] = pd.to_datetime(out['Date'])
        return out
    return None


def compute_returns_from_raw(preds: pd.DataFrame):
    # compute daily returns from raw OHLCV and merge with preds
    raw = pd.read_parquet(RAW_FILE)
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw = raw[['Date', 'Ticker', 'daily_return']]

    preds = preds.copy()
    preds['Date'] = pd.to_datetime(preds['Date'])
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()

    merged = pd.merge(preds, raw, on=['Date', 'Ticker'], how='inner')

    # apply regime-filter signal same as elsewhere
    merged['ratio_sma200'] = merged.get('ratio_sma200', 0).fillna(0)
    merged['Signal'] = ((merged['prob_ens'] > 0.50) & (merged['ratio_sma200'] > 1.0)).astype(int)

    merged['strategy_return'] = merged['Signal'] * merged['daily_return']
    merged['benchmark_return'] = merged['daily_return']

    return merged[['Date', 'Ticker', 'strategy_return', 'benchmark_return']]


def aggregate_daily_portfolio(returns_df: pd.DataFrame):
    # equal-weight average across tickers per date
    df = returns_df.groupby('Date').agg({
        'strategy_return': 'mean',
        'benchmark_return': 'mean'
    }).reset_index().sort_values('Date')
    return df


def validate_and_plot(portfolio_daily: pd.DataFrame):
    # compute cumulative wealth from $1
    portfolio_daily = portfolio_daily.copy()
    portfolio_daily['strategy_wealth'] = (1 + portfolio_daily['strategy_return']).cumprod()
    portfolio_daily['benchmark_wealth'] = (1 + portfolio_daily['benchmark_return']).cumprod()

    final_strat = portfolio_daily['strategy_wealth'].iloc[-1]
    final_bench = portfolio_daily['benchmark_wealth'].iloc[-1]

    # Validation: cannot be negative
    if (portfolio_daily['strategy_wealth'] <= 0).any() or (portfolio_daily['benchmark_wealth'] <= 0).any():
        raise RuntimeError('BUG: Cumulative wealth cannot be negative')

    # Validation: final wealth should be roughly the expected values
    # Accept small tolerance
    if not (0.9 <= final_strat <= 2.5):
        print(f'WARNING: final strategy wealth = {final_strat:.6f} is outside expected ~1.57', file=sys.stderr)
    if not (5.0 <= final_bench <= 30.0):
        print(f'WARNING: final benchmark wealth = {final_bench:.6f} is outside expected ~12.96', file=sys.stderr)

    # Plotting
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(portfolio_daily['Date'], portfolio_daily['benchmark_wealth'], label=f'Buy & Hold: ${final_bench:.2f}', color='#1f77b4', linewidth=3)
    ax.plot(portfolio_daily['Date'], portfolio_daily['strategy_wealth'], label=f'ML Strategy: ${final_strat:.2f}', color='#ff7f0e', linewidth=3)

    # COVID vertical
    covid_date = pd.to_datetime('2020-03-15')
    if portfolio_daily['Date'].min() <= covid_date <= portfolio_daily['Date'].max():
        ax.axvline(covid_date, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-04-22'))

    # Y-axis start at 0
    ax.set_ylim(0, max(portfolio_daily['benchmark_wealth'].max(), portfolio_daily['strategy_wealth'].max()) * 1.05)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Wealth ($1 Invested)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Performance: Regime-Filtered Ensemble vs Buy & Hold (2015-2025)', fontsize=14, fontweight='bold')

    ax.legend(fontsize=11)
    ax.grid(color='gray', alpha=0.3)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig.savefig(OUT_PNG, dpi=300)
    print(f'Saved plot to {OUT_PNG}')


def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f'{PRED_FILE} not found')

    preds = pd.read_csv(PRED_FILE)

    returns_df = load_returns_from_preds(preds)
    if returns_df is None:
        print('strategy_return / benchmark_return not found in predictions; computing from raw data...')
        returns_df = compute_returns_from_raw(preds)

    daily = aggregate_daily_portfolio(returns_df)
    validate_and_plot(daily)


if __name__ == '__main__':
    main()
