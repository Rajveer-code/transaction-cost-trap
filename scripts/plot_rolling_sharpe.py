from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / 'results' / 'predictions_ensemble.csv'
OUT_PNG = ROOT / 'results' / 'rolling_sharpe.png'


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Predictions file not found: {path}')
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date':'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    # Expect columns: strategy_return, benchmark_return
    if 'strategy_return' in df.columns and 'benchmark_return' in df.columns:
        agg = df.groupby('Date')[['strategy_return','benchmark_return']].mean().sort_index()
        return agg

    # Fallback: compute daily returns from raw OHLCV and derive strategy/benchmark returns
    RAW = Path(__file__).resolve().parents[1] / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
    if not RAW.exists():
        raise RuntimeError(f"Required columns not found and raw data missing. Available columns: {df.columns.tolist()}")

    raw = pd.read_parquet(RAW)
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker','Date']).reset_index(drop=True)
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw = raw[['Date','Ticker','daily_return']]

    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()
    merged = pd.merge(df, raw, on=['Date','Ticker'], how='inner')

    # Use same regime filter as other scripts: ratio_sma200 and prob_ens
    merged['ratio_sma200'] = merged.get('ratio_sma200', 0).fillna(0)
    merged['Signal'] = ((merged.get('prob_ens', 0) > 0.50) & (merged['ratio_sma200'] > 1.0)).astype(int)

    merged['strategy_return'] = merged['Signal'] * merged['daily_return']
    merged['benchmark_return'] = merged['daily_return']

    agg = merged.groupby('Date')[['strategy_return','benchmark_return']].mean().sort_index()
    return agg


def compute_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    # rolling mean and std (daily)
    roll_mean = returns.rolling(window=window, min_periods=30).mean()
    roll_std = returns.rolling(window=window, min_periods=30).std()
    # annualization factor sqrt(252)
    ann = np.sqrt(252.0)
    rolling_sharpe = (roll_mean * 252.0) / (roll_std * ann)
    return rolling_sharpe


def plot(agg: pd.DataFrame):
    # cumulative returns
    cum = (1.0 + agg).cumprod()

    strat_sharpe = compute_rolling_sharpe(agg['strategy_return'])
    bench_sharpe = compute_rolling_sharpe(agg['benchmark_return'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[1,1]})

    # Panel A: Cumulative returns
    ax1.plot(cum.index, cum['benchmark_return'], color='tab:blue', label='Benchmark')
    ax1.plot(cum.index, cum['strategy_return'], color='tab:orange', label='Strategy')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Panel B: Rolling Sharpe
    ax2.plot(bench_sharpe.index, bench_sharpe, color='tab:blue', label='Benchmark Sharpe')
    ax2.plot(strat_sharpe.index, strat_sharpe, color='tab:orange', label='Strategy Sharpe')
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    # Shade where strategy Sharpe < benchmark Sharpe
    cond = (strat_sharpe < bench_sharpe)
    ax2.fill_between(strat_sharpe.index, strat_sharpe, bench_sharpe, where=cond, interpolate=True, color='red', alpha=0.2)
    ax2.set_ylabel('Rolling Sharpe (252-day)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    title = 'Panel A: Cumulative Returns | Panel B: Rolling Sharpe Ratio (1Y Window)'
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('Saved rolling sharpe figure to', OUT_PNG)


def main():
    agg = load_returns(PRED_CSV)
    plot(agg)


if __name__ == '__main__':
    main()
