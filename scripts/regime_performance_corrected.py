"""Recompute regimes from raw price data and regenerate regime performance.

Steps implemented:
- Load raw prices from `data/extended/all_free_data_2015_2025.parquet`.
- Compute per-ticker: `sma_200`, `return_1d`, `ATR_14`.
- Merge computed fields into `results/predictions_ensemble.csv` by Date+Ticker.
- Define regimes with priority: High Volatility -> Bull/Bear -> Sideways.
- Extract trades as contiguous runs of signals (one trade per run), assign trade entry regime.
- Compute per-regime metrics and perform HARD validation checks.
- Save figure `results/regime_performance_corrected.png` and CSV `results/regime_performance_corrected.csv`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).resolve().parents[1]
RAW_PRICES = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
PRED_PATH = ROOT / 'results' / 'predictions_ensemble.csv'
OUT_PNG = ROOT / 'results' / 'regime_performance_corrected.png'
OUT_CSV = ROOT / 'results' / 'regime_performance_corrected.csv'
DIAG_TRADES = ROOT / 'results' / 'regime_trades_corrected.csv'


def compute_indicators(df_prices: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: Date, Ticker, Open/High/Low/Close/Volume (case-insensitive)
    cols_lower = {c.lower(): c for c in df_prices.columns}
    rename_map = {}
    if 'date' in cols_lower:
        rename_map[cols_lower['date']] = 'Date'
    if 'ticker' in cols_lower:
        rename_map[cols_lower['ticker']] = 'Ticker'
    if 'close' in cols_lower:
        rename_map[cols_lower['close']] = 'Close'
    if 'high' in cols_lower:
        rename_map[cols_lower['high']] = 'High'
    if 'low' in cols_lower:
        rename_map[cols_lower['low']] = 'Low'
    if 'open' in cols_lower:
        rename_map[cols_lower['open']] = 'Open'
    df_prices = df_prices.rename(columns=rename_map)
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])

    # compute per-ticker indicators
    def per_ticker(g):
        g = g.sort_values('Date').copy()
        g['sma_200'] = g['Close'].rolling(window=200, min_periods=1).mean()
        g['return_1d'] = g['Close'].pct_change(1)
        # ATR_14
        high = g['High']
        low = g['Low']
        close = g['Close']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        g['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
        return g

    out = df_prices.groupby('Ticker', group_keys=False).apply(per_ticker).reset_index(drop=True)
    return out


def extract_trades_from_preds(merged: pd.DataFrame):
    # Determine signal boolean
    if 'y_pred' in merged.columns:
        merged['signal'] = merged['y_pred'].astype(int) == 1
    elif 'prob_ens' in merged.columns:
        merged['signal'] = merged['prob_ens'] > 0.5
    else:
        merged['signal'] = False

    merged = merged.sort_values(['Ticker','Date']).reset_index(drop=True)
    trades = []
    for ticker, g in merged.groupby('Ticker'):
        g = g.reset_index(drop=True)
        sig = g['signal'].fillna(False).astype(bool)
        starts = sig & (~sig.shift(1).fillna(False))
        trade_id = (starts.cumsum() * sig).replace(0, np.nan)
        g['trade_id'] = trade_id
        trade_rows = g.dropna(subset=['trade_id'])
        for tid, tg in trade_rows.groupby('trade_id'):
            entry = tg.iloc[0]
            exit = tg.iloc[-1]
            entry_date = entry['Date']
            exit_date = exit['Date']
            # trade return computed from Close: next available close after exit / close at entry - 1
            # find index of entry in g
            entry_idx = tg.index[0]
            exit_idx = tg.index[-1]
            closes = g['Close'] if 'Close' in g.columns else pd.Series([np.nan]*len(g))
            try:
                close_entry = float(closes.loc[entry_idx])
            except Exception:
                close_entry = np.nan
            try:
                close_after = float(closes.loc[exit_idx + 1])
            except Exception:
                # fallback to exit day's close
                try:
                    close_after = float(closes.loc[exit_idx])
                except Exception:
                    close_after = np.nan
            if pd.isna(close_entry) or pd.isna(close_after) or close_entry == 0:
                tr = np.nan
            else:
                tr = close_after / close_entry - 1.0
            # regime at entry
            entry_regime = entry.get('regime') if 'regime' in entry.index else None
            trades.append({'Ticker': ticker, 'entry_date': entry_date, 'exit_date': exit_date, 'entry_regime': entry_regime, 'trade_return': tr})
    trades_df = pd.DataFrame(trades)
    return trades_df


def annualized_sharpe(returns, days_in_regime):
    # returns: array of trade returns (decimal)
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return np.nan
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return np.nan
    # trades per year approx = (n_trades / days_in_regime) * 252
    n_trades = len(returns)
    if days_in_regime <= 0:
        return np.nan
    trades_per_year = (n_trades / days_in_regime) * 252.0
    if trades_per_year <= 0:
        return np.nan
    return (mean / std) * math.sqrt(trades_per_year)


def main():
    # Load raw prices
    if not RAW_PRICES.exists():
        raise FileNotFoundError(f'Raw price file not found: {RAW_PRICES}')
    prices = pd.read_parquet(RAW_PRICES)
    prices_cols = [c.lower() for c in prices.columns]
    # Normalize column names
    colmap = {c: c for c in prices.columns}
    # lowercase keys mapping handled in compute_indicators
    prices_ind = compute_indicators(prices)

    # Load predictions
    preds = pd.read_csv(PRED_PATH)
    preds.columns = [c.strip() for c in preds.columns]
    # Ensure Date, Ticker column names match
    if 'Date' not in preds.columns and 'date' in preds.columns:
        preds = preds.rename(columns={'date':'Date'})
    if 'Ticker' not in preds.columns and 'ticker' in preds.columns:
        preds = preds.rename(columns={'ticker':'Ticker'})
    preds['Date'] = pd.to_datetime(preds['Date'])

    # Merge computed indicators into preds by Date+Ticker
    merged = pd.merge(preds, prices_ind[['Date','Ticker','Close','sma_200','return_1d','ATR_14']], on=['Date','Ticker'], how='left')

    # Compute regime per priority: High Volatility -> Bull/Bear -> Sideways
    atr_med = merged['ATR_14'].median()

    def regime_for_row(r):
        close = r.get('Close')
        sma = r.get('sma_200')
        ret1 = r.get('return_1d')
        atr = r.get('ATR_14')
        # High Volatility
        if pd.notna(atr) and pd.notna(atr_med) and (atr > 2.0 * atr_med):
            return 'High Volatility'
        # Bull/Bear (require sma and ret)
        if pd.notna(close) and pd.notna(sma) and pd.notna(ret1):
            if (close > sma) and (ret1 > 0):
                return 'Bull'
            if (close < sma) and (ret1 < 0):
                return 'Bear'
        # Sideways
        try:
            if pd.notna(close) and pd.notna(sma) and pd.notna(ret1):
                if (abs(close - sma) / sma < 0.02) and (abs(ret1) < 0.005):
                    return 'Sideways'
        except Exception:
            pass
        return 'Sideways'

    merged['regime'] = merged.apply(regime_for_row, axis=1)

    # Extract trades
    trades = extract_trades_from_preds(merged)
    if trades.empty:
        raise RuntimeError('No trades found')

    # Attach regime from merged (entry date)
    merged_key = merged[['Date','Ticker','regime','Close']].copy()
    merged_key = merged_key.rename(columns={'Date':'entry_date','regime':'entry_regime','Close':'Close_at_date'})
    # Ensure datetimes align for merging
    if 'entry_date' in trades.columns:
        trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    merged_key['entry_date'] = pd.to_datetime(merged_key['entry_date'])

    # Primary merge attempt
    trades = trades.merge(merged_key, on=['Ticker','entry_date'], how='left')

    # Defensive check: if merge did not produce `entry_regime`, try alternate merge and warn
    if 'entry_regime' not in trades.columns:
        # Attempt merge using right_on Date to be robust to naming differences
        alt = merged[['Date','Ticker','regime','Close']].rename(columns={'Date':'entry_date','regime':'entry_regime','Close':'Close_at_date'}).copy()
        alt['entry_date'] = pd.to_datetime(alt['entry_date'])
        trades = trades.merge(alt, on=['Ticker','entry_date'], how='left')
    # Final guard: ensure column exists
    if 'entry_regime' not in trades.columns:
        trades['entry_regime'] = np.nan
        print('Warning: entry_regime not found after merge — filling with NaN')

    # Per-regime aggregates
    regimes = ['Bull','Bear','High Volatility','Sideways']
    rows = []
    total_trades = len(trades)
    for reg in regimes:
        days = int(merged[merged['regime']==reg]['Date'].nunique())
        reg_trades = trades[trades['entry_regime']==reg]
        n_trades = len(reg_trades)
        if days == 0 or n_trades == 0:
            raise RuntimeError('Invalid regime definition — check SMA_200 or momentum computation')
        # compute stats on valid returns
        valid = reg_trades['trade_return'].dropna().astype(float).values
        wins = int(np.sum(valid > 0)) if len(valid)>0 else 0
        win_rate = (wins / len(valid) * 100.0) if len(valid)>0 else float('nan')
        mean_ret = np.nanmean(valid) * 100.0 if len(valid)>0 else float('nan')
        sharpe = annualized_sharpe(valid, days)
        rows.append({'Regime': reg, 'Days': days, 'Trades': n_trades, 'Win Rate': win_rate, 'Mean Return': mean_ret, 'Sharpe': sharpe})

    summary = pd.DataFrame(rows).set_index('Regime')

    # Validations
    if summary.loc['Bull','Days'] <= 500 or summary.loc['Bear','Days'] <= 500:
        raise RuntimeError('HARD VALIDATION FAILED: Bull and Bear must each have > 500 days')
    # No regime may have Win Rate <45% unless labeled Fail (not applicable)
    for idx, r in summary.iterrows():
        if pd.notna(r['Win Rate']) and r['Win Rate'] < 45.0:
            raise RuntimeError(f'HARD VALIDATION FAILED: {idx} Win Rate {r['Win Rate']:.2f}% < 45%')
    # Sum of trades
    if summary['Trades'].sum() != total_trades:
        raise RuntimeError('HARD VALIDATION FAILED: Sum of trades across regimes != total trades')

    # Plot
    win_rates = summary['Win Rate'].tolist()
    trades_counts = summary['Trades'].tolist()
    colors = [('#2ca02c' if w > 55.0 else '#ffcc00' if w >= 50.0 else '#d62728') for w in win_rates]
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(summary.index.tolist(), win_rates, color=colors, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Win Rate (%)')
    for b, tr_count in zip(bars, trades_counts):
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}%\n{tr_count} trades", ha='center')
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    summary.to_csv(OUT_CSV)
    trades.to_csv(DIAG_TRADES, index=False)
    print('Saved corrected regime performance figure to', OUT_PNG)
    print('Saved corrected summary to', OUT_CSV)


if __name__ == '__main__':
    main()
