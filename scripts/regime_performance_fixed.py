"""Fixed regime performance analysis with trade-level signal accounting.

Signals (trades) are defined as contiguous runs where the model prediction
 (`y_pred==1` or `prob_ens>0.5`) is True for a given ticker. Each such run
 is a single trade counted once, with entry date = first day of the run.

Regime definitions:
 - Bull: price > SMA_200 AND momentum > 0
 - Bear: price < SMA_200 AND momentum < 0
 - High Volatility: ATR > 2 * median(ATR) OR VIX > 25
 - Sideways: otherwise

Trade return: cumulative product of (1 + next_day_return) over the trade holding
period minus 1, expressed as percent.

Outputs:
 - results/regime_performance_fixed.png
 - results/regime_performance_fixed.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / 'results' / 'predictions_ensemble.csv'
FEATURES_PATH = ROOT / 'data' / 'combined' / 'all_features.parquet'
OUT_PNG = ROOT / 'results' / 'regime_performance_fixed.png'
OUT_CSV = ROOT / 'results' / 'regime_performance_fixed.csv'


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def annualized_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    # returns is array of trade returns (decimal, not percent)
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return float('nan')
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return float('nan')
    return (mean / std) * math.sqrt(trading_days)


def define_regime(row, atr_median, has_vix):
    price = row.get('close')
    sma200 = row.get('sma_200')
    momentum = row.get('return_1d')
    atr = row.get('atr')
    vix = row.get('vix') if has_vix else None

    if pd.notna(price) and pd.notna(sma200) and pd.notna(momentum):
        if (price > sma200) and (momentum > 0):
            return 'Bull'
        if (price < sma200) and (momentum < 0):
            return 'Bear'

    if pd.notna(atr) and (atr_median is not None) and pd.notna(atr_median) and (atr > 2.0 * atr_median):
        return 'High Volatility'
    if has_vix and pd.notna(vix) and (vix > 25.0):
        return 'High Volatility'
    return 'Sideways'


def extract_trades(merged: pd.DataFrame):
    # Identify trades per ticker as contiguous runs where signal==True
    trades = []
    merged = merged.sort_values(['Ticker','Date']).reset_index(drop=True)
    for ticker, g in merged.groupby('Ticker'):
        g = g.reset_index(drop=True)
        # signal boolean
        sig = g['signal'].fillna(False).astype(bool)
        # start of trade when sig True and previous False (or first row)
        starts = sig & (~sig.shift(1).fillna(False))
        # assign trade ids
        trade_id = (starts.cumsum() * sig).replace(0, np.nan)
        g['trade_id'] = trade_id
        # drop rows with no trade_id
        trade_rows = g.dropna(subset=['trade_id'])
        # We'll compute trade returns using close prices: return = close_after_exit / close_at_entry - 1
        closes = g['close'].astype(float) if 'close' in g.columns else pd.Series([np.nan]*len(g))
        for tid, tg in trade_rows.groupby('trade_id'):
            entry_idx = tg.index[0]
            exit_idx = tg.index[-1]
            entry_date = tg['Date'].iloc[0]
            exit_date = tg['Date'].iloc[-1]
            # find close at entry and close at next available row after exit
            try:
                close_entry = closes.loc[entry_idx]
            except Exception:
                close_entry = np.nan
            try:
                close_after_exit = closes.loc[exit_idx + 1]
            except Exception:
                # if next row not available, try using exit day's close (intra-day exit)
                try:
                    close_after_exit = closes.loc[exit_idx]
                except Exception:
                    close_after_exit = np.nan
            if pd.isna(close_entry) or pd.isna(close_after_exit):
                trade_return = np.nan
            else:
                trade_return = (float(close_after_exit) / float(close_entry)) - 1.0
            trades.append({'Ticker': ticker, 'entry_date': entry_date, 'exit_date': exit_date, 'trade_return': trade_return, 'n_days': len(tg)})
    trades_df = pd.DataFrame(trades)
    return trades_df


def main():
    preds = load_predictions(PRED_PATH)
    feats = load_features(FEATURES_PATH)

    # standardize cols
    preds = preds.rename(columns={c: c.strip() for c in preds.columns})
    date_col = next((c for c in preds.columns if c.lower()=='date'), None)
    ticker_col = next((c for c in preds.columns if c.lower()=='ticker'), None)
    if date_col is None or ticker_col is None:
        raise RuntimeError('Predictions file missing Date or Ticker columns')
    preds['Date'] = pd.to_datetime(preds[date_col])
    preds['Ticker'] = preds[ticker_col]

    # normalize feature names
    feats_cols_lower = {c.lower(): c for c in feats.columns}
    for need in ('date','ticker','close','sma_200','atr','return_1d','vix','next_day_return'):
        if need in feats_cols_lower:
            feats = feats.rename(columns={feats_cols_lower[need]: need})
    feats['date'] = pd.to_datetime(feats['date'])
    feats = feats.rename(columns={'date':'Date','ticker':'Ticker'})

    merged = pd.merge(preds, feats, on=['Date','Ticker'], how='left')

    # signal boolean
    if 'y_pred' in merged.columns:
        merged['signal'] = merged['y_pred'].astype(int) == 1
    elif 'prob_ens' in merged.columns:
        merged['signal'] = merged['prob_ens'] > 0.5
    else:
        raise RuntimeError('No signal column found (y_pred or prob_ens)')

    # prepare next_ret
    if 'next_day_return' in merged.columns:
        merged['next_ret'] = merged['next_day_return']
    elif 'return_1d' in merged.columns:
        merged['next_ret'] = merged['return_1d']
    else:
        merged['next_ret'] = np.nan

    # regime computation
    atr_median = merged['atr'].median() if 'atr' in merged.columns else np.nan
    has_vix = 'vix' in merged.columns
    merged['regime'] = merged.apply(lambda r: define_regime(r, atr_median, has_vix), axis=1)

    # extract trades
    trades = extract_trades(merged)
    if trades.empty:
        raise RuntimeError('No trades found in predictions (no contiguous signal runs)')

    # attach regime at entry: find regime from merged using Date+Ticker
    trades = trades.merge(merged[['Ticker','Date','regime']].rename(columns={'Date':'entry_date','regime':'entry_regime'}), on=['Ticker','entry_date'], how='left')

    # Aggregate per regime
    regimes = ['Bull','Bear','High Volatility','Sideways']
    rows = []
    for reg in regimes:
        # days in regime: unique dates where merged.regime==reg
        days = int(merged[merged['regime']==reg]['Date'].nunique())
        reg_trades = trades[trades['entry_regime']==reg]
        signals = len(reg_trades)
        # validation signals <= days
        if signals > days:
            raise RuntimeError(f'HARD VALIDATION FAILED: Signals ({signals}) > Days ({days}) for regime {reg}')
        # compute stats only on trades with valid returns
        valid_trade_returns = reg_trades['trade_return'].dropna().astype(float).values
        valid_n = len(valid_trade_returns)
        if valid_n > 0:
            wins = np.sum(valid_trade_returns > 0)
            win_rate = wins / valid_n * 100.0
            mean_trade_ret = np.nanmean(valid_trade_returns) * 100.0
            sharpe = annualized_sharpe(valid_trade_returns)
        else:
            win_rate = float('nan')
            mean_trade_ret = float('nan')
            sharpe = float('nan')
        rows.append({'Regime': reg, 'Days': days, 'Signals': signals, 'Win Rate': win_rate, 'Avg Return': mean_trade_ret, 'Sharpe': sharpe})

    summary = pd.DataFrame(rows).set_index('Regime')

    # HARD checks
    # Win rate in [30,70]
    for idx, r in summary.iterrows():
        if pd.notna(r['Win Rate']):
            if not (30.0 <= r['Win Rate'] <= 70.0):
                raise RuntimeError(f'HARD VALIDATION FAILED: Win Rate {r["Win Rate"]:.2f}% for {idx} not in [30,70]')
        # Sharpe sign matches avg return sign
        if pd.notna(r['Avg Return']) and pd.notna(r['Sharpe']):
            if r['Avg Return'] > 0 and r['Sharpe'] <= 0:
                raise RuntimeError(f'HARD VALIDATION FAILED: Sharpe {r["Sharpe"]:.4f} non-positive while Avg Return {r["Avg Return"]:.4f}% positive for {idx}')
            if r['Avg Return'] < 0 and r['Sharpe'] >= 0:
                raise RuntimeError(f'HARD VALIDATION FAILED: Sharpe {r["Sharpe"]:.4f} non-negative while Avg Return {r["Avg Return"]:.4f}% negative for {idx}')
        # If Win rate <45% then Sharpe must be <=0
        if pd.notna(r['Win Rate']) and r['Win Rate'] < 45.0 and pd.notna(r['Sharpe']):
            if r['Sharpe'] > 0:
                raise RuntimeError(f'HARD VALIDATION FAILED: Win rate {r["Win Rate"]:.2f}% <45% but Sharpe {r["Sharpe"]:.4f} > 0 for {idx}')

    # Plot
    win_rates = summary['Win Rate'].fillna(0).tolist()
    signals = summary['Signals'].tolist()
    colors = [('#2ca02c' if w > 55.0 else '#ffcc00' if w >= 50.0 else '#d62728') if not np.isnan(w) else '#cccccc' for w in win_rates]

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2,1, height_ratios=[3,1], hspace=0.4)
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(summary.index.tolist(), win_rates, color=colors, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Model Performance Across Market Regimes (2015-2025) — Fixed (trade-level)')
    for bar, wr, sig in zip(bars, win_rates, signals):
        h = bar.get_height()
        if np.isnan(wr):
            txt = 'n/a'
        else:
            txt = f"{wr:.1f}%\n{sig} trades"
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, txt, ha='center', va='bottom', fontsize=9)

    # table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table_df = summary.reset_index()
    table_df['Win Rate'] = table_df['Win Rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else 'n/a')
    table_df['Sharpe'] = table_df['Sharpe'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'n/a')
    table_df['Avg Return'] = table_df['Avg Return'].apply(lambda x: f"{x:.3f}%" if pd.notna(x) else 'n/a')
    cols = ['Regime','Days','Signals','Win Rate','Sharpe','Avg Return']
    cell_text = table_df[cols].values.tolist()
    table = ax_table.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1,1.2)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    summary.to_csv(OUT_CSV)
    print('Saved fixed regime performance figure to', OUT_PNG)
    print('Saved fixed regime summary CSV to', OUT_CSV)


if __name__ == '__main__':
    main()
