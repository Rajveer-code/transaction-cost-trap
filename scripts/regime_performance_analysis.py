"""Recompute regimes, produce regime performance chart and table.

Definitions (documented here and used in code):
 - Bull: price > SMA_200 AND momentum > 0
 - Bear: price < SMA_200 AND momentum < 0
 - High Volatility: ATR > 2 * median(ATR) OR VIX > 25
 - Sideways: all remaining periods

Notes:
 - `momentum` is implemented as `return_1d` (one-day return) from features.
 - If `VIX` is missing in the dataset, the VIX condition is ignored.
 - Signals are taken from `results/predictions_ensemble.csv` using `y_pred==1`.
 - Win = next-day return > 0. Mean return and Sharpe are computed on the set of signals in the regime.

Saves:
 - `results/regime_performance.png` (300 DPI)
 - `results/regime_performance_table.csv`
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / 'results' / 'predictions_ensemble.csv'
FEATURES_PATH = ROOT / 'data' / 'combined' / 'all_features.parquet'
OUT_PNG = ROOT / 'results' / 'regime_performance.png'
OUT_CSV = ROOT / 'results' / 'regime_performance_table.csv'

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df

def annualized_sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    if len(returns) < 2 or np.all(np.isclose(returns, 0)):
        return float('nan')
    mean = np.nanmean(returns)
    std = np.nanstd(returns, ddof=1)
    if std == 0 or np.isnan(std):
        return float('nan')
    return (mean / std) * math.sqrt(trading_days)

def define_regime(row, atr_median, has_vix):
    price = row.get('close')
    sma200 = row.get('sma_200')
    momentum = row.get('return_1d')
    atr = row.get('atr')
    vix = row.get('vix') if has_vix else None

    # Bull
    if pd.notna(price) and pd.notna(sma200) and pd.notna(momentum):
        if (price > sma200) and (momentum > 0):
            return 'Bull'
    # Bear
    if pd.notna(price) and pd.notna(sma200) and pd.notna(momentum):
        if (price < sma200) and (momentum < 0):
            return 'Bear'
    # High Volatility
    try:
        if pd.notna(atr) and (atr > 2.0 * atr_median):
            return 'High Volatility'
    except Exception:
        pass
    if has_vix and pd.notna(vix) and (vix > 25.0):
        return 'High Volatility'
    # Sideways default
    return 'Sideways'

def main():
    preds = load_predictions(PRED_PATH)
    feats = load_features(FEATURES_PATH)

    # normalize column names to lowercase for safe merging
    preds_cols = {c: c for c in preds.columns}
    # Expect Date and Ticker (or date,ticker)
    # Clean and lowercase column names for merge
    preds = preds.rename(columns={c: c.strip() for c in preds.columns})
    # pick date and ticker column names
    date_col = next((c for c in preds.columns if c.lower() in ('date','dt','timestamp')), None)
    ticker_col = next((c for c in preds.columns if c.lower() in ('ticker','symbol')), None)
    if date_col is None or ticker_col is None:
        raise RuntimeError('Predictions file missing date or ticker column')

    preds[date_col] = pd.to_datetime(preds[date_col])

    # features side
    feats_cols_lower = {c.lower(): c for c in feats.columns}
    # standardize column names we need
    for needed in ('date','ticker','close','sma_200','atr','return_1d','vix','next_day_return'):
        if needed in feats_cols_lower:
            feats = feats.rename(columns={feats_cols_lower[needed]: needed})

    if 'date' not in feats.columns or 'ticker' not in feats.columns:
        raise RuntimeError('Features table missing date or ticker columns')

    feats['date'] = pd.to_datetime(feats['date'])

    # Merge predictions with features on date+ticker
    merged = pd.merge(preds, feats, left_on=[date_col, ticker_col], right_on=['date','ticker'], how='left')

    # Signals: rows where model produced a positive prediction
    if 'y_pred' in merged.columns:
        merged['signal'] = merged['y_pred'].astype(int) == 1
    else:
        # fallback: use prob_ens > 0.5 if available
        if 'prob_ens' in merged.columns:
            merged['signal'] = merged['prob_ens'] > 0.5
        else:
            raise RuntimeError('No signal column found (y_pred or prob_ens) in predictions')

    # Compute ATR median for volatility rule
    atr_median = merged['atr'].median() if 'atr' in merged.columns else float('nan')
    has_vix = 'vix' in merged.columns

    # Compute regime per row
    merged['regime'] = merged.apply(lambda r: define_regime(r, atr_median, has_vix), axis=1)

    # For next-day return use 'next_day_return' if present else try to compute from returns
    if 'next_day_return' in merged.columns:
        merged['next_ret'] = merged['next_day_return']
    else:
        # try return_1d shifted? if not available, use y_true mapped
        if 'return_1d' in merged.columns:
            merged['next_ret'] = merged['return_1d']
        else:
            merged['next_ret'] = merged['y_true'] if 'y_true' in merged.columns else np.nan

    # Aggregate per regime
    rows = []
    for regime, g in merged.groupby('regime'):
        days = g['date'].nunique()
        signals_df = g[g['signal'] == True]
        signals = len(signals_df)
        if signals > 0:
            wins = (signals_df['next_ret'] > 0).sum()
            win_rate = wins / signals * 100.0
            mean_ret = signals_df['next_ret'].mean() * 100.0
            sharpe = annualized_sharpe(signals_df['next_ret'].values)
        else:
            win_rate = float('nan')
            mean_ret = float('nan')
            sharpe = float('nan')
        rows.append({'Regime': regime, 'Days': int(days), 'Signals': int(signals), 'Win Rate': win_rate, 'Sharpe': sharpe, 'Avg Return': mean_ret})

    summary = pd.DataFrame(rows).set_index('Regime')

    # HARD validations
    # Bear-market win rate MUST be < 50%
    bear_wr = summary.loc['Bear','Win Rate'] if 'Bear' in summary.index else float('nan')
    bull_wr = summary.loc['Bull','Win Rate'] if 'Bull' in summary.index else float('nan')
    if pd.notna(bear_wr) and bear_wr >= 50.0:
        raise RuntimeError(f'HARD VALIDATION FAILED: Bear-market win rate {bear_wr:.2f}% >= 50%')
    if pd.notna(bull_wr) and pd.notna(bear_wr) and not (bull_wr > bear_wr):
        raise RuntimeError(f'HARD VALIDATION FAILED: Bull win rate {bull_wr:.2f}% is not greater than Bear {bear_wr:.2f}%')

    # Signals != Days (for at least one regime) — raise if equal for any regime
    for idx, r in summary.iterrows():
        if r['Signals'] == r['Days']:
            raise RuntimeError(f'HARD VALIDATION FAILED: Signals equals Days for regime {idx} ({r["Signals"]}) — check mapping between dates and signals')

    # Plot
    regimes = summary.index.tolist()
    win_rates = summary['Win Rate'].fillna(0).tolist()
    signals = summary['Signals'].tolist()

    colors = [('#2ca02c' if w > 55.0 else '#ffcc00' if w >= 50.0 else '#d62728') if not np.isnan(w) else '#cccccc' for w in win_rates]

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2,1, height_ratios=[3,1], hspace=0.4)
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(regimes, win_rates, color=colors, edgecolor='black')
    ax.axhline(50, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Model Performance Across Market Regimes (2015-2025)')
    for bar, wr, sig in zip(bars, win_rates, signals):
        h = bar.get_height()
        if np.isnan(wr):
            txt = 'n/a'
        else:
            txt = f"{wr:.1f}%\n{sig} signals"
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, txt, ha='center', va='bottom', fontsize=9)

    # Table below
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table_df = summary.reset_index()
    # format numbers
    table_df['Win Rate'] = table_df['Win Rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else 'n/a')
    table_df['Sharpe'] = table_df['Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'n/a')
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
    print('Saved regime performance figure to', OUT_PNG)
    print('Saved regime summary CSV to', OUT_CSV)


if __name__ == '__main__':
    main()
