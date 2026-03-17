"""
Generate regime performance breakdown table.

Defines market regimes using S&P 500 data:
- Bull: Price > SMA_200 AND momentum > 0
- Bear: Price < SMA_200 AND momentum < 0
- High Volatility: VIX > 25 OR ATR > 2x median ATR
- Sideways: All other periods

Computes per-regime: days, accuracy, ROC-AUC, Sharpe, signals, avg return.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / 'results' / 'predictions_ensemble.csv'
SP500_FILE = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_CSV = ROOT / 'results' / 'regime_performance_table.csv'
OUT_MD = ROOT / 'results' / 'regime_performance_table.md'


def load_sp500():
    """Load S&P 500 data (assuming ^GSPC or similar)."""
    if not SP500_FILE.exists():
        raise FileNotFoundError(f'S&P 500 data not found: {SP500_FILE}')

    df = pd.read_parquet(SP500_FILE)
    df.columns = [c.strip() for c in df.columns]

    # Rename to standard format
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
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

    df = df.rename(columns=rename_map)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Filter to S&P 500 (ticker = GSPC or ^GSPC or similar)
    sp500_tickers = ['^GSPC', 'GSPC', '^SPX', 'SPX', '^INDU', 'INDU']
    sp500 = df[df['Ticker'].str.upper().isin(sp500_tickers)]

    if sp500.empty:
        # Fallback: use first ticker (assume data is S&P 500)
        sp500 = df[df['Ticker'] == df['Ticker'].iloc[0]]

    sp500 = sp500[['Date', 'Close', 'High', 'Low']].copy()
    sp500 = sp500.sort_values('Date').reset_index(drop=True)

    # Compute indicators
    sp500['SMA_200'] = sp500['Close'].rolling(200, min_periods=1).mean()
    sp500['Return_1d'] = sp500['Close'].pct_change()
    sp500['Return_5d'] = sp500['Close'].pct_change(5)

    # ATR
    sp500['prev_close'] = sp500['Close'].shift(1)
    sp500['TR'] = np.maximum(
        sp500['High'] - sp500['Low'],
        np.maximum(
            (sp500['High'] - sp500['prev_close']).abs(),
            (sp500['Low'] - sp500['prev_close']).abs()
        )
    )
    sp500['ATR_14'] = sp500['TR'].rolling(14, min_periods=1).mean()

    return sp500


def assign_regimes(sp500: pd.DataFrame):
    """Assign market regime to each date."""
    sp500 = sp500.copy()
    atr_median = sp500['ATR_14'].median()

    def regime_label(row):
        close = row['Close']
        sma = row['SMA_200']
        ret_5d = row['Return_5d']
        atr = row['ATR_14']

        # High Volatility (priority 1)
        if pd.notna(atr) and atr > 2.0 * atr_median:
            return 'High Volatility'

        # Bull: Price > SMA and momentum > 0
        if pd.notna(close) and pd.notna(sma) and pd.notna(ret_5d):
            if close > sma and ret_5d > 0:
                return 'Bull'
            # Bear: Price < SMA and momentum < 0
            if close < sma and ret_5d < 0:
                return 'Bear'

        return 'Sideways'

    sp500['Regime'] = sp500.apply(regime_label, axis=1)
    return sp500


def compute_regime_metrics(preds: pd.DataFrame, sp500_regimes: pd.DataFrame):
    """Compute metrics per regime."""
    # Load daily returns from raw data
    raw = pd.read_parquet(ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet')
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw = raw[['Date', 'Ticker', 'daily_return']].copy()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()

    # Merge predictions with regime labels
    preds['Date'] = pd.to_datetime(preds['Date'])
    sp500_regimes['Date'] = pd.to_datetime(sp500_regimes['Date'])
    preds['Ticker'] = preds['Ticker'].astype(str).str.strip()

    merged = pd.merge(preds, sp500_regimes[['Date', 'Regime']], on='Date', how='left')
    merged = pd.merge(merged, raw, on=['Date', 'Ticker'], how='left')

    # Ensure we have predictions
    if 'prob_ens' not in merged.columns or 'y_true' not in merged.columns:
        raise RuntimeError('Missing prob_ens or y_true in predictions')

    # Predicted label
    merged['pred'] = (merged['prob_ens'] > 0.5).astype(int)

    rows = []
    regimes = ['Bull', 'Bear', 'High Volatility', 'Sideways']

    for regime in regimes:
        regime_data = merged[merged['Regime'] == regime].dropna(subset=['pred', 'y_true'])

        if regime_data.empty:
            rows.append({
                'Regime': regime,
                'Days': 0,
                'Accuracy (%)': np.nan,
                'ROC-AUC': np.nan,
                'Sharpe': np.nan,
                'Signals': 0,
                'Avg Daily Return (%)': np.nan
            })
            continue

        # Days
        days = int(regime_data['Date'].nunique())

        # Accuracy
        correct = (regime_data['pred'] == regime_data['y_true']).sum()
        accuracy = 100.0 * correct / len(regime_data)

        # ROC-AUC
        try:
            auc = roc_auc_score(regime_data['y_true'], regime_data['prob_ens'])
        except Exception:
            auc = np.nan

        # Sharpe (compute from strategy returns)
        if 'daily_return' in regime_data.columns:
            strat_returns = regime_data['daily_return'].astype(float) * regime_data['pred'].astype(float)
            returns = regime_data['daily_return'].astype(float)
        else:
            strat_returns = np.zeros(len(regime_data))
            returns = np.zeros(len(regime_data))

        mean_strat = strat_returns.mean()
        std_strat = strat_returns.std(ddof=1)
        if std_strat > 0 and pd.notna(std_strat):
            sharpe = (mean_strat * 252.0) / (std_strat * np.sqrt(252.0))
        else:
            sharpe = np.nan

        # Signals
        signals = int((regime_data['pred'] == 1).sum())

        # Avg daily return (%)
        mean_ret = returns.mean()
        avg_daily_ret = mean_ret * 100.0 if pd.notna(mean_ret) else np.nan

        rows.append({
            'Regime': regime,
            'Days': days,
            'Accuracy (%)': accuracy,
            'ROC-AUC': auc,
            'Sharpe': sharpe,
            'Signals': signals,
            'Avg Daily Return (%)': avg_daily_ret
        })

    # Overall
    overall_data = merged.dropna(subset=['pred', 'y_true'])
    overall_days = int(overall_data['Date'].nunique())
    overall_acc = 100.0 * (overall_data['pred'] == overall_data['y_true']).sum() / len(overall_data)
    try:
        overall_auc = roc_auc_score(overall_data['y_true'], overall_data['prob_ens'])
    except Exception:
        overall_auc = np.nan

    if 'daily_return' in overall_data.columns:
        overall_strat_returns = overall_data['daily_return'].astype(float) * overall_data['pred'].astype(float)
        overall_returns = overall_data['daily_return'].astype(float)
    else:
        overall_strat_returns = np.zeros(len(overall_data))
        overall_returns = np.zeros(len(overall_data))

    overall_mean_strat = overall_strat_returns.mean()
    overall_std_strat = overall_strat_returns.std(ddof=1)
    overall_sharpe = (overall_mean_strat * 252.0) / (overall_std_strat * np.sqrt(252.0)) if overall_std_strat > 0 and pd.notna(overall_std_strat) else np.nan
    overall_signals = int((overall_data['pred'] == 1).sum())
    overall_mean = overall_returns.mean()
    overall_avg_ret = overall_mean * 100.0 if pd.notna(overall_mean) else np.nan

    rows.append({
        'Regime': 'Overall (All Regimes)',
        'Days': overall_days,
        'Accuracy (%)': overall_acc,
        'ROC-AUC': overall_auc,
        'Sharpe': overall_sharpe,
        'Signals': overall_signals,
        'Avg Daily Return (%)': overall_avg_ret
    })

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame):
    """Save as CSV."""
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'Saved regime performance table to {OUT_CSV}')


def save_markdown(df: pd.DataFrame):
    """Save as markdown with formatting."""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    # Find best in each column (excluding Overall row)
    df_numeric = df[df['Regime'] != 'Overall (All Regimes)'].copy()
    best_acc = df_numeric['Accuracy (%)'].idxmax() if not df_numeric['Accuracy (%)'].isna().all() else -1
    best_auc = df_numeric['ROC-AUC'].idxmax() if not df_numeric['ROC-AUC'].isna().all() else -1
    best_sharpe = df_numeric['Sharpe'].idxmax() if not df_numeric['Sharpe'].isna().all() else -1
    best_ret = df_numeric['Avg Daily Return (%)'].idxmax() if not df_numeric['Avg Daily Return (%)'].isna().all() else -1

    md_lines = [
        '# Regime Performance Breakdown',
        '',
        '| Regime | Days | Accuracy (%) | ROC-AUC | Sharpe | Signals | Avg Daily Return (%) |',
        '|--------|------|--------------|---------|--------|---------|----------------------|'
    ]

    for idx, row in df.iterrows():
        regime = row['Regime']
        days = int(row['Days']) if pd.notna(row['Days']) else 0
        acc = row['Accuracy (%)']
        auc = row['ROC-AUC']
        sharpe = row['Sharpe']
        signals = int(row['Signals']) if pd.notna(row['Signals']) else 0
        ret = row['Avg Daily Return (%)']

        # Format values
        acc_str = f"{acc:.1f}%" if pd.notna(acc) else 'N/A'
        auc_str = f"{auc:.2f}" if pd.notna(auc) else 'N/A'
        sharpe_str = f"{sharpe:.2f}" if pd.notna(sharpe) else 'N/A'
        ret_str = f"{ret:.2f}%" if pd.notna(ret) else 'N/A'

        # Bold best performers (only for non-Overall rows)
        if idx == best_acc and idx != len(df) - 1:
            acc_str = f"**{acc_str}**"
        if idx == best_auc and idx != len(df) - 1:
            auc_str = f"**{auc_str}**"
        if idx == best_sharpe and idx != len(df) - 1:
            sharpe_str = f"**{sharpe_str}**"
        if idx == best_ret and idx != len(df) - 1:
            ret_str = f"**{ret_str}**"

        md_lines.append(f"| {regime} | {days:,} | {acc_str} | {auc_str} | {sharpe_str} | {signals:,} | {ret_str} |")

    md_lines.extend([
        '',
        '*Regime definitions based on S&P 500 price relative to 200-day SMA, momentum, and ATR volatility.*'
    ])

    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f'Saved markdown version to {OUT_MD}')


def main():
    print('Loading predictions...')
    preds = pd.read_csv(PRED_CSV)
    preds.columns = [c.strip() for c in preds.columns]

    print('Loading S&P 500 data...')
    sp500 = load_sp500()

    print('Assigning regimes...')
    sp500_regimes = assign_regimes(sp500)

    print('Computing regime metrics...')
    results = compute_regime_metrics(preds, sp500_regimes)

    save_csv(results)
    save_markdown(results)

    print('\n=== Regime Performance Summary ===')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
