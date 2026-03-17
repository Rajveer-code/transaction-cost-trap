from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / 'results' / 'predictions_ensemble.csv'
RAW_PRICES = ROOT / 'data' / 'extended' / 'all_free_data_2015_2025.parquet'
OUT_PNG = ROOT / 'results' / 'ticker_performance_heatmap.png'
OUT_CSV = ROOT / 'results' / 'ticker_performance_summary.csv'

TICKERS = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA']


def load_predictions():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f'Predictions file not found: {PRED_CSV}')
    df = pd.read_csv(PRED_CSV)
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date':'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    return df


def ensure_daily_returns(preds: pd.DataFrame):
    # If preds already has daily_return, use it; else compute from raw prices
    if 'daily_return' in preds.columns:
        return preds

    if not RAW_PRICES.exists():
        raise FileNotFoundError(f'Raw prices missing and preds lacks daily_return: {RAW_PRICES}')

    raw = pd.read_parquet(RAW_PRICES)
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values(['Ticker','Date']).reset_index(drop=True)
    raw['daily_return'] = raw.groupby('Ticker')['Close'].pct_change()
    raw['Ticker'] = raw['Ticker'].astype(str).str.strip()

    merged = pd.merge(preds, raw[['Date','Ticker','daily_return']], on=['Date','Ticker'], how='left')
    return merged


def compute_metrics(df: pd.DataFrame):
    rows = []
    for t in TICKERS:
        d = df[df['Ticker'] == t].copy()
        if d.empty:
            rows.append({'Ticker': t, 'Win Rate (%)': np.nan, 'Avg Return (%)': np.nan, 'Sharpe': np.nan, 'Times Traded': 0})
            continue

        # determine signal
        if 'y_pred' in d.columns:
            d['signal'] = (d['y_pred'].astype(float) == 1)
        else:
            d['signal'] = (d.get('prob_ens', 0) > 0.5)

        # correctness
        if 'y_true' in d.columns and 'y_pred' in d.columns:
            valid = d.dropna(subset=['y_true','y_pred'])
            correct = (valid['y_true'] == valid['y_pred']).sum()
            total = len(valid)
            win_rate = 100.0 * correct / total if total>0 else np.nan
        else:
            win_rate = np.nan

        # avg return when signaled (in percent)
        if 'daily_return' in d.columns:
            strat_returns = d['daily_return'] * d['signal'].astype(float)
            # average on signaled days
            if d['signal'].sum() > 0:
                avg_ret = d.loc[d['signal'],'daily_return'].mean() * 100.0
            else:
                avg_ret = 0.0
            # Sharpe computed on strategy returns (including zeros)
            mean = strat_returns.mean()
            std = strat_returns.std(ddof=1)
            if std == 0 or np.isnan(std):
                sharpe = np.nan
            else:
                sharpe = (mean * 252.0) / (std * np.sqrt(252.0))
        else:
            avg_ret = np.nan
            sharpe = np.nan

        times_traded = int(d['signal'].sum()) if 'signal' in d.columns else 0

        rows.append({'Ticker': t, 'Win Rate (%)': win_rate, 'Avg Return (%)': avg_ret, 'Sharpe': sharpe, 'Times Traded': times_traded})

    out = pd.DataFrame(rows).set_index('Ticker')
    # portfolio average
    avg = out.mean(numeric_only=True)
    avg_row = pd.DataFrame(avg).T
    avg_row.index = ['Portfolio Average']
    result = pd.concat([out, avg_row])
    return result


def plot_heatmap(df: pd.DataFrame):
    # df: index tickers + Portfolio Average row at bottom
    labels = ['Win Rate (%)','Avg Return (%)','Sharpe','Times Traded']
    data = df[labels].copy()

    # Normalize for colormap per column
    norm_data = data.copy()
    for col in labels:
        col_vals = data[col].astype(float)
        vmin = np.nanmin(col_vals)
        vmax = np.nanmax(col_vals)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax>vmin:
            norm_data[col] = (col_vals - vmin) / (vmax - vmin)
        else:
            norm_data[col] = 0.5

    arr = norm_data.values

    fig, ax = plt.subplots(figsize=(10,6))

    cmap = plt.get_cmap('RdYlGn')
    im = ax.imshow(arr, aspect='auto', cmap=cmap, interpolation='nearest')

    # Annotate
    nrows, ncols = arr.shape
    for i in range(nrows):
        for j in range(ncols):
            val = data.iloc[i,j]
            txt = f"{val:.2f}" if pd.notna(val) else 'NA'
            ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=10)

    # Row/col ticks
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(df.index.tolist(), fontsize=10)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_title('Per-Ticker Performance Breakdown (2015-2025)')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label('Normalized (per-column)')

    # Bold borders for best/worst by Win Rate
    win_rates = df['Win Rate (%)'].iloc[:-1]  # exclude portfolio avg
    best_idx = int(np.nanargmax(win_rates.values))
    worst_idx = int(np.nanargmin(win_rates.values))

    # draw rectangles around best and worst rows
    def draw_row_border(row_idx, color='black'):
        rect = patches.Rectangle((-0.5, row_idx-0.5), ncols, 1, linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    draw_row_border(best_idx, color='green')
    draw_row_border(worst_idx, color='red')

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('Saved heatmap to', OUT_PNG)


def main():
    preds = load_predictions()
    merged = ensure_daily_returns(preds)
    metrics = compute_metrics(merged)
    metrics.to_csv(OUT_CSV)
    plot_heatmap(metrics)


if __name__ == '__main__':
    main()
