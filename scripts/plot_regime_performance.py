"""Plot regime-specific performance breakdown.

Loads `results/regime_stats.csv` and creates a grouped bar chart of
win rates per regime with annotations for number of trading days and a
summary table below the chart. Saves to `results/regime_performance.png`.

If the input CSV doesn't contain a `signals` column, the script uses
`total_days` as a proxy for `Signals`.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / 'results' / 'regime_stats.csv'
OUT_PNG = ROOT / 'results' / 'regime_performance.png'


def load_regime_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Regime stats CSV not found: {path}")
    df = pd.read_csv(path)
    # Accept multiple possible column schemas; normalize to expected names
    cols = {c.lower(): c for c in df.columns}
    # Preferred schema: regime, win_rate, total_days, sharpe_ratio
    if {'regime', 'win_rate', 'total_days'}.issubset({c.lower() for c in df.columns}):
        # Normalize lowercase names to expected
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc == 'regime':
                rename_map[c] = 'regime'
            elif lc == 'win_rate':
                rename_map[c] = 'win_rate'
            elif lc == 'total_days':
                rename_map[c] = 'total_days'
            elif lc == 'sharpe_ratio':
                rename_map[c] = 'sharpe_ratio'
        df = df.rename(columns=rename_map)
        if 'sharpe_ratio' not in df.columns:
            df['sharpe_ratio'] = float('nan')
        return df

    # Alternate schema (existing file): Regime,N_Days,Baseline_Win%,Model_Win%,Improvement
    alt_map = {c.lower(): c for c in df.columns}
    if 'regime' in {c.lower() for c in df.columns} and 'model_win%' in {c.lower() for c in df.columns}:
        # Map to expected
        df = df.rename(columns={
            [c for c in df.columns if c.lower() == 'regime'][0]: 'regime',
            [c for c in df.columns if c.lower() in ('model_win%','model_win')][0]: 'win_rate',
            [c for c in df.columns if c.lower() in ('n_days','n_days','total_days')][0]: 'total_days'
        })
        # Ensure numeric types
        df['win_rate'] = pd.to_numeric(df['win_rate'], errors='coerce')
        df['total_days'] = pd.to_numeric(df['total_days'], errors='coerce')
        df['sharpe_ratio'] = float('nan')
        return df

    raise RuntimeError('CSV missing required columns and no known alternative schema was detected')


def to_pct(series: pd.Series) -> pd.Series:
    # Convert to percentage if values appear in [0,1]
    s = pd.to_numeric(series, errors='coerce')
    if s.max() <= 1.01:
        return s * 100.0
    return s


def color_for_winrate(pct: float) -> str:
    if pct > 55.0:
        return '#2ca02c'  # green
    if pct >= 50.0:
        return '#ffcc00'  # yellow
    return '#d62728'      # red


def main():
    df = load_regime_stats(CSV_PATH)

    df['win_rate_pct'] = to_pct(df['win_rate'])
    df['total_days'] = pd.to_numeric(df['total_days'], errors='coerce').fillna(0).astype(int)
    df['sharpe_ratio'] = pd.to_numeric(df['sharpe_ratio'], errors='coerce')
    # Signals: use column if present, otherwise fall back to total_days
    if 'signals' in df.columns:
        df['signals'] = pd.to_numeric(df['signals'], errors='coerce').fillna(0).astype(int)
    else:
        df['signals'] = df['total_days']

    regimes = df['regime'].astype(str).tolist()
    win_rates = df['win_rate_pct'].tolist()
    days = df['total_days'].tolist()
    sharpe = df['sharpe_ratio'].tolist()
    signals = df['signals'].tolist()

    colors = [color_for_winrate(w) for w in win_rates]

    # Figure with two rows: bar chart on top, table below
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    bars = ax.bar(regimes, win_rates, color=colors, edgecolor='black')
    ax.set_ylim(0, max(60, max(win_rates) * 1.15))
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Model Performance Across Market Regimes (2015-2025)')

    # horizontal baseline at 50%
    ax.axhline(50, color='black', linestyle='--', linewidth=1)

    # annotate number of trading days on each bar
    for bar, d in zip(bars, days):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{d}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Legend: color meaning
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color='#2ca02c', label='>55% (Good)'),
                      mpatches.Patch(color='#ffcc00', label='50-55% (Marginal)'),
                      mpatches.Patch(color='#d62728', label='<50% (Fail)')]
    ax.legend(handles=legend_patches, loc='upper right')

    # Summary table below
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    table_cols = ['Regime', 'Days', 'Win Rate', 'Sharpe', 'Signals']
    table_data = []
    for r, d, w, s, sig in zip(regimes, days, win_rates, sharpe, signals):
        table_data.append([r, int(d), f"{w:.1f}%", f"{s:.2f}" if pd.notna(s) else 'n/a', int(sig)])

    table = ax_table.table(cellText=table_data, colLabels=table_cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('Saved regime performance figure to', OUT_PNG)


if __name__ == '__main__':
    main()
