from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / 'results' / 'transaction_cost_analysis.csv'
CSV_OUT = ROOT / 'results' / 'transaction_cost_analysis.csv'
OUT_PNG = ROOT / 'results' / 'transaction_cost_sensitivity.png'


def compute_table():
    # replicate the calculation used elsewhere
    from scripts import transaction_cost_sensitivity as tcs

    # constants mirrored from tcs
    TOTAL_TRADES = getattr(tcs, 'TOTAL_TRADES', 4711)
    YEARS = getattr(tcs, 'YEARS', 10)
    ANNUAL_TRADES = getattr(tcs, 'ANNUAL_TRADES', 471)
    costs_bps = [0, 5, 10, 15, 20, 25]
    vol = getattr(tcs, 'vol', 0.15)

    final_strategy_wealth, _ = tcs.compute_final_wealth_from_predictions()
    base_annual_return = final_strategy_wealth ** (1.0 / YEARS) - 1.0

    rows = []
    for c in costs_bps:
        annual_cost_pct = (ANNUAL_TRADES * 2 * c) / 10000.0
        net_return = base_annual_return - annual_cost_pct
        sharpe = net_return / vol if vol > 0 else np.nan
        rows.append({'cost_bps': c, 'annual_return': net_return * 100.0, 'sharpe_ratio': sharpe})

    df = pd.DataFrame(rows)
    return df


def load_or_compute():
    if CSV_IN.exists():
        df = pd.read_csv(CSV_IN)
        expected = set(['cost_bps','annual_return','sharpe_ratio'])
        if not expected.issubset(set(df.columns)):
            raise RuntimeError(f'Found {CSV_IN} but missing required columns: {df.columns.tolist()}')
        return df.sort_values('cost_bps')
    df = compute_table()
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    return df


def plot_df(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12,6))

    x = df['cost_bps']
    y_return = df['annual_return']
    y_sharpe = df['sharpe_ratio']

    color_ret = 'tab:blue'
    color_sh = 'tab:orange'

    ax1.set_xlabel('Transaction cost (bps)')
    ax1.set_ylabel('Annual Return (%)', color=color_ret)
    ax1.plot(x, y_return, color=color_ret, marker='o', label='Annual Return')
    ax1.tick_params(axis='y', labelcolor=color_ret)
    ax1.axhline(0.0, color='red', linestyle='--', linewidth=1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Sharpe Ratio', color=color_sh)
    ax2.plot(x, y_sharpe, color=color_sh, marker='s', label='Sharpe')
    ax2.tick_params(axis='y', labelcolor=color_sh)
    ax2.axhline(1.22, color='gray', linestyle='--', linewidth=1)

    # Annotations
    # breakeven cost where annual_return crosses 0 (interpolate)
    try:
        breakeven = float(np.interp(0.0, y_return[::-1], x[::-1]))
    except Exception:
        # fallback: first where return <=0
        be_idx = df[df['annual_return'] <= 0]
        breakeven = float(be_idx['cost_bps'].iloc[0]) if not be_idx.empty else None

    if breakeven is not None:
        ax1.annotate(f'Breakeven ≈ {breakeven:.1f} bps', xy=(breakeven, 0.0), xytext=(breakeven, max(y_return)*0.3),
                     arrowprops=dict(arrowstyle='->', color='black'))

    # cost where Sharpe < benchmark
    benchmark = 1.22
    below = df[df['sharpe_ratio'] < benchmark]
    first_below = int(below['cost_bps'].iloc[0]) if not below.empty else None
    if first_below is not None:
        ax2.annotate(f'Sharpe < {benchmark} at {first_below} bps', xy=(first_below, df[df['cost_bps']==first_below]['sharpe_ratio'].iloc[0]),
                     xytext=(first_below, min(y_sharpe)-0.2), arrowprops=dict(arrowstyle='->', color='black'))

    # assumed cost vertical line at 5 bps
    assumed = 5
    ax1.axvline(assumed, color='green', linestyle='-', linewidth=1)
    ax1.text(assumed, ax1.get_ylim()[1]*0.95, f'Assumed {assumed} bps', color='green', ha='center', va='top')

    # summary table below chart
    table_data = []
    for _, row in df.iterrows():
        profitable = 'Yes' if row['annual_return'] > 0 else 'No'
        table_data.append([int(row['cost_bps']), f"{row['annual_return']:.2f}%", f"{row['sharpe_ratio']:.2f}", profitable])

    table = plt.table(cellText=table_data, colLabels=['Cost (bps)','Annual Return','Sharpe','Profitable?'],
                      cellLoc='center', loc='bottom', bbox=[0.0, -0.35, 1.0, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.title('Transaction Cost Sensitivity: Strategy Profitability vs Execution Costs')
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('Saved transaction cost sensitivity plot to', OUT_PNG)


def main():
    df = load_or_compute()
    plot_df(df)


if __name__ == '__main__':
    main()
