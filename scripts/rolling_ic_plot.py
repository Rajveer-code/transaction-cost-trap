"""
rolling_ic_plot.py
==================
Generates rolling Information Coefficient plot for the paper.
Shows IC stability over time to address the marginal IC concern.

Run: python scripts/rolling_ic_plot.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr

PARQUET_PATH = Path("results/predictions/predictions_ensemble.parquet")
OUTPUT_DIR   = Path("results/figures")

def compute_daily_ic(df: pd.DataFrame) -> pd.Series:
    """Compute Spearman IC for each trading day."""
    daily_ic = {}
    for date, day in df.groupby(level="date"):
        probs   = day["prob"].values
        returns = day["actual_return"].values
        if np.var(probs) < 1e-12 or np.var(returns) < 1e-12:
            daily_ic[date] = 0.0
            continue
        corr, _ = spearmanr(probs, returns)
        daily_ic[date] = float(corr) if not np.isnan(corr) else 0.0
    return pd.Series(daily_ic).sort_index()

def plot_rolling_ic(daily_ic: pd.Series, output_dir: Path) -> None:
    """Generate publication-quality rolling IC figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rolling windows
    ic_63d  = daily_ic.rolling(63,  min_periods=21).mean()
    ic_126d = daily_ic.rolling(126, min_periods=42).mean()
    cumulative_mean = daily_ic.expanding().mean()

    # Overall stats
    mean_ic = daily_ic.mean()
    from scipy.stats import ttest_1samp
    t_stat, p_two = ttest_1samp(daily_ic.dropna(), 0)
    p_one = p_two / 2

    # Style
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.linewidth":    0.6,
        "grid.alpha":        0.4,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        f"Rolling Information Coefficient Stability (Mean IC = {mean_ic:.4f}, "
        f"p = {p_one:.3f}, N = {len(daily_ic)} days)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # ── Top panel: rolling IC ────────────────────────────────────────────
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax1.axhline(mean_ic, color="#4DAC26", linewidth=1.5, linestyle="--",
                label=f"Overall mean IC = {mean_ic:.4f} (p = {p_one:.3f})")

    ax1.fill_between(daily_ic.index, 0, ic_63d,
                     where=(ic_63d >= 0), alpha=0.15, color="#2166AC",
                     label="Positive 63d IC")
    ax1.fill_between(daily_ic.index, 0, ic_63d,
                     where=(ic_63d < 0),  alpha=0.15, color="#D7191C",
                     label="Negative 63d IC")

    ax1.plot(ic_63d.index,  ic_63d,
             color="#2166AC", linewidth=1.8, label="63-day rolling IC")
    ax1.plot(ic_126d.index, ic_126d,
             color="#762A83", linewidth=1.5, linestyle="--",
             label="126-day rolling IC")

    # Shade sub-periods
    periods = [
        ("2018-10-19", "2018-12-31", "#D7191C", "Period 1 start"),
        ("2019-01-01", "2021-12-31", "#4DAC26", "COVID/Growth"),
        ("2022-01-01", "2024-10-23", "#4575B4", "Rate Shock"),
    ]
    for start, end, color, label in periods:
        ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.04, color=color)

    ax1.set_ylabel("Spearman IC (Cross-Sectional)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, axis="y", linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Annotate COVID period
    ax1.annotate("COVID/Growth\n(IC concentrated here)",
                 xy=(pd.Timestamp("2020-06-01"), ic_63d.loc["2020-06-01"]),
                 xytext=(pd.Timestamp("2021-01-01"), 0.25),
                 fontsize=8, color="#4DAC26",
                 arrowprops=dict(arrowstyle="->", color="#4DAC26", lw=1))

    # ── Bottom panel: cumulative IC ──────────────────────────────────────
    ax2.plot(cumulative_mean.index, cumulative_mean,
             color="#2166AC", linewidth=2, label="Cumulative mean IC")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(mean_ic, color="#4DAC26", linewidth=1.2,
                linestyle="--", alpha=0.7)
    ax2.fill_between(cumulative_mean.index, 0, cumulative_mean,
                     where=(cumulative_mean >= 0), alpha=0.2, color="#2166AC")
    ax2.set_ylabel("Cumulative Mean IC")
    ax2.set_xlabel("Date")
    ax2.grid(True, axis="y", linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.legend(loc="lower right")

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = output_dir / f"fig7_rolling_ic.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)

    # Print summary stats for paper
    n_positive = (ic_63d.dropna() > 0).sum()
    n_total    = ic_63d.dropna().count()
    print(f"\n  63-day rolling windows positive: {n_positive}/{n_total} "
          f"({100*n_positive/n_total:.0f}%)")
    pct_positive_days = (daily_ic > 0).mean()
    print(f"  Fraction of individual days IC > 0: {pct_positive_days:.1%}")
    print(f"\n  LaTeX caption:")
    print(f'  \\caption{{Rolling 63-day (solid) and 126-day (dashed) '
          f'Information Coefficient over the out-of-sample period '
          f'(2018--2024). The bottom panel shows the cumulative mean IC '
          f'converging toward the overall mean of {mean_ic:.4f} '
          f'($p = {p_one:.3f}$, $N = {len(daily_ic)}$ days). '
          f'Shaded regions denote sub-period regimes. '
          f'The positive IC concentration during the COVID/Growth regime '
          f'(2019--2021) is consistent with the regime-conditional '
          f'performance documented in Table 2.}}')

if __name__ == "__main__":
    print("Loading predictions...")
    df = pd.read_parquet(PARQUET_PATH)

    print("Computing daily IC values...")
    daily_ic = compute_daily_ic(df)
    print(f"  {len(daily_ic)} daily IC values computed")

    print("Generating rolling IC plot...")
    plot_rolling_ic(daily_ic, OUTPUT_DIR)

    # Save daily IC values
    daily_ic.to_csv(Path("results/metrics") / "daily_ic_values.csv",
                    header=True, index=True)
    print("\n[DONE] Figure saved as results/figures/fig7_rolling_ic.png")