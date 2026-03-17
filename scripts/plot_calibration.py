"""Calibration plot for ensemble predictions.

Loads `results/predictions_ensemble.csv` and produces a calibration plot and
summary CSV. Requirements implemented:
- 10 decile bins [0.0,0.1),...,[0.9,1.0]
- Per-bin mean predicted probability, realized fraction, 95% CI (Wilson)
- Brier score and ECE
- Save plot `results/calibration_plot.png` (300 DPI)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / 'results' / 'predictions_ensemble.csv'
OUT_PLOT = ROOT / 'results' / 'calibration_plot.png'
OUT_SUM = ROOT / 'results' / 'calibration_summary.csv'


def wilson_interval(k, n, alpha=0.05):
    """Wilson score interval for proportion k/n.
    Returns (lower, upper). If n==0 returns (nan,nan).
    """
    if n == 0:
        return (np.nan, np.nan)
    z = -1 * np.quantile([0,1], 0.975) if False else 1.96
    # use z=1.96 for 95% CI
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def compute_calibration(df, prob_col='ensemble_prob', label_col='actual_label'):
    probs = df[prob_col].astype(float).values
    labels = df[label_col].astype(int).values

    bins = np.linspace(0.0, 1.0, 11)  # 0.0..1.0 step 0.1
    # Use right=False so intervals are [0.0,0.1),..., [0.9,1.0]
    inds = np.digitize(probs, bins[:-1], right=False)
    # digitize returns 1..10 for our bins

    rows = []
    N = len(probs)
    for b in range(1, 11):
        mask = inds == b
        n = int(mask.sum())
        if n == 0:
            mean_prob = np.nan
            frac_pos = np.nan
            lo = np.nan
            hi = np.nan
        else:
            mean_prob = float(np.mean(probs[mask]))
            k = int(labels[mask].sum())
            frac_pos = float(k / n)
            lo, hi = wilson_interval(k, n)
        rows.append({'bin': b, 'count': n, 'mean_pred': mean_prob, 'frac_pos': frac_pos, 'ci_lower': lo, 'ci_upper': hi})

    summary = pd.DataFrame(rows)

    # Brier score
    brier = float(np.mean((probs - labels) ** 2))

    # Expected Calibration Error (ECE): weighted avg of |mean_pred - frac_pos|
    ece = 0.0
    for _, r in summary.iterrows():
        if np.isnan(r['mean_pred']) or np.isnan(r['frac_pos']) or r['count'] == 0:
            continue
        ece += (r['count'] / N) * abs(r['mean_pred'] - r['frac_pos'])

    return summary, brier, ece


def plot_calibration(summary, brier, ece):
    fig, ax = plt.subplots(figsize=(10, 10))

    # remove bins with nan
    valid = summary.dropna(subset=['mean_pred', 'frac_pos'])

    x = valid['mean_pred'].values
    y = valid['frac_pos'].values
    lower = valid['ci_lower'].values
    upper = valid['ci_upper'].values
    counts = valid['count'].values

    # diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    # calibration curve
    ax.errorbar(x, y, yerr=[y - lower, upper - y], fmt='-o', color='blue', capsize=4, label='Model')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Plot: Predicted vs Realized Probabilities')
    ax.grid(True)

    # metrics box
    text_lines = [f'Brier score: {brier:.4f}', f'ECE: {ece:.4f}']
    bbox = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=bbox)

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print('Saved calibration plot to', OUT_PLOT)


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_CSV}")
    df = pd.read_csv(PRED_CSV)
    # Accept alternate column names
    if 'ensemble_prob' not in df.columns:
        # try common alternatives
        # accept 'prob_ens', 'prob_ensemble', or any column containing 'prob' and ('ens'|'ensemble')
        alt = [c for c in df.columns if ('prob' in c.lower() and ('ens' in c.lower() or 'ensemble' in c.lower()))]
        if 'prob_ens' in df.columns:
            df = df.rename(columns={'prob_ens': 'ensemble_prob'})
        elif alt:
            df = df.rename(columns={alt[0]: 'ensemble_prob'})
        else:
            raise RuntimeError('Could not find `ensemble_prob` column in predictions CSV')
    if 'actual_label' not in df.columns:
        # accept 'y_true', 'actual', 'label', 'y'
        if 'y_true' in df.columns:
            df = df.rename(columns={'y_true': 'actual_label'})
        else:
            alt = [c for c in df.columns if c.lower() in ('actual', 'label', 'y')]
            if alt:
                df = df.rename(columns={alt[0]: 'actual_label'})
            else:
                raise RuntimeError('Could not find `actual_label` column in predictions CSV')

    summary, brier, ece = compute_calibration(df, prob_col='ensemble_prob', label_col='actual_label')
    summary.to_csv(OUT_SUM, index=False)
    print('Saved calibration summary to', OUT_SUM)
    plot_calibration(summary, brier, ece)


if __name__ == '__main__':
    main()
