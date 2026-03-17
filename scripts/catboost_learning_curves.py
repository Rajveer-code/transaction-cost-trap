"""Plot CatBoost learning curves and overlay per-fold test accuracies.

Behavior:
- Searches the workspace for any `catboost_training.json` files (these contain per-iteration
  learn/test metrics saved by CatBoost). If multiple are found, each is plotted.
- Loads `results/walk_forward_results.csv` and computes mean+std test accuracy per fold
  (across tickers). These per-fold test accuracies are shown as green dots (right axis).
- If no per-fold per-epoch histories are available, the script will still plot the
  available training log (if any) and the per-fold test accuracies.

This script is defensive and prints explanations about what's plotted and any limitations.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


ROOT = Path(__file__).resolve().parents[1]
OUT_PLOT = ROOT / 'results' / 'catboost_learning_curves.png'
WALK_RESULTS = ROOT / 'results' / 'walk_forward_results.csv'


def find_catboost_jsons(root: Path):
    return list(root.rglob('catboost_training.json'))


def load_catboost_json(p: Path):
    try:
        with open(p, 'r', encoding='utf-8') as f:
            j = json.load(f)
        iterations = j.get('iterations', [])
        iters = [int(x.get('iteration')) for x in iterations if 'iteration' in x]
        # extract the first learn metric value if present
        learn_vals = []
        for x in iterations:
            lv = x.get('learn')
            if isinstance(lv, list) and len(lv) > 0:
                learn_vals.append(float(lv[0]))
            elif lv is None:
                learn_vals.append(math.nan)
            else:
                learn_vals.append(float(lv))

        return pd.DataFrame({'iteration': iters, 'learn': learn_vals})
    except Exception as e:
        print(f"Warning: failed to read {p}: {e}")
        return pd.DataFrame(columns=['iteration', 'learn'])


def main():
    print('Reading walk-forward results...')
    if not WALK_RESULTS.exists():
        print(f"ERROR: {WALK_RESULTS} not found. Cannot plot per-fold accuracies.")
        return

    wf = pd.read_csv(WALK_RESULTS)
    if 'fold' not in wf.columns or 'accuracy' not in wf.columns:
        print('ERROR: walk_forward_results.csv missing expected columns `fold` and `accuracy`.')
        return

    # compute mean+std accuracy per fold
    acc_summary = wf.groupby('fold')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    acc_summary = acc_summary.sort_values('fold')

    print('Searching for CatBoost training JSON files...')
    json_files = find_catboost_jsons(ROOT)
    print(f'Found {len(json_files)} catboost_training.json file(s).')

    curves = []
    for jf in json_files:
        df = load_catboost_json(jf)
        if not df.empty:
            curves.append((jf, df))

    # Prepare plot
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    if curves:
        max_iter = max(df['iteration'].max() for _, df in curves)
        for i, (jf, df) in enumerate(curves):
            label = jf.parent.name if jf.parent.name else jf.name
            ax1.plot(df['iteration'], df['learn'], label=f"{label}", color=colors[i % len(colors)], lw=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Learn metric (e.g. Logloss)')
        ax1.set_title('CatBoost Training Curve(s) and Per-Fold Test Accuracies')
        ax1.legend(loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'No CatBoost training JSON found', ha='center', va='center')
        max_iter = 100

    # Overlay per-fold mean test accuracies as green dots on a second y-axis
    ax2 = ax1.twinx()
    folds = acc_summary['fold'].values
    acc_means = acc_summary['mean'].values
    acc_stds = acc_summary['std'].fillna(0).values

    # Map fold indices to x positions spanning the iteration range for visual alignment
    if max_iter <= 0:
        xs = folds
    else:
        xs = np.linspace(0, max_iter, num=len(folds))

    ax2.errorbar(xs, acc_means, yerr=acc_stds, fmt='o', color='green', label='Per-fold test accuracy', markersize=8)
    for xi, f, a in zip(xs, folds, acc_means):
        ax2.annotate(f"fold {int(f)}\n{a:.3f}", (xi, a), textcoords='offset points', xytext=(0,8), ha='center', fontsize=8)

    ax2.set_ylabel('Test Accuracy (per-fold mean)', color='green')
    ax2.set_ylim(0, 1.0)

    # Legend for accuracies
    ax2.get_legend()

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved learning curve + per-fold accuracies to {OUT_PLOT}")

    # Print summary and limitations
    print('\nSummary:')
    print(f" - Plotted {len(curves)} CatBoost training curve(s).")
    print(f" - Plotted per-fold mean test accuracy for {len(folds)} folds (from {WALK_RESULTS}).")
    if not curves:
        msg = (
            "\nNote: No per-fold per-epoch training histories were found. If you trained CatBoost separately "
            "and saved learning curves per fold, place their `catboost_training.json` files somewhere under the "
            "project root (e.g. `catboost_info/<run_name>/catboost_training.json`) and re-run this script to "
            "plot per-fold learning curves (train metric per iteration)."
        )
        print(msg)


if __name__ == '__main__':
    main()
