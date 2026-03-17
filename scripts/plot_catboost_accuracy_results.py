"""Plot aggregated TRAIN vs VALIDATION accuracy from saved CatBoost fold histories.

Reads per-fold JSONs from `catboost_info/fold_histories/`, aggregates mean train/val
accuracy per iteration, shades ±1 std for validation, and plots green dots for
fold test accuracies. Failing folds (test_acc < 0.45) are marked as red X.

Saves:
- `results/learning_curves_catboost_accuracy.png` (DPI=300)
- `results/learning_curves_catboost_accuracy_summary.csv`
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / 'catboost_info' / 'fold_histories'
OUT_PLOT = ROOT / 'results' / 'learning_curves_catboost_accuracy.png'
OUT_SUM = ROOT / 'results' / 'learning_curves_catboost_accuracy_summary.csv'

def load_histories(hist_dir):
    files = sorted(hist_dir.glob('*.json'))
    entries = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            entries.append({'file': f.name, 'train_acc': d.get('train_acc'), 'val_acc': d.get('val_acc'), 'test_acc': d.get('test_acc'), 'final_iter': d.get('final_iter')})
        except Exception:
            continue
    return entries


def aggregate(entries):
    # determine max iterations
    max_iter = 0
    for e in entries:
        if e['train_acc']:
            max_iter = max(max_iter, len(e['train_acc']))
        if e['val_acc']:
            max_iter = max(max_iter, len(e['val_acc']))
    if max_iter == 0:
        raise RuntimeError('No per-iteration data found')

    n = len(entries)
    train_matrix = np.full((n, max_iter), np.nan)
    val_matrix = np.full((n, max_iter), np.nan)

    for i, e in enumerate(entries):
        if e['train_acc']:
            arr = np.array(e['train_acc'], dtype=float)
            train_matrix[i, :len(arr)] = arr
        if e['val_acc']:
            arr = np.array(e['val_acc'], dtype=float)
            val_matrix[i, :len(arr)] = arr

    train_mean = np.nanmean(train_matrix, axis=0)
    val_mean = np.nanmean(val_matrix, axis=0)
    val_std = np.nanstd(val_matrix, axis=0)

    return train_mean, val_mean, val_std, train_matrix, val_matrix


def plot_and_save(entries, train_mean, val_mean, val_std):
    max_iter = len(train_mean)
    iters = np.arange(max_iter)

    # classify folds
    good = [e for e in entries if isinstance(e['test_acc'], (int, float)) and e['test_acc'] >= 0.45]
    fail = [e for e in entries if isinstance(e['test_acc'], (int, float)) and e['test_acc'] < 0.45]

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iters, train_mean, color='blue', linewidth=2, label='Train')
    ax.plot(iters, val_mean, color='orange', linewidth=2, label='Validation')
    ax.fill_between(iters, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random Baseline')

    # green dots for good folds
    if good:
        xs = [min(int(e['final_iter']) if e['final_iter'] is not None else max_iter-1, max_iter-1) for e in good]
        ys = [e['test_acc'] for e in good]
        ax.scatter(xs, ys, color='green', s=40, label='Test (good)')

    # red X for failing folds
    if fail:
        xs_f = [min(int(e['final_iter']) if e['final_iter'] is not None else max_iter-1, max_iter-1) for e in fail]
        ys_f = [e['test_acc'] for e in fail]
        ax.scatter(xs_f, ys_f, marker='x', color='red', s=60, label='Test (failing)')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves: CatBoost Accuracy (Walk-Forward Validation)')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # add a small summary box
    total = len(entries)
    num_fail = len(fail)
    test_accs = [e['test_acc'] for e in entries if isinstance(e['test_acc'], (int, float))]
    summary_lines = [f'Total folds: {total}', f'Failing folds (<0.45): {num_fail}', f'Mean test acc: {np.nanmean(test_accs):.3f}', f'Min test acc: {np.nanmin(test_accs):.3f}', f'Max test acc: {np.nanmax(test_accs):.3f}']
    bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.98, '\n'.join(summary_lines), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print('Saved plot to', OUT_PLOT)

    # save CSV summary
    rows = []
    for e in entries:
        rows.append({'file': e['file'], 'test_acc': e['test_acc'], 'final_iter': e['final_iter'], 'status': 'fail' if isinstance(e['test_acc'], (int,float)) and e['test_acc']<0.45 else 'good'})
    pd.DataFrame(rows).to_csv(OUT_SUM, index=False)
    print('Saved summary CSV to', OUT_SUM)


def main():
    entries = load_histories(HIST_DIR)
    train_mean, val_mean, val_std, _, _ = aggregate(entries)
    plot_and_save(entries, train_mean, val_mean, val_std)


if __name__ == '__main__':
    main()
