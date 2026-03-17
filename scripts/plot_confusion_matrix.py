from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / 'results' / 'predictions_ensemble.csv'
OUT_PNG = ROOT / 'results' / 'confusion_matrix.png'
OUT_CSV = ROOT / 'results' / 'confusion_matrix_metrics.csv'


def load_and_compute_cm():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f'Predictions file not found: {PRED_CSV}')

    df = pd.read_csv(PRED_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns
    if 'prob_ens' not in df.columns:
        raise RuntimeError('prob_ens not found in predictions')
    if 'y_true' not in df.columns:
        raise RuntimeError('y_true not found in predictions')

    # predicted_label = 1 if prob_ens > 0.5, else 0
    df['predicted_label'] = (df['prob_ens'] > 0.5).astype(int)
    df['actual_label'] = df['y_true'].astype(int)

    # Remove rows with NaN in key columns
    df = df.dropna(subset=['predicted_label', 'actual_label'])

    # Compute confusion matrix
    tp = ((df['predicted_label'] == 1) & (df['actual_label'] == 1)).sum()
    fp = ((df['predicted_label'] == 1) & (df['actual_label'] == 0)).sum()
    fn = ((df['predicted_label'] == 0) & (df['actual_label'] == 1)).sum()
    tn = ((df['predicted_label'] == 0) & (df['actual_label'] == 0)).sum()

    total = tp + fp + fn + tn

    # Metrics
    accuracy = (tp + tn) / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'total': int(total),
        'accuracy': float(accuracy), 'precision': float(precision),
        'recall': float(recall), 'f1': float(f1)
    }


def plot_cm(cm_dict):
    tp = cm_dict['tp']
    fp = cm_dict['fp']
    fn = cm_dict['fn']
    tn = cm_dict['tn']
    total = cm_dict['total']

    # Build row-normalized matrix (percentage per actual class)
    # Row 0: Actual Down (0) -> [TN%, FP%]
    # Row 1: Actual Up (1) -> [FN%, TP%]
    actual_down_total = tn + fp
    actual_up_total = fn + tp

    matrix = np.array([
        [100.0 * tn / actual_down_total, 100.0 * fp / actual_down_total],
        [100.0 * fn / actual_up_total, 100.0 * tp / actual_up_total]
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Heatmap using imshow
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=100)

    # Annotate cells with row-normalized percentages only
    for i in range(2):
        for j in range(2):
            pct = matrix[i, j]
            text = f'{pct:.1f}%'
            ax.text(j, i, text, ha='center', va='center', color='black',
                   fontsize=16, fontweight='bold')

    # Ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Down (0)', 'Predicted Up (1)'], fontsize=11)
    ax.set_yticklabels(['Actual Down (0)', 'Actual Up (1)'], fontsize=11)

    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Ensemble Predictions (All Tickers, 2015-2025)', fontsize=13, fontweight='bold')

    fig.colorbar(im, ax=ax, label='Row Percentage (%)')

    # Caption with metrics below figure
    caption = (
        f"Row-normalized percentages. Precision: {cm_dict['precision']:.1%} | "
        f"Recall: {cm_dict['recall']:.1%} | F1: {cm_dict['f1']:.3f}"
    )
    fig.text(0.5, 0.02, caption, ha='center', fontsize=10, style='italic')

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print(f'Saved confusion matrix to {OUT_PNG}')


def main():
    cm_dict = load_and_compute_cm()
    plot_cm(cm_dict)

    # Save metrics CSV
    metrics_df = pd.DataFrame([cm_dict])
    metrics_df.to_csv(OUT_CSV, index=False)
    print(f'Saved metrics to {OUT_CSV}')

    # Print summary
    print('\n=== Confusion Matrix Summary ===')
    print(f"TP (True Positives):   {cm_dict['tp']:6d}")
    print(f"FP (False Positives):  {cm_dict['fp']:6d}")
    print(f"FN (False Negatives):  {cm_dict['fn']:6d}")
    print(f"TN (True Negatives):   {cm_dict['tn']:6d}")
    print(f"Total:                 {cm_dict['total']:6d}")
    print('\n=== Performance Metrics ===')
    print(f"Precision: {cm_dict['precision']:.4f} ({cm_dict['precision']*100:.2f}%)")
    print(f"Recall:    {cm_dict['recall']:.4f} ({cm_dict['recall']*100:.2f}%)")
    print(f"F1-Score:  {cm_dict['f1']:.4f}")


if __name__ == '__main__':
    main()
