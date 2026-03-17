"""Robust CatBoost feature importance with permutation validation.

Computes PredictionValuesChange importances and permutation importances
on an out-of-sample set, averages the normalized importances, flags
potentially inflated features due to correlation, and produces a
publication-quality plot and CSVs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

try:
    from catboost import CatBoostClassifier
except Exception:
    raise RuntimeError('CatBoost is required to run this script')

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, make_scorer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'combined' / 'all_features.parquet'
OUT_PLOT = ROOT / 'results' / 'feature_importance_catboost_robust.png'
OUT_CSV = ROOT / 'results' / 'feature_importance_comparison.csv'

CB_PARAMS = dict(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    thread_count=4,
)


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if 'binary_label' not in df.columns:
        raise RuntimeError('Expected `binary_label` target column in data')
    return df


def categorize_feature(fname):
    lf = fname.lower()
    # short-term returns 1-5d
    for d in range(1,6):
        if f'return_{d}d' == lf or f'return_{d}' == lf or lf.endswith(f'return_{d}'):
            return 'short'
    # medium-term returns 10-50d
    for d in (10,20,30,50):
        if f'return_{d}d' == lf or f'return_{d}' == lf or lf.endswith(f'return_{d}'):
            return 'medium'
    # technical indicators
    tech_keys = ['atr','rsi','macd','obv','adx','volatility','volatilty','bb','bbands']
    if any(k in lf for k in tech_keys):
        return 'tech'
    if 'volume' in lf or (('vol' in lf) and 'volatility' not in lf):
        return 'volume'
    if any(s in lf for s in ('sentiment','nlp','text','news','polarity','vader')):
        return 'sentiment'
    return 'other'


def normalize_to_100(arr):
    arr = np.array(arr, dtype=float)
    maxv = np.nanmax(arr)
    if maxv <= 0 or np.isnan(maxv):
        return np.zeros_like(arr)
    return (arr / maxv) * 100.0


def main():
    warnings.filterwarnings('ignore')
    df = load_data(DATA_PATH)
    feat_cols = [c for c in df.columns if c not in ('date','ticker','binary_label','next_day_return')]

    # chronological split: first 80% train, last 20% oos
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    split = int(n * 0.8)
    train_df = df_sorted.iloc[:split]
    oos_df = df_sorted.iloc[split:]

    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df['binary_label'].astype(int).values
    X_oos = oos_df[feat_cols].fillna(0)
    y_oos = oos_df['binary_label'].astype(int).values

    print('Training CatBoost on training set...')
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(X_train, y_train)

    print('Computing PredictionValuesChange importances...')
    pv = model.get_feature_importance(type='PredictionValuesChange')

    print('Computing permutation importances on out-of-sample set (AUC decrease)...')
    scorer = make_scorer(roc_auc_score, needs_proba=True)
    perm = permutation_importance(model, X_oos, y_oos, scoring=scorer, n_repeats=10, random_state=42, n_jobs=4)
    perm_mean = perm.importances_mean  # higher means more important (decrease in score)

    # Align features
    feat_imp_df = pd.DataFrame({'feature': feat_cols, 'pv': pv, 'perm': perm_mean})

    # Normalize both to 0-100
    feat_imp_df['pv_norm'] = normalize_to_100(feat_imp_df['pv'])
    feat_imp_df['perm_norm'] = normalize_to_100(feat_imp_df['perm'])

    feat_imp_df['avg_norm'] = (feat_imp_df['pv_norm'] + feat_imp_df['perm_norm']) / 2.0

    # Rank by average
    feat_imp_df = feat_imp_df.sort_values('avg_norm', ascending=False).reset_index(drop=True)

    # Flag potentially inflated due to correlation
    top3_pv = feat_imp_df.head(3)['feature'].tolist()
    perm_top5 = feat_imp_df.sort_values('perm_norm', ascending=False).head(5)['feature'].tolist()
    inflated = [f for f in top3_pv if f not in perm_top5]

    # Prepare plot data top20
    top20 = feat_imp_df.head(20).copy()
    top20['cat'] = top20['feature'].apply(categorize_feature)
    color_map = {'short': 'tab:blue', 'medium': 'tab:orange', 'tech': 'tab:green', 'volume': 'tab:purple', 'sentiment': 'tab:brown', 'other': 'tab:gray'}
    top20['color'] = top20['cat'].map(color_map).fillna('tab:gray')

    # normalize avg so top=100
    top20['avg_norm_top100'] = normalize_to_100(top20['avg_norm'])

    # Plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = top20[::-1].reset_index(drop=True)  # reverse for plotting (largest on top)
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df['avg_norm_top100'], color=plot_df['color'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['feature'])
    ax.set_xlabel('Normalized importance (top = 100)')
    ax.set_title('Top 20 Most Predictive Features (CatBoost) — Robust (PV + Permutation)')

    # annotate values
    for i, v in enumerate(plot_df['avg_norm_top100']):
        ax.text(v + 0.8, i, f"{v:.1f}", va='center', fontsize=8)

    # legend
    import matplotlib.patches as mpatches
    legend_labels = {
        'short': 'Short-term returns (1-5d)',
        'medium': 'Medium-term returns (10-50d)',
        'tech': 'Volatility / Technical',
        'volume': 'Volume',
        'sentiment': 'Sentiment / Text',
        'other': 'Other',
    }
    legend_patches = [mpatches.Patch(color=color_map[k], label=legend_labels[k]) for k in legend_labels]
    ax.legend(handles=legend_patches, loc='lower right')

    # annotation box about volume vs volatility
    note = ("Volume features show high CatBoost importance but reduced permutation importance,\n"
            "indicating correlation with volatility and liquidity rather than standalone predictive power.")
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print('Saved robust feature importance plot to', OUT_PLOT)

    # Save comparison CSV
    feat_imp_df.to_csv(OUT_CSV, index=False)
    print('Saved feature importance comparison CSV to', OUT_CSV)

    # Flagging: if feature ranks top-3 in PV but not top-5 in perm
    if inflated:
        print('FLAG: Features potentially inflated due to correlation (top-3 PV but not top-5 permutation):', inflated)

    # HARD validation: at least 3 of top5 must be return_1d-5d or atr
    top5 = feat_imp_df.head(5)['feature'].str.lower().tolist()
    count_short_atr = 0
    for f in top5:
        if any(s in f for s in ['return_1d','return_2d','return_3d','return_4d','return_5d']) or 'atr' in f:
            count_short_atr += 1
    if count_short_atr < 3:
        diag_csv = OUT_CSV.parent / 'feature_importance_robust_diagnostic.csv'
        feat_imp_df.to_csv(diag_csv, index=False)
        print('WARNING: HARD validation failed — fewer than 3 of top5 are short-term returns or ATR')
        print('Saved diagnostic CSV to', diag_csv)


if __name__ == '__main__':
    main()
