"""Train a CatBoost model on combined features and plot top-20 feature importances.

This script will:
- Load `data/combined/all_features.parquet`.
- Train a CatBoostClassifier (uses modest iterations for speed).
- Extract feature importances with `model.get_feature_importance()`.
- Select top 20 features, normalize importance so top = 100, and plot a horizontal bar chart.

Saves: `results/feature_importance.png` (300 DPI)
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

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'combined' / 'all_features.parquet'
OUT_PLOT = ROOT / 'results' / 'feature_importance.png'

CB_PARAMS = dict(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    thread_count=4,
)


def load_data(path):
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
        if lf == f'return_{d}d' or lf == f'return_{d}':
            return 'short'
    # medium-term returns 10-50d (10,20,30,50)
    for d in (10,20,30,50):
        if lf == f'return_{d}d' or lf == f'return_{d}':
            return 'medium'
    # technical indicators
    tech_keys = ['atr','rsi','macd','vol','volatility','sma','ema','bbands','adx']
    if any(k in lf for k in tech_keys):
        return 'tech'
    return 'other'


def main():
    warnings.filterwarnings('ignore')
    df = load_data(DATA_PATH)
    feat_cols = [c for c in df.columns if c not in ('date','ticker','binary_label','next_day_return')]

    X = df[feat_cols].fillna(0)
    y = df['binary_label'].astype(int).values

    print('Training CatBoost on full dataset (this may take a moment)...')
    model = CatBoostClassifier(**CB_PARAMS)
    model.fit(X, y)

    # Use PredictionValuesChange importance for more interpretable effect sizes
    imp = model.get_feature_importance(type='PredictionValuesChange')
    if len(imp) != len(feat_cols):
        raise RuntimeError('Feature importance length mismatch')

    feat_imp = pd.DataFrame({'feature': feat_cols, 'importance': imp})
    feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)

    topn = feat_imp.head(20).copy()

    # normalize to top = 100
    max_val = topn['importance'].iloc[0]
    if max_val <= 0:
        topn['importance_norm'] = 0.0
    else:
        topn['importance_norm'] = topn['importance'] / max_val * 100.0

    # categorize colors
    color_map = {'short': 'tab:blue', 'medium': 'tab:orange', 'tech': 'tab:green', 'other': 'tab:gray', 'volume': 'tab:purple', 'sentiment': 'tab:brown'}
    topn['cat'] = topn['feature'].apply(categorize_feature)
    # extend categorize to detect volume and sentiment
    def categorize_extended(fname, base_cat):
        lf = fname.lower()
        if 'volume' in lf or lf.endswith('_vol') or 'vol' in lf and 'volatility' not in lf:
            return 'volume'
        if any(s in lf for s in ('sentiment','nlp','text','news','polarity','vader')):
            return 'sentiment'
        return base_cat

    topn['cat'] = [categorize_extended(f, c) for f, c in zip(topn['feature'], topn['cat'])]
    topn['color'] = topn['cat'].map(color_map).fillna('tab:gray')

    # plotting horizontal bar (most important at top)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 8))
    topn_plot = topn[::-1].reset_index(drop=True)  # reverse for plotting so largest on top
    y_pos = np.arange(len(topn_plot))
    bars = ax.barh(y_pos, topn_plot['importance_norm'], color=topn_plot['color'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topn_plot['feature'])
    ax.set_xlabel('Importance (normalized, top=100)')
    ax.set_title('Top 20 Most Predictive Features (CatBoost)')

    # annotate exact importance values (normalized)
    for i, (val, orig) in enumerate(zip(topn_plot['importance_norm'], topn_plot['importance'])):
        ax.text(val + 0.8, i, f"{val:.1f}", va='center', fontsize=8)

    # legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color_map[k], label={'short':'Short-term returns (1-5d)','medium':'Medium-term returns (10-50d)','tech':'Volatility/Tech indicators','volume':'Volume','sentiment':'Sentiment/Text','other':'Other'}[k]) for k in ['short','medium','tech','volume','sentiment','other']]
    ax.legend(handles=legend_patches, loc='lower right')

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PLOT.parent / 'feature_importance_catboost.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print('Saved feature importance plot to', out_path)

    # Save top20 table
    topn.to_csv(OUT_PLOT.parent / 'feature_importance_top20.csv', index=False)

    # Validation checks (WARN if violated)
    top5 = topn['feature'].head(5).str.lower().tolist()
    short_or_atr = sum(1 for f in top5 if any(s in f for s in ('return_1d','return_2d','return_3d','return_4d','return_5d')) or 'atr' in f)
    if short_or_atr < 3:
        print('WARNING: Less than 3 of top 5 features are short-term returns or ATR (found', short_or_atr, ')')

    # Volume-based features in top3
    top3 = topn['feature'].head(3).str.lower().tolist()
    vol_in_top3 = any('volume' in f or (('vol' in f) and 'volatility' not in f) for f in top3)
    if vol_in_top3:
        print('WARNING: Volume-based feature present in top 3:', [f for f in top3 if 'volume' in f or (('vol' in f) and 'volatility' not in f)])

    # Sentiment in top10
    top10 = topn['feature'].head(10).str.lower().tolist()
    sentiment_in_top10 = [f for f in top10 if any(s in f for s in ('sentiment','nlp','text','news','polarity','vader'))]
    if sentiment_in_top10:
        print('NOTE: Sentiment features in top 10:', sentiment_in_top10)


if __name__ == '__main__':
    main()
