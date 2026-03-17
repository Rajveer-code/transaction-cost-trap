"""Generate publication tables (CSV + LaTeX) from results and dataset files.

Outputs saved to: research_outputs/final_final_final/tables/

Handles missing files gracefully (skips table and logs warning).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuration
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
DATA_DIR = ROOT / 'data' / 'combined'
OUTPUT_DIR = ROOT / 'research_outputs' / 'final_final_final' / 'tables'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('generate_all_tables')


def load_csv(fname, required=True):
    path = RESULTS_DIR / fname
    if not path.exists():
        if required:
            logger.warning(f"Missing required CSV: {path}. Skipping related table.")
        else:
            logger.info(f"Optional CSV not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        logger.info(f"✅ Loaded {fname} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None


def load_parquet(path):
    p = Path(path)
    if not p.exists():
        logger.warning(f"Missing parquet file: {p}")
        return None
    try:
        df = pd.read_parquet(p)
        logger.info(f"✅ Loaded {p.name} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to read parquet {p}: {e}")
        return None


def df_to_latex(df, tex_path, caption, label, numeric_fmt=None):
    """Write a DataFrame to a LaTeX table using booktabs.

    - numeric_fmt: dict mapping column -> format string like '{:.2f}'
    """
    if numeric_fmt is None:
        numeric_fmt = {}

    # Determine column alignments: left for object, center for numeric
    aligns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            aligns.append('c')
        else:
            aligns.append('l')

    col_align = ' '.join(aligns)

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[ht]\\centering\\n')
        f.write(f'\\caption{{{caption}}}\\n')
        f.write(f'\\label{{{label}}}\\n')
        f.write(f'\\begin{{tabular}}{{{col_align}}}\\n')
        f.write('\\toprule\\n')

        # Header
        header_cells = [f"\\textbf{{{c}}}" for c in df.columns]
        f.write(' & '.join(header_cells) + ' \\\\ \n')
        f.write('\\midrule\\n')

        # Rows
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    cells.append('')
                    continue
                if col in numeric_fmt and pd.api.types.is_numeric_dtype(type(val)):
                    try:
                        cells.append(numeric_fmt[col].format(val))
                    except Exception:
                        cells.append(str(val))
                elif pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float, np.floating)):
                    # default numeric formatting
                    if isinstance(val, float):
                        cells.append(f"{val:.2f}")
                    else:
                        cells.append(str(val))
                else:
                    cells.append(str(val))
            f.write(' & '.join(cells) + ' \\\\ \n')

        f.write('\\bottomrule\\n')
        f.write('\\end{tabular}\\n')
        f.write('\\end{table}\\n')


def save_table(df, base_name, caption, label, numeric_fmt=None):
    csv_path = OUTPUT_DIR / f"{base_name}.csv"
    tex_path = OUTPUT_DIR / f"{base_name}.tex"

    # CSV: apply formatting where requested (string formatting)
    df_csv = df.copy()
    if numeric_fmt:
        for col, fmt in numeric_fmt.items():
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')

    df_csv.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved {csv_path}")

    # LaTeX
    try:
        df_to_latex(df, tex_path, caption, label, numeric_fmt=numeric_fmt)
        logger.info(f"✅ Saved {tex_path}")
    except Exception as e:
        logger.error(f"Failed to write LaTeX for {base_name}: {e}")


# ------------------------------
# Table 1: Dataset summary
def table1_dataset_summary():
    p = DATA_DIR / 'all_features.parquet'
    df = load_parquet(p)
    if df is None:
        return

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    grouped = df.groupby('ticker')
    rows = []
    for ticker, g in grouped:
        n = len(g)
        min_date = g['date'].min().strftime('%Y-%m-%d') if 'date' in g else ''
        max_date = g['date'].max().strftime('%Y-%m-%d') if 'date' in g else ''
        mean_ret = g['next_day_return'].mean() * 100 if 'next_day_return' in g else np.nan
        std_ret = g['next_day_return'].std() * 100 if 'next_day_return' in g else np.nan
        up_pct = g['binary_label'].mean() * 100 if 'binary_label' in g else np.nan
        down_pct = 100.0 - up_pct if pd.notna(up_pct) else np.nan

        rows.append({
            'Ticker': ticker,
            'Observations': n,
            'Date_Range': f"{min_date} to {max_date}",
            'Mean_Return_%': mean_ret,
            'Std_Return_%': std_ret,
            'Upward_Days_%': up_pct,
            'Downward_Days_%': down_pct
        })

    df_sum = pd.DataFrame(rows)

    # Summary row
    total_obs = df_sum['Observations'].sum()
    full_min = df['date'].min().strftime('%Y-%m-%d') if 'date' in df else ''
    full_max = df['date'].max().strftime('%Y-%m-%d') if 'date' in df else ''
    mean_of_means = df_sum['Mean_Return_%'].mean()
    mean_up = df_sum['Upward_Days_%'].mean()
    mean_down = df_sum['Downward_Days_%'].mean()

    summary = {
        'Ticker': 'Total',
        'Observations': int(total_obs),
        'Date_Range': f"{full_min} to {full_max}",
        'Mean_Return_%': mean_of_means,
        'Std_Return_%': np.nan,
        'Upward_Days_%': mean_up,
        'Downward_Days_%': mean_down
    }

    df_sum = pd.concat([df_sum, pd.DataFrame([summary])], ignore_index=True)

    numeric_fmt = {
        'Mean_Return_%': '{:.2f}',
        'Std_Return_%': '{:.2f}',
        'Upward_Days_%': '{:.2f}',
        'Downward_Days_%': '{:.2f}'
    }

    save_table(df_sum, 'table1_dataset_summary',
               'Dataset Summary by Ticker', 'tab:table1_dataset_summary', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 2: Feature groups (hardcoded)
def table2_feature_groups():
    data = [
        ['NLP Features', 23, 'finbert_score, vader_polarity, textblob_polarity, ensemble_mean, entity_sentiment'],
        ['Technical Indicators', 15, 'RSI, MACD, Bollinger_Bands, ATR, OBV, CMF, VWAP'],
        ['Lagged Features', 4, 'sentiment_lag_1d, sentiment_lag_3d, sentiment_lag_7d, return_lag_1d'],
        ['Metadata', 2, 'date, ticker'],
        ['ML Targets', 2, 'next_day_return, binary_label']
    ]
    df = pd.DataFrame(data, columns=['Feature_Group', 'Count', 'Example_Features'])
    save_table(df, 'table2_feature_groups', 'Feature Groups and Example Features', 'tab:table2_feature_groups')


# ------------------------------
# Table 3: Walk-forward results by fold
def table3_walk_forward_results():
    df = load_csv('walk_forward_results.csv')
    if df is None:
        return

    if 'fold' not in df.columns:
        logger.warning('walk_forward_results.csv missing fold column; skipping Table 3')
        return

    grouped = df.groupby('fold').agg({
        'accuracy': 'mean',
        'precision': 'mean' if 'precision' in df.columns else (lambda x: np.nan),
        'recall': 'mean' if 'recall' in df.columns else (lambda x: np.nan),
        'f1': 'mean' if 'f1' in df.columns else (lambda x: np.nan),
        'roc_auc': 'mean' if 'roc_auc' in df.columns else (lambda x: np.nan)
    })

    grouped = grouped.reset_index().rename(columns={
        'accuracy': 'Accuracy_%',
        'precision': 'Precision_%',
        'recall': 'Recall_%',
        'f1': 'F1_Score',
        'roc_auc': 'ROC_AUC'
    })

    # Convert to percentages where required
    grouped['Accuracy_%'] = grouped['Accuracy_%'] * 100
    if 'Precision_%' in grouped.columns:
        grouped['Precision_%'] = grouped['Precision_%'] * 100
    if 'Recall_%' in grouped.columns:
        grouped['Recall_%'] = grouped['Recall_%'] * 100

    # Summary row
    mean_row = {
        'fold': 'Mean',
        'Accuracy_%': grouped['Accuracy_%'].mean(),
        'Precision_%': grouped['Precision_%'].mean() if 'Precision_%' in grouped.columns else np.nan,
        'Recall_%': grouped['Recall_%'].mean() if 'Recall_%' in grouped.columns else np.nan,
        'F1_Score': grouped['F1_Score'].mean() if 'F1_Score' in grouped.columns else np.nan,
        'ROC_AUC': grouped['ROC_AUC'].mean() if 'ROC_AUC' in grouped.columns else np.nan
    }

    # Rename fold column to Fold for output
    grouped = grouped.rename(columns={'fold': 'Fold'})
    grouped = pd.concat([grouped, pd.DataFrame([mean_row]).rename(columns={'fold': 'Fold'})], ignore_index=True)

    numeric_fmt = {
        'Accuracy_%': '{:.2f}',
        'Precision_%': '{:.2f}',
        'Recall_%': '{:.2f}',
        'F1_Score': '{:.3f}',
        'ROC_AUC': '{:.3f}'
    }

    save_table(grouped, 'table3_walk_forward_results', 'Walk-Forward Results by Fold', 'tab:table3_walk_forward_results', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 4: Cross-ticker matrix
def table4_cross_ticker_matrix():
    df = load_csv('cross_ticker_results.csv')
    if df is None:
        return

    # Expect columns: train_tickers (comma-separated) or train_ticker, test_ticker, accuracy
    if 'train_ticker' in df.columns:
        train_col = 'train_ticker'
    elif 'train_tickers' in df.columns:
        # For our result format, train_tickers might be comma-separated; we will expand into single row per test_ticker
        # create pivot with train as each ticker (we'll set train values to NaN except mean)
        # If input only has one row per test_ticker, we'll put the accuracy in the column for that test
        df_exp = []
        for _, r in df.iterrows():
            test = r.get('test_ticker')
            acc = r.get('accuracy')
            # assign same acc for all train tickers as approximation
            df_exp.append({'train_ticker': 'ALL', 'test_ticker': test, 'accuracy': acc})
        df = pd.DataFrame(df_exp)
        train_col = 'train_ticker'
    else:
        logger.warning('cross_ticker_results.csv missing expected columns; skipping Table 4')
        return

    pivot = df.pivot_table(index='train_ticker', columns='test_ticker', values='accuracy', aggfunc='mean')

    # Convert accuracy to percentage
    pivot = pivot * 100

    # Add mean row and column
    pivot['Mean'] = pivot.mean(axis=1)
    mean_col = pivot.mean(axis=0)
    pivot.loc['Mean'] = mean_col

    pivot = pivot.reset_index().rename(columns={'train_ticker': 'Train_Ticker'})

    # Format numeric columns
    numeric_fmt = {c: '{:.2f}' for c in pivot.columns if c != 'Train_Ticker'}

    save_table(pivot, 'table4_cross_ticker_matrix', 'Cross-Ticker Accuracy Matrix (%)', 'tab:table4_cross_ticker_matrix', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 5: Ablation studies
def table5_ablation_studies():
    df = load_csv('ablation_studies.csv')
    if df is None:
        return

    # Expect a column describing the feature_set / experiment and accuracy
    key_col = 'feature_set' if 'feature_set' in df.columns else ('experiment' if 'experiment' in df.columns else None)
    if key_col is None or 'accuracy' not in df.columns:
        logger.warning('ablation_studies.csv missing expected columns; skipping Table 5')
        return

    agg = df.groupby(key_col)['accuracy'].mean()

    # Identify full model accuracy
    full_key = None
    for k in agg.index:
        if str(k).lower().startswith('full') or str(k).lower() == 'full':
            full_key = k
            break
    if full_key is None:
        # fallback: take max accuracy as full
        full_key = agg.idxmax()

    full_acc = agg.loc[full_key]

    rows = []
    for k, v in agg.items():
        delta = full_acc - v
        rows.append({'Model_Variant': str(k), 'Accuracy_%': v * 100, 'Delta_vs_Full_Model': delta * 100})

    df_out = pd.DataFrame(rows).sort_values('Accuracy_%', ascending=False).reset_index(drop=True)

    numeric_fmt = {'Accuracy_%': '{:.2f}', 'Delta_vs_Full_Model': '{:.2f}'}
    save_table(df_out, 'table5_ablation_studies', 'Ablation Study Results', 'tab:table5_ablation_studies', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 6: Feature importance top 20
def table6_feature_importance_top():
    df = load_csv('feature_importance_walkforward.csv')
    if df is None:
        return

    # input columns expected: feature, importance
    if 'feature' not in df.columns or 'importance' not in df.columns:
        logger.warning('feature_importance_walkforward.csv missing expected columns; skipping Table 6')
        return

    agg = df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    agg = agg.rename(columns={'mean': 'Mean_Importance', 'std': 'Std_Importance'})
    agg = agg.sort_values('Mean_Importance', ascending=False).head(20)
    agg = agg.reset_index(drop=True)
    agg.insert(0, 'Rank', agg.index + 1)

    numeric_fmt = {'Mean_Importance': '{:.4f}', 'Std_Importance': '{:.4f}'}
    save_table(agg, 'table6_feature_importance_top', 'Top 20 Feature Importances', 'tab:table6_feature_importance_top', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 7: Baseline comparison per ticker
def table7_baseline_comparison():
    df_rand = load_csv('baseline_random.csv', required=False)
    df_lr = load_csv('baseline_logistic_regression.csv', required=False)
    df_tech = load_csv('baseline_technical_only.csv', required=False)
    df_wf = load_csv('walk_forward_results.csv')
    if df_wf is None:
        logger.warning('walk_forward_results.csv required for Table 7; skipping')
        return

    tickers = sorted(df_wf['ticker'].unique()) if 'ticker' in df_wf.columns else []
    rows = []
    for t in tickers:
        r = {'Ticker': t}
        if df_rand is not None and 'accuracy' in df_rand.columns:
            try:
                r['Random_%'] = df_rand[df_rand['ticker'] == t]['accuracy'].mean() * 100
            except Exception:
                r['Random_%'] = np.nan
        else:
            r['Random_%'] = np.nan

        if df_lr is not None and 'accuracy' in df_lr.columns:
            r['Logistic_Regression_%'] = df_lr[df_lr['ticker'] == t]['accuracy'].mean() * 100
        else:
            r['Logistic_Regression_%'] = np.nan

        if df_tech is not None and 'accuracy' in df_tech.columns:
            r['Technical_Only_%'] = df_tech[df_tech['ticker'] == t]['accuracy'].mean() * 100
        else:
            r['Technical_Only_%'] = np.nan

        r['Walk_Forward_CatBoost_%'] = df_wf[df_wf['ticker'] == t]['accuracy'].mean() * 100 if 'accuracy' in df_wf.columns else np.nan

        rows.append(r)

    df_out = pd.DataFrame(rows)
    # summary row
    summary = {'Ticker': 'Mean',
               'Random_%': df_out['Random_%'].mean(),
               'Logistic_Regression_%': df_out['Logistic_Regression_%'].mean(),
               'Technical_Only_%': df_out['Technical_Only_%'].mean(),
               'Walk_Forward_CatBoost_%': df_out['Walk_Forward_CatBoost_%'].mean()}
    df_out = pd.concat([df_out, pd.DataFrame([summary])], ignore_index=True)

    numeric_fmt = {c: '{:.2f}' for c in df_out.columns if c != 'Ticker'}
    save_table(df_out, 'table7_baseline_comparison', 'Baseline Model Comparison by Ticker', 'tab:table7_baseline_comparison', numeric_fmt=numeric_fmt)


# ------------------------------
# Table 8: Backtest statistics aggregated
def table8_backtest_statistics():
    df = load_csv('backtest_results.csv')
    if df is None:
        return

    # Use available columns; compute means across tickers
    metrics = []

    # Total Return
    ml_ret_col = 'strategy_total_return' if 'strategy_total_return' in df.columns else None
    bh_ret_col = 'buy_hold_total_return' if 'buy_hold_total_return' in df.columns else None

    if ml_ret_col and bh_ret_col:
        ml_tot = df[ml_ret_col].mean() * 100
        bh_tot = df[bh_ret_col].mean() * 100
        metrics.append(('Total Return %', ml_tot, bh_tot, ml_tot - bh_tot))
    else:
        logger.warning('Total return columns not found in backtest_results.csv')

    # Sharpe Ratio
    ml_sh = df['strategy_sharpe'].mean() if 'strategy_sharpe' in df.columns else np.nan
    bh_sh = df['buy_hold_sharpe'].mean() if 'buy_hold_sharpe' in df.columns else np.nan
    if not (pd.isna(ml_sh) and pd.isna(bh_sh)):
        metrics.append(('Sharpe Ratio', ml_sh, bh_sh, ml_sh - bh_sh))

    # Max Drawdown
    ml_dd = df['strategy_max_drawdown'].mean() * 100 if 'strategy_max_drawdown' in df.columns else np.nan
    bh_dd = df['buy_hold_max_drawdown'].mean() * 100 if 'buy_hold_max_drawdown' in df.columns else np.nan
    if not (pd.isna(ml_dd) and pd.isna(bh_dd)):
        metrics.append(('Max Drawdown %', ml_dd, bh_dd, ml_dd - bh_dd))

    # Win Rate: try to use 'win_rate' or 'strategy_win_rate' if present
    ml_wr = None
    bh_wr = None
    if 'strategy_win_rate' in df.columns:
        ml_wr = df['strategy_win_rate'].mean() * 100
    elif 'win_rate' in df.columns:
        ml_wr = df['win_rate'].mean() * 100

    if 'buy_hold_win_rate' in df.columns:
        bh_wr = df['buy_hold_win_rate'].mean() * 100

    if ml_wr is not None or bh_wr is not None:
        ml_wr = ml_wr if ml_wr is not None else np.nan
        bh_wr = bh_wr if bh_wr is not None else np.nan
        metrics.append(('Win Rate %', ml_wr, bh_wr, (ml_wr - bh_wr) if (not pd.isna(ml_wr) and not pd.isna(bh_wr)) else np.nan))
    else:
        logger.warning('Win Rate not available in backtest_results.csv; skipping Win Rate metric')

    if not metrics:
        logger.warning('No backtest metrics computed; skipping Table 8')
        return

    df_out = pd.DataFrame([{'Metric': m[0], 'ML_Strategy': m[1], 'Buy_Hold_Baseline': m[2], 'Difference': m[3]} for m in metrics])

    # Format numeric columns: returns & drawdowns 2 decimals, sharpe 3 decimals
    numeric_fmt = {}
    for col in ['ML_Strategy', 'Buy_Hold_Baseline', 'Difference']:
        numeric_fmt[col] = '{:.2f}'

    save_table(df_out, 'table8_backtest_statistics', 'Backtest Aggregated Statistics', 'tab:table8_backtest_statistics', numeric_fmt=numeric_fmt)


def main():
    logger.info('Starting table generation')
    table1_dataset_summary()
    table2_feature_groups()
    table3_walk_forward_results()
    table4_cross_ticker_matrix()
    table5_ablation_studies()
    table6_feature_importance_top()
    table7_baseline_comparison()
    table8_backtest_statistics()
    logger.info('Table generation finished')


if __name__ == '__main__':
    main()
