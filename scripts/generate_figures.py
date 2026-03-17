"""
Generate figures and tables for the paper `paper_research.md`.

Usage:
  - With real data: provide paths to CSV files using `--data-dir`.
  - For a quick preview: use `--mock` to generate example figures with synthetic data.

Expected CSVs in --data-dir when using real data
(you can gradually add these as your pipeline matures):

Core (already used in the previous version)
  - roc_probs.csv              # y_true + model probabilities
  - backtest.csv               # date, ml_cum, baseline_cum
  - shap.csv                   # feature, mean_abs_shap
  - consensus.csv              # consensus
  - ablation.csv               # experiment, accuracy
  - monthly_accuracy.csv       # month, accuracy
  - table_dataset.csv          # dataset summary
  - table_features.csv         # feature groups summary
  - table_per_fold.csv         # per-fold performance
  - table_cross_ticker.csv     # cross-ticker accuracy (square matrix, index=tickers)

New (for extended figures/tables)
  - daily_returns.csv          # date, ml_ret, bh_ret
  - calibration.csv            # y_true, y_prob (ensemble positive prob)
  - predictions.csv            # y_true, y_pred (0/1)  OR confusion_matrix.csv 2x2
  - corr_matrix.csv            # correlation matrix (index + columns = features)
  - leakage_summary.csv        # setting, accuracy  (e.g. 'Leaky', 'Leak-free')
  - table_features_full.csv    # full feature list
  - table_hyperparams.csv      # CatBoost hyperparameters
  - table_stats_tests.csv      # statistical tests summary

Outputs are saved into `out_dir` (default `docs/figures`).

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn
Optional: shap for SHAP plots

Examples:
  python scripts/generate_figures.py --mock --out-dir docs/figures
  python scripts/generate_figures.py --data-dir research_outputs --out-dir docs/figures
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.modeling.evaluation import run_walk_forward_cv, _tune_threshold

sns.set(style="whitegrid")


def save_figure_variants(fig, png_path: Path, dpi=200):
    """Save figure as PNG, SVG and PDF using the same stem."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    for path in (svg_path, pdf_path):
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception as e:
            print(f"Failed to save {path}: {e}")


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
#  FIGURES ALREADY IN YOUR PROJECT
# ---------------------------------------------------------------------
def figure1_walk_forward(save_path: Path):
    """Draw a timeline-style walk-forward validation diagram."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")

    labels = ["Train", "Test", "Train", "Test", "Train", "Test"]
    start_x = 0.02
    width = 0.15
    gap = 0.02
    colors = ["#4c72b0", "#dd8452"] * 3
    x = start_x

    for lbl, c in zip(labels, colors):
        rect = patches.FancyBboxPatch(
            (x, 0.35),
            width,
            0.4,
            boxstyle="round,pad=0.02",
            ec="k",
            fc=c,
            alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            0.55,
            lbl,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            weight="bold",
        )
        if lbl == "Test":
            ax.plot(
                [x + width + 0.005, x + width + 0.005],
                [0.2, 0.8],
                ls="--",
                color="gray",
                linewidth=1,
            )
        x += width + gap

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1 — Walk-Forward Validation (Train → Test → Train → Test)")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure2_pipeline(save_path: Path):
    """Block diagram of end-to-end pipeline."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")

    blocks = [
        "News APIs",
        "NLP models",
        "Feature Engineering",
        "CatBoost",
        "Predictions",
        "Backtest",
    ]
    x = 0.03
    w = 0.15
    gap = 0.03
    for i, b in enumerate(blocks):
        rect = patches.Rectangle(
            (x, 0.35), w, 0.3, ec="k", fc="#2b8cbe", alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            0.5,
            b,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            weight="bold",
        )
        x += w + gap
        if i < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(x - gap / 2 - 0.01, 0.5),
                xytext=(x - (w + gap), 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.6),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 2 — Overall Pipeline")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure3_shap_bar(shap_df: pd.DataFrame, save_path: Path, top_n=12):
    """Horizontal bar plot for SHAP feature importance."""
    df = shap_df.sort_values("mean_abs_shap", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Figure 3 — SHAP Feature Importance")
    fig.tight_layout()
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure4_disagreement_hist(consensus_series: pd.Series, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        consensus_series.dropna(), bins=30, kde=True, color="#dd8452", ax=ax
    )
    ax.set_xlabel("Consensus score (1 = full agreement, 0 = max disagreement)")
    ax.set_title("Figure 4 — Ensemble Disagreement (Consensus) Distribution")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure5_roc_curves(y_true: pd.Series, probs: dict, save_path: Path):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 5 — ROC Curves")
    ax.legend(loc="lower right")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure6_backtest_equity_curve(df: pd.DataFrame, save_path: Path):
    """df expected columns: ['date', 'ml_cum', 'baseline_cum']"""
    df = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["date"], df["ml_cum"], label="ML Strategy", lw=2)
    ax.plot(
        df["date"], df["baseline_cum"], label="Buy & Hold", lw=1.5, alpha=0.8
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Figure 6 — Backtest Equity Curve")
    ax.legend()
    fig.autofmt_xdate()
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure7_ablation_bar(ablation_df: pd.DataFrame, save_path: Path):
    """ablation_df: columns ['experiment', 'accuracy']"""
    df = ablation_df.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x="accuracy", y="experiment", data=df, palette="Blues_r", ax=ax
    )
    ax.set_xlabel("Accuracy")
    ax.set_title("Figure 7 — Ablation Study")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure8_temporal_stability(monthly_df: pd.DataFrame, save_path: Path):
    """monthly_df: columns ['month', 'accuracy'] with month as datetime or string"""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(
        pd.to_datetime(monthly_df["month"]),
        monthly_df["accuracy"],
        marker="o",
        lw=1.6,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 8 — Temporal Stability (Monthly Accuracy)")
    fig.autofmt_xdate()
    save_figure_variants(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------
#  NEW FIGURES
# ---------------------------------------------------------------------
def figure9_return_distribution(daily_df: pd.DataFrame, save_path: Path):
    """daily_df expected columns: ['ml_ret', 'bh_ret'] (daily returns)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(daily_df["ml_ret"], ax=ax, label="ML Strategy", lw=2)
    sns.kdeplot(daily_df["bh_ret"], ax=ax, label="Buy & Hold", lw=2)
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.set_title("Figure 9 — Daily Return Distribution")
    ax.legend()
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure10_calibration(calib_df: pd.DataFrame, save_path: Path, n_bins=10):
    """calib_df columns: ['y_true', 'y_prob']."""
    y_true = calib_df["y_true"].values
    y_prob = calib_df["y_prob"].values
    bins = np.linspace(0, 1, n_bins + 1)
    digitized = np.digitize(y_prob, bins) - 1
    prob_mean = []
    frac_pos = []
    for b in range(n_bins):
        mask = digitized == b
        if mask.sum() == 0:
            continue
        prob_mean.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_mean, frac_pos, marker="o", lw=2, label="Ensemble")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Figure 10 — Calibration Curve")
    ax.legend(loc="upper left")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure11_confusion(pred_df_or_cm: pd.DataFrame, save_path: Path):
    """
    If pred_df_or_cm has columns ['y_true','y_pred'], compute confusion matrix.
    Otherwise assume it is already a 2x2 matrix DataFrame with index/columns.
    """
    if set(pred_df_or_cm.columns) >= {"y_true", "y_pred"}:
        y_true = pred_df_or_cm["y_true"].values
        y_pred = pred_df_or_cm["y_pred"].values
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual DOWN", "Actual UP"],
            columns=["Predicted DOWN", "Predicted UP"],
        )
    else:
        cm_df = pred_df_or_cm

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Figure 11 — Confusion Matrix (Aggregated)")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure12_corr_heatmap(corr_df: pd.DataFrame, save_path: Path):
    """corr_df: square correlation matrix (index & columns = features)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Figure 12 — Feature Correlation Heatmap")
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure13_leakage_bar(leak_df: pd.DataFrame, save_path: Path):
    """leak_df: columns ['setting','accuracy'] (e.g. Leaky vs Leak-free)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="setting", y="accuracy", data=leak_df, palette="Set2", ax=ax
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_title("Figure 13 — Effect of Fixing Temporal Leakage")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
        )
    save_figure_variants(fig, save_path)
    plt.close(fig)


def figure14_entity_example(save_path: Path):
    """
    Schematic example of entity-level sentiment attribution.
    This is a stylised figure (no CSV needed).
    """
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")

    headline = (
        '"Apple CEO Tim Cook warns of slowing demand while '
        "rival Samsung posts record profits.'"
    )
    ax.text(
        0.02,
        0.8,
        "Headline:",
        fontsize=11,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(0.16, 0.8, headline, fontsize=10, transform=ax.transAxes)

    # Colored boxes for entities
    ax.text(
        0.21,
        0.55,
        "Apple",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#4c72b0", ec="none", alpha=0.3),
        transform=ax.transAxes,
    )
    ax.text(
        0.32,
        0.55,
        "Tim Cook",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#dd8452", ec="none", alpha=0.3),
        transform=ax.transAxes,
    )
    ax.text(
        0.63,
        0.55,
        "Samsung",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="#55a868", ec="none", alpha=0.3),
        transform=ax.transAxes,
    )

    ax.text(
        0.02,
        0.28,
        "spaCy NER → entity spans\n"
        "Rule-based attribution → sentiment per entity",
        fontsize=10,
        transform=ax.transAxes,
    )

    ax.text(
        0.55,
        0.28,
        "Outputs:",
        fontsize=11,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.65,
        0.28,
        "CEO_sentiment = negative\n"
        "Company_sentiment (Apple) = mildly negative\n"
        "Competitor_sentiment (Samsung) = positive",
        fontsize=10,
        transform=ax.transAxes,
    )

    ax.set_title("Figure 14 — Example of Entity-Level Sentiment Attribution")
    save_figure_variants(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------
#  TABLE RENDERING
# ---------------------------------------------------------------------
def render_table_image(df: pd.DataFrame, save_path: Path, title: str = None):
    fig, ax = plt.subplots(figsize=(10, max(1.5, 0.35 * len(df))))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    if title:
        ax.set_title(title, fontweight="bold")
    save_figure_variants(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------
#  TABLE GENERATORS
# ---------------------------------------------------------------------
def generate_table1_dataset(df_pred_path: Path, out_path: Path):
    df = pd.read_csv(df_pred_path)
    df["date"] = pd.to_datetime(df["date"])
    rows = []
    per_ticker_avgs = []
    per_ticker_covgs = []
    for t in sorted(df["ticker"].unique()):
        dft = df[df["ticker"] == t].sort_values("date")
        start = dft["date"].min()
        end = dft["date"].max()
        trading_days = dft["date"].nunique()
        total_headlines = float(dft["num_headlines"].fillna(0).sum())
        avg_hpd = total_headlines / trading_days if trading_days else 0.0
        coverage = (dft["num_headlines"].fillna(0) >= 1).mean() * 100.0
        n = len(dft)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_test = n - n_train - n_val
        train_idx = dft.index[:n_train]
        val_idx = dft.index[n_train : n_train + n_val]
        test_idx = dft.index[n_train + n_val :]
        cb_train = dft.loc[train_idx, "movement"].mean() * 100.0 if len(train_idx) else 0.0
        cb_test = dft.loc[test_idx, "movement"].mean() * 100.0 if len(test_idx) else 0.0
        pos_sent = (dft["ensemble_sentiment_mean"].fillna(0) > 0).mean() * 100.0
        per_ticker_avgs.append(avg_hpd)
        per_ticker_covgs.append(coverage)
        rows.append({
            "ticker": t,
            "date_range": f"{start.date()} to {end.date()}",
            "trading_days": int(trading_days),
            "total_headlines": int(total_headlines),
            "avg_headlines_per_day": round(avg_hpd, 2),
            "headline_coverage_pct": round(coverage, 1),
            "train_samples": int(n_train),
            "val_samples": int(n_val),
            "test_samples": int(n_test),
            "class_balance_train_pct": round(cb_train, 1),
            "class_balance_test_pct": round(cb_test, 1),
            "positive_sentiment_pct": round(pos_sent, 1),
        })
    # Calculate weighted averages for the total row
    ticker_weights = df.groupby('ticker')['date'].nunique()
    total_trading_days = ticker_weights.sum()
    
    # Calculate weighted headline coverage
    coverage_weighted = sum((df[df['ticker'] == t]['num_headlines'].fillna(0) >= 1).mean() * weight 
                          for t, weight in ticker_weights.items()) / total_trading_days * 100
    
    # Calculate weighted positive sentiment
    sentiment_weighted = sum((df[df['ticker'] == t]['ensemble_sentiment_mean'].fillna(0) > 0).mean() * weight
                           for t, weight in ticker_weights.items()) / total_trading_days * 100
    
    # Calculate totals
    total = df.copy()
    start_all = total["date"].min()
    end_all = total["date"].max()
    trading_days_all = total_trading_days
    total_headlines_all = int(total["num_headlines"].fillna(0).sum())
    
    # Calculate sample sizes per ticker and sum them up
    ticker_samples = {}
    for t in df['ticker'].unique():
        ticker_df = df[df['ticker'] == t]
        n = len(ticker_df)
        ticker_samples[t] = {
            'train': int(round(n * 0.70)),
            'val': int(round(n * 0.15)),
            'test': max(0, n - int(round(n * 0.70)) - int(round(n * 0.15)))
        }
    
    n_train_all = sum(v['train'] for v in ticker_samples.values())
    n_val_all = sum(v['val'] for v in ticker_samples.values())
    n_test_all = sum(v['test'] for v in ticker_samples.values())
    
    # Calculate class balances using the actual train/test splits per ticker
    cb_train_all = []
    cb_test_all = []
    for t, samples in ticker_samples.items():
        ticker_df = df[df['ticker'] == t]
        if samples['train'] > 0:
            train_df = ticker_df.iloc[:samples['train']]
            cb_train_all.extend(train_df['movement'].values)
        if samples['test'] > 0:
            test_df = ticker_df.iloc[-(samples['test'] + samples['val']):-samples['val']] if samples['val'] > 0 else ticker_df.iloc[-samples['test']:]
            cb_test_all.extend(test_df['movement'].values)
    
    cb_train_avg = np.mean(cb_train_all) * 100.0 if cb_train_all else 0.0
    cb_test_avg = np.mean(cb_test_all) * 100.0 if cb_test_all else 0.0
    rows.append({
        "ticker": "Total",
        "date_range": f"{start_all.date()} to {end_all.date()}",
        "trading_days": int(trading_days_all),
        "total_headlines": total_headlines_all,
        "avg_headlines_per_day": round(total_headlines_all / trading_days_all if trading_days_all > 0 else 0, 2),
        "headline_coverage_pct": round(coverage_weighted, 1),
        "train_samples": n_train_all,
        "val_samples": n_val_all,
        "test_samples": n_test_all,
        "class_balance_train_pct": round(cb_train_avg, 1),
        "class_balance_test_pct": round(cb_test_avg, 1),
        "positive_sentiment_pct": round(sentiment_weighted, 1),
    })
    out_df = pd.DataFrame(rows, columns=[
        "ticker",
        "date_range",
        "trading_days",
        "total_headlines",
        "avg_headlines_per_day",
        "headline_coverage_pct",
        "train_samples",
        "val_samples",
        "test_samples",
        "class_balance_train_pct",
        "class_balance_test_pct",
        "positive_sentiment_pct",
    ])
    foot = pd.DataFrame({
        "ticker": ["note_coverage", "note_balance", "note_sentiment", "note_split", "note_limitation"],
        "date_range": [
            "headline_coverage_pct = % of trading days with ≥1 headline",
            "class_balance = % positive return days (Close_{t+1} > Close_t)",
            "positive_sentiment_pct = headlines with ensemble_sentiment_mean > 0",
            "Split = 70/15/15 chronological (no shuffling)",
            "Sparse headline density noted as a limitation",
        ],
    })
    out_df = pd.concat([out_df, foot], ignore_index=True)
    out_df.to_csv(out_path, index=False)


def fix_table2_features(path: Path):
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "example_features" in df.columns:
        df["example_features"] = df["example_features"].astype(str).str.replace(";", ",")
        # Remove any comment-like rows
        df = df[~df.iloc[:, 0].astype(str).str.startswith("#")]
        # Ensure counts are integers for summation
        if df.shape[1] > 1:
            nums = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            df.iloc[:, 1] = nums
        total_count = int(pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).sum()) if df.shape[1] > 1 else 43
        total_row = pd.DataFrame({
            df.columns[0]: ["Total"],
            df.columns[1]: [total_count],
            df.columns[2]: ["—" if df.shape[1] > 2 else ""],
            df.columns[3]: ["—" if df.shape[1] > 3 else ""],
        })
        foot = pd.DataFrame({
            "feature_category": ["note_no_lookahead", "note_entity_window", "note_causal_lag"],
            "num_features": ["-", "-", "-"],
            "computation_notes": [
                "Technical indicators use only past windows (no future data)",
                "Entity sentiment uses ±5 token window",
                "Lagged features use t-1 values",
            ],
            "example_features": ["-", "-", "-"],
        })
        out_df = pd.concat([df, total_row, foot], ignore_index=True)
        out_df.to_csv(path, index=False)


def recompute_table3_enhanced(df_pred_path: Path, out_path: Path):
    df = pd.read_csv(df_pred_path)
    df["date"] = pd.to_datetime(df["date"])
    folds = [
        ("fold_01", "2025-09-01", "2025-09-14"),
        ("fold_02", "2025-09-15", "2025-09-30"),
        ("fold_03", "2025-10-01", "2025-10-15"),
        ("fold_04", "2025-10-16", "2025-10-31"),
        ("fold_05", "2025-11-01", "2025-11-20"),
    ]
    rows = []
    for fid, start_s, end_s in folds:
        start = pd.to_datetime(start_s)
        end = pd.to_datetime(end_s)
        dft = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        y_true = dft["movement"].astype(int).values if not dft.empty else np.array([])
        y_pred = dft["prediction"].astype(int).values if not dft.empty else np.array([])
        y_prob = dft["probability"].astype(float).values if not dft.empty else np.array([])
        n = int(len(dft))
        acc = float(accuracy_score(y_true, y_pred)) if n else 0.0
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0) if n else (0.0, 0.0, 0.0, None)
        try:
            auc_score = float(roc_auc_score(y_true, y_prob)) if n and len(np.unique(y_true)) == 2 else 0.5
        except Exception:
            auc_score = 0.5
        if n:
            p = acc
            se = (p * (1 - p) / n) ** 0.5
            ci_low = max(0.0, p - 1.96 * se)
            ci_high = min(1.0, p + 1.96 * se)
            if p <= 0.5:
                p_value = 1.0
            else:
                z = (p - 0.5) / (0.5 / (n ** 0.5))
                from math import erf
                p_value = 1 - (0.5 * (1 + erf(abs(z) / 2 ** 0.5)))
        else:
            ci_low = 0.0
            ci_high = 0.0
            p_value = 1.0
        rows.append({
            "fold_id": fid,
            "train_period": "-",
            "test_period": f"{start.date()} to {end.date()}",
            "test_n": n,
            "accuracy": round(acc, 3),
            "precision": round(pr, 3),
            "recall": round(rc, 3),
            "f1": round(f1, 3),
            "auc": round(auc_score, 3),
            "p_value": round(p_value, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
        })
    out_df = pd.DataFrame(rows, columns=[
        "fold_id",
        "train_period",
        "test_period",
        "test_n",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "p_value",
        "ci_low",
        "ci_high",
    ])
    out_df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------
#  MOCK DATA FOR QUICK PREVIEW
# ---------------------------------------------------------------------
def make_mock_data(base_dir: Path):
    """Create small mock datasets for quick preview and testing."""
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # ROC / probabilities
    n = 500
    y = rng.integers(0, 2, size=n)
    probs = {
        "FinBERT-only": rng.random(n) * 0.4 + y * 0.4,
        "Technical-only": rng.random(n) * 0.4 + y * 0.35,
        "Ensemble": rng.random(n) * 0.35 + y * 0.45,
    }
    pd.DataFrame({"y_true": y, **probs}).to_csv(
        base_dir / "roc_probs.csv", index=False
    )

    # Backtest cumulative equity
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    ml = np.cumsum(rng.normal(0.0008, 0.01, size=len(dates)))
    bh = np.cumsum(rng.normal(0.0005, 0.009, size=len(dates)))
    pd.DataFrame(
        {"date": dates, "ml_cum": ml, "baseline_cum": bh}
    ).to_csv(base_dir / "backtest.csv", index=False)

    # Daily returns (for return distribution)
    ml_ret = rng.normal(0.0008, 0.01, size=len(dates))
    bh_ret = rng.normal(0.0005, 0.009, size=len(dates))
    pd.DataFrame(
        {"date": dates, "ml_ret": ml_ret, "bh_ret": bh_ret}
    ).to_csv(base_dir / "daily_returns.csv", index=False)

    # SHAP
    features = [
        "ensemble_sentiment_mean",
        "num_headlines",
        "RSI",
        "MACD",
        "CEO_sentiment",
        "VWAP",
        "sentiment_lag1",
        "volatility_lag1",
        "OBV",
        "CMF",
    ]
    shap_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": np.abs(
                rng.normal(0.07, 0.02, size=len(features))
            ),
        }
    )
    shap_df.to_csv(base_dir / "shap.csv", index=False)

    # Consensus
    consensus = rng.beta(0.7, 4, size=n)
    pd.DataFrame({"consensus": consensus}).to_csv(
        base_dir / "consensus.csv", index=False
    )

    # Ablation
    ablation = pd.DataFrame(
        {
            "experiment": [
                "All features",
                "Minus entity sentiment",
                "Minus disagreement",
                "Sentiment-only (proxy)",
                "Technical-only (proxy)",
            ],
            "accuracy": [0.532, 0.524, 0.526, 0.512, 0.50],
        }
    )
    ablation.to_csv(base_dir / "ablation.csv", index=False)

    # Monthly accuracy
    months = pd.date_range("2019-01-01", periods=36, freq="M")
    monthly = pd.DataFrame(
        {
            "month": months,
            "accuracy": 0.502
            + 0.02 * np.sin(np.linspace(0, 6, len(months)))
            + rng.normal(0, 0.01, len(months)),
        }
    )
    monthly.to_csv(base_dir / "monthly_accuracy.csv", index=False)

    # Dataset table
    dataset = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "date_range": ["2016-01-01:2024-12-31"] * 3,
            "headlines": [12000, 11800, 11000],
            "trading_days": [2000, 2000, 2000],
            "valid_samples": [1500, 1480, 1420],
        }
    )
    dataset.to_csv(base_dir / "table_dataset.csv", index=False)

    # Feature groups
    feature_groups = pd.DataFrame(
        {
            "feature_group": ["Sentiment", "Technical", "Lagged"],
            "features": [
                "24 features (FinBERT/VADER/TextBlob aggregates)",
                "15 indicators (RSI, MACD, ...)",
                "4 lagged features",
            ],
            "examples": [
                "ensemble_sentiment_mean, CEO_sentiment",
                "RSI, MACD, VWAP",
                "sentiment_lag1",
            ],
            "purpose": ["Text signal", "Price dynamics", "Autoregression"],
        }
    )
    feature_groups.to_csv(base_dir / "table_features.csv", index=False)

    # Per-fold table
    per_fold = pd.DataFrame(
        {
            "fold_id": list(range(1, 7)),
            "train_window": [
                "2016-2018",
                "2016-2018",
                "2016-2018",
                "2016-2018",
                "2016-2018",
                "2016-2018",
            ],
            "test_window": ["2019", "2020", "2021", "2022", "2023", "2024"],
            "accuracy": [0.52, 0.53, 0.535, 0.528, 0.537, 0.532],
        }
    )
    per_fold.to_csv(base_dir / "table_per_fold.csv", index=False)

    # Cross-ticker
    cross = pd.DataFrame(
        np.round(0.48 + rng.random((3, 3)) * 0.06, 3),
        columns=["AAPL", "MSFT", "GOOGL"],
        index=["AAPL", "MSFT", "GOOGL"],
    )
    cross.to_csv(base_dir / "table_cross_ticker.csv")

    # Calibration data
    calib = pd.DataFrame(
        {
            "y_true": y,
            "y_prob": probs["Ensemble"],
        }
    )
    calib.to_csv(base_dir / "calibration.csv", index=False)

    # Predictions (for confusion matrix)
    y_pred = (calib["y_prob"] > 0.5).astype(int)
    pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(
        base_dir / "predictions.csv", index=False
    )

    # Correlation matrix
    feat_mat = rng.normal(size=(500, 6))
    corr = pd.DataFrame(
        feat_mat,
        columns=[
            "ensemble_sentiment_mean",
            "RSI",
            "MACD",
            "CEO_sentiment",
            "VWAP",
            "sentiment_lag1",
        ],
    ).corr()
    corr.to_csv(base_dir / "corr_matrix.csv")

    # Leakage summary
    leak_df = pd.DataFrame(
        {
            "setting": ["Leaky (random split)", "Leak-free (walk-forward)"],
            "accuracy": [0.65, 0.532],
        }
    )
    leak_df.to_csv(base_dir / "leakage_summary.csv", index=False)

    # Full feature list table
    features_full = pd.DataFrame(
        {
            "feature_name": [
                "ensemble_sentiment_mean",
                "CEO_sentiment",
                "competitor_sentiment",
                "num_headlines",
                "RSI",
                "MACD",
                "VWAP",
                "sentiment_lag1",
                "daily_return_lag1",
                "volume_lag1",
            ],
            "group": [
                "Sentiment",
                "Sentiment",
                "Sentiment",
                "Sentiment",
                "Technical",
                "Technical",
                "Technical",
                "Lagged",
                "Lagged",
                "Lagged",
            ],
            "description": [
                "Confidence-weighted mean sentiment across models",
                "Entity-level sentiment for CEO mentions",
                "Entity-level sentiment for competitors",
                "Number of headlines for the day",
                "14-day relative strength index",
                "12/26 EMA moving-average convergence divergence",
                "Volume-weighted average price",
                "Previous-day ensemble sentiment mean",
                "Previous-day return",
                "Previous-day volume (normalised)",
            ],
        }
    )
    features_full.to_csv(base_dir / "table_features_full.csv", index=False)

    # Hyperparameters table
    hyperparams = pd.DataFrame(
        {
            "parameter": [
                "iterations",
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "loss_function",
                "random_seed",
                "early_stopping_rounds",
            ],
            "value": [
                "1000",
                "0.03",
                "6",
                "3.0",
                "Logloss",
                "42",
                "50 (on rolling validation)",
            ],
        }
    )
    hyperparams.to_csv(base_dir / "table_hyperparams.csv", index=False)

    # Statistical tests summary
    stats = pd.DataFrame(
        {
            "test": [
                "Accuracy vs random (binomial)",
                "ROC-AUC vs 0.5 (DeLong)",
                "Diebold–Mariano (ML vs buy&hold returns)",
            ],
            "statistic": [4.1, 2.3, 2.15],
            "p_value": ["< 0.001", "0.021", "0.032"],
            "interpretation": [
                "Accuracy significantly better than 50%",
                "AUC significantly above random",
                "ML strategy returns differ from baseline",
            ],
        }
    )
    stats.to_csv(base_dir / "table_stats_tests.csv", index=False)


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory with prepared CSVs "
            "(roc_probs.csv, backtest.csv, shap.csv, consensus.csv, ablation.csv, "
            "monthly_accuracy.csv, daily_returns.csv, calibration.csv, predictions.csv "
            "or confusion_matrix.csv, corr_matrix.csv, leakage_summary.csv, "
            "table_*.csv)"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="docs/figures",
        help="Output directory for figures",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Generate mock data and plots",
    )
    p.add_argument(
        "--generate-tables",
        action="store_true",
        help="Generate corrected tables from df_pred.csv and existing templates",
    )
    p.add_argument(
        "--run-wf-cv",
        action="store_true",
        help="Run walk-forward CV with class balancing, calibration, and threshold tuning",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    data_dir = Path(args.data_dir) if args.data_dir else out_dir / "mock_data"
    if args.mock:
        make_mock_data(data_dir)

    if args.generate_tables:
        try:
            df_pred_path = data_dir / "df_pred.csv"
            if df_pred_path.exists():
                generate_table1_dataset(df_pred_path, data_dir / "table_dataset.csv")
                recompute_table3_enhanced(df_pred_path, data_dir / "table3_per_fold_enhanced.csv")
            fix_table2_features(data_dir / "table2_features.csv")
        except Exception as e:
            print(f"Failed to generate tables: {e}")

    if args.run_wf_cv:
        try:
            model_ready = pd.read_csv(data_dir / "model_ready_full.csv")
            probs_dir = data_dir
            cv_df = run_walk_forward_cv(model_ready, "date", "movement", probs_dir)
            foot = pd.DataFrame({
                "fold_id": ["note_binomial", "note_bootstrap"],
                "train_period": ["-", "-"],
                "test_period": ["-", "-"],
                "test_n": ["-", "-"],
                "accuracy": [
                    "p-value is one-sided vs 0.50",
                    "95% CI via 1000 bootstrap resamples",
                ],
                "precision": ["-", "-"],
                "recall": ["-", "-"],
                "f1": ["-", "-"],
                "auc": ["-", "-"],
                "p_value": ["-", "-"],
                "ci_low": ["-", "-"],
                "ci_high": ["-", "-"],
                "class_balance_train_pct": ["-", "-"],
                "class_balance_test_pct": ["-", "-"],
                "threshold": ["-", "-"],
                "train_time_s": ["-", "-"],
            })
            out_df = pd.concat([cv_df, foot], ignore_index=True)
            out_df.to_csv(data_dir / "table_per_fold.csv", index=False)

            # Table 4: Cross-Ticker Generalization (last window)
            model_ready["date"] = pd.to_datetime(model_ready["date"])
            last_start = pd.to_datetime("2025-11-01")
            last_end = pd.to_datetime("2025-11-20")
            tickers = sorted(model_ready["ticker"].unique())
            feat_cols = [c for c in model_ready.columns if c not in {"date", "ticker", "movement"}]
            within_acc = []
            cross_acc = []
            for t in tickers:
                tr_df = model_ready[(model_ready["ticker"] == t) & (model_ready["date"] < last_start)]
                te_df_t = model_ready[(model_ready["ticker"] == t) & (model_ready["date"] >= last_start) & (model_ready["date"] <= last_end)]
                if len(tr_df) == 0 or len(te_df_t) == 0:
                    continue
                X_tr = tr_df[feat_cols]
                y_tr = tr_df["movement"].astype(int)
                model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3, random_seed=42, verbose=False, auto_class_weights="Balanced")
                model.fit(X_tr, y_tr)
                calibrated = CalibratedClassifierCV(model, cv=3)
                calibrated.fit(X_tr, y_tr)
                X_val = X_tr.iloc[-int(max(1, round(len(X_tr) * 0.15))):]
                y_val = y_tr.iloc[-int(max(1, round(len(y_tr) * 0.15))):]
                val_prob = calibrated.predict_proba(X_val)[:, 1] if len(X_val) else np.array([])
                t_star = 0.5 if len(val_prob) == 0 else _tune_threshold(y_val.values, val_prob)
                X_te_t = te_df_t[feat_cols]
                y_te_t = te_df_t["movement"].astype(int)
                prob_t = calibrated.predict_proba(X_te_t)[:, 1]
                pred_t = (prob_t >= t_star).astype(int)
                within_acc.append(accuracy_score(y_te_t, pred_t))
                other = model_ready[(model_ready["ticker"] != t) & (model_ready["date"] >= last_start) & (model_ready["date"] <= last_end)]
                if len(other) > 0:
                    prob_o = calibrated.predict_proba(other[feat_cols])[:, 1]
                    pred_o = (prob_o >= t_star).astype(int)
                    cross_acc.append(accuracy_score(other["movement"].astype(int), pred_o))
            def _ci(vals):
                if len(vals) == 0:
                    return ("-", "-")
                # Remove NaN values for CI calculation
                vals = np.array(vals)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    return ("-", "-")
                rng = np.random.default_rng(42)
                boots = []
                for _ in range(1000):
                    idx = rng.integers(0, len(vals), size=len(vals))
                    boots.append(np.mean(vals[idx]))
                boots = np.array(boots)
                return (float(np.round(np.quantile(boots, 0.025), 3)), 
                        float(np.round(np.quantile(boots, 0.975), 3)))
            # Calculate means with NaN handling
            w_mean = float(np.round(np.nanmean(within_acc) if len(within_acc) > 0 else np.nan, 3))
            c_mean = float(np.round(np.nanmean(cross_acc) if len(cross_acc) > 0 else np.nan, 3))
            
            # Calculate CIs with NaN handling
            w_low, w_high = _ci(within_acc)
            c_low, c_high = _ci(cross_acc)
            
            # Calculate generalization gap, handling NaN cases
            if np.isnan(w_mean) or np.isnan(c_mean):
                g_gap = "-"
            else:
                g_gap = float(np.round(w_mean - c_mean, 3))
            table4 = pd.DataFrame({
                "metric": ["Within-ticker mean", "Cross-ticker mean", "Generalization gap", "Random baseline"],
                "value": [w_mean if not np.isnan(w_mean) else "-", 
                         c_mean if not np.isnan(c_mean) else "-", 
                         g_gap, 0.5],
                "ci_low": [w_low, c_low, "-", "-"],
                "ci_high": [w_high, c_high, "-", "-"],
                "note": [
                    "Last fold window (NaN values excluded from mean/CI)",
                    "Applied each model to other tickers (NaN values excluded from mean/CI)",
                    f"Difference (NaN if either mean is NaN)",
                    "50% (theoretical minimum)",
                ],
            })
            table4.to_csv(data_dir / "table_cross_ticker_summary.csv", index=False)

            # Table 5: Ablation Study (summary)
            full_mean = float(out_df[out_df["fold_id"] == "mean"]["accuracy"].values[0]) if (out_df["fold_id"] == "mean").any() else 0.0
            ab = pd.DataFrame({
                "experiment": ["Full model"],
                "accuracy": [full_mean],
                "ci_low": ["-"],
                "ci_high": ["-"],
                "delta_vs_full": [0.0],
                "protocol": ["walk-forward"],
            })
            ab.to_csv(data_dir / "table_ablation.csv", index=False)

            # Table 6: Computational Efficiency
            train_time_mean = float(out_df[out_df["fold_id"] == "mean"]["train_time_s"].values[0]) if (out_df["fold_id"] == "mean").any() else 0.0
            eff = pd.DataFrame({
                "component": ["CatBoost training", "CatBoost inference", "FinBERT training"],
                "time": [train_time_mean, "~ ms/sample", "N/A"],
                "cpu": ["CPU-only", "CPU-only", "N/A"],
                "params": ["iterations=500, depth=6", "batch_size=1", "N/A"],
            })
            eff.to_csv(data_dir / "table_efficiency.csv", index=False)

            # Table 7: Cross-Ticker Matrix (supplementary decision)
            sup = pd.DataFrame({"note": ["Not included due to n=13 instability in last window"]})
            sup.to_csv(data_dir / "table_cross_ticker_matrix_note.csv", index=False)
        except Exception as e:
            print(f"Failed to run walk-forward CV: {e}")

    # Core figures
    figure1_walk_forward(out_dir / "figure1_walk_forward.png")
    figure2_pipeline(out_dir / "figure2_pipeline.png")

    try:
        shap_df = pd.read_csv(data_dir / "shap.csv")
        figure3_shap_bar(shap_df, out_dir / "figure3_shap.png")
    except Exception as e:
        print("shap.csv not found — skipping Figure 3", e)

    try:
        cons = pd.read_csv(data_dir / "consensus.csv")["consensus"]
        figure4_disagreement_hist(cons, out_dir / "figure4_consensus_hist.png")
    except Exception as e:
        print("consensus.csv not found — skipping Figure 4", e)

    try:
        roc = pd.read_csv(data_dir / "roc_probs.csv")
        y = roc["y_true"]
        probs = {c: roc[c].values for c in roc.columns if c != "y_true"}
        figure5_roc_curves(y, probs, out_dir / "figure5_roc.png")
    except Exception as e:
        print("roc_probs.csv not found — skipping Figure 5", e)

    try:
        back = pd.read_csv(data_dir / "backtest.csv")
        back["date"] = pd.to_datetime(back["date"])
        figure6_backtest_equity_curve(back, out_dir / "figure6_backtest.png")
    except Exception as e:
        print("backtest.csv not found — skipping Figure 6", e)

    try:
        ab = pd.read_csv(data_dir / "ablation.csv")
        figure7_ablation_bar(ab, out_dir / "figure7_ablation.png")
    except Exception as e:
        print("ablation.csv not found — skipping Figure 7", e)

    try:
        monthly = pd.read_csv(data_dir / "monthly_accuracy.csv")
        figure8_temporal_stability(monthly, out_dir / "figure8_monthly_accuracy.png")
    except Exception as e:
        print("monthly_accuracy.csv not found — skipping Figure 8", e)

    # New figures
    try:
        daily = pd.read_csv(data_dir / "daily_returns.csv")
        figure9_return_distribution(daily, out_dir / "figure9_return_dist.png")
    except Exception as e:
        print("daily_returns.csv not found — skipping Figure 9", e)

    try:
        calib = pd.read_csv(data_dir / "calibration.csv")
        figure10_calibration(calib, out_dir / "figure10_calibration.png")
    except Exception as e:
        print("calibration.csv not found — skipping Figure 10", e)

    try:
        # prefer predictions.csv; fall back to confusion_matrix.csv if present
        pred_path = data_dir / "predictions.csv"
        if pred_path.exists():
            preds = pd.read_csv(pred_path)
        else:
            preds = pd.read_csv(data_dir / "confusion_matrix.csv").set_index(
                preds.columns[0]
            )
        figure11_confusion(preds, out_dir / "figure11_confusion.png")
    except Exception as e:
        print("predictions.csv / confusion_matrix.csv not found — skipping Figure 11", e)

    try:
        corr = pd.read_csv(data_dir / "corr_matrix.csv", index_col=0)
        figure12_corr_heatmap(corr, out_dir / "figure12_corr_heatmap.png")
    except Exception as e:
        print("corr_matrix.csv not found — skipping Figure 12", e)

    try:
        leak_df = pd.read_csv(data_dir / "leakage_summary.csv")
        figure13_leakage_bar(leak_df, out_dir / "figure13_leakage.png")
    except Exception as e:
        print("leakage_summary.csv not found — skipping Figure 13", e)

    # Entity-level example does not need data
    figure14_entity_example(out_dir / "figure14_entity_example.png")

    # Tables
    try:
        df = pd.read_csv(data_dir / "table_dataset.csv")
        render_table_image(
            df, out_dir / "table1_dataset.png", title="Table 1 — Dataset Summary"
        )
    except Exception as e:
        print("table_dataset.csv not found — skipping Table 1", e)

    try:
        tf2 = data_dir / "table2_features.csv"
        tf_legacy = data_dir / "table_features.csv"
        if tf2.exists():
            df = pd.read_csv(tf2)
        else:
            df = pd.read_csv(tf_legacy)
        render_table_image(
            df, out_dir / "table2_features.png", title="Table 2 — Feature Engineering Schema"
        )
    except Exception as e:
        print("table2_features.csv / table_features.csv not found — skipping Table 2", e)

    try:
        df = pd.read_csv(data_dir / "table_per_fold.csv")
        render_table_image(
            df,
            out_dir / "table3_per_fold.png",
            title="Table 3 — Per-Fold Performance",
        )
    except Exception as e:
        print("table_per_fold.csv not found — skipping Table 3", e)

    try:
        df = pd.read_csv(data_dir / "table_cross_ticker_summary.csv")
        render_table_image(
            df,
            out_dir / "table4_cross_ticker.png",
            title="Table 4 — Cross-Ticker Generalization",
        )
    except Exception as e:
        print("table_cross_ticker_summary.csv not found — skipping Table 4", e)

    try:
        df = pd.read_csv(data_dir / "table_ablation.csv")
        render_table_image(
            df,
            out_dir / "table5_ablation.png",
            title="Table 5 — Ablation Study",
        )
    except Exception as e:
        print("table_ablation.csv not found — skipping Table 5", e)

    try:
        df = pd.read_csv(data_dir / "table_efficiency.csv")
        render_table_image(
            df,
            out_dir / "table6_efficiency.png",
            title="Table 6 — Computational Efficiency",
        )
    except Exception as e:
        print("table_efficiency.csv not found — skipping Table 6", e)

    try:
        df = pd.read_csv(data_dir / "table_cross_ticker_matrix_note.csv")
        render_table_image(
            df,
            out_dir / "table7_cross_ticker_note.png",
            title="Table 7 — Cross-Ticker Matrix (Supplementary)",
        )
    except Exception as e:
        print("table_cross_ticker_matrix_note.csv not found — skipping Table 7", e)

    print(f"All generated figures and tables saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
