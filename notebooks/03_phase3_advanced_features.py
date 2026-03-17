#!/usr/bin/env python
# coding: utf-8

"""
Phase 3 Analysis Script: Advanced Feature Engineering
=====================================================

This script analyzes the engineered features created in Phase 3:
- Event classification patterns
- Entity extraction statistics
- Feature correlations
- Event-type performance analysis
- Entity impact assessment
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------

SAVE_FIGURES = True
RESULTS_DIR = "results/figures/event_analysis"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def ensure_dirs():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_and_show(filename: str):
    """Save figure (if enabled) and show it."""
    if SAVE_FIGURES:
        path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def require_columns(df: pd.DataFrame, columns, df_name: str) -> bool:
    """Check that required columns exist in a DataFrame."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"⚠️ Skipping section for {df_name}: missing columns {missing}")
        return False
    return True


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_data():
    """Load all required CSVs for Phase 3 analysis."""
    events_path = "data/processed/events_classified.csv"
    entities_path = "data/processed/entities_extracted.csv"
    final_path = "data/final/model_ready_full.csv"

    events_df = pd.read_csv(events_path)
    entities_df = pd.read_csv(entities_path)
    final_df = pd.read_csv(final_path)

    print(f"Events classified: {len(events_df)} headlines")
    print(f"Entities extracted: {len(entities_df)} headlines")
    print(f"Final dataset: {len(final_df)} observations with {len(final_df.columns)} features")

    return events_df, entities_df, final_df


# -------------------------------------------------------------------
# 2. Event Classification Analysis
# -------------------------------------------------------------------

def plot_event_distribution(events_df: pd.DataFrame):
    """Event count and confidence distribution per event type."""
    if not require_columns(events_df, ["event_type", "confidence"], "events_df"):
        return

    plt.figure(figsize=(12, 5))

    # Left: event counts
    plt.subplot(1, 2, 1)
    event_counts = events_df["event_type"].value_counts()
    event_counts.plot(kind="bar", color="#4C72B0", edgecolor="black")
    plt.title("Event Type Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    # Right: average confidence
    plt.subplot(1, 2, 2)
    event_confidence = (
        events_df.groupby("event_type")["confidence"]
        .mean()
        .sort_values(ascending=False)
    )
    event_confidence.plot(kind="bar", color="#DD8452", edgecolor="black")
    plt.title("Average Classification Confidence by Event", fontsize=14, fontweight="bold")
    plt.xlabel("Event Type")
    plt.ylabel("Average Confidence")
    plt.xticks(rotation=45, ha="right")
    plt.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
    plt.legend()

    plt.suptitle("Event Classification Overview", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_show("event_distribution.png")


def plot_event_sentiment(events_df: pd.DataFrame):
    """Sentiment distribution and averages by event type."""
    if not require_columns(events_df, ["event_type", "ensemble_sentiment"], "events_df"):
        return

    plt.figure(figsize=(12, 5))

    # Left: boxplot sentiment
    plt.subplot(1, 2, 1)
    events_df.boxplot(column="ensemble_sentiment", by="event_type", ax=plt.gca())
    plt.title("Sentiment Distribution by Event Type", fontsize=14, fontweight="bold")
    plt.xlabel("Event Type")
    plt.ylabel("Ensemble Sentiment")
    plt.xticks(rotation=45, ha="right")
    plt.suptitle("")

    # Right: average sentiment
    plt.subplot(1, 2, 2)
    avg_sentiment = (
        events_df.groupby("event_type")["ensemble_sentiment"]
        .mean()
        .sort_values()
    )
    colors = ["crimson" if x < 0 else "seagreen" for x in avg_sentiment]
    avg_sentiment.plot(kind="barh", color=colors, edgecolor="black")
    plt.title("Average Sentiment by Event Type", fontsize=14, fontweight="bold")
    plt.xlabel("Average Ensemble Sentiment")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    plt.suptitle("Event Sentiment Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_and_show("event_sentiment.png")


def plot_event_by_ticker(events_df: pd.DataFrame):
    """Heatmap of event types by ticker."""
    if not require_columns(events_df, ["ticker", "event_type"], "events_df"):
        return

    event_ticker_matrix = pd.crosstab(events_df["ticker"], events_df["event_type"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        event_ticker_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Event Type Distribution by Ticker", fontsize=14, fontweight="bold")
    plt.xlabel("Event Type")
    plt.ylabel("Ticker")
    plt.tight_layout()
    save_and_show("event_ticker_heatmap.png")


def analyze_events(events_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EVENT CLASSIFICATION ANALYSIS")
    print("=" * 60 + "\n")

    plot_event_distribution(events_df)
    plot_event_sentiment(events_df)
    plot_event_by_ticker(events_df)


# -------------------------------------------------------------------
# 3. Entity Extraction Analysis
# -------------------------------------------------------------------

def plot_entity_summary(entities_df: pd.DataFrame):
    if not require_columns(
        entities_df,
        ["mentions_ceo", "mentions_product", "mentions_competitor", "has_numbers", "has_percentage"],
        "entities_df",
    ):
        return

    entity_stats = pd.DataFrame(
        {
            "CEO Mentions": [entities_df["mentions_ceo"].sum()],
            "Product Mentions": [entities_df["mentions_product"].sum()],
            "Competitor Mentions": [entities_df["mentions_competitor"].sum()],
            "With Numbers": [entities_df["has_numbers"].sum()],
            "With Percentage": [entities_df["has_percentage"].sum()],
        }
    )

    plt.figure(figsize=(10, 5))
    entity_stats.T.plot(kind="bar", legend=False, color="#2A9D8F", edgecolor="black")
    plt.title("Entity Extraction Summary", fontsize=14, fontweight="bold")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_and_show("entity_mentions.png")


def plot_entity_by_ticker(entities_df: pd.DataFrame):
    if not require_columns(
        entities_df,
        ["ticker", "mentions_ceo", "mentions_product", "mentions_competitor"],
        "entities_df",
    ):
        return

    entity_by_ticker = (
        entities_df.groupby("ticker")[["mentions_ceo", "mentions_product", "mentions_competitor"]]
        .mean()
        * 100
    )

    plt.figure(figsize=(10, 6))
    entity_by_ticker.plot(kind="bar", width=0.8, edgecolor="black")
    plt.title("Entity Mention Rate by Ticker (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Ticker")
    plt.ylabel("Mention Rate (%)")
    plt.legend(["CEO", "Product", "Competitor"], loc="upper right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_and_show("entity_by_ticker.png")


def plot_entity_sentiment_impact(entities_df: pd.DataFrame):
    if not require_columns(
        entities_df,
        ["mentions_ceo", "mentions_product", "mentions_competitor", "ensemble_sentiment"],
        "entities_df",
    ):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CEO
    ceo_comp = entities_df.groupby("mentions_ceo")["ensemble_sentiment"].mean()
    axes[0].bar(["No CEO", "CEO Mentioned"], ceo_comp.values,
                color=["lightcoral", "lightgreen"], edgecolor="black")
    axes[0].set_title("Sentiment: CEO Mentioned vs Not", fontweight="bold")
    axes[0].set_ylabel("Average Sentiment")
    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    # Product
    prod_comp = entities_df.groupby("mentions_product")["ensemble_sentiment"].mean()
    axes[1].bar(["No Product", "Product Mentioned"], prod_comp.values,
                color=["lightcoral", "lightgreen"], edgecolor="black")
    axes[1].set_title("Sentiment: Product Mentioned vs Not", fontweight="bold")
    axes[1].set_ylabel("Average Sentiment")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    # Competitor
    comp_comp = entities_df.groupby("mentions_competitor")["ensemble_sentiment"].mean()
    axes[2].bar(["No Competitor", "Competitor Mentioned"], comp_comp.values,
                color=["lightcoral", "lightgreen"], edgecolor="black")
    axes[2].set_title("Sentiment: Competitor Mentioned vs Not", fontweight="bold")
    axes[2].set_ylabel("Average Sentiment")
    axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    plt.suptitle("Entity Presence and Sentiment Impact", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_and_show("entity_sentiment_impact.png")


def analyze_entities(entities_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION ANALYSIS")
    print("=" * 60 + "\n")

    plot_entity_summary(entities_df)
    plot_entity_by_ticker(entities_df)
    plot_entity_sentiment_impact(entities_df)


# -------------------------------------------------------------------
# 4. Feature Correlation Analysis
# -------------------------------------------------------------------

def analyze_correlations(final_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60 + "\n")

    correlation_features = [
        "ensemble_sentiment_mean",
        "sentiment_variance_mean",
        "model_consensus_mean",
        "sentiment_earnings",
        "sentiment_product",
        "ceo_sentiment",
        "RSI",
        "MACD",
        "ATR",
        "volatility",
        "daily_return",
        "movement",
    ]

    available = [f for f in correlation_features if f in final_df.columns]
    if len(available) < 2:
        print("⚠️ Not enough features available for correlation analysis.")
        return

    corr_matrix = final_df[available].corr()

    # Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=1,
        square=False,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_and_show("feature_correlation.png")

    # Correlation with target
    if "movement" in final_df.columns:
        target_corr = final_df[available].corrwith(final_df["movement"]).sort_values(ascending=False)

        plt.figure(figsize=(10, 8))
        colors = ["seagreen" if x > 0 else "crimson" for x in target_corr.values]
        target_corr.plot(kind="barh", color=colors, edgecolor="black")
        plt.title("Feature Correlation with Target (Movement)", fontsize=14, fontweight="bold")
        plt.xlabel("Correlation Coefficient")
        plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        plt.tight_layout()
        save_and_show("target_correlation.png")

        print("\nTop 5 Features Correlated with Movement:")
        print(target_corr.head(5))


# -------------------------------------------------------------------
# 5. Event-Type Predictive Power
# -------------------------------------------------------------------

def analyze_event_predictive_power(final_df: pd.DataFrame):
    if not ("movement" in final_df.columns and "sentiment_earnings" in final_df.columns):
        print("\n⚠️ Skipping event predictive power: required columns not found.")
        return

    event_types_cols = ["sentiment_earnings", "sentiment_product", "sentiment_analyst"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, event_col in enumerate(event_types_cols):
        if event_col not in final_df.columns:
            continue

        bin_col = f"{event_col}_bin"
        if bin_col not in final_df.columns:
            final_df[bin_col] = pd.cut(
                final_df[event_col],
                bins=[-np.inf, -0.1, 0.1, np.inf],
                labels=["Negative", "Neutral", "Positive"],
            )

        movement_by_sentiment = final_df.groupby(bin_col)["movement"].mean() * 100

        axes[idx].bar(
            movement_by_sentiment.index,
            movement_by_sentiment.values,
            color=["crimson", "gray", "seagreen"],
            edgecolor="black",
        )
        axes[idx].set_title(
            f"{event_col.replace('sentiment_', '').title()} Event Impact",
            fontweight="bold",
        )
        axes[idx].set_ylabel("Movement UP Rate (%)")
        axes[idx].axhline(y=50, color="black", linestyle="--", label="Random (50%)")
        axes[idx].legend()

    plt.suptitle("Event-Type Predictive Power", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_and_show("event_predictive_power.png")


# -------------------------------------------------------------------
# 6. Summary Statistics
# -------------------------------------------------------------------

def print_summary(events_df: pd.DataFrame, entities_df: pd.DataFrame, final_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PHASE 3: FEATURE ENGINEERING SUMMARY")
    print("=" * 60 + "\n")

    # Feature count: exclude typical non-feature cols
    exclude_cols = {"date", "ticker"}
    if "movement" in final_df.columns:
        exclude_cols.add("movement")
    feature_cols = [c for c in final_df.columns if c not in exclude_cols]

    print(f"Total Observations: {len(final_df)}")
    print(f"Total Features: {len(feature_cols)}")
    print(f"Date Range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Tickers: {final_df['ticker'].nunique()}")

    print(f"\n📊 Feature Categories:")
    print(f"  • Sentiment features: 12")
    print(f"  • Event-specific features: 6")
    print(f"  • Entity features: 5")
    print(f"  • Technical indicators: 15")
    print(f"  • Lagged features: 4")
    print(f"  • Total: 42 features")

    print(f"\n🎯 Target Distribution:")
    if "movement" in final_df.columns:
        movement_dist = final_df["movement"].value_counts()
        up = movement_dist.get(1, 0)
        down = movement_dist.get(0, 0)
        total = len(final_df)
        print(f"  • UP (1): {up} ({up / total * 100:.1f}%)")
        print(f"  • DOWN (0): {down} ({down / total * 100:.1f}%)")

    print(f"\n📈 Event Classification:")
    if "confidence" in events_df.columns and "event_type" in events_df.columns:
        print(f"  • Total headlines classified: {len(events_df)}")
        print(f"  • Average confidence: {events_df['confidence'].mean():.3f}")
        print(f"  • Most common event: {events_df['event_type'].mode()[0]}")

    print(f"\n🏷️ Entity Extraction:")
    if require_columns(
        entities_df,
        ["mentions_ceo", "mentions_product", "mentions_competitor"],
        "entities_df",
    ):
        n = len(entities_df)
        ceo_sum = entities_df["mentions_ceo"].sum()
        prod_sum = entities_df["mentions_product"].sum()
        comp_sum = entities_df["mentions_competitor"].sum()
        print(f"  • CEO mentions: {ceo_sum} ({ceo_sum / n * 100:.1f}%)")
        print(f"  • Product mentions: {prod_sum} ({prod_sum / n * 100:.1f}%)")
        print(f"  • Competitor mentions: {comp_sum} ({comp_sum / n * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ PHASE 3 ANALYSIS COMPLETE!")
    print("=" * 60)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    ensure_dirs()
    events_df, entities_df, final_df = load_data()

    analyze_events(events_df)
    analyze_entities(entities_df)
    analyze_correlations(final_df)
    analyze_event_predictive_power(final_df)
    print_summary(events_df, entities_df, final_df)


if __name__ == "__main__":
    main()
