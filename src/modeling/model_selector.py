"""
Model Selector - Phase 4 (Day 26-27)
====================================
Loads baseline and advanced performance metrics, merges them safely,
and selects the best model using F1 score (then accuracy).
"""

import json
from pathlib import Path
import pandas as pd


def load_metrics():
    baseline_path = Path("results/metrics/baseline_performance.csv")
    advanced_path = Path("results/metrics/advanced_model_performance.csv")

    dfs = []

    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)

        # keep only required columns & rename f1_score → f1
        df_base = df_base[["model", "accuracy", "f1_score"]].rename(
            columns={"f1_score": "f1"}
        )
        dfs.append(df_base)

    if advanced_path.exists():
        df_adv = pd.read_csv(advanced_path)

        # advanced has only model + f1_cv
        df_adv = df_adv[["model", "f1_cv"]].rename(
            columns={"f1_cv": "f1"}
        )

        # add accuracy placeholder so both dataframes align
        df_adv["accuracy"] = None

        dfs.append(df_adv)

    if not dfs:
        raise FileNotFoundError("No metrics files found.")

    # merge vertically
    all_metrics = pd.concat(dfs, ignore_index=True)

    return all_metrics


def select_best_model(df):
    # ensure no duplicates
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # sort: highest F1 first, then accuracy
    df_sorted = df.sort_values(
        by=["f1", "accuracy"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best = df_sorted.iloc[0].to_dict()
    return best, df_sorted


def main():
    print("\n" + "="*70)
    print("PHASE 4 - MODEL SELECTION")
    print("="*70 + "\n")

    all_metrics = load_metrics()

    print("📊 All Models:")
    print(all_metrics.to_string(index=False))

    best, ranked = select_best_model(all_metrics)

    print("\n🏆 Best Model Selected:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    summary = {
        "best_model_name": best["model"],
        "best_f1_score": float(best["f1"]),
        "best_accuracy": float(best["accuracy"]) if best["accuracy"] is not None else None,
        "all_models_sorted": ranked.to_dict(orient="records")
    }

    output_path = Path("models/final_model_summary.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved final summary: {output_path}")


if __name__ == "__main__":
    main()
