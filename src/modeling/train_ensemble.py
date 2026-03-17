"""
Advanced Model Training - Phase 4 (Day 24-26)
=============================================
Train advanced models:

- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble (XGB + LGBM + CB → Logistic Regression)

Uses:
- TimeSeriesSplit for CV
- HyperparameterTuner for RandomizedSearchCV
- Class weights for slight imbalance
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Gradient boosting libs
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Custom modules
import sys
sys.path.append("src/evaluation")
from time_series_cv import TimeSeriesSplit

sys.path.append("src/modeling")
from hyperparameter_tuner import HyperparameterTuner


class EnsembleTrainer:
    def __init__(self, cv_config_path: str = "models/cv_config.json"):
        # Load CV config
        if Path(cv_config_path).exists():
            with open(cv_config_path, "r") as f:
                cfg = json.load(f)
            self.n_splits = cfg["n_splits"]
            self.test_size = cfg["test_size"]
        else:
            self.n_splits = 5
            self.test_size = 30

        self.scaler = None
        self.class_weights_dict = None

    def prepare_data(self, df: pd.DataFrame):
        """
        Same logic as baseline: drop date, ticker, movement.
        """
        df_sorted = df.sort_values("date").reset_index(drop=True)
        exclude_cols = ["date", "ticker", "movement"]

        feature_cols = [c for c in df_sorted.columns if c not in exclude_cols]
        X = df_sorted[feature_cols].values
        y = df_sorted["movement"].values.astype(int)

        print("\n📊 Ensemble Data Preparation:")
        print(f"   Samples:  {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"   Target distribution: {dict(zip(unique, counts))}")

        # Compute class weights for boosting models
        n_pos = counts[unique.tolist().index(1)]
        n_neg = counts[unique.tolist().index(0)]
        # scale_pos_weight = neg / pos (classic XGBoost heuristic)
        scale_pos_weight = n_neg / n_pos
        self.class_weights_dict = {
            "scale_pos_weight": scale_pos_weight,
            "lgbm_class_weight": "balanced",
            "catboost_class_weights": [n_neg / (n_pos + n_neg), n_pos / (n_pos + n_neg)]
        }

        return X, y, feature_cols

    def get_xgb_base(self) -> XGBClassifier:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=self.class_weights_dict["scale_pos_weight"],
        )

    def get_lgbm_base(self) -> LGBMClassifier:
        return LGBMClassifier(
            objective="binary",
            class_weight=self.class_weights_dict["lgbm_class_weight"],
            random_state=42,
            n_jobs=-1
        )

    def get_catboost_base(self) -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            class_weights=self.class_weights_dict["catboost_class_weights"],
            random_seed=42,
            verbose=False
        )

    def get_param_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Parameter grids for each model.
        Keep modest but meaningful ranges.
        """
        xgb_params = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
        }

        lgbm_params = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [-1, 3, 5, 7, 9],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        catboost_params = {
            "depth": [3, 4, 5, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "iterations": [200, 300, 400, 500],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "border_count": [32, 64, 128],
        }

        return {
            "xgb": xgb_params,
            "lgbm": lgbm_params,
            "cat": catboost_params,
        }

    def time_series_cv_evaluate(self, model, X, y, model_name: str) -> dict:
        """
        Simple walk-forward evaluation (no tuning), mainly for stacking evaluation.
        Uses the stored n_splits and test_size.
        """
        print(f"\n🔍 Evaluating {model_name} with walk-forward CV...")

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=0
        )

        metrics = {
            "accuracy": [],
            "f1_score": [],
            "roc_auc": []
        }

        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["f1_score"].append(f1_score(y_test, y_pred))
            metrics["roc_auc"].append(roc_auc_score(y_test, y_proba))

            print(f"  Fold {fold}: "
                  f"Acc={metrics['accuracy'][-1]:.4f}, "
                  f"F1={metrics['f1_score'][-1]:.4f}, "
                  f"AUC={metrics['roc_auc'][-1]:.4f}")
            fold += 1

        summary = {
            "model": model_name,
            "accuracy_mean": float(np.mean(metrics["accuracy"])),
            "accuracy_std": float(np.std(metrics["accuracy"])),
            "f1_mean": float(np.mean(metrics["f1_score"])),
            "f1_std": float(np.std(metrics["f1_score"])),
            "roc_auc_mean": float(np.mean(metrics["roc_auc"])),
            "roc_auc_std": float(np.std(metrics["roc_auc"])),
        }

        print(f"\n📊 {model_name} CV Summary:")
        print(f"  Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        print(f"  F1:       {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"  ROC-AUC:  {summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}")

        return summary

    def train(self, df: pd.DataFrame):
        """
        Full training flow:
        - Prepare data
        - Tune XGB, LGBM, CatBoost
        - Train stacking ensemble
        - Save all models + metrics
        """
        os.makedirs("models", exist_ok=True)
        os.makedirs("results/metrics", exist_ok=True)

        X, y, feature_names = self.prepare_data(df)
        param_grids = self.get_param_distributions()

        tuner = HyperparameterTuner(
            n_splits=self.n_splits,
            test_size=self.test_size,
            n_iter=50,          # research-level
            scoring="f1"
        )

        # 1) Tune XGBoost
        xgb_base = self.get_xgb_base()
        xgb_best, xgb_cv = tuner.tune_model(
            xgb_base,
            param_grids["xgb"],
            X, y,
            model_name="XGBoost",
            n_jobs=-1,
            verbose=1
        )

        # 2) Tune LightGBM
        lgbm_base = self.get_lgbm_base()
        lgbm_best, lgbm_cv = tuner.tune_model(
            lgbm_base,
            param_grids["lgbm"],
            X, y,
            model_name="LightGBM",
            n_jobs=-1,
            verbose=1
        )

        # 3) Tune CatBoost
        cat_base = self.get_catboost_base()
        cat_best, cat_cv = tuner.tune_model(
            cat_base,
            param_grids["cat"],
            X, y,
            model_name="CatBoost",
            n_jobs=1,   # CatBoost handles threads internally
            verbose=1
        )

        # Save tuned base models
        with open("models/xgb_best.pkl", "wb") as f:
            pickle.dump(xgb_best, f)
        with open("models/lgbm_best.pkl", "wb") as f:
            pickle.dump(lgbm_best, f)
        with open("models/catboost_best.pkl", "wb") as f:
            pickle.dump(cat_best, f)

        print("\n💾 Saved tuned base models to models/")

        # 4) Stacking Ensemble
        print("\n" + "="*70)
        print("TRAINING STACKING ENSEMBLE")
        print("="*70)

        estimators = [
            ("xgb", xgb_best),
            ("lgbm", lgbm_best),
            ("cat", cat_best),
        ]

        final_estimator = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        # Important: scale features for meta-learner
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            stack_method="predict_proba",
            n_jobs=-1,
            passthrough=False
        )

        ensemble_cv_summary = self.time_series_cv_evaluate(
            stacking_clf, X_scaled, y, "Stacking Ensemble"
        )

        # Fit final ensemble on full data
        stacking_clf.fit(X_scaled, y)

        # Save ensemble + scaler
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(stacking_clf, f)
        with open("models/scaler_ensemble.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        print("\n💾 Saved ensemble model to models/best_model.pkl")
        print("💾 Saved ensemble scaler to models/scaler_ensemble.pkl")

        # Save metrics summary
        perf_rows = []

        def cv_to_row(cv_dict, name_key="model_name"):
            return {
                "model": cv_dict.get(name_key, cv_dict.get("model_name", "Unknown")),
                "f1_cv": cv_dict["best_score"],
            }

        perf_rows.append({"model": "XGBoost", "f1_cv": xgb_cv["best_score"]})
        perf_rows.append({"model": "LightGBM", "f1_cv": lgbm_cv["best_score"]})
        perf_rows.append({"model": "CatBoost", "f1_cv": cat_cv["best_score"]})
        perf_rows.append({
            "model": "Stacking Ensemble",
            "f1_cv": ensemble_cv_summary["f1_mean"]
        })

        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_csv("results/metrics/advanced_model_performance.csv", index=False)
        print("\n💾 Saved advanced model performance to results/metrics/advanced_model_performance.csv")

        # Also dump a JSON summary for model_selector.py
        comparison = {
            "xgboost": xgb_cv,
            "lightgbm": lgbm_cv,
            "catboost": cat_cv,
            "ensemble": ensemble_cv_summary
        }

        with open("models/model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print("💾 Saved model comparison JSON to models/model_comparison.json")


def main():
    print("\n" + "="*70)
    print("PHASE 4 - ADVANCED MODEL TRAINING (XGB, LGBM, CatBoost, Ensemble)")
    print("="*70 + "\n")

    data_file = "data/final/model_ready_full.csv"
    print(f"📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} rows")

    trainer = EnsembleTrainer()
    trainer.train(df)

    print("\n✅ Advanced model training complete!")


if __name__ == "__main__":
    main()
