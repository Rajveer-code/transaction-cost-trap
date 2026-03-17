"""
Hyperparameter Tuning Module - Phase 4 (Day 24-26)
==================================================
Centralized tuner using RandomizedSearchCV with custom TimeSeriesSplit.

- Uses walk-forward CV (no shuffling)
- Scoring: F1-score (binary classification)
- n_iter: 50 by default (research-grade)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

# Custom time-series CV
import sys
sys.path.append("src/evaluation")
from time_series_cv import TimeSeriesSplit


class HyperparameterTuner:
    """
    Wrapper around RandomizedSearchCV using TimeSeriesSplit.
    """

    def __init__(
        self,
        n_splits: int = None,
        test_size: int = None,
        n_iter: int = 50,
        scoring: str = "f1",
        random_state: int = 42,
    ):
        # Load CV config if available
        cv_config_path = Path("models/cv_config.json")
        if cv_config_path.exists():
            with open(cv_config_path, "r") as f:
                cfg = json.load(f)
            self.n_splits = n_splits or cfg["n_splits"]
            self.test_size = test_size or cfg["test_size"]
        else:
            self.n_splits = n_splits or 5
            self.test_size = test_size or 30

        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state

    def tune_model(
        self,
        model,
        param_distributions: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        n_jobs: int = -1,
        verbose: int = 1,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run RandomizedSearchCV with time-series cross-validation.

        Returns:
            best_estimator, results_dict
        """
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'='*70}")

        print(f"Samples: {len(X)}")
        print(f"Folds:   {self.n_splits}")
        print(f"Test size per fold: {self.test_size}")
        print(f"Scoring: {self.scoring}")
        print(f"Iterations: {self.n_iter}")

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=0
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=tscv,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,   # refit on best params
            return_train_score=False
        )

        start = time.time()
        search.fit(X, y)
        elapsed = time.time() - start

        print(f"\n✅ Tuning complete in {elapsed:.2f} seconds")
        print(f"Best F1-score (CV): {search.best_score_:.4f}")
        print(f"Best params:")
        for k, v in search.best_params_.items():
            print(f"  - {k}: {v}")

        # Collect CV results summary
        best_estimator = search.best_estimator_
        cv_results = {
            "model_name": model_name,
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "n_iter": self.n_iter,
            "scoring": self.scoring,
            "cv_time_sec": float(elapsed),
        }

        return best_estimator, cv_results
