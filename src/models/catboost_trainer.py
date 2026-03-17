"""
CatBoost trainer for per-ticker stock movement prediction.

Handles:
- Per-ticker model training with chronological train/val/test splits
- Walk-forward validation with expanding windows
- Hyperparameter tuning on validation set
- Threshold optimization
- Cross-ticker generalization evaluation
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import warnings
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.evaluation.metrics import (
    compute_metrics, find_optimal_threshold, bootstrap_confidence_interval,
    binomial_significance_test, diebold_mariano_test
)

logger = logging.getLogger(__name__)


class PerTickerSplitter:
    """
    Chronological train/val/test splitter for per-ticker data.
    
    No random shuffling, no cross-ticker mixing.
    """
    
    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """
        Parameters
        ----------
        train_ratio : float
            Ratio for training set
        val_ratio : float
            Ratio for validation set
        test_ratio : float
            Ratio for test set
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, df, ticker):
        """
        Split ticker data chronologically.
        
        Parameters
        ----------
        df : pd.DataFrame
            Full dataset
        ticker : str
            Stock ticker symbol
        
        Returns
        -------
        tuple
            (train_df, val_df, test_df)
        """
        # Filter to ticker and sort by date
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date').reset_index(drop=True)
        
        n = len(ticker_df)
        train_end = int(np.ceil(n * self.train_ratio))
        val_end = train_end + int(np.ceil(n * self.val_ratio))
        
        train_df = ticker_df.iloc[:train_end].copy()
        val_df = ticker_df.iloc[train_end:val_end].copy()
        test_df = ticker_df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df


class WalkForwardValidator:
    """
    Walk-forward validation with expanding training windows.
    """
    
    def __init__(self, min_train_size=150, step_size=20, test_size=None):
        """
        Parameters
        ----------
        min_train_size : int
            Minimum training window size
        step_size : int
            Step size for expanding window
        test_size : int, optional
            Fixed test set size; if None, use remaining data
        """
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.test_size = test_size
    
    def iter_folds(self, df, ticker):
        """
        Iterate over walk-forward folds.
        
        Parameters
        ----------
        df : pd.DataFrame
            Ticker-filtered dataset (already sorted chronologically)
        ticker : str
            Stock ticker
        
        Yields
        ------
        tuple
            (fold_num, train_df, test_df)
        """
        df = df.sort_values('date').reset_index(drop=True)
        n = len(df)
        
        fold_num = 0
        train_end = self.min_train_size
        
        while train_end < n:
            if self.test_size is not None:
                test_end = min(train_end + self.test_size, n)
            else:
                test_end = n
            
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            if len(test_df) > 0:
                yield fold_num, train_df, test_df
                fold_num += 1
            
            train_end += self.step_size


class CatBoostTrainer:
    """
    CatBoost trainer with hyperparameter tuning and threshold optimization.
    """
    
    def __init__(self, random_state=42, verbose=100, n_jobs=4, ticker=None):
        """
        Parameters
        ----------
        random_state : int
            Random seed
        verbose : int
            CatBoost verbosity level
        n_jobs : int
            Number of CPU cores to use
        ticker : str, optional
            Stock ticker - if provided, generates per-ticker seed for model diversity
        """
        # If ticker is provided, derive a unique seed from it for model diversity
        if ticker is not None:
            import hashlib
            ticker_seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16) % (2**32 - 1)
            self.random_state = ticker_seed % (2**31 - 1)  # Ensure valid 32-bit int
        else:
            self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.feature_names = None
        self.model = None
        self.threshold = 0.5
    
    def _build_model(self, iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3):
        """Create CatBoost model with specified hyperparameters."""
        return CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function='Logloss',
            eval_metric='AUC',
            random_state=self.random_state,
            verbose=self.verbose,
            thread_count=self.n_jobs,
            use_best_model=False,  # Only use best_model if eval_set is provided
            has_time=False,
            od_type='Iter',
            od_wait=50,
            allow_writing_files=False
        )
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Grid search hyperparameter tuning on validation set.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : array-like
            Training targets
        X_val : pd.DataFrame or np.ndarray
            Validation features
        y_val : array-like
            Validation targets
        
        Returns
        -------
        dict
            Best hyperparameters and validation accuracy
        """
        param_grid = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.03, 0.05],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        best_accuracy = 0
        best_params = param_grid
        
        logger.info(f"Tuning hyperparameters... ({np.prod([len(v) for v in param_grid.values()])} combinations)")
        
        combo_count = 0
        for iterations in param_grid['iterations']:
            for learning_rate in param_grid['learning_rate']:
                for depth in param_grid['depth']:
                    for l2_leaf_reg in param_grid['l2_leaf_reg']:
                        combo_count += 1
                        
                        model = self._build_model(
                            iterations=iterations,
                            learning_rate=learning_rate,
                            depth=depth,
                            l2_leaf_reg=l2_leaf_reg
                        )
                        
                        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                        
                        y_pred = model.predict(X_val)
                        accuracy = accuracy_score(y_val, y_pred)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                'iterations': iterations,
                                'learning_rate': learning_rate,
                                'depth': depth,
                                'l2_leaf_reg': l2_leaf_reg
                            }
                            logger.info(f"  [{combo_count}] New best: accuracy={best_accuracy:.4f}, params={best_params}")
        
        logger.info(f"Hyperparameter tuning complete. Best validation accuracy: {best_accuracy:.4f}")
        
        return best_params, best_accuracy
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, tune_hyperparameters=True,
            optimize_threshold=True):
        """
        Train model with optional hyperparameter tuning and threshold optimization.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features
        y_train : array-like
            Training targets
        X_val : pd.DataFrame or np.ndarray, optional
            Validation features (used for tuning and calibration)
        y_val : array-like, optional
            Validation targets
        tune_hyperparameters : bool
            Whether to perform hyperparameter tuning
        optimize_threshold : bool
            Whether to optimize decision threshold
        
        Returns
        -------
        dict
            Training results including final hyperparameters and validation metrics
        """
        self.feature_names = X_train.columns if hasattr(X_train, 'columns') else None
        
        results = {}
        
        # Hyperparameter tuning
        if tune_hyperparameters and X_val is not None and y_val is not None:
            best_params, val_accuracy = self.tune_hyperparameters(X_train, y_train, X_val, y_val)
            results['best_params'] = best_params
            results['val_accuracy'] = val_accuracy
        else:
            best_params = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3
            }
        
        # Train final model
        logger.info(f"Training final model with params: {best_params}")
        self.model = self._build_model(**best_params)
        
        if X_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        # Threshold optimization (on validation set if available)
        if optimize_threshold and X_val is not None:
            y_val_proba = self.model.predict_proba(X_val)[:, 1]
            threshold_results = find_optimal_threshold(y_val, y_val_proba, metric='f1', penalty=0.1)
            self.threshold = threshold_results['optimal_threshold']
            results['threshold_results'] = threshold_results
            logger.info(f"Optimal threshold: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
        
        return results
    
    def predict(self, X, return_proba=False):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features
        return_proba : bool
            Whether to return probabilities
        
        Returns
        -------
        array or tuple
            Predictions (and probabilities if return_proba=True)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        if return_proba:
            return y_pred, y_pred_proba
        return y_pred
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from trained model.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame
            Feature importance (sorted by gain)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        importance = self.model.get_feature_importance()
        
        if self.feature_names is not None:
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            })
        
        feature_df = feature_df.sort_values('importance', ascending=False)
        
        return feature_df.head(top_n)


class BaselineModels:
    """
    Simple baseline models for comparison.
    """
    
    @staticmethod
    def random_classifier(y_test, random_state=42):
        """
        Random classifier (50% baseline).
        """
        np.random.seed(random_state)
        y_pred = np.random.binomial(1, 0.5, len(y_test))
        y_pred_proba = np.random.uniform(0, 1, len(y_test))
        
        return y_pred, y_pred_proba
    
    @staticmethod
    def logistic_regression(X_train, y_train, X_test):
        """
        Logistic regression baseline.
        
        Drops any columns with NaN values before training.
        """
        # Drop columns with NaN
        if isinstance(X_train, pd.DataFrame):
            valid_cols = X_train.columns[~X_train.isna().any()]
            X_train_clean = X_train[valid_cols].values
            X_test_clean = X_test[valid_cols].values
        else:
            X_train_clean = X_train
            X_test_clean = X_test
        
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train_clean, y_train)
        
        y_pred = model.predict(X_test_clean)
        y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
        
        return y_pred, y_pred_proba
    
    @staticmethod
    def technical_only(X_train, y_train, X_test, exclude_cols=None):
        """
        CatBoost model trained on technical features only.
        """
        if exclude_cols is None:
            # NLP features to exclude
            nlp_features = [
                'finbert_score', 'vader_polarity', 'textblob_polarity',
                'ensemble_mean', 'model_disagreement', 'consensus_score',
                'num_headlines', 'num_positive', 'num_negative',
                'ceo_sentiment', 'competitor_sentiment',
                'earnings_flag', 'upgrade_flag', 'downgrade_flag',
                'finbert_weight', 'vader_weight', 'textblob_weight', 'ensemble_confidence'
            ]
            exclude_cols = [c for c in nlp_features if c in X_train.columns]
        
        feature_cols = [c for c in X_train.columns if c not in exclude_cols]
        X_train_tech = X_train[feature_cols]
        X_test_tech = X_test[feature_cols]
        
        trainer = CatBoostTrainer(verbose=0)
        trainer.fit(X_train_tech, y_train, X_val=None, y_val=None, 
                   tune_hyperparameters=False, optimize_threshold=False)
        
        y_pred, y_pred_proba = trainer.predict(X_test_tech, return_proba=True)
        
        return y_pred, y_pred_proba


if __name__ == '__main__':
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10), columns=[f'feat_{i}' for i in range(10)])
    y = np.random.binomial(1, 0.5, 200)
    
    splitter = PerTickerSplitter()
    trainer = CatBoostTrainer(verbose=0)
    
    # Dummy split (normally done per-ticker)
    train_size = 140
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    trainer.fit(X_train, y_train, X_val=None, optimize_threshold=False)
    y_pred, y_pred_proba = trainer.predict(X_test, return_proba=True)
    
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Feature importance:\n{trainer.get_feature_importance(top_n=5)}")
