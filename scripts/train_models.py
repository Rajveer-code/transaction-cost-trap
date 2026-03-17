"""
Main training script for stock movement prediction pipeline.

Executes:
1. Baseline models (random, logistic regression, technical-only)
2. Per-ticker CatBoost with walk-forward validation
3. Cross-ticker generalization evaluation
4. Ablation studies (full model, technical-only, rolling-returns-only)
5. Feature importance analysis
6. Backtest simulation
7. Comprehensive results reporting
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from src.models.catboost_trainer import (
    PerTickerSplitter, WalkForwardValidator, CatBoostTrainer, BaselineModels
)
from src.evaluation.metrics import (
    compute_metrics, find_optimal_threshold, bootstrap_confidence_interval,
    binomial_significance_test, diebold_mariano_test
)


class StockPredictionPipeline:
    """
    Complete modeling pipeline for stock movement prediction.
    """
    
    def __init__(self, data_dir='data/combined', output_dir='results',
                 figures_dir='research_outputs/figures'):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        data_dir : str
            Directory containing all_features.parquet
        output_dir : str
            Directory for results CSV files
        figures_dir : str
            Directory for figure outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.results = {}
        self.feature_importance_global = None
        
        logger.info(f"Pipeline initialized. Output: {self.output_dir}")
    
    def load_data(self):
        """Load combined dataset."""
        data_path = self.data_dir / 'all_features.parquet'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        self.df = pd.read_parquet(data_path)
        
        logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        logger.info(f"Tickers: {sorted(self.df['ticker'].unique())}")
        logger.info(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def prepare_features(self):
        """Prepare feature sets."""
        # Identify feature columns
        exclude_cols = {'date', 'ticker', 'next_day_return', 'binary_label'}
        
        nlp_features = [
            'finbert_score', 'vader_polarity', 'textblob_polarity',
            'ensemble_mean', 'model_disagreement', 'consensus_score',
            'num_headlines', 'num_positive', 'num_negative',
            'ceo_sentiment', 'competitor_sentiment',
            'earnings_flag', 'upgrade_flag', 'downgrade_flag',
            'finbert_weight', 'vader_weight', 'textblob_weight', 'ensemble_confidence'
        ]
        
        rolling_return_features = [
            'return_1d', 'return_2d', 'return_3d', 'return_5d', 
            'return_10d', 'return_20d', 'return_50d'
        ]
        
        # All features (excluding NaN and targets)
        all_features = [c for c in self.df.columns 
                       if c not in exclude_cols and c not in nlp_features]
        
        # Technical only (exclude rolling returns)
        technical_features = [c for c in all_features 
                             if c not in rolling_return_features]
        
        # Rolling returns only
        rolling_features = [c for c in rolling_return_features 
                           if c in self.df.columns]
        
        logger.info(f"Feature sets prepared:")
        logger.info(f"  - All features: {len(all_features)}")
        logger.info(f"  - Technical only: {len(technical_features)}")
        logger.info(f"  - Rolling returns: {len(rolling_features)}")
        
        return {
            'all': all_features,
            'technical': technical_features,
            'rolling_returns': rolling_features
        }
    
    def run_baseline_models(self, tickers, feature_cols):
        """
        Run baseline models (random, logistic regression, technical-only).
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        feature_cols : dict
            Feature column sets
        
        Returns
        -------
        dict
            Baseline results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING BASELINE MODELS")
        logger.info("="*80)
        
        baseline_results = {
            'random': [],
            'logistic_regression': [],
            'technical_only': []
        }
        
        splitter = PerTickerSplitter()
        
        for ticker in tickers:
            logger.info(f"\n--- {ticker} ---")
            
            train_df, val_df, test_df = splitter.split(self.df, ticker)
            
            # Use only non-NaN columns for features
            X_train = train_df[feature_cols['all']].dropna(axis=1, how='all')
            y_train = train_df['binary_label'].values
            X_test = test_df[feature_cols['all']][X_train.columns]
            y_test = test_df['binary_label'].values
            
            # Random baseline
            y_pred_rand, y_pred_proba_rand = BaselineModels.random_classifier(y_test)
            metrics_rand = compute_metrics(y_test, y_pred_rand, y_pred_proba_rand)
            baseline_results['random'].append({
                'ticker': ticker,
                **metrics_rand
            })
            logger.info(f"Random: accuracy={metrics_rand['accuracy']:.4f}")
            
            # Logistic regression
            y_pred_lr, y_pred_proba_lr = BaselineModels.logistic_regression(X_train, y_train, X_test)
            metrics_lr = compute_metrics(y_test, y_pred_lr, y_pred_proba_lr)
            baseline_results['logistic_regression'].append({
                'ticker': ticker,
                **metrics_lr
            })
            logger.info(f"Logistic Regression: accuracy={metrics_lr['accuracy']:.4f}")
            
            # Technical-only CatBoost
            y_pred_tech, y_pred_proba_tech = BaselineModels.technical_only(
                X_train, y_train, X_test
            )
            metrics_tech = compute_metrics(y_test, y_pred_tech, y_pred_proba_tech)
            baseline_results['technical_only'].append({
                'ticker': ticker,
                **metrics_tech
            })
            logger.info(f"Technical-only: accuracy={metrics_tech['accuracy']:.4f}")
        
        # Save baseline results
        for model_name, results_list in baseline_results.items():
            df_baseline = pd.DataFrame(results_list)
            output_path = self.output_dir / f'baseline_{model_name}.csv'
            df_baseline.to_csv(output_path, index=False)
            logger.info(f"\nSaved baseline results: {output_path}")
        
        return baseline_results
    
    def run_per_ticker_catboost(self, tickers, feature_cols):
        """
        Run per-ticker CatBoost with walk-forward validation.
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        feature_cols : dict
            Feature column sets
        
        Returns
        -------
        dict
            Walk-forward results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING PER-TICKER CATBOOST WITH WALK-FORWARD VALIDATION")
        logger.info("="*80)
        
        wf_results = []
        feature_importance_list = []
        
        for ticker in tickers:
            logger.info(f"\n--- {ticker} ---")
            
            ticker_df = self.df[self.df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            validator = WalkForwardValidator(min_train_size=150, step_size=20)
            
            fold_results = []
            
            for fold_num, train_df, test_df in validator.iter_folds(ticker_df, ticker):
                logger.info(f"  Fold {fold_num}: train={len(train_df)}, test={len(test_df)}")
                
                # Drop NaN columns
                X_train = train_df[feature_cols['all']].dropna(axis=1, how='all')
                y_train = train_df['binary_label'].values
                X_test = test_df[feature_cols['all']][X_train.columns]
                y_test = test_df['binary_label'].values
                
                # Train model (with per-ticker seed for diversity)
                trainer = CatBoostTrainer(verbose=0, ticker=ticker)
                trainer.fit(X_train, y_train, X_val=None, optimize_threshold=True)
                
                # Predict
                y_pred, y_pred_proba = trainer.predict(X_test, return_proba=True)
                
                # Metrics
                metrics = compute_metrics(y_test, y_pred, y_pred_proba, threshold=trainer.threshold)
                
                # Bootstrap CI
                ci_acc = bootstrap_confidence_interval(y_test, y_pred, metric='accuracy', n_iterations=1000)
                
                # Binomial significance test
                binom_test = binomial_significance_test(metrics['accuracy'], len(y_test))
                
                fold_result = {
                    'ticker': ticker,
                    'fold': fold_num,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'threshold': trainer.threshold,
                    'accuracy': metrics['accuracy'],
                    'accuracy_ci_lower': ci_acc['lower'],
                    'accuracy_ci_upper': ci_acc['upper'],
                    'roc_auc': metrics['roc_auc'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'binom_p_value': binom_test['p_value'],
                    'binom_significant': binom_test['is_significant']
                }
                
                fold_results.append(fold_result)
                wf_results.append(fold_result)
                
                logger.info(f"    Accuracy: {metrics['accuracy']:.4f} "
                          f"[{ci_acc['lower']:.4f}-{ci_acc['upper']:.4f}], "
                          f"AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
                
                # Feature importance
                feature_imp = trainer.get_feature_importance(top_n=15)
                feature_imp['ticker'] = ticker
                feature_imp['fold'] = fold_num
                feature_importance_list.append(feature_imp)
            
            # Per-ticker summary
            fold_df = pd.DataFrame(fold_results)
            mean_acc = fold_df['accuracy'].mean()
            logger.info(f"  {ticker} Mean Accuracy: {mean_acc:.4f}")
        
        # Save walk-forward results
        df_wf = pd.DataFrame(wf_results)
        output_path = self.output_dir / 'walk_forward_results.csv'
        df_wf.to_csv(output_path, index=False)
        logger.info(f"\nSaved walk-forward results: {output_path}")
        
        # Save feature importance
        df_fi = pd.concat(feature_importance_list, ignore_index=True)
        output_path = self.output_dir / 'feature_importance_walkforward.csv'
        df_fi.to_csv(output_path, index=False)
        logger.info(f"Saved feature importance: {output_path}")
        
        return wf_results, df_fi
    
    def run_cross_ticker_evaluation(self, tickers, feature_cols):
        """
        Evaluate cross-ticker generalization.
        
        Train on 6 tickers, test on held-out 7th.
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        feature_cols : dict
            Feature column sets
        
        Returns
        -------
        dict
            Cross-ticker results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING CROSS-TICKER GENERALIZATION EVALUATION")
        logger.info("="*80)
        
        cross_ticker_results = []
        
        for test_ticker in tickers:
            train_tickers = [t for t in tickers if t != test_ticker]
            logger.info(f"\nTrain: {train_tickers} | Test: {test_ticker}")
            
            # Combine training data from 6 tickers
            train_data = self.df[self.df['ticker'].isin(train_tickers)].sort_values('date')
            test_data = self.df[self.df['ticker'] == test_ticker].sort_values('date')
            
            # Use chronological split: 80% train, 20% test
            train_size = int(0.8 * len(train_data))
            X_train = train_data.iloc[:train_size][feature_cols['all']].dropna(axis=1, how='all')
            y_train = train_data.iloc[:train_size]['binary_label'].values
            X_test = test_data[feature_cols['all']][X_train.columns]
            y_test = test_data['binary_label'].values
            
            # Train model (with per-test-ticker seed)
            trainer = CatBoostTrainer(verbose=0, ticker=test_ticker)
            trainer.fit(X_train, y_train, X_val=None, optimize_threshold=False)
            
            # Predict
            y_pred, y_pred_proba = trainer.predict(X_test, return_proba=True)
            
            # Metrics
            metrics = compute_metrics(y_test, y_pred, y_pred_proba)
            
            result = {
                'train_tickers': ','.join(train_tickers),
                'test_ticker': test_ticker,
                'train_size': len(train_data.iloc[:train_size]),
                'test_size': len(test_data),
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
            
            cross_ticker_results.append(result)
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        # Save cross-ticker results
        df_cross = pd.DataFrame(cross_ticker_results)
        output_path = self.output_dir / 'cross_ticker_results.csv'
        df_cross.to_csv(output_path, index=False)
        logger.info(f"\nSaved cross-ticker results: {output_path}")
        
        return cross_ticker_results
    
    def run_ablation_studies(self, tickers, feature_cols):
        """
        Run ablation studies (full vs technical-only vs rolling-returns-only).
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        feature_cols : dict
            Feature column sets
        
        Returns
        -------
        dict
            Ablation results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING ABLATION STUDIES")
        logger.info("="*80)
        
        ablation_results = []
        splitter = PerTickerSplitter()
        
        feature_sets = {
            'full': feature_cols['all'],
            'technical_only': feature_cols['technical'],
            'rolling_returns_only': feature_cols['rolling_returns']
        }
        
        for feature_set_name, features in feature_sets.items():
            logger.info(f"\n--- {feature_set_name.upper()} ({len(features)} features) ---")
            
            for ticker in tickers:
                train_df, val_df, test_df = splitter.split(self.df, ticker)
                
                X_train = train_df[features]
                y_train = train_df['binary_label'].values
                X_test = test_df[features]
                y_test = test_df['binary_label'].values
                
                # Train model (with per-ticker seed)
                trainer = CatBoostTrainer(verbose=0, ticker=ticker)
                trainer.fit(X_train, y_train, X_val=None, optimize_threshold=False)
                
                # Predict
                y_pred, y_pred_proba = trainer.predict(X_test, return_proba=True)
                
                # Metrics
                metrics = compute_metrics(y_test, y_pred, y_pred_proba)
                
                result = {
                    'feature_set': feature_set_name,
                    'n_features': len(features),
                    'ticker': ticker,
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                    'f1': metrics['f1']
                }
                
                ablation_results.append(result)
                logger.info(f"  {ticker}: accuracy={metrics['accuracy']:.4f}")
        
        # Save ablation results
        df_ablation = pd.DataFrame(ablation_results)
        output_path = self.output_dir / 'ablation_studies.csv'
        df_ablation.to_csv(output_path, index=False)
        logger.info(f"\nSaved ablation results: {output_path}")
        
        return ablation_results
    
    def run_backtest_simulation(self, tickers, feature_cols):
        """
        Simulate daily trading strategy based on model predictions.
        
        Parameters
        ----------
        tickers : list
            List of ticker symbols
        feature_cols : dict
            Feature column sets
        
        Returns
        -------
        dict
            Backtest results
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING BACKTEST SIMULATION")
        logger.info("="*80)
        
        backtest_results = []
        splitter = PerTickerSplitter()
        
        for ticker in tickers:
            logger.info(f"\n--- {ticker} ---")
            
            train_df, val_df, test_df = splitter.split(self.df, ticker)
            
            X_train = train_df[feature_cols['all']]
            y_train = train_df['binary_label'].values
            X_test = test_df[feature_cols['all']]
            y_test = test_df['binary_label'].values
            returns_test = test_df['next_day_return'].values
            
            # Train model (with per-ticker seed)
            trainer = CatBoostTrainer(verbose=0, ticker=ticker)
            trainer.fit(X_train, y_train, X_val=None, optimize_threshold=True)
            
            # Predict
            y_pred, y_pred_proba = trainer.predict(X_test, return_proba=True)
            
            # Backtest: long when prediction=1, cash (0) when prediction=0
            strategy_returns = np.where(y_pred == 1, returns_test, 0)
            cumulative_strategy_returns = np.cumprod(1 + strategy_returns) - 1
            cumulative_buy_hold_returns = np.cumprod(1 + returns_test) - 1
            
            # Metrics
            strategy_total_return = cumulative_strategy_returns[-1] if len(cumulative_strategy_returns) > 0 else np.nan
            buy_hold_total_return = cumulative_buy_hold_returns[-1] if len(cumulative_buy_hold_returns) > 0 else np.nan
            
            # Sharpe ratio
            strategy_sharpe = (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)) \
                if np.std(strategy_returns) > 0 else np.nan
            buy_hold_sharpe = (np.mean(returns_test) / np.std(returns_test) * np.sqrt(252)) \
                if np.std(returns_test) > 0 else np.nan
            
            # Max drawdown
            def max_drawdown(returns):
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                return np.min(drawdown)
            
            strategy_max_dd = max_drawdown(strategy_returns)
            buy_hold_max_dd = max_drawdown(returns_test)
            
            # Diebold-Mariano test
            dm_results = diebold_mariano_test(y_test, y_pred_proba, np.full_like(y_pred_proba, 0.5))
            
            result = {
                'ticker': ticker,
                'test_size': len(test_df),
                'strategy_total_return': strategy_total_return,
                'buy_hold_total_return': buy_hold_total_return,
                'excess_return': strategy_total_return - buy_hold_total_return,
                'strategy_sharpe': strategy_sharpe,
                'buy_hold_sharpe': buy_hold_sharpe,
                'strategy_max_drawdown': strategy_max_dd,
                'buy_hold_max_drawdown': buy_hold_max_dd,
                'model_accuracy': np.mean(y_pred == y_test),
                'dm_statistic': dm_results['dm_statistic'],
                'dm_p_value': dm_results['p_value']
            }
            
            backtest_results.append(result)
            logger.info(f"  Strategy Return: {strategy_total_return:.4f}, B&H Return: {buy_hold_total_return:.4f}")
            logger.info(f"  Excess Return: {strategy_total_return - buy_hold_total_return:.4f}")
        
        # Save backtest results
        df_backtest = pd.DataFrame(backtest_results)
        output_path = self.output_dir / 'backtest_results.csv'
        df_backtest.to_csv(output_path, index=False)
        logger.info(f"\nSaved backtest results: {output_path}")
        
        return backtest_results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("="*80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_rows': len(self.df),
                'tickers': sorted(self.df['ticker'].unique().tolist()),
                'date_range': {
                    'start': str(self.df['date'].min()),
                    'end': str(self.df['date'].max())
                },
                'target_distribution': {
                    'upward_movement': float(self.df['binary_label'].sum()),
                    'downward_movement': float((1 - self.df['binary_label']).sum()),
                    'upward_pct': float(self.df['binary_label'].mean())
                }
            }
        }
        
        # Save report
        output_path = self.output_dir / 'summary_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved summary report: {output_path}")
        
        return report
    
    def run_full_pipeline(self):
        """Run complete modeling pipeline."""
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL MODELING PIPELINE")
        logger.info("="*80)
        
        # Load data
        self.load_data()
        
        # Prepare features
        feature_cols = self.prepare_features()
        
        tickers = sorted(self.df['ticker'].unique())
        
        try:
            # Run all analyses
            self.run_baseline_models(tickers, feature_cols)
            self.run_per_ticker_catboost(tickers, feature_cols)
            self.run_cross_ticker_evaluation(tickers, feature_cols)
            self.run_ablation_studies(tickers, feature_cols)
            self.run_backtest_simulation(tickers, feature_cols)
            self.generate_summary_report()
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    pipeline = StockPredictionPipeline()
    pipeline.run_full_pipeline()
