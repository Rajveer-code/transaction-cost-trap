"""
Baseline Model Training - Phase 4 (Day 23-24)
==============================================
Train baseline models: Logistic Regression and Random Forest.
Establish performance benchmarks for comparison with advanced models.
"""

import pandas as pd
import numpy as np
import json
import pickle
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Import custom time-series CV
import sys
sys.path.append('src/evaluation')
from time_series_cv import TimeSeriesSplit


class BaselineTrainer:
    """
    Trainer for baseline models with time-series cross-validation.
    """
    
    def __init__(self, cv_config_path: str = 'models/cv_config.json'):
        """
        Initialize baseline trainer.
        
        Args:
            cv_config_path: Path to CV configuration file
        """
        self.models = {}
        self.results = {}
        self.scaler = None
        
        # Load CV configuration
        if Path(cv_config_path).exists():
            with open(cv_config_path, 'r') as f:
                cv_config = json.load(f)
            self.n_splits = cv_config['n_splits']
            self.test_size = cv_config['test_size']
        else:
            print("⚠️  CV config not found, using defaults")
            self.n_splits = 5
            self.test_size = 30
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target for modeling.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Exclude non-feature columns
        exclude_cols = ['date', 'ticker', 'movement']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['movement'].values
        
        print(f"📊 Data Preparation:")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Samples: {len(X)}")
        print(f"   Target distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def train_logistic_regression(self, X_train, y_train, class_weight='balanced'):
        """
        Train Logistic Regression baseline.
        
        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Class balancing strategy
            
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print("TRAINING: LOGISTIC REGRESSION")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"✅ Training complete in {training_time:.2f} seconds")
        print(f"   Coefficients: {len(model.coef_[0])}")
        print(f"   Intercept: {model.intercept_[0]:.4f}")
        
        return model, training_time
    
    def train_random_forest(self, X_train, y_train, class_weight='balanced'):
        """
        Train Random Forest baseline.
        
        Args:
            X_train: Training features
            y_train: Training target
            class_weight: Class balancing strategy
            
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print("TRAINING: RANDOM FOREST")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        print(f"✅ Training complete in {training_time:.2f} seconds")
        print(f"   Trees: {model.n_estimators}")
        print(f"   Max depth: {model.max_depth}")
        print(f"   Features used: {model.n_features_in_}")
        
        return model, training_time
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def cross_validate_model(self, model_fn, X, y, model_name: str) -> dict:
        """
        Perform time-series cross-validation.
        
        Args:
            model_fn: Function that returns a trained model
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            
        Returns:
            Dictionary with CV results
        """
        print(f"\n🔄 Cross-validating {model_name}...")
        print(f"   Folds: {self.n_splits}")
        print(f"   Test size per fold: {self.test_size}")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        cv_results = {
            'accuracy': [],
            'f1_score': [],
            'roc_auc': [],
            'training_time': []
        }
        
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            print(f"\n   Fold {fold}/{self.n_splits}:")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model, train_time = model_fn(X_train_scaled, y_train)
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test_scaled, y_test, model_name)
            
            cv_results['accuracy'].append(metrics['accuracy'])
            cv_results['f1_score'].append(metrics['f1_score'])
            cv_results['roc_auc'].append(metrics['roc_auc'])
            cv_results['training_time'].append(train_time)
            
            print(f"     Accuracy: {metrics['accuracy']:.4f}")
            print(f"     F1 Score: {metrics['f1_score']:.4f}")
            print(f"     ROC-AUC:  {metrics['roc_auc']:.4f}")
            
            fold += 1
        
        # Aggregate results
        aggregated = {
            'model': model_name,
            'accuracy_mean': np.mean(cv_results['accuracy']),
            'accuracy_std': np.std(cv_results['accuracy']),
            'f1_score_mean': np.mean(cv_results['f1_score']),
            'f1_score_std': np.std(cv_results['f1_score']),
            'roc_auc_mean': np.mean(cv_results['roc_auc']),
            'roc_auc_std': np.std(cv_results['roc_auc']),
            'training_time_mean': np.mean(cv_results['training_time']),
            'cv_scores': cv_results
        }
        
        print(f"\n📊 {model_name} CV Results:")
        print(f"   Accuracy: {aggregated['accuracy_mean']:.4f} (±{aggregated['accuracy_std']:.4f})")
        print(f"   F1 Score: {aggregated['f1_score_mean']:.4f} (±{aggregated['f1_score_std']:.4f})")
        print(f"   ROC-AUC:  {aggregated['roc_auc_mean']:.4f} (±{aggregated['roc_auc_std']:.4f})")
        print(f"   Avg Training Time: {aggregated['training_time_mean']:.2f}s")
        
        return aggregated
    
    def train_and_save_final_model(self, model_fn, X, y, model_name: str, save_path: str):
        """
        Train model on full dataset and save.
        
        Args:
            model_fn: Function that returns a trained model
            X: Full feature matrix
            y: Full target vector
            model_name: Name of the model
            save_path: Path to save the model
        """
        print(f"\n💾 Training final {model_name} on full dataset...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model, train_time = model_fn(X_scaled, y)
        
        # Save model
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"   ✅ Model saved to: {save_path}")
        
        return model


def main():
    """
    Main execution function for baseline training.
    """
    import os
    
    print("\n" + "="*60)
    print("PHASE 4 - DAY 23-24: BASELINE MODEL TRAINING")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    
    # Load data
    data_file = 'data/final/model_ready_full.csv'
    print(f"📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    print(f"   Loaded {len(df)} observations")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize trainer
    trainer = BaselineTrainer()
    
    # Prepare data
    X, y, feature_names = trainer.prepare_data(df)
    
    # Train and evaluate models
    results = []
    
    # 1. Logistic Regression
    lr_results = trainer.cross_validate_model(
        trainer.train_logistic_regression,
        X, y,
        'Logistic Regression'
    )
    results.append(lr_results)
    
    # Save final LR model
    lr_model = trainer.train_and_save_final_model(
        trainer.train_logistic_regression,
        X, y,
        'Logistic Regression',
        'models/baseline_lr.pkl'
    )
    
    # 2. Random Forest
    rf_results = trainer.cross_validate_model(
        trainer.train_random_forest,
        X, y,
        'Random Forest'
    )
    results.append(rf_results)
    
    # Save final RF model
    rf_model = trainer.train_and_save_final_model(
        trainer.train_random_forest,
        X, y,
        'Random Forest',
        'models/baseline_rf.pkl'
    )
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(trainer.scaler, f)
    print(f"\n💾 Scaler saved to: models/scaler.pkl")
    
    # Save results
    results_df = pd.DataFrame([{
        'model': r['model'],
        'accuracy': r['accuracy_mean'],
        'accuracy_std': r['accuracy_std'],
        'f1_score': r['f1_score_mean'],
        'f1_std': r['f1_score_std'],
        'roc_auc': r['roc_auc_mean'],
        'roc_auc_std': r['roc_auc_std'],
        'training_time': r['training_time_mean']
    } for r in results])
    
    results_df.to_csv('results/metrics/baseline_performance.csv', index=False)
    print(f"\n💾 Results saved to: results/metrics/baseline_performance.csv")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("BASELINE MODEL COMPARISON")
    print(f"{'='*60}\n")
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ BASELINE TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    print("  • models/baseline_lr.pkl")
    print("  • models/baseline_rf.pkl")
    print("  • models/scaler.pkl")
    print("\nNext: Run train_ensemble.py for advanced models")


if __name__ == "__main__":
    main()