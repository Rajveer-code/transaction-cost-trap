#!/usr/bin/env python3
"""Train LSTM for next-day stock direction using sentiment + technical features.

Implements:
- Data loading and time-series split (chronological)
- Feature selection, scaling and sequence preparation (lookback window)
- LSTM model training with early stopping and ReduceLROnPlateau
- Baseline models (random, logistic on sentiment-only, simple ensembles)
- Evaluation: classification metrics, financial metrics, plots, SHAP feature importance
- Save artifacts: model checkpoint, scaler, metrics JSON, plots

Usage (PowerShell):
  python .\scripts\train_lstm_model.py --input data/combined/features_for_modeling.parquet

Requirements: Python3.10+, torch>=2.0, scikit-learn>=1.3, matplotlib, seaborn, shap
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             classification_report, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import trange

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


@dataclass
class Config:
    input_path: str
    output_dir: str
    hidden_size: int = 64
    num_layers: int = 2
    epochs: int = 50
    batch_size: int = 8
    lr: float = 0.001
    lookback: int = 3
    device: str = "auto"
    patience: int = 10
    lr_patience: int = 5


class SentimentLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def select_features(df: pd.DataFrame) -> List[str]:
    """Return the 15 selected features according to spec."""
    features = [
        # Sentiment
        'sentiment_weighted_mean', 'vader_mean', 'textblob_mean',
        # Sentiment lags
        'sentiment_lag_1d', 'sentiment_lag_3d',
        # Sentiment interactions
        'sentiment_x_volume', 'sentiment_x_volatility', 'sentiment_momentum',
        # Technical
        'rsi_14d', 'macd', 'rolling_volatility_5d', 'volume_change',
        # Price
        'daily_return', 'open_to_close',
        # Headline metrics
        'headline_count', 'positive_ratio',
    ]
    # Keep only those present in df
    return [c for c in features if c in df.columns]


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Create sequences and labels across tickers using chronological order.

    Returns: (X_arr, y_arr, scaler, feature_cols_used, seq_dates, seq_tickers)
    where feature_cols_used is the list after removing zero-variance features.
    """
    # Copy and sort
    df2 = df.copy()
    df2 = df2.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Ensure feature cols exist
    for col in feature_cols:
        if col not in df2.columns:
            df2[col] = 0.0

    # Fit scaler if not provided
    if scaler is None:
        # Fill NaN values before variance check
        df2_filled = df2[feature_cols].fillna(0.0)
        
        # Check for zero variance (constant features) using nanvar for robustness
        feature_variances = np.nanvar(df2_filled.values, axis=0)
        zero_var_mask = feature_variances < 1e-10
        zero_var_features = [feature_cols[i] for i in range(len(feature_cols)) if zero_var_mask[i]]

        if zero_var_features:
            logging.warning(f"Removing {len(zero_var_features)} zero-variance features: {zero_var_features}")
            # Remove constant features from feature list
            feature_cols = [f for f in feature_cols if f not in zero_var_features]
            logging.info(f"Remaining features ({len(feature_cols)}): {feature_cols}")

        # Fit scaler on valid features only
        if len(feature_cols) > 0:
            scaler = StandardScaler()
            scaler.fit(df2[feature_cols].fillna(0.0))
        else:
            logging.error("No valid features remaining after removing zero-variance columns")
            raise ValueError("All features have zero variance")

    # Fill NaN and transform the (possibly reduced) set of features
    df2[feature_cols] = scaler.transform(df2[feature_cols].fillna(0.0))

    X_list = []
    y_list = []
    seq_dates = []
    seq_tickers = []

    # For each ticker, build sliding windows
    for ticker in df2['ticker'].unique():
        df_t = df2[df2['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        values = df_t[feature_cols].values
        targets = df_t['target_direction_1d'].values
        dates = pd.to_datetime(df_t['date']).dt.date.values

        # We need windows of length lookback where target is the next day
        for i in range(len(df_t) - lookback):
            X = values[i:i + lookback]
            y = targets[i + lookback]  # next day
            X_list.append(X)
            y_list.append(int(y))
            seq_dates.append(dates[i + lookback])
            seq_tickers.append(ticker)

    X_arr = np.stack(X_list) if len(X_list) > 0 else np.empty((0, lookback, len(feature_cols)))
    y_arr = np.array(y_list)

    return X_arr, y_arr, scaler, feature_cols, seq_dates, seq_tickers


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Config,
    output_dir: str,
) -> Tuple[nn.Module, dict]:
    device = cfg.device if cfg.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=cfg.lr_patience)

    best_val_loss = float('inf')
    best_epoch = -1
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                preds = model(Xv)
                loss = criterion(preds, yv)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}/{cfg.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
            # Save checkpoint
            ckpt_path = os.path.join(output_dir, 'lstm_sentiment_best.pth')
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                logging.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Load best model
    try:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'lstm_sentiment_best.pth')))
    except Exception:
        logging.warning("Could not load saved checkpoint; returning last model state")

    return model, history


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: Config) -> dict:
    device = cfg.device if cfg.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy().reshape(-1)

    y_true = y.astype(int)
    y_pred = (preds >= 0.5).astype(int)

    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, preds))
    except Exception:
        metrics['roc_auc'] = float('nan')

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Financial metrics: simulate simple daily returns when predicting direction
    # Assume a strategy that goes long if prediction==1, else 0 (no shorting)
    # Use daily_return column which must be provided separately by caller if needed
    return metrics, preds


def baseline_models_and_eval(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> dict:
    """Compute baseline models and simple logistic baselines."""
    results = {}
    # Random baseline
    y = df['target_direction_1d'].values
    rand_preds = np.random.binomial(1, 0.5, size=len(y))
    results['random_accuracy'] = float(accuracy_score(y, rand_preds))

    # FinBERT-only logistic (use sentiment_weighted_mean as proxy)
    if 'sentiment_weighted_mean' in df.columns:
        X = df[['sentiment_weighted_mean']].fillna(0).values.reshape(-1, 1)
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X, y)
        preds = lr.predict(X)
        results['finbert_logistic_acc'] = float(accuracy_score(y, preds))
    else:
        results['finbert_logistic_acc'] = None

    # Simple ensemble: average of three simple logistic regressions trained on three sentiment features
    sent_feats = [c for c in ['sentiment_weighted_mean', 'vader_mean', 'textblob_mean'] if c in df.columns]
    if len(sent_feats) >= 1:
        preds_stack = []
        for feat in sent_feats:
            Xf = df[[feat]].fillna(0).values.reshape(-1, 1)
            lr = LogisticRegression(solver='liblinear')
            lr.fit(Xf, y)
            preds_stack.append(lr.predict_proba(Xf)[:, 1])

        preds_stack = np.vstack(preds_stack)
        # Equal ensemble
        avg_preds = preds_stack.mean(axis=0)
        results['ensemble_equal_acc'] = float(accuracy_score(y, (avg_preds >= 0.5).astype(int)))
        # Weighted 50/30/20
        weights = np.array([0.5, 0.3, 0.2])[:preds_stack.shape[0]]
        weighted = np.average(preds_stack, axis=0, weights=weights)
        results['ensemble_weighted_acc'] = float(accuracy_score(y, (weighted >= 0.5).astype(int)))
    else:
        results['ensemble_equal_acc'] = None
        results['ensemble_weighted_acc'] = None

    return results


def plot_confusion_matrix(cm, out_path: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc(y_true, y_scores, out_path: str):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label='ROC')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        logging.warning(f"ROC plot failed: {e}")


def save_json(d: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, default=float)


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for next-day direction')
    parser.add_argument('--input', type=str, default=os.path.join('data', 'combined', 'features_for_modeling.parquet'))
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--lookback', type=int, default=3)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    cfg = Config(
        input_path=args.input,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        lookback=args.lookback,
        device=args.device,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    results_dir = os.path.join(cfg.output_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    logs_path = os.path.join(results_dir, 'training_errors.log')

    # Setup logging to file as well
    fh = logging.FileHandler(logs_path)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    try:
        df = pd.read_parquet(cfg.input_path)
    except Exception as e:
        logging.error(f"Could not read input features: {e}")
        sys.exit(1)

    # Check sample size
    if len(df) < 30:
        logging.warning("Dataset has <30 samples — high overfitting risk")

    # Ensure date and target exist
    if 'date' not in df.columns or 'target_direction_1d' not in df.columns:
        logging.error("Input must contain 'date' and 'target_direction_1d' columns")
        sys.exit(1)

    # Chronological split by unique date
    df['date'] = pd.to_datetime(df['date'])
    unique_dates = sorted(df['date'].dt.date.unique())
    logging.info(f"Found {len(unique_dates)} unique dates")

    # According to spec: Days 1-5 train, Days 6-7 val, Day 8 test holdout (not trained)
    train_dates = unique_dates[:5]
    val_dates = unique_dates[5:7]
    test_dates = unique_dates[7:8] if len(unique_dates) > 7 else []

    logging.info(f"Train dates: {train_dates}")
    logging.info(f"Val dates: {val_dates}")
    logging.info(f"Test dates (holdout): {test_dates}")

    feature_cols = select_features(df)
    logging.info(f"Selected features ({len(feature_cols)}): {feature_cols}")

    # Build train / val subsets (rows whose target day falls into the date buckets)
    # We will create sequences later; here we select rows by date for splitting
    df_train = df[df['date'].dt.date.isin(train_dates)].copy()
    df_val = df[df['date'].dt.date.isin(val_dates)].copy()
    df_test = df[df['date'].dt.date.isin(test_dates)].copy()

    # Prepare sequences using full df but let prepare_sequences handle scaler fitting
    X_all, y_all, scaler_out, feature_cols_used, seq_dates, seq_tickers = prepare_sequences(df, feature_cols, cfg.lookback, scaler=None)

    # Log which features were actually used (some may have been removed)
    logging.info(f"Features after zero-variance removal ({len(feature_cols_used)}): {feature_cols_used}")
    feature_cols = feature_cols_used  # Update feature_cols to match what was actually used

    # Save scaler returned from prepare_sequences
    scaler_path = os.path.join(cfg.output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_out, f)

    # Build index arrays to select samples where target date in train/val/test
    seq_dates = np.array(seq_dates)
    train_mask = np.isin(seq_dates, np.array(train_dates))
    val_mask = np.isin(seq_dates, np.array(val_dates))
    test_mask = np.isin(seq_dates, np.array(test_dates))

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    logging.info(f"Sequences: total={len(X_all)} train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Baselines on the dataset aggregated by target rows (we use df filtered by target dates)
    baseline_stats = baseline_models_and_eval(df[df['date'].dt.date.isin(train_dates + val_dates)], feature_cols, cfg)

    # Torch datasets and loaders
    train_dataset = TorchDataset(X_train, y_train)
    val_dataset = TorchDataset(X_val, y_val)
    test_dataset = TorchDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Build model
    input_size = X_train.shape[2] if len(X_train) > 0 else (len(feature_cols))
    model = SentimentLSTM(input_size=input_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)

    # Train
    model, history = train_loop(model, train_loader, val_loader, cfg, cfg.output_dir)

    # Evaluate on validation and test
    metrics = {}
    if len(X_val) > 0:
        val_metrics, val_scores = evaluate_model(model, X_val, y_val, cfg)
        metrics['val'] = val_metrics
    else:
        metrics['val'] = {}

    if len(X_test) > 0:
        test_metrics, test_scores = evaluate_model(model, X_test, y_test, cfg)
        metrics['test'] = test_metrics
    else:
        metrics['test'] = {}

    metrics['baseline'] = baseline_stats

    # Save metrics and history
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.json')
    save_json(metrics, metrics_path)

    # Save model (best was saved during training; also save final state)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'lstm_sentiment_final.pth'))

    # Confusion matrix and ROC for validation if available
    if len(X_val) > 0:
        _, val_scores = evaluate_model(model, X_val, y_val, cfg)
        val_preds = (val_scores >= 0.5).astype(int)
        cm = confusion_matrix(y_val, val_preds)
        plot_confusion_matrix(cm, os.path.join(results_dir, 'confusion_matrix.png'))
        plot_roc(y_val, val_scores, os.path.join(results_dir, 'roc_curve.png'))

    # Try SHAP feature importance (may be slow); fallback gracefully
    try:
        import shap
        if len(X_val) > 0:
            # Use a small background sample
            bg = X_train.reshape(X_train.shape[0], -1)
            bg = bg[np.random.choice(len(bg), min(len(bg), 30), replace=False)]
            explainer = shap.KernelExplainer(lambda x: model(torch.tensor(x.reshape(-1, cfg.lookback, input_size), dtype=torch.float32)).detach().numpy().reshape(-1), bg)
            shap_vals = explainer.shap_values(X_val.reshape(X_val.shape[0], -1), nsamples=50)
            # Aggregate importance across timesteps by mean abs
            shap_arr = np.mean(np.abs(np.array(shap_vals)), axis=0)
            feat_imp = shap_arr.reshape(cfg.lookback, input_size).mean(axis=0)
            feat_names = feature_cols
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=feat_imp, y=feat_names, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, 'feature_importance.png'))
            plt.close(fig)
    except Exception as e:
        logging.warning(f"SHAP feature importance failed: {e}")

    # Optionally, cumulative returns plot — needs daily_return aligned with seq dates
    try:
        if 'daily_return' in df.columns and len(X_val) > 0:
            # For each prediction on val, compute return if predicted long
            _, val_scores = evaluate_model(model, X_val, y_val, cfg)
            val_preds = (val_scores >= 0.5).astype(int)
            # Map seq_dates of val_mask
            val_dates_arr = np.array(seq_dates)[val_mask]
            returns = []
            for i, p in enumerate(val_preds):
                date_i = val_dates_arr[i]
                ticker_i = np.array(seq_tickers)[val_mask][i]
                # find the row in df with ticker/date
                r = df[(df['ticker'] == ticker_i) & (df['date'].dt.date == date_i)]
                if not r.empty:
                    returns.append(float(r['daily_return'].iloc[0]) if p == 1 else 0.0)
            cum_returns = np.cumsum(returns)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(cum_returns)
            ax.set_title('Cumulative Returns (Val)')
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, 'cumulative_returns.png'))
            plt.close(fig)
    except Exception as e:
        logging.warning(f"Cumulative returns plot failed: {e}")

    logging.info("Training and evaluation complete. Artifacts saved.")


if __name__ == '__main__':
    main()
