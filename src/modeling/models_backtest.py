"""
models_backtest.py (FIXED)
===========================
Model inference engine aligned with unified feature pipeline.

Changes from original:
- Removed fetch_technical_features() (now in feature_pipeline.py)
- Removed prepare_model_features() (now in feature_pipeline.py)
- Uses assemble_model_features() for complete feature assembly
- Simplified predict() to accept pre-assembled features

Author: Rajveer Singh Pall
"""

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import log_info, log_error, log_warning
from src.feature_engineering.feature_pipeline import assemble_model_features, create_model_input_dataframe
from FEATURE_SCHEMA import MODEL_FEATURES

# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_best.pkl"
SCALER_PATH = MODELS_DIR / "scaler_ensemble.pkl"

DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PredictionResult:
    """Container for model prediction results."""
    ticker: str
    date: datetime
    prediction: int  # 0 = DOWN, 1 = UP
    probability: float  # Probability of UP class
    confidence: float  # max(prob_up, prob_down)
    signal: str  # "BUY", "SELL", "HOLD"
    features: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "signal": self.signal,
        }


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    """Handles loading and caching of trained models."""

    def __init__(
        self,
        model_path: Path = CATBOOST_MODEL_PATH,
        scaler_path: Path = SCALER_PATH
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def load_model(self) -> Any:
        """Load CatBoost model from pickle."""
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            log_info(f"Loaded CatBoost model from {self.model_path.name}", "MODEL")

        return self.model

    def load_scaler(self) -> Optional[Any]:
        """Load feature scaler from pickle."""
        if self.scaler is None:
            if not self.scaler_path.exists():
                log_warning(
                    f"Scaler not found at {self.scaler_path}, using raw features",
                    "MODEL"
                )
                return None

            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            log_info(f"Loaded scaler from {self.scaler_path.name}", "MODEL")

        return self.scaler


# ============================================================
# PREDICTION ENGINE
# ============================================================

class PredictionEngine:
    """Handles model inference and signal generation."""

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.loader = ModelLoader()
        self.model = self.loader.load_model()
        self.scaler = self.loader.load_scaler()
        self.confidence_threshold = confidence_threshold

    def _get_probabilities(self, X: pd.DataFrame) -> tuple[float, float]:
        """Return (prob_down, prob_up) in a robust way."""
        prob = self.model.predict_proba(X)[0]

        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            if 1 in classes and 0 in classes:
                idx_up = classes.index(1)
                idx_down = classes.index(0)
            else:
                idx_down, idx_up = 0, 1
        else:
            idx_down, idx_up = 0, 1

        prob_down = float(prob[idx_down])
        prob_up = float(prob[idx_up])
        return prob_down, prob_up

    def predict(
        self,
        ticker: str,
        sentiment_features: Dict[str, float],
        date: Optional[datetime] = None,
    ) -> PredictionResult:
        """
        Generate prediction for a single ticker.
        
        Args:
            ticker: Stock ticker
            sentiment_features: 24 features from nlp_pipeline
            date: Prediction date (defaults to now)
            
        Returns:
            PredictionResult object
        """
        if date is None:
            date = datetime.now()

        log_info(f"Predicting for {ticker} on {date.strftime('%Y-%m-%d')}", "PREDICT")

        # Assemble complete features (sentiment + technical + lagged)
        complete_features = assemble_model_features(ticker, sentiment_features)

        # Create model input DataFrame
        X = create_model_input_dataframe(
            complete_features,
            ticker,
            date.strftime("%Y-%m-%d")
        )

        # Drop metadata columns for prediction
        X_pred = X[MODEL_FEATURES]

        # Apply scaling if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_pred)
            X_pred = pd.DataFrame(X_scaled, columns=X_pred.columns)

        # Predict
        prediction = int(self.model.predict(X_pred)[0])
        prob_down, prob_up = self._get_probabilities(X_pred)
        confidence = max(prob_up, prob_down)

        # Generate signal
        if confidence >= self.confidence_threshold:
            signal = "BUY" if prediction == 1 else "SELL"
        else:
            signal = "HOLD"

        log_info(
            f"✅ {ticker}: {signal} (prob={prob_up:.3f}, conf={confidence:.3f})",
            "PREDICT"
        )

        return PredictionResult(
            ticker=ticker,
            date=date,
            prediction=prediction,
            probability=prob_up,
            confidence=confidence,
            signal=signal,
            features=complete_features,
        )

    def predict_batch(
        self,
        ticker_sentiment_dict: Dict[str, Dict[str, float]],
        date: Optional[datetime] = None
    ) -> List[PredictionResult]:
        """
        Generate predictions for multiple tickers.
        
        Args:
            ticker_sentiment_dict: Dictionary mapping ticker -> sentiment_features
            date: Prediction date (defaults to now)
            
        Returns:
            List of PredictionResult objects
        """
        results: List[PredictionResult] = []
        
        for ticker, sentiment_features in ticker_sentiment_dict.items():
            try:
                result = self.predict(ticker, sentiment_features, date)
                results.append(result)
            except Exception as e:
                log_error(f"Prediction failed for {ticker}: {e}", "PREDICT")
                continue
        
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def quick_predict(
    ticker: str,
    sentiment_features: Dict[str, float],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> PredictionResult:
    """
    Quick single-ticker prediction.

    Args:
        ticker: Stock ticker
        sentiment_features: From nlp_pipeline (24 features)
        confidence_threshold: Minimum confidence for BUY/SELL

    Returns:
        PredictionResult
    """
    engine = PredictionEngine(confidence_threshold=confidence_threshold)
    return engine.predict(ticker, sentiment_features)


# ============================================================
# BACKTESTING (STUB - Preserved from Original)
# ============================================================

@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    def to_dict(self) -> Dict:
        return {
            "Total Return (%)": f"{self.total_return:.2f}",
            "Annualized Return (%)": f"{self.annualized_return:.2f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Max Drawdown (%)": f"{self.max_drawdown:.2f}",
            "Win Rate (%)": f"{self.win_rate:.2f}",
            "Total Trades": self.total_trades,
        }


class BacktestEngine:
    """
    Portfolio backtesting engine (STUB).
    
    Original implementation preserved but simplified.
    Full backtest requires historical predictions database.
    """
    
    def __init__(self, tickers: List[str], start_date: datetime, end_date: datetime):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        log_warning("Backtest engine is a stub - full implementation requires historical data", "BACKTEST")
    
    def run_ml_strategy(self, predictions: pd.DataFrame) -> tuple[pd.DataFrame, BacktestMetrics]:
        """Stub for ML strategy backtest."""
        equity_curve = pd.DataFrame({
            "date": pd.date_range(self.start_date, self.end_date, freq="D"),
            "equity": [100000] * len(pd.date_range(self.start_date, self.end_date, freq="D"))
        })
        
        metrics = BacktestMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0
        )
        
        return equity_curve, metrics
    
    def run_buy_and_hold(self) -> tuple[pd.DataFrame, BacktestMetrics]:
        """Stub for buy-and-hold backtest."""
        return self.run_ml_strategy(pd.DataFrame())


def compare_strategies(
    tickers: List[str],
    predictions: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compare ML strategy vs Buy & Hold (STUB).
    
    Full implementation requires historical predictions database.
    """
    engine = BacktestEngine(tickers, start_date, end_date)
    
    ml_equity, ml_metrics = engine.run_ml_strategy(predictions)
    bh_equity, bh_metrics = engine.run_buy_and_hold()
    
    return {
        "ml_equity": ml_equity,
        "ml_metrics": ml_metrics,
        "bh_equity": bh_equity,
        "bh_metrics": bh_metrics,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing models_backtest.py...")
    print("=" * 60)
    
    # Test 1: Model Loading
    try:
        loader = ModelLoader()
        model = loader.load_model()
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Model loading failed: {e}\n")
    
    # Test 2: Single Prediction
    try:
        mock_sentiment = {
            "finbert_sentiment_score_mean": 0.35,
            "vader_sentiment_score_mean": 0.28,
            "textblob_sentiment_score_mean": 0.32,
            "ensemble_sentiment_mean": 0.317,
            "sentiment_variance_mean": 0.001,
            "model_consensus_mean": 0.95,
            "ensemble_sentiment_max": 0.45,
            "ensemble_sentiment_min": 0.20,
            "ensemble_sentiment_std": 0.08,
            "confidence_mean": 0.65,
            "num_headlines": 12,
            "headline_length_avg": 85.5,
            "sentiment_earnings": 0.40,
            "sentiment_product": 0.30,
            "sentiment_analyst": 0.25,
            "count_positive_earnings": 3,
            "count_negative_regulatory": 0,
            "has_macroeconomic_news": 0,
            "ceo_mention_count": 2,
            "ceo_sentiment": 0.38,
            "competitor_mention_count": 1,
            "entity_density": 1.5,
            "entity_sentiment_gap": 0.05,
        }
        
        result = quick_predict("AAPL", mock_sentiment)
        
        print("✅ Prediction generated:")
        print(f"   Signal: {result.signal}")
        print(f"   Probability (UP): {result.probability:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print()
    except Exception as e:
        print(f"❌ Prediction failed: {e}\n")
    
    print("=" * 60)
    print("✅ models_backtest.py test complete")