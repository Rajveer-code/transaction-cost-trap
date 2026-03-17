"""
FEATURE_SCHEMA.py
==================
Central feature schema definition for the financial sentiment NLP model.

This module defines:
1. MODEL_FEATURES: Exact list of 42 features in order (excluding target)
2. FEATURE_DEFAULTS: Default values for each feature
3. Validation functions: Ensure feature dictionaries match schema

CRITICAL: Feature order must match exactly between:
- Training data (model_ready_full.csv)
- Model artifacts (catboost_best.pkl, scaler_ensemble.pkl)
- Inference pipeline (feature_pipeline.py)

Author: Rajveer Singh Pall
"""

from typing import Dict, List, Tuple

# ============================================================
# FEATURE DEFINITIONS (42 features total)
# ============================================================

# Order matches model_ready_full.csv exactly (excluding date, ticker, movement)
MODEL_FEATURES = [
    # Sentiment features (23)
    "finbert_sentiment_score_mean",
    "vader_sentiment_score_mean",
    "textblob_sentiment_score_mean",
    "ensemble_sentiment_mean",
    "sentiment_variance_mean",
    "model_consensus_mean",
    "ensemble_sentiment_max",
    "ensemble_sentiment_min",
    "ensemble_sentiment_std",
    "confidence_mean",
    "num_headlines",
    "headline_length_avg",
    "sentiment_earnings",
    "sentiment_product",
    "sentiment_analyst",
    "count_positive_earnings",
    "count_negative_regulatory",
    "has_macroeconomic_news",
    "ceo_mention_count",
    "ceo_sentiment",
    "competitor_mention_count",
    "entity_density",
    "entity_sentiment_gap",
    
    # Technical indicators (15)
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "ATR",
    "OBV",
    "ADX",
    "Stochastic_K",
    "Stochastic_D",
    "VWAP",
    "CMF",
    "Williams_R",
    "EMA_12",
    
    # Lagged features (4)
    "ensemble_sentiment_mean_lag1",
    "daily_return_lag1",
    "Volume_lag1",
    "volatility_lag1",
]

# Total: 23 sentiment + 15 technical + 4 lagged = 42 features
# (movement is target, not a feature)
assert len(MODEL_FEATURES) == 42, f"Expected 42 features, got {len(MODEL_FEATURES)}"

# ============================================================
# DEFAULT VALUES
# ============================================================

FEATURE_DEFAULTS = {
    # Sentiment defaults (neutral/zero)
    "finbert_sentiment_score_mean": 0.0,
    "vader_sentiment_score_mean": 0.0,
    "textblob_sentiment_score_mean": 0.0,
    "ensemble_sentiment_mean": 0.0,
    "sentiment_variance_mean": 0.0,
    "model_consensus_mean": 0.0,
    "ensemble_sentiment_max": 0.0,
    "ensemble_sentiment_min": 0.0,
    "ensemble_sentiment_std": 0.0,
    "confidence_mean": 0.0,
    "num_headlines": 0.0,
    "headline_length_avg": 0.0,
    "sentiment_earnings": 0.0,
    "sentiment_product": 0.0,
    "sentiment_analyst": 0.0,
    "count_positive_earnings": 0.0,
    "count_negative_regulatory": 0.0,
    "has_macroeconomic_news": 0.0,
    "ceo_mention_count": 0.0,
    "ceo_sentiment": 0.0,
    "competitor_mention_count": 0.0,
    "entity_density": 0.0,
    "entity_sentiment_gap": 0.0,
    
    # Technical defaults (neutral values)
    "RSI": 50.0,  # Neutral RSI
    "MACD": 0.0,
    "MACD_signal": 0.0,
    "BB_upper": 0.0,
    "BB_middle": 0.0,
    "BB_lower": 0.0,
    "ATR": 0.0,
    "OBV": 0.0,
    "ADX": 20.0,  # Neutral ADX
    "Stochastic_K": 50.0,  # Neutral stochastic
    "Stochastic_D": 50.0,
    "VWAP": 0.0,
    "CMF": 0.0,
    "Williams_R": -50.0,  # Neutral Williams %R
    "EMA_12": 0.0,
    
    # Lagged defaults
    "ensemble_sentiment_mean_lag1": 0.0,  # ⚠️ Placeholder - should use historical predictions
    "daily_return_lag1": 0.0,
    "Volume_lag1": 0.0,
    "volatility_lag1": 0.15,  # Typical annualized volatility
}

# Verify all features have defaults
assert set(MODEL_FEATURES) == set(FEATURE_DEFAULTS.keys()), \
    "Feature mismatch between MODEL_FEATURES and FEATURE_DEFAULTS"

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def validate_feature_dict(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate that a feature dictionary contains all required features.
    
    Args:
        features: Dictionary of feature name -> value
        
    Returns:
        (is_valid, missing_features)
    """
    missing = [f for f in MODEL_FEATURES if f not in features]
    return len(missing) == 0, missing


def normalize_feature_dict(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize feature dictionary to ensure all MODEL_FEATURES are present.
    
    Missing features are filled with defaults.
    Extra features are removed.
    Features are reordered to match MODEL_FEATURES order.
    
    Args:
        features: Partial or complete feature dictionary
        
    Returns:
        Normalized dictionary with all MODEL_FEATURES in correct order
    """
    normalized = {}
    
    # Add all required features (use provided value or default)
    for feat in MODEL_FEATURES:
        normalized[feat] = features.get(feat, FEATURE_DEFAULTS[feat])
    
    return normalized


def get_feature_order() -> List[str]:
    """Return the canonical feature order."""
    return MODEL_FEATURES.copy()


def get_feature_count() -> int:
    """Return the number of features."""
    return len(MODEL_FEATURES)


# ============================================================
# FEATURE GROUPS (for analysis)
# ============================================================

# Correct slicing: 23 sentiment + 15 technical + 4 lagged = 42 total
SENTIMENT_FEATURES = MODEL_FEATURES[:23]  # Indices 0-22 (23 features)
TECHNICAL_FEATURES = MODEL_FEATURES[23:38]  # Indices 23-37 (15 features)
LAGGED_FEATURES = MODEL_FEATURES[38:]  # Indices 38-41 (4 features)

assert len(SENTIMENT_FEATURES) == 23, f"Expected 23 sentiment features, got {len(SENTIMENT_FEATURES)}"
assert len(TECHNICAL_FEATURES) == 15, f"Expected 15 technical features, got {len(TECHNICAL_FEATURES)}"
assert len(LAGGED_FEATURES) == 4, f"Expected 4 lagged features, got {len(LAGGED_FEATURES)}"

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing FEATURE_SCHEMA.py...")
    print(f"✅ Total features: {len(MODEL_FEATURES)}")
    print(f"✅ Sentiment features: {len(SENTIMENT_FEATURES)}")
    print(f"✅ Technical features: {len(TECHNICAL_FEATURES)}")
    print(f"✅ Lagged features: {len(LAGGED_FEATURES)}")
    
    # Test validation
    test_features = {f: 0.0 for f in MODEL_FEATURES}
    is_valid, missing = validate_feature_dict(test_features)
    assert is_valid, f"Validation failed: {missing}"
    print("✅ Validation works")
    
    # Test normalization
    partial = {"finbert_sentiment_score_mean": 0.5}
    normalized = normalize_feature_dict(partial)
    assert len(normalized) == len(MODEL_FEATURES), "Normalization failed"
    assert normalized["finbert_sentiment_score_mean"] == 0.5, "Value not preserved"
    print("✅ Normalization works")
    
    print("\n✅ FEATURE_SCHEMA.py test passed")

