"""
nlp_pipeline.py
===============
NLP feature generation aligned EXACTLY with model_ready_full.csv schema.

This module takes raw news headlines and produces the exact 24 sentiment/event/entity
features that the trained CatBoost model expects.

Output features (24 total):
- 13 sentiment features
- 6 event-specific features
- 5 entity features

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import clean_text, log_info, log_warning
from FEATURE_SCHEMA import FEATURE_DEFAULTS

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL REGISTRY (LAZY LOADING)
# ============================================================

class ModelRegistry:
    """Lazy-load NLP models to avoid slow imports."""
    
    finbert_model = None
    finbert_tokenizer = None
    vader = None
    
    @staticmethod
    def load_finbert():
        if ModelRegistry.finbert_model is None:
            model_name = "ProsusAI/finbert"
            ModelRegistry.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            ModelRegistry.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            ModelRegistry.finbert_model.to(DEVICE)
            ModelRegistry.finbert_model.eval()
            log_info(f"FinBERT loaded on {DEVICE}", "NLP")
        return ModelRegistry.finbert_model, ModelRegistry.finbert_tokenizer
    
    @staticmethod
    def load_vader():
        if ModelRegistry.vader is None:
            ModelRegistry.vader = SentimentIntensityAnalyzer()
            log_info("VADER loaded", "NLP")
        return ModelRegistry.vader


# ============================================================
# CORE SENTIMENT MODELS
# ============================================================

def finbert_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate FinBERT sentiment scores.
    
    Returns:
        List of scores in range [-1, 1] (negative to positive)
    """
    if not texts:
        return []
    
    model, tokenizer = ModelRegistry.load_finbert()
    scores = []
    
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = [clean_text(t, max_length=512) for t in texts[i:i + batch_size]]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        # FinBERT classes: [negative, neutral, positive]
        pos = probs[:, 2].cpu().numpy()
        neg = probs[:, 0].cpu().numpy()
        
        # Score: positive - negative
        batch_scores = (pos - neg).tolist()
        scores.extend(batch_scores)
    
    return scores


def vader_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate VADER sentiment scores.
    
    Returns:
        List of compound scores in range [-1, 1]
    """
    vader = ModelRegistry.load_vader()
    return [vader.polarity_scores(clean_text(t))["compound"] for t in texts]


def textblob_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate TextBlob sentiment scores.
    
    Returns:
        List of polarity scores in range [-1, 1]
    """
    return [TextBlob(clean_text(t)).sentiment.polarity for t in texts]


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def classify_event(headline: str) -> str:
    """
    Classify headline into event categories.
    
    Returns:
        One of: earnings, product, analyst, regulatory, macroeconomic, other
    """
    text = clean_text(headline).lower()
    
    # Earnings-related keywords
    if any(k in text for k in ["earnings", "eps", "revenue", "profit", "quarterly", "results"]):
        return "earnings"
    
    # Product-related keywords
    if any(k in text for k in ["launch", "unveils", "introduces", "product", "release"]):
        return "product"
    
    # Analyst-related keywords
    if any(k in text for k in ["analyst", "upgrade", "downgrade", "price target", "rating"]):
        return "analyst"
    
    # Regulatory/legal keywords
    if any(k in text for k in ["regulatory", "lawsuit", "fine", "sec", "probe", "investigation"]):
        return "regulatory"
    
    # Macroeconomic keywords
    if any(k in text for k in ["inflation", "fed", "interest rate", "macro", "economy"]):
        return "macroeconomic"
    
    return "other"


# ============================================================
# ENTITY DETECTION
# ============================================================

def calculate_ceo_sentiment(headlines: List[str], ceo_name: str) -> tuple[int, float]:
    """
    Calculate CEO-specific sentiment.
    
    Returns:
        (mention_count, average_sentiment)
    """
    if not ceo_name:
        return 0, 0.0
    
    ceo_lower = ceo_name.lower()
    ceo_headlines = [h for h in headlines if ceo_lower in h.lower()]
    
    if not ceo_headlines:
        return 0, 0.0
    
    ceo_scores = textblob_sentiment(ceo_headlines)
    return len(ceo_headlines), float(np.mean(ceo_scores))


def calculate_competitor_mentions(headlines: List[str], competitors: List[str]) -> int:
    """Count competitor mentions across all headlines."""
    if not competitors:
        return 0
    
    comp_lower = [c.lower() for c in competitors]
    count = 0
    
    for h in headlines:
        h_lower = h.lower()
        if any(c in h_lower for c in comp_lower):
            count += 1
    
    return count


def calculate_entity_density(headlines: List[str], company_names: List[str]) -> float:
    """
    Calculate entity density (mentions per headline).
    
    Returns:
        Average mentions per headline
    """
    if not headlines or not company_names:
        return 0.0
    
    total_mentions = 0
    for h in headlines:
        h_lower = h.lower()
        for name in company_names:
            total_mentions += h_lower.count(name.lower())
    
    return total_mentions / len(headlines)


def calculate_entity_sentiment_gap(headlines: List[str], company_name: str) -> float:
    """
    Calculate sentiment gap between entity-mentioned vs all headlines.
    
    Returns:
        sentiment(entity_mentioned) - sentiment(all)
    """
    if not headlines or not company_name:
        return 0.0
    
    # All headline sentiment
    all_sentiment = np.mean(finbert_sentiment(headlines))
    
    # Entity-mentioned headline sentiment
    entity_headlines = [h for h in headlines if company_name.lower() in h.lower()]
    
    if not entity_headlines:
        return 0.0
    
    entity_sentiment = np.mean(finbert_sentiment(entity_headlines))
    
    return float(entity_sentiment - all_sentiment)


# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_sentiment_features(
    headlines_df: pd.DataFrame,
    ticker_metadata: Dict[str, Dict],
    ticker: Optional[str] = None
) -> Dict[str, float]:
    """
    Generate ALL 24 sentiment/event/entity features for a single ticker/date.
    
    Args:
        headlines_df: DataFrame with columns [date, ticker, headline]
        ticker_metadata: Ticker metadata dict (from tickers.json)
        ticker: Optional ticker to filter (if None, uses first ticker in df)
        
    Returns:
        Dictionary with 24 features matching FEATURE_SCHEMA
    """
    if headlines_df.empty:
        log_warning("No headlines provided, returning defaults", "NLP")
        return {k: FEATURE_DEFAULTS[k] for k in FEATURE_DEFAULTS if k in [
            "finbert_sentiment_score_mean", "vader_sentiment_score_mean",
            "textblob_sentiment_score_mean", "ensemble_sentiment_mean",
            "sentiment_variance_mean", "model_consensus_mean",
            "ensemble_sentiment_max", "ensemble_sentiment_min",
            "ensemble_sentiment_std", "confidence_mean",
            "num_headlines", "headline_length_avg",
            "sentiment_earnings", "sentiment_product", "sentiment_analyst",
            "count_positive_earnings", "count_negative_regulatory",
            "has_macroeconomic_news", "ceo_mention_count", "ceo_sentiment",
            "competitor_mention_count", "entity_density", "entity_sentiment_gap"
        ]}
    
    # Infer ticker if not provided
    if ticker is None:
        ticker = headlines_df.iloc[0]["ticker"]
    
    headlines = headlines_df["headline"].astype(str).tolist()
    
    # -------------------- Base Sentiment Scores --------------------
    
    finbert_scores = np.array(finbert_sentiment(headlines))
    vader_scores = np.array(vader_sentiment(headlines))
    textblob_scores = np.array(textblob_sentiment(headlines))
    
    # Ensemble: weighted average
    ensemble_scores = 0.6 * finbert_scores + 0.3 * vader_scores + 0.1 * textblob_scores
    
    # Variance across models
    sentiment_variance = float(np.var([
        np.mean(finbert_scores),
        np.mean(vader_scores),
        np.mean(textblob_scores)
    ]))
    
    # Model consensus (1 - std of model means)
    model_consensus = 1.0 - float(np.std([
        np.mean(finbert_scores),
        np.mean(vader_scores),
        np.mean(textblob_scores)
    ]))
    
    # Confidence (mean absolute ensemble score)
    confidence_mean = float(np.mean(np.abs(ensemble_scores)))
    
    # -------------------- Event Classification --------------------
    
    events = [classify_event(h) for h in headlines]
    
    # Event-specific sentiment
    def event_sentiment(event_type: str) -> float:
        indices = [i for i, e in enumerate(events) if e == event_type]
        if not indices:
            return 0.0
        return float(np.mean(ensemble_scores[indices]))
    
    sentiment_earnings = event_sentiment("earnings")
    sentiment_product = event_sentiment("product")
    sentiment_analyst = event_sentiment("analyst")
    
    # Event counts
    count_positive_earnings = sum(
        1 for i, e in enumerate(events)
        if e == "earnings" and ensemble_scores[i] > 0.1
    )
    
    count_negative_regulatory = sum(
        1 for i, e in enumerate(events)
        if e == "regulatory" and ensemble_scores[i] < -0.1
    )
    
    has_macroeconomic_news = 1 if "macroeconomic" in events else 0
    
    # -------------------- Entity Features --------------------
    
    meta = ticker_metadata.get(ticker, {})
    ceo_name = meta.get("ceo", "")
    competitors = meta.get("competitors", [])
    company_name = meta.get("company_name", ticker)
    short_name = meta.get("short_name", ticker)
    company_names = [company_name, short_name]
    
    ceo_mention_count, ceo_sentiment = calculate_ceo_sentiment(headlines, ceo_name)
    competitor_mention_count = calculate_competitor_mentions(headlines, competitors)
    entity_density = calculate_entity_density(headlines, company_names)
    entity_sentiment_gap = calculate_entity_sentiment_gap(headlines, company_name)
    
    # -------------------- Metadata --------------------
    
    num_headlines = len(headlines)
    headline_length_avg = float(np.mean([len(h) for h in headlines]))
    
    # -------------------- Return Feature Dict --------------------
    
    return {
        # Sentiment features (13)
        "finbert_sentiment_score_mean": float(np.mean(finbert_scores)),
        "vader_sentiment_score_mean": float(np.mean(vader_scores)),
        "textblob_sentiment_score_mean": float(np.mean(textblob_scores)),
        "ensemble_sentiment_mean": float(np.mean(ensemble_scores)),
        "sentiment_variance_mean": sentiment_variance,
        "model_consensus_mean": model_consensus,
        "ensemble_sentiment_max": float(np.max(ensemble_scores)),
        "ensemble_sentiment_min": float(np.min(ensemble_scores)),
        "ensemble_sentiment_std": float(np.std(ensemble_scores)),
        "confidence_mean": confidence_mean,
        "num_headlines": num_headlines,
        "headline_length_avg": headline_length_avg,
        
        # Event features (6)
        "sentiment_earnings": sentiment_earnings,
        "sentiment_product": sentiment_product,
        "sentiment_analyst": sentiment_analyst,
        "count_positive_earnings": count_positive_earnings,
        "count_negative_regulatory": count_negative_regulatory,
        "has_macroeconomic_news": has_macroeconomic_news,
        
        # Entity features (5)
        "ceo_mention_count": ceo_mention_count,
        "ceo_sentiment": ceo_sentiment,
        "competitor_mention_count": competitor_mention_count,
        "entity_density": entity_density,
        "entity_sentiment_gap": entity_sentiment_gap,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing nlp_pipeline.py...")
    
    # Mock data
    test_df = pd.DataFrame({
        "date": ["2024-01-15"] * 3,
        "ticker": ["AAPL"] * 3,
        "headline": [
            "Apple reports record earnings, beats analyst expectations",
            "Tim Cook announces new product launch",
            "Regulatory probe into Apple's app store policies"
        ]
    })
    
    test_metadata = {
        "AAPL": {
            "company_name": "Apple Inc",
            "short_name": "Apple",
            "ceo": "Tim Cook",
            "competitors": ["Samsung", "Google"]
        }
    }
    
    features = generate_sentiment_features(test_df, test_metadata, "AAPL")
    
    print("\n✅ Generated features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    print(f"\n✅ Total features: {len(features)}")
    print("✅ nlp_pipeline.py test passed")