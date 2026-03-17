"""
Multi-Model Sentiment Fusion System
Combines FinBERT, VADER, and TextBlob into a unified sentiment score

NOVEL CONTRIBUTION: Sentiment Disagreement Metrics
When models disagree, it indicates:
1. Ambiguous news (mixed signals)
2. Market uncertainty
3. Potential for volatility

This disagreement itself becomes a predictive feature!

RESEARCH HYPOTHESIS:
High disagreement → Higher volatility → Less predictable movement
Low disagreement → Strong signal → More predictable movement
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../..')
from config.config import *


class SentimentFusion:
    """
    Fuses multiple sentiment models into unified scores
    
    KEY FEATURES CREATED:
    1. ensemble_sentiment: Weighted average of all models
    2. sentiment_variance: How much models disagree
    3. sentiment_agreement: Pairwise correlation
    4. confidence_weighted: Sentiment weighted by agreement
    5. model_consensus: Binary flag (all models agree?)
    """
    
    def __init__(self, weights=None):
        """
        Initialize fusion system
        
        Parameters:
        -----------
        weights : dict, optional
            Weight for each model in ensemble
            Default: {'finbert': 0.6, 'vader': 0.25, 'textblob': 0.15}
            
        WHY THESE WEIGHTS?
        - FinBERT: 0.6 (highest weight, financial domain expert)
        - VADER: 0.25 (good at intensity, proven on social media)
        - TextBlob: 0.15 (baseline, less specialized)
        
        These weights are based on:
        1. Domain specificity (FinBERT is financial-specific)
        2. Empirical testing (FinBERT typically performs best)
        3. Ensemble diversity (don't want one model to dominate)
        """
        
        if weights is None:
            self.weights = {
                'finbert': 0.6,
                'vader': 0.25,
                'textblob': 0.15
            }
        else:
            self.weights = weights
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            print(f"⚠️ Warning: Weights sum to {total_weight}, normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print("\n" + "="*70)
        print("SENTIMENT FUSION SYSTEM INITIALIZED")
        print("="*70)
        print("Model weights:")
        for model, weight in self.weights.items():
            print(f"  {model.upper():12s}: {weight:.2%}")
        print("="*70)
    
    def load_sentiment_data(self, finbert_path=None, vader_path=None, textblob_path=None):
        """
        Load sentiment results from all three models
        
        Parameters:
        -----------
        finbert_path : str, optional
            Path to FinBERT results CSV
        vader_path : str, optional
            Path to VADER results CSV
        textblob_path : str, optional
            Path to TextBlob results CSV
            
        Returns:
        --------
        dict with DataFrames: {'finbert': df1, 'vader': df2, 'textblob': df3}
        """
        
        # Default paths
        if finbert_path is None:
            finbert_path = '../../data/processed/sentiment_finbert.csv'
        if vader_path is None:
            vader_path = '../../data/processed/sentiment_vader.csv'
        if textblob_path is None:
            textblob_path = '../../data/processed/sentiment_textblob.csv'
        
        print("\n" + "="*70)
        print("LOADING SENTIMENT DATA")
        print("="*70)
        
        data = {}
        
        # Load FinBERT
        try:
            df_finbert = pd.read_csv(finbert_path)
            data['finbert'] = df_finbert
            print(f"✅ FinBERT: {len(df_finbert)} rows")
        except FileNotFoundError:
            print(f"❌ FinBERT file not found: {finbert_path}")
        
        # Load VADER
        try:
            df_vader = pd.read_csv(vader_path)
            data['vader'] = df_vader
            print(f"✅ VADER: {len(df_vader)} rows")
        except FileNotFoundError:
            print(f"❌ VADER file not found: {vader_path}")
        
        # Load TextBlob
        try:
            df_textblob = pd.read_csv(textblob_path)
            data['textblob'] = df_textblob
            print(f"✅ TextBlob: {len(df_textblob)} rows")
        except FileNotFoundError:
            print(f"❌ TextBlob file not found: {textblob_path}")
        
        print("="*70)
        
        if len(data) == 0:
            raise ValueError("No sentiment data loaded! Run sentiment analyzers first.")
        
        return data
    
    def normalize_scores(self, df_finbert, df_vader, df_textblob):
        """
        Normalize all sentiment scores to [-1, +1] range
        
        WHY NORMALIZE?
        - FinBERT: sentiment_score ranges ~[-0.8, +0.8]
        - VADER: compound ranges [-1, +1] ← Already normalized
        - TextBlob: polarity ranges [-1, +1] ← Already normalized
        
        We need all scores on same scale to compare/combine them fairly
        
        Parameters:
        -----------
        df_finbert : DataFrame
            FinBERT results
        df_vader : DataFrame
            VADER results
        df_textblob : DataFrame
            TextBlob results
            
        Returns:
        --------
        DataFrame with normalized scores for all models
        
        NORMALIZATION METHOD:
        For each score:
        1. Find min and max in dataset
        2. Scale to [-1, +1]: normalized = 2 * (score - min) / (max - min) - 1
        """
        
        print("\n" + "="*70)
        print("NORMALIZING SENTIMENT SCORES")
        print("="*70)
        
        # Extract sentiment columns
        finbert_score = df_finbert['finbert_sentiment_score'].values
        vader_score = df_vader['vader_sentiment_score'].values
        textblob_score = df_textblob['textblob_sentiment_score'].values
        
        # Show original ranges
        print(f"Original ranges:")
        print(f"  FinBERT:  [{finbert_score.min():.3f}, {finbert_score.max():.3f}]")
        print(f"  VADER:    [{vader_score.min():.3f}, {vader_score.max():.3f}]")
        print(f"  TextBlob: [{textblob_score.min():.3f}, {textblob_score.max():.3f}]")
        
        # Normalize FinBERT (usually needs it most)
        if finbert_score.max() - finbert_score.min() > 0:
            finbert_normalized = 2 * (finbert_score - finbert_score.min()) / \
                                (finbert_score.max() - finbert_score.min()) - 1
        else:
            finbert_normalized = finbert_score
        
        # VADER and TextBlob are usually already in [-1, 1], but normalize anyway for consistency
        if vader_score.max() - vader_score.min() > 0:
            vader_normalized = 2 * (vader_score - vader_score.min()) / \
                              (vader_score.max() - vader_score.min()) - 1
        else:
            vader_normalized = vader_score
        
        if textblob_score.max() - textblob_score.min() > 0:
            textblob_normalized = 2 * (textblob_score - textblob_score.min()) / \
                                 (textblob_score.max() - textblob_score.min()) - 1
        else:
            textblob_normalized = textblob_score
        
        # Show normalized ranges
        print(f"\nNormalized ranges (all should be [-1, +1]):")
        print(f"  FinBERT:  [{finbert_normalized.min():.3f}, {finbert_normalized.max():.3f}]")
        print(f"  VADER:    [{vader_normalized.min():.3f}, {vader_normalized.max():.3f}]")
        print(f"  TextBlob: [{textblob_normalized.min():.3f}, {textblob_normalized.max():.3f}]")
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'finbert_normalized': finbert_normalized,
            'vader_normalized': vader_normalized,
            'textblob_normalized': textblob_normalized
        })
        
        print("✅ Normalization complete")
        print("="*70)
        
        return result_df
    
    def calculate_ensemble_sentiment(self, normalized_df):
        """
        Calculate weighted average sentiment (ensemble)
        
        Formula:
        ensemble = w1*finbert + w2*vader + w3*textblob
        
        Where w1 + w2 + w3 = 1.0
        
        Parameters:
        -----------
        normalized_df : DataFrame
            Normalized sentiment scores
            
        Returns:
        --------
        np.array of ensemble sentiment scores
        """
        
        ensemble = (
            self.weights['finbert'] * normalized_df['finbert_normalized'] +
            self.weights['vader'] * normalized_df['vader_normalized'] +
            self.weights['textblob'] * normalized_df['textblob_normalized']
        )
        
        return ensemble.values
    
    def calculate_disagreement_metrics(self, normalized_df):
        """
        Calculate NOVEL disagreement metrics
        
        THIS IS OUR RESEARCH CONTRIBUTION!
        
        Metrics:
        1. sentiment_variance: Statistical variance across 3 models
           High variance = models strongly disagree
           Low variance = models agree
        
        2. sentiment_range: Max score - Min score
           Simple measure of disagreement spread
        
        3. sentiment_agreement: Average pairwise correlation
           Measures how similarly models rank headlines
        
        4. model_consensus: Binary flag
           1 if all 3 models agree on direction (pos/neg/neu)
           0 if they disagree
        
        Parameters:
        -----------
        normalized_df : DataFrame
            Normalized sentiment scores
            
        Returns:
        --------
        DataFrame with disagreement metrics
        
        INTERPRETATION:
        Headline: "Apple beats earnings but faces regulatory scrutiny"
        
        FinBERT: +0.3  (slightly positive, understands finance)
        VADER:   +0.1  (weak positive, "beats" but "scrutiny")
        TextBlob: -0.2 (negative, focuses on "scrutiny")
        
        → High variance (0.065) → Models disagree → Ambiguous news
        → This disagreement is valuable information!
        """
        
        print("\n" + "="*70)
        print("CALCULATING DISAGREEMENT METRICS")
        print("="*70)
        
        scores = normalized_df[['finbert_normalized', 'vader_normalized', 'textblob_normalized']].values
        
        # 1. Variance across models (row-wise)
        sentiment_variance = np.var(scores, axis=1)
        
        # 2. Range (max - min per row)
        sentiment_range = np.max(scores, axis=1) - np.min(scores, axis=1)
        
        # 3. Pairwise agreement (correlation)
        # Calculate correlation between each pair of models
        corr_finbert_vader = np.corrcoef(
            normalized_df['finbert_normalized'],
            normalized_df['vader_normalized']
        )[0, 1]
        
        corr_finbert_textblob = np.corrcoef(
            normalized_df['finbert_normalized'],
            normalized_df['textblob_normalized']
        )[0, 1]
        
        corr_vader_textblob = np.corrcoef(
            normalized_df['vader_normalized'],
            normalized_df['textblob_normalized']
        )[0, 1]
        
        # Average pairwise correlation
        avg_correlation = (corr_finbert_vader + corr_finbert_textblob + corr_vader_textblob) / 3
        
        print(f"Pairwise correlations:")
        print(f"  FinBERT-VADER:    {corr_finbert_vader:.3f}")
        print(f"  FinBERT-TextBlob: {corr_finbert_textblob:.3f}")
        print(f"  VADER-TextBlob:   {corr_vader_textblob:.3f}")
        print(f"  Average:          {avg_correlation:.3f}")
        
        # 4. Model consensus (all agree on sign?)
        # Get sign of each score (-1, 0, or +1)
        signs = np.sign(scores)
        
        # Check if all 3 signs are the same (consensus)
        consensus = (signs[:, 0] == signs[:, 1]) & (signs[:, 1] == signs[:, 2])
        model_consensus = consensus.astype(int)
        
        # Create result DataFrame
        disagreement_df = pd.DataFrame({
            'sentiment_variance': sentiment_variance,
            'sentiment_range': sentiment_range,
            'sentiment_agreement': avg_correlation,  # Same for all rows
            'model_consensus': model_consensus
        })
        
        print(f"\nDisagreement statistics:")
        print(f"  Avg variance: {sentiment_variance.mean():.4f}")
        print(f"  Avg range: {sentiment_range.mean():.4f}")
        print(f"  Consensus rate: {model_consensus.mean():.2%} (models agree)")
        
        print("✅ Disagreement metrics calculated")
        print("="*70)
        
        return disagreement_df
    
    def calculate_confidence_weighted_sentiment(self, ensemble_sentiment, sentiment_variance):
        """
        Weight sentiment by confidence (inverse of disagreement)
        
        LOGIC:
        - Low variance → High confidence → Keep sentiment as-is
        - High variance → Low confidence → Reduce sentiment magnitude
        
        Formula:
        confidence = exp(-variance)  ← Exponential decay
        weighted_sentiment = sentiment * confidence
        
        EXAMPLE:
        Headline 1: Ensemble = +0.5, Variance = 0.01 (low)
        → Confidence = 0.99
        → Weighted = +0.5 * 0.99 = +0.495 (almost unchanged)
        
        Headline 2: Ensemble = +0.5, Variance = 0.20 (high)
        → Confidence = 0.82
        → Weighted = +0.5 * 0.82 = +0.41 (reduced by 18%)
        
        Parameters:
        -----------
        ensemble_sentiment : np.array
            Weighted average sentiment
        sentiment_variance : np.array
            Disagreement variance
            
        Returns:
        --------
        np.array of confidence-weighted sentiment
        """
        
        # Calculate confidence (inverse of variance)
        # Use exponential to map variance [0, ∞) to confidence [1, 0)
        confidence = np.exp(-sentiment_variance)
        
        # Weight sentiment by confidence
        weighted_sentiment = ensemble_sentiment * confidence
        
        return weighted_sentiment, confidence
    
    def fuse_all(self, data):
        """
        Main fusion function - combines all models
        
        Parameters:
        -----------
        data : dict
            Dictionary with 'finbert', 'vader', 'textblob' DataFrames
            
        Returns:
        --------
        DataFrame with:
        - Original data (date, ticker, headline, etc.)
        - All individual model scores
        - Normalized scores
        - Ensemble sentiment
        - Disagreement metrics
        - Confidence-weighted sentiment
        
        This becomes our final sentiment dataset for Phase 3!
        """
        
        print("\n" + "="*70)
        print("FUSING ALL SENTIMENT MODELS")
        print("="*70)
        
        # Get DataFrames
        df_finbert = data['finbert']
        df_vader = data['vader']
        df_textblob = data['textblob']
        
        # Normalize scores
        normalized_df = self.normalize_scores(df_finbert, df_vader, df_textblob)
        
        # Calculate ensemble
        ensemble_sentiment = self.calculate_ensemble_sentiment(normalized_df)
        
        # Calculate disagreement metrics
        disagreement_df = self.calculate_disagreement_metrics(normalized_df)
        
        # Calculate confidence-weighted sentiment
        confidence_weighted, confidence = self.calculate_confidence_weighted_sentiment(
            ensemble_sentiment,
            disagreement_df['sentiment_variance'].values
        )
        
        # Combine everything
        result_df = df_finbert.copy()  # Start with FinBERT as base
        
        # Add VADER columns
        result_df['vader_compound'] = df_vader['vader_compound']
        result_df['vader_sentiment_score'] = df_vader['vader_sentiment_score']
        
        # Add TextBlob columns
        result_df['textblob_polarity'] = df_textblob['textblob_polarity']
        result_df['textblob_subjectivity'] = df_textblob['textblob_subjectivity']
        result_df['textblob_sentiment_score'] = df_textblob['textblob_sentiment_score']
        
        # Add normalized scores
        result_df['finbert_normalized'] = normalized_df['finbert_normalized']
        result_df['vader_normalized'] = normalized_df['vader_normalized']
        result_df['textblob_normalized'] = normalized_df['textblob_normalized']
        
        # Add ensemble
        result_df['ensemble_sentiment'] = ensemble_sentiment
        
        # Add disagreement metrics
        result_df['sentiment_variance'] = disagreement_df['sentiment_variance']
        result_df['sentiment_range'] = disagreement_df['sentiment_range']
        result_df['sentiment_agreement'] = disagreement_df['sentiment_agreement']
        result_df['model_consensus'] = disagreement_df['model_consensus']
        
        # Add confidence-weighted sentiment
        result_df['confidence'] = confidence
        result_df['confidence_weighted_sentiment'] = confidence_weighted
        
        print(f"\n✅ Fusion complete!")
        print(f"Final dataset: {len(result_df)} rows × {len(result_df.columns)} columns")
        print("="*70)
        
        return result_df
    
    def save_fused_data(self, df, filename='../../data/processed/sentiment_fused.csv'):
        """
        Save fused sentiment data
        
        Parameters:
        -----------
        df : DataFrame
            Fused data
        filename : str
            Output file path
        """
        
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*70}")
        print("SAVE COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Saved to {filename}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"{'='*70}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Run sentiment fusion
    
    Usage:
        python sentiment_fusion.py
    """
    
    print("\n" + "="*70)
    print("MULTI-MODEL SENTIMENT FUSION")
    print("="*70)
    
    # Initialize fusion system
    fusion = SentimentFusion()
    
    # Load sentiment data
    try:
        data = fusion.load_sentiment_data()
        
        # Fuse all models
        fused_df = fusion.fuse_all(data)
        
        # Save results
        fusion.save_fused_data(fused_df)
        
        # Display sample
        print("\n" + "="*70)
        print("SAMPLE FUSED DATA (First 3 rows)")
        print("="*70)
        
        display_cols = [
            'headline',
            'finbert_sentiment_score',
            'vader_sentiment_score',
            'textblob_sentiment_score',
            'ensemble_sentiment',
            'sentiment_variance',
            'model_consensus'
        ]
        
        print(fused_df[display_cols].head(3).to_string())
        
        # Show statistics
        print("\n" + "="*70)
        print("FUSION STATISTICS")
        print("="*70)
        print(f"Ensemble sentiment: mean={fused_df['ensemble_sentiment'].mean():.4f}, "
              f"std={fused_df['ensemble_sentiment'].std():.4f}")
        print(f"Sentiment variance: mean={fused_df['sentiment_variance'].mean():.4f}, "
              f"std={fused_df['sentiment_variance'].std():.4f}")
        print(f"Model consensus: {fused_df['model_consensus'].mean():.2%} agreement")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to run sentiment analyzers first:")
        print("1. python nlp_analyzer.py")
        print("2. python baseline_sentiment.py")
        print("3. python sentiment_fusion.py")