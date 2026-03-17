"""
Baseline Sentiment Analyzers - VADER & TextBlob
These are simpler, faster sentiment models that complement FinBERT

WHY USE THESE?
1. VADER: Excellent at understanding intensity ("very good" vs "good")
2. TextBlob: Fast, handles subjectivity (opinion vs fact)
3. Both are rule-based/lexicon-based (no neural networks)
4. 100x faster than transformers
5. Provide diversity in our ensemble

WHEN TO USE EACH:
- FinBERT: Best for financial domain accuracy
- VADER: Best for social media, intensity, emojis
- TextBlob: Best for general text, quick prototyping
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# VADER - Valence Aware Dictionary and sEntiment Reasoner
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("⚠️ VADER not installed. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# TextBlob - Simple text processing
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("⚠️ TextBlob not installed. Install with: pip install textblob")
    TEXTBLOB_AVAILABLE = False

import sys
sys.path.append('../..')
from config.config import *


class VADERAnalyzer:
    """
    VADER Sentiment Analyzer
    
    WHAT IS VADER?
    - Lexicon-based (dictionary of 7,500+ words with sentiment scores)
    - Rule-based (understands negation, capitalization, punctuation)
    - Created specifically for social media text
    
    VADER UNDERSTANDS:
    1. Intensity modifiers:
       "good" = 0.44
       "very good" = 0.60  ← Modifier increases score
       "VERY GOOD" = 0.68  ← Capitalization increases more
    
    2. Negation:
       "good" = 0.44
       "not good" = -0.34  ← Negation flips sentiment
    
    3. Punctuation:
       "good" = 0.44
       "good!" = 0.50  ← Exclamation increases intensity
       "good!!!" = 0.55
    
    4. Conjunctions:
       "The food was great but service was terrible"
       ← VADER correctly balances both sentiments
    
    OUTPUT:
    {
        'compound': -0.5 to +0.5 (overall sentiment)
        'pos': 0 to 1 (positive proportion)
        'neu': 0 to 1 (neutral proportion)
        'neg': 0 to 1 (negative proportion)
    }
    Note: pos + neu + neg = 1.0 (they are proportions, not probabilities)
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        
        if not VADER_AVAILABLE:
            raise ImportError("VADER not installed")
        
        print("\n" + "="*70)
        print("INITIALIZING VADER SENTIMENT ANALYZER")
        print("="*70)
        
        # Initialize VADER
        self.analyzer = SentimentIntensityAnalyzer()
        
        print("✅ VADER initialized")
        print("Lexicon size: ~7,500 words")
        print("Processing speed: ~10,000 texts/second")
        print("="*70)
    
    def analyze_single(self, text):
        """
        Analyze sentiment of a single text
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict with keys:
            - compound: overall sentiment (-1 to +1)
            - pos: positive proportion (0 to 1)
            - neu: neutral proportion (0 to 1)
            - neg: negative proportion (0 to 1)
            - sentiment_score: normalized compound score
            
        EXAMPLE:
        Input: "Apple stock soars on stellar earnings report!"
        Output: {
            'compound': 0.68,
            'pos': 0.52,
            'neu': 0.48,
            'neg': 0.0,
            'sentiment_score': 0.68
        }
        """
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)
        
        # VADER returns 4 scores: compound, pos, neu, neg
        # We add 'sentiment_score' as alias for 'compound' for consistency
        result = {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg'],
            'sentiment_score': scores['compound']  # Main score for comparison
        }
        
        return result
    
    def analyze_dataframe(self, df, text_column='headline', show_progress=True):
        """
        Analyze sentiment for all texts in a DataFrame
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        text_column : str
            Column name containing text
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        DataFrame with added VADER sentiment columns:
            - vader_compound
            - vader_pos
            - vader_neu
            - vader_neg
            - vader_sentiment_score
            
        SPEED:
        VADER is MUCH faster than FinBERT:
        - 1,000 headlines: ~0.1 seconds (vs ~15 seconds for FinBERT)
        - No GPU needed
        - No batching needed (already fast enough)
        """
        
        print(f"\n{'='*70}")
        print(f"ANALYZING SENTIMENT WITH VADER")
        print(f"{'='*70}")
        print(f"Total texts: {len(df)}")
        
        # Extract texts
        texts = df[text_column].astype(str).tolist()
        
        # Analyze each text
        results = []
        
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="Processing with VADER")
        
        for text in iterator:
            result = self.analyze_single(text)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Rename columns with 'vader_' prefix
        results_df = results_df.rename(columns={
            'compound': 'vader_compound',
            'pos': 'vader_pos',
            'neu': 'vader_neu',
            'neg': 'vader_neg',
            'sentiment_score': 'vader_sentiment_score'
        })
        
        # Combine with original DataFrame
        df_combined = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Show statistics
        print(f"\n✅ VADER analysis complete!")
        print(f"Average compound score: {results_df['vader_compound'].mean():.4f}")
        print(f"Score range: [{results_df['vader_compound'].min():.4f}, {results_df['vader_compound'].max():.4f}]")
        
        # Classify sentiment based on compound score
        # VADER convention: compound >= 0.05 is positive, <= -0.05 is negative
        positive = (results_df['vader_compound'] >= 0.05).sum()
        negative = (results_df['vader_compound'] <= -0.05).sum()
        neutral = len(results_df) - positive - negative
        
        print(f"\nSentiment distribution:")
        print(f"  Positive: {positive} ({positive/len(results_df)*100:.1f}%)")
        print(f"  Negative: {negative} ({negative/len(results_df)*100:.1f}%)")
        print(f"  Neutral: {neutral} ({neutral/len(results_df)*100:.1f}%)")
        
        return df_combined


class TextBlobAnalyzer:
    """
    TextBlob Sentiment Analyzer
    
    WHAT IS TEXTBLOB?
    - Built on top of NLTK (Natural Language Toolkit)
    - Uses pattern analyzer by default (lexicon + rules)
    - Returns polarity AND subjectivity
    
    TEXTBLOB OUTPUT:
    1. Polarity: -1 (negative) to +1 (positive)
       Example: "This movie is terrible" → -0.7
    
    2. Subjectivity: 0 (objective) to 1 (subjective)
       Objective: "The stock closed at $150" → 0.0
       Subjective: "I think this stock is amazing!" → 0.8
    
    WHY SUBJECTIVITY MATTERS:
    - Objective statements (facts) might be more reliable
    - Subjective statements (opinions) might be less predictive
    - We can use this as a feature!
    
    LIMITATIONS:
    - Not domain-specific (worse than FinBERT for finance)
    - Simple rule-based approach
    - But: VERY fast, easy to use, handles general language well
    """
    
    def __init__(self):
        """Initialize TextBlob analyzer"""
        
        if not TEXTBLOB_AVAILABLE:
            raise ImportError("TextBlob not installed")
        
        print("\n" + "="*70)
        print("INITIALIZING TEXTBLOB SENTIMENT ANALYZER")
        print("="*70)
        
        print("✅ TextBlob initialized")
        print("Method: Pattern analyzer (lexicon + rules)")
        print("Processing speed: ~5,000 texts/second")
        print("="*70)
    
    def analyze_single(self, text):
        """
        Analyze sentiment of a single text
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict with keys:
            - polarity: sentiment (-1 to +1)
            - subjectivity: objectivity measure (0 to 1)
            - sentiment_score: same as polarity (for consistency)
            
        EXAMPLE:
        Input: "Apple announces record earnings"
        Output: {
            'polarity': 0.35,        ← Positive
            'subjectivity': 0.25,    ← Mostly objective (factual)
            'sentiment_score': 0.35
        }
        
        Input: "I absolutely love this company's products!"
        Output: {
            'polarity': 0.65,        ← Very positive
            'subjectivity': 0.85,    ← Very subjective (opinion)
            'sentiment_score': 0.65
        }
        """
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentiment
        # blob.sentiment returns a named tuple: (polarity, subjectivity)
        sentiment = blob.sentiment
        
        result = {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'sentiment_score': sentiment.polarity  # For consistency with other models
        }
        
        return result
    
    def analyze_dataframe(self, df, text_column='headline', show_progress=True):
        """
        Analyze sentiment for all texts in a DataFrame
        
        Parameters:
        -----------
        df : pandas DataFrame
            Input data
        text_column : str
            Column name containing text
        show_progress : bool
            Show progress bar
            
        Returns:
        --------
        DataFrame with added TextBlob sentiment columns:
            - textblob_polarity
            - textblob_subjectivity
            - textblob_sentiment_score
        """
        
        print(f"\n{'='*70}")
        print(f"ANALYZING SENTIMENT WITH TEXTBLOB")
        print(f"{'='*70}")
        print(f"Total texts: {len(df)}")
        
        # Extract texts
        texts = df[text_column].astype(str).tolist()
        
        # Analyze each text
        results = []
        
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="Processing with TextBlob")
        
        for text in iterator:
            result = self.analyze_single(text)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Rename columns with 'textblob_' prefix
        results_df = results_df.rename(columns={
            'polarity': 'textblob_polarity',
            'subjectivity': 'textblob_subjectivity',
            'sentiment_score': 'textblob_sentiment_score'
        })
        
        # Combine with original DataFrame
        df_combined = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Show statistics
        print(f"\n✅ TextBlob analysis complete!")
        print(f"Average polarity: {results_df['textblob_polarity'].mean():.4f}")
        print(f"Average subjectivity: {results_df['textblob_subjectivity'].mean():.4f}")
        print(f"Polarity range: [{results_df['textblob_polarity'].min():.4f}, {results_df['textblob_polarity'].max():.4f}]")
        
        # Classify sentiment
        positive = (results_df['textblob_polarity'] > 0.1).sum()
        negative = (results_df['textblob_polarity'] < -0.1).sum()
        neutral = len(results_df) - positive - negative
        
        print(f"\nSentiment distribution:")
        print(f"  Positive: {positive} ({positive/len(results_df)*100:.1f}%)")
        print(f"  Negative: {negative} ({negative/len(results_df)*100:.1f}%)")
        print(f"  Neutral: {neutral} ({neutral/len(results_df)*100:.1f}%)")
        
        # Subjectivity analysis
        objective = (results_df['textblob_subjectivity'] < 0.5).sum()
        subjective = len(results_df) - objective
        
        print(f"\nSubjectivity distribution:")
        print(f"  Objective (<0.5): {objective} ({objective/len(results_df)*100:.1f}%)")
        print(f"  Subjective (>=0.5): {subjective} ({subjective/len(results_df)*100:.1f}%)")
        
        return df_combined


class MultiModelSentimentAnalyzer:
    """
    Combines all three sentiment models (FinBERT, VADER, TextBlob)
    
    This is the MAIN CLASS you'll use for Phase 2
    
    WORKFLOW:
    1. Load news data
    2. Run FinBERT (slow but accurate)
    3. Run VADER (fast, intensity-aware)
    4. Run TextBlob (fast, subjectivity-aware)
    5. Save all results to separate CSVs
    6. Next step: Fusion (Module 3)
    """
    
    def __init__(self):
        """Initialize all three sentiment analyzers"""
        
        print("\n" + "="*70)
        print("MULTI-MODEL SENTIMENT ANALYZER")
        print("="*70)
        
        self.models = {}
        
        # Initialize each model
        try:
            from nlp_analyzer import FinBERTAnalyzer
            self.models['finbert'] = FinBERTAnalyzer()
            print("✅ FinBERT loaded")
        except Exception as e:
            print(f"⚠️ FinBERT not available: {e}")
        
        try:
            if VADER_AVAILABLE:
                self.models['vader'] = VADERAnalyzer()
                print("✅ VADER loaded")
        except Exception as e:
            print(f"⚠️ VADER not available: {e}")
        
        try:
            if TEXTBLOB_AVAILABLE:
                self.models['textblob'] = TextBlobAnalyzer()
                print("✅ TextBlob loaded")
        except Exception as e:
            print(f"⚠️ TextBlob not available: {e}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        print("="*70)
    
    def analyze_all(self, df, text_column='headline'):
        """
        Run all sentiment models on the data
        
        Parameters:
        -----------
        df : DataFrame
            Input news data
        text_column : str
            Column containing text to analyze
            
        Returns:
        --------
        dict with DataFrames for each model:
            {
                'finbert': df_with_finbert_columns,
                'vader': df_with_vader_columns,
                'textblob': df_with_textblob_columns
            }
        """
        
        results = {}
        
        # Run each model
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"RUNNING {model_name.upper()}")
            print(f"{'='*70}")
            
            df_result = model.analyze_dataframe(df, text_column=text_column)
            results[model_name] = df_result
        
        return results
    
    def save_all(self, results):
        """
        Save all results to separate CSV files
        
        Parameters:
        -----------
        results : dict
            Dictionary of DataFrames from analyze_all()
        """
        
        import os
        os.makedirs('../../data/processed', exist_ok=True)
        
        for model_name, df in results.items():
            filename = f'../../data/processed/sentiment_{model_name}.csv'
            df.to_csv(filename, index=False)
            print(f"✅ Saved {model_name} results to {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Test VADER and TextBlob analyzers
    
    Usage:
        python baseline_sentiment.py
    """
    
    print("\n" + "="*70)
    print("BASELINE SENTIMENT ANALYZERS TEST")
    print("="*70)
    
    # Test headlines
    test_headlines = [
        "Apple stock soars to record high on stellar earnings!",
        "Tesla faces major setback, production targets missed",
        "Microsoft maintains steady performance in cloud services",
        "Amazon criticized for labor practices, stock drops",
        "Google unveils revolutionary AI technology"
    ]
    
    # Test VADER
    if VADER_AVAILABLE:
        print("\n" + "="*70)
        print("TESTING VADER")
        print("="*70)
        
        vader = VADERAnalyzer()
        
        for headline in test_headlines:
            result = vader.analyze_single(headline)
            print(f"\nHeadline: {headline}")
            print(f"Compound: {result['compound']:+.3f}")
            print(f"Pos: {result['pos']:.3f}, Neg: {result['neg']:.3f}, Neu: {result['neu']:.3f}")
    
    # Test TextBlob
    if TEXTBLOB_AVAILABLE:
        print("\n" + "="*70)
        print("TESTING TEXTBLOB")
        print("="*70)
        
        textblob = TextBlobAnalyzer()
        
        for headline in test_headlines:
            result = textblob.analyze_single(headline)
            print(f"\nHeadline: {headline}")
            print(f"Polarity: {result['polarity']:+.3f}")
            print(f"Subjectivity: {result['subjectivity']:.3f}")
    
    # Test on actual data if available
    try:
        news_df = pd.read_csv('../../data/raw/news_yahoo.csv')
        
        print("\n" + "="*70)
        print("PROCESSING ACTUAL NEWS DATA")
        print("="*70)
        
        # Run VADER
        if VADER_AVAILABLE:
            vader = VADERAnalyzer()
            news_vader = vader.analyze_dataframe(news_df, text_column='headline')
            news_vader.to_csv('../../data/processed/sentiment_vader.csv', index=False)
            print("\n✅ VADER results saved")
        
        # Run TextBlob
        if TEXTBLOB_AVAILABLE:
            textblob = TextBlobAnalyzer()
            news_textblob = textblob.analyze_dataframe(news_df, text_column='headline')
            news_textblob.to_csv('../../data/processed/sentiment_textblob.csv', index=False)
            print("✅ TextBlob results saved")
        
    except FileNotFoundError:
        print("\n⚠️ News data not found. Run scrapers first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")