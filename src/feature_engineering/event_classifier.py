"""
Event Classification Module - Phase 3 (Day 15-16)
=================================================
Zero-shot classification of financial news headlines into 6 event categories.

Event Categories:
1. Earnings: Quarterly reports, revenue announcements
2. Product: Launches, updates, recalls
3. Analyst: Upgrades, downgrades, price targets
4. Regulatory: SEC filings, lawsuits, investigations
5. Macroeconomic: Fed rates, GDP, inflation
6. M&A: Mergers, acquisitions, partnerships
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EventClassifier:
    """
    Zero-shot classification of financial news headlines using facebook/bart-large-mnli.
    """
    
    # Define the 6 event categories with detailed labels for better classification
    EVENT_CATEGORIES = [
        "earnings report or quarterly financial results",
        "product launch or product announcement or product update",
        "analyst rating or stock upgrade or price target",
        "regulatory filing or lawsuit or investigation",
        "macroeconomic news or federal reserve or GDP or inflation",
        "merger or acquisition or business partnership"
    ]
    
    # Simplified labels for output
    EVENT_LABELS = [
        "earnings",
        "product",
        "analyst",
        "regulatory",
        "macroeconomic",
        "m&a"
    ]
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", batch_size: int = 8):
        """
        Initialize the zero-shot classifier.
        
        Args:
            model_name: Hugging Face model for zero-shot classification
            batch_size: Batch size for processing headlines
        """
        print(f"🔄 Loading zero-shot classifier: {model_name}")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1  # CPU; change to 0 for GPU
        )
        self.batch_size = batch_size
        print("✅ Event classifier loaded successfully!")
    
    def classify_single_headline(self, headline: str) -> Dict:
        """
        Classify a single headline into one of the 6 event categories.
        
        Args:
            headline: News headline text
            
        Returns:
            Dictionary with event_type, confidence, and all scores
        """
        try:
            result = self.classifier(
                headline,
                candidate_labels=self.EVENT_CATEGORIES,
                multi_label=False
            )

            # top predicted category (long form description)
            predicted_label = result["labels"][0]
            confidence = float(result["scores"][0])

            # map long label → short label
            top_label_idx = self.EVENT_CATEGORIES.index(predicted_label)
            event_type = self.EVENT_LABELS[top_label_idx]

            # map scores properly
            all_scores = {}
            for long_label, short_label in zip(self.EVENT_CATEGORIES, self.EVENT_LABELS):
                idx = result["labels"].index(long_label)
                all_scores[f"score_{short_label}"] = float(result["scores"][idx])

            return {
                "event_type": event_type,
                "confidence": confidence,
                **all_scores
            }

        except Exception as e:
            print(f"⚠️ Error classifying headline: {str(e)}")
            return {
                'event_type': 'unknown',
                'confidence': 0.0,
                **{f'score_{label}': 0.0 for label in self.EVENT_LABELS}
            }
    
    def classify_batch(self, headlines: List[str]) -> List[Dict]:
        """
        Classify a batch of headlines efficiently.
        
        Args:
            headlines: List of news headlines
            
        Returns:
            List of classification dictionaries
        """
        results = []
        total = len(headlines)
        
        print(f"📊 Classifying {total} headlines in batches of {self.batch_size}...")
        
        for i in range(0, total, self.batch_size):
            batch = headlines[i:i + self.batch_size]
            
            for headline in batch:
                result = self.classify_single_headline(headline)
                results.append(result)
            
            processed = min(i + self.batch_size, total)
            print(f"   Processed {processed}/{total} headlines ({processed/total*100:.1f}%)")
        
        print("✅ Event classification complete!")
        return results
    
    def classify_dataframe(self, df: pd.DataFrame, headline_col: str = 'headline') -> pd.DataFrame:
        """
        Classify all headlines in a DataFrame and add event features.
        
        Args:
            df: DataFrame containing headlines
            headline_col: Name of the headline column
            
        Returns:
            DataFrame with added event classification columns
        """
        print(f"\n{'='*60}")
        print("EVENT CLASSIFICATION - PHASE 3")
        print(f"{'='*60}\n")
        
        # Create a copy to avoid modifying original
        df_events = df.copy()
        
        # Classify all headlines
        headlines = df_events[headline_col].tolist()
        classifications = self.classify_batch(headlines)
        
        # Convert to DataFrame and merge
        events_df = pd.DataFrame(classifications)
        df_events = pd.concat([df_events, events_df], axis=1)
        
        # Add binary flags for each event type
        for event in self.EVENT_LABELS:
            df_events[f'is_{event}'] = (df_events['event_type'] == event).astype(int)
        
        # Summary statistics
        print(f"\n📈 Event Distribution:")
        print(df_events['event_type'].value_counts())
        df_events = df_events.loc[:, ~df_events.columns.duplicated()]

        # --- FIXED: use float() instead of np.float64().item() ---
        avg_confidence = float(df_events['confidence'].mean())
        median_confidence = float(df_events['confidence'].median())
        min_confidence = float(df_events['confidence'].min())
        max_confidence = float(df_events['confidence'].max())

        print(f"\n📊 Average Confidence: {avg_confidence:.3f}")
        print(f"  Median Confidence: {median_confidence:.3f}")
        print(f"  Min Confidence: {min_confidence:.3f}")
        print(f"  Max Confidence: {max_confidence:.3f}")
        
        return df_events
def main():
    """
    Main execution function for event classification.
    """
    import os
    
    # Paths
    INPUT_FILE = 'data/processed/sentiment_fused.csv'
    OUTPUT_FILE = 'data/processed/events_classified.csv'
    
    # Create output directory if needed
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 3 - DAY 15-16: EVENT CLASSIFICATION")
    print("="*60 + "\n")
    
    # Load sentiment data
    print(f"📂 Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} headlines")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Tickers: {df['ticker'].unique().tolist()}")
    
    # Initialize classifier
    classifier = EventClassifier(batch_size=8)
    
    # Classify events
    df_classified = classifier.classify_dataframe(df)
    
    # Save results
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df_classified.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved {len(df_classified)} rows with {len(df_classified.columns)} columns")
    
    print("\n" + "="*60)
    print("✅ EVENT CLASSIFICATION COMPLETE!")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Columns added: event_type, confidence, score_* (6), is_* (6)")
    print(f"Total columns: {len(df_classified.columns)}")

if __name__ == "__main__":
    main()