"""
Named Entity Recognition Module - Phase 3 (Day 16-17)
======================================================
Hybrid approach for extracting financial entities: CEOs, Products, Competitors.

Strategy:
- CEOs: Dictionary + SpaCy PERSON verification
- Products: Keyword-based detection with predefined lists
- Competitors: Dictionary + SpaCy ORG verification
"""

import pandas as pd
import numpy as np
import spacy
from typing import Dict, List, Set, Tuple
import re
import warnings
warnings.filterwarnings('ignore')


class FinancialEntityExtractor:
    """
    Hybrid NER system for financial entities using dictionaries + SpaCy.
    """
    
    # CEO Dictionary (Ticker -> List of CEO names/variants)
    CEO_DICT = {
        'AAPL': ['Tim Cook', 'Cook', 'Timothy Cook'],
        'MSFT': ['Satya Nadella', 'Nadella', 'Satya'],
        'GOOGL': ['Sundar Pichai', 'Pichai', 'Sundar'],
        'AMZN': ['Andy Jassy', 'Jassy', 'Andrew Jassy'],
        'TSLA': ['Elon Musk', 'Musk', 'Elon'],
        'NVDA': ['Jensen Huang', 'Huang', 'Jensen'],
        'META': ['Mark Zuckerberg', 'Zuckerberg', 'Mark']
    }
    
    # Product Dictionary (Ticker -> List of major products)
    PRODUCT_DICT = {
        'AAPL': ['iPhone', 'iPad', 'Mac', 'MacBook', 'AirPods', 'Apple Watch', 
                 'iOS', 'MacOS', 'Apple TV', 'Vision Pro', 'App Store'],
        'MSFT': ['Windows', 'Office', 'Azure', 'Xbox', 'Surface', 'Teams', 
                 'LinkedIn', 'GitHub', 'Bing', 'Copilot'],
        'GOOGL': ['Search', 'Android', 'Chrome', 'YouTube', 'Gmail', 'Maps',
                  'Cloud', 'Pixel', 'Gemini', 'Bard', 'Workspace'],
        'AMZN': ['AWS', 'Prime', 'Alexa', 'Echo', 'Kindle', 'Fire TV',
                 'Amazon Web Services', 'Prime Video'],
        'TSLA': ['Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck',
                 'Powerwall', 'Solar', 'Autopilot', 'FSD'],
        'NVDA': ['GPU', 'GeForce', 'RTX', 'Tensor Core', 'CUDA', 'DGX',
                 'Jetson', 'DRIVE', 'Omniverse'],
        'META': ['Facebook', 'Instagram', 'WhatsApp', 'Messenger', 'Threads',
                 'Oculus', 'Quest', 'Ray-Ban Stories', 'Llama']
    }
    
    # Competitor Dictionary (Ticker -> List of main competitors)
    COMPETITOR_DICT = {
        'AAPL': ['Samsung', 'Google', 'Microsoft', 'Huawei', 'Xiaomi'],
        'MSFT': ['Apple', 'Google', 'Amazon', 'Oracle', 'IBM'],
        'GOOGL': ['Microsoft', 'Apple', 'Meta', 'Amazon', 'TikTok'],
        'AMZN': ['Microsoft', 'Google', 'Alibaba', 'Walmart', 'eBay'],
        'TSLA': ['Ford', 'GM', 'Toyota', 'Volkswagen', 'BYD', 'Rivian'],
        'NVDA': ['AMD', 'Intel', 'Qualcomm', 'ARM', 'Broadcom'],
        'META': ['TikTok', 'Snapchat', 'Twitter', 'X', 'YouTube', 'LinkedIn']
    }
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """
        Initialize the entity extractor with SpaCy model.
        
        Args:
            spacy_model: SpaCy model name (must be downloaded)
        """
        print(f"🔄 Loading SpaCy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
            print("✅ SpaCy model loaded successfully!")
        except OSError:
            print(f"⚠️ Model '{spacy_model}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
            print("✅ SpaCy model downloaded and loaded!")
    
    def extract_ceo_mentions(self, headline: str, ticker: str) -> Dict:
        """
        Extract CEO mentions using dictionary + SpaCy verification.
        
        Args:
            headline: News headline text
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with CEO mention flags and details
        """
        mentions_ceo = 0
        ceo_names_found = []
        
        # Primary: Dictionary lookup
        if ticker in self.CEO_DICT:
            for ceo_name in self.CEO_DICT[ticker]:
                if ceo_name.lower() in headline.lower():
                    mentions_ceo = 1
                    ceo_names_found.append(ceo_name)
        
        # Secondary: SpaCy PERSON verification
        doc = self.nlp(headline)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if detected person matches any CEO
                for ticker_key, ceo_list in self.CEO_DICT.items():
                    if any(ceo.lower() in ent.text.lower() for ceo in ceo_list):
                        mentions_ceo = 1
                        if ent.text not in ceo_names_found:
                            ceo_names_found.append(ent.text)
        
        return {
            'mentions_ceo': mentions_ceo,
            'ceo_names': ', '.join(ceo_names_found) if ceo_names_found else ''
        }
    
    def extract_product_mentions(self, headline: str, ticker: str) -> Dict:
        """
        Extract product mentions using keyword-based detection.
        
        Args:
            headline: News headline text
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with product mention flags and details
        """
        mentions_product = 0
        products_found = []
        
        if ticker in self.PRODUCT_DICT:
            for product in self.PRODUCT_DICT[ticker]:
                # Case-insensitive match with word boundaries
                pattern = r'\b' + re.escape(product) + r'\b'
                if re.search(pattern, headline, re.IGNORECASE):
                    mentions_product = 1
                    products_found.append(product)
        
        return {
            'mentions_product': mentions_product,
            'products': ', '.join(products_found) if products_found else '',
            'product_count': len(products_found)
        }
    
    def extract_competitor_mentions(self, headline: str, ticker: str) -> Dict:
        """
        Extract competitor mentions using dictionary + SpaCy ORG verification.
        
        Args:
            headline: News headline text
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with competitor mention flags and details
        """
        mentions_competitor = 0
        competitors_found = []
        
        # Primary: Dictionary lookup
        if ticker in self.COMPETITOR_DICT:
            for competitor in self.COMPETITOR_DICT[ticker]:
                if competitor.lower() in headline.lower():
                    mentions_competitor = 1
                    competitors_found.append(competitor)
        
        # Secondary: SpaCy ORG verification
        doc = self.nlp(headline)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Check if it's a known competitor
                for ticker_key, comp_list in self.COMPETITOR_DICT.items():
                    if any(comp.lower() in ent.text.lower() for comp in comp_list):
                        if ent.text not in competitors_found:
                            competitors_found.append(ent.text)
                            mentions_competitor = 1
        
        return {
            'mentions_competitor': mentions_competitor,
            'competitors': ', '.join(competitors_found) if competitors_found else '',
            'competitor_count': len(competitors_found)
        }
    
    def extract_numeric_features(self, headline: str) -> Dict:
        """
        Extract numeric features (numbers, percentages).
        
        Args:
            headline: News headline text
            
        Returns:
            Dictionary with numeric feature flags
        """
        has_numbers = 1 if re.search(r'\d+', headline) else 0
        has_percentage = 1 if re.search(r'\d+%', headline) else 0
        
        return {
            'has_numbers': has_numbers,
            'has_percentage': has_percentage
        }
    
    def extract_all_entities(self, headline: str, ticker: str) -> Dict:
        """
        Extract all entity types from a headline.
        
        Args:
            headline: News headline text
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with all entity features
        """
        # Extract all entity types
        ceo_features = self.extract_ceo_mentions(headline, ticker)
        product_features = self.extract_product_mentions(headline, ticker)
        competitor_features = self.extract_competitor_mentions(headline, ticker)
        numeric_features = self.extract_numeric_features(headline)
        
        # Combine all features
        all_features = {
            **ceo_features,
            **product_features,
            **competitor_features,
            **numeric_features
        }
        
        # Calculate entity density (total entities per headline)
        entity_count = (
            ceo_features['mentions_ceo'] +
            product_features['product_count'] +
            competitor_features['competitor_count']
        )
        all_features['entity_count'] = entity_count
        
        return all_features
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame and extract entities from all headlines.
        
        Args:
            df: DataFrame with 'headline' and 'ticker' columns
            
        Returns:
            DataFrame with added entity features
        """
        print(f"\n{'='*60}")
        print("NAMED ENTITY RECOGNITION - PHASE 3")
        print(f"{'='*60}\n")
        
        print(f"📊 Processing {len(df)} headlines...")
        
        # Extract entities for all rows
        entity_features = []
        for idx, row in df.iterrows():
            features = self.extract_all_entities(row['headline'], row['ticker'])
            entity_features.append(features)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} headlines ({(idx+1)/len(df)*100:.1f}%)")
        
        print(f"   Processed {len(df)}/{len(df)} headlines (100.0%)")
        
        # Convert to DataFrame and merge
        entities_df = pd.DataFrame(entity_features)
        df_with_entities = pd.concat([df, entities_df], axis=1)
        
        # Summary statistics
        print(f"\n📈 Entity Extraction Summary:")
        print(f"   CEO mentions: {entities_df['mentions_ceo'].sum()} ({entities_df['mentions_ceo'].mean()*100:.1f}%)")
        print(f"   Product mentions: {entities_df['mentions_product'].sum()} ({entities_df['mentions_product'].mean()*100:.1f}%)")
        print(f"   Competitor mentions: {entities_df['mentions_competitor'].sum()} ({entities_df['mentions_competitor'].mean()*100:.1f}%)")
        print(f"   Headlines with numbers: {entities_df['has_numbers'].sum()} ({entities_df['has_numbers'].mean()*100:.1f}%)")
        print(f"   Headlines with percentages: {entities_df['has_percentage'].sum()} ({entities_df['has_percentage'].mean()*100:.1f}%)")
        print(f"   Average entity count per headline: {entities_df['entity_count'].mean():.2f}")
        
        return df_with_entities


def main():
    """
    Main execution function for entity extraction.
    """
    import os
    
    # Paths
    INPUT_FILE = 'data/processed/events_classified.csv'
    OUTPUT_FILE = 'data/processed/entities_extracted.csv'
    
    # Create output directory if needed
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 3 - DAY 16-17: NAMED ENTITY RECOGNITION")
    print("="*60 + "\n")
    
    # Load event-classified data
    print(f"📂 Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df)} headlines")
    print(f"   Columns: {len(df.columns)}")
    
    # Initialize extractor
    extractor = FinancialEntityExtractor(spacy_model="en_core_web_lg")
    
    # Extract entities
    df_with_entities = extractor.process_dataframe(df)
    
    # Save results
    print(f"\n💾 Saving results to: {OUTPUT_FILE}")
    df_with_entities.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved {len(df_with_entities)} rows with {len(df_with_entities.columns)} columns")
    
    print("\n" + "="*60)
    print("✅ NAMED ENTITY RECOGNITION COMPLETE!")
    print("="*60)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"New entity features: mentions_ceo, mentions_product, mentions_competitor,")
    print(f"                     has_numbers, has_percentage, entity_count")
    print(f"Total columns: {len(df_with_entities.columns)}")


if __name__ == "__main__":
    main()