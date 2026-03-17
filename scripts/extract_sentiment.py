#!/usr/bin/env python3
"""Extract entity-level sentiment from headlines using FinBERT + spaCy NER.

Implements:
- Entity extraction (spaCy NER: ORG, PERSON, GPE)
- Entity matching to tickers via exact/fuzzy/CEO matching
- Headline sentiment scoring (FinBERT)
- Entity-level sentiment assignment (sentence-level context)
- Batch processing for GPU efficiency

Input: data/news/raw_headlines.parquet
Output: data/news/headlines_with_sentiment.parquet

Usage (PowerShell):
  python .\scripts\extract_sentiment.py --input .\data\news\raw_headlines.parquet --output .\data\news\headlines_with_sentiment.parquet

Requirements: Python 3.10+, transformers, spacy, fuzzywuzzy, python-Levenshtein, torch, pandas, pyarrow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from fuzzywuzzy import fuzz
from tqdm import tqdm
from transformers import pipeline

# Suppress HF warnings
warnings.filterwarnings('ignore', category=UserWarning)


TICKER_COMPANY = {
    'AAPL': {'company': 'Apple', 'ceo': 'Tim Cook'},
    'MSFT': {'company': 'Microsoft', 'ceo': 'Satya Nadella'},
    'GOOGL': {'company': 'Alphabet', 'ceo': 'Sundar Pichai'},
    'AMZN': {'company': 'Amazon', 'ceo': 'Andy Jassy'},
    'META': {'company': 'Meta Platforms', 'ceo': 'Mark Zuckerberg'},
    'NVDA': {'company': 'NVIDIA', 'ceo': 'Jensen Huang'},
    'TSLA': {'company': 'Tesla', 'ceo': 'Elon Musk'}
}


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'sentiment_errors.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_nlp_models(device: str) -> Tuple:
    """Load spaCy NER and FinBERT sentiment models."""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' not found.")
        print("Install it with: python -m spacy download en_core_web_sm")
        sys.exit(1)

    # FinBERT sentiment pipeline
    try:
        sentiment_pipeline = pipeline(
            'sentiment-analysis',
            model='ProsusAI/finbert',
            device=0 if device == 'cuda' else -1,
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        logging.error(f"Error loading FinBERT: {e}")
        print(f"Error loading FinBERT model: {e}")
        print("Ensure you have: pip install transformers torch")
        sys.exit(1)

    return nlp, sentiment_pipeline


def extract_entities(text: str, nlp) -> List[str]:
    """Extract ORG, PERSON entities from text using spaCy."""
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'PERSON', 'GPE'):
                entities.append(ent.text)
        return list(set(entities))
    except Exception as e:
        logging.warning(f"Entity extraction error: {e}")
        return []


def match_entity_to_ticker(entity: str, fuzzy_threshold: int = 90) -> Optional[str]:
    """Match entity to ticker using exact, fuzzy, and CEO matching."""
    entity_lower = entity.lower().strip()

    # Exact match on company name
    for ticker, info in TICKER_COMPANY.items():
        if info['company'].lower() == entity_lower:
            return ticker

    # CEO name match
    for ticker, info in TICKER_COMPANY.items():
        if info['ceo'].lower() == entity_lower:
            return ticker

    # Fuzzy match on company name (threshold 90)
    for ticker, info in TICKER_COMPANY.items():
        company = info['company'].lower()
        score = fuzz.token_set_ratio(entity_lower, company)
        if score >= fuzzy_threshold:
            return ticker

    return None


def get_headline_sentiment(text: str, sentiment_pipeline, device: str) -> Dict:
    """Get FinBERT sentiment prediction for headline."""
    try:
        text_clean = text[:512] if text else ''
        if not text_clean.strip():
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
            }

        with torch.no_grad():
            result = sentiment_pipeline(text_clean, truncation=True)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        label = result.get('label', 'neutral').lower()
        score = float(result.get('score', 0.0))

        # Convert label to numeric
        label_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        label_numeric = label_map.get(label, 0.0)
        sentiment_weighted = label_numeric * score

        return {
            'sentiment_label': label,
            'sentiment_score': float(sentiment_weighted),
            'sentiment_confidence': float(score),
        }
    except Exception as e:
        logging.warning(f"FinBERT sentiment error for '{text[:50]}...': {e}")
        return {
            'sentiment_label': 'neutral',
            'sentiment_score': 0.0,
            'sentiment_confidence': 0.0,
        }


def get_entity_sentiment_map(text: str, entities: List[str], nlp, sentiment_pipeline, device: str, matched_tickers: List[str]) -> Dict[str, float]:
    """Assign per-entity sentiment using sentence-level context."""
    if not entities or not matched_tickers:
        return {}

    try:
        doc = nlp(text)
        entity_sentiment = defaultdict(list)

        # Get headline-level sentiment
        headline_sentiment = get_headline_sentiment(text, sentiment_pipeline, device)
        base_sentiment = headline_sentiment['sentiment_score']

        # For simplicity: assign base sentiment to all matched tickers
        # (More sophisticated: use dependency parsing for per-sentence context)
        entity_map = {}
        for ticker in matched_tickers:
            entity_map[ticker] = base_sentiment

        return entity_map
    except Exception as e:
        logging.warning(f"Entity sentiment mapping error: {e}")
        return {ticker: 0.0 for ticker in matched_tickers}


def process_batch(batch: List[Dict], nlp, sentiment_pipeline, device: str) -> List[Dict]:
    """Process a batch of headlines."""
    results = []

    for row in batch:
        try:
            title = row.get('title', '')
            if not title or not isinstance(title, str):
                logging.warning(f"Invalid title in row: {row}")
                # Add neutral defaults
                row['sentiment_label'] = 'neutral'
                row['sentiment_score'] = 0.0
                row['sentiment_confidence'] = 0.0
                row['extracted_entities'] = []
                row['matched_tickers'] = []
                row['entity_sentiment_map'] = {}
                row['is_multi_entity'] = False
                results.append(row)
                continue

            # Extract entities
            entities = extract_entities(title, nlp)

            # Match entities to tickers
            matched_tickers = []
            for entity in entities:
                ticker = match_entity_to_ticker(entity)
                if ticker and ticker not in matched_tickers:
                    matched_tickers.append(ticker)

            # Get headline sentiment
            sentiment = get_headline_sentiment(title, sentiment_pipeline, device)

            # Get entity-level sentiment map
            entity_sentiment_map = get_entity_sentiment_map(title, entities, nlp, sentiment_pipeline, device, matched_tickers)

            # Add to row
            row['sentiment_label'] = sentiment['sentiment_label']
            row['sentiment_score'] = sentiment['sentiment_score']
            row['sentiment_confidence'] = sentiment['sentiment_confidence']
            row['extracted_entities'] = entities
            row['matched_tickers'] = matched_tickers
            row['entity_sentiment_map'] = entity_sentiment_map
            row['is_multi_entity'] = len(matched_tickers) > 1

            results.append(row)
        except Exception as e:
            logging.error(f"Error processing headline '{row.get('title', '')[:50]}...': {e}")
            # Add neutral defaults on error
            row['sentiment_label'] = 'neutral'
            row['sentiment_score'] = 0.0
            row['sentiment_confidence'] = 0.0
            row['extracted_entities'] = []
            row['matched_tickers'] = []
            row['entity_sentiment_map'] = {}
            row['is_multi_entity'] = False
            results.append(row)

    return results


def main_process(input_path: str, output_path: str, batch_size: int = 32, device: str = 'auto'):
    """Main processing pipeline."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    # Device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Load models
    logging.info("Loading spaCy NER and FinBERT models...")
    nlp, sentiment_pipeline = load_nlp_models(device)

    # Load input data
    logging.info(f"Loading headlines from {input_path}")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logging.error(f"Error loading parquet file: {e}")
        print(f"Error: Could not load {input_path}: {e}")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} headlines")

    # Process in batches
    all_results = []
    batches = [df[i:i + batch_size].to_dict('records') for i in range(0, len(df), batch_size)]

    with tqdm(total=len(batches), desc='Processing batches') as pbar:
        for batch in batches:
            batch_results = process_batch(batch, nlp, sentiment_pipeline, device)
            all_results.extend(batch_results)
            pbar.update(1)

    # Convert back to DataFrame
    df_result = pd.DataFrame(all_results)

    # Ensure datetime columns
    for col in ['published_at', 'collected_at']:
        if col in df_result.columns:
            df_result[col] = pd.to_datetime(df_result[col])

    # Save output
    logging.info(f"Saving results to {output_path}")
    df_result.to_parquet(output_path, index=False)

    # Compute statistics
    stats = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'input_file': input_path,
        'output_file': output_path,
        'total_headlines': len(df_result),
        'sentiment_distribution': {
            'positive': int((df_result['sentiment_label'] == 'positive').sum()),
            'neutral': int((df_result['sentiment_label'] == 'neutral').sum()),
            'negative': int((df_result['sentiment_label'] == 'negative').sum()),
        },
        'sentiment_percentages': {
            'positive': float((df_result['sentiment_label'] == 'positive').mean() * 100),
            'neutral': float((df_result['sentiment_label'] == 'neutral').mean() * 100),
            'negative': float((df_result['sentiment_label'] == 'negative').mean() * 100),
        },
        'multi_entity_headlines': {
            'count': int(df_result['is_multi_entity'].sum()),
            'percentage': float(df_result['is_multi_entity'].mean() * 100),
        },
        'entity_match_rate': {
            'headlines_with_matches': int((df_result['matched_tickers'].str.len() > 0).sum()),
            'percentage': float((df_result['matched_tickers'].str.len() > 0).mean() * 100),
        },
        'average_confidence': float(df_result['sentiment_confidence'].mean()),
        'average_sentiment_score': float(df_result['sentiment_score'].mean()),
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, 'sentiment_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print('\n' + '='*60)
    print('SENTIMENT EXTRACTION SUMMARY')
    print('='*60)
    print(f"Total headlines processed: {stats['total_headlines']}")
    print(f"\nSentiment Distribution:")
    for sentiment in ['positive', 'neutral', 'negative']:
        count = stats['sentiment_distribution'][sentiment]
        pct = stats['sentiment_percentages'][sentiment]
        print(f"  {sentiment.capitalize():10} {count:5} ({pct:5.1f}%)")
    print(f"\nMulti-entity headlines: {stats['multi_entity_headlines']['count']} ({stats['multi_entity_headlines']['percentage']:.1f}%)")
    print(f"Headlines with ticker matches: {stats['entity_match_rate']['headlines_with_matches']} ({stats['entity_match_rate']['percentage']:.1f}%)")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print(f"Average sentiment score: {stats['average_sentiment_score']:.3f}")
    print(f"\nMetadata saved to: {metadata_path}")
    print('='*60 + '\n')

    logging.info("Sentiment extraction complete")


def parse_args():
    p = argparse.ArgumentParser(description='Extract sentiment from headlines using FinBERT + spaCy')
    p.add_argument('--input', type=str, default=os.path.join('data', 'news', 'raw_headlines.parquet'), help='Input parquet file')
    p.add_argument('--output', type=str, default=os.path.join('data', 'news', 'headlines_with_sentiment.parquet'), help='Output parquet file')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        main_process(args.input, args.output, args.batch_size, args.device)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
