#!/usr/bin/env python3
"""Download and filter FNSPID dataset for specific tickers."""
import json
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Target tickers
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

print("Loading FNSPID news data (streaming mode)...")
# Load FNSPID subset - use fnspid_news only (most relevant)
ds = load_dataset(
    "Brianferrell787/financial-news-multisource",
    data_files="data/fnspid_news/*.parquet",
    split="train",
    streaming=True  # Streaming avoids loading 15M rows into memory
)

# Filter and collect
filtered_rows = []
print(f"Filtering for tickers: {TICKERS}")
print("Processing (this takes ~10 minutes for 15M records)...")

for i, row in enumerate(tqdm(ds, desc="Scanning")):
    try:
        # Parse extra_fields JSON
        extras = json.loads(row['extra_fields'])
        stocks = extras.get('stocks', [])
        
        # Check if any target ticker is mentioned
        if any(ticker in stocks for ticker in TICKERS):
            # Filter for 2022-2023 dates
            date_str = row['date'][:10]  # Get YYYY-MM-DD
            year = int(date_str[:4])
            
            if 2022 <= year <= 2023:
                filtered_rows.append({
                    'date': date_str,
                    'text': row['text'],
                    'tickers': [t for t in stocks if t in TICKERS],
                    'url': extras.get('url', ''),
                    'publisher': extras.get('publisher', ''),
                })
        
        # Progress update every 100k rows
        if (i + 1) % 100000 == 0:
            print(f"Processed {i+1:,} rows, found {len(filtered_rows):,} matches")
            
    except Exception as e:
        continue

# Convert to DataFrame
print(f"\nTotal matches found: {len(filtered_rows):,}")
df = pd.DataFrame(filtered_rows)

# Expand multi-ticker rows (one row per ticker)
expanded_rows = []
for _, row in df.iterrows():
    for ticker in row['tickers']:
        expanded_rows.append({
            'ticker': ticker,
            'published_at': row['date'],
            'title': row['text'][:200],  # First 200 chars as title
            'url': row['url'],
            'source': row['publisher'],
        })

df_final = pd.DataFrame(expanded_rows)
print(f"\nExpanded to {len(df_final):,} ticker-headline pairs")
print(f"Date range: {df_final['published_at'].min()} to {df_final['published_at'].max()}")
print(f"\nPer-ticker counts:")
print(df_final['ticker'].value_counts())

# Save
output_path = 'data/news/fnspid_raw_headlines.parquet'
df_final.to_parquet(output_path, index=False)
print(f"\n✅ Saved to {output_path}")