# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

def generate_sample_price_data(ticker: str, days: int = 480) -> None:
    """
    Generate sample OHLCV price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of data to generate
    """
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    days = len(dates)
    
    # Derive a reproducible per-ticker seed so each ticker gets different data
    # Use MD5 hash of the ticker to create a deterministic seed
    seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16) % (2**32 - 1)
    np.random.seed(seed)
    base_price = np.random.normal(150, 50)  # Random base price around 150
    
    df = pd.DataFrame({
        'date': dates,
        'open': np.cumprod(1 + np.random.normal(0.001, 0.02, days)) * base_price,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000000, 10000000, size=days)
    })
    
    # Set high and low based on open with some randomness
    df['high'] = df['open'] * (1 + np.abs(np.random.normal(0.01, 0.01, days)))
    df['low'] = df['open'] * (1 - np.abs(np.random.normal(0.01, 0.01, days)))
    df['close'] = (df['high'] + df['low']) / 2  # Simple close price
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Ensure volume is integer
    df['volume'] = df['volume'].astype(int)
    
    # Save to file
    output_dir = Path('data/prices')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{ticker}_extended.csv"
    df.to_csv(output_file, index=False)
    print(f"[OK] Generated {days} days of sample data for {ticker} at {output_file}")

def main():
    """Main function to generate sample data for multiple tickers"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    print("Generating sample price data...")
    for ticker in tickers:
        generate_sample_price_data(ticker)
    print("All sample data generated successfully!")

if __name__ == "__main__":
    main()