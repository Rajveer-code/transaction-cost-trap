import requests
import pandas as pd
import os

API_KEY = "YOUR_API_KEY"

tickers = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA']
os.makedirs("data/news", exist_ok=True)

for ticker in tickers:
    print("\nGetting news for:", ticker)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": 1000,
        "apikey": API_KEY
    }

    r = requests.get(url, params=params).json()
    feed = r.get("feed", [])

    df = pd.DataFrame(feed)
    df["ticker"] = ticker
    df.to_csv(f"data/news/{ticker}_alphavantage.csv", index=False)

    print(f"{ticker} → {len(df)} articles")
