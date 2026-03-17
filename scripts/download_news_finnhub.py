import finnhub
import pandas as pd
import time
from datetime import datetime
import os

API_KEY = "d4ktr2pr01qt7v177lqgd4ktr2pr01qt7v177lr0"
client = finnhub.Client(api_key=API_KEY)

tickers = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA']
os.makedirs("data/news", exist_ok=True)

def get_month_range(ym):
    start = ym.strftime("%Y-%m-%d")
    end = (ym + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    return start, end

for ticker in tickers:
    print("\nCollecting:", ticker)
    all_news = []

    for month in pd.date_range("2024-01-01", "2025-11-01", freq="MS"):
        start, end = get_month_range(month)
        
        try:
            news = client.company_news(ticker, _from=start, to=end)
            for n in news:
                n["datetime"] = datetime.fromtimestamp(n["datetime"])
                n["ticker"] = ticker
            all_news.extend(news)
            print(ticker, month.strftime("%Y-%m"), len(news))

        except Exception as e:
            print("Error:", e)

        time.sleep(1)

    pd.DataFrame(all_news).to_csv(f"data/news/{ticker}_finnhub.csv", index=False)
    print("Saved:", ticker)
