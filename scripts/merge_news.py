import pandas as pd
import os

tickers = ['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA']

for t in tickers:
    dfs = []

    for src in ["finnhub", "alphavantage"]:
        file = f"data/news/{t}_{src}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df["source"] = src
            dfs.append(df)

    if not dfs:
        continue

    merged = pd.concat(dfs, ignore_index=True)

    if "headline" in merged.columns:
        merged["headline_clean"] = merged["headline"].astype(str).str.lower().str.strip()
        merged = merged.drop_duplicates(subset="headline_clean")

    merged.to_csv(f"data/news/{t}_merged.csv", index=False)
    print(t, "→", len(merged), "unique articles")
