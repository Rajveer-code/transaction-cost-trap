import yfinance as yf
import os

tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
start_date = '2024-01-01'
end_date = '2025-11-30'

os.makedirs('data/prices', exist_ok=True)

print("Downloading prices...\n")

for ticker in tickers:
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False          # ← THIS FIXES THE ERROR
        )

        if df.empty:
            print(f"{ticker}: empty dataframe (retrying...)")
            continue

        df.to_csv(f"data/prices/{ticker}_extended.csv")
        print(f"{ticker} → saved {len(df)} rows")

    except Exception as e:
        print(f"{ticker} ERROR:", e)

print("\nDone!")
