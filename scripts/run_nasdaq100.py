"""
run_nasdaq100.py
================
Downloads NASDAQ-100 price data one ticker at a time,
cleans it, computes daily returns, and saves a parquet file.

This version avoids MultiIndex problems from yfinance multi-ticker downloads.

Run:
    python scripts/run_nasdaq100.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yfinance as yf


START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
OUTPUT_PATH = Path("data/nasdaq100_prices.parquet")
AVAILABLE_TICKERS_PATH = Path("data/nasdaq100_available_tickers.csv")

NASDAQ100: List[str] = [
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA","NVDA","AVGO","COST",
    "ADBE","CSCO","TMUS","CMCSA","TXN","QCOM","INTC","HON","INTU","AMGN",
    "AMAT","MU","ISRG","BKNG","LRCX","ADI","REGN","VRTX","KLAC","MDLZ",
    "PANW","ADP","SBUX","GILD","SNPS","CDNS","MELI","ASML","ORLY","CTAS",
    "NFLX","PEP","MNST","FTNT","FAST","ODFL","ROST","CPRT","BIIB","IDXX",
    "PAYX","VRSK","EXC","XEL","SGEN","MCHP","LULU","DLTR","MSCI","ANSS",
    "ALGN","DXCM","ZS","CRWD","TEAM","DOCU","OKTA","ZM","WDAY","SPLK",
    "MTCH","SIRI","WBA","CTSH","MAR","PCAR","ILMN","FISV","NXPI","SWKS",
    "ZBRA","TTWO","CDW","CHKP","NTAP","VRSN","BMRN","HOLX","CERN","EXPE",
    "FOXA","FOX","DISH","LBTYA","QRTEA","VIAV","MXIM","XLNX","ALXN","WLTW"
]


def download_one_ticker(ticker: str):
    import yfinance as yf
    import pandas as pd

    try:
        df = yf.download(
            ticker,
            start="2015-01-01",
            end="2025-01-01",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"❌ {ticker} error: {e}")
        return None

    # 🚫 Empty dataframe
    if df is None or df.empty:
        print(f"❌ {ticker}: empty")
        return None

    df = df.copy()

    # 🔥 CRITICAL FIX: handle weird column formats
    # Sometimes columns are MultiIndex like ('Close', '')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = [str(c).strip().title() for c in df.columns]

    # 🔍 Debug (only first few runs)
    # print(f"{ticker} columns:", df.columns)

    # 🚫 If still no Close → skip
    if "Close" not in df.columns:
        print(f"❌ {ticker}: no Close → {list(df.columns)}")
        return None

    # ✅ Clean data
    df.index.name = "date"
    df = df.reset_index()

    df["ticker"] = ticker

    # Keep only needed columns safely
    keep_cols = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Drop rows where Close missing
    df = df.dropna(subset=["Close"])

    return df


def main() -> None:
    print("Downloading NASDAQ-100 data...")

    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    for i, ticker in enumerate(NASDAQ100, start=1):
        df_t = download_one_ticker(ticker)
        if df_t is None or df_t.empty:
            failed.append(ticker)
            continue

        frames.append(df_t)
        print(f"  [{i:03d}/{len(NASDAQ100)}] loaded {ticker}  ({len(df_t)} rows)")

    if not frames:
        raise RuntimeError("No ticker data downloaded successfully.")

    df = pd.concat(frames, ignore_index=True)

    # Set MultiIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()

    # Compute daily returns
    df["actual_return"] = df.groupby(level="ticker")["Close"].pct_change()

    # IMPORTANT:
    # Do NOT use dropna() on the whole dataframe.
    # That can wipe out everything if any column has missing values.
    df = df.dropna(subset=["Close", "actual_return"]).copy()

    # Save outputs
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)

    avail = pd.DataFrame(
        {
            "available_tickers": sorted(df.index.get_level_values("ticker").unique()),
        }
    )
    AVAILABLE_TICKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    avail.to_csv(AVAILABLE_TICKERS_PATH, index=False)

    print("\nData Summary:")
    print("Tickers:", df.index.get_level_values("ticker").nunique())
    print("Date range:", df.index.get_level_values("date").min(), "to", df.index.get_level_values("date").max())
    print("Shape:", df.shape)

    if failed:
        print(f"\nFailed / skipped tickers ({len(failed)}):")
        print(", ".join(failed))

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Saved available ticker list to: {AVAILABLE_TICKERS_PATH}")
    print("[DONE]")


if __name__ == "__main__":
    main()