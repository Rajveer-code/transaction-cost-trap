#!/usr/bin/env python3
"""
scripts/download_news_historical.py

Fetch historical news headlines (Aug 2024 -> present) for tickers defined in config/tickers.json
Primary provider: yfinance (yf.Ticker(symbol).news)
Fallback: Yahoo Finance RSS via feedparser

Output:
    data/news/raw_headlines.parquet

Schema (exact columns):
    - ticker (str)
    - published_at (datetime64[ns, UTC])
    - title (str)
    - source (str)
    - url (str)
    - description (str)

Usage:
    python scripts/download_news_historical.py --start-date 2024-08-01
"""
from __future__ import annotations

import json
import logging
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
import argparse

import pandas as pd
import yfinance as yf
import feedparser
from dateutil import parser as dateparser

# Project paths
ROOT = Path(__file__).resolve().parents[1]
TICKERS_FILE = ROOT / "config" / "tickers.json"
OUTPUT_DIR = ROOT / "data" / "news"
OUTPUT_FILE = OUTPUT_DIR / "raw_headlines.parquet"

# Date range default
START_DATE = datetime(2024, 8, 1, tzinfo=timezone.utc)

# Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("download_news_historical")


def load_tickers(path: Path) -> list[str]:
    """Load tickers from config file. Accepts dict or list formats."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # If file maps ticker->meta, return keys
            return [str(k).upper() for k in data.keys()]
        if isinstance(data, list):
            return [str(x).upper() for x in data]
    except Exception as e:
        logger.error(f"Failed to load tickers from {path}: {e}")
    return []


def _to_utc(val) -> datetime | None:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(int(val), tz=timezone.utc)
        if isinstance(val, datetime):
            if val.tzinfo is None:
                return val.replace(tzinfo=timezone.utc)
            return val.astimezone(timezone.utc)
        # parse string
        parsed = dateparser.parse(str(val))
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _hash_id(ticker: str, published_at: datetime | None, title: str) -> str:
    published_iso = published_at.isoformat() if published_at is not None else ""
    s = f"{ticker}|{published_iso}|{title or ''}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def fetch_with_yf(ticker: str) -> list[dict]:
    articles = []
    try:
        t = yf.Ticker(ticker)
        raw = getattr(t, "news", None)
        if not raw:
            return []
        for item in raw:
            try:
                title = (item.get("title") or "").strip()
                url = item.get("link") or item.get("url") or ""
                source = (item.get("publisher") or item.get("provider") or item.get("source") or "").strip() or "yfinance"
                published_at = None
                if "providerPublishTime" in item:
                    published_at = _to_utc(item.get("providerPublishTime"))
                elif "pubDate" in item:
                    published_at = _to_utc(item.get("pubDate"))
                else:
                    published_at = _to_utc(item.get("published_at") or item.get("date"))
                description = (item.get("summary") or item.get("description") or "") or ""
                articles.append({
                    "ticker": ticker.upper(),
                    "title": title,
                    "url": url,
                    "source": source,
                    "published_at": published_at,
                    "description": description,
                })
            except Exception:
                continue
    except Exception as e:
        logger.exception(f"yfinance failed for {ticker}: {e}")
    return articles


def fetch_with_rss(ticker: str) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    articles = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            try:
                title = (entry.get("title") or "").strip()
                link = entry.get("link") or ""
                src = (entry.get("source", {}).get("title") or "YahooRSS")
                published_at = _to_utc(entry.get("published") or entry.get("updated") or entry.get("pubDate"))
                description = (entry.get("summary") or entry.get("description") or "") or ""
                articles.append({
                    "ticker": ticker.upper(),
                    "title": title,
                    "url": link,
                    "source": src,
                    "published_at": published_at,
                    "description": description,
                })
            except Exception:
                continue
    except Exception as e:
        logger.exception(f"RSS fallback failed for {ticker}: {e}")
    return articles


def run(tickers_file: Path = TICKERS_FILE, output_file: Path = OUTPUT_FILE, start_date: datetime = START_DATE,
        end_date: datetime | None = None, pause_between: float = 0.5, max_per_ticker: int | None = None):
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    OUTPUT_DIR = output_file.parent
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers(tickers_file)
    if not tickers:
        logger.error("No tickers found; exiting")
        return

    seen_ids: set[str] = set()
    rows: list[dict] = []

    for ticker in tickers:
        ticker = str(ticker).upper()
        logger.info(f"Fetching news for {ticker} (yfinance)")
        yf_articles = fetch_with_yf(ticker)
        yf_filtered = []
        for a in yf_articles:
            pa = a.get("published_at")
            if pa is None:
                continue
            if pa < start_date or pa > end_date:
                continue
            yf_filtered.append(a)
        logger.info(f"Fetched {len(yf_filtered)} headlines for {ticker} via yfinance")
        count_added = 0
        for a in yf_filtered:
            uid = _hash_id(ticker, a.get("published_at"), a.get("title", ""))
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
            rows.append({
                "ticker": ticker,
                "published_at": a.get("published_at"),
                "title": a.get("title") or "",
                "source": a.get("source") or "",
                "url": a.get("url") or "",
                "description": a.get("description") or "",
            })
            count_added += 1
            if max_per_ticker and count_added >= max_per_ticker:
                break

        if not max_per_ticker or count_added < (max_per_ticker or 999999):
            logger.info(f"Using RSS fallback for {ticker} (if needed)")
            rss_articles = fetch_with_rss(ticker)
            rss_filtered = []
            for a in rss_articles:
                pa = a.get("published_at")
                if pa is None:
                    continue
                if pa < start_date or pa > end_date:
                    continue
                rss_filtered.append(a)
            logger.info(f"Fetched {len(rss_filtered)} headlines for {ticker} via RSS")
            for a in rss_filtered:
                uid = _hash_id(ticker, a.get("published_at"), a.get("title", ""))
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                rows.append({
                    "ticker": ticker,
                    "published_at": a.get("published_at"),
                    "title": a.get("title") or "",
                    "source": a.get("source") or "",
                    "url": a.get("url") or "",
                    "description": a.get("description") or "",
                })
                count_added += 1
                if max_per_ticker and count_added >= max_per_ticker:
                    break

        logger.info(f"Added {count_added} unique headlines for {ticker}")
        time.sleep(pause_between)

    # Build DataFrame
    if not rows:
        logger.warning("No headlines collected; creating empty DataFrame with schema")
        df = pd.DataFrame(columns=["ticker", "published_at", "title", "source", "url", "description"])
    else:
        df = pd.DataFrame(rows)
        df = df[["ticker", "published_at", "title", "source", "url", "description"]]
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df[(df["published_at"] >= pd.Timestamp(start_date)) & (df["published_at"] <= pd.Timestamp(end_date))]
        df = df.sort_values(["ticker", "published_at"], ascending=[True, False]).reset_index(drop=True)

    # Deduplicate by hash
    if not df.empty:
        df["_uid"] = df.apply(lambda r: hashlib.md5(f"{r.ticker}|{r.published_at.isoformat()}|{r.title}".encode("utf-8")).hexdigest(), axis=1)
        df = df.drop_duplicates(subset=["_uid"]).drop(columns=["_uid"]).reset_index(drop=True)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

    # Save parquet
    try:
        df.to_parquet(output_file, index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} headlines to {output_file}")
    except Exception as e:
        logger.exception(f"Failed to save parquet to {output_file}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Download historical news headlines (yfinance + RSS fallback)")
    p.add_argument("--tickers-file", type=Path, default=TICKERS_FILE)
    p.add_argument("--output-file", type=Path, default=OUTPUT_FILE)
    p.add_argument("--start-date", type=str, default=START_DATE.isoformat())
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--max-per-ticker", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        sd = dateparser.parse(args.start_date)
        if sd.tzinfo is None:
            sd = sd.replace(tzinfo=timezone.utc)
    except Exception:
        sd = START_DATE
    if args.end_date:
        try:
            ed = dateparser.parse(args.end_date)
            if ed.tzinfo is None:
                ed = ed.replace(tzinfo=timezone.utc)
        except Exception:
            ed = datetime.now(timezone.utc)
    else:
        ed = datetime.now(timezone.utc)

    run(
        tickers_file=args.tickers_file,
        output_file=args.output_file,
        start_date=sd,
        end_date=ed,
        pause_between=args.pause,
        max_per_ticker=args.max_per_ticker,
    )
