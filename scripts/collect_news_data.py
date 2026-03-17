#!/usr/bin/env python3
"""Collect news headlines for tickers into Parquet with deduplication.

Implements:
- Finnhub -> /api/v1/company-news (requests)
- Yahoo Finance RSS -> feedparser
- Async orchestration using asyncio + requests via to_thread
- TF-IDF deduplication (sklearn)
- Partial saves and error logging

Notes:
- NewsAPI removed due to free-tier date-range limitations (HTTP 426 on from/to queries).

Usage (PowerShell):
    python .\scripts\collect_news_data.py --start-date 2024-08-19 --end-date 2025-12-10

Requirements: Python 3.10+, pandas, requests, feedparser, scikit-learn, tqdm, python-dateutil, pyarrow
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import requests
from dateutil import parser as date_parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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
    log_file = os.path.join(output_dir, 'collection_errors.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_api_keys(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API keys file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def _md5_hash(ticker: str, title: str, published_at: str) -> str:
    key = f"{ticker}|{title}|{published_at}"
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def _normalize_title(title: Optional[str]) -> str:
    if title is None:
        return ''
    return title.strip()[:500]


def _to_utc(dt_str: str) -> datetime:
    # dateutil parser handles most formats; ensure tz-aware UTC
    dt = date_parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    async def wait(self):
        now = time.time()
        to_wait = self.min_interval - (now - self._last)
        if to_wait > 0:
            await asyncio.sleep(to_wait)
        self._last = time.time()


async def retry_request(fn, *args, retries=3, base_delay=2, **kwargs):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:  # broad catch; logged by caller
            last_exc = exc
            wait = base_delay * (2 ** (attempt - 1))
            logging.warning(f"Request failed (attempt {attempt}/{retries}): {exc}; retrying in {wait}s")
            await asyncio.sleep(wait)
    raise last_exc


def _finnhub_request(api_key: str, params: Dict) -> Dict:
    url = 'https://finnhub.io/api/v1/company-news'
    params = dict(params)
    params['token'] = api_key
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_yahoo_rss_sync(ticker: str) -> List[Dict]:
    url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}'
    parsed = feedparser.parse(url)
    items = []
    for entry in parsed.entries:
        published = getattr(entry, 'published', getattr(entry, 'updated', None))
        items.append({
            'title': entry.get('title'),
            'description': entry.get('summary'),
            'source': entry.get('source', {}).get('title', '') if isinstance(entry.get('source'), dict) else entry.get('source', ''),
            'publishedAt': published,
            'url': entry.get('link')
        })
    return items


# NewsAPI removed - use Finnhub + Yahoo RSS only


async def fetch_finnhub(ticker: str, start_date: str, end_date: str, api_key: str, rate_limiter: RateLimiter) -> List[Dict]:
    await rate_limiter.wait()
    params = {'symbol': ticker, 'from': start_date, 'to': end_date}
    try:
        payload = await retry_request(_finnhub_request, api_key, params)
    except Exception as exc:
        logging.error(f"Finnhub fetch error for {ticker}: {exc}\n{traceback.format_exc()}")
        return []
    results = []
    for art in payload:
        results.append({
            'title': art.get('headline'),
            'description': None,
            'source': art.get('source'),
            'publishedAt': art.get('datetime') and datetime.fromtimestamp(art.get('datetime'), tz=timezone.utc).isoformat(),
            'url': art.get('url')
        })
    return results


async def fetch_yahoo(ticker: str, rate_limiter: RateLimiter) -> List[Dict]:
    await rate_limiter.wait()
    try:
        payload = await asyncio.to_thread(fetch_yahoo_rss_sync, ticker)
    except Exception as exc:
        logging.error(f"Yahoo RSS fetch error for {ticker}: {exc}\n{traceback.format_exc()}")
        return []
    return payload


def transform_item(raw: Dict, ticker: str) -> Dict:
    title = _normalize_title(raw.get('title'))
    published_raw = raw.get('publishedAt')
    try:
        if published_raw is None:
            published_at = datetime.now(timezone.utc)
        elif isinstance(published_raw, (int, float)):
            published_at = datetime.fromtimestamp(int(published_raw), tz=timezone.utc)
        else:
            published_at = _to_utc(published_raw)
    except Exception:
        published_at = datetime.now(timezone.utc)
    collected_at = datetime.now(timezone.utc)
    return {
        'headline_id': _md5_hash(ticker, title, published_at.isoformat()),
        'ticker': ticker,
        'title': title,
        'description': (raw.get('description') or '')[:1000] if raw.get('description') else None,
        'source': raw.get('source') or None,
        'published_at': published_at,
        'url': raw.get('url') or None,
        'collected_at': collected_at,
    }


def deduplicate(headlines: List[Dict], output_dir: str) -> List[Dict]:
    if not headlines:
        return []
    df = pd.DataFrame(headlines)
    # ensure published_at is datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    # remove tz info safely and convert to POSIX seconds
    df['published_at_ts'] = df['published_at'].dt.tz_localize(None).astype('int64') // 10**9
    keep_mask = [True] * len(df)
    # operate per-ticker
    duplicates_log = []
    grouped = df.groupby('ticker')
    for ticker, group in grouped:
        texts = group['title'].fillna('').tolist()
        if len(texts) <= 1:
            continue
        vec = TfidfVectorizer().fit_transform(texts)
        sim = cosine_similarity(vec)
        idxs = group.index.tolist()
        for i_pos, i in enumerate(idxs):
            if not keep_mask[i_pos + group.index.min() - group.index.min()]:
                continue
            for j_pos, j in enumerate(idxs):
                if i == j:
                    continue
                # ensure j is later than i (we keep earliest)
                t_i = group.loc[i, 'published_at']
                t_j = group.loc[j, 'published_at']
                time_diff = abs((t_j - t_i).total_seconds())
                # indices in similarity matrix correspond to relative positions
                s = sim[i_pos, j_pos]
                if s > 0.85 and time_diff <= 6 * 3600:
                    # mark the later one as duplicate
                    if t_i <= t_j:
                        dup_idx = j
                        keep_mask[df.index.get_loc(dup_idx)] = False
                        duplicates_log.append({
                            'ticker': ticker,
                            'original_id': group.loc[i, 'headline_id'],
                            'duplicate_id': group.loc[j, 'headline_id'],
                            'similarity': float(s),
                            'original_published': group.loc[i, 'published_at'].isoformat(),
                            'duplicate_published': group.loc[j, 'published_at'].isoformat(),
                        })
                    else:
                        dup_idx = i
                        keep_mask[df.index.get_loc(dup_idx)] = False
                        duplicates_log.append({
                            'ticker': ticker,
                            'original_id': group.loc[j, 'headline_id'],
                            'duplicate_id': group.loc[i, 'headline_id'],
                            'similarity': float(s),
                            'original_published': group.loc[j, 'published_at'].isoformat(),
                            'duplicate_published': group.loc[i, 'published_at'].isoformat(),
                        })
    # save duplicates log
    if duplicates_log:
        dedup_path = os.path.join(output_dir, 'deduplication_log.csv')
        dd = pd.DataFrame(duplicates_log)
        if os.path.exists(dedup_path):
            dd.to_csv(dedup_path, mode='a', header=False, index=False)
        else:
            dd.to_csv(dedup_path, index=False)
    kept = df[keep_mask].reset_index(drop=True)
    # convert back to records, dropping helper
    kept = kept.drop(columns=['published_at_ts'])
    # convert published_at and collected_at to datetimes (already datetimes)
    return kept.to_dict(orient='records')


def save_partial(headlines: List[Dict], output_dir: str, partial_name: str = 'raw_headlines_partial.parquet'):
    if not headlines:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, partial_name)
    df_new = pd.DataFrame(headlines)
    # ensure datetime columns
    for c in ['published_at', 'collected_at']:
        if c in df_new.columns:
            df_new[c] = pd.to_datetime(df_new[c])
    if os.path.exists(path):
        try:
            df_old = pd.read_parquet(path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_parquet(path, index=False)


async def collect_for_ticker(ticker: str, company: str, start_date: str, end_date: str, apis: Dict[str, str], out_dir: str, semaphores: Dict[str, RateLimiter], partial_save_fn, progress):
    collected = []
    # run the two sources concurrently (respecting per-source rate limiter)
    finnhub_task = fetch_finnhub(ticker, start_date, end_date, apis.get('finnhub', ''), semaphores['finnhub'])
    yahoo_task = fetch_yahoo(ticker, semaphores['yahoo'])
    try:
        results = await asyncio.gather(finnhub_task, yahoo_task)
    except Exception as exc:
        logging.error(f"Error gathering sources for {ticker}: {exc}\n{traceback.format_exc()}")
        results = [[], []]
    raw_all = []
    for src in results:
        raw_all.extend(src or [])
    # transform
    for raw in raw_all:
        try:
            item = transform_item(raw, ticker)
            collected.append(item)
            if len(collected) % 50 == 0:
                partial_save_fn(collected[-50:], out_dir)
        except Exception as exc:
            logging.error(f"Transform error for ticker {ticker}: {exc}\n{traceback.format_exc()}")
    # progress update
    progress.update(1)
    return collected


async def main_async(tickers: List[str], start_date: str, end_date: str, output_dir: str, api_keys_path: str):
    apis = load_api_keys(api_keys_path)
    setup_logging(output_dir)
    # rate limiters (NewsAPI removed)
    semaphores = {
        'finnhub': RateLimiter(1.0),  # 1 req/sec (~60/min)
        'yahoo': RateLimiter(2.0),  # politeness 2 sec
    }
    tasks = []
    collected_all = []
    with ThreadPoolExecutor(max_workers=8):
        # run per-ticker concurrently but limited by rate-limiters
        with tqdm(total=len(tickers), desc='Tickers') as progress:
            for t in tickers:
                company = TICKER_COMPANY.get(t, t)
                tasks.append(collect_for_ticker(t, company, start_date, end_date, apis, output_dir, semaphores, save_partial, progress))
            results = await asyncio.gather(*tasks)
            for res in results:
                collected_all.extend(res or [])
    # deduplicate (fix timezone handling)
    deduped = deduplicate(collected_all, output_dir)
    # final save
    if deduped:
        df = pd.DataFrame(deduped)
        for c in ['published_at', 'collected_at']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        out_path = os.path.join(output_dir, 'raw_headlines.parquet')
        df.to_parquet(out_path, index=False)
    # metadata
    metadata = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'requested_start_date': start_date,
        'requested_end_date': end_date,
        'tickers': tickers,
        'sources': ['finnhub', 'yahoo'],
        'total_collected': len(collected_all),
        'total_after_dedup': len(deduped),
    }
    # per-ticker counts and date ranges
    if deduped:
        df = pd.DataFrame(deduped)
        per_ticker = df.groupby('ticker').agg(
            count=('headline_id', 'count'),
            earliest=('published_at', 'min'),
            latest=('published_at', 'max')
        ).reset_index()
        per_ticker['earliest'] = per_ticker['earliest'].astype(str)
        per_ticker['latest'] = per_ticker['latest'].astype(str)
        metadata['per_ticker'] = per_ticker.to_dict(orient='records')
        # date coverage gaps (days with zero headlines)
        gaps = {}
        requested_start = date_parser.parse(start_date).date()
        requested_end = date_parser.parse(end_date).date()
        for t in tickers:
            df_t = df[df['ticker'] == t]
            if df_t.empty:
                gaps[t] = {'missing_days': (requested_end - requested_start).days + 1}
                continue
            dates = pd.to_datetime(df_t['published_at']).dt.date.unique()
            present = set(dates.tolist())
            missing = []
            cur = requested_start
            while cur <= requested_end:
                if cur not in present:
                    missing.append(str(cur))
                cur += timedelta(days=1)
            gaps[t] = {'missing_days': len(missing), 'missing_dates_sample': missing[:5]}
        metadata['coverage_gaps'] = gaps
    # save metadata
    meta_path = os.path.join(output_dir, 'collection_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    # print summary
    print('\nCollection Summary:')
    print(json.dumps(metadata, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description='Collect headlines for tickers into Parquet')
    p.add_argument('--tickers', nargs='+', default=list(TICKER_COMPANY.keys()), help='Tickers to collect')
    p.add_argument('--start-date', type=str, default='2024-08-19', help='Start date YYYY-MM-DD')
    p.add_argument('--end-date', type=str, default='2025-12-10', help='End date YYYY-MM-DD')
    p.add_argument('--output-dir', type=str, default=os.path.join('data', 'news'), help='Output directory')
    p.add_argument('--api-keys-file', type=str, default=os.path.join('config', 'api_keys.json'), help='API keys JSON file')
    return p.parse_args()


def main():
    args = parse_args()
    # validate api keys file
    if not os.path.exists(args.api_keys_file):
        print(f"API keys file not found: {args.api_keys_file}. Please add keys to config/api_keys.json")
        sys.exit(2)
    # ensure output dir
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        asyncio.run(main_async(args.tickers, args.start_date, args.end_date, args.output_dir, args.api_keys_file))
    except KeyboardInterrupt:
        logging.info('Interrupted by user')


if __name__ == '__main__':
    main()
