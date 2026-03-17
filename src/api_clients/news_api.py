"""
news_api.py (FIXED)
===================

Now loads API keys from config/api_keys.json via api_key_manager.

Key changes:
- get_default_providers() now uses load_api_keys()
- Falls back to environment variables if file doesn't exist
- Better error logging when keys are missing

Author: Rajveer Singh Pall
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORT API KEY MANAGER
# ============================================================

try:
    from src.utils.api_key_manager import load_api_keys
    API_KEY_MANAGER_AVAILABLE = True
except ImportError:
    API_KEY_MANAGER_AVAILABLE = False
    print("⚠️  api_key_manager not available, falling back to environment variables")

# ============================================================
# LOGGING CONFIG
# ============================================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

# ============================================================
# DATE / TIME HELPERS
# ============================================================

def _to_datetime(dt: Any) -> datetime:
    """Convert various date formats to datetime."""
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, str):
        return pd.to_datetime(dt)
    raise ValueError(f"Unsupported date format: {type(dt)}")


def _default_date_range(days_back: int = 3) -> Tuple[datetime, datetime]:
    """Return a default [from_date, to_date] window."""
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=days_back)
    return from_dt, to_dt

# ============================================================
# QUERY BUILDER
# ============================================================

def build_news_query(
    ticker: str,
    ticker_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: str = "default",
) -> str:
    """Build a sensible news query string for a given ticker."""
    ticker = ticker.upper()
    base_meta = ticker_metadata.get(ticker, {}) if ticker_metadata else {}

    company_name = base_meta.get("company_name", "").strip()
    short_name = base_meta.get("short_name", "").strip()
    ceo_name = base_meta.get("ceo", "").strip()

    tokens = []
    if company_name:
        tokens.append(f'"{company_name}"')
    if short_name and short_name.lower() not in company_name.lower():
        tokens.append(f'"{short_name}"')
    tokens.append(ticker)

    base_query = " OR ".join(tokens)

    if mode == "ceo":
        if ceo_name:
            return f"({base_query}) AND (\"{ceo_name}\")"
    elif mode == "macro":
        macro_terms = ["stock", "shares", "earnings", "guidance"]
        return f"({base_query}) AND ({' OR '.join(macro_terms)})"

    return base_query

# ============================================================
# BASE PROVIDER CLASS
# ============================================================

class BaseNewsProvider:
    """Abstract base class for news providers."""
    name: str = "base"

    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def fetch(
        self,
        ticker: str,
        query: str,
        from_date: datetime,
        to_date: datetime,
        max_articles: int = 50,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("fetch must be implemented by subclasses.")

# ============================================================
# NEWSAPI PROVIDER
# ============================================================

class NewsAPIProvider(BaseNewsProvider):
    """NewsAPI (newsapi.org) provider."""
    name = "newsapi"
    BASE_URL = "https://newsapi.org/v2/everything"

    def fetch(
        self,
        ticker: str,
        query: str,
        from_date: datetime,
        to_date: datetime,
        max_articles: int = 50,
    ) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "from": from_date.isoformat(timespec="seconds"),
            "to": to_date.isoformat(timespec="seconds"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),
            "apiKey": self.api_key,
        }

        logger.info(f"[NewsAPI] Fetching news for {ticker} | query='{query}'")
        resp = self.session.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", []) or []
        normalized: List[Dict[str, Any]] = []

        for item in articles:
            title = item.get("title") or ""
            desc = item.get("description") or ""
            headline = title if title else desc

            published_str = item.get("publishedAt")
            published_dt = pd.to_datetime(published_str) if published_str else None

            normalized.append({
                "ticker": ticker.upper(),
                "headline": headline.strip(),
                "source": (item.get("source") or {}).get("name", "NewsAPI"),
                "published_at": published_dt,
                "url": item.get("url"),
                "provider": self.name,
                "raw": item,
            })

        return normalized

# ============================================================
# FINNHUB PROVIDER
# ============================================================

class FinnhubNewsProvider(BaseNewsProvider):
    """Finnhub company news provider (finnhub.io)."""
    name = "finnhub"
    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def fetch(
        self,
        ticker: str,
        query: str,
        from_date: datetime,
        to_date: datetime,
        max_articles: int = 50,
    ) -> List[Dict[str, Any]]:
        params = {
            "symbol": ticker.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": self.api_key,
        }

        logger.info(f"[Finnhub] Fetching company news for {ticker}")
        resp = self.session.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data or []
        normalized: List[Dict[str, Any]] = []

        for item in articles[:max_articles]:
            headline = (item.get("headline") or "").strip()
            if not headline:
                continue

            dt_unix = item.get("datetime")
            published_dt = (
                datetime.utcfromtimestamp(dt_unix) if isinstance(dt_unix, (int, float)) else None
            )

            normalized.append({
                "ticker": ticker.upper(),
                "headline": headline,
                "source": item.get("source") or "Finnhub",
                "published_at": published_dt,
                "url": item.get("url"),
                "provider": self.name,
                "raw": item,
            })

        return normalized

# ============================================================
# ALPHA VANTAGE NEWS PROVIDER
# ============================================================

class AlphaVantageNewsProvider(BaseNewsProvider):
    """Alpha Vantage News & Sentiment API provider."""
    name = "alphavantage"
    BASE_URL = "https://www.alphavantage.co/query"

    def fetch(
        self,
        ticker: str,
        query: str,
        from_date: datetime,
        to_date: datetime,
        max_articles: int = 50,
    ) -> List[Dict[str, Any]]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "apikey": self.api_key,
        }

        logger.info(f"[AlphaVantage] Fetching news sentiment for {ticker}")
        resp = self.session.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        feeds = data.get("feed", []) or []
        normalized: List[Dict[str, Any]] = []

        for item in feeds[:max_articles]:
            title = item.get("title") or ""
            headline = title.strip()
            if not headline:
                continue

            time_str = item.get("time_published")
            published_dt = pd.to_datetime(time_str) if time_str else None

            if published_dt is not None:
                if published_dt < from_date or published_dt > to_date:
                    continue

            normalized.append({
                "ticker": ticker.upper(),
                "headline": headline,
                "source": item.get("source") or "AlphaVantage",
                "published_at": published_dt,
                "url": item.get("url"),
                "provider": self.name,
                "raw": item,
            })

        return normalized

# ============================================================
# PROVIDER FACTORY (FIXED TO USE FILE-BASED KEYS)
# ============================================================

def get_default_providers() -> List[BaseNewsProvider]:
    """
    Build list of provider instances using keys from config file or environment.
    
    Priority:
    1. config/api_keys.json (via api_key_manager)
    2. Environment variables (fallback)
    
    Returns:
        List of instantiated providers
    """
    providers: List[BaseNewsProvider] = []
    session = requests.Session()

    # Load keys from file (or environment as fallback)
    if API_KEY_MANAGER_AVAILABLE:
        keys = load_api_keys()
        finnhub_key = keys.get("finnhub")
        newsapi_key = keys.get("newsapi")
        av_key = keys.get("alphavantage")
        
        logger.info("✅ Loaded API keys via api_key_manager")
    else:
        # Fallback to environment variables
        import os
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        newsapi_key = os.getenv("NEWSAPI_KEY")
        av_key = os.getenv("ALPHAVANTAGE_API_KEY")
        
        logger.warning("⚠️  Using environment variables (api_key_manager not available)")

    # Initialize providers with available keys
    if finnhub_key:
        providers.append(FinnhubNewsProvider(finnhub_key, session=session))
        logger.info("✅ Finnhub provider initialized")
    else:
        logger.warning("⚠️  Finnhub API key not found - skipping Finnhub")

    if newsapi_key:
        providers.append(NewsAPIProvider(newsapi_key, session=session))
        logger.info("✅ NewsAPI provider initialized")
    else:
        logger.warning("⚠️  NewsAPI key not found - skipping NewsAPI")

    if av_key:
        providers.append(AlphaVantageNewsProvider(av_key, session=session))
        logger.info("✅ Alpha Vantage provider initialized")
    else:
        logger.warning("⚠️  Alpha Vantage API key not found - skipping Alpha Vantage")

    if not providers:
        logger.error("❌ NO API PROVIDERS INITIALIZED! Please configure API keys in Settings.")

    return providers

# ============================================================
# CORE FETCH FUNCTION WITH FALLBACK
# ============================================================

def fetch_news_with_fallback(
    ticker: str,
    ticker_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    from_date: Optional[Any] = None,
    to_date: Optional[Any] = None,
    mode: str = "default",
    max_articles: int = 50,
    providers: Optional[List[BaseNewsProvider]] = None,
) -> List[Dict[str, Any]]:
    """Fetch news for a given ticker using multiple providers with fallback."""
    if from_date is None or to_date is None:
        from_dt, to_dt = _default_date_range(days_back=3)
    else:
        from_dt = _to_datetime(from_date)
        to_dt = _to_datetime(to_date)

    query = build_news_query(ticker, ticker_metadata or {}, mode=mode)

    if providers is None:
        providers = get_default_providers()

    if not providers:
        logger.error("❌ No providers available. Check API keys in Settings.")
        return []

    all_articles: List[Dict[str, Any]] = []

    for provider in providers:
        try:
            logger.info(f"🔍 Trying provider: {provider.name}")
            articles = provider.fetch(
                ticker=ticker,
                query=query,
                from_date=from_dt,
                to_date=to_dt,
                max_articles=max_articles,
            )

            if articles:
                logger.info(f"✅ {provider.name} returned {len(articles)} articles for {ticker}")
                all_articles = articles
                break
            else:
                logger.warning(f"⚠️  {provider.name} returned 0 articles, trying next...")
        except Exception as e:
            logger.error(f"❌ Error with {provider.name}: {str(e)[:200]}")
            continue

    if not all_articles:
        logger.warning(f"⚠️  No articles found for {ticker} across all providers")

    return all_articles

# ============================================================
# DATAFRAME HELPERS
# ============================================================

def articles_to_dataframe(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of article dicts into a pandas DataFrame."""
    if not articles:
        return pd.DataFrame(
            columns=["date", "ticker", "headline", "source", "published_at", "url", "provider"]
        )

    df = pd.DataFrame(articles)

    for col in ["ticker", "headline", "source", "published_at", "url", "provider"]:
        if col not in df.columns:
            df[col] = None

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["date"] = df["published_at"].dt.date

    df = df[["date", "ticker", "headline", "source", "published_at", "url", "provider"]]

    return df


def fetch_news_dataframe_for_ticker(
    ticker: str,
    ticker_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    from_date: Optional[Any] = None,
    to_date: Optional[Any] = None,
    mode: str = "default",
    max_articles: int = 50,
    providers: Optional[List[BaseNewsProvider]] = None,
) -> pd.DataFrame:
    """Fetch news and return as DataFrame."""
    articles = fetch_news_with_fallback(
        ticker=ticker,
        ticker_metadata=ticker_metadata,
        from_date=from_date,
        to_date=to_date,
        mode=mode,
        max_articles=max_articles,
        providers=providers,
    )
    df = articles_to_dataframe(articles)
    return df

# ============================================================
# MODULE SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Testing news_api.py...")
    print("=" * 60)
    
    # Test provider initialization
    providers = get_default_providers()
    print(f"\n✅ Initialized {len(providers)} providers")
    
    if providers:
        # Test fetch
        ticker = "AAPL"
        print(f"\n🔍 Fetching news for {ticker}...")
        
        df = fetch_news_dataframe_for_ticker(ticker, max_articles=5)
        print(f"\n✅ Fetched {len(df)} articles")
        
        if not df.empty:
            print("\n📰 Sample headlines:")
            for i, row in df.head(3).iterrows():
                print(f"  - {row['headline'][:80]}...")
    else:
        print("\n⚠️  No providers initialized. Configure API keys in Settings.")
    
    print("\n" + "=" * 60)