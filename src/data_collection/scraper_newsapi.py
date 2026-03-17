"""
NewsAPI Scraper - Structured Financial News with Guaranteed Dates
This module uses NewsAPI.org to fetch financial news with accurate timestamps

WHY NewsAPI?
1. Guaranteed accurate publication dates (no estimation needed)
2. Rich metadata (author, source, description)
3. Structured JSON format (easy to parse)
4. Filtered by financial sources (Bloomberg, Reuters, WSJ)

HOW TO GET API KEY:
1. Go to https://newsapi.org/
2. Sign up for free account
3. Get API key (100 requests/day free)
4. Put key in config/config.py
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
sys.path.append('../..')
from config.config import *

class NewsAPIScraper:
    """
    Scrapes financial news from NewsAPI with guaranteed accurate dates
    
    API Limits (Free Tier):
    - 100 requests per day
    - 1 month of historical data
    - 100 articles per request
    """
    
    def __init__(self, api_key=None):
        """
        Initialize NewsAPI scraper
        
        Parameters:
        -----------
        api_key : str, optional
            NewsAPI key. If None, loads from config
        """
        self.api_key = api_key or NEWSAPI_KEY
        
        if self.api_key == "YOUR_API_KEY_HERE":
            print("\n" + "="*70)
            print("⚠️ WARNING: No NewsAPI key configured!")
            print("="*70)
            print("To use NewsAPI:")
            print("1. Go to https://newsapi.org/register")
            print("2. Get your free API key")
            print("3. Update NEWSAPI_KEY in config/config.py")
            print("="*70)
            raise ValueError("NewsAPI key not configured")
        
        # Base URL for NewsAPI
        self.base_url = "https://newsapi.org/v2/everything"
        
        # Track API usage
        self.requests_made = 0
        self.max_requests = 100  # Free tier limit
    
    def search_ticker_news(self, ticker, from_date=None, to_date=None, 
                          page_size=100, language='en'):
        """
        Search news articles for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL')
        from_date : str, optional
            Start date 'YYYY-MM-DD'. Defaults to 30 days ago
        to_date : str, optional
            End date 'YYYY-MM-DD'. Defaults to today
        page_size : int
            Number of articles per request (max 100)
        language : str
            Article language code
            
        Returns:
        --------
        DataFrame with columns: [date, ticker, headline, url, source, 
                                 author, description, published_at]
        
        HOW IT WORKS:
        1. Constructs API request with search query
        2. Filters by financial news sources
        3. Sends GET request to NewsAPI
        4. Parses JSON response
        5. Returns structured DataFrame
        """
        
        print(f"\n{'='*60}")
        print(f"Fetching NewsAPI data for {ticker}...")
        print(f"{'='*60}")
        
        # Set default date range (last 30 days)
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Date range: {from_date} to {to_date}")
        
        # Get company name for better search results
        # This mapping helps find more relevant articles
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'NVDA': 'Nvidia',
            'META': 'Meta OR Facebook'
        }
        
        # Build search query
        # We search for both ticker AND company name for better results
        company = company_names.get(ticker, ticker)
        query = f'({ticker} OR {company}) AND (stock OR shares OR market OR trading)'
        
        print(f"Search query: {query}")
        
        # API request parameters
        params = {
            'q': query,                    # Search query
            'from': from_date,             # Start date
            'to': to_date,                 # End date
            'language': language,          # Language filter
            'sortBy': 'publishedAt',       # Sort by publication date
            'pageSize': page_size,         # Articles per request
            'apiKey': self.api_key,        # Authentication
            'domains': self._get_domains() # Filter by financial sources
        }
        
        articles = []
        
        try:
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            self.requests_made += 1
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check API status
            if data['status'] != 'ok':
                print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
            
            # Extract articles
            total_results = data['totalResults']
            articles_data = data['articles']
            
            print(f"Found {total_results} articles (retrieving {len(articles_data)})")
            
            # Process each article
            for article in articles_data:
                try:
                    # Extract article metadata
                    # NewsAPI provides rich structured data
                    
                    # CRITICAL: Extract accurate publication date
                    # Format: "2024-11-24T10:30:00Z" (ISO 8601)
                    published_at = article.get('publishedAt', '')
                    
                    if published_at:
                        # Parse ISO format and extract date
                        date_obj = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        date_str = date_obj.strftime('%Y-%m-%d')
                    else:
                        # Skip if no date (should never happen with NewsAPI)
                        print(f"  ⚠️ Skipping article without date")
                        continue
                    
                    # Extract other fields
                    headline = article.get('title', '')
                    url = article.get('url', '')
                    source = article.get('source', {}).get('name', 'Unknown')
                    author = article.get('author', 'Unknown')
                    description = article.get('description', '')
                    
                    # Skip if essential fields are missing
                    if not headline or not url:
                        continue
                    
                    articles.append({
                        'date': date_str,
                        'ticker': ticker,
                        'headline': headline,
                        'url': url,
                        'source': source,
                        'author': author,
                        'description': description,
                        'published_at': published_at  # Keep full timestamp
                    })
                    
                except Exception as e:
                    print(f"  ⚠️ Error parsing article: {str(e)[:50]}...")
                    continue
            
            print(f"✅ Successfully retrieved {len(articles)} articles for {ticker}")
            
            # Check API usage
            print(f"API requests used: {self.requests_made}/{self.max_requests}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(articles)
    
    def _get_domains(self):
        """
        Get comma-separated list of trusted financial news domains
        
        WHY FILTER BY DOMAIN?
        - Focus on high-quality financial journalism
        - Avoid low-quality blog posts or spam
        - Ensure relevance to stock market
        """
        
        # Trusted financial news sources
        # These sources have professional financial reporting
        domains = [
            'bloomberg.com',
            'reuters.com',
            'wsj.com',              # Wall Street Journal
            'ft.com',               # Financial Times
            'cnbc.com',
            'marketwatch.com',
            'fool.com',             # Motley Fool
            'seekingalpha.com',
            'barrons.com',
            'businessinsider.com',
            'forbes.com',
            'fortune.com'
        ]
        
        # Convert to comma-separated string
        return ','.join(domains)
    
    def scrape_multiple_tickers(self, tickers=None, from_date=None, 
                               to_date=None, max_articles_per_ticker=100):
        """
        Scrape news for multiple tickers
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols. If None, uses config.TICKERS
        from_date : str, optional
            Start date 'YYYY-MM-DD'
        to_date : str, optional
            End date 'YYYY-MM-DD'
        max_articles_per_ticker : int
            Max articles per ticker
            
        Returns:
        --------
        Combined DataFrame with all articles
        
        NOTE: Be mindful of API limits (100 requests/day)
        Each ticker = 1 request
        """
        
        if tickers is None:
            tickers = TICKERS
        
        print("\n" + "="*70)
        print(f"NEWSAPI MULTI-TICKER SCRAPING")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Articles per ticker: {max_articles_per_ticker}")
        print("="*70)
        
        # Check if we'll exceed API limit
        if len(tickers) > (self.max_requests - self.requests_made):
            print(f"\n⚠️ WARNING: This will use {len(tickers)} API requests")
            print(f"You have {self.max_requests - self.requests_made} requests remaining today")
            
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return pd.DataFrame()
        
        all_articles = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            # Fetch articles for this ticker
            df = self.search_ticker_news(
                ticker=ticker,
                from_date=from_date,
                to_date=to_date,
                page_size=max_articles_per_ticker
            )
            
            if not df.empty:
                all_articles.append(df)
            
            # Small delay between requests (be polite)
            if i < len(tickers):
                time.sleep(1)
        
        # Combine all articles
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            
            print("\n" + "="*70)
            print("POST-PROCESSING")
            print("="*70)
            print(f"Total articles before deduplication: {len(combined_df)}")
            
            # Remove duplicates by URL
            combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
            
            print(f"Total articles after deduplication: {len(combined_df)}")
            
            # Sort by date
            combined_df = combined_df.sort_values('date', ascending=False)
            
            return combined_df
        else:
            print("\n❌ No articles retrieved!")
            return pd.DataFrame()
    
    def save_to_csv(self, df, filename=NEWS_NEWSAPI_PATH):
        """
        Save NewsAPI data to CSV
        
        Parameters:
        -----------
        df : DataFrame
            Data to save
        filename : str
            Output file path
        """
        
        if df.empty:
            print("⚠️ Cannot save empty DataFrame!")
            return
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        print("\n" + "="*70)
        print("SAVE SUMMARY")
        print("="*70)
        print(f"✅ Saved {len(df)} articles to {filename}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Tickers: {df['ticker'].nunique()} unique")
        print(f"\nArticles per source:")
        print(df['source'].value_counts().head(10))
        print("="*70)
    
    def get_usage_stats(self):
        """
        Get current API usage statistics
        
        Helps track daily quota
        """
        print("\n" + "="*70)
        print("NEWSAPI USAGE STATISTICS")
        print("="*70)
        print(f"Requests made today: {self.requests_made}")
        print(f"Requests remaining: {self.max_requests - self.requests_made}")
        print(f"Daily limit: {self.max_requests}")
        print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Run this script to fetch news from NewsAPI
    
    REQUIREMENTS:
    1. Valid NewsAPI key in config.py
    2. Internet connection
    3. API quota available (100/day)
    
    Usage:
        python scraper_newsapi.py
    """
    
    print("\n" + "="*70)
    print("NEWSAPI FINANCIAL NEWS SCRAPER")
    print("="*70)
    
    try:
        # Initialize scraper
        scraper = NewsAPIScraper()
        
        # Fetch news for all tickers (last 30 days)
        news_df = scraper.scrape_multiple_tickers(
            tickers=TICKERS,
            from_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            to_date=datetime.now().strftime('%Y-%m-%d'),
            max_articles_per_ticker=100
        )
        
        # Save results
        if not news_df.empty:
            scraper.save_to_csv(news_df)
            
            # Display sample
            print("\n" + "="*70)
            print("SAMPLE DATA (First 5 rows)")
            print("="*70)
            print(news_df[['date', 'ticker', 'headline', 'source']].head())
            
            # Display statistics
            print("\n" + "="*70)
            print("STATISTICS")
            print("="*70)
            print(f"Total articles: {len(news_df)}")
            print(f"\nArticles per ticker:")
            print(news_df['ticker'].value_counts())
            
        # Show API usage
        scraper.get_usage_stats()
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease configure NewsAPI key before running.")
    except Exception as e:
        print(f"\n❌ Error: {e}")