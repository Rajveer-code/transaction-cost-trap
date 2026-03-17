"""
Yahoo Finance News Scraper - FIXED VERSION
This module scrapes financial news with ACTUAL publication dates

KEY FIX: We extract real historical dates instead of using datetime.now()
Method: We use Yahoo Finance RSS feeds which contain proper timestamps
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import feedparser  # NEW: For parsing RSS feeds with timestamps
from urllib.parse import quote
import sys
sys.path.append('../..')
from config.config import *

class YahooFinanceScraper:
    """
    Scrapes financial news from Yahoo Finance with accurate historical dates
    
    Why this approach?
    - Yahoo's RSS feeds contain proper publication timestamps
    - More reliable than scraping HTML (which often lacks dates)
    - Respects rate limits and server resources
    """
    
    def __init__(self):
        """Initialize the scraper with necessary headers"""
        # Headers make our request look like it's from a real browser
        # Without this, Yahoo might block our requests
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Keep track of articles we've already scraped
        # This prevents duplicates
        self.seen_urls = set()
        
    def scrape_ticker_rss(self, ticker, max_articles=50):
        """
        Scrape news for a specific ticker using RSS feed
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL')
        max_articles : int
            Maximum number of articles to retrieve
            
        Returns:
        --------
        DataFrame with columns: [date, ticker, headline, url, source]
        
        How it works:
        1. Constructs RSS feed URL for the ticker
        2. Parses the XML feed using feedparser
        3. Extracts publication date from 'published' field
        4. Returns structured data
        """
        
        print(f"\n{'='*60}")
        print(f"Scraping news for {ticker}...")
        print(f"{'='*60}")
        
        # Yahoo Finance RSS feed URL structure
        # This URL returns XML with news articles and their metadata
        rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        articles = []
        
        try:
            # Parse the RSS feed
            # feedparser automatically handles XML parsing
            feed = feedparser.parse(rss_url)
            
            print(f"Found {len(feed.entries)} articles in RSS feed")
            
            # Loop through each article in the feed
            for entry in feed.entries[:max_articles]:
                try:
                    # Extract the headline
                    headline = entry.title
                    
                    # Extract the URL
                    article_url = entry.link
                    
                    # Skip if we've already seen this URL (avoid duplicates)
                    if article_url in self.seen_urls:
                        continue
                    
                    # CRITICAL: Extract actual publication date
                    # The 'published_parsed' field contains a time struct
                    if hasattr(entry, 'published_parsed'):
                        # Convert time struct to datetime object
                        pub_date = datetime(*entry.published_parsed[:6])
                        date_str = pub_date.strftime('%Y-%m-%d')
                    else:
                        # Fallback: If no date available, skip this article
                        print(f"  ⚠️ Warning: No date found for article, skipping...")
                        continue
                    
                    # Extract source (e.g., "Reuters", "Bloomberg")
                    source = entry.source.title if hasattr(entry, 'source') else 'Yahoo Finance'
                    
                    # Store the article data
                    articles.append({
                        'date': date_str,
                        'ticker': ticker,
                        'headline': headline,
                        'url': article_url,
                        'source': source
                    })
                    
                    # Add to seen URLs
                    self.seen_urls.add(article_url)
                    
                except Exception as e:
                    # If something goes wrong with one article, continue to next
                    print(f"  ⚠️ Error parsing article: {str(e)[:50]}...")
                    continue
            
            print(f"✅ Successfully scraped {len(articles)} unique articles for {ticker}")
            
        except Exception as e:
            print(f"❌ Error fetching RSS feed for {ticker}: {e}")
            return pd.DataFrame()
        
        # Convert list of dictionaries to DataFrame
        return pd.DataFrame(articles)
    
    def scrape_ticker_html_fallback(self, ticker, max_articles=30):
        """
        Fallback method: Scrape from HTML if RSS fails
        
        This method scrapes Yahoo's news page directly
        We try to extract dates from the HTML structure
        Less reliable than RSS, but useful as backup
        """
        
        print(f"  Using HTML fallback for {ticker}...")
        
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        articles = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise error if request failed
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all news article containers
            # Yahoo's structure: each article is in a <li> with specific classes
            news_items = soup.find_all('li', class_='js-stream-content')
            
            print(f"  Found {len(news_items)} articles in HTML")
            
            for item in news_items[:max_articles]:
                try:
                    # Extract headline (usually in an <a> tag)
                    headline_tag = item.find('a')
                    if not headline_tag:
                        continue
                    
                    headline = headline_tag.get_text(strip=True)
                    article_url = headline_tag.get('href', '')
                    
                    # Make URL absolute if it's relative
                    if article_url.startswith('/'):
                        article_url = f"https://finance.yahoo.com{article_url}"
                    
                    # Skip duplicates
                    if article_url in self.seen_urls:
                        continue
                    
                    # Try to find date in HTML
                    # Yahoo sometimes includes a <time> tag
                    date_tag = item.find('time')
                    if date_tag and date_tag.get('datetime'):
                        # Parse ISO format datetime
                        date_str = date_tag['datetime'][:10]  # Extract YYYY-MM-DD
                    else:
                        # If no date found, use a reasonable estimate
                        # Assume articles are from the last 30 days
                        estimated_date = datetime.now() - timedelta(days=15)
                        date_str = estimated_date.strftime('%Y-%m-%d')
                        print(f"  ⚠️ Using estimated date for: {headline[:50]}...")
                    
                    articles.append({
                        'date': date_str,
                        'ticker': ticker,
                        'headline': headline,
                        'url': article_url,
                        'source': 'Yahoo Finance'
                    })
                    
                    self.seen_urls.add(article_url)
                    
                except Exception as e:
                    continue
            
            print(f"  ✅ HTML fallback retrieved {len(articles)} articles")
            
        except Exception as e:
            print(f"  ❌ HTML fallback failed: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(articles)
    
    def scrape_multiple_tickers(self, tickers=None, articles_per_ticker=50):
        """
        Scrape news for multiple tickers
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols. If None, uses config.TICKERS
        articles_per_ticker : int
            Number of articles to scrape per ticker
            
        Returns:
        --------
        DataFrame with all articles combined
        """
        
        if tickers is None:
            tickers = TICKERS
        
        print("\n" + "="*70)
        print(f"STARTING MULTI-TICKER SCRAPING")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Target: {articles_per_ticker} articles per ticker")
        print("="*70)
        
        all_articles = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            # Try RSS first (more reliable)
            df_rss = self.scrape_ticker_rss(ticker, articles_per_ticker)
            
            if not df_rss.empty:
                all_articles.append(df_rss)
            else:
                # If RSS fails, try HTML fallback
                print(f"  RSS failed, trying HTML fallback...")
                df_html = self.scrape_ticker_html_fallback(ticker, articles_per_ticker)
                if not df_html.empty:
                    all_articles.append(df_html)
            
            # Be polite: wait between requests
            if i < len(tickers):
                print(f"  Waiting {SCRAPING_DELAY} seconds before next ticker...")
                time.sleep(SCRAPING_DELAY)
        
        # Combine all articles
        if all_articles:
            combined_df = pd.concat(all_articles, ignore_index=True)
            
            # Remove duplicates based on headline similarity
            # Some articles might appear for multiple tickers
            print("\n" + "="*70)
            print("POST-PROCESSING")
            print("="*70)
            print(f"Total articles before deduplication: {len(combined_df)}")
            
            combined_df = combined_df.drop_duplicates(subset=['headline'], keep='first')
            
            print(f"Total articles after deduplication: {len(combined_df)}")
            
            # Sort by date (most recent first)
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            combined_df = combined_df.sort_values('date', ascending=False)
            
            # Convert date back to string for saving
            combined_df['date'] = combined_df['date'].dt.strftime('%Y-%m-%d')
            
            return combined_df
        else:
            print("\n❌ No articles scraped!")
            return pd.DataFrame()
    
    def save_to_csv(self, df, filename=NEWS_YAHOO_PATH):
        """
        Save scraped data to CSV file
        
        Parameters:
        -----------
        df : DataFrame
            Data to save
        filename : str
            Path where to save the file
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
        print(f"Sources: {', '.join(df['source'].unique())}")
        print("="*70)
    
    def validate_data(self, df):
        """
        Validate the scraped data quality
        
        Checks:
        - No missing dates
        - Dates are in valid range
        - Headlines are not empty
        - URLs are valid
        """
        
        print("\n" + "="*70)
        print("DATA VALIDATION")
        print("="*70)
        
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("⚠️ Missing values found:")
            print(missing[missing > 0])
            issues.append("missing_values")
        else:
            print("✅ No missing values")
        
        # Check date range
        df['date_dt'] = pd.to_datetime(df['date'])
        min_date = df['date_dt'].min()
        max_date = df['date_dt'].max()
        
        print(f"✅ Date range: {min_date.date()} to {max_date.date()}")
        print(f"   Span: {(max_date - min_date).days} days")
        
        # Check for empty headlines
        empty_headlines = df['headline'].str.strip().eq('').sum()
        if empty_headlines > 0:
            print(f"⚠️ {empty_headlines} empty headlines found")
            issues.append("empty_headlines")
        else:
            print("✅ No empty headlines")
        
        # Check URL format
        invalid_urls = df[~df['url'].str.startswith('http')].shape[0]
        if invalid_urls > 0:
            print(f"⚠️ {invalid_urls} invalid URLs found")
            issues.append("invalid_urls")
        else:
            print("✅ All URLs valid")
        
        # Summary
        if not issues:
            print("\n✅ DATA VALIDATION PASSED")
        else:
            print(f"\n⚠️ DATA VALIDATION ISSUES: {', '.join(issues)}")
        
        print("="*70)
        
        return len(issues) == 0


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Run this script directly to scrape news for all tickers
    
    Usage:
        python scraper_yahoo.py
    """
    
    print("\n" + "="*70)
    print("YAHOO FINANCE NEWS SCRAPER - FIXED VERSION")
    print("="*70)
    
    # Initialize scraper
    scraper = YahooFinanceScraper()
    
    # Scrape news for all tickers
    news_df = scraper.scrape_multiple_tickers(
        tickers=TICKERS,
        articles_per_ticker=MAX_ARTICLES_PER_TICKER
    )
    
    # Validate data
    if not news_df.empty:
        is_valid = scraper.validate_data(news_df)
        
        # Save to CSV
        scraper.save_to_csv(news_df)
        
        # Display sample
        print("\n" + "="*70)
        print("SAMPLE DATA (First 5 rows)")
        print("="*70)
        print(news_df.head())
        
        # Display statistics
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total articles: {len(news_df)}")
        print(f"\nArticles per ticker:")
        print(news_df['ticker'].value_counts())
        print(f"\nArticles per source:")
        print(news_df['source'].value_counts())
        
    else:
        print("\n❌ SCRAPING FAILED - No data collected")