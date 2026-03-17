"""
Configuration file for Financial Sentiment NLP Project
Stores all project settings in one centralized location
"""

from datetime import datetime, timedelta

# ============================================================================
# PROJECT METADATA
# ============================================================================
PROJECT_NAME = "Financial Sentiment Stock Prediction"
VERSION = "1.0.0"
AUTHOR = "Rajveer Singh Pall"

# ============================================================================
# STOCK TICKERS TO ANALYZE
# ============================================================================
# We're focusing on major tech stocks because:
# 1. High news coverage (lots of data)
# 2. High volatility (easier to predict movements)
# 3. Well-known companies (easier to validate results)
TICKERS = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'GOOGL',  # Google
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    'NVDA',   # Nvidia
    'META'    # Meta (Facebook)
]

# ============================================================================
# DATE RANGES FOR DATA COLLECTION
# ============================================================================
# We want 6 months of historical data
# Why 6 months? Balance between:
# - Enough data for training (not too little)
# - Recent enough to be relevant (not too old)
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

# ============================================================================
# WEB SCRAPING SETTINGS
# ============================================================================
# User-Agent: Makes our scraper look like a real browser
# Without this, websites might block us thinking we're a bot
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Request timeout: How long to wait for a response before giving up
REQUEST_TIMEOUT = 10  # seconds

# Delay between requests: Be polite to servers, don't spam them
SCRAPING_DELAY = 2  # seconds between requests

# Maximum articles per ticker
MAX_ARTICLES_PER_TICKER = 50

# ============================================================================
# NEWSAPI SETTINGS (if you have an API key)
# ============================================================================
# Get free API key from: https://newsapi.org/
# Free tier: 100 requests per day, 1 month of historical data
NEWSAPI_KEY = "e2346ae51d68490084fe54166ae36582"  # Replace with your actual key

# News sources to prioritize (more reliable)
NEWSAPI_SOURCES = [
    'bloomberg',
    'business-insider',
    'financial-post',
    'fortune',
    'the-wall-street-journal',
    'reuters'
]

# ============================================================================
# DATA PATHS
# ============================================================================
# Where to save collected data
DATA_DIR = 'data/'
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
FINAL_DATA_DIR = 'data/final/'

# Specific file paths
NEWS_YAHOO_PATH = 'data/raw/news_yahoo.csv'
NEWS_NEWSAPI_PATH = 'data/raw/news_newsapi.csv'
STOCK_DATA_PATH = 'data/raw/stock_with_ta.csv'

# ============================================================================
# MODEL SETTINGS
# ============================================================================
# Random seed for reproducibility
# Same seed = same results every time
RANDOM_SEED = 42

# Train/test split (but we'll use time-series split later)
TEST_SIZE = 0.2

# ============================================================================
# NLP SETTINGS
# ============================================================================
# FinBERT model from HuggingFace
FINBERT_MODEL = 'yiyanghkust/finbert-tone'

# Batch size for processing (process multiple headlines at once)
# Larger = faster but more memory
BATCH_SIZE = 32

# Maximum text length for models
MAX_TEXT_LENGTH = 512

# ============================================================================
# TECHNICAL INDICATORS TO CALCULATE
# ============================================================================
# These are financial metrics calculated from price data
TECHNICAL_INDICATORS = [
    'RSI',              # Relative Strength Index (momentum)
    'MACD',             # Moving Average Convergence Divergence (trend)
    'MACD_signal',      # MACD signal line
    'BB_upper',         # Bollinger Band upper (volatility)
    'BB_middle',        # Bollinger Band middle
    'BB_lower',         # Bollinger Band lower
    'ATR',              # Average True Range (volatility)
    'OBV',              # On-Balance Volume (volume indicator)
    'ADX',              # Average Directional Index (trend strength)
    'Stochastic_K',     # Stochastic oscillator
    'Stochastic_D',     # Stochastic signal
    'VWAP',             # Volume Weighted Average Price
    'CMF',              # Chaikin Money Flow
    'Williams_R',       # Williams %R (momentum)
    'EMA_12'            # Exponential Moving Average
]

# ============================================================================
# EVENT CLASSIFICATION CATEGORIES
# ============================================================================
# Types of financial news events
EVENT_TYPES = [
    'earnings',        # Quarterly earnings reports
    'product',         # New product launches
    'analyst',         # Analyst upgrades/downgrades
    'regulatory',      # Legal/regulatory news
    'macroeconomic',   # Fed rates, GDP, inflation
    'merger'           # M&A, acquisitions
]

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'logs/project.log'

# ============================================================================
# HELPER FUNCTION: Print configuration summary
# ============================================================================
def print_config():
    """Print current configuration settings"""
    print("="*70)
    print(f"PROJECT: {PROJECT_NAME} v{VERSION}")
    print("="*70)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Technical Indicators: {len(TECHNICAL_INDICATORS)} indicators")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*70)

if __name__ == "__main__":
    print_config()