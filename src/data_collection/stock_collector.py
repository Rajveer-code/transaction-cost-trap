"""
Stock Data Collector with Technical Indicators
Fetches historical stock prices and calculates 15+ technical indicators

WHAT ARE TECHNICAL INDICATORS?
- Mathematical calculations based on price/volume
- Used by traders to predict future price movements
- Examples: RSI (momentum), MACD (trend), Bollinger Bands (volatility)

WHY USE THEM?
- Complement sentiment analysis with price patterns
- Capture market momentum and trends
- Improve prediction accuracy by 10-15%
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('../..')
from config.config import *

# Try to import ta library for technical indicators
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: 'ta' library not installed")
    print("Install with: pip install ta")
    TA_AVAILABLE = False

class StockDataCollector:
    """
    Collects historical stock data and calculates technical indicators
    
    Data Sources:
    - Price data: Yahoo Finance (via yfinance)
    - Technical indicators: ta library
    """
    
    def __init__(self):
        """Initialize the stock data collector"""
        self.data = {}  # Store data for each ticker
        
    def fetch_stock_data(self, ticker, start_date=None, end_date=None):
        """
        Fetch historical stock price data
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str, optional
            Start date 'YYYY-MM-DD'
        end_date : str, optional
            End date 'YYYY-MM-DD'
            
        Returns:
        --------
        DataFrame with columns: [Date, Open, High, Low, Close, Volume]
        
        HOW IT WORKS:
        1. Uses yfinance library (free Yahoo Finance API)
        2. Downloads OHLCV data (Open, High, Low, Close, Volume)
        3. Returns as pandas DataFrame
        """
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"\n{'='*60}")
        print(f"Fetching stock data for {ticker}...")
        print(f"Date range: {start_date} to {end_date}")
        print(f"{'='*60}")
        
        try:
            # Create yfinance Ticker object
            stock = yf.Ticker(ticker)
            
            # Download historical data
            # period: not used when start/end specified
            # interval: '1d' = daily data
            df = stock.history(start=start_date, end=end_date, interval='1d')
            
            if df.empty:
                print(f"❌ No data found for {ticker}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to standard format
            df = df.rename(columns={'Date': 'date'})
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Keep only OHLCV columns
            df = df[['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Convert date to string format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            print(f"✅ Retrieved {len(df)} days of data")
            print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            print(f"   Avg volume: {df['Volume'].mean():,.0f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_basic_features(self, df):
        """
        Calculate basic price-based features
        
        Features calculated:
        - daily_return: Percentage change in closing price
        - next_day_return: Tomorrow's return (for target variable)
        - movement: Binary target (1=UP, 0=DOWN)
        
        Parameters:
        -----------
        df : DataFrame
            Stock data with OHLC columns
            
        Returns:
        --------
        DataFrame with additional feature columns
        
        EXPLANATION:
        - daily_return = (Close_today - Close_yesterday) / Close_yesterday * 100
        - If next_day_return > 0 → movement = 1 (UP)
        - If next_day_return <= 0 → movement = 0 (DOWN)
        """
        
        print("  Calculating basic features...")
        
        # Calculate daily return (percentage change)
        # pct_change() calculates: (value - previous_value) / previous_value
        df['daily_return'] = df['Close'].pct_change() * 100
        
        # Calculate next day's return (shift -1 means look ahead)
        df['next_day_return'] = df['daily_return'].shift(-1)
        
        # Create binary target variable
        # True converts to 1, False converts to 0
        df['movement'] = (df['next_day_return'] > 0).astype(int)
        
        # Calculate price volatility (rolling standard deviation)
        # window=5 means use last 5 days
        df['volatility'] = df['daily_return'].rolling(window=5).std()
        
        print(f"  ✅ Basic features calculated")
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate 15+ technical indicators using ta library
        
        Indicator Categories:
        1. TREND: SMA, EMA, MACD, ADX
        2. MOMENTUM: RSI, Stochastic, Williams %R
        3. VOLATILITY: Bollinger Bands, ATR
        4. VOLUME: OBV, VWAP, CMF
        
        Parameters:
        -----------
        df : DataFrame
            Stock data with OHLCV columns
            
        Returns:
        --------
        DataFrame with 15+ new indicator columns
        
        WHAT EACH INDICATOR MEANS:
        
        1. RSI (Relative Strength Index):
           - Measures momentum (0-100 scale)
           - >70 = overbought (might drop)
           - <30 = oversold (might rise)
        
        2. MACD (Moving Average Convergence Divergence):
           - Trend-following indicator
           - Positive = uptrend, Negative = downtrend
        
        3. Bollinger Bands:
           - Price volatility indicator
           - Price touching upper band = might reverse down
           - Price touching lower band = might reverse up
        
        4. ATR (Average True Range):
           - Measures volatility
           - High ATR = high volatility
        
        5. OBV (On-Balance Volume):
           - Volume-based momentum
           - Rising OBV = buying pressure
        
        ... and 10 more!
        """
        
        if not TA_AVAILABLE:
            print("  ⚠️ Skipping technical indicators (ta library not installed)")
            return df
        
        print("  Calculating technical indicators...")
        
        try:
            # Ensure we have proper numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # ========== TREND INDICATORS ==========
            
            # 1. Simple Moving Average (20-day)
            # Average of last 20 closing prices
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            
            # 2. Exponential Moving Average (12-day)
            # Recent prices weighted more than older prices
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            
            # 3. MACD (Moving Average Convergence Divergence)
            # Difference between fast and slow EMAs
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # 4. ADX (Average Directional Index)
            # Measures trend strength (0-100)
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            df['ADX'] = adx.adx()
            
            # ========== MOMENTUM INDICATORS ==========
            
            # 5. RSI (Relative Strength Index)
            # Momentum oscillator (0-100)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # 6. Stochastic Oscillator
            # Compares closing price to price range
            stoch = ta.momentum.StochasticOscillator(
                df['High'], df['Low'], df['Close']
            )
            df['Stochastic_K'] = stoch.stoch()
            df['Stochastic_D'] = stoch.stoch_signal()
            
            # 7. Williams %R
            # Momentum indicator similar to Stochastic
            df['Williams_R'] = ta.momentum.williams_r(
                df['High'], df['Low'], df['Close']
            )
            
            # ========== VOLATILITY INDICATORS ==========
            
            # 8-10. Bollinger Bands
            # Volatility bands around moving average
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # 11. ATR (Average True Range)
            # Measures market volatility
            df['ATR'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close']
            )
            
            # ========== VOLUME INDICATORS ==========
            
            # 12. OBV (On-Balance Volume)
            # Cumulative volume based on price direction
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # 13. CMF (Chaikin Money Flow)
            # Money flow volume indicator
            df['CMF'] = ta.volume.chaikin_money_flow(
                df['High'], df['Low'], df['Close'], df['Volume']
            )
            
            # 14. VWAP (Volume Weighted Average Price)
            # Average price weighted by volume
            df['VWAP'] = ta.volume.volume_weighted_average_price(
                df['High'], df['Low'], df['Close'], df['Volume']
            )
            
            print(f"  ✅ Calculated 15 technical indicators")
            
        except Exception as e:
            print(f"  ⚠️ Error calculating some indicators: {e}")
        
        return df
    
    def add_all_features(self, df):
        """
        Add both basic features and technical indicators
        
        This is the master function that adds all 15+ features
        """
        
        # Calculate basic features first
        df = self.calculate_basic_features(df)
        
        # Then calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Drop rows with NaN values
        # (first few rows will have NaN due to rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            print(f"  ℹ️ Dropped {dropped_rows} rows with NaN values")
        
        return df
    
    def fetch_multiple_tickers(self, tickers=None, start_date=None, end_date=None):
        """
        Fetch stock data with technical indicators for multiple tickers
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        start_date : str, optional
            Start date 'YYYY-MM-DD'
        end_date : str, optional
            End date 'YYYY-MM-DD'
            
        Returns:
        --------
        Combined DataFrame with all tickers and their technical indicators
        """
        
        if tickers is None:
            tickers = TICKERS
        
        if start_date is None:
            start_date = START_DATE
        if end_date is None:
            end_date = END_DATE
        
        print("\n" + "="*70)
        print(f"STOCK DATA COLLECTION WITH TECHNICAL INDICATORS")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Date range: {start_date} to {end_date}")
        print("="*70)
        
        all_data = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            # Fetch price data
            df = self.fetch_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                print(f"  ⚠️ Skipping {ticker} - no data")
                continue
            
            # Add all features
            df = self.add_all_features(df)
            
            if not df.empty:
                all_data.append(df)
                self.data[ticker] = df  # Store for later use
        
        # Combine all tickers
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            print("\n" + "="*70)
            print("DATA COLLECTION COMPLETE")
            print("="*70)
            print(f"Total rows: {len(combined_df)}")
            print(f"Tickers: {combined_df['ticker'].nunique()}")
            print(f"Features: {combined_df.shape[1]} columns")
            print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            
            # Show target variable distribution
            print(f"\nTarget variable distribution:")
            print(f"  UP days (movement=1): {(combined_df['movement']==1).sum()} ({(combined_df['movement']==1).mean()*100:.1f}%)")
            print(f"  DOWN days (movement=0): {(combined_df['movement']==0).sum()} ({(combined_df['movement']==0).mean()*100:.1f}%)")
            
            return combined_df
        else:
            print("\n❌ No data collected!")
            return pd.DataFrame()
    
    def save_to_csv(self, df, filename=STOCK_DATA_PATH):
        """
        Save stock data with technical indicators to CSV
        
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
        print(f"✅ Saved {len(df)} rows to {filename}")
        print(f"Columns: {', '.join(df.columns.tolist()[:10])}... (+{len(df.columns)-10} more)")
        print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    """
    Run this script to fetch stock data with technical indicators
    
    Usage:
        python stock_collector.py
    """
    
    print("\n" + "="*70)
    print("STOCK DATA COLLECTOR WITH TECHNICAL INDICATORS")
    print("="*70)
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Fetch data for all tickers
    stock_df = collector.fetch_multiple_tickers(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Save to CSV
    if not stock_df.empty:
        collector.save_to_csv(stock_df)
        
        # Display sample
        print("\n" + "="*70)
        print("SAMPLE DATA (First 3 rows)")
        print("="*70)
        print(stock_df.head(3))
        
        # Display feature list
        print("\n" + "="*70)
        print("FEATURES CREATED")
        print("="*70)
        features = [col for col in stock_df.columns 
                   if col not in ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
        for i, feature in enumerate(features, 1):
            print(f"{i:2d}. {feature}")
        
        print(f"\n✅ Total features: {len(features)}")
    else:
        print("\n❌ COLLECTION FAILED")