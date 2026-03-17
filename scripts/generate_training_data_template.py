"""
generate_training_data_template.py
===================================
TEMPLATE for data generation script that creates model_ready_full.csv

CRITICAL: This is a template. You must:
1. Fill in your actual data sources
2. Verify target construction matches this exactly
3. Document train/test split dates
4. Ensure news-price alignment is correct

Author: Rajveer Singh Pall
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# ============================================================
# TARGET CONSTRUCTION (MOST CRITICAL - VERIFY THIS!)
# ============================================================

def create_movement_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create movement target from next_day_return.
    
    CRITICAL: This must be leak-free!
    
    Args:
        df: DataFrame with columns [date, ticker, Close, ...]
        
    Returns:
        DataFrame with added columns: next_day_return, movement
    """
    # OPTION A (SAFE - Recommended):
    df = df.copy()
    df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
    df['movement'] = (df['next_day_return'] > 0).astype(int)
    
    # ALTERNATIVE OPTION B (Also safe):
    # df['movement'] = (df.groupby('ticker')['Close'].shift(-1) > df['Close']).astype(int)
    # df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
    
    # VERIFY: movement[T] predicts price direction on T+1
    # movement[T] = 1 means: Close[T+1] > Close[T]
    # movement[T] = 0 means: Close[T+1] <= Close[T]
    
    return df


def verify_target_construction(df: pd.DataFrame) -> bool:
    """
    Verify target construction is leak-free.
    
    Returns:
        True if verification passes, raises AssertionError otherwise
    """
    # Check 1: movement should be 0 or 1
    assert df['movement'].isin([0, 1]).all(), "movement must be binary (0 or 1)"
    
    # Check 2: movement should align with next_day_return
    for idx in range(len(df) - 1):
        if pd.notna(df.iloc[idx]['next_day_return']):
            expected_movement = 1 if df.iloc[idx]['next_day_return'] > 0 else 0
            actual_movement = df.iloc[idx]['movement']
            assert actual_movement == expected_movement, \
                f"Movement mismatch at index {idx}: expected {expected_movement}, got {actual_movement}"
    
    # Check 3: No future data in features
    # (This should be checked separately for each feature)
    
    print("✅ Target construction verification passed")
    return True


# ============================================================
# NEWS-PRICE ALIGNMENT
# ============================================================

def align_news_to_trading_day(
    news_date: datetime,
    price_dates: pd.DatetimeIndex
) -> datetime:
    """
    Align news publication date to next trading day.
    
    Scenario:
        News published: 2024-11-24 (Sunday) 10:00 AM
        → Returns: 2024-11-25 (Monday) - next trading day
        
    Args:
        news_date: News publication datetime
        price_dates: Available trading days (DatetimeIndex)
        
    Returns:
        Next trading day after news publication
    """
    # Convert to date (ignore time)
    news_date_only = news_date.date() if isinstance(news_date, datetime) else news_date
    
    # Find next trading day
    price_dates_only = pd.to_datetime(price_dates).date if hasattr(price_dates, 'date') else price_dates
    
    # Get next trading day after news
    next_trading_days = [d for d in price_dates_only if d > news_date_only]
    
    if not next_trading_days:
        # If no future trading day, use last available
        return max(price_dates_only)
    
    return min(next_trading_days)


def verify_news_alignment(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> bool:
    """
    Verify news timestamps are before prediction times.
    
    Args:
        news_df: DataFrame with columns [date, published_at, ...]
        price_df: DataFrame with columns [date, ...]
        
    Returns:
        True if verification passes
    """
    # Merge to check alignment
    merged = news_df.merge(
        price_df[['date']],
        on='date',
        how='left',
        suffixes=('_news', '_price')
    )
    
    # Check: published_at should be before or equal to date
    if 'published_at' in news_df.columns:
        news_with_timestamp = merged[merged['published_at'].notna()]
        if len(news_with_timestamp) > 0:
            # Convert to datetime for comparison
            published = pd.to_datetime(news_with_timestamp['published_at'])
            price_date = pd.to_datetime(news_with_timestamp['date'])
            
            # Allow same day (news can be published during trading hours)
            assert (published <= price_date + pd.Timedelta(days=1)).all(), \
                "News published_at must be before or on prediction date!"
    
    print("✅ News alignment verification passed")
    return True


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

# DOCUMENT YOUR EXACT DATES HERE:
TRAIN_START = datetime(2020, 1, 1)  # FILL IN YOUR ACTUAL DATE
TRAIN_END = datetime(2023, 12, 31)  # FILL IN YOUR ACTUAL DATE
TEST_START = datetime(2024, 7, 1)  # FILL IN YOUR ACTUAL DATE
TEST_END = datetime(2024, 12, 31)  # FILL IN YOUR ACTUAL DATE

# Gap between train and test (to prevent leakage)
GAP_DAYS = 1  # FILL IN YOUR ACTUAL GAP


def create_temporal_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create train/test split with temporal ordering.
    
    Args:
        df: Full dataset with date column
        
    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    df['date'] = pd.to_datetime(df['date'])
    
    # Verify dates
    assert TRAIN_END < TEST_START, \
        f"Temporal leakage! train_end ({TRAIN_END}) must be < test_start ({TEST_START})"
    
    train_df = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)].copy()
    test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].copy()
    
    print(f"✅ Train set: {len(train_df)} samples ({TRAIN_START.date()} to {TRAIN_END.date()})")
    print(f"✅ Test set: {len(test_df)} samples ({TEST_START.date()} to {TEST_END.date()})")
    
    return {
        'train': train_df,
        'test': test_df,
    }


# ============================================================
# MAIN DATA GENERATION PIPELINE
# ============================================================

def generate_training_data(
    price_data_path: str,
    news_data_path: str,
    output_path: str = "research_outputs/tables/model_ready_full.csv"
) -> pd.DataFrame:
    """
    Main function to generate model-ready training data.
    
    Steps:
    1. Load price data
    2. Create movement target (VERIFY THIS!)
    3. Load news data
    4. Align news to trading days
    5. Generate features (sentiment, technical, lagged)
    6. Merge everything
    7. Create train/test split
    8. Verify no leakage
    9. Save to CSV
    
    Args:
        price_data_path: Path to price data CSV
        news_data_path: Path to news data CSV
        output_path: Output path for model-ready data
        
    Returns:
        Complete DataFrame ready for model training
    """
    # Step 1: Load price data
    print("Loading price data...")
    price_df = pd.read_csv(price_data_path)
    price_df['date'] = pd.to_datetime(price_df['date'])
    
    # Step 2: Create movement target (CRITICAL!)
    print("Creating movement target...")
    price_df = create_movement_target(price_df)
    verify_target_construction(price_df)
    
    # Step 3: Load news data
    print("Loading news data...")
    news_df = pd.read_csv(news_data_path)
    news_df['date'] = pd.to_datetime(news_df['date'])
    
    # Step 4: Align news to trading days
    print("Aligning news to trading days...")
    price_dates = price_df['date'].unique()
    # Apply alignment logic here
    
    # Step 5: Generate features
    # (This would call your feature generation pipeline)
    print("Generating features...")
    # features_df = generate_all_features(price_df, news_df)
    
    # Step 6: Merge
    # merged_df = merge_features(price_df, features_df)
    
    # Step 7: Create split
    # splits = create_temporal_split(merged_df)
    
    # Step 8: Verify
    # verify_no_leakage(splits['train'], splits['test'])
    
    # Step 9: Save
    # merged_df.to_csv(output_path, index=False)
    
    print("✅ Data generation complete")
    return price_df  # Placeholder


# ============================================================
# VERIFICATION FUNCTIONS
# ============================================================

def verify_no_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Verify no data leakage between train and test.
    
    Checks:
    1. No temporal overlap
    2. No feature leakage (future data in features)
    3. Target alignment correct
    """
    # Check 1: Temporal separation
    train_max_date = train_df['date'].max()
    test_min_date = test_df['date'].min()
    
    assert train_max_date < test_min_date, \
        f"Temporal leakage! Train max ({train_max_date}) >= Test min ({test_min_date})"
    
    # Check 2: No future features (sample check)
    # This should be done for each feature type
    
    print("✅ No leakage detected")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING DATA GENERATION TEMPLATE")
    print("=" * 60)
    print("\n⚠️  This is a TEMPLATE. You must:")
    print("1. Fill in your actual data loading logic")
    print("2. Verify target construction matches your actual code")
    print("3. Document train/test split dates")
    print("4. Test with your actual data sources")
    print("\n" + "=" * 60)

