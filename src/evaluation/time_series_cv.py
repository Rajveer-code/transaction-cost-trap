"""
Time-Series Cross-Validation Module - Phase 4 (Day 22-23)
==========================================================
Walk-forward validation to prevent data leakage in time-series modeling.

CRITICAL: Never shuffle time-series data. Always respect temporal order.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator
from sklearn.model_selection import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time-series cross-validation with walk-forward approach.
    
    Strategy:
    - Expanding window: Training set grows with each fold
    - Fixed test size: Test set remains constant
    - No shuffling: Respects temporal ordering
    - No data leakage: Future data never in training
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 30, gap: int = 0):
        """
        Initialize time-series splitter.
        
        Args:
            n_splits: Number of CV folds
            test_size: Number of samples in each test set
            gap: Number of samples to skip between train and test (optional)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            groups: Group labels (optional, unused)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate minimum training size needed
        min_train_size = n_samples - (self.n_splits * self.test_size) - (self.n_splits * self.gap)
        
        if min_train_size < 50:
            raise ValueError(
                f"Dataset too small for {self.n_splits} folds with test_size={self.test_size}. "
                f"Need at least {50 + self.n_splits * (self.test_size + self.gap)} samples."
            )
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test set: next test_size samples
            test_start = min_train_size + (i * (self.test_size + self.gap))
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
            
            # Training set: all data up to gap before test
            train_end = test_start - self.gap
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def create_time_series_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 30,
    date_col: str = 'date'
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Create time-series splits ensuring data is sorted by date.
    
    Args:
        df: DataFrame with date column
        n_splits: Number of CV folds
        test_size: Size of each test set
        date_col: Name of date column
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Ensure data is sorted by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    
    # Create splitter
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    # Generate splits
    splits = []
    for train_idx, test_idx in tscv.split(df_sorted):
        splits.append((df_sorted.index[train_idx], df_sorted.index[test_idx]))
    
    return splits


def visualize_splits(splits: List[Tuple[np.ndarray, np.ndarray]], n_samples: int):
    """
    Visualize train/test splits to verify no data leakage.
    
    Args:
        splits: List of (train_indices, test_indices)
        n_samples: Total number of samples
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # Plot train indices
        ax.scatter(train_idx, [i] * len(train_idx), c='blue', marker='|', s=100, label='Train' if i == 0 else '')
        # Plot test indices
        ax.scatter(test_idx, [i] * len(test_idx), c='red', marker='|', s=100, label='Test' if i == 0 else '')
    
    ax.set_xlabel('Sample Index (Time →)', fontsize=12)
    ax.set_ylabel('Fold Number', fontsize=12)
    ax.set_title('Time-Series Cross-Validation Splits', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(len(splits))])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_split_summary(splits: List[Tuple[np.ndarray, np.ndarray]], dates: pd.Series = None):
    """
    Print summary of cross-validation splits.
    
    Args:
        splits: List of (train_indices, test_indices)
        dates: Series of dates (optional)
    """
    print(f"\n{'='*60}")
    print("TIME-SERIES CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"Total Folds: {len(splits)}")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {i+1}:")
        print(f"  Train size: {len(train_idx)}")
        print(f"  Test size:  {len(test_idx)}")
        
        if dates is not None:
            train_dates = dates.iloc[train_idx]
            test_dates = dates.iloc[test_idx]
            print(f"  Train period: {train_dates.min()} to {train_dates.max()}")
            print(f"  Test period:  {test_dates.min()} to {test_dates.max()}")
        
        # Verify no overlap
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} overlapping samples!")
        else:
            print(f"  ✅ No data leakage")


def main():
    """
    Test the time-series cross-validation on actual data.
    """
    import os
    
    print("\n" + "="*60)
    print("PHASE 4 - DAY 22-23: TIME-SERIES CROSS-VALIDATION")
    print("="*60 + "\n")
    
    # Load final dataset
    data_file = 'data/final/model_ready_full.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print("   Please complete Phase 3 first.")
        return
    
    print(f"📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Loaded {len(df)} observations")
    
    # Sort by date
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Determine optimal split parameters based on dataset size
    n_samples = len(df_sorted)
    
    # Calculate test_size (approximately 5% of data or minimum 20)
    test_size = max(20, int(n_samples * 0.05))
    
    # Calculate n_splits (ensure we have enough data)
    max_possible_splits = (n_samples - 100) // test_size  # Leave 100 for initial training
    n_splits = min(5, max_possible_splits)
    
    print(f"\n📊 Cross-Validation Configuration:")
    print(f"   Total samples: {n_samples}")
    print(f"   Test size per fold: {test_size}")
    print(f"   Number of folds: {n_splits}")
    print(f"   Strategy: Walk-forward expanding window")
    
    # Create splits
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        splits = list(tscv.split(df_sorted))
        
        # Print summary
        print_split_summary(splits, df_sorted['date'])
        
        # Visualize splits
        print("\n📈 Generating visualization...")
        fig = visualize_splits(splits, n_samples)
        
        # Save visualization
        os.makedirs('results/figures', exist_ok=True)
        fig.savefig('results/figures/time_series_cv_splits.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: results/figures/time_series_cv_splits.png")
        
        print("\n" + "="*60)
        print("✅ TIME-SERIES CV SETUP COMPLETE!")
        print("="*60)
        print(f"\nConfiguration saved:")
        print(f"  • n_splits: {n_splits}")
        print(f"  • test_size: {test_size}")
        print(f"  • No data leakage verified ✅")
        
        # Save configuration for use in training scripts
        cv_config = {
            'n_splits': n_splits,
            'test_size': test_size,
            'strategy': 'walk_forward_expanding'
        }
        
        import json
        with open('models/cv_config.json', 'w') as f:
            json.dump(cv_config, f, indent=2)
        
        print(f"\n💾 CV configuration saved to: models/cv_config.json")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nSuggestion: Reduce n_splits or test_size")
    
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()