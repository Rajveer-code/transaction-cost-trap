"""
merge_features.py

Merge technical and trend feature sets on ['Date','Ticker'] (inner join),
resolve duplicate target columns, and save an ensemble features parquet.

Usage:
    python scripts/merge_features.py

Output:
    data/extended/features_ensemble.parquet
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TECH_FILE = ROOT / 'data' / 'extended' / 'features_5d_horizon.parquet'
TREND_FILE = ROOT / 'data' / 'extended' / 'features_trend_5d.parquet'
OUT_FILE = ROOT / 'data' / 'extended' / 'features_ensemble.parquet'


def main():
    if not TECH_FILE.exists():
        print('Technical features file not found:', TECH_FILE)
        return
    if not TREND_FILE.exists():
        print('Trend features file not found:', TREND_FILE)
        return

    print('Loading technical features:', TECH_FILE)
    tech = pd.read_parquet(TECH_FILE)
    print('Loading trend features:', TREND_FILE)
    trend = pd.read_parquet(TREND_FILE)

    # Normalize column names to strings
    tech.columns = [str(c) for c in tech.columns]
    trend.columns = [str(c) for c in trend.columns]

    # Ensure Date and Ticker columns exist and use same names
    def find_col(cols, key):
        key = key.lower()
        for c in cols:
            if c.lower() == key:
                return c
        for c in cols:
            if key in c.lower():
                return c
        return None

    date_col_tech = find_col(tech.columns, 'date')
    date_col_trend = find_col(trend.columns, 'date')
    ticker_col_tech = find_col(tech.columns, 'ticker')
    ticker_col_trend = find_col(trend.columns, 'ticker')

    if date_col_tech is None or ticker_col_tech is None:
        raise RuntimeError('Date or Ticker column not found in technical features')
    if date_col_trend is None or ticker_col_trend is None:
        raise RuntimeError('Date or Ticker column not found in trend features')

    # Standardize column names
    tech = tech.rename(columns={date_col_tech: 'Date', ticker_col_tech: 'Ticker'})
    trend = trend.rename(columns={date_col_trend: 'Date', ticker_col_trend: 'Ticker'})

    # Inner join on Date and Ticker; add suffixes to detect duplicates
    merged = pd.merge(tech, trend, on=['Date', 'Ticker'], how='inner', suffixes=('_tech', '_trend'))

    # Resolve duplicate target columns: keep _tech versions if present
    for col in ['binary_label', 'target_return_5d']:
        col_tech = f'{col}_tech'
        col_trend = f'{col}_trend'
        if col_tech in merged.columns and col_trend in merged.columns:
            # prefer technical label
            merged = merged.drop(columns=[col_trend])
            merged = merged.rename(columns={col_tech: col})
        elif col_tech in merged.columns:
            merged = merged.rename(columns={col_tech: col})
        elif col_trend in merged.columns:
            merged = merged.rename(columns={col_trend: col})

    # Drop any remaining suffixed duplicate columns where full duplicates exist
    # e.g., columns like 'Open_tech' and 'Open_trend' - prefer technical values
    cols = merged.columns.tolist()
    base_names = {}
    for c in cols:
        if c.endswith('_tech'):
            base = c[:-5]
            base_names.setdefault(base, []).append(c)
        if c.endswith('_trend'):
            base = c[:-6]
            base_names.setdefault(base, []).append(c)

    for base, variants in base_names.items():
        tech_variant = f'{base}_tech'
        trend_variant = f'{base}_trend'
        if tech_variant in merged.columns and trend_variant in merged.columns:
            # keep tech, drop trend
            merged = merged.drop(columns=[trend_variant])
            merged = merged.rename(columns={tech_variant: base})
        else:
            # rename any existing suffixed variant back to base
            if tech_variant in merged.columns:
                merged = merged.rename(columns={tech_variant: base})
            if trend_variant in merged.columns:
                merged = merged.rename(columns={trend_variant: base})

    # Final sanity: ensure neither target nor date/ticker missing
    if 'Date' not in merged.columns or 'Ticker' not in merged.columns:
        raise RuntimeError('Merged frame missing Date or Ticker')

    # Save
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_FILE, index=False)
    print('Saved ensemble features to', OUT_FILE)
    print('Final shape:', merged.shape)
    print('Columns sample (contains RSI and ratio_sma50?):')
    print([c for c in merged.columns if 'rsi' in c.lower() or 'ratio_sma' in c.lower()][:20])


if __name__ == '__main__':
    main()
