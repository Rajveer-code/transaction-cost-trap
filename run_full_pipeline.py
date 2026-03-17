#!/usr/bin/env python
"""Run the complete modeling pipeline with proper initialization."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from scripts.train_models import StockPredictionPipeline

def main():
    print("=" * 80)
    print("STOCK MOVEMENT PREDICTION - COMPLETE MODELING PIPELINE")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Initialize and load data
        print("\n[INIT] Initializing pipeline...")
        pipeline = StockPredictionPipeline()
        pipeline.load_data()
        print(f"  Data loaded: {pipeline.df.shape}")
        
        # Run full pipeline
        print("\n[EXEC] Running complete pipeline (6 analyses)...")
        pipeline.run_full_pipeline()
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"SUCCESS: Pipeline completed in {total_time:.1f}s ({total_time/60:.1f}m)")
        print("=" * 80)
        
        # List output files
        print("\nOutput files:")
        output_dir = Path('research_outputs/results')
        for f in sorted(output_dir.glob('*.csv')) + sorted(output_dir.glob('*.json')):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name:50s} {size_kb:7.1f} KB")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
