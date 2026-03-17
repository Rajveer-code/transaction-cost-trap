"""
Wrapper script to run the full pipeline with proper argument handling.
"""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run stock movement prediction pipeline')
    parser.add_argument('--data', default='data/combined/all_features.parquet',
                       help='Path to combined dataset')
    parser.add_argument('--output', default='research_outputs/results',
                       help='Output directory for results')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of CPU cores to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with detailed logging')
    
    args = parser.parse_args()
    
    # Import after arg parsing so we can control logging
    from scripts.train_models import StockPredictionPipeline
    import logging
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.output) / 'pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline with args: {args}")
    
    try:
        # Create and run pipeline
        pipeline = StockPredictionPipeline(
            data_dir=str(Path(args.data).parent),
            output_dir=args.output,
            figures_dir=str(Path(args.output) / '../figures')
        )
        
        # Temporarily update n_jobs if different
        # This would require modifying the pipeline class to accept this parameter
        
        pipeline.run_full_pipeline()
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
