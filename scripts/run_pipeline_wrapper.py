import traceback
from scripts.train_models import StockPredictionPipeline

try:
    p = StockPredictionPipeline(output_dir="results")
    p.run_full_pipeline()
except Exception:
    print("\n\n--------- PIPELINE ERROR BELOW ---------")
    traceback.print_exc()
    print("----------------------------------------")
