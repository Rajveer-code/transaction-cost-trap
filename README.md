# The Transaction Cost Trap

**Why Machine Learning Stock Prediction Fails Economically Under Realistic Market Frictions**

Rajveer Singh Pall  
Department of Computer Science and Business Systems  
Gyan Ganga Institute of Technology and Sciences, Jabalpur, India  
📧 rajveerpall05@gmail.com

> Under review at *Quantitative Finance and Economics (QFE)*

---

## Key Finding

A regime-filtered CatBoost ensemble achieves up to **73.3% conditional directional accuracy** on 7 large-cap tech stocks (2015–2025), yet generates **-42.49% annually** (Sharpe -2.83) after realistic 5 bps execution costs, while passive buy-and-hold earns **+34.77%** (Sharpe 1.21).

At zero cost the strategy earns **+4.61% annually** (Sharpe 0.31), confirming the gap arises entirely from transaction cost drag — the **transaction cost trap**.

---

## Replication

### Requirements
`\ash
pip install -r requirements.txt
`\

### Step 1 — Run the full modeling pipeline
`\ash
python run_pipeline.py --data data/combined/all_features.parquet --output outputs/ --verbose
`\

### Step 2 — Reproduce transaction cost sensitivity table
`\ash
python scripts/transaction_cost_sensitivity.py
`\

This reproduces Table 4 from the paper exactly.

---

## Repository Structure

`\
├── src/                        # Core pipeline modules
│   ├── data_collection/        # News and price data scrapers
│   ├── feature_engineering/    # 47 technical features
│   ├── modeling/               # CatBoost, RF, DNN trainers
│   ├── evaluation/             # Walk-forward CV, backtesting, SHAP
│   └── utils/                  # Helpers
├── scripts/                    # Analysis and figure scripts
├── data/
│   └── combined/               # all_features.parquet (17,773 obs)
├── models/                     # Trained model files
├── results/                    # Output CSVs and figures
├── notebooks/                  # Phase notebooks
├── config/                     # Configuration (no API keys)
├── run_pipeline.py             # Main entry point
└── requirements.txt
`\

---

## Dataset

- **7 tickers:** AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- **Period:** January 2015 – April 2025
- **Observations:** 17,773 stock-day records
- **Features:** 47 technical indicators
- **Validation:** Expanding walk-forward (10 folds, ~2,160 OOS observations)

---

## License

MIT License — see LICENSE file.
