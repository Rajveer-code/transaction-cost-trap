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

```bash
pip install -r requirements.txt
```

### Step 1 — Run the full modeling pipeline

```bash
python run_pipeline.py
```

This will:
1. Load the combined dataset (`data/combined/all_features.parquet`)
2. Run baseline models (random, logistic regression, technical-only)
3. Run per-ticker CatBoost with walk-forward validation
4. Evaluate cross-ticker generalization
5. Run ablation studies (full vs. technical-only vs. rolling-returns)
6. Run backtest simulation

Results are saved to `outputs/`.

For verbose logging:

```bash
python run_pipeline.py --verbose
```

### Step 2 — Reproduce transaction cost sensitivity table

```bash
python scripts/transaction_cost_sensitivity.py
```

This reproduces Table 4 from the paper exactly.

---

## Repository Structure

```
├── run_pipeline.py                  # Main entry point
├── requirements.txt                 # Python dependencies
├── FEATURE_SCHEMA.py                # 42-feature schema documentation
│
├── src/
│   ├── models/
│   │   └── catboost_trainer.py      # CatBoost, walk-forward, baselines
│   └── evaluation/
│       └── metrics.py               # Metrics, bootstrap CI, significance tests
│
├── scripts/
│   ├── train_models.py              # StockPredictionPipeline class
│   └── transaction_cost_sensitivity.py  # Table 4 reproduction
│
├── config/
│   ├── config.py                    # Project configuration
│   └── tickers.json                 # Ticker metadata
│
├── data/
│   ├── combined/                    # all_features.parquet (17,773 obs)
│   └── extended/                    # Extended 10-year dataset
│
├── results/                         # Pre-computed predictions
├── outputs/                         # Pipeline outputs (generated)
├── notebooks/                       # Exploration notebooks (phases 3–6)
└── paper/                           # Manuscript
```

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
