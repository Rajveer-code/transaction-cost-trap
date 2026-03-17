# 📈 Advanced NLP for Financial Sentiment Analysis

> A research-grade machine learning system for predicting stock movements using multi-model sentiment fusion, event classification, and entity extraction.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## 🎯 Project Overview

This project implements an **end-to-end NLP pipeline** for financial sentiment analysis that combines:
- **Multi-model sentiment fusion** (FinBERT + VADER + TextBlob)
- **Event-aware classification** (6 financial event types)
- **Entity extraction** (CEO names, products, competitors)
- **Technical indicators** (15+ market signals)
- **Gradient boosting models** with hyperparameter tuning
- **SHAP explainability** for model interpretability
- **Portfolio backtesting** with realistic trading simulation

### Key Innovation

The system introduces **novel sentiment disagreement metrics** that capture market uncertainty by measuring variance across different sentiment models. This, combined with event-specific sentiment analysis and entity context, achieves superior predictive performance compared to single-model approaches.

---

## ✨ Key Features

### 🔬 Research-Grade Implementation
- Walk-forward time-series cross-validation (no data leakage)
- Statistical significance testing (T-test, McNemar, Bootstrap, Permutation)
- Temporal sentiment decay analysis (half-life calculation)
- SHAP-based feature importance and interaction analysis

### 🤖 Advanced NLP
- **FinBERT**: Domain-specific transformer (finance-tuned)
- **VADER**: Lexicon-based sentiment (fast, interpretable)
- **TextBlob**: Rule-based baseline
- **Zero-shot classification**: Event categorization without training
- **SpaCy NER**: Entity extraction with financial context

### 📊 Comprehensive Evaluation
- **Financial metrics**: Sharpe, Sortino, Calmar ratios
- **ML metrics**: Accuracy, F1, ROC-AUC
- **Backtesting**: Transaction costs, slippage, win rate
- **Explainability**: Global + local SHAP analysis

### 🎨 Interactive Dashboard
- **Streamlit web app**: Real-time predictions
- **Multi-tab interface**: Dashboard, Sentiment Analysis, Predictions, Insights
- **Live visualization**: Candlestick charts, sentiment timelines
- **Custom headline analysis**: On-demand sentiment scoring

---

## 📂 Project Structure

```
financial-sentiment-nlp/
│
├── app/
│   └── streamlit_app.py           # Interactive web application
│
├── config/
│   ├── config.py                  # Configuration settings
│   └── tickers.json               # Stock ticker list
│
├── data/
│   ├── raw/                       # Scraped news + stock data
│   ├── processed/                 # Sentiment + events + entities
│   └── final/                     # 42-feature model-ready dataset
│
├── models/
│   ├── best_model.pkl             # CatBoost (best performer)
│   ├── catboost_best.pkl          # Tuned CatBoost
│   ├── xgb_best.pkl               # Tuned XGBoost
│   ├── lgbm_best.pkl              # Tuned LightGBM
│   ├── scaler_ensemble.pkl        # Feature scaler
│   └── model_comparison.json      # Performance comparison
│
├── notebooks/
│   ├── 01_phase1_data_foundation.ipynb
│   ├── 02_phase2_nlp_innovation.ipynb
│   ├── 03_phase3_advanced_features.ipynb
│   ├── 04_phase4_model_training.ipynb
│   ├── 05_phase5_shap_explainability.ipynb
│   └── 06_phase5_portfolio_backtest.ipynb
│
├── results/
│   ├── figures/                   # All visualizations
│   └── metrics/                   # Performance CSVs
│
├── src/
│   ├── data_collection/
│   ├── feature_engineering/
│   ├── modeling/
│   ├── evaluation/
│   └── utils/
│
├── Documentation/
│   ├── README.md                  # This file
│   ├── METHODOLOGY.md             # Detailed methods
│   └── RESULTS.md                 # Results summary
│
├── requirements.txt
└── LICENSE
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/financial-sentiment-nlp.git
cd financial-sentiment-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_lg
```

### Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

### Execute Full Pipeline

```bash
# Phase 1: Data Collection
python src/data_collection/scraper_yahoo.py
python src/data_collection/stock_collector.py

# Phase 2: NLP Processing
python src/feature_engineering/nlp_analyzer.py
python src/data_collection/sentiment_fusion.py

# Phase 3: Feature Engineering
python src/feature_engineering/event_classifier.py
python src/feature_engineering/entity_extractor.py
python src/feature_engineering/feature_pipeline.py

# Phase 4: Model Training
python src/evaluation/time_series_cv.py
python src/modeling/train_baseline.py
python src/modeling/train_ensemble.py

# Phase 5: Evaluation
python src/evaluation/shap_explainer.py
python src/evaluation/backtesting_engine.py
python src/evaluation/significance_tests.py
python src/evaluation/temporal_analysis.py
```

---

## 📊 Results Summary

### Model Performance

| Model | F1-Score (CV) | Accuracy | ROC-AUC |
|-------|---------------|----------|---------|
| **CatBoost** | **0.608** | 0.642 | 0.721 |
| XGBoost | 0.576 | 0.619 | 0.698 |
| LightGBM | 0.544 | 0.591 | 0.672 |
| Random Forest | 0.501 | 0.520 | 0.632 |
| Stacking Ensemble | 0.463 | 0.498 | 0.615 |
| Logistic Regression | 0.364 | 0.507 | 0.568 |

### Financial Performance

| Metric | ML Strategy | Buy & Hold |
|--------|-------------|------------|
| Total Return | 18.7% | 12.5% |
| Sharpe Ratio | 1.24 | 0.78 |
| Max Drawdown | -9.8% | -15.2% |
| Win Rate | 58% | N/A |
| Calmar Ratio | 1.91 | 0.82 |

### Statistical Validation

- ✅ **Accuracy vs Random** (50%): t = 8.45, p < 0.001
- ✅ **McNemar's Test** (vs Baseline): χ² = 12.3, p = 0.004
- ✅ **Permutation Test**: p < 0.001
- ✅ **Bootstrap 95% CI**: [0.575, 0.641] for F1-score

### Top 10 Features (SHAP)

1. sentiment_variance_mean (model disagreement)
2. finbert_sentiment_score_mean
3. CMF (Chaikin Money Flow)
4. ceo_sentiment
5. ensemble_sentiment_min
6. volatility_lag1
7. headline_length_avg
8. entity_density
9. MACD
10. daily_return_lag1

---

## 🔬 Methodology Highlights

### Data Collection
- **News Sources**: Yahoo Finance RSS + NewsAPI
- **Historical Date Extraction**: Fixed scraping with accurate timestamps
- **Technical Indicators**: 15+ TAs via `ta` library
- **Coverage**: 602 observations, 45 features, 5 tickers

### Sentiment Analysis
- **Multi-Model Ensemble**: Weighted combination (FinBERT 60%, VADER 30%, TextBlob 10%)
- **Novel Disagreement Metrics**: Variance and consensus across models
- **Batch Optimization**: 7.5x speedup with PyTorch batching

### Feature Engineering
- **Event Classification**: Zero-shot (Earnings, Product, Analyst, Regulatory, Macro, M&A)
- **Entity Extraction**: Hybrid dictionary + SpaCy NER
- **Daily Aggregation**: Multiple headlines → single day features
- **Lagged Features**: T-1, T-3, T-5 temporal patterns

### Modeling
- **Time-Series CV**: Walk-forward expanding window (5 folds)
- **Hyperparameter Tuning**: RandomizedSearchCV (50 iterations)
- **Class Weighting**: Balanced for slight imbalance
- **Best Model**: CatBoost with optimized parameters

### Evaluation
- **SHAP Analysis**: Global importance + local explanations + interactions
- **Backtesting**: Realistic simulation with costs (0.2%) and slippage (0.05%)
- **Temporal Decay**: Sentiment half-life = 1.8 days
- **Significance Tests**: T-test, McNemar, Bootstrap, Permutation

---

## 📚 Dependencies

### Core Libraries
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
```

### NLP & Transformers
```
transformers==4.30.2
torch==2.0.1
vaderSentiment==3.3.2
textblob==0.17.1
spacy==3.5.3
```

### Gradient Boosting
```
xgboost==1.7.6
lightgbm==4.0.0
catboost==1.2
```

### Explainability & Visualization
```
shap==0.42.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
```

### Web Framework
```
streamlit==1.25.0
```

See `requirements.txt` for complete list.

---

## 🎓 Academic Applications

This project demonstrates skills for **Master's program applications** in:
- **Data Science**: End-to-end ML pipeline
- **NLP**: Transformer models, sentiment analysis, entity extraction
- **Financial Engineering**: Technical indicators, backtesting, risk metrics
- **Research Methods**: Statistical testing, experimental design, reproducibility

### Suitable For:
- Portfolio projects for graduate school applications
- Research papers in computational finance
- Industry ML engineer roles
- Capstone projects in data science programs

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Anthropic Claude** for project guidance
- **Yahoo Finance** & **NewsAPI** for data
- **CatBoost team** for excellent documentation

---

## 📞 Contact & Support

For questions or issues:
- Open an [Issue](https://github.com/yourusername/financial-sentiment-nlp/issues)
- Email: your.email@example.com
- LinkedIn: [Connect](https://linkedin.com/in/yourprofile)

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ for advancing NLP in finance**