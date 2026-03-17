# src/reporting/reporting.py

"""
reporting.py
============
PDF reporting utilities for the Financial Sentiment Analysis app.

Generates an IEEE-style, professional PDF summary containing:
- Ticker and prediction overview
- Sentiment feature summary (FinBERT, VADER, TextBlob, ensemble)
- Event and CEO sentiment breakdown
- Optional backtest performance summary
- News snapshot (top headlines)

Author: Rajveer Singh Pall
"""

from io import BytesIO
from typing import Dict, List, Optional, Any
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors


def _draw_header(c: canvas.Canvas, title: str, author: str, date_str: str):
    """Draws the title block in IEEE-style top section."""
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2.0, height - 1.0 * inch, title)

    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2.0, height - 1.3 * inch, author)
    c.drawCentredString(width / 2.0, height - 1.6 * inch, date_str)

    # Horizontal line
    c.setLineWidth(0.5)
    c.line(1.0 * inch, height - 1.8 * inch, width - 1.0 * inch, height - 1.8 * inch)


def _draw_section_title(c: canvas.Canvas, text: str, y: float) -> float:
    """Draws a section title like 'I. INTRODUCTION' and returns new y."""
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.0 * inch, y, text)
    return y - 0.25 * inch


def _draw_paragraph(c: canvas.Canvas, text: str, y: float, max_width: float) -> float:
    """
    Draw simple wrapped paragraph.
    Very lightweight word-wrap for IEEE-style text blocks.
    """
    c.setFont("Helvetica", 10)
    words = text.split()
    line = ""
    line_height = 0.18 * inch
    x = 1.0 * inch

    for w in words:
        test_line = f"{line} {w}".strip()
        if c.stringWidth(test_line, "Helvetica", 10) < max_width:
            line = test_line
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w

        # Page break safety
        if y < 0.75 * inch:
            c.showPage()
            y = A4[1] - 1.0 * inch
            c.setFont("Helvetica", 10)

    if line:
        c.drawString(x, y, line)
        y -= line_height

    return y - 0.1 * inch


def _format_percent(x: float) -> str:
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"


def generate_pdf_report(
    ticker: str,
    prediction: Any,
    sentiment_features: Dict[str, float],
    articles: List[Dict],
    backtest_results: Optional[Dict[str, Any]] = None,
    author: str = "Rajveer Singh Pall",
) -> bytes:
    """
    Generate an IEEE-style PDF report.

    Args:
        ticker: e.g. 'AAPL'
        prediction: PredictionResult instance (from models_backtest)
        sentiment_features: dict from nlp_pipeline.generate_sentiment_features()
        articles: list of article dicts from news_api.get_news_cached()
        backtest_results: optional dict with keys like:
            {
                'ml_metrics': BacktestMetrics,
                'bh_metrics': BacktestMetrics
            }
        author: author name for report header

    Returns:
        PDF bytes, suitable for sending to a Streamlit download_button.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ---------- HEADER ----------
    today_str = datetime.now().strftime("%d %b %Y")
    title = f"Financial Sentiment & Prediction Report: {ticker}"
    _draw_header(c, title, author, today_str)

    y = height - 2.1 * inch
    max_text_width = width - 2.0 * inch

    # ---------- ABSTRACT ----------
    y = _draw_section_title(c, "Abstract", y)
    abstract_text = (
        f"This report summarizes the news-driven sentiment, model prediction, and optional "
        f"backtest performance for the equity {ticker}. A multi-model NLP pipeline "
        f"(FinBERT, VADER, TextBlob) was combined with event classification and entity "
        f"sentiment to generate features for a machine learning model (CatBoost), which "
        f"produced the trading signal reported below."
    )
    y = _draw_paragraph(c, abstract_text, y, max_text_width)

    # ---------- 1. PREDICTION SUMMARY ----------
    y = _draw_section_title(c, "I. Prediction Summary", y)

    signal = getattr(prediction, "signal", "N/A")
    prob_up = getattr(prediction, "probability", 0.0)
    confidence = getattr(prediction, "confidence", 0.0)
    direction = "UP" if getattr(prediction, "prediction", 1) == 1 else "DOWN"
    date_str = getattr(prediction, "date", datetime.now()).strftime("%Y-%m-%d")

    summary_text = (
        f"Ticker: {ticker}\n"
        f"Prediction Date: {date_str}\n"
        f"Model Signal: {signal} (direction: {direction})\n"
        f"Probability of UP class: {_format_percent(prob_up)}\n"
        f"Model Confidence: {_format_percent(confidence)}"
    )

    for line in summary_text.split("\n"):
        c.setFont("Helvetica", 10)
        c.drawString(1.0 * inch, y, line)
        y -= 0.18 * inch

    y -= 0.15 * inch

    # ---------- 2. SENTIMENT OVERVIEW ----------
    y = _draw_section_title(c, "II. Sentiment Overview", y)

    finbert_mean = sentiment_features.get("finbert_sentiment_mean", 0.0)
    vader_mean = sentiment_features.get("vader_sentiment_mean", 0.0)
    tb_mean = sentiment_features.get("textblob_sentiment_mean", 0.0)
    ensemble_mean = sentiment_features.get("ensemble_sentiment_mean", 0.0)
    sentiment_var = sentiment_features.get("sentiment_variance", 0.0)
    headline_count = sentiment_features.get("headline_count", 0)
    avg_len = sentiment_features.get("avg_headline_length", 0.0)

    sentiment_text = (
        f"Number of headlines analyzed: {headline_count}\n"
        f"Average headline length: {avg_len:.1f} characters\n"
        f"FinBERT sentiment mean: {finbert_mean:.3f}\n"
        f"VADER sentiment mean: {vader_mean:.3f}\n"
        f"TextBlob sentiment mean: {tb_mean:.3f}\n"
        f"Ensemble sentiment mean: {ensemble_mean:.3f}\n"
        f"Sentiment variance across models: {sentiment_var:.4f}"
    )

    for line in sentiment_text.split("\n"):
        c.setFont("Helvetica", 10)
        c.drawString(1.0 * inch, y, line)
        y -= 0.18 * inch

    y -= 0.15 * inch

    # ---------- 3. EVENT & CEO SENTIMENT ----------
    y = _draw_section_title(c, "III. Event and CEO Sentiment", y)

    ceo_score = sentiment_features.get("ceo_sentiment_score", 0.0)
    entity_score = sentiment_features.get("entity_sentiment_score", 0.0)

    event_earn = sentiment_features.get("event_earnings", 0)
    event_prod = sentiment_features.get("event_product", 0)
    event_ma = sentiment_features.get("event_ma", 0)
    event_reg = sentiment_features.get("event_regulatory", 0)
    event_macro = sentiment_features.get("event_macro", 0)
    event_other = sentiment_features.get("event_other", 0)

    ceo_text = (
        f"CEO-specific sentiment score: {ceo_score:.3f}\n"
        f"Entity-level (company) sentiment score: {entity_score:.3f}\n"
    )

    for line in ceo_text.split("\n"):
        if line.strip():
            c.setFont("Helvetica", 10)
            c.drawString(1.0 * inch, y, line)
            y -= 0.18 * inch

    y -= 0.1 * inch
    c.setFont("Helvetica-Bold", 10)
    c.drawString(1.0 * inch, y, "Event-type counts from news:")
    y -= 0.2 * inch

    c.setFont("Helvetica", 10)
    event_lines = [
        f"Earnings-related: {event_earn}",
        f"Product launch: {event_prod}",
        f"Mergers & acquisitions: {event_ma}",
        f"Regulatory / legal: {event_reg}",
        f"Macroeconomic: {event_macro}",
        f"Other / uncategorized: {event_other}",
    ]
    for line in event_lines:
        c.drawString(1.2 * inch, y, f"- {line}")
        y -= 0.18 * inch

    y -= 0.15 * inch

    # ---------- 4. BACKTEST SUMMARY (OPTIONAL) ----------
    if backtest_results is not None:
        y = _draw_section_title(c, "IV. Backtest Summary", y)

        ml_metrics = backtest_results.get("ml_metrics")
        bh_metrics = backtest_results.get("bh_metrics")

        if ml_metrics:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(1.0 * inch, y, "ML Strategy Performance:")
            y -= 0.2 * inch

            ml_dict = ml_metrics.to_dict() if hasattr(ml_metrics, "to_dict") else ml_metrics
            c.setFont("Helvetica", 9)
            for k, v in ml_dict.items():
                c.drawString(1.2 * inch, y, f"- {k}: {v}")
                y -= 0.16 * inch
                if y < 0.75 * inch:
                    c.showPage()
                    y = A4[1] - 1.0 * inch

            y -= 0.1 * inch

        if bh_metrics:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(1.0 * inch, y, "Buy & Hold Benchmark:")
            y -= 0.2 * inch

            bh_dict = bh_metrics.to_dict() if hasattr(bh_metrics, "to_dict") else bh_metrics
            c.setFont("Helvetica", 9)
            for k, v in bh_dict.items():
                c.drawString(1.2 * inch, y, f"- {k}: {v}")
                y -= 0.16 * inch
                if y < 0.75 * inch:
                    c.showPage()
                    y = A4[1] - 1.0 * inch

            y -= 0.1 * inch

    # ---------- 5. NEWS SNAPSHOT ----------
    # New page if too low
    if y < 1.5 * inch:
        c.showPage()
        y = A4[1] - 1.0 * inch

    y = _draw_section_title(c, "V. News Snapshot", y)

    c.setFont("Helvetica", 10)
    if not articles:
        c.drawString(1.0 * inch, y, "No news articles were available for this period.")
        y -= 0.2 * inch
    else:
        c.drawString(1.0 * inch, y, "Top headlines:")
        y -= 0.25 * inch

        max_articles = min(5, len(articles))
        for i in range(max_articles):
            art = articles[i]
            title = art.get("title", "No title")
            source = art.get("source", "Unknown source")
            published = art.get("published_at", "N/A")

            c.setFont("Helvetica-Bold", 9)
            c.drawString(1.0 * inch, y, f"[{i+1}] {title}")
            y -= 0.18 * inch

            c.setFont("Helvetica", 8)
            c.drawString(1.0 * inch, y, f"Source: {source} | Published: {published}")
            y -= 0.2 * inch

            if y < 0.75 * inch:
                c.showPage()
                y = A4[1] - 1.0 * inch

    # ---------- FOOTER ----------
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawCentredString(
        width / 2.0,
        0.5 * inch,
        f"Generated by Financial Sentiment Analysis System – {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
