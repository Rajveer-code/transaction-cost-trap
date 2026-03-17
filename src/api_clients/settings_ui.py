"""
settings_ui.py (FIXED)
======================

Streamlit UI module for configuring API keys.
Now saves keys persistently to config/api_keys.json.

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_key_manager import save_api_keys, load_api_keys, validate_api_keys


def render_api_settings() -> None:
    """Render API key configuration UI in the Settings tab."""

    st.markdown("### 🔑 API Key Configuration")
    st.info(
        """
        Add your API keys below.  
        Keys are saved **persistently** to `config/api_keys.json`.

        **Free APIs:**
        - Finnhub → https://finnhub.io (Company news, 60 calls/min free)
        - NewsAPI → https://newsapi.org (General news, 100 calls/day free)
        - Alpha Vantage → https://www.alphavantage.co (News + sentiment, 500 calls/day free)
        """
    )

    # Load existing keys from disk
    existing_keys = load_api_keys()

    # -----------------------------
    # Finnhub Key
    # -----------------------------
    finnhub_key = st.text_input(
        "Finnhub API Key",
        type="password",
        value=existing_keys.get("finnhub", "") or "",
        help="Used for financial company news",
        key="finnhub_input"
    )

    # -----------------------------
    # NewsAPI Key
    # -----------------------------
    newsapi_key = st.text_input(
        "NewsAPI Key",
        type="password",
        value=existing_keys.get("newsapi", "") or "",
        help="General news provider (100 req/day)",
        key="newsapi_input"
    )

    # -----------------------------
    # Alpha Vantage Key
    # -----------------------------
    av_key = st.text_input(
        "Alpha Vantage API Key",
        type="password",
        value=existing_keys.get("alphavantage", "") or "",
        help="News + sentiment scoring (500 req/day)",
        key="alphavantage_input"
    )

    st.markdown("---")

    # Save button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("💾 Save Keys", type="primary", use_container_width=True):
            # Prepare keys dictionary
            keys_to_save = {
                "finnhub": finnhub_key,
                "newsapi": newsapi_key,
                "alphavantage": av_key,
            }
            
            # Save to disk
            success = save_api_keys(keys_to_save)
            
            if success:
                st.success("✅ API keys saved successfully!")
                st.balloons()
                
                # Show which keys are active
                validation = validate_api_keys()
                active = [k for k, v in validation.items() if v]
                if active:
                    st.info(f"🔑 Active providers: **{', '.join(active)}**")
            else:
                st.error("❌ Failed to save API keys. Check file permissions.")
    
    with col2:
        if st.button("🗑️ Clear All Keys", use_container_width=True):
            from src.utils.api_key_manager import clear_api_keys
            if clear_api_keys():
                st.success("✅ All API keys cleared!")
                st.rerun()

    # Display current status
    st.markdown("---")
    st.markdown("### 📊 Current Status")
    
    validation = validate_api_keys()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅" if validation.get("finnhub") else "❌"
        st.metric("Finnhub", status)
    
    with col2:
        status = "✅" if validation.get("newsapi") else "❌"
        st.metric("NewsAPI", status)
    
    with col3:
        status = "✅" if validation.get("alphavantage") else "❌"
        st.metric("Alpha Vantage", status)

    # Show config file location
    from src.utils.api_key_manager import get_config_file_path
    config_path = get_config_file_path()
    
    st.markdown("---")
    st.markdown("### 📁 Storage Location")
    st.code(str(config_path), language="text")
    
    if config_path.exists():
        st.success("✅ Config file exists")
    else:
        st.warning("⚠️  Config file not created yet. Click 'Save Keys' to create it.")