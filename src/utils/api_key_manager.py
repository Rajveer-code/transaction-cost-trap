"""
api_key_manager.py
==================
Centralized API key storage and retrieval.

Stores keys persistently in config/api_keys.json.
Works with both Streamlit UI and backend modules.

Author: Rajveer Singh Pall
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# CONFIGURATION
# ============================================================

# Determine project root (this file is in src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

# Environment variable names (fallback)
ENV_KEY_NAMES = {
    "finnhub": "FINNHUB_API_KEY",
    "newsapi": "NEWSAPI_KEY",
    "alphavantage": "ALPHAVANTAGE_API_KEY",
}


# ============================================================
# KEY STORAGE
# ============================================================

def save_api_keys(keys: Dict[str, str]) -> bool:
    """
    Save API keys to persistent config file.
    
    Args:
        keys: Dictionary with keys like:
            {
                "finnhub": "your_key_here",
                "newsapi": "your_key_here",
                "alphavantage": "your_key_here"
            }
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing keys if file exists
        existing_keys = {}
        if API_KEYS_FILE.exists():
            with open(API_KEYS_FILE, "r") as f:
                existing_keys = json.load(f)
        
        # Update with new keys (don't overwrite empty values)
        for key, value in keys.items():
            if value:  # Only save non-empty values
                existing_keys[key] = value
        
        # Write to file
        with open(API_KEYS_FILE, "w") as f:
            json.dump(existing_keys, f, indent=2)
        
        print(f"✅ API keys saved to {API_KEYS_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save API keys: {e}")
        return False


def load_api_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from config file or environment variables.
    
    Priority:
    1. config/api_keys.json (highest priority)
    2. Environment variables (fallback)
    3. None (if not found anywhere)
    
    Returns:
        Dictionary with keys: finnhub, newsapi, alphavantage
    """
    keys = {
        "finnhub": None,
        "newsapi": None,
        "alphavantage": None,
    }
    
    # Try loading from config file first
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, "r") as f:
                file_keys = json.load(f)
            
            # Update keys from file
            for key in keys.keys():
                if key in file_keys and file_keys[key]:
                    keys[key] = file_keys[key]
            
            print(f"✅ Loaded API keys from {API_KEYS_FILE}")
        except Exception as e:
            print(f"⚠️  Failed to load API keys from file: {e}")
    
    # Fallback to environment variables
    for key, env_var in ENV_KEY_NAMES.items():
        if keys[key] is None:  # Only check env if not loaded from file
            env_value = os.getenv(env_var)
            if env_value:
                keys[key] = env_value
                print(f"✅ Loaded {key} from environment variable")
    
    # Report status
    active_keys = [k for k, v in keys.items() if v is not None]
    if active_keys:
        print(f"✅ Active API providers: {', '.join(active_keys)}")
    else:
        print("⚠️  No API keys found. Please configure in Settings.")
    
    return keys


def get_api_key(provider: str) -> Optional[str]:
    """
    Get a single API key for a specific provider.
    
    Args:
        provider: One of "finnhub", "newsapi", "alphavantage"
    
    Returns:
        API key string or None
    """
    keys = load_api_keys()
    return keys.get(provider.lower())


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate which API keys are configured.
    
    Returns:
        Dictionary mapping provider -> is_configured (bool)
    """
    keys = load_api_keys()
    return {
        provider: (key is not None and key != "")
        for provider, key in keys.items()
    }


def clear_api_keys() -> bool:
    """
    Clear all saved API keys (delete config file).
    
    Returns:
        True if cleared successfully
    """
    try:
        if API_KEYS_FILE.exists():
            API_KEYS_FILE.unlink()
            print(f"✅ Cleared API keys from {API_KEYS_FILE}")
        return True
    except Exception as e:
        print(f"❌ Failed to clear API keys: {e}")
        return False


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def get_config_file_path() -> Path:
    """Get the path to the API keys config file."""
    return API_KEYS_FILE


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("\nTesting api_key_manager.py...")
    print("=" * 60)
    
    # Test 1: Save keys
    print("\n📝 Test 1: Saving API keys...")
    test_keys = {
        "finnhub": "test_finnhub_key_123",
        "newsapi": "test_newsapi_key_456",
        "alphavantage": "test_alphavantage_key_789"
    }
    success = save_api_keys(test_keys)
    print(f"Save result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 2: Load keys
    print("\n📖 Test 2: Loading API keys...")
    loaded_keys = load_api_keys()
    print("Loaded keys:")
    for provider, key in loaded_keys.items():
        masked = f"{key[:8]}..." if key else "None"
        print(f"  {provider}: {masked}")
    
    # Test 3: Validate keys
    print("\n✔️  Test 3: Validating API keys...")
    validation = validate_api_keys()
    print("Validation results:")
    for provider, is_valid in validation.items():
        status = "✅ Configured" if is_valid else "❌ Missing"
        print(f"  {provider}: {status}")
    
    # Test 4: Get single key
    print("\n🔑 Test 4: Getting single key...")
    finnhub_key = get_api_key("finnhub")
    masked = f"{finnhub_key[:8]}..." if finnhub_key else "None"
    print(f"Finnhub key: {masked}")
    
    # Test 5: Clear keys
    print("\n🗑️  Test 5: Clearing API keys...")
    clear_success = clear_api_keys()
    print(f"Clear result: {'✅ Success' if clear_success else '❌ Failed'}")
    
    print("\n" + "=" * 60)
    print("✅ All tests complete!")