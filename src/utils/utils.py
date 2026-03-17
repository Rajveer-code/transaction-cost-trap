"""
utils.py
========
Shared utility functions used across the entire project.

Contains:
- JSON loading (with UTF-8-BOM handling)
- Text cleaning
- Date formatting
- Logging helpers
- Schema validation

Author: Rajveer Singh Pall
"""

import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional, Union


# ============================================================
# JSON UTILITIES
# ============================================================

def safe_json_load(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file safely, handling UTF-8-BOM encoding issues.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    # Try UTF-8 with BOM first (common in Windows-created files)
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to regular UTF-8
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def safe_json_save(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary as JSON file safely.
    
    Args:
        data: Dictionary to save
        filepath: Output path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Clean and normalize text for NLP processing.
    
    Args:
        text: Raw text string
        max_length: Optional maximum length (truncates if exceeded)
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


# ============================================================
# DATE UTILITIES
# ============================================================

def format_date(dt: Union[datetime, date, str]) -> str:
    """
    Convert various date formats to YYYY-MM-DD string.
    
    Args:
        dt: Date in various formats
        
    Returns:
        Standardized date string (YYYY-MM-DD)
    """
    if isinstance(dt, str):
        # Already a string, try to parse and reformat
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt  # Return as-is if can't parse
    
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    
    if isinstance(dt, date):
        return dt.strftime("%Y-%m-%d")
    
    raise ValueError(f"Unsupported date type: {type(dt)}")


def today_str() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return datetime.now().strftime("%Y-%m-%d")


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime object.
    
    Supports common formats:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS
    - ISO 8601
    
    Args:
        date_str: Date string
        
    Returns:
        datetime object
    """
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        # Fallback: try common formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Could not parse date: {date_str}")


# ============================================================
# LOGGING UTILITIES
# ============================================================

def log_info(message: str, module: str = "APP") -> None:
    """Print timestamped info message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{module}] INFO: {message}")


def log_warning(message: str, module: str = "APP") -> None:
    """Print timestamped warning message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{module}] ⚠️  WARNING: {message}")


def log_error(message: str, module: str = "APP") -> None:
    """Print timestamped error message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{module}] ❌ ERROR: {message}")


def log_success(message: str, module: str = "APP") -> None:
    """Print timestamped success message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{module}] ✅ SUCCESS: {message}")


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format.
    
    Args:
        ticker: Stock ticker string
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip().upper()
    
    # Valid ticker: 1-5 uppercase letters
    return bool(re.match(r"^[A-Z]{1,5}$", ticker))


def validate_feature_value(value: Any, feature_name: str) -> bool:
    """
    Validate that a feature value is appropriate.
    
    Args:
        value: Feature value
        feature_name: Name of the feature
        
    Returns:
        True if valid, False otherwise
    """
    # Check for None
    if value is None:
        return False
    
    # Check for NaN
    try:
        import math
        if math.isnan(value):
            return False
    except (TypeError, ValueError):
        pass
    
    # Check for inf
    try:
        import math
        if math.isinf(value):
            return False
    except (TypeError, ValueError):
        pass
    
    return True


# ============================================================
# PATH UTILITIES
# ============================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Assumes this file is in src/utils/utils.py
    
    Returns:
        Path to project root
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing utils.py...")
    
    # Test date formatting
    print(f"Today: {today_str()}")
    
    # Test text cleaning
    dirty_text = "  This   has   extra    spaces!!!  "
    clean = clean_text(dirty_text)
    print(f"Cleaned: '{clean}'")
    
    # Test ticker validation
    print(f"Valid ticker 'AAPL': {validate_ticker('AAPL')}")
    print(f"Invalid ticker '123': {validate_ticker('123')}")
    
    # Test project root
    print(f"Project root: {get_project_root()}")
    
    print("✅ utils.py tests passed")