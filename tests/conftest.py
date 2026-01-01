"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1h"],
        "lookback_periods": 100,
        "indicators": {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"period": 20, "std_dev": 2.0},
            "ema": {"fast": 9, "slow": 21},
        },
        "signal_weights": {
            "rsi": 0.25,
            "macd": 0.25,
            "bollinger": 0.25,
            "ema": 0.25,
        },
    }

