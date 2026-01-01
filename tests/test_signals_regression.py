"""Regression tests to ensure same data produces same signals."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.indicators import calculate_indicators
from src.research.signals import generate_signal, generate_signals


def create_deterministic_df() -> pd.DataFrame:
    """Create a deterministic OHLCV DataFrame for regression testing.
    
    Uses fixed values to ensure reproducibility across runs.
    """
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(100)
    ]

    # Fixed price pattern: oscillating with trend
    prices = []
    base = 100.0
    for i in range(100):
        # Add trend
        trend = i * 0.1
        # Add oscillation
        oscillation = 5 * (1 if i % 10 < 5 else -1)
        prices.append(base + trend + oscillation)

    data = {
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000.0 + i * 10 for i in range(100)],
    }

    return pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))


class TestSignalDeterminism:
    """Tests to ensure signals are deterministic."""

    def test_same_data_same_signal(self):
        """Same input data should produce identical signals."""
        df = create_deterministic_df()

        signal1 = generate_signal(df, "BTCUSDT", "1h")
        signal2 = generate_signal(df, "BTCUSDT", "1h")

        assert signal1.score == signal2.score
        assert signal1.direction == signal2.direction
        assert signal1.confidence == signal2.confidence

    def test_indicator_values_match(self):
        """Indicator values should be identical for same input."""
        df = create_deterministic_df()

        signal1 = generate_signal(df, "BTCUSDT", "1h")
        signal2 = generate_signal(df, "BTCUSDT", "1h")

        # Compare all indicator values
        ind1 = signal1.indicators
        ind2 = signal2.indicators

        assert ind1.rsi == ind2.rsi
        assert ind1.macd == ind2.macd
        assert ind1.bb_percent == ind2.bb_percent
        assert ind1.ema_fast == ind2.ema_fast

    def test_multiple_calls_consistent(self):
        """Multiple calls should produce consistent results."""
        df = create_deterministic_df()

        signals = [generate_signal(df, "BTCUSDT", "1h") for _ in range(10)]

        # All signals should be identical
        first = signals[0]
        for signal in signals[1:]:
            assert signal.score == first.score
            assert signal.direction == first.direction


class TestSignalResultMetadata:
    """Tests for signal result metadata."""

    def test_result_contains_metadata(self):
        """SignalResult should contain correct metadata."""
        df = create_deterministic_df()

        result = generate_signals(df, "BTCUSDT", "1h")

        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
        assert result.raw_data_points == len(df)
        assert result.calculation_time_ms >= 0

    def test_price_matches_last_close(self):
        """Signal price should match last close price."""
        df = create_deterministic_df()

        signal = generate_signal(df, "BTCUSDT", "1h")

        assert signal.price == df["close"].iloc[-1]


class TestKnownOutputs:
    """Tests against known expected outputs."""

    def test_uptrend_produces_buy_signal(self):
        """Strong uptrend should produce buy signal."""
        timestamps = [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            for i in range(100)
        ]

        # Create strong uptrend
        prices = [100 + i * 0.5 for i in range(100)]

        data = {
            "open": [p - 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000.0] * 100,
        }

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))
        signal = generate_signal(df, "BTCUSDT", "1h")

        # EMA crossover should be bullish in uptrend
        assert signal.indicators.ema_score is not None
        assert signal.indicators.ema_score > 0

    def test_downtrend_produces_sell_signal(self):
        """Strong downtrend should produce sell signal."""
        timestamps = [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
            for i in range(100)
        ]

        # Create strong downtrend
        prices = [150 - i * 0.5 for i in range(100)]

        data = {
            "open": [p + 0.1 for p in prices],
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [1000.0] * 100,
        }

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))
        signal = generate_signal(df, "BTCUSDT", "1h")

        # EMA crossover should be bearish in downtrend
        assert signal.indicators.ema_score is not None
        assert signal.indicators.ema_score < 0

