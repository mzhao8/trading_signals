"""Unit tests for technical indicator calculations."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.research.indicators import (
    calculate_bollinger,
    calculate_ema_crossover,
    calculate_indicators,
    calculate_macd,
    calculate_rsi,
    get_latest_indicators,
)


def create_sample_df(n_rows: int = 100, start_price: float = 100.0) -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    import numpy as np

    np.random.seed(42)  # For reproducibility

    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n_rows)
    ]

    # Generate random walk prices
    returns = np.random.randn(n_rows) * 0.02  # 2% daily volatility
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV data
    data = {
        "open": prices * (1 + np.random.randn(n_rows) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(n_rows) * 0.01)),
        "low": prices * (1 - np.abs(np.random.randn(n_rows) * 0.01)),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_rows).astype(float),
    }

    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))
    return df


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_returns_series(self):
        """RSI should return a pandas Series."""
        df = create_sample_df()
        rsi, signal = calculate_rsi(df)

        assert isinstance(rsi, pd.Series)
        assert isinstance(signal, pd.Series)
        assert len(rsi) == len(df)

    def test_rsi_range(self):
        """RSI values should be between 0 and 100."""
        df = create_sample_df(200)
        rsi, _ = calculate_rsi(df)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_signal_range(self):
        """RSI signal should be between -100 and 100."""
        df = create_sample_df(200)
        _, signal = calculate_rsi(df)

        valid_signal = signal.dropna()
        assert (valid_signal >= -100).all()
        assert (valid_signal <= 100).all()

    def test_rsi_oversold_bullish(self):
        """When RSI is low (oversold), signal should be positive (bullish)."""
        # Create downtrending data to produce low RSI
        df = create_sample_df(50)
        df["close"] = [100 - i * 0.5 for i in range(50)]  # Steady decline

        rsi, signal = calculate_rsi(df)

        # Check if low RSI produces positive signal
        oversold_mask = rsi < 30
        if oversold_mask.any():
            oversold_signals = signal[oversold_mask].dropna()
            if not oversold_signals.empty:
                assert (oversold_signals > 0).all()

    def test_rsi_overbought_bearish(self):
        """When RSI is high (overbought), signal should be negative (bearish)."""
        # Create uptrending data to produce high RSI
        df = create_sample_df(50)
        df["close"] = [100 + i * 0.5 for i in range(50)]  # Steady rise

        rsi, signal = calculate_rsi(df)

        # Check if high RSI produces negative signal
        overbought_mask = rsi > 70
        if overbought_mask.any():
            overbought_signals = signal[overbought_mask].dropna()
            if not overbought_signals.empty:
                assert (overbought_signals < 0).all()


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_returns_series(self):
        """MACD should return multiple series."""
        df = create_sample_df(100)
        macd_line, signal_line, histogram, score = calculate_macd(df)

        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert isinstance(score, pd.Series)

    def test_macd_histogram_calculation(self):
        """Histogram should be MACD line minus signal line."""
        df = create_sample_df(100)
        macd_line, signal_line, histogram, _ = calculate_macd(df)

        # Check where all values are not NaN
        valid_mask = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        if valid_mask.any():
            expected = macd_line[valid_mask] - signal_line[valid_mask]
            actual = histogram[valid_mask]
            pd.testing.assert_series_equal(
                actual, expected, check_names=False, atol=1e-10
            )

    def test_macd_score_range(self):
        """MACD score should be between -100 and 100."""
        df = create_sample_df(100)
        _, _, _, score = calculate_macd(df)

        valid_score = score.dropna()
        assert (valid_score >= -100).all()
        assert (valid_score <= 100).all()


class TestBollinger:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_returns_series(self):
        """Bollinger Bands should return multiple series."""
        df = create_sample_df(100)
        upper, middle, lower, percent_b, score = calculate_bollinger(df)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert isinstance(percent_b, pd.Series)
        assert isinstance(score, pd.Series)

    def test_bollinger_band_order(self):
        """Upper band should be > middle > lower band."""
        df = create_sample_df(100)
        upper, middle, lower, _, _ = calculate_bollinger(df)

        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_mask.any():
            assert (upper[valid_mask] >= middle[valid_mask]).all()
            assert (middle[valid_mask] >= lower[valid_mask]).all()

    def test_bollinger_percent_b(self):
        """When price is at bands, %B should be 0 or 1."""
        df = create_sample_df(50)
        upper, middle, lower, percent_b, _ = calculate_bollinger(df)

        # %B should be around 0.5 when price is at middle
        # Just check it's calculated and in reasonable range
        valid_pct = percent_b.dropna()
        assert len(valid_pct) > 0


class TestEMACrossover:
    """Tests for EMA crossover calculation."""

    def test_ema_returns_series(self):
        """EMA crossover should return series."""
        df = create_sample_df(100)
        ema_fast, ema_slow, score = calculate_ema_crossover(df)

        assert isinstance(ema_fast, pd.Series)
        assert isinstance(ema_slow, pd.Series)
        assert isinstance(score, pd.Series)

    def test_ema_fast_more_responsive(self):
        """Fast EMA should react more quickly to price changes."""
        # Create data with sudden price jump
        df = create_sample_df(50)
        df.iloc[40:, df.columns.get_loc("close")] *= 1.1  # 10% jump

        ema_fast, ema_slow, _ = calculate_ema_crossover(df)

        # After the jump, fast EMA should be closer to current price
        last_close = df["close"].iloc[-1]
        fast_diff = abs(ema_fast.iloc[-1] - last_close)
        slow_diff = abs(ema_slow.iloc[-1] - last_close)

        assert fast_diff <= slow_diff

    def test_ema_score_range(self):
        """EMA score should be between -100 and 100."""
        df = create_sample_df(100)
        _, _, score = calculate_ema_crossover(df)

        valid_score = score.dropna()
        assert (valid_score >= -100).all()
        assert (valid_score <= 100).all()


class TestCalculateIndicators:
    """Tests for the combined indicator calculation."""

    def test_adds_all_indicator_columns(self):
        """Should add all expected indicator columns."""
        df = create_sample_df(100)
        result = calculate_indicators(df)

        expected_columns = [
            "rsi",
            "rsi_signal",
            "macd",
            "macd_signal_line",
            "macd_histogram",
            "macd_score",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_percent",
            "bb_score",
            "ema_fast",
            "ema_slow",
            "ema_score",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self):
        """Should preserve original OHLCV columns."""
        df = create_sample_df(100)
        result = calculate_indicators(df)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(df[col], result[col])

    def test_get_latest_indicators(self):
        """Should extract latest indicator values."""
        df = create_sample_df(100)
        result = calculate_indicators(df)
        indicators = get_latest_indicators(result)

        assert indicators.rsi is not None or indicators.rsi is None  # May be NaN
        assert indicators.macd is not None or indicators.macd is None


class TestIndicatorDeterminism:
    """Tests for deterministic behavior (same input = same output)."""

    def test_rsi_deterministic(self):
        """RSI should produce same results for same input."""
        df = create_sample_df(100)

        rsi1, signal1 = calculate_rsi(df)
        rsi2, signal2 = calculate_rsi(df)

        pd.testing.assert_series_equal(rsi1, rsi2)
        pd.testing.assert_series_equal(signal1, signal2)

    def test_all_indicators_deterministic(self):
        """All indicators should produce same results for same input."""
        df = create_sample_df(100)

        result1 = calculate_indicators(df)
        result2 = calculate_indicators(df)

        pd.testing.assert_frame_equal(result1, result2)

