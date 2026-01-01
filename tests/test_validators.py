"""Unit tests for data validation."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.validators import (
    ValidationError,
    clean_ohlcv,
    ensure_sorted,
    remove_duplicates,
    validate_ohlcv,
)


def create_valid_df() -> pd.DataFrame:
    """Create a valid OHLCV DataFrame for testing."""
    timestamps = [
        datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(24)
    ]

    data = {
        "open": [100.0] * 24,
        "high": [105.0] * 24,
        "low": [95.0] * 24,
        "close": [102.0] * 24,
        "volume": [1000.0] * 24,
    }

    return pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))


class TestValidateOHLCV:
    """Tests for validate_ohlcv function."""

    def test_valid_df_passes(self):
        """Valid DataFrame should pass validation."""
        df = create_valid_df()
        is_valid, errors = validate_ohlcv(df)

        assert is_valid
        assert len(errors) == 0

    def test_empty_df_fails(self):
        """Empty DataFrame should fail validation."""
        df = pd.DataFrame()
        is_valid, errors = validate_ohlcv(df)

        assert not is_valid
        assert "DataFrame is empty" in errors[0]

    def test_missing_columns_fails(self):
        """DataFrame with missing columns should fail."""
        df = create_valid_df()
        df = df.drop(columns=["volume"])

        is_valid, errors = validate_ohlcv(df)

        assert not is_valid
        assert any("Missing columns" in e for e in errors)

    def test_nan_values_detected(self):
        """NaN values should be detected."""
        df = create_valid_df()
        df.iloc[5, df.columns.get_loc("close")] = float("nan")

        is_valid, errors = validate_ohlcv(df)

        assert not is_valid
        assert any("NaN" in e for e in errors)

    def test_negative_price_fails(self):
        """Negative prices should fail validation."""
        df = create_valid_df()
        df.iloc[5, df.columns.get_loc("close")] = -100.0

        is_valid, errors = validate_ohlcv(df, check_values=True)

        assert not is_valid
        assert any("Non-positive" in e for e in errors)

    def test_negative_volume_fails(self):
        """Negative volume should fail validation."""
        df = create_valid_df()
        df.iloc[5, df.columns.get_loc("volume")] = -1000.0

        is_valid, errors = validate_ohlcv(df, check_values=True)

        assert not is_valid
        assert any("Negative volume" in e for e in errors)

    def test_invalid_high_detected(self):
        """High lower than low should be detected."""
        df = create_valid_df()
        df.iloc[5, df.columns.get_loc("high")] = 90.0  # Lower than low

        is_valid, errors = validate_ohlcv(df, check_values=True)

        assert not is_valid
        assert any("Invalid high" in e for e in errors)

    def test_invalid_low_detected(self):
        """Low higher than open/close should be detected."""
        df = create_valid_df()
        df.iloc[5, df.columns.get_loc("low")] = 110.0  # Higher than high

        is_valid, errors = validate_ohlcv(df, check_values=True)

        assert not is_valid
        # This will trigger both high and low validation errors

    def test_gap_detection(self):
        """Gaps in timestamp should be detected."""
        timestamps = [
            datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(10)
        ]
        # Remove some timestamps to create gap
        timestamps = timestamps[:5] + timestamps[8:]

        df = create_valid_df().iloc[: len(timestamps)]
        df.index = pd.DatetimeIndex(timestamps)

        is_valid, errors = validate_ohlcv(df, expected_interval="1h", check_gaps=True)

        assert not is_valid
        assert any("Gap detected" in e or "missing candles" in e for e in errors)

    def test_skip_gap_check(self):
        """Gap check should be skipped if disabled."""
        timestamps = [
            datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc) for i in [0, 1, 2, 10, 11]
        ]

        df = create_valid_df().iloc[:5]
        df.index = pd.DatetimeIndex(timestamps)

        is_valid, errors = validate_ohlcv(df, check_gaps=False)

        assert is_valid


class TestEnsureSorted:
    """Tests for ensure_sorted function."""

    def test_already_sorted(self):
        """Already sorted DataFrame should be unchanged."""
        df = create_valid_df()
        result = ensure_sorted(df)

        pd.testing.assert_frame_equal(df, result)

    def test_unsorted_gets_sorted(self):
        """Unsorted DataFrame should be sorted."""
        df = create_valid_df()
        # Reverse the order
        df = df.iloc[::-1]

        result = ensure_sorted(df)

        assert result.index.is_monotonic_increasing


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""

    def test_no_duplicates(self):
        """DataFrame without duplicates should be unchanged."""
        df = create_valid_df()
        result = remove_duplicates(df)

        pd.testing.assert_frame_equal(df, result)

    def test_duplicates_removed(self):
        """Duplicate timestamps should be removed."""
        df = create_valid_df()
        # Add duplicate
        dup = df.iloc[[5]].copy()
        df = pd.concat([df, dup])

        assert df.index.has_duplicates

        result = remove_duplicates(df)

        assert not result.index.has_duplicates
        assert len(result) == 24  # Original length


class TestCleanOHLCV:
    """Tests for clean_ohlcv function."""

    def test_clean_handles_unsorted_duplicates(self):
        """Clean should handle both unsorted and duplicate data."""
        df = create_valid_df()

        # Add duplicate and reverse
        dup = df.iloc[[5]].copy()
        df = pd.concat([df, dup])
        df = df.iloc[::-1]

        result = clean_ohlcv(df)

        assert not result.index.has_duplicates
        assert result.index.is_monotonic_increasing

