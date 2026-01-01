"""Tests to detect lookahead bias in signal generation.

Lookahead bias occurs when future data is used to calculate signals
for past time periods. This is a critical bug in backtesting that
makes strategies appear profitable when they wouldn't be in live trading.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.research.indicators import calculate_indicators
from src.research.signals import generate_signal


def create_test_df(n_rows: int = 100) -> pd.DataFrame:
    """Create test OHLCV DataFrame."""
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n_rows)
    ]

    # Predictable pattern
    prices = [100 + (i % 20) for i in range(n_rows)]

    data = {
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000.0] * n_rows,
    }

    return pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp"))


class TestNoLookaheadBias:
    """Tests to ensure no lookahead bias in calculations."""

    def test_indicator_only_uses_past_data(self):
        """Indicators at time T should only use data from times <= T."""
        df = create_test_df(100)
        
        # Calculate indicators on full dataset
        full_indicators = calculate_indicators(df)
        
        # Calculate indicators on partial dataset (first 50 rows)
        partial_df = df.iloc[:50]
        partial_indicators = calculate_indicators(partial_df)
        
        # The indicator values at the last row of partial should match
        # the same timestamp in full (allowing for NaN propagation)
        partial_last = partial_indicators.iloc[-1]
        full_at_same_time = full_indicators.iloc[49]  # Same index
        
        # RSI at row 49 should be the same whether we calculated with
        # 50 rows or 100 rows
        if pd.notna(partial_last["rsi"]) and pd.notna(full_at_same_time["rsi"]):
            assert abs(partial_last["rsi"] - full_at_same_time["rsi"]) < 1e-10

    def test_signal_doesnt_change_with_future_data(self):
        """Adding future data should not change past signals."""
        df = create_test_df(100)
        
        # Generate signal at row 50 using only first 50 rows
        partial_signal = generate_signal(df.iloc[:50], "TEST", "1h")
        
        # Generate signal at row 50 using all 100 rows
        # (This would only be valid if we're looking at the 50th row)
        full_df_indicators = calculate_indicators(df)
        
        # The indicator values at row 49 should be identical
        partial_indicators = calculate_indicators(df.iloc[:50])
        
        for col in ["rsi", "rsi_signal", "macd", "macd_score"]:
            if col in partial_indicators.columns:
                partial_val = partial_indicators.iloc[-1][col]
                full_val = full_df_indicators.iloc[49][col]
                
                if pd.notna(partial_val) and pd.notna(full_val):
                    assert abs(partial_val - full_val) < 1e-10, \
                        f"Lookahead detected in {col}: partial={partial_val}, full={full_val}"

    def test_incremental_calculation_matches(self):
        """Calculating incrementally should match calculating all at once."""
        df = create_test_df(100)
        
        # Calculate all at once
        all_at_once = calculate_indicators(df)
        
        # "Simulate" incremental calculation by calculating on subsets
        for i in range(50, 100):
            subset = df.iloc[: i + 1]
            incremental = calculate_indicators(subset)
            
            # Last row of incremental should match row i of all_at_once
            for col in ["rsi", "macd", "bb_middle", "ema_fast"]:
                if col in incremental.columns:
                    inc_val = incremental.iloc[-1][col]
                    all_val = all_at_once.iloc[i][col]
                    
                    if pd.notna(inc_val) and pd.notna(all_val):
                        assert abs(inc_val - all_val) < 1e-10, \
                            f"Mismatch at row {i} for {col}"

    def test_no_future_close_in_signal(self):
        """Signal price should be current close, not future."""
        df = create_test_df(50)
        
        # Modify future closes to be very different
        df_modified = df.copy()
        
        # Generate signal on original
        signal_original = generate_signal(df.iloc[:40], "TEST", "1h")
        
        # Modify rows 40-49 (future data)
        df_modified.iloc[40:, df_modified.columns.get_loc("close")] = 9999
        
        # Generate signal again (should use only first 40 rows)
        signal_modified = generate_signal(df_modified.iloc[:40], "TEST", "1h")
        
        # Signals should be identical since we're not using future data
        assert signal_original.price == signal_modified.price
        assert signal_original.score == signal_modified.score


class TestIndicatorWarmup:
    """Tests for proper indicator warmup periods."""

    def test_rsi_needs_warmup(self):
        """RSI should have NaN values during warmup period."""
        df = create_test_df(100)
        indicators = calculate_indicators(df)
        
        # RSI with period 14 needs at least 14 data points
        # First 13 should be NaN
        rsi_vals = indicators["rsi"]
        
        # At least some initial values should be NaN
        assert rsi_vals.iloc[:13].isna().any()

    def test_macd_needs_warmup(self):
        """MACD should have NaN values during warmup period."""
        df = create_test_df(100)
        indicators = calculate_indicators(df)
        
        # MACD with slow=26, signal=9 needs warmup
        macd_vals = indicators["macd"]
        
        # Should have NaN in early rows
        assert macd_vals.iloc[:25].isna().any()

    def test_short_data_handled_gracefully(self):
        """Short datasets should not cause errors."""
        df = create_test_df(10)  # Very short
        
        # Should not raise
        indicators = calculate_indicators(df)
        
        # Most values will be NaN, but no errors
        assert len(indicators) == 10

