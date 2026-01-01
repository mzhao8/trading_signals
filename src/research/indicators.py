"""Technical indicator calculations using pandas-ta."""

from typing import Optional

import pandas as pd
import pandas_ta as ta

from src.shared.config import get_config
from src.shared.types import IndicatorValues


def calculate_rsi(
    df: pd.DataFrame,
    period: Optional[int] = None,
    oversold: Optional[float] = None,
    overbought: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:
    """Calculate RSI and its signal score.

    Args:
        df: DataFrame with 'close' column.
        period: RSI period (default from config).
        oversold: Oversold threshold (default from config).
        overbought: Overbought threshold (default from config).

    Returns:
        Tuple of (RSI values, RSI signal scores from -100 to 100).
    """
    config = get_config()
    period = period or config.indicators.rsi.period
    oversold = oversold or config.indicators.rsi.oversold
    overbought = overbought or config.indicators.rsi.overbought

    rsi = ta.rsi(df["close"], length=period)

    # Convert RSI to signal score
    # RSI < oversold = bullish (positive score)
    # RSI > overbought = bearish (negative score)
    # Linear scaling between thresholds
    def rsi_to_score(value: float) -> float:
        if pd.isna(value):
            return 0.0
        if value <= oversold:
            # Oversold: bullish signal (0 to 100)
            return 100 * (oversold - value) / oversold
        elif value >= overbought:
            # Overbought: bearish signal (0 to -100)
            return -100 * (value - overbought) / (100 - overbought)
        else:
            # Neutral zone: scale between -20 and 20
            mid = (oversold + overbought) / 2
            if value < mid:
                return 20 * (mid - value) / (mid - oversold)
            else:
                return -20 * (value - mid) / (overbought - mid)

    signal = rsi.apply(rsi_to_score)

    return rsi, signal


def calculate_macd(
    df: pd.DataFrame,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal_period: Optional[int] = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate MACD and its signal score.

    Args:
        df: DataFrame with 'close' column.
        fast: Fast EMA period (default from config).
        slow: Slow EMA period (default from config).
        signal_period: Signal line period (default from config).

    Returns:
        Tuple of (MACD line, signal line, histogram, MACD score from -100 to 100).
    """
    config = get_config()
    fast = fast or config.indicators.macd.fast
    slow = slow or config.indicators.macd.slow
    signal_period = signal_period or config.indicators.macd.signal

    macd_result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal_period)

    macd_line = macd_result.iloc[:, 0]  # MACD line
    signal_line = macd_result.iloc[:, 1]  # Signal line
    histogram = macd_result.iloc[:, 2]  # Histogram

    # Score based on histogram direction and magnitude
    # Normalize histogram to price percentage for comparability
    price_avg = df["close"].rolling(slow).mean()
    normalized_hist = (histogram / price_avg) * 1000  # Scale to meaningful range

    def hist_to_score(value: float) -> float:
        if pd.isna(value):
            return 0.0
        # Clip to reasonable range and scale to -100 to 100
        clamped = max(min(value, 10), -10)
        return clamped * 10

    score = normalized_hist.apply(hist_to_score)

    return macd_line, signal_line, histogram, score


def calculate_bollinger(
    df: pd.DataFrame,
    period: Optional[int] = None,
    std_dev: Optional[float] = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands and signal score.

    Args:
        df: DataFrame with 'close' column.
        period: Moving average period (default from config).
        std_dev: Number of standard deviations (default from config).

    Returns:
        Tuple of (upper band, middle band, lower band, %B, BB score from -100 to 100).
    """
    config = get_config()
    period = period or config.indicators.bollinger.period
    std_dev = std_dev or config.indicators.bollinger.std_dev

    bb_result = ta.bbands(df["close"], length=period, std=std_dev)

    lower = bb_result.iloc[:, 0]  # Lower band
    middle = bb_result.iloc[:, 1]  # Middle band
    upper = bb_result.iloc[:, 2]  # Upper band

    # Calculate %B (position within bands, 0 = lower, 1 = upper)
    band_width = upper - lower
    percent_b = (df["close"] - lower) / band_width

    # Score based on %B
    # %B < 0 = very oversold (bullish)
    # %B > 1 = very overbought (bearish)
    # %B around 0.5 = neutral
    def bb_to_score(value: float) -> float:
        if pd.isna(value):
            return 0.0
        if value <= 0:
            # Below lower band: very bullish
            return min(100, 50 + abs(value) * 50)
        elif value >= 1:
            # Above upper band: very bearish
            return max(-100, -50 - (value - 1) * 50)
        elif value < 0.5:
            # Between lower and middle: bullish
            return (0.5 - value) * 100
        else:
            # Between middle and upper: bearish
            return (0.5 - value) * 100

    score = percent_b.apply(bb_to_score)

    return upper, middle, lower, percent_b, score


def calculate_ema_crossover(
    df: pd.DataFrame,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate EMA crossover and signal score.

    Args:
        df: DataFrame with 'close' column.
        fast: Fast EMA period (default from config).
        slow: Slow EMA period (default from config).

    Returns:
        Tuple of (fast EMA, slow EMA, EMA score from -100 to 100).
    """
    config = get_config()
    fast = fast or config.indicators.ema.fast
    slow = slow or config.indicators.ema.slow

    ema_fast = ta.ema(df["close"], length=fast)
    ema_slow = ta.ema(df["close"], length=slow)

    # Score based on EMA relationship and momentum
    # Fast > Slow = bullish
    # Fast < Slow = bearish
    # Magnitude based on percentage difference
    def ema_to_score(fast_val: float, slow_val: float) -> float:
        if pd.isna(fast_val) or pd.isna(slow_val) or slow_val == 0:
            return 0.0

        # Percentage difference
        diff_pct = (fast_val - slow_val) / slow_val * 100

        # Scale to -100 to 100 (capped at +/- 5% difference)
        score = diff_pct * 20  # 5% diff = 100 score
        return max(min(score, 100), -100)

    score = pd.Series(
        [ema_to_score(f, s) for f, s in zip(ema_fast, ema_slow)],
        index=df.index,
    )

    return ema_fast, ema_slow, score


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all indicators and add them to the DataFrame.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        DataFrame with added indicator columns.
    """
    result = df.copy()

    # RSI
    rsi, rsi_signal = calculate_rsi(df)
    result["rsi"] = rsi
    result["rsi_signal"] = rsi_signal

    # MACD
    macd_line, signal_line, histogram, macd_score = calculate_macd(df)
    result["macd"] = macd_line
    result["macd_signal_line"] = signal_line
    result["macd_histogram"] = histogram
    result["macd_score"] = macd_score

    # Bollinger Bands
    upper, middle, lower, percent_b, bb_score = calculate_bollinger(df)
    result["bb_upper"] = upper
    result["bb_middle"] = middle
    result["bb_lower"] = lower
    result["bb_percent"] = percent_b
    result["bb_score"] = bb_score

    # EMA Crossover
    ema_fast, ema_slow, ema_score = calculate_ema_crossover(df)
    result["ema_fast"] = ema_fast
    result["ema_slow"] = ema_slow
    result["ema_score"] = ema_score

    return result


def get_latest_indicators(df: pd.DataFrame) -> IndicatorValues:
    """Get the latest indicator values from a DataFrame with indicators.

    Args:
        df: DataFrame with indicator columns (from calculate_indicators).

    Returns:
        IndicatorValues with latest values.
    """
    if df.empty:
        return IndicatorValues()

    latest = df.iloc[-1]

    return IndicatorValues(
        rsi=_safe_float(latest.get("rsi")),
        rsi_signal=_safe_float(latest.get("rsi_signal")),
        macd=_safe_float(latest.get("macd")),
        macd_signal=_safe_float(latest.get("macd_signal_line")),
        macd_histogram=_safe_float(latest.get("macd_histogram")),
        macd_score=_safe_float(latest.get("macd_score")),
        bb_upper=_safe_float(latest.get("bb_upper")),
        bb_middle=_safe_float(latest.get("bb_middle")),
        bb_lower=_safe_float(latest.get("bb_lower")),
        bb_percent=_safe_float(latest.get("bb_percent")),
        bb_score=_safe_float(latest.get("bb_score")),
        ema_fast=_safe_float(latest.get("ema_fast")),
        ema_slow=_safe_float(latest.get("ema_slow")),
        ema_score=_safe_float(latest.get("ema_score")),
    )


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float, handling NaN."""
    if value is None or pd.isna(value):
        return None
    return float(value)
