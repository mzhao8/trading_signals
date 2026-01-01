"""Signal generation from indicators."""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.shared.config import get_config
from src.shared.types import IndicatorValues, Signal, SignalDirection, SignalResult

from .indicators import calculate_indicators, get_latest_indicators


def calculate_aggregate_score(
    indicators: IndicatorValues,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Calculate weighted aggregate score from individual indicator scores.

    Args:
        indicators: IndicatorValues with individual scores.
        weights: Optional weight overrides. Defaults to config values.

    Returns:
        Aggregate score from -100 to 100.
    """
    config = get_config()
    weights = weights or {
        "rsi": config.signal_weights.rsi,
        "macd": config.signal_weights.macd,
        "bollinger": config.signal_weights.bollinger,
        "ema": config.signal_weights.ema,
    }

    # Collect available scores
    scores = []
    total_weight = 0.0

    if indicators.rsi_signal is not None:
        scores.append(indicators.rsi_signal * weights["rsi"])
        total_weight += weights["rsi"]

    if indicators.macd_score is not None:
        scores.append(indicators.macd_score * weights["macd"])
        total_weight += weights["macd"]

    if indicators.bb_score is not None:
        scores.append(indicators.bb_score * weights["bollinger"])
        total_weight += weights["bollinger"]

    if indicators.ema_score is not None:
        scores.append(indicators.ema_score * weights["ema"])
        total_weight += weights["ema"]

    if total_weight == 0:
        return 0.0

    # Normalize by actual weights used
    return sum(scores) / total_weight


def calculate_confidence(indicators: IndicatorValues) -> float:
    """Calculate signal confidence based on indicator agreement.

    Args:
        indicators: IndicatorValues with individual scores.

    Returns:
        Confidence from 0 to 1.
    """
    scores = []

    if indicators.rsi_signal is not None:
        scores.append(indicators.rsi_signal)
    if indicators.macd_score is not None:
        scores.append(indicators.macd_score)
    if indicators.bb_score is not None:
        scores.append(indicators.bb_score)
    if indicators.ema_score is not None:
        scores.append(indicators.ema_score)

    if len(scores) < 2:
        return 0.5  # Low confidence with few indicators

    # Check if all indicators agree on direction
    positive = sum(1 for s in scores if s > 0)
    negative = sum(1 for s in scores if s < 0)
    neutral = sum(1 for s in scores if s == 0)

    total = len(scores)

    # High confidence if all agree
    if positive == total or negative == total:
        return 1.0

    # Medium confidence if most agree
    max_agreement = max(positive, negative, neutral)
    return max_agreement / total


def generate_signal(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> Signal:
    """Generate a trading signal from OHLCV data.

    Args:
        df: DataFrame with OHLCV data.
        symbol: Trading pair symbol.
        timeframe: Candle timeframe.

    Returns:
        Signal with direction, score, and indicators.
    """
    # Calculate all indicators
    df_with_indicators = calculate_indicators(df)

    # Get latest values
    indicators = get_latest_indicators(df_with_indicators)

    # Calculate aggregate score
    score = calculate_aggregate_score(indicators)

    # Calculate confidence
    confidence = calculate_confidence(indicators)

    # Get current price
    price = float(df["close"].iloc[-1])

    # Get timestamp
    timestamp = df.index[-1]
    if not isinstance(timestamp, datetime):
        timestamp = pd.Timestamp(timestamp).to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        direction=SignalDirection.from_score(score),
        score=round(score, 2),
        indicators=indicators,
        price=price,
        confidence=round(confidence, 2),
    )


def generate_signals(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> SignalResult:
    """Generate signals with timing and metadata.

    Args:
        df: DataFrame with OHLCV data.
        symbol: Trading pair symbol.
        timeframe: Candle timeframe.

    Returns:
        SignalResult with signal and metadata.
    """
    start_time = time.perf_counter()

    signal = generate_signal(df, symbol, timeframe)

    calc_time = (time.perf_counter() - start_time) * 1000

    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        signal=signal,
        raw_data_points=len(df),
        calculation_time_ms=round(calc_time, 2),
    )

