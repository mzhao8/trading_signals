"""Market regime detection.

Classifies market conditions as bull, bear, or sideways based on
price action and technical indicators.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    """Market regime classification."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass
class RegimeResult:
    """Result of regime detection.

    Attributes:
        regime: Current market regime.
        confidence: Confidence in the classification (0 to 1).
        trend_strength: Strength of the trend (-100 to +100).
        volatility: Current volatility level (0 to 1, relative to history).
        components: Individual component scores that informed the decision.
        timestamp: When this was calculated.
    """

    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility: float
    components: dict[str, float]
    timestamp: datetime

    @property
    def is_trending(self) -> bool:
        """Whether the market is in a trending state."""
        return self.regime in (MarketRegime.BULL, MarketRegime.BEAR)


def calculate_trend_strength(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
) -> float:
    """Calculate trend strength from -100 to +100.

    Uses:
    - Price vs moving averages
    - Moving average slopes
    - Higher highs / lower lows

    Args:
        df: OHLCV DataFrame.
        fast_period: Fast MA period.
        slow_period: Slow MA period.

    Returns:
        Trend strength score from -100 (strong downtrend) to +100 (strong uptrend).
    """
    if len(df) < slow_period:
        return 0.0

    close = df["close"]

    # Calculate moving averages
    ma_fast = close.rolling(fast_period).mean()
    ma_slow = close.rolling(slow_period).mean()

    current_price = close.iloc[-1]
    current_ma_fast = ma_fast.iloc[-1]
    current_ma_slow = ma_slow.iloc[-1]

    scores = []

    # 1. Price vs MAs (weight: 30%)
    price_vs_fast = (current_price - current_ma_fast) / current_ma_fast * 100
    price_vs_slow = (current_price - current_ma_slow) / current_ma_slow * 100
    price_score = (price_vs_fast + price_vs_slow) / 2
    price_score = max(-100, min(100, price_score * 10))  # Scale
    scores.append(("price_vs_ma", price_score, 0.3))

    # 2. MA crossover (weight: 25%)
    ma_diff = (current_ma_fast - current_ma_slow) / current_ma_slow * 100
    ma_score = max(-100, min(100, ma_diff * 20))  # Scale
    scores.append(("ma_crossover", ma_score, 0.25))

    # 3. MA slope (weight: 25%)
    slope_period = min(10, len(df) - slow_period)
    if slope_period > 1:
        fast_slope = (ma_fast.iloc[-1] - ma_fast.iloc[-slope_period]) / ma_fast.iloc[-slope_period] * 100
        slow_slope = (ma_slow.iloc[-1] - ma_slow.iloc[-slope_period]) / ma_slow.iloc[-slope_period] * 100
        slope_score = (fast_slope + slow_slope) / 2
        slope_score = max(-100, min(100, slope_score * 20))
    else:
        slope_score = 0.0
    scores.append(("ma_slope", slope_score, 0.25))

    # 4. Higher highs / lower lows (weight: 20%)
    lookback = min(20, len(df) - 1)
    if lookback >= 5:
        highs = df["high"].iloc[-lookback:]
        lows = df["low"].iloc[-lookback:]

        # Count higher highs
        higher_highs = sum(
            1 for i in range(1, len(highs)) if highs.iloc[i] > highs.iloc[i - 1]
        )
        # Count lower lows
        lower_lows = sum(
            1 for i in range(1, len(lows)) if lows.iloc[i] < lows.iloc[i - 1]
        )

        hh_ll_score = (higher_highs - lower_lows) / (lookback - 1) * 100
    else:
        hh_ll_score = 0.0
    scores.append(("hh_ll", hh_ll_score, 0.2))

    # Weighted average
    total_score = sum(score * weight for _, score, weight in scores)

    return total_score


def calculate_volatility(
    df: pd.DataFrame,
    period: int = 20,
    lookback: int = 100,
) -> float:
    """Calculate current volatility relative to recent history.

    Args:
        df: OHLCV DataFrame.
        period: ATR period.
        lookback: Historical lookback for comparison.

    Returns:
        Volatility score from 0 (low) to 1 (high relative to history).
    """
    if len(df) < max(period, lookback):
        return 0.5

    # Calculate ATR
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    # Normalize ATR by price
    atr_pct = atr / close * 100

    current_vol = atr_pct.iloc[-1]
    historical_vol = atr_pct.iloc[-lookback:]

    # Calculate percentile
    percentile = (historical_vol < current_vol).sum() / len(historical_vol)

    return percentile


def detect_regime(
    df: pd.DataFrame,
    trend_threshold: float = 30.0,
    volatility_lookback: int = 100,
) -> RegimeResult:
    """Detect current market regime.

    Args:
        df: OHLCV DataFrame with at least 50 periods.
        trend_threshold: Minimum trend strength to classify as trending.
        volatility_lookback: Historical periods for volatility comparison.

    Returns:
        RegimeResult with classification and metrics.
    """
    timestamp = datetime.now(timezone.utc)

    if len(df) < 50:
        return RegimeResult(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.3,
            trend_strength=0.0,
            volatility=0.5,
            components={"error": "insufficient_data"},
            timestamp=timestamp,
        )

    # Calculate components
    trend_strength = calculate_trend_strength(df)
    volatility = calculate_volatility(df, lookback=min(volatility_lookback, len(df)))

    # Additional regime indicators
    components = {}

    # RSI for momentum confirmation
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    components["rsi"] = current_rsi

    # ADX for trend strength (simplified)
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr = (high - low).rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(14).mean()
    current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
    components["adx"] = current_adx

    # Price distance from 200 MA
    if len(df) >= 200:
        ma_200 = close.rolling(200).mean().iloc[-1]
        price_vs_200 = (close.iloc[-1] - ma_200) / ma_200 * 100
        components["price_vs_200ma"] = price_vs_200

    components["trend_strength"] = trend_strength
    components["volatility_pct"] = volatility * 100

    # Classify regime
    if trend_strength > trend_threshold:
        regime = MarketRegime.BULL
        # Higher confidence if ADX confirms trend
        base_confidence = min(abs(trend_strength) / 100, 1.0)
        adx_boost = 0.2 if current_adx > 25 else 0
        confidence = min(base_confidence + adx_boost, 1.0)

    elif trend_strength < -trend_threshold:
        regime = MarketRegime.BEAR
        base_confidence = min(abs(trend_strength) / 100, 1.0)
        adx_boost = 0.2 if current_adx > 25 else 0
        confidence = min(base_confidence + adx_boost, 1.0)

    else:
        regime = MarketRegime.SIDEWAYS
        # Higher confidence for sideways if ADX is low
        adx_confidence = 0.3 if current_adx < 20 else 0.1
        range_confidence = 1 - abs(trend_strength) / trend_threshold
        confidence = min(range_confidence * 0.5 + adx_confidence + 0.3, 1.0)

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        trend_strength=trend_strength,
        volatility=volatility,
        components=components,
        timestamp=timestamp,
    )


def get_regime_emoji(regime: MarketRegime) -> str:
    """Get emoji for regime display."""
    return {
        MarketRegime.BULL: "🐂",
        MarketRegime.BEAR: "🐻",
        MarketRegime.SIDEWAYS: "↔️",
    }.get(regime, "❓")


def get_regime_color(regime: MarketRegime) -> str:
    """Get color for regime display."""
    return {
        MarketRegime.BULL: "green",
        MarketRegime.BEAR: "red",
        MarketRegime.SIDEWAYS: "yellow",
    }.get(regime, "white")

