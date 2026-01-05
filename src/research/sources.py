"""Signal source abstraction for pluggable signal generation.

This module defines the protocol for signal sources and provides
implementations for different data types (technical, on-chain, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import pandas as pd


class SignalCategory(str, Enum):
    """Category of signal source."""

    TECHNICAL = "technical"
    ONCHAIN = "onchain"
    FUNDAMENTAL = "fundamental"
    SUPPLY = "supply"
    SENTIMENT = "sentiment"


@dataclass
class SignalOutput:
    """Output from a signal source.

    Attributes:
        score: Signal score from -100 (bearish) to +100 (bullish).
        confidence: Confidence in the signal (0 to 1).
        components: Individual component scores that make up this signal.
        timestamp: When this signal was calculated.
        data_timestamp: When the underlying data was last updated.
        metadata: Additional metadata about the signal.
    """

    score: float
    confidence: float = 1.0
    components: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_timestamp: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Clamp score to valid range
        self.score = max(-100, min(100, self.score))
        self.confidence = max(0, min(1, self.confidence))


@runtime_checkable
class SignalSource(Protocol):
    """Protocol for signal sources.

    All signal sources must implement this interface to be used
    in the weighted aggregator.
    """

    @property
    def name(self) -> str:
        """Unique name for this signal source."""
        ...

    @property
    def category(self) -> SignalCategory:
        """Category of this signal source."""
        ...

    @property
    def update_frequency_hours(self) -> float:
        """How often this signal's underlying data updates.

        Used for staleness calculations. For example:
        - Technical indicators: ~0 (updates every candle)
        - On-chain daily metrics: 24
        - Fundamental data: 168 (weekly)
        """
        ...

    def calculate(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SignalOutput:
        """Calculate the signal for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT).
            timeframe: Candle timeframe (e.g., 4h, 1d).
            df: Optional OHLCV DataFrame (required for technical signals).

        Returns:
            SignalOutput with score, confidence, and components.
        """
        ...

    def is_available(self, symbol: str) -> bool:
        """Check if this signal source supports the given symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            True if data is available for this symbol.
        """
        ...


class BaseSignalSource(ABC):
    """Abstract base class for signal sources with common functionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def category(self) -> SignalCategory:
        pass

    @property
    def update_frequency_hours(self) -> float:
        return 0.0  # Default: updates every candle

    @abstractmethod
    def calculate(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SignalOutput:
        pass

    def is_available(self, symbol: str) -> bool:
        return True  # Default: available for all symbols

    def calculate_staleness_factor(
        self,
        timeframe: str,
        data_age_hours: float,
    ) -> float:
        """Calculate staleness decay factor for this signal.

        Args:
            timeframe: Trading timeframe.
            data_age_hours: How old the data is in hours.

        Returns:
            Factor from 0 to 1 to apply to weight (1 = fresh, 0 = stale).
        """
        if self.update_frequency_hours == 0:
            return 1.0  # No staleness for real-time data

        # Calculate how many update cycles old the data is
        cycles_old = data_age_hours / self.update_frequency_hours

        if cycles_old <= 1:
            return 1.0  # Within one update cycle: full weight
        elif cycles_old <= 2:
            return 0.8  # 1-2 cycles: 80% weight
        elif cycles_old <= 3:
            return 0.6  # 2-3 cycles: 60% weight
        else:
            return 0.4  # Older: 40% weight minimum


class TechnicalSignalSource(BaseSignalSource):
    """Signal source that wraps existing technical indicators.

    This provides backwards compatibility with the existing indicator
    system while conforming to the SignalSource protocol.
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """Initialize with optional custom weights.

        Args:
            weights: Weight for each indicator (rsi, macd, bollinger, ema).
                     Defaults to config values.
        """
        self._weights = weights

    @property
    def name(self) -> str:
        return "technical"

    @property
    def category(self) -> SignalCategory:
        return SignalCategory.TECHNICAL

    @property
    def update_frequency_hours(self) -> float:
        return 0.0  # Updates every candle

    def calculate(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SignalOutput:
        """Calculate technical signal from OHLCV data.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            df: OHLCV DataFrame (required).

        Returns:
            SignalOutput with technical analysis score.
        """
        if df is None or len(df) == 0:
            return SignalOutput(score=0.0, confidence=0.0)

        from src.shared.config import get_config
        from .indicators import calculate_indicators, get_latest_indicators

        # Get weights
        config = get_config()
        weights = self._weights or {
            "rsi": config.signal_weights.rsi,
            "macd": config.signal_weights.macd,
            "bollinger": config.signal_weights.bollinger,
            "ema": config.signal_weights.ema,
        }

        # Calculate indicators
        df_with_indicators = calculate_indicators(df)
        indicators = get_latest_indicators(df_with_indicators)

        # Collect component scores
        components = {}
        scores = []
        total_weight = 0.0

        if indicators.rsi_signal is not None:
            components["rsi"] = indicators.rsi_signal
            scores.append(indicators.rsi_signal * weights["rsi"])
            total_weight += weights["rsi"]

        if indicators.macd_score is not None:
            components["macd"] = indicators.macd_score
            scores.append(indicators.macd_score * weights["macd"])
            total_weight += weights["macd"]

        if indicators.bb_score is not None:
            components["bollinger"] = indicators.bb_score
            scores.append(indicators.bb_score * weights["bollinger"])
            total_weight += weights["bollinger"]

        if indicators.ema_score is not None:
            components["ema"] = indicators.ema_score
            scores.append(indicators.ema_score * weights["ema"])
            total_weight += weights["ema"]

        # Calculate aggregate score
        if total_weight == 0:
            score = 0.0
        else:
            score = sum(scores) / total_weight

        # Calculate confidence based on indicator agreement
        confidence = self._calculate_confidence(list(components.values()))

        # Get data timestamp
        data_timestamp = df.index[-1]
        if not isinstance(data_timestamp, datetime):
            data_timestamp = pd.Timestamp(data_timestamp).to_pydatetime()
        if data_timestamp.tzinfo is None:
            data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)

        return SignalOutput(
            score=score,
            confidence=confidence,
            components=components,
            data_timestamp=data_timestamp,
            metadata={
                "rsi_value": indicators.rsi,
                "macd_value": indicators.macd,
                "bb_percent": indicators.bb_percent,
            },
        )

    def _calculate_confidence(self, scores: list[float]) -> float:
        """Calculate confidence based on indicator agreement."""
        if len(scores) < 2:
            return 0.5

        positive = sum(1 for s in scores if s > 0)
        negative = sum(1 for s in scores if s < 0)
        neutral = sum(1 for s in scores if s == 0)

        total = len(scores)

        if positive == total or negative == total:
            return 1.0

        max_agreement = max(positive, negative, neutral)
        return max_agreement / total
