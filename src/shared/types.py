"""Pydantic models for data contracts between agents."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    """Trading signal direction."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @classmethod
    def from_score(cls, score: float) -> "SignalDirection":
        """Convert a score (-100 to 100) to a signal direction."""
        if score >= 60:
            return cls.STRONG_BUY
        elif score >= 20:
            return cls.BUY
        elif score > -20:
            return cls.NEUTRAL
        elif score > -60:
            return cls.SELL
        else:
            return cls.STRONG_SELL


class OHLCV(BaseModel):
    """Single candlestick data point."""

    model_config = {"frozen": True}

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorValues(BaseModel):
    """Container for all calculated indicator values."""

    # RSI
    rsi: Optional[float] = None
    rsi_signal: Optional[float] = Field(
        None, description="RSI contribution to score (-100 to 100)"
    )

    # MACD
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_score: Optional[float] = Field(
        None, description="MACD contribution to score (-100 to 100)"
    )

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_percent: Optional[float] = Field(
        None, description="Price position within bands (0-1)"
    )
    bb_score: Optional[float] = Field(
        None, description="Bollinger contribution to score (-100 to 100)"
    )

    # EMA
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_score: Optional[float] = Field(
        None, description="EMA contribution to score (-100 to 100)"
    )


class Signal(BaseModel):
    """Trading signal output."""

    model_config = {"frozen": True}

    symbol: str
    timeframe: str
    timestamp: datetime
    direction: SignalDirection
    score: float = Field(..., ge=-100, le=100)
    indicators: IndicatorValues
    price: float
    confidence: float = Field(default=1.0, ge=0, le=1)


class SignalResult(BaseModel):
    """Result of signal generation for a single symbol/timeframe."""

    symbol: str
    timeframe: str
    signal: Signal
    raw_data_points: int = Field(..., description="Number of candles used")
    calculation_time_ms: float = Field(..., description="Time taken to calculate")
