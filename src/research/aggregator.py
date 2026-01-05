"""Weighted signal aggregator with regime-based adjustments.

Combines signals from multiple sources (technical, on-chain, etc.)
using configurable weights that adapt to market regime.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .regime import MarketRegime, RegimeResult, detect_regime
from .sources import SignalCategory, SignalOutput, SignalSource


@dataclass
class WeightProfile:
    """Weight configuration for a specific regime.

    Attributes:
        technical: Weight for technical signals.
        onchain: Weight for on-chain signals.
        fundamental: Weight for fundamental signals.
        supply: Weight for supply signals.
        sentiment: Weight for sentiment signals.
    """

    technical: float = 0.5
    onchain: float = 0.3
    fundamental: float = 0.1
    supply: float = 0.05
    sentiment: float = 0.05

    def __post_init__(self):
        # Normalize weights to sum to 1
        total = (
            self.technical
            + self.onchain
            + self.fundamental
            + self.supply
            + self.sentiment
        )
        if total > 0:
            self.technical /= total
            self.onchain /= total
            self.fundamental /= total
            self.supply /= total
            self.sentiment /= total

    def get_weight(self, category: SignalCategory) -> float:
        """Get weight for a signal category."""
        return {
            SignalCategory.TECHNICAL: self.technical,
            SignalCategory.ONCHAIN: self.onchain,
            SignalCategory.FUNDAMENTAL: self.fundamental,
            SignalCategory.SUPPLY: self.supply,
            SignalCategory.SENTIMENT: self.sentiment,
        }.get(category, 0.0)


# Default weight profiles for each regime
DEFAULT_REGIME_WEIGHTS = {
    MarketRegime.BULL: WeightProfile(
        technical=0.5,  # Technical works well in trends
        onchain=0.25,
        fundamental=0.1,
        supply=0.1,
        sentiment=0.05,
    ),
    MarketRegime.BEAR: WeightProfile(
        technical=0.4,  # On-chain more important in bear markets
        onchain=0.35,  # Watch for accumulation signals
        fundamental=0.1,
        supply=0.1,
        sentiment=0.05,
    ),
    MarketRegime.SIDEWAYS: WeightProfile(
        technical=0.6,  # Mean reversion works in ranges
        onchain=0.2,
        fundamental=0.1,
        supply=0.05,
        sentiment=0.05,
    ),
}


@dataclass
class AggregatedSignal:
    """Result of signal aggregation.

    Attributes:
        score: Final aggregated score (-100 to +100).
        confidence: Confidence in the signal.
        regime: Detected market regime.
        sources: Individual source signals.
        weights_used: Weights applied to each source.
        timestamp: When this was calculated.
    """

    score: float
    confidence: float
    regime: RegimeResult
    sources: dict[str, SignalOutput]
    weights_used: dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def direction(self) -> str:
        """Get signal direction label."""
        if self.score >= 60:
            return "STRONG_BUY"
        elif self.score >= 20:
            return "BUY"
        elif self.score <= -60:
            return "STRONG_SELL"
        elif self.score <= -20:
            return "SELL"
        else:
            return "NEUTRAL"


class WeightedAggregator:
    """Aggregates signals from multiple sources with regime-aware weighting.

    Example:
        >>> from src.research.sources import TechnicalSignalSource
        >>> from src.research.onchain import GlassnodeSignalSource
        >>>
        >>> aggregator = WeightedAggregator()
        >>> aggregator.add_source(TechnicalSignalSource())
        >>> aggregator.add_source(GlassnodeSignalSource())
        >>>
        >>> # Get aggregated signal
        >>> result = aggregator.aggregate("BTCUSDT", "1d", df)
        >>> print(f"Score: {result.score}, Regime: {result.regime.regime}")
    """

    def __init__(
        self,
        regime_weights: Optional[dict[MarketRegime, WeightProfile]] = None,
        auto_detect_regime: bool = True,
    ):
        """Initialize aggregator.

        Args:
            regime_weights: Weight profiles for each regime.
            auto_detect_regime: Whether to auto-detect regime from price data.
        """
        self.sources: list[SignalSource] = []
        self.regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS
        self.auto_detect_regime = auto_detect_regime
        self._forced_regime: Optional[MarketRegime] = None

    def add_source(self, source: SignalSource) -> None:
        """Add a signal source."""
        self.sources.append(source)

    def remove_source(self, name: str) -> bool:
        """Remove a signal source by name."""
        for i, source in enumerate(self.sources):
            if source.name == name:
                self.sources.pop(i)
                return True
        return False

    def set_regime(self, regime: Optional[MarketRegime]) -> None:
        """Force a specific regime (or None to auto-detect)."""
        self._forced_regime = regime

    def _detect_regime(self, df: Optional[pd.DataFrame]) -> RegimeResult:
        """Detect or use forced regime."""
        if self._forced_regime is not None:
            return RegimeResult(
                regime=self._forced_regime,
                confidence=1.0,
                trend_strength=0.0,
                volatility=0.5,
                components={"forced": True},
                timestamp=datetime.now(timezone.utc),
            )

        if df is not None and len(df) >= 50 and self.auto_detect_regime:
            return detect_regime(df)

        # Default to sideways if no data
        return RegimeResult(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            trend_strength=0.0,
            volatility=0.5,
            components={"default": True},
            timestamp=datetime.now(timezone.utc),
        )

    def aggregate(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
        override_weights: Optional[dict[str, float]] = None,
    ) -> AggregatedSignal:
        """Aggregate signals from all sources.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            df: OHLCV DataFrame (required for technical signals and regime detection).
            override_weights: Override weights by source name (ignores regime).

        Returns:
            AggregatedSignal with combined score and details.
        """
        # Detect regime
        regime = self._detect_regime(df)

        # Get weight profile for regime
        weight_profile = self.regime_weights.get(
            regime.regime, DEFAULT_REGIME_WEIGHTS[MarketRegime.SIDEWAYS]
        )

        # Calculate signals from each source
        source_signals: dict[str, SignalOutput] = {}
        weights_used: dict[str, float] = {}
        weighted_scores: list[float] = []
        total_weight: float = 0.0

        for source in self.sources:
            # Check if source is available for this symbol
            if not source.is_available(symbol):
                continue

            try:
                signal = source.calculate(symbol, timeframe, df)

                # Skip sources with no confidence
                if signal.confidence == 0:
                    continue

                source_signals[source.name] = signal

                # Determine weight
                if override_weights and source.name in override_weights:
                    weight = override_weights[source.name]
                else:
                    weight = weight_profile.get_weight(source.category)

                # Apply staleness factor
                if signal.data_timestamp:
                    age_hours = (
                        datetime.now(timezone.utc) - signal.data_timestamp
                    ).total_seconds() / 3600
                    staleness = source.calculate_staleness_factor(timeframe, age_hours)
                    weight *= staleness

                # Apply confidence factor
                weight *= signal.confidence

                weights_used[source.name] = weight
                weighted_scores.append(signal.score * weight)
                total_weight += weight

            except Exception as e:
                # Log error but continue with other sources
                source_signals[source.name] = SignalOutput(
                    score=0.0,
                    confidence=0.0,
                    metadata={"error": str(e)},
                )

        # Calculate aggregate score
        if total_weight > 0:
            aggregate_score = sum(weighted_scores) / total_weight
        else:
            aggregate_score = 0.0

        # Calculate overall confidence
        if source_signals:
            confidences = [s.confidence for s in source_signals.values() if s.confidence > 0]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                # Boost confidence if sources agree
                scores = [s.score for s in source_signals.values() if s.confidence > 0]
                if all(s > 0 for s in scores) or all(s < 0 for s in scores):
                    agreement_boost = 0.2
                else:
                    agreement_boost = 0.0
                overall_confidence = min(avg_confidence + agreement_boost, 1.0)
            else:
                overall_confidence = 0.0
        else:
            overall_confidence = 0.0

        return AggregatedSignal(
            score=aggregate_score,
            confidence=overall_confidence,
            regime=regime,
            sources=source_signals,
            weights_used=weights_used,
        )


def create_default_aggregator(include_onchain: bool = True) -> WeightedAggregator:
    """Create an aggregator with default sources.

    Args:
        include_onchain: Whether to include on-chain signals (requires API key).

    Returns:
        Configured WeightedAggregator.
    """
    from .sources import TechnicalSignalSource

    aggregator = WeightedAggregator()
    aggregator.add_source(TechnicalSignalSource())

    if include_onchain:
        try:
            from .onchain import GlassnodeSignalSource

            aggregator.add_source(GlassnodeSignalSource())
        except Exception:
            pass  # Glassnode not configured

    return aggregator

