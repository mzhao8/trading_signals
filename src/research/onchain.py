"""On-chain signal sources using Glassnode data.

Converts on-chain metrics into trading signals scored from -100 to +100.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.data.glassnode import GlassnodeClient
from .sources import BaseSignalSource, SignalCategory, SignalOutput


@dataclass
class OnchainMetricConfig:
    """Configuration for converting a metric to a signal score."""

    name: str
    weight: float
    # Thresholds for scoring
    strong_bullish: float  # Score = +100 when value >= this
    bullish: float  # Score = +50 when value >= this
    bearish: float  # Score = -50 when value <= this
    strong_bearish: float  # Score = -100 when value <= this
    # Whether higher values are bullish (True) or bearish (False)
    higher_is_bullish: bool = True


# Default configurations for each metric
DEFAULT_METRIC_CONFIGS = {
    "exchange_net_flow": OnchainMetricConfig(
        name="exchange_net_flow",
        weight=0.35,
        # Negative flow = outflow from exchanges = bullish
        strong_bullish=-50000,  # Large outflow
        bullish=-10000,
        bearish=10000,
        strong_bearish=50000,  # Large inflow
        higher_is_bullish=False,  # Negative (outflow) is bullish
    ),
    "mvrv_z_score": OnchainMetricConfig(
        name="mvrv_z_score",
        weight=0.30,
        # Low MVRV = undervalued = bullish
        strong_bullish=-0.5,  # Deep undervaluation
        bullish=0.5,
        bearish=3.0,
        strong_bearish=7.0,  # Extreme overvaluation
        higher_is_bullish=False,  # Lower is bullish
    ),
    "sopr": OnchainMetricConfig(
        name="sopr",
        weight=0.20,
        # SOPR < 1 = selling at loss = capitulation = bullish
        strong_bullish=0.95,  # Deep capitulation
        bullish=0.98,
        bearish=1.02,
        strong_bearish=1.05,  # Taking profits
        higher_is_bullish=False,  # Lower is bullish
    ),
    "active_addresses": OnchainMetricConfig(
        name="active_addresses",
        weight=0.15,
        # More active addresses = more network activity = bullish
        # These are relative thresholds (will be compared to rolling average)
        strong_bullish=1.3,  # 30% above average
        bullish=1.1,  # 10% above average
        bearish=0.9,  # 10% below average
        strong_bearish=0.7,  # 30% below average
        higher_is_bullish=True,
    ),
}


def normalize_to_score(
    value: float,
    config: OnchainMetricConfig,
) -> float:
    """Convert a metric value to a score from -100 to +100.

    Uses linear interpolation between thresholds.

    Args:
        value: The metric value.
        config: Configuration with thresholds.

    Returns:
        Score from -100 to +100.
    """
    if config.higher_is_bullish:
        # Higher values are bullish
        if value >= config.strong_bullish:
            return 100.0
        elif value >= config.bullish:
            # Interpolate between 50 and 100
            ratio = (value - config.bullish) / (config.strong_bullish - config.bullish)
            return 50.0 + ratio * 50.0
        elif value > config.bearish:
            # Interpolate between -50 and 50
            ratio = (value - config.bearish) / (config.bullish - config.bearish)
            return -50.0 + ratio * 100.0
        elif value > config.strong_bearish:
            # Interpolate between -100 and -50
            ratio = (value - config.strong_bearish) / (config.bearish - config.strong_bearish)
            return -100.0 + ratio * 50.0
        else:
            return -100.0
    else:
        # Lower values are bullish (invert the logic)
        if value <= config.strong_bullish:
            return 100.0
        elif value <= config.bullish:
            ratio = (config.bullish - value) / (config.bullish - config.strong_bullish)
            return 50.0 + ratio * 50.0
        elif value < config.bearish:
            ratio = (config.bearish - value) / (config.bearish - config.bullish)
            return -50.0 + ratio * 100.0
        elif value < config.strong_bearish:
            ratio = (config.strong_bearish - value) / (config.strong_bearish - config.bearish)
            return -100.0 + ratio * 50.0
        else:
            return -100.0


class GlassnodeSignalSource(BaseSignalSource):
    """Signal source using Glassnode on-chain metrics.

    NOTE: Only supports BTC and ETH symbols.

    Combines multiple on-chain metrics into a single score:
    - Exchange Net Flow: Coins moving to/from exchanges
    - MVRV Z-Score: Market vs realized value
    - SOPR: Profit/loss of spent outputs
    - Active Addresses: Network activity
    """

    # Symbols that have on-chain data
    SUPPORTED_SYMBOLS = {"BTCUSDT", "BTCUSD", "BTC", "ETHUSDT", "ETHUSD", "ETH"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        metric_configs: Optional[dict[str, OnchainMetricConfig]] = None,
        progress_callback: Optional[callable] = None,
    ):
        """Initialize Glassnode signal source.

        Args:
            api_key: Glassnode API key (defaults to env var).
            metric_configs: Custom metric configurations.
            progress_callback: Optional callback for progress updates.
        """
        self._client: Optional[GlassnodeClient] = None
        self._api_key = api_key
        self._metric_configs = metric_configs or DEFAULT_METRIC_CONFIGS
        self._cache: dict[str, tuple[datetime, dict]] = {}  # symbol -> (timestamp, metrics)
        self._progress_callback = progress_callback

    @property
    def client(self) -> GlassnodeClient:
        """Lazy initialization of Glassnode client."""
        if self._client is None:
            self._client = GlassnodeClient(
                api_key=self._api_key,
                progress_callback=self._progress_callback,
            )
        return self._client

    @property
    def name(self) -> str:
        return "onchain"

    @property
    def category(self) -> SignalCategory:
        return SignalCategory.ONCHAIN

    @property
    def update_frequency_hours(self) -> float:
        return 24.0  # Daily data

    def is_available(self, symbol: str) -> bool:
        """Check if on-chain data is available for this symbol.

        Only BTC and ETH are supported.
        """
        return symbol.upper() in self.SUPPORTED_SYMBOLS

    def _get_metrics(self, symbol: str) -> tuple[dict[str, float], datetime]:
        """Get metrics with caching.

        Returns:
            Tuple of (metrics dict, data timestamp).
        """
        # Check cache
        if symbol in self._cache:
            cache_time, metrics = self._cache[symbol]
            age_hours = (datetime.now(timezone.utc) - cache_time).total_seconds() / 3600
            if age_hours < 1:  # Use cache if less than 1 hour old
                return metrics, cache_time

        # Fetch from API
        metrics = self.client.get_latest_metrics(symbol)
        timestamp = datetime.now(timezone.utc)
        self._cache[symbol] = (timestamp, metrics)

        return metrics, timestamp

    def calculate(
        self,
        symbol: str,
        timeframe: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SignalOutput:
        """Calculate on-chain signal for a symbol.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe (used for staleness calculation).
            df: Optional OHLCV data (not used for on-chain signals).

        Returns:
            SignalOutput with aggregated on-chain score.
        """
        if not self.is_available(symbol):
            return SignalOutput(
                score=0.0,
                confidence=0.0,
                metadata={"error": f"On-chain data not available for {symbol}"},
            )

        try:
            metrics, data_timestamp = self._get_metrics(symbol)
        except Exception as e:
            return SignalOutput(
                score=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
            )

        if not metrics:
            return SignalOutput(
                score=0.0,
                confidence=0.0,
                metadata={"error": "No metrics available"},
            )

        # Calculate individual metric scores
        components = {}
        weighted_scores = []
        total_weight = 0.0

        for metric_name, config in self._metric_configs.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]
            score = normalize_to_score(value, config)
            components[metric_name] = score
            weighted_scores.append(score * config.weight)
            total_weight += config.weight

        # Calculate aggregate score
        if total_weight == 0:
            return SignalOutput(
                score=0.0,
                confidence=0.0,
                metadata={"error": "No valid metrics"},
            )

        aggregate_score = sum(weighted_scores) / total_weight

        # Calculate confidence based on metric agreement
        if len(components) < 2:
            confidence = 0.5
        else:
            scores = list(components.values())
            positive = sum(1 for s in scores if s > 0)
            negative = sum(1 for s in scores if s < 0)
            max_agreement = max(positive, negative)
            confidence = max_agreement / len(scores)

        return SignalOutput(
            score=aggregate_score,
            confidence=confidence,
            components=components,
            data_timestamp=data_timestamp,
            metadata={
                "raw_metrics": metrics,
                "metrics_used": len(components),
                "metrics_available": len(metrics),
            },
        )


def get_onchain_signal(
    symbol: str,
    timeframe: str = "1d",
) -> SignalOutput:
    """Convenience function to get on-chain signal.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT).
        timeframe: Timeframe for staleness calculation.

    Returns:
        SignalOutput with on-chain analysis.
    """
    source = GlassnodeSignalSource()
    return source.calculate(symbol, timeframe)

