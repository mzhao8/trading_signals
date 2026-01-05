"""Quant Research domain - Indicators, signals, backtesting, and sentiment."""

from .indicators import calculate_indicators, get_latest_indicators
from .signals import generate_signals, generate_signal, SignalResult
from .backtest import Backtester, BacktestResult, STRATEGY_PRESETS, TradingMode
from .reports import generate_backtest_report
from .sentiment import SentimentAnalyzer, get_symbol_sentiment, SentimentResult
from .analysis import analyze_backtest, BacktestAnalysis
from .sources import (
    SignalSource,
    SignalOutput,
    SignalCategory,
    BaseSignalSource,
    TechnicalSignalSource,
)
from .onchain import GlassnodeSignalSource, get_onchain_signal
from .regime import MarketRegime, RegimeResult, detect_regime, get_regime_emoji, get_regime_color
from .aggregator import (
    WeightedAggregator,
    AggregatedSignal,
    WeightProfile,
    create_default_aggregator,
)

__all__ = [
    "calculate_indicators",
    "get_latest_indicators",
    "generate_signals",
    "generate_signal",
    "SignalResult",
    "Backtester",
    "BacktestResult",
    "STRATEGY_PRESETS",
    "TradingMode",
    "generate_backtest_report",
    "SentimentAnalyzer",
    "get_symbol_sentiment",
    "SentimentResult",
    "analyze_backtest",
    "BacktestAnalysis",
    # Signal source abstraction
    "SignalSource",
    "SignalOutput",
    "SignalCategory",
    "BaseSignalSource",
    "TechnicalSignalSource",
    "GlassnodeSignalSource",
    "get_onchain_signal",
    # Regime detection
    "MarketRegime",
    "RegimeResult",
    "detect_regime",
    "get_regime_emoji",
    "get_regime_color",
    # Weighted aggregation
    "WeightedAggregator",
    "AggregatedSignal",
    "WeightProfile",
    "create_default_aggregator",
]
