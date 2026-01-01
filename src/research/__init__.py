"""Quant Research domain - Indicators, signals, backtesting, and sentiment."""

from .indicators import calculate_indicators, get_latest_indicators
from .signals import generate_signals, generate_signal, SignalResult
from .backtest import Backtester, BacktestResult
from .reports import generate_backtest_report
from .sentiment import SentimentAnalyzer, get_symbol_sentiment, SentimentResult

__all__ = [
    "calculate_indicators",
    "get_latest_indicators",
    "generate_signals",
    "generate_signal",
    "SignalResult",
    "Backtester",
    "BacktestResult",
    "generate_backtest_report",
    "SentimentAnalyzer",
    "get_symbol_sentiment",
    "SentimentResult",
]
