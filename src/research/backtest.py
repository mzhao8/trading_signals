"""Backtesting engine for strategy validation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd

from src.shared.types import SignalDirection

from .indicators import calculate_indicators


class TradingMode(str, Enum):
    """Trading mode for backtester."""

    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    entry_price: float
    direction: str  # "long" or "short"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    strategy_name: str = "custom"
    trades: list[Trade] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """Total return as a percentage."""
        return (
            (self.final_capital - self.initial_capital) / self.initial_capital
        ) * 100

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl and t.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.pnl and t.pnl < 0)

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def avg_win(self) -> float:
        """Average winning trade PnL percentage."""
        wins = [
            t.pnl_percent for t in self.trades if t.pnl_percent and t.pnl_percent > 0
        ]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade PnL percentage."""
        losses = [
            t.pnl_percent for t in self.trades if t.pnl_percent and t.pnl_percent < 0
        ]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a percentage."""
        if not self.trades:
            return 0.0

        peak = self.initial_capital
        max_dd = 0.0
        capital = self.initial_capital

        for trade in self.trades:
            if trade.pnl:
                capital += trade.pnl
                peak = max(peak, capital)
                dd = (peak - capital) / peak * 100
                max_dd = max(max_dd, dd)

        return max_dd

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio (annualized).

        Assumes daily returns if using daily timeframe.
        """
        if not self.trades:
            return 0.0

        returns = [t.pnl_percent for t in self.trades if t.pnl_percent is not None]
        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0

        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance**0.5

        if std_dev == 0:
            return 0.0

        # Annualize (assume 252 trading days)
        annualized_return = avg_return * 252
        annualized_std = std_dev * (252**0.5)

        return (annualized_return - risk_free_rate) / annualized_std


# Pre-built strategy presets
STRATEGY_PRESETS: dict[str, dict] = {
    "long_strong_signals": {
        "description": "Long only on STRONG_BUY, exit on STRONG_SELL",
        "mode": "long_only",
        "long_entry_score": 60.0,
        "long_exit_score": -60.0,
    },
    "long_moderate_signals": {
        "description": "Long only on BUY signals, exit on SELL",
        "mode": "long_only",
        "long_entry_score": 30.0,
        "long_exit_score": -30.0,
    },
    "long_any_signals": {
        "description": "Long on any bullish signal, exit on bearish",
        "mode": "long_only",
        "long_entry_score": 15.0,
        "long_exit_score": -15.0,
    },
    "short_strong_signals": {
        "description": "Short only on STRONG_SELL, exit on STRONG_BUY",
        "mode": "short_only",
        "short_entry_score": -60.0,
        "short_exit_score": 60.0,
    },
    "short_moderate_signals": {
        "description": "Short only on SELL signals, exit on BUY",
        "mode": "short_only",
        "short_entry_score": -30.0,
        "short_exit_score": 30.0,
    },
    "short_any_signals": {
        "description": "Short on any bearish signal, exit on bullish",
        "mode": "short_only",
        "short_entry_score": -15.0,
        "short_exit_score": 15.0,
    },
    "bidirectional_strong": {
        "description": "Both directions, strong signals only",
        "mode": "bidirectional",
        "long_entry_score": 60.0,
        "short_entry_score": -60.0,
        "long_exit_score": -60.0,
        "short_exit_score": 60.0,
    },
    "bidirectional_moderate": {
        "description": "Both directions, moderate signals",
        "mode": "bidirectional",
        "long_entry_score": 30.0,
        "short_entry_score": -30.0,
        "long_exit_score": -30.0,
        "short_exit_score": 30.0,
    },
    "trend_following": {
        "description": "Enter on any buy/sell, hold until opposite",
        "mode": "bidirectional",
        "long_entry_score": 20.0,
        "short_entry_score": -20.0,
        "exit_on_opposite": True,
    },
    "mean_reversion": {
        "description": "Enter on extreme signals, exit at neutral",
        "mode": "bidirectional",
        "long_entry_score": 80.0,
        "short_entry_score": -80.0,
        "long_exit_score": 0.0,
        "short_exit_score": 0.0,
    },
}


class Backtester:
    """Configurable backtesting engine for signal-based strategies.

    Supports multiple trading modes and customizable entry/exit thresholds.
    Can be configured directly or via pre-built strategy presets.

    Examples:
        # Use a preset
        backtester = Backtester.from_preset("long_strong_signals")

        # Custom configuration
        backtester = Backtester(
            mode="long_only",
            long_entry_score=60,
            long_exit_score=-60,
        )
    """

    def __init__(
        self,
        # Capital settings
        initial_capital: float = 10000.0,
        position_size: float = 1.0,
        fee_rate: float = 0.001,
        # Trading mode
        mode: str = "bidirectional",
        # Entry conditions
        long_entry_score: float = 60.0,
        short_entry_score: float = -60.0,
        # Exit conditions
        long_exit_score: float = -60.0,
        short_exit_score: float = 60.0,
        # Alternative exit mode
        exit_on_opposite: bool = False,
        # Strategy name (for reports)
        strategy_name: str = "custom",
    ):
        """Initialize the backtester.

        Args:
            initial_capital: Starting capital.
            position_size: Fraction of capital to use per trade (0.0 to 1.0).
            fee_rate: Trading fee as decimal (0.001 = 0.1%).
            mode: Trading mode - "long_only", "short_only", or "bidirectional".
            long_entry_score: Minimum score to enter long position.
            short_entry_score: Maximum score to enter short position.
            long_exit_score: Exit long when score falls to this level.
            short_exit_score: Exit short when score rises to this level.
            exit_on_opposite: If True, exit on any opposite signal instead of threshold.
            strategy_name: Name for this strategy (used in reports).
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.fee_rate = fee_rate

        # Validate and set mode
        try:
            self.mode = TradingMode(mode)
        except ValueError:
            valid_modes = [m.value for m in TradingMode]
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        self.long_entry_score = long_entry_score
        self.short_entry_score = short_entry_score
        self.long_exit_score = long_exit_score
        self.short_exit_score = short_exit_score
        self.exit_on_opposite = exit_on_opposite
        self.strategy_name = strategy_name

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        initial_capital: float = 10000.0,
        position_size: float = 1.0,
        fee_rate: float = 0.001,
    ) -> "Backtester":
        """Create a Backtester from a pre-defined strategy preset.

        Args:
            preset_name: Name of the preset (see STRATEGY_PRESETS).
            initial_capital: Starting capital.
            position_size: Fraction of capital per trade.
            fee_rate: Trading fee rate.

        Returns:
            Configured Backtester instance.

        Raises:
            ValueError: If preset_name is not found.
        """
        if preset_name not in STRATEGY_PRESETS:
            available = ", ".join(STRATEGY_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {available}"
            )

        preset = STRATEGY_PRESETS[preset_name].copy()
        # Remove description, it's just for documentation
        preset.pop("description", None)

        return cls(
            initial_capital=initial_capital,
            position_size=position_size,
            fee_rate=fee_rate,
            strategy_name=preset_name,
            **preset,
        )

    @classmethod
    def list_presets(cls) -> dict[str, str]:
        """List available strategy presets with descriptions.

        Returns:
            Dict mapping preset name to description.
        """
        return {
            name: config.get("description", "No description")
            for name, config in STRATEGY_PRESETS.items()
        }

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.

        Returns:
            BacktestResult with performance metrics.
        """
        # Calculate indicators
        df_with_indicators = calculate_indicators(df)

        capital = self.initial_capital
        trades: list[Trade] = []
        current_trade: Optional[Trade] = None

        for i in range(len(df_with_indicators)):
            row = df_with_indicators.iloc[i]
            timestamp = df_with_indicators.index[i]
            price = row["close"]

            # Calculate aggregate score for this row
            score = self._calculate_score(row)

            if current_trade is None:
                # Look for entry
                current_trade = self._check_entry(score, timestamp, price)
            else:
                # Look for exit
                should_exit = self._check_exit(score, current_trade.direction)

                if should_exit:
                    current_trade = self._close_trade(
                        current_trade, timestamp, price, capital
                    )
                    capital += current_trade.pnl or 0
                    trades.append(current_trade)
                    current_trade = None

        # Close any open trade at the end
        if current_trade is not None:
            last_row = df_with_indicators.iloc[-1]
            current_trade = self._close_trade(
                current_trade,
                df_with_indicators.index[-1],
                last_row["close"],
                capital,
            )
            capital += current_trade.pnl or 0
            trades.append(current_trade)

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df_with_indicators.index[0],
            end_date=df_with_indicators.index[-1],
            initial_capital=self.initial_capital,
            final_capital=capital,
            strategy_name=self.strategy_name,
            trades=trades,
        )

    def _check_entry(
        self, score: float, timestamp: datetime, price: float
    ) -> Optional[Trade]:
        """Check if we should enter a position.

        Args:
            score: Current signal score.
            timestamp: Current timestamp.
            price: Current price.

        Returns:
            Trade if entering, None otherwise.
        """
        # Check for long entry
        if self.mode in (TradingMode.LONG_ONLY, TradingMode.BIDIRECTIONAL):
            if score >= self.long_entry_score:
                return Trade(
                    entry_time=timestamp,
                    entry_price=price * (1 + self.fee_rate),
                    direction="long",
                )

        # Check for short entry
        if self.mode in (TradingMode.SHORT_ONLY, TradingMode.BIDIRECTIONAL):
            if score <= self.short_entry_score:
                return Trade(
                    entry_time=timestamp,
                    entry_price=price * (1 - self.fee_rate),
                    direction="short",
                )

        return None

    def _check_exit(self, score: float, direction: str) -> bool:
        """Check if we should exit the current position.

        Args:
            score: Current signal score.
            direction: Current position direction ("long" or "short").

        Returns:
            True if should exit.
        """
        if direction == "long":
            if self.exit_on_opposite:
                # Exit on any negative score
                return score < 0
            else:
                # Exit when score falls to threshold
                return score <= self.long_exit_score

        elif direction == "short":
            if self.exit_on_opposite:
                # Exit on any positive score
                return score > 0
            else:
                # Exit when score rises to threshold
                return score >= self.short_exit_score

        return False

    def _close_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        price: float,
        capital: float,
    ) -> Trade:
        """Close a trade and calculate PnL.

        Args:
            trade: The trade to close.
            timestamp: Exit timestamp.
            price: Exit price.
            capital: Current capital for position sizing.

        Returns:
            Trade with exit details filled in.
        """
        if trade.direction == "long":
            exit_price = price * (1 - self.fee_rate)
            trade.pnl_percent = (
                (exit_price - trade.entry_price) / trade.entry_price * 100
            )
        else:
            exit_price = price * (1 + self.fee_rate)
            trade.pnl_percent = (
                (trade.entry_price - exit_price) / trade.entry_price * 100
            )

        trade.exit_time = timestamp
        trade.exit_price = exit_price

        trade_capital = capital * self.position_size
        trade.pnl = trade_capital * (trade.pnl_percent / 100)

        return trade

    def _calculate_score(self, row: pd.Series) -> float:
        """Calculate aggregate score from a row with indicators."""
        scores = []
        weights = []

        if pd.notna(row.get("rsi_signal")):
            scores.append(row["rsi_signal"])
            weights.append(0.25)

        if pd.notna(row.get("macd_score")):
            scores.append(row["macd_score"])
            weights.append(0.25)

        if pd.notna(row.get("bb_score")):
            scores.append(row["bb_score"])
            weights.append(0.25)

        if pd.notna(row.get("ema_score")):
            scores.append(row["ema_score"])
            weights.append(0.25)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def __repr__(self) -> str:
        """String representation of the backtester."""
        return (
            f"Backtester(strategy='{self.strategy_name}', mode='{self.mode.value}', "
            f"long_entry={self.long_entry_score}, long_exit={self.long_exit_score})"
        )
