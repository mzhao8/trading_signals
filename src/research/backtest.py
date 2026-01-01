"""Backtesting engine for strategy validation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.shared.types import SignalDirection

from .indicators import calculate_indicators


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
    trades: list[Trade] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """Total return as a percentage."""
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

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
        wins = [t.pnl_percent for t in self.trades if t.pnl_percent and t.pnl_percent > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade PnL percentage."""
        losses = [t.pnl_percent for t in self.trades if t.pnl_percent and t.pnl_percent < 0]
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


class Backtester:
    """Simple backtesting engine for signal-based strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 1.0,  # Fraction of capital per trade
        entry_threshold: float = 60.0,  # Minimum score to enter
        exit_threshold: float = 0.0,  # Score to exit (opposite direction)
        fee_rate: float = 0.001,  # 0.1% per trade
    ):
        """Initialize the backtester.

        Args:
            initial_capital: Starting capital.
            position_size: Fraction of capital to use per trade.
            entry_threshold: Minimum absolute score to enter a position.
            exit_threshold: Score threshold for exit (when signal reverses).
            fee_rate: Trading fee as decimal (0.001 = 0.1%).
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.fee_rate = fee_rate

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
                if score >= self.entry_threshold:
                    # Enter long
                    current_trade = Trade(
                        entry_time=timestamp,
                        entry_price=price * (1 + self.fee_rate),  # Include fee
                        direction="long",
                    )
                elif score <= -self.entry_threshold:
                    # Enter short
                    current_trade = Trade(
                        entry_time=timestamp,
                        entry_price=price * (1 - self.fee_rate),  # Include fee
                        direction="short",
                    )
            else:
                # Look for exit
                should_exit = False

                if current_trade.direction == "long" and score <= -self.exit_threshold:
                    should_exit = True
                elif current_trade.direction == "short" and score >= self.exit_threshold:
                    should_exit = True

                if should_exit:
                    # Close trade
                    exit_price = price * (1 - self.fee_rate if current_trade.direction == "long" else 1 + self.fee_rate)
                    current_trade.exit_time = timestamp
                    current_trade.exit_price = exit_price

                    # Calculate PnL
                    if current_trade.direction == "long":
                        current_trade.pnl_percent = ((exit_price - current_trade.entry_price) / current_trade.entry_price) * 100
                    else:
                        current_trade.pnl_percent = ((current_trade.entry_price - exit_price) / current_trade.entry_price) * 100

                    trade_capital = capital * self.position_size
                    current_trade.pnl = trade_capital * (current_trade.pnl_percent / 100)
                    capital += current_trade.pnl

                    trades.append(current_trade)
                    current_trade = None

        # Close any open trade at the end
        if current_trade is not None:
            last_row = df_with_indicators.iloc[-1]
            exit_price = last_row["close"]
            current_trade.exit_time = df_with_indicators.index[-1]
            current_trade.exit_price = exit_price

            if current_trade.direction == "long":
                current_trade.pnl_percent = ((exit_price - current_trade.entry_price) / current_trade.entry_price) * 100
            else:
                current_trade.pnl_percent = ((current_trade.entry_price - exit_price) / current_trade.entry_price) * 100

            trade_capital = capital * self.position_size
            current_trade.pnl = trade_capital * (current_trade.pnl_percent / 100)
            capital += current_trade.pnl

            trades.append(current_trade)

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=df_with_indicators.index[0],
            end_date=df_with_indicators.index[-1],
            initial_capital=self.initial_capital,
            final_capital=capital,
            trades=trades,
        )

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

