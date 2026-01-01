"""Risk management utilities."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PositionSize:
    """Position sizing result.

    Attributes:
        size: Position size in base currency.
        risk_amount: Dollar amount at risk.
        stop_distance: Distance to stop loss.
        risk_reward_ratio: Expected risk/reward ratio.
    """

    size: float
    risk_amount: float
    stop_distance: float
    risk_reward_ratio: Optional[float] = None


class RiskManager:
    """Manages position sizing and risk limits."""

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,  # 2% per trade
        max_portfolio_risk: float = 0.10,  # 10% total portfolio risk
        max_position_size: float = 0.25,  # 25% of portfolio in single position
    ):
        """Initialize the risk manager.

        Args:
            max_risk_per_trade: Maximum risk per trade as fraction of portfolio.
            max_portfolio_risk: Maximum total portfolio risk.
            max_position_size: Maximum single position size as fraction.
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_price: float,
        target_price: Optional[float] = None,
    ) -> PositionSize:
        """Calculate position size based on risk parameters.

        Uses fixed fractional position sizing where the position size is
        determined by the maximum risk per trade and the distance to stop loss.

        Args:
            portfolio_value: Total portfolio value.
            entry_price: Planned entry price.
            stop_price: Stop loss price.
            target_price: Optional target price for R:R calculation.

        Returns:
            PositionSize with calculated values.
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        stop_percent = stop_distance / entry_price

        # Maximum risk amount
        max_risk = portfolio_value * self.max_risk_per_trade

        # Position size based on risk
        if stop_percent > 0:
            risk_based_size = max_risk / stop_distance
        else:
            risk_based_size = 0

        # Maximum position size based on portfolio limit
        max_position = (portfolio_value * self.max_position_size) / entry_price

        # Use the smaller of risk-based and max position
        position_size = min(risk_based_size, max_position)

        # Calculate actual risk amount
        actual_risk = position_size * stop_distance

        # Calculate risk/reward ratio
        rr_ratio = None
        if target_price and stop_distance > 0:
            target_distance = abs(target_price - entry_price)
            rr_ratio = target_distance / stop_distance

        return PositionSize(
            size=position_size,
            risk_amount=actual_risk,
            stop_distance=stop_distance,
            risk_reward_ratio=rr_ratio,
        )

    def validate_trade(
        self,
        portfolio_value: float,
        current_exposure: float,
        proposed_size: float,
        entry_price: float,
    ) -> tuple[bool, str]:
        """Validate if a trade meets risk requirements.

        Args:
            portfolio_value: Total portfolio value.
            current_exposure: Current total exposure.
            proposed_size: Proposed position size.
            entry_price: Entry price.

        Returns:
            Tuple of (is_valid, reason).
        """
        proposed_notional = proposed_size * entry_price

        # Check position size limit
        position_fraction = proposed_notional / portfolio_value
        if position_fraction > self.max_position_size:
            return False, f"Position too large: {position_fraction:.1%} > {self.max_position_size:.1%}"

        # Check total exposure
        new_exposure = current_exposure + proposed_notional
        exposure_fraction = new_exposure / portfolio_value
        if exposure_fraction > self.max_portfolio_risk * 10:  # Allow 10x risk as exposure
            return False, f"Total exposure too high: {exposure_fraction:.1%}"

        return True, "OK"


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0,
    side: str = "long",
) -> float:
    """Calculate stop loss using ATR.

    Args:
        entry_price: Entry price.
        atr: Average True Range value.
        multiplier: ATR multiplier for stop distance.
        side: Position side ("long" or "short").

    Returns:
        Stop loss price.
    """
    stop_distance = atr * multiplier

    if side == "long":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def calculate_take_profit(
    entry_price: float,
    stop_price: float,
    risk_reward: float = 2.0,
    side: str = "long",
) -> float:
    """Calculate take profit based on risk/reward ratio.

    Args:
        entry_price: Entry price.
        stop_price: Stop loss price.
        risk_reward: Target risk/reward ratio.
        side: Position side ("long" or "short").

    Returns:
        Take profit price.
    """
    risk = abs(entry_price - stop_price)
    reward = risk * risk_reward

    if side == "long":
        return entry_price + reward
    else:
        return entry_price - reward


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Calculate Kelly Criterion optimal bet size.

    The Kelly Criterion determines the optimal fraction of capital to risk
    to maximize long-term growth.

    Args:
        win_rate: Historical win rate (0 to 1).
        avg_win: Average winning trade return (positive).
        avg_loss: Average losing trade return (positive, represents loss magnitude).

    Returns:
        Optimal fraction of capital to bet (usually use half-Kelly in practice).
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    # Kelly = (p * b - q) / b
    # where p = win rate, q = loss rate, b = win/loss ratio
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate

    kelly = (p * b - q) / b

    # Clip to reasonable range
    return max(0.0, min(kelly, 0.5))

