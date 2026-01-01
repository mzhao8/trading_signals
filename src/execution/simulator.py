"""Fee and slippage simulation for realistic backtesting."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class FeeStructure:
    """Trading fee structure.

    Attributes:
        maker_fee: Fee for limit orders that add liquidity (e.g., 0.001 = 0.1%)
        taker_fee: Fee for market orders that remove liquidity (e.g., 0.001 = 0.1%)
    """

    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%

    @classmethod
    def binance_default(cls) -> "FeeStructure":
        """Binance default fee structure."""
        return cls(maker_fee=0.001, taker_fee=0.001)

    @classmethod
    def binance_vip(cls, level: int = 1) -> "FeeStructure":
        """Binance VIP fee structure."""
        vip_fees = {
            0: (0.001, 0.001),
            1: (0.0009, 0.001),
            2: (0.0008, 0.001),
            3: (0.0007, 0.0009),
        }
        maker, taker = vip_fees.get(level, (0.001, 0.001))
        return cls(maker_fee=maker, taker_fee=taker)


@dataclass
class SlippageModel:
    """Slippage model for order execution.

    Slippage occurs when the execution price differs from expected price
    due to market movement, order book depth, etc.

    Attributes:
        base_slippage: Base slippage as fraction (e.g., 0.0001 = 0.01%)
        volume_impact: Additional slippage per unit volume
        volatility_factor: Multiplier for volatility-adjusted slippage
    """

    base_slippage: float = 0.0001  # 0.01% base
    volume_impact: float = 0.0  # Per-volume impact (usually 0 for crypto)
    volatility_factor: float = 1.0  # Multiplier for high volatility

    def calculate_slippage(
        self,
        order_size: float,
        volatility: Optional[float] = None,
    ) -> float:
        """Calculate expected slippage for an order.

        Args:
            order_size: Order size in quote currency.
            volatility: Optional volatility measure (e.g., ATR ratio).

        Returns:
            Expected slippage as a fraction.
        """
        slippage = self.base_slippage

        # Add volume impact
        slippage += self.volume_impact * order_size

        # Adjust for volatility
        if volatility:
            slippage *= self.volatility_factor * volatility

        return slippage


@dataclass
class ExecutionResult:
    """Result of simulated order execution.

    Attributes:
        requested_price: The price at which the order was placed.
        executed_price: The simulated execution price after slippage.
        size: Order size.
        fees: Total fees paid.
        slippage_cost: Cost due to slippage.
        total_cost: Total cost including fees and slippage.
    """

    requested_price: float
    executed_price: float
    size: float
    fees: float
    slippage_cost: float

    @property
    def total_cost(self) -> float:
        """Total cost of execution."""
        return self.fees + self.slippage_cost


class ExecutionSimulator:
    """Simulates order execution with realistic fees and slippage."""

    def __init__(
        self,
        fee_structure: Optional[FeeStructure] = None,
        slippage_model: Optional[SlippageModel] = None,
    ):
        """Initialize the execution simulator.

        Args:
            fee_structure: Fee structure to use. Defaults to Binance default.
            slippage_model: Slippage model to use. Defaults to minimal slippage.
        """
        self.fees = fee_structure or FeeStructure.binance_default()
        self.slippage = slippage_model or SlippageModel()

    def simulate_execution(
        self,
        price: float,
        size: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        volatility: Optional[float] = None,
    ) -> ExecutionResult:
        """Simulate order execution.

        Args:
            price: Requested execution price.
            size: Order size in base currency.
            side: Order side (buy/sell).
            order_type: Order type (market/limit).
            volatility: Optional volatility for slippage calculation.

        Returns:
            ExecutionResult with simulated execution details.
        """
        # Calculate fees
        if order_type == OrderType.MARKET:
            fee_rate = self.fees.taker_fee
        else:
            fee_rate = self.fees.maker_fee

        notional = price * size
        fees = notional * fee_rate

        # Calculate slippage
        slippage_rate = self.slippage.calculate_slippage(notional, volatility)

        if side == OrderSide.BUY:
            # Slippage works against us when buying (price goes up)
            executed_price = price * (1 + slippage_rate)
        else:
            # Slippage works against us when selling (price goes down)
            executed_price = price * (1 - slippage_rate)

        slippage_cost = abs(executed_price - price) * size

        return ExecutionResult(
            requested_price=price,
            executed_price=executed_price,
            size=size,
            fees=fees,
            slippage_cost=slippage_cost,
        )

    def calculate_breakeven_move(
        self,
        entry_price: float,
        order_type: OrderType = OrderType.MARKET,
    ) -> float:
        """Calculate the price move needed to break even after fees.

        Args:
            entry_price: Entry price.
            order_type: Order type for fee calculation.

        Returns:
            Required price move as a fraction (e.g., 0.002 = 0.2%).
        """
        if order_type == OrderType.MARKET:
            fee_rate = self.fees.taker_fee
        else:
            fee_rate = self.fees.maker_fee

        # Need to cover entry + exit fees + slippage on both sides
        total_fee = 2 * fee_rate
        total_slippage = 2 * self.slippage.base_slippage

        return total_fee + total_slippage


def calculate_volatility(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate volatility as ATR ratio.

    Args:
        df: OHLCV DataFrame.
        period: ATR period.

    Returns:
        Series of volatility values (ATR / Close).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR
    atr = tr.rolling(window=period).mean()

    # Volatility ratio
    return atr / close

