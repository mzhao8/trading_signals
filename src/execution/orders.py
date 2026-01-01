"""Order generation and management."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from src.shared.types import Signal, SignalDirection


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class Order:
    """Represents a trading order.

    Attributes:
        id: Unique order identifier.
        symbol: Trading pair symbol.
        side: Order side (buy/sell).
        order_type: Order type.
        size: Order size in base currency.
        price: Limit price (None for market orders).
        stop_price: Stop trigger price.
        status: Current order status.
        created_at: Order creation timestamp.
        filled_at: Fill timestamp.
        filled_price: Actual fill price.
        signal_score: Original signal score that generated this order.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    signal_score: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": self.size,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "signal_score": self.signal_score,
        }


class OrderGenerator:
    """Generates orders from signals."""

    def __init__(
        self,
        min_score_to_trade: float = 60.0,
        default_size: float = 0.01,  # Default position size
        use_stop_loss: bool = True,
        stop_loss_percent: float = 0.02,  # 2% stop loss
        use_take_profit: bool = True,
        take_profit_percent: float = 0.04,  # 4% take profit (2:1 R:R)
    ):
        """Initialize the order generator.

        Args:
            min_score_to_trade: Minimum absolute score to generate order.
            default_size: Default position size.
            use_stop_loss: Whether to generate stop loss orders.
            stop_loss_percent: Stop loss distance as percent.
            use_take_profit: Whether to generate take profit orders.
            take_profit_percent: Take profit distance as percent.
        """
        self.min_score = min_score_to_trade
        self.default_size = default_size
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_percent
        self.use_take_profit = use_take_profit
        self.take_profit_pct = take_profit_percent

    def generate_orders(
        self,
        signal: Signal,
        size: Optional[float] = None,
    ) -> list[Order]:
        """Generate orders from a signal.

        Args:
            signal: Trading signal.
            size: Optional position size override.

        Returns:
            List of orders (entry + optional stop/take profit).
        """
        orders = []
        size = size or self.default_size

        # Check if signal is strong enough
        if abs(signal.score) < self.min_score:
            return orders

        # Determine order side
        if signal.direction in (SignalDirection.STRONG_BUY, SignalDirection.BUY):
            side = OrderSide.BUY
            stop_price = signal.price * (1 - self.stop_loss_pct)
            take_profit_price = signal.price * (1 + self.take_profit_pct)
        elif signal.direction in (SignalDirection.STRONG_SELL, SignalDirection.SELL):
            side = OrderSide.SELL
            stop_price = signal.price * (1 + self.stop_loss_pct)
            take_profit_price = signal.price * (1 - self.take_profit_pct)
        else:
            # Neutral signal - no orders
            return orders

        # Create entry order
        entry_order = Order(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            size=size,
            signal_score=signal.score,
        )
        orders.append(entry_order)

        # Create stop loss order
        if self.use_stop_loss:
            stop_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            stop_order = Order(
                symbol=signal.symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS,
                size=size,
                stop_price=stop_price,
                signal_score=signal.score,
            )
            orders.append(stop_order)

        # Create take profit order
        if self.use_take_profit:
            tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            tp_order = Order(
                symbol=signal.symbol,
                side=tp_side,
                order_type=OrderType.TAKE_PROFIT,
                size=size,
                price=take_profit_price,
                signal_score=signal.score,
            )
            orders.append(tp_order)

        return orders

    def format_orders(self, orders: list[Order]) -> str:
        """Format orders as readable text.

        Args:
            orders: List of orders.

        Returns:
            Formatted string.
        """
        if not orders:
            return "No orders to display."

        lines = ["Generated Orders:", "-" * 40]

        for order in orders:
            if order.order_type == OrderType.MARKET:
                line = f"  {order.side.value.upper()} {order.size} {order.symbol} @ MARKET"
            elif order.order_type == OrderType.STOP_LOSS:
                line = f"  STOP LOSS: {order.side.value.upper()} @ ${order.stop_price:,.2f}"
            elif order.order_type == OrderType.TAKE_PROFIT:
                line = f"  TAKE PROFIT: {order.side.value.upper()} @ ${order.price:,.2f}"
            else:
                line = f"  {order.order_type.value}: {order.side.value} {order.size}"

            lines.append(line)

        return "\n".join(lines)

