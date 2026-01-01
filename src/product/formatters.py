"""Signal formatting utilities for CLI and notifications."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.shared.types import Signal, SignalDirection


def get_direction_emoji(direction: SignalDirection) -> str:
    """Get emoji for signal direction."""
    emojis = {
        SignalDirection.STRONG_BUY: "🟢🟢",
        SignalDirection.BUY: "🟢",
        SignalDirection.NEUTRAL: "🟡",
        SignalDirection.SELL: "🔴",
        SignalDirection.STRONG_SELL: "🔴🔴",
    }
    return emojis.get(direction, "⚪")


def get_direction_color(direction: SignalDirection) -> str:
    """Get color for signal direction."""
    colors = {
        SignalDirection.STRONG_BUY: "bright_green",
        SignalDirection.BUY: "green",
        SignalDirection.NEUTRAL: "yellow",
        SignalDirection.SELL: "red",
        SignalDirection.STRONG_SELL: "bright_red",
    }
    return colors.get(direction, "white")


def format_signal_card(signal: Signal) -> Panel:
    """Format a signal as a Rich panel (signal card).

    Args:
        signal: Signal to format.

    Returns:
        Rich Panel with formatted signal information.
    """
    direction_color = get_direction_color(signal.direction)
    
    # Build the main content
    content_lines = []
    
    # Direction and score
    direction_text = Text()
    direction_text.append(f"{signal.direction.value}", style=f"bold {direction_color}")
    direction_text.append(f"  Score: {signal.score:+.1f}", style=direction_color)
    content_lines.append(direction_text)
    
    content_lines.append(Text())  # Empty line
    
    # Price and confidence
    content_lines.append(Text(f"Price: ${signal.price:,.2f}", style="bold"))
    content_lines.append(Text(f"Confidence: {signal.confidence:.0%}"))
    content_lines.append(Text(f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"))
    
    content_lines.append(Text())  # Empty line
    
    # Indicator summary
    ind = signal.indicators
    indicator_text = Text("Indicators: ", style="dim")
    
    scores = []
    if ind.rsi_signal is not None:
        color = "green" if ind.rsi_signal > 0 else "red" if ind.rsi_signal < 0 else "yellow"
        scores.append(f"[{color}]RSI:{ind.rsi_signal:+.0f}[/{color}]")
    if ind.macd_score is not None:
        color = "green" if ind.macd_score > 0 else "red" if ind.macd_score < 0 else "yellow"
        scores.append(f"[{color}]MACD:{ind.macd_score:+.0f}[/{color}]")
    if ind.bb_score is not None:
        color = "green" if ind.bb_score > 0 else "red" if ind.bb_score < 0 else "yellow"
        scores.append(f"[{color}]BB:{ind.bb_score:+.0f}[/{color}]")
    if ind.ema_score is not None:
        color = "green" if ind.ema_score > 0 else "red" if ind.ema_score < 0 else "yellow"
        scores.append(f"[{color}]EMA:{ind.ema_score:+.0f}[/{color}]")
    
    content_lines.append(Text.from_markup(" | ".join(scores) if scores else "N/A"))
    
    # Create panel
    panel = Panel(
        Group(*content_lines),
        title=f"[bold]{signal.symbol}[/bold] ({signal.timeframe})",
        border_style=direction_color,
        padding=(1, 2),
    )
    
    return panel


def format_discord_message(signal: Signal) -> dict:
    """Format a signal as a Discord webhook message.

    Args:
        signal: Signal to format.

    Returns:
        Dict suitable for Discord webhook JSON payload.
    """
    # Color based on direction (Discord uses decimal colors)
    colors = {
        SignalDirection.STRONG_BUY: 0x00FF00,  # Green
        SignalDirection.BUY: 0x90EE90,  # Light green
        SignalDirection.NEUTRAL: 0xFFFF00,  # Yellow
        SignalDirection.SELL: 0xFFA500,  # Orange
        SignalDirection.STRONG_SELL: 0xFF0000,  # Red
    }
    
    emoji = get_direction_emoji(signal.direction)
    
    embed = {
        "title": f"{emoji} {signal.symbol} - {signal.direction.value}",
        "color": colors.get(signal.direction, 0xFFFFFF),
        "fields": [
            {
                "name": "Score",
                "value": f"{signal.score:+.1f}",
                "inline": True,
            },
            {
                "name": "Price",
                "value": f"${signal.price:,.2f}",
                "inline": True,
            },
            {
                "name": "Timeframe",
                "value": signal.timeframe,
                "inline": True,
            },
            {
                "name": "Confidence",
                "value": f"{signal.confidence:.0%}",
                "inline": True,
            },
        ],
        "footer": {
            "text": f"Trading Signals • {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        },
    }
    
    # Add indicator breakdown
    ind = signal.indicators
    indicator_parts = []
    if ind.rsi is not None:
        indicator_parts.append(f"RSI: {ind.rsi:.1f}")
    if ind.macd is not None:
        indicator_parts.append(f"MACD: {ind.macd:.4f}")
    if ind.bb_percent is not None:
        indicator_parts.append(f"BB%: {ind.bb_percent:.2f}")
    
    if indicator_parts:
        embed["fields"].append({
            "name": "Indicators",
            "value": " | ".join(indicator_parts),
            "inline": False,
        })
    
    return {
        "embeds": [embed],
    }


def format_telegram_message(signal: Signal) -> str:
    """Format a signal as a Telegram message.

    Args:
        signal: Signal to format.

    Returns:
        Formatted string for Telegram (supports HTML).
    """
    emoji = get_direction_emoji(signal.direction)
    
    # Build message
    lines = [
        f"{emoji} <b>{signal.symbol}</b> ({signal.timeframe})",
        "",
        f"<b>Signal:</b> {signal.direction.value}",
        f"<b>Score:</b> {signal.score:+.1f}",
        f"<b>Price:</b> ${signal.price:,.2f}",
        f"<b>Confidence:</b> {signal.confidence:.0%}",
        "",
    ]
    
    # Add indicator details
    ind = signal.indicators
    indicator_lines = []
    if ind.rsi is not None:
        indicator_lines.append(f"RSI: {ind.rsi:.1f}")
    if ind.macd is not None:
        indicator_lines.append(f"MACD: {ind.macd:.4f}")
    if ind.bb_percent is not None:
        indicator_lines.append(f"BB%: {ind.bb_percent:.2f}")
    
    if indicator_lines:
        lines.append("<b>Indicators:</b>")
        lines.extend(indicator_lines)
    
    lines.append("")
    lines.append(f"<i>{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</i>")
    
    return "\n".join(lines)


def format_plain_text(signal: Signal) -> str:
    """Format a signal as plain text.

    Args:
        signal: Signal to format.

    Returns:
        Plain text representation.
    """
    lines = [
        f"=== {signal.symbol} ({signal.timeframe}) ===",
        f"Signal: {signal.direction.value}",
        f"Score: {signal.score:+.1f}",
        f"Price: ${signal.price:,.2f}",
        f"Confidence: {signal.confidence:.0%}",
        f"Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    
    return "\n".join(lines)

