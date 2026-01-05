"""Product domain - CLI dashboard and notifications."""

# Note: cli module is not imported here to avoid RuntimeWarning when running
# python -m src.product.cli. The CLI is always run directly, not imported.

from .formatters import (
    format_signal_card,
    format_discord_message,
    format_telegram_message,
    format_plain_text,
)
from .alerts import AlertManager, send_signal_alert, get_alert_manager
from .charts import create_signal_heatmap, create_multi_timeframe_heatmap

__all__ = [
    "format_signal_card",
    "format_discord_message",
    "format_telegram_message",
    "format_plain_text",
    "AlertManager",
    "send_signal_alert",
    "get_alert_manager",
    "create_signal_heatmap",
    "create_multi_timeframe_heatmap",
]
