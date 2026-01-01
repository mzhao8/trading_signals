"""Product domain - CLI dashboard and notifications."""

from .cli import app as cli_app
from .formatters import (
    format_signal_card,
    format_discord_message,
    format_telegram_message,
    format_plain_text,
)
from .alerts import AlertManager, send_signal_alert, get_alert_manager

__all__ = [
    "cli_app",
    "format_signal_card",
    "format_discord_message",
    "format_telegram_message",
    "format_plain_text",
    "AlertManager",
    "send_signal_alert",
    "get_alert_manager",
]
