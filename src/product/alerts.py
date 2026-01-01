"""Notification system for Discord and Telegram alerts."""

import json
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.shared.config import get_config
from src.shared.types import Signal, SignalDirection

from .formatters import format_discord_message, format_telegram_message


class AlertManager:
    """Manages sending alerts via Discord and Telegram."""

    def __init__(self):
        """Initialize the alert manager."""
        self.config = get_config()
        self._last_alerts: dict[str, datetime] = {}  # symbol -> last alert time

    def should_alert(self, signal: Signal) -> bool:
        """Check if an alert should be sent for this signal.

        Args:
            signal: The signal to check.

        Returns:
            True if alert should be sent.
        """
        # Check if signal meets threshold
        if signal.direction == SignalDirection.STRONG_BUY:
            if signal.score < self.config.alerts.strong_buy_threshold:
                return False
        elif signal.direction == SignalDirection.STRONG_SELL:
            if signal.score > self.config.alerts.strong_sell_threshold:
                return False
        elif signal.direction in (SignalDirection.NEUTRAL, SignalDirection.BUY, SignalDirection.SELL):
            # Only alert on strong signals by default
            return False

        # Check cooldown
        key = f"{signal.symbol}_{signal.timeframe}"
        last_alert = self._last_alerts.get(key)
        if last_alert:
            cooldown = self.config.alerts.cooldown_seconds
            elapsed = (datetime.now(timezone.utc) - last_alert).total_seconds()
            if elapsed < cooldown:
                return False

        return True

    def send_alert(self, signal: Signal) -> dict[str, bool]:
        """Send alert for a signal to all configured channels.

        Args:
            signal: The signal to alert.

        Returns:
            Dict mapping channel name to success status.
        """
        results = {}

        if self.config.notifications.discord.enabled:
            results["discord"] = self._send_discord(signal)

        if self.config.notifications.telegram.enabled:
            results["telegram"] = self._send_telegram(signal)

        # Update last alert time if any succeeded
        if any(results.values()):
            key = f"{signal.symbol}_{signal.timeframe}"
            self._last_alerts[key] = datetime.now(timezone.utc)

        return results

    def _send_discord(self, signal: Signal) -> bool:
        """Send alert to Discord webhook.

        Args:
            signal: The signal to send.

        Returns:
            True if successful.
        """
        webhook_url = self.config.notifications.discord.webhook_url
        if not webhook_url:
            return False

        try:
            payload = format_discord_message(signal)

            with httpx.Client() as client:
                response = client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            return True

        except Exception as e:
            print(f"Discord alert failed: {e}")
            return False

    def _send_telegram(self, signal: Signal) -> bool:
        """Send alert to Telegram.

        Args:
            signal: The signal to send.

        Returns:
            True if successful.
        """
        bot_token = self.config.notifications.telegram.bot_token
        chat_id = self.config.notifications.telegram.chat_id

        if not bot_token or not chat_id:
            return False

        try:
            message = format_telegram_message(signal)
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

            with httpx.Client() as client:
                response = client.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                    },
                )
                response.raise_for_status()

            return True

        except Exception as e:
            print(f"Telegram alert failed: {e}")
            return False


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def send_signal_alert(signal: Signal) -> dict[str, bool]:
    """Convenience function to send an alert.

    Args:
        signal: The signal to alert.

    Returns:
        Dict mapping channel name to success status.
    """
    manager = get_alert_manager()

    if not manager.should_alert(signal):
        return {}

    return manager.send_alert(signal)

