"""Configuration loading and validation."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RSIConfig(BaseModel):
    """RSI indicator configuration."""

    period: int = 14
    oversold: float = 30
    overbought: float = 70


class MACDConfig(BaseModel):
    """MACD indicator configuration."""

    fast: int = 12
    slow: int = 26
    signal: int = 9


class BollingerConfig(BaseModel):
    """Bollinger Bands configuration."""

    period: int = 20
    std_dev: float = 2.0


class EMAConfig(BaseModel):
    """EMA crossover configuration."""

    fast: int = 9
    slow: int = 21


class IndicatorsConfig(BaseModel):
    """All indicator configurations."""

    rsi: RSIConfig = Field(default_factory=RSIConfig)
    macd: MACDConfig = Field(default_factory=MACDConfig)
    bollinger: BollingerConfig = Field(default_factory=BollingerConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)


class SignalWeightsConfig(BaseModel):
    """Signal scoring weights."""

    rsi: float = 0.25
    macd: float = 0.25
    bollinger: float = 0.25
    ema: float = 0.25


class AlertsConfig(BaseModel):
    """Alert threshold configuration."""

    strong_buy_threshold: float = 60
    strong_sell_threshold: float = -60
    cooldown_seconds: int = 3600


class DiscordConfig(BaseModel):
    """Discord notification configuration."""

    enabled: bool = False
    webhook_url: str = ""


class TelegramConfig(BaseModel):
    """Telegram notification configuration."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class NotificationsConfig(BaseModel):
    """All notification configurations."""

    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    directory: str = "data"
    max_age_hours: int = 1


class BinanceAPIConfig(BaseModel):
    """Binance API configuration."""

    base_url: str = "https://api.binance.com"
    rate_limit_per_minute: int = 1200


class APIConfig(BaseModel):
    """API configurations."""

    binance: BinanceAPIConfig = Field(default_factory=BinanceAPIConfig)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    initial_capital: float = 10000.0
    position_size: float = 1.0
    fee_rate: float = 0.001

    mode: str = "bidirectional"

    long_entry_score: float = 60.0
    short_entry_score: float = -60.0
    long_exit_score: float = -60.0
    short_exit_score: float = 60.0

    exit_on_opposite: bool = False
    default_strategy: Optional[str] = None


class Config(BaseSettings):
    """Main application configuration."""

    model_config = {"env_prefix": "TRADING_"}

    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: list[str] = Field(default_factory=lambda: ["4h", "1d"])
    lookback_periods: int = 200

    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    signal_weights: SignalWeightsConfig = Field(default_factory=SignalWeightsConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.

    Returns:
        Loaded configuration object.
    """
    if config_path is None:
        # Look for config.yaml in current directory or parent directories
        current = Path.cwd()
        for path in [current, *current.parents]:
            candidate = path / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        # Return default config if no file found
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config(**data)


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
