# Crypto Trading Signals Tracker

A Python-based cryptocurrency trading signals application that combines technical analysis, AI sentiment analysis, and on-chain metrics to generate buy/sell/short signals.

## Features

- **Technical Indicators**: RSI, MACD, Bollinger Bands, EMA Crossovers
- **Signal Aggregation**: Weighted scoring system (-100 to +100)
- **CLI Dashboard**: Rich terminal interface for viewing signals
- **Notifications**: Discord and Telegram alerts (optional)
- **Backtesting**: Validate strategies against historical data

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python -m src.product.cli signals --symbol BTCUSDT --timeframe 4h
```

## Project Structure

```
trading_signals/
├── src/
│   ├── data/           # Data fetching and caching
│   ├── research/       # Indicators and signal generation
│   ├── execution/      # Risk management and order simulation
│   ├── product/        # CLI and notifications
│   └── shared/         # Shared types and configuration
├── tests/              # Unit and regression tests
├── config.yaml         # User configuration
└── requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- Symbols to track
- Timeframes
- Indicator parameters
- Alert thresholds
- Notification settings

## Usage

```bash
# Get current signals for BTC on 4h timeframe
python -m src.product.cli signals --symbol BTCUSDT --timeframe 4h

# Get signals for multiple symbols
python -m src.product.cli signals --symbol BTCUSDT --symbol ETHUSDT

# Run in watch mode (refresh every N seconds)
python -m src.product.cli watch --interval 300
```

## License

MIT

