# Crypto Trading Signals Tracker

A Python-based cryptocurrency trading signals application that combines technical analysis, AI sentiment analysis, and on-chain metrics to generate buy/sell/short signals.

## Features

- **Technical Indicators**: RSI, MACD, Bollinger Bands, EMA Crossovers
- **On-Chain Metrics**: Exchange flows, MVRV Z-Score, SOPR, Active Addresses (BTC/ETH via Glassnode)
- **Market Regime Detection**: Auto-detect bull/bear/sideways markets
- **Signal Aggregation**: Regime-aware weighted scoring system (-100 to +100)
- **Multi-Exchange Support**: Uses CCXT for reliable data from multiple exchanges
- **CLI Dashboard**: Rich terminal interface for viewing signals
- **Backtesting**: Validate strategies against historical data with AI analysis
- **Notifications**: Discord and Telegram alerts (optional)

## Quick Start

```bash
# Clone and navigate to project
cd trading_signals

# Create virtual environment (requires Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python -m src.product.cli signals --symbol BTCUSDT --timeframe 4h
```

## CLI Usage

The CLI provides several commands for analyzing crypto markets. All commands are run via:

```bash
python -m src.product.cli <command> [options]
```

### Get Current Signals

Fetch and display trading signals for one or more symbols:

```bash
# Single symbol
python -m src.product.cli signals --symbol BTCUSDT --timeframe 4h

# Multiple symbols
python -m src.product.cli signals -s BTCUSDT -s ETHUSDT -s SOLUSDT

# Multiple timeframes
python -m src.product.cli signals -s BTCUSDT -t 1h -t 4h -t 1d

# Output as JSON (for scripting)
python -m src.product.cli signals -s BTCUSDT -t 4h --json
```

**Output:**

```
                            Trading Signals
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━┓
┃ Symbol  ┃ Timeframe ┃ Direction ┃ Score ┃      Price ┃ Confid. ┃  RSI ┃   MACD ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━┩
│ BTCUSDT │ 4h        │ NEUTRAL   │  -1.8 │ $88,229.75 │     50% │ 51.7 │ 18.76  │
└─────────┴───────────┴───────────┴───────┴────────────┴─────────┴──────┴────────┘
```

### Detailed Analysis

Get in-depth analysis for a single symbol with all indicator values:

```bash
python -m src.product.cli detail BTCUSDT --timeframe 4h
```

**Output:**

```
╭──────────────────────────────── BTCUSDT (4h) ────────────────────────────────╮
│  NEUTRAL  Score: -1.8                                                        │
│  Price: $88,229.75                                                           │
│  Confidence: 50%                                                             │
│  Time: 2026-01-01 20:00 UTC                                                  │
│  RSI:-2 | MACD:+3 | BB:-9 | EMA:+0                                           │
╰──────────────────────────────────────────────────────────────────────────────╯

                          Indicator Details
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Indicator    ┃               Value ┃ Signal Score ┃ Interpretation ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ RSI (14)     │                51.7 │         -1.7 │ Neutral        │
│ MACD         │             18.7608 │         +3.1 │ Bullish        │
│ Bollinger %B │                0.59 │         -8.8 │ Neutral        │
│ EMA (9/21)   │ 88091.22 / 88081.73 │         +0.2 │ Bullish        │
└──────────────┴─────────────────────┴──────────────┴────────────────┘
```

For BTC/ETH, the detail command also shows on-chain metrics from Glassnode:

```bash
# Requires GLASSNODE_API_KEY in .env
python -m src.product.cli detail BTCUSDT --timeframe 1d
```

### Aggregated Multi-Source Signal

Get a combined signal from multiple sources (technical + on-chain) with automatic market regime detection:

```bash
# Full aggregated analysis for BTC
python -m src.product.cli aggregate BTCUSDT --timeframe 1d

# Disable regime detection (use default weights)
python -m src.product.cli aggregate BTCUSDT --no-regime
```

**Output:**

```
╭────────────────────────────── Regime Detection ──────────────────────────────╮
│  Market Regime: ↔️ SIDEWAYS                                                   │
│     Confidence: 72%                                                          │
│ Trend Strength: +10.7                                                        │
│     Volatility: 12% (percentile)                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
                   Signal Sources
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ Score ┃ Weight ┃ Weighted ┃ Status   ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Technical │ -29.1 │   0.45 │    -13.1 │ ✓ active │
│ Onchain   │ +21.1 │   0.15 │     +3.2 │ ✓ active │
└───────────┴───────┴────────┴──────────┴──────────┘

╭─────────────────────── 📊 Aggregated Signal: BTCUSDT ────────────────────────╮
│  Direction: NEUTRAL                                                          │
│      Score: -16.5                                                            │
│ Confidence: 75%                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**Note:** On-chain signals are only available for BTC and ETH. Other symbols will use technical analysis only.

### Watch Mode

Monitor signals in real-time with auto-refresh:

```bash
# Refresh every 60 seconds (default)
python -m src.product.cli watch --symbol BTCUSDT

# Custom refresh interval (300 seconds = 5 minutes)
python -m src.product.cli watch -s BTCUSDT -s ETHUSDT --interval 300

# Press Ctrl+C to stop
```

### Run Backtest

Test your strategy against historical data. The backtester supports both pre-built strategy presets and fully custom configurations.

#### Using Strategy Presets

```bash
# List available strategies
python -m src.product.cli backtest --list-strategies

# Use a preset strategy
python -m src.product.cli backtest BTCUSDT --strategy long_strong_signals -t 1d -d 365
python -m src.product.cli backtest BTCUSDT --strategy trend_following -t 4h -d 90
```

**Available Presets:**

| Preset                 | Description                                   | Mode          | Entry            | Exit            |
| ---------------------- | --------------------------------------------- | ------------- | ---------------- | --------------- |
| `long_strong_signals`  | Long only on STRONG_BUY, exit on STRONG_SELL  | long_only     | score >= 60      | score <= -60    |
| `short_strong_signals` | Short only on STRONG_SELL, exit on STRONG_BUY | short_only    | score <= -60     | score >= 60     |
| `bidirectional_strong` | Both directions, strong signals only          | bidirectional | abs(score) >= 60 | opposite strong |
| `trend_following`      | Enter on any buy/sell, hold until opposite    | bidirectional | abs(score) >= 20 | opposite signal |
| `mean_reversion`       | Enter on extreme signals, exit at neutral     | bidirectional | abs(score) >= 80 | score crosses 0 |

#### Custom Configuration

```bash
# Long-only strategy with custom thresholds
python -m src.product.cli backtest BTCUSDT --mode long_only --long-entry 50 --long-exit -40

# Bidirectional with exit on any opposite signal
python -m src.product.cli backtest BTCUSDT --mode bidirectional --long-entry 40 --exit-opposite
```

#### Compare Strategies

Compare multiple strategies side-by-side:

```bash
python -m src.product.cli backtest BTCUSDT --compare long_strong_signals,trend_following,mean_reversion -t 1d -d 180
```

**Output:**

```
      Strategy Comparison: BTCUSDT (1d, 180 days)
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric        ┃ long_strong_signals ┃ trend_following ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Total Return  │              +0.00% │          +2.45% │
│ Total Trades  │                   0 │               2 │
│ Win Rate      │                0.0% │           50.0% │
│ Profit Factor │                0.00 │            1.27 │
│ Max Drawdown  │               0.00% │           9.21% │
│ Sharpe Ratio  │                0.00 │            1.85 │
│ Final Capital │             $10,000 │         $10,245 │
└───────────────┴─────────────────────┴─────────────────┘
```

#### Basic Backtest

```bash
# Backtest BTC on daily timeframe for 1 year
python -m src.product.cli backtest BTCUSDT --timeframe 1d --days 365

# Save report to file
python -m src.product.cli backtest BTCUSDT -t 1d -d 365 --output reports/btc_backtest.md
```

**Output:**

```
           Backtest Results: BTCUSDT
            (long_strong_signals)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric          ┃               Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Strategy        │  long_strong_signals│
│ Period          │ 2025-01-01 to ...   │
│ Initial Capital │          $10,000.00 │
│ Final Capital   │          $12,450.00 │
│ Total Return    │             +24.50% │
│ Total Trades    │                  42 │
│ Win Rate        │               54.8% │
│ Profit Factor   │                1.85 │
│ Max Drawdown    │              12.30% │
│ Sharpe Ratio    │                1.42 │
└─────────────────┴─────────────────────┘
```

### Show Configuration

Display current configuration settings:

```bash
python -m src.product.cli config-show
```

### Command Reference

| Command       | Description                             |
| ------------- | --------------------------------------- |
| `signals`     | Get current trading signals for symbols |
| `detail`      | Detailed analysis for a single symbol   |
| `watch`       | Real-time monitoring with auto-refresh  |
| `backtest`    | Run backtest on historical data         |
| `config-show` | Display current configuration           |

### Options Reference

**General Options:**

| Option        | Short | Description                                    |
| ------------- | ----- | ---------------------------------------------- |
| `--symbol`    | `-s`  | Trading pair (e.g., BTCUSDT, ETH/USDT)         |
| `--timeframe` | `-t`  | Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w) |
| `--interval`  | `-i`  | Refresh interval in seconds (watch mode)       |
| `--days`      | `-d`  | Number of days to backtest                     |
| `--output`    | `-o`  | Output file path for reports                   |
| `--json`      | `-j`  | Output as JSON                                 |

**Backtest Options:**

| Option              | Short | Description                                              |
| ------------------- | ----- | -------------------------------------------------------- |
| `--strategy`        | `-S`  | Use a pre-built strategy preset                          |
| `--list-strategies` |       | List available strategy presets                          |
| `--compare`         |       | Compare multiple strategies (comma-separated)            |
| `--mode`            | `-m`  | Trading mode: long_only, short_only, bidirectional       |
| `--long-entry`      |       | Score threshold to enter long position                   |
| `--long-exit`       |       | Score threshold to exit long position                    |
| `--short-entry`     |       | Score threshold to enter short position                  |
| `--short-exit`      |       | Score threshold to exit short position                   |
| `--exit-opposite`   |       | Exit on any opposite signal instead of waiting threshold |

## Signal Interpretation

The system generates a score from -100 to +100:

| Score Range | Direction       | Interpretation                        |
| ----------- | --------------- | ------------------------------------- |
| +60 to +100 | **STRONG_BUY**  | Very bullish, consider long position  |
| +20 to +60  | **BUY**         | Bullish, favorable conditions         |
| -20 to +20  | **NEUTRAL**     | No clear direction, stay flat         |
| -60 to -20  | **SELL**        | Bearish, unfavorable conditions       |
| -100 to -60 | **STRONG_SELL** | Very bearish, consider short position |

### Indicators Used

1. **RSI (14)**: Relative Strength Index - Oversold (<30) is bullish, overbought (>70) is bearish
2. **MACD (12/26/9)**: Trend momentum - Positive histogram is bullish
3. **Bollinger Bands (20, 2)**: Volatility - Price at lower band is bullish
4. **EMA Crossover (9/21)**: Trend direction - Fast above slow is bullish

## API Keys Setup

Create a `.env` file in the project root for API keys:

```bash
# Required for on-chain metrics (BTC/ETH)
GLASSNODE_API_KEY=your_glassnode_api_key

# Optional: For AI-powered backtest analysis
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Glassnode** provides institutional-grade on-chain data. Sign up at https://studio.glassnode.com/

The `.env` file is git-ignored for security.

## Configuration

Edit `config.yaml` to customize:

```yaml
# Symbols to track
symbols:
  - BTCUSDT
  - ETHUSDT

# Timeframes to analyze
timeframes:
  - 4h
  - 1d

# Indicator settings
indicators:
  rsi:
    period: 14
    oversold: 30
    overbought: 70
  macd:
    fast: 12
    slow: 26
    signal: 9

# Signal weights (must sum to 1.0)
signal_weights:
  rsi: 0.25
  macd: 0.25
  bollinger: 0.25
  ema: 0.25
```

## Project Structure

```
trading_signals/
├── src/
│   ├── data/           # Data fetching (CCXT) and caching
│   ├── research/       # Indicators, signals, backtesting
│   ├── execution/      # Risk management, order simulation
│   ├── product/        # CLI and notifications
│   └── shared/         # Types and configuration
├── tests/              # Unit and regression tests
├── config.yaml         # User configuration
└── requirements.txt    # Dependencies
```

## Supported Exchanges

The system uses CCXT and automatically tries these exchanges in order:

1. Binance US
2. Kraken
3. Coinbase
4. KuCoin

If one exchange is unavailable, it automatically falls back to the next.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_indicators.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Development

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check (optional)
mypy src/
```

## Disclaimer

This software is for educational purposes only. It is not financial advice. Trading cryptocurrencies involves significant risk of loss. Always do your own research and never invest more than you can afford to lose.

## License

MIT
