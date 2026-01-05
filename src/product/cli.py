"""CLI dashboard for trading signals."""

from datetime import datetime
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.data.cache import DataCache
from src.data.fetcher import CCXTFetcher
from src.research.backtest import Backtester, STRATEGY_PRESETS
from src.research.indicators import calculate_indicators
from src.research.analysis import analyze_backtest
from src.research.reports import generate_backtest_report
from src.research.signals import generate_signals
from src.shared.config import get_config, load_config
from src.shared.types import SignalDirection

from .formatters import format_signal_card

app = typer.Typer(
    name="trading-signals",
    help="Cryptocurrency trading signals tracker",
    add_completion=False,
)
console = Console()


def get_direction_style(direction: SignalDirection) -> str:
    """Get Rich style for signal direction."""
    styles = {
        SignalDirection.STRONG_BUY: "bold green",
        SignalDirection.BUY: "green",
        SignalDirection.NEUTRAL: "yellow",
        SignalDirection.SELL: "red",
        SignalDirection.STRONG_SELL: "bold red",
    }
    return styles.get(direction, "white")


def get_score_color(score: float) -> str:
    """Get color for score value."""
    if score >= 60:
        return "green"
    elif score >= 20:
        return "bright_green"
    elif score > -20:
        return "yellow"
    elif score > -60:
        return "bright_red"
    else:
        return "red"


@app.command()
def signals(
    symbol: list[str] = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Symbol(s) to analyze (e.g., BTCUSDT). Uses config if not specified.",
    ),
    timeframe: list[str] = typer.Option(
        None,
        "--timeframe",
        "-t",
        help="Timeframe(s) to analyze (e.g., 4h, 1d). Uses config if not specified.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON instead of formatted table.",
    ),
):
    """Get current trading signals for specified symbols."""
    config = get_config()

    symbols = symbol if symbol else config.symbols
    timeframes = timeframe if timeframe else config.timeframes

    with CCXTFetcher() as fetcher:
        cache = DataCache()

        results = []

        for sym in symbols:
            for tf in timeframes:
                try:
                    with console.status(f"Fetching {sym} {tf}..."):
                        df = cache.get_or_fetch(
                            fetcher, sym, tf, config.lookback_periods
                        )
                        result = generate_signals(df, sym, tf)
                        results.append(result)
                except Exception as e:
                    console.print(f"[red]Error fetching {sym} {tf}: {e}[/red]")

    if not results:
        console.print("[yellow]No signals generated.[/yellow]")
        return

    if json_output:
        import json

        output = [
            {
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "direction": r.signal.direction.value,
                "score": r.signal.score,
                "price": r.signal.price,
                "confidence": r.signal.confidence,
                "timestamp": r.signal.timestamp.isoformat(),
            }
            for r in results
        ]
        console.print_json(json.dumps(output))
        return

    # Create table
    table = Table(
        title="Trading Signals",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Symbol", style="bold")
    table.add_column("Timeframe")
    table.add_column("Direction", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("MACD", justify="right")

    for r in results:
        sig = r.signal
        direction_text = Text(
            sig.direction.value, style=get_direction_style(sig.direction)
        )
        score_text = Text(f"{sig.score:+.1f}", style=get_score_color(sig.score))

        rsi_val = f"{sig.indicators.rsi:.1f}" if sig.indicators.rsi else "N/A"
        macd_val = f"{sig.indicators.macd:.4f}" if sig.indicators.macd else "N/A"

        table.add_row(
            r.symbol,
            r.timeframe,
            direction_text,
            score_text,
            f"${sig.price:,.2f}",
            f"{sig.confidence:.0%}",
            rsi_val,
            macd_val,
        )

    console.print(table)
    console.print(
        f"\n[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
    )


@app.command()
def detail(
    symbol: str = typer.Argument(..., help="Symbol to analyze (e.g., BTCUSDT)"),
    timeframe: str = typer.Option(
        "4h", "--timeframe", "-t", help="Timeframe to analyze"
    ),
    onchain: bool = typer.Option(
        True,
        "--onchain/--no-onchain",
        help="Include on-chain metrics (BTC/ETH only)",
    ),
):
    """Get detailed signal analysis for a single symbol."""
    from src.research.onchain import GlassnodeSignalSource

    config = get_config()
    onchain_signal = None
    onchain_available = False

    with CCXTFetcher() as fetcher:
        cache = DataCache()

        with console.status(f"Analyzing {symbol} {timeframe}..."):
            df = cache.get_or_fetch(fetcher, symbol, timeframe, config.lookback_periods)
            result = generate_signals(df, symbol, timeframe)

            # Fetch on-chain data if available (BTC/ETH only)
            if onchain:
                onchain_source = GlassnodeSignalSource()
                if onchain_source.is_available(symbol):
                    try:
                        onchain_signal = onchain_source.calculate(symbol, timeframe)
                        if onchain_signal.confidence > 0:
                            onchain_available = True
                    except Exception:
                        pass

    # Calculate scores
    tech_score = result.signal.score
    if onchain_available and onchain_signal:
        onchain_score = onchain_signal.score
        combined_score = (tech_score * 0.6) + (onchain_score * 0.4)
    else:
        onchain_score = None
        combined_score = tech_score

    # Determine direction from combined score
    if combined_score >= 60:
        direction = "STRONG BUY"
        direction_style = "bold green"
    elif combined_score >= 20:
        direction = "BUY"
        direction_style = "green"
    elif combined_score <= -60:
        direction = "STRONG SELL"
        direction_style = "bold red"
    elif combined_score <= -20:
        direction = "SELL"
        direction_style = "red"
    else:
        direction = "NEUTRAL"
        direction_style = "yellow"

    # Build main signal card with COMBINED score
    score_color = get_score_color(combined_score)
    ind = result.signal.indicators

    # Header with combined score
    header_grid = Table.grid(padding=0)
    header_grid.add_column(justify="left")
    header_grid.add_row(f"[{direction_style}]{direction}[/]  Score: [{score_color}]{combined_score:+.1f}[/]")
    header_grid.add_row("")
    header_grid.add_row(f"Price: ${result.signal.price:,.2f}")
    header_grid.add_row(f"Confidence: {int(result.signal.confidence * 100)}%")
    header_grid.add_row(f"Time: {result.signal.timestamp.strftime('%Y-%m-%d %H:%M')} UTC")

    # Score breakdown line
    if onchain_available:
        header_grid.add_row("")
        header_grid.add_row(
            f"[dim]Technical:[/dim] [{get_score_color(tech_score)}]{tech_score:+.1f}[/]  "
            f"[dim]On-Chain:[/dim] [{get_score_color(onchain_score)}]{onchain_score:+.1f}[/]"
        )

    console.print(
        Panel(
            header_grid,
            title=f"{symbol} ({timeframe})",
            border_style=score_color,
        )
    )

    # Technical indicators table
    indicator_table = Table(title="Technical Indicators", show_header=True)
    indicator_table.add_column("Indicator")
    indicator_table.add_column("Value", justify="right")
    indicator_table.add_column("Signal Score", justify="right")
    indicator_table.add_column("Interpretation")

    # RSI
    if ind.rsi is not None:
        rsi_interp = (
            "Oversold" if ind.rsi < 30 else "Overbought" if ind.rsi > 70 else "Neutral"
        )
        indicator_table.add_row(
            "RSI (14)",
            f"{ind.rsi:.1f}",
            f"{ind.rsi_signal:+.1f}" if ind.rsi_signal else "N/A",
            rsi_interp,
        )

    # MACD
    if ind.macd is not None:
        macd_interp = (
            "Bullish" if ind.macd_histogram and ind.macd_histogram > 0 else "Bearish"
        )
        indicator_table.add_row(
            "MACD",
            f"{ind.macd:.4f}",
            f"{ind.macd_score:+.1f}" if ind.macd_score else "N/A",
            macd_interp,
        )

    # Bollinger
    if ind.bb_percent is not None:
        bb_interp = (
            "Oversold"
            if ind.bb_percent < 0.2
            else "Overbought" if ind.bb_percent > 0.8 else "Neutral"
        )
        indicator_table.add_row(
            "Bollinger %B",
            f"{ind.bb_percent:.2f}",
            f"{ind.bb_score:+.1f}" if ind.bb_score else "N/A",
            bb_interp,
        )

    # EMA
    if ind.ema_fast is not None and ind.ema_slow is not None:
        ema_interp = "Bullish" if ind.ema_fast > ind.ema_slow else "Bearish"
        indicator_table.add_row(
            "EMA (9/21)",
            f"{ind.ema_fast:.2f} / {ind.ema_slow:.2f}",
            f"{ind.ema_score:+.1f}" if ind.ema_score else "N/A",
            ema_interp,
        )

    console.print(indicator_table)

    # On-chain metrics table (if available)
    if onchain_available and onchain_signal:
        onchain_table = Table(
            title="On-Chain Metrics (Glassnode)", show_header=True
        )
        onchain_table.add_column("Metric")
        onchain_table.add_column("Value", justify="right")
        onchain_table.add_column("Signal Score", justify="right")
        onchain_table.add_column("Interpretation")

        raw_metrics = onchain_signal.metadata.get("raw_metrics", {})

        # Exchange Net Flow
        if "exchange_net_flow" in onchain_signal.components:
            flow = raw_metrics.get("exchange_net_flow", 0)
            score = onchain_signal.components["exchange_net_flow"]
            interp = "Bullish (outflow)" if flow < 0 else "Bearish (inflow)"
            onchain_table.add_row(
                "Exchange Net Flow",
                f"{flow:,.0f} BTC",
                f"{score:+.1f}",
                interp,
            )

        # MVRV Z-Score
        if "mvrv_z_score" in onchain_signal.components:
            mvrv = raw_metrics.get("mvrv_z_score", 0)
            score = onchain_signal.components["mvrv_z_score"]
            if mvrv < 0:
                interp = "Undervalued"
            elif mvrv < 2:
                interp = "Fair value"
            elif mvrv < 4:
                interp = "Overvalued"
            else:
                interp = "Extremely overvalued"
            onchain_table.add_row(
                "MVRV Z-Score",
                f"{mvrv:.2f}",
                f"{score:+.1f}",
                interp,
            )

        # SOPR
        if "sopr" in onchain_signal.components:
            sopr = raw_metrics.get("sopr", 1)
            score = onchain_signal.components["sopr"]
            if sopr < 1:
                interp = "Selling at loss (capitulation)"
            else:
                interp = "Selling at profit"
            onchain_table.add_row(
                "SOPR",
                f"{sopr:.4f}",
                f"{score:+.1f}",
                interp,
            )

        # Active Addresses
        if "active_addresses" in onchain_signal.components:
            addrs = raw_metrics.get("active_addresses", 0)
            score = onchain_signal.components["active_addresses"]
            onchain_table.add_row(
                "Active Addresses",
                f"{addrs:,.0f}",
                f"{score:+.1f}",
                "Network activity",
            )

        # NUPL (if available in raw metrics)
        if "nupl" in raw_metrics:
            nupl = raw_metrics.get("nupl", 0)
            if nupl < 0:
                interp = "Capitulation"
            elif nupl < 0.25:
                interp = "Hope/Fear"
            elif nupl < 0.5:
                interp = "Optimism"
            elif nupl < 0.75:
                interp = "Belief/Greed"
            else:
                interp = "Euphoria"
            onchain_table.add_row(
                "NUPL",
                f"{nupl:.4f}",
                "[dim]n/a[/dim]",
                interp,
            )

        console.print()
        console.print(onchain_table)
    elif onchain and not onchain_available:
        # Check if it's because the symbol isn't supported
        onchain_source = GlassnodeSignalSource()
        if not onchain_source.is_available(symbol):
            console.print(
                f"\n[dim]On-chain metrics not available for {symbol} (BTC/ETH only)[/dim]"
            )
        else:
            console.print(
                "\n[yellow]On-chain data unavailable or error occurred[/yellow]"
            )


@app.command()
def aggregate(
    symbol: str = typer.Argument("BTCUSDT", help="Symbol to analyze (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe"),
    regime: bool = typer.Option(
        True, "--regime/--no-regime", help="Auto-detect market regime"
    ),
):
    """Get multi-source aggregated signal with regime detection.

    Combines technical indicators with on-chain metrics (BTC/ETH) using
    regime-aware weighting. Shows breakdown of each signal source.
    """
    from src.research.aggregator import create_default_aggregator
    from src.research.regime import get_regime_emoji, get_regime_color

    config = get_config()

    with CCXTFetcher() as fetcher:
        cache = DataCache()

        with console.status(f"Analyzing {symbol} {timeframe}..."):
            df = cache.get_or_fetch(fetcher, symbol, timeframe, config.lookback_periods)

            # Create aggregator with optional on-chain
            aggregator = create_default_aggregator(include_onchain=True)
            aggregator.auto_detect_regime = regime

            result = aggregator.aggregate(symbol, timeframe, df)

    # Regime panel
    regime_result = result.regime
    regime_emoji = get_regime_emoji(regime_result.regime)
    regime_color = get_regime_color(regime_result.regime)

    regime_grid = Table.grid(padding=1)
    regime_grid.add_column(justify="right", style="dim")
    regime_grid.add_column(justify="left")
    regime_grid.add_row("Market Regime:", f"[bold {regime_color}]{regime_emoji} {regime_result.regime.value.upper()}[/]")
    regime_grid.add_row("Confidence:", f"{regime_result.confidence:.0%}")
    regime_grid.add_row("Trend Strength:", f"{regime_result.trend_strength:+.1f}")
    regime_grid.add_row("Volatility:", f"{regime_result.volatility:.0%} (percentile)")

    console.print(Panel(regime_grid, title="Regime Detection", border_style="cyan"))

    # Source signals table
    source_table = Table(title="Signal Sources", show_header=True)
    source_table.add_column("Source")
    source_table.add_column("Score", justify="right")
    source_table.add_column("Weight", justify="right")
    source_table.add_column("Weighted", justify="right")
    source_table.add_column("Status")

    for source_name, signal in result.sources.items():
        weight = result.weights_used.get(source_name, 0)
        weighted = signal.score * weight if signal.confidence > 0 else 0

        if signal.confidence == 0:
            status = "[red]✗ unavailable[/red]"
            score_str = "[dim]n/a[/dim]"
            weight_str = "[dim]n/a[/dim]"
            weighted_str = "[dim]n/a[/dim]"
        else:
            status = "[green]✓ active[/green]"
            score_str = f"[{get_score_color(signal.score)}]{signal.score:+.1f}[/]"
            weight_str = f"{weight:.2f}"
            weighted_str = f"[{get_score_color(weighted)}]{weighted:+.1f}[/]"

        source_table.add_row(
            source_name.title(),
            score_str,
            weight_str,
            weighted_str,
            status,
        )

    console.print(source_table)

    # Final signal panel
    score_color = get_score_color(result.score)
    direction_style = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "NEUTRAL": "yellow",
        "SELL": "red",
        "STRONG_SELL": "bold red",
    }.get(result.direction, "white")

    signal_grid = Table.grid(padding=1)
    signal_grid.add_column(justify="right", style="dim")
    signal_grid.add_column(justify="left")
    signal_grid.add_row(
        "Direction:",
        f"[{direction_style}]{result.direction.replace('_', ' ')}[/]"
    )
    signal_grid.add_row("Score:", f"[bold {score_color}]{result.score:+.1f}[/]")
    signal_grid.add_row("Confidence:", f"{result.confidence:.0%}")

    console.print()
    console.print(
        Panel(
            signal_grid,
            title=f"📊 Aggregated Signal: {symbol}",
            border_style=score_color,
        )
    )


@app.command()
def watch(
    symbol: list[str] = typer.Option(None, "--symbol", "-s", help="Symbol(s) to watch"),
    timeframe: str = typer.Option("4h", "--timeframe", "-t", help="Timeframe"),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Refresh interval in seconds"
    ),
):
    """Watch signals in real-time with auto-refresh."""
    import signal
    import time

    config = get_config()
    symbols = symbol if symbol else config.symbols

    # Flag to handle graceful shutdown
    stop_watching = False

    def handle_interrupt(signum, frame):
        nonlocal stop_watching
        stop_watching = True

    # Register signal handler for Ctrl+C
    original_handler = signal.signal(signal.SIGINT, handle_interrupt)

    console.print(f"[bold]Watching {', '.join(symbols)} on {timeframe}[/bold]")
    console.print(
        f"[dim]Refreshing every {interval} seconds. Press Ctrl+C to stop.[/dim]\n"
    )

    try:
        while not stop_watching:
            with CCXTFetcher() as fetcher:
                cache = DataCache()

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Symbol", style="bold")
                table.add_column("Direction", justify="center")
                table.add_column("Score", justify="right")
                table.add_column("Price", justify="right")
                table.add_column("Updated")

                for sym in symbols:
                    if stop_watching:
                        break
                    try:
                        # Force fresh fetch by invalidating cache
                        cache.invalidate(sym, timeframe)
                        df = cache.get_or_fetch(
                            fetcher, sym, timeframe, config.lookback_periods
                        )
                        result = generate_signals(df, sym, timeframe)
                        sig = result.signal

                        direction_text = Text(
                            sig.direction.value,
                            style=get_direction_style(sig.direction),
                        )
                        score_text = Text(
                            f"{sig.score:+.1f}", style=get_score_color(sig.score)
                        )

                        table.add_row(
                            sym,
                            direction_text,
                            score_text,
                            f"${sig.price:,.2f}",
                            datetime.now().strftime("%H:%M:%S"),
                        )
                    except Exception as e:
                        table.add_row(sym, "[red]ERROR[/red]", str(e)[:20], "", "")

                if not stop_watching:
                    console.clear()
                    console.print(table)
                    console.print(
                        f"\n[dim]Next refresh in {interval}s. Ctrl+C to stop.[/dim]"
                    )

            # Interruptible sleep - check every second
            for _ in range(interval):
                if stop_watching:
                    break
                time.sleep(1)

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        console.print("\n[yellow]Stopped watching.[/yellow]")


@app.command()
def backtest(
    symbol: Optional[str] = typer.Argument(
        None, help="Symbol to backtest (e.g., BTCUSDT)"
    ),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe"),
    days: int = typer.Option(365, "--days", "-d", help="Number of days to backtest"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file for report (markdown)"
    ),
    # Strategy options
    strategy: Optional[str] = typer.Option(
        None,
        "--strategy",
        "-S",
        help="Use a pre-built strategy preset (see --list-strategies)",
    ),
    list_strategies: bool = typer.Option(
        False, "--list-strategies", help="List available strategy presets and exit"
    ),
    compare: Optional[str] = typer.Option(
        None,
        "--compare",
        help="Compare multiple strategies (comma-separated preset names)",
    ),
    # Custom configuration
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Trading mode: long_only, short_only, bidirectional"
    ),
    long_entry: Optional[float] = typer.Option(
        None, "--long-entry", help="Score threshold to enter long position"
    ),
    long_exit: Optional[float] = typer.Option(
        None, "--long-exit", help="Score threshold to exit long position"
    ),
    short_entry: Optional[float] = typer.Option(
        None, "--short-entry", help="Score threshold to enter short position"
    ),
    short_exit: Optional[float] = typer.Option(
        None, "--short-exit", help="Score threshold to exit short position"
    ),
    exit_on_opposite: bool = typer.Option(
        False, "--exit-opposite", help="Exit on any opposite signal"
    ),
    analyze: bool = typer.Option(
        True,
        "--analyze/--no-analyze",
        "-a",
        help="Show strategy analysis and recommendations",
    ),
    ai: bool = typer.Option(
        False, "--ai", help="Include AI-powered commentary (requires ANTHROPIC_API_KEY)"
    ),
):
    """Run backtest on historical data.

    Examples:

        # Use a preset strategy
        backtest BTCUSDT --strategy long_strong_signals

        # Custom configuration
        backtest BTCUSDT --mode long_only --long-entry 60 --long-exit -60

        # Compare strategies
        backtest BTCUSDT --compare long_strong_signals,trend_following

        # Skip analysis
        backtest BTCUSDT --strategy trend_following --no-analyze
    """
    from pathlib import Path
    import pandas as pd

    # Handle --list-strategies
    if list_strategies:
        _show_strategy_presets()
        return

    # Handle --compare
    if compare:
        if not symbol:
            console.print("[red]Error: Symbol required for comparison[/red]")
            raise typer.Exit(1)
        _compare_strategies(symbol, timeframe, days, compare.split(","))
        return

    # Normal backtest - symbol is required
    if not symbol:
        console.print("[red]Error: Symbol argument required[/red]")
        console.print("Usage: backtest BTCUSDT [OPTIONS]")
        console.print("Use --list-strategies to see available presets")
        raise typer.Exit(1)

    # Create backtester
    if strategy:
        # Use preset
        try:
            backtester = Backtester.from_preset(strategy)
            console.print(f"[cyan]Using strategy preset: {strategy}[/cyan]")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Build custom config
        kwargs = {}
        if mode:
            kwargs["mode"] = mode
        if long_entry is not None:
            kwargs["long_entry_score"] = long_entry
        if long_exit is not None:
            kwargs["long_exit_score"] = long_exit
        if short_entry is not None:
            kwargs["short_entry_score"] = short_entry
        if short_exit is not None:
            kwargs["short_exit_score"] = short_exit
        if exit_on_opposite:
            kwargs["exit_on_opposite"] = True

        backtester = Backtester(**kwargs)

    # Fetch data
    with CCXTFetcher() as fetcher:
        with console.status(f"Fetching {days} days of {symbol} {timeframe} data..."):
            # Estimate candles needed
            if timeframe == "1d":
                limit = days
            elif timeframe == "4h":
                limit = days * 6
            elif timeframe == "1h":
                limit = days * 24
            else:
                limit = min(days * 24, 1000)

            df = fetcher.fetch_dataframe(symbol, timeframe, limit=limit)

    console.print(f"[green]Fetched {len(df)} candles[/green]")

    # Run backtest
    with console.status("Running backtest..."):
        result = backtester.run(df, symbol, timeframe)

    # Print summary
    _print_backtest_result(result)

    # Run analysis if requested
    if analyze:
        with console.status("Analyzing strategy performance..."):
            # Calculate indicators and scores for analysis
            df_ind = calculate_indicators(df)
            scores = _calculate_scores_series(df_ind)

            # Determine strategy mode
            strategy_mode = mode or backtester.mode.value

            analysis = analyze_backtest(result, df, scores, strategy_mode)

        _print_analysis(analysis)

        # LLM analysis if requested
        if ai:
            from src.research.llm_analysis import (
                get_llm_commentary,
                format_llm_commentary,
            )

            with console.status("[cyan]Getting AI analysis...[/cyan]"):
                commentary = get_llm_commentary(result, analysis)

            if commentary:
                console.print("\n[bold]🤖 AI Analysis:[/bold]")
                console.print(
                    Panel(
                        format_llm_commentary(commentary),
                        border_style="cyan",
                    )
                )
            else:
                console.print(
                    "\n[dim]AI analysis unavailable. Set ANTHROPIC_API_KEY environment variable.[/dim]"
                )

    if output:
        output_path = Path(output)
        generate_backtest_report(result, output_path)
        console.print(f"\n[green]Report saved to {output_path}[/green]")


def _show_strategy_presets():
    """Display available strategy presets."""
    table = Table(title="Available Strategy Presets", show_header=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Mode")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")

    for name, config in STRATEGY_PRESETS.items():
        mode = config.get("mode", "bidirectional")
        description = config.get("description", "")

        if mode == "long_only":
            entry = f">= {config.get('long_entry_score', 60)}"
            exit_val = f"<= {config.get('long_exit_score', -60)}"
        elif mode == "short_only":
            entry = f"<= {config.get('short_entry_score', -60)}"
            exit_val = f">= {config.get('short_exit_score', 60)}"
        else:
            long_entry = config.get("long_entry_score", 60)
            short_entry = config.get("short_entry_score", -60)
            entry = f"L>={long_entry}, S<={short_entry}"
            if config.get("exit_on_opposite"):
                exit_val = "opposite signal"
            else:
                exit_val = f"L<={config.get('long_exit_score', 0)}, S>={config.get('short_exit_score', 0)}"

        table.add_row(name, description, mode, entry, exit_val)

    console.print(table)
    console.print("\n[dim]Usage: backtest BTCUSDT --strategy <preset_name>[/dim]")


def _compare_strategies(symbol: str, timeframe: str, days: int, strategies: list[str]):
    """Compare multiple strategies side by side."""
    # Validate strategies
    for s in strategies:
        if s not in STRATEGY_PRESETS:
            console.print(f"[red]Error: Unknown strategy '{s}'[/red]")
            console.print(f"Available: {', '.join(STRATEGY_PRESETS.keys())}")
            raise typer.Exit(1)

    # Fetch data once
    with CCXTFetcher() as fetcher:
        with console.status(f"Fetching {days} days of {symbol} {timeframe} data..."):
            if timeframe == "1d":
                limit = days
            elif timeframe == "4h":
                limit = days * 6
            elif timeframe == "1h":
                limit = days * 24
            else:
                limit = min(days * 24, 1000)

            df = fetcher.fetch_dataframe(symbol, timeframe, limit=limit)

    console.print(f"[green]Fetched {len(df)} candles[/green]\n")

    # Run backtests
    results = []
    for strategy_name in strategies:
        with console.status(f"Testing {strategy_name}..."):
            backtester = Backtester.from_preset(strategy_name)
            result = backtester.run(df, symbol, timeframe)
            results.append(result)

    # Create comparison table
    table = Table(
        title=f"Strategy Comparison: {symbol} ({timeframe}, {days} days)",
        show_header=True,
    )
    table.add_column("Metric")
    for s in strategies:
        table.add_column(s, justify="right")

    metrics = [
        ("Total Return", lambda r: f"{r.total_return:+.2f}%"),
        ("Total Trades", lambda r: str(r.total_trades)),
        ("Win Rate", lambda r: f"{r.win_rate:.1f}%"),
        ("Profit Factor", lambda r: f"{r.profit_factor:.2f}"),
        ("Max Drawdown", lambda r: f"{r.max_drawdown:.2f}%"),
        ("Sharpe Ratio", lambda r: f"{r.sharpe_ratio():.2f}"),
        ("Final Capital", lambda r: f"${r.final_capital:,.0f}"),
    ]

    for metric_name, metric_fn in metrics:
        values = [metric_fn(r) for r in results]
        # Highlight best value for return and Sharpe
        if metric_name in ("Total Return", "Sharpe Ratio"):
            numeric_vals = [
                r.total_return if metric_name == "Total Return" else r.sharpe_ratio()
                for r in results
            ]
            best_idx = numeric_vals.index(max(numeric_vals))
            values[best_idx] = f"[bold green]{values[best_idx]}[/bold green]"
        table.add_row(metric_name, *values)

    console.print(table)


def _print_backtest_result(result):
    """Print backtest result summary table."""
    title = f"Backtest Results: {result.symbol}"
    if result.strategy_name != "custom":
        title += f" ({result.strategy_name})"

    summary_table = Table(title=title, show_header=True)
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")

    return_style = "green" if result.total_return > 0 else "red"

    summary_table.add_row("Strategy", result.strategy_name)
    summary_table.add_row(
        "Period", f"{result.start_date.date()} to {result.end_date.date()}"
    )
    summary_table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
    summary_table.add_row("Final Capital", f"${result.final_capital:,.2f}")
    summary_table.add_row(
        "Total Return",
        Text(f"{result.total_return:+.2f}%", style=return_style),
    )
    summary_table.add_row("Total Trades", str(result.total_trades))
    summary_table.add_row("Win Rate", f"{result.win_rate:.1f}%")
    summary_table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    summary_table.add_row("Max Drawdown", f"{result.max_drawdown:.2f}%")
    summary_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio():.2f}")

    console.print(summary_table)


def _calculate_scores_series(df: "pd.DataFrame") -> "pd.Series":
    """Calculate aggregate signal scores for each row.

    Args:
        df: DataFrame with indicator columns.

    Returns:
        Series of signal scores.
    """
    import pandas as pd

    scores = []
    for i in range(len(df)):
        row = df.iloc[i]
        s = []
        w = []

        if pd.notna(row.get("rsi_signal")):
            s.append(row["rsi_signal"])
            w.append(0.25)
        if pd.notna(row.get("macd_score")):
            s.append(row["macd_score"])
            w.append(0.25)
        if pd.notna(row.get("bb_score")):
            s.append(row["bb_score"])
            w.append(0.25)
        if pd.notna(row.get("ema_score")):
            s.append(row["ema_score"])
            w.append(0.25)

        if s:
            score = sum(x * y for x, y in zip(s, w)) / sum(w)
        else:
            score = 0.0
        scores.append(score)

    return pd.Series(scores, index=df.index)


def _print_analysis(analysis: "BacktestAnalysis"):
    """Print the backtest analysis report.

    Args:
        analysis: BacktestAnalysis object.
    """
    from src.research.analysis import PerformanceGrade, MarketRegime

    console.print()

    # Grade styling
    grade_styles = {
        PerformanceGrade.EXCELLENT: ("bold green", "✅"),
        PerformanceGrade.GOOD: ("green", "👍"),
        PerformanceGrade.FAIR: ("yellow", "➖"),
        PerformanceGrade.POOR: ("red", "⚠️"),
        PerformanceGrade.VERY_POOR: ("bold red", "❌"),
    }

    grade_style, grade_icon = grade_styles.get(analysis.grade, ("white", ""))

    # Main analysis panel
    grade_text = Text()
    grade_text.append(f"Performance Grade: ", style="bold")
    grade_text.append(
        f"{analysis.grade.value.upper()} {grade_icon}",
        style=grade_style,
    )
    grade_text.append(f"\n{analysis.grade_rationale}")

    console.print(
        Panel(
            grade_text,
            title="[bold]Strategy Analysis[/bold]",
            border_style="cyan",
        )
    )

    # Buy and hold comparison
    bh = analysis.buy_hold
    bh_table = Table(show_header=False, box=None, padding=(0, 2))
    bh_table.add_column("Metric", style="dim")
    bh_table.add_column("Value")

    bh_style = "green" if bh.beat_benchmark else "red"
    bh_table.add_row("Buy & Hold Return", f"{bh.buy_hold_return:+.2f}%")
    bh_table.add_row("Strategy Return", f"{bh.strategy_return:+.2f}%")
    bh_table.add_row(
        "Outperformance",
        Text(f"{bh.outperformance:+.2f}%", style=bh_style),
    )

    console.print(
        Panel(
            bh_table,
            title="[bold]vs Buy & Hold[/bold]",
            border_style="blue",
        )
    )

    # Market regime
    regime = analysis.regime
    regime_styles = {
        MarketRegime.STRONG_BULL: ("bold green", "🚀"),
        MarketRegime.BULL: ("green", "📈"),
        MarketRegime.SIDEWAYS: ("yellow", "➡️"),
        MarketRegime.BEAR: ("red", "📉"),
        MarketRegime.STRONG_BEAR: ("bold red", "💥"),
    }
    regime_style, regime_icon = regime_styles.get(regime.regime, ("white", ""))

    regime_text = Text()
    regime_text.append(f"Market Regime: ", style="bold")
    regime_text.append(
        f"{regime.regime.value.replace('_', ' ').upper()} {regime_icon}",
        style=regime_style,
    )
    regime_text.append(f"\nPrice Change: {regime.price_change_pct:+.1f}%")
    regime_text.append(f"\nAvg Signal Score: {regime.avg_score:+.1f}")
    regime_text.append(
        f"\nBullish: {regime.bullish_periods_pct:.0f}% | Bearish: {regime.bearish_periods_pct:.0f}%"
    )

    console.print(
        Panel(
            regime_text,
            title="[bold]Market Conditions[/bold]",
            border_style="magenta",
        )
    )

    # Strategy fit
    fit_color = (
        "green"
        if analysis.strategy_fit_score >= 60
        else "yellow" if analysis.strategy_fit_score >= 40 else "red"
    )
    console.print(
        f"\n[bold]Strategy Fit Score:[/bold] [{fit_color}]{analysis.strategy_fit_score:.0f}/100[/{fit_color}]"
    )
    console.print(f"[dim]{analysis.strategy_fit_rationale}[/dim]")

    # Issues
    if analysis.issues:
        console.print("\n[bold]⚠️ Issues Identified:[/bold]")
        for issue in analysis.issues:
            severity_styles = {
                "critical": "bold red",
                "warning": "yellow",
                "info": "dim",
            }
            style = severity_styles.get(issue.severity, "white")
            console.print(f"  [{style}]• {issue.message}[/{style}]")
            if issue.suggestion:
                console.print(f"    [dim]→ {issue.suggestion}[/dim]")

    # Recommendations
    if analysis.recommendations:
        console.print("\n[bold]💡 Recommendations:[/bold]")
        for i, rec in enumerate(analysis.recommendations[:3], 1):  # Top 3
            console.print(f"\n  [cyan]{i}. {rec.action}[/cyan]")
            console.print(f"     [dim]{rec.rationale}[/dim]")
            if rec.cli_example:
                console.print(f"     [green]$ {rec.cli_example}[/green]")


@app.command()
def chart(
    symbol: str = typer.Argument("BTCUSDT", help="Symbol to chart (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe"),
    days: int = typer.Option(90, "--days", "-d", help="Number of days to display"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save chart to file (PNG)"
    ),
    no_show: bool = typer.Option(
        False, "--no-show", help="Don't display chart (use with --output)"
    ),
    multi: bool = typer.Option(
        False, "--multi", "-m", help="Show multi-timeframe comparison (1h, 4h, 1d)"
    ),
):
    """Display price chart with signal score heatmap.

    Shows a visual representation of price action colored by signal strength:
    - Green = bullish signals (positive scores)
    - Red = bearish signals (negative scores)

    Examples:

        # Show BTC daily chart for 90 days
        chart BTCUSDT -t 1d -d 90

        # Save chart to file
        chart BTCUSDT -t 4h -d 30 --output charts/btc_signals.png

        # Multi-timeframe comparison
        chart BTCUSDT --multi -d 30
    """
    from pathlib import Path
    from .charts import create_signal_heatmap, create_multi_timeframe_heatmap

    output_path = Path(output) if output else None

    with console.status(f"Generating chart for {symbol}..."):
        try:
            if multi:
                create_multi_timeframe_heatmap(
                    symbol=symbol,
                    timeframes=["1h", "4h", "1d"],
                    days=days,
                    output_path=output_path,
                    show=not no_show,
                )
            else:
                create_signal_heatmap(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    output_path=output_path,
                    show=not no_show,
                )

            if output_path:
                console.print(f"[green]Chart saved to {output_path}[/green]")

        except Exception as e:
            console.print(f"[red]Error generating chart: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    console.print(
        Panel.fit(
            f"""[bold]Trading Signals Configuration[/bold]

[cyan]Symbols:[/cyan] {', '.join(config.symbols)}
[cyan]Timeframes:[/cyan] {', '.join(config.timeframes)}
[cyan]Lookback Periods:[/cyan] {config.lookback_periods}

[bold]Indicator Settings[/bold]
  RSI: period={config.indicators.rsi.period}, oversold={config.indicators.rsi.oversold}, overbought={config.indicators.rsi.overbought}
  MACD: fast={config.indicators.macd.fast}, slow={config.indicators.macd.slow}, signal={config.indicators.macd.signal}
  Bollinger: period={config.indicators.bollinger.period}, std_dev={config.indicators.bollinger.std_dev}
  EMA: fast={config.indicators.ema.fast}, slow={config.indicators.ema.slow}

[bold]Signal Weights[/bold]
  RSI: {config.signal_weights.rsi:.0%}
  MACD: {config.signal_weights.macd:.0%}
  Bollinger: {config.signal_weights.bollinger:.0%}
  EMA: {config.signal_weights.ema:.0%}

[bold]Alerts[/bold]
  Strong Buy Threshold: {config.alerts.strong_buy_threshold}
  Strong Sell Threshold: {config.alerts.strong_sell_threshold}
""",
            title="config.yaml",
        )
    )


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
