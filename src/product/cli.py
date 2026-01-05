"""CLI dashboard for trading signals."""

from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.data.cache import DataCache
from src.data.fetcher import CCXTFetcher
from src.research.backtest import Backtester, STRATEGY_PRESETS
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
):
    """Get detailed signal analysis for a single symbol."""
    config = get_config()

    with CCXTFetcher() as fetcher:
        cache = DataCache()

        with console.status(f"Analyzing {symbol} {timeframe}..."):
            df = cache.get_or_fetch(fetcher, symbol, timeframe, config.lookback_periods)
            result = generate_signals(df, symbol, timeframe)

    # Print signal card
    card = format_signal_card(result.signal)
    console.print(card)

    # Print indicator details
    ind = result.signal.indicators

    indicator_table = Table(title="Indicator Details", show_header=True)
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


@app.command()
def watch(
    symbol: list[str] = typer.Option(None, "--symbol", "-s", help="Symbol(s) to watch"),
    timeframe: str = typer.Option("4h", "--timeframe", "-t", help="Timeframe"),
    interval: int = typer.Option(
        60, "--interval", "-i", help="Refresh interval in seconds"
    ),
):
    """Watch signals in real-time with auto-refresh."""
    import time

    config = get_config()
    symbols = symbol if symbol else config.symbols

    console.print(f"[bold]Watching {', '.join(symbols)} on {timeframe}[/bold]")
    console.print(
        f"[dim]Refreshing every {interval} seconds. Press Ctrl+C to stop.[/dim]\n"
    )

    try:
        while True:
            with CCXTFetcher() as fetcher:
                cache = DataCache()

                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Symbol", style="bold")
                table.add_column("Direction", justify="center")
                table.add_column("Score", justify="right")
                table.add_column("Price", justify="right")
                table.add_column("Updated")

                for sym in symbols:
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

                console.clear()
                console.print(table)
                console.print(
                    f"\n[dim]Next refresh in {interval}s. Ctrl+C to stop.[/dim]"
                )

            time.sleep(interval)

    except KeyboardInterrupt:
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
):
    """Run backtest on historical data.

    Examples:

        # Use a preset strategy
        backtest BTCUSDT --strategy long_strong_signals

        # Custom configuration
        backtest BTCUSDT --mode long_only --long-entry 60 --long-exit -60

        # Compare strategies
        backtest BTCUSDT --compare long_strong_signals,trend_following
    """
    from pathlib import Path

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
            numeric_vals = [r.total_return if metric_name == "Total Return" else r.sharpe_ratio() for r in results]
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

    summary_table.add_row(
        "Strategy", result.strategy_name
    )
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
