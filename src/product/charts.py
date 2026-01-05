"""Price charts with signal heatmap visualization."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from src.data.fetcher import CCXTFetcher
from src.research.indicators import calculate_indicators


def calculate_scores(df: pd.DataFrame) -> pd.Series:
    """Calculate aggregate signal scores for each row.

    Args:
        df: DataFrame with indicator columns.

    Returns:
        Series of signal scores (-100 to +100).
    """
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


def create_signal_heatmap(
    symbol: str = "BTCUSDT",
    timeframe: str = "1d",
    days: int = 90,
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple[int, int] = (14, 8),
) -> Optional[Path]:
    """Create a price chart with signal score heatmap.

    The chart shows:
    - Price line colored by signal score (green=bullish, red=bearish)
    - Background heatmap showing signal strength
    - Score histogram on the right

    Args:
        symbol: Trading pair symbol.
        timeframe: Candle timeframe.
        days: Number of days of data to display.
        output_path: If provided, save chart to this path.
        show: If True, display the chart interactively.
        figsize: Figure size as (width, height).

    Returns:
        Path to saved file if output_path provided, else None.
    """
    # Fetch data
    with CCXTFetcher() as fetcher:
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

    # Calculate indicators and scores
    df_ind = calculate_indicators(df)
    scores = calculate_scores(df_ind)
    df_ind["score"] = scores

    # Create figure with subplots
    fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")

    # Main price chart (larger)
    ax_price = fig.add_axes([0.08, 0.35, 0.75, 0.55])  # [left, bottom, width, height]

    # Score subplot below
    ax_score = fig.add_axes([0.08, 0.1, 0.75, 0.2], sharex=ax_price)

    # Score distribution on right
    ax_hist = fig.add_axes([0.85, 0.35, 0.12, 0.55])

    # Style settings
    bg_color = "#1a1a2e"
    text_color = "#e0e0e0"
    grid_color = "#2d2d44"

    for ax in [ax_price, ax_score, ax_hist]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    # Create colormap (red -> yellow -> green)
    colors_list = ["#ff4444", "#ff6b6b", "#ffd93d", "#6bcf6b", "#44ff44"]
    cmap = mcolors.LinearSegmentedColormap.from_list("signal", colors_list)

    # Normalize scores to 0-1 range for colormap
    norm = mcolors.Normalize(vmin=-60, vmax=60)

    # --- Main Price Chart with Colored Line ---
    dates = df_ind.index.to_numpy()
    prices = df_ind["close"].to_numpy()
    score_values = scores.to_numpy()

    # Create line segments for colored line
    points = np.array([mdates.date2num(dates), prices]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color each segment by score
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(score_values[:-1])  # Color by score
    ax_price.add_collection(lc)

    # Set axis limits
    ax_price.set_xlim(mdates.date2num(dates[0]), mdates.date2num(dates[-1]))
    price_margin = (prices.max() - prices.min()) * 0.05
    ax_price.set_ylim(prices.min() - price_margin, prices.max() + price_margin)

    # Add background heatmap (vertical bars)
    for i in range(len(dates) - 1):
        color = cmap(norm(score_values[i]))
        ax_price.axvspan(
            mdates.date2num(dates[i]),
            mdates.date2num(dates[i + 1]),
            alpha=0.15,
            color=color,
            linewidth=0,
        )

    # Format price axis
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax_price.set_ylabel("Price", color=text_color, fontsize=11)
    ax_price.grid(True, alpha=0.2, color=grid_color)
    ax_price.xaxis.set_visible(False)  # Hide x-axis, shown on score plot

    # Title
    ax_price.set_title(
        f"{symbol} Price with Signal Heatmap ({timeframe})",
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # Add current price annotation
    current_price = prices[-1]
    current_score = score_values[-1]
    score_color = cmap(norm(current_score))
    ax_price.annotate(
        f"${current_price:,.0f}\nScore: {current_score:+.1f}",
        xy=(mdates.date2num(dates[-1]), current_price),
        xytext=(10, 0),
        textcoords="offset points",
        color=score_color,
        fontsize=10,
        fontweight="bold",
        va="center",
    )

    # --- Score Bar Chart ---
    bar_colors = [cmap(norm(s)) for s in score_values]
    ax_score.bar(
        mdates.date2num(dates),
        score_values,
        width=0.8 * (mdates.date2num(dates[1]) - mdates.date2num(dates[0])),
        color=bar_colors,
        alpha=0.8,
    )

    # Add zero line
    ax_score.axhline(y=0, color=text_color, linewidth=0.5, alpha=0.5)

    # Add threshold lines
    ax_score.axhline(y=60, color="#44ff44", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_score.axhline(y=30, color="#6bcf6b", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_score.axhline(y=-30, color="#ff6b6b", linewidth=0.8, linestyle=":", alpha=0.5)
    ax_score.axhline(y=-60, color="#ff4444", linewidth=0.8, linestyle="--", alpha=0.5)

    ax_score.set_ylabel("Signal Score", color=text_color, fontsize=10)
    ax_score.set_ylim(-80, 80)
    ax_score.grid(True, alpha=0.2, color=grid_color)

    # Format x-axis dates
    ax_score.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_score.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_score.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Score Distribution Histogram ---
    n, bins, patches = ax_hist.hist(
        score_values[~np.isnan(score_values)],
        bins=20,
        orientation="horizontal",
        alpha=0.8,
        edgecolor=bg_color,
    )

    # Color histogram bars
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for patch, center in zip(patches, bin_centers):
        patch.set_facecolor(cmap(norm(center)))

    ax_hist.axhline(y=0, color=text_color, linewidth=0.5, alpha=0.5)
    ax_hist.axhline(y=60, color="#44ff44", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_hist.axhline(y=-60, color="#ff4444", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_hist.set_ylim(-80, 80)
    ax_hist.set_xlabel("Count", color=text_color, fontsize=10)
    ax_hist.set_title("Distribution", color=text_color, fontsize=10)
    ax_hist.yaxis.set_visible(False)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.08, 0.02, 0.75, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Signal Score", color=text_color)
    cbar.ax.tick_params(colors=text_color)
    cbar.ax.set_xlabel(
        "← SELL                                                              BUY →",
        color=text_color,
        fontsize=9,
    )

    # Add statistics text
    stats_text = (
        f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\n"
        f"Score Range: {score_values.min():.1f} to {score_values.max():.1f}\n"
        f"Avg Score: {score_values.mean():.1f}"
    )
    fig.text(
        0.88,
        0.15,
        stats_text,
        color=text_color,
        fontsize=9,
        ha="left",
        va="top",
        family="monospace",
    )

    # Layout is manually set via add_axes, skip tight_layout

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=bg_color, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path if output_path else None


def create_multi_timeframe_heatmap(
    symbol: str = "BTCUSDT",
    timeframes: list[str] = ["1h", "4h", "1d"],
    days: int = 30,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Create a multi-timeframe signal heatmap comparison.

    Shows signal scores across multiple timeframes for the same symbol.

    Args:
        symbol: Trading pair symbol.
        timeframes: List of timeframes to compare.
        days: Number of days of data.
        output_path: If provided, save chart to this path.
        show: If True, display the chart interactively.

    Returns:
        Path to saved file if output_path provided, else None.
    """
    fig, axes = plt.subplots(
        len(timeframes), 1, figsize=(14, 3 * len(timeframes)), facecolor="#1a1a2e"
    )

    if len(timeframes) == 1:
        axes = [axes]

    # Style settings
    bg_color = "#1a1a2e"
    text_color = "#e0e0e0"
    grid_color = "#2d2d44"

    # Create colormap
    colors_list = ["#ff4444", "#ff6b6b", "#ffd93d", "#6bcf6b", "#44ff44"]
    cmap = mcolors.LinearSegmentedColormap.from_list("signal", colors_list)
    norm = mcolors.Normalize(vmin=-60, vmax=60)

    with CCXTFetcher() as fetcher:
        for ax, tf in zip(axes, timeframes):
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(grid_color)

            # Fetch data
            if tf == "1d":
                limit = days
            elif tf == "4h":
                limit = days * 6
            elif tf == "1h":
                limit = days * 24
            else:
                limit = min(days * 24, 500)

            df = fetcher.fetch_dataframe(symbol, tf, limit=limit)
            df_ind = calculate_indicators(df)
            scores = calculate_scores(df_ind)

            dates = df_ind.index.to_numpy()
            score_values = scores.to_numpy()

            # Plot score bars
            bar_colors = [cmap(norm(s)) for s in score_values]
            ax.bar(
                mdates.date2num(dates),
                score_values,
                width=0.8 * (mdates.date2num(dates[1]) - mdates.date2num(dates[0])),
                color=bar_colors,
                alpha=0.8,
            )

            # Reference lines
            ax.axhline(y=0, color=text_color, linewidth=0.5, alpha=0.5)
            ax.axhline(y=30, color="#44ff44", linewidth=0.8, linestyle=":", alpha=0.5)
            ax.axhline(y=-30, color="#ff4444", linewidth=0.8, linestyle=":", alpha=0.5)

            ax.set_ylabel(f"{tf}", color=text_color, fontsize=12, fontweight="bold")
            ax.set_ylim(-80, 80)
            ax.grid(True, alpha=0.2, color=grid_color)

            # Format x-axis on bottom plot only
            if ax == axes[-1]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            else:
                ax.xaxis.set_visible(False)

    fig.suptitle(
        f"{symbol} Signal Scores - Multi-Timeframe",
        color=text_color,
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=bg_color, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path if output_path else None
