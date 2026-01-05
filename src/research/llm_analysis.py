"""LLM-powered backtest analysis using Anthropic Claude."""

import os
from typing import Optional

from .analysis import BacktestAnalysis
from .backtest import BacktestResult


def get_llm_commentary(
    result: BacktestResult,
    analysis: BacktestAnalysis,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Get AI-powered commentary on backtest results.

    Uses Claude to provide nuanced analysis and insights beyond
    the rule-based analysis.

    Args:
        result: Backtest result.
        analysis: Rule-based analysis.
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).

    Returns:
        LLM commentary as string, or None if API unavailable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    # Build context for LLM
    context = f"""You are a quantitative trading analyst reviewing backtest results.

BACKTEST RESULTS:
- Symbol: {result.symbol}
- Timeframe: {result.timeframe}
- Period: {result.start_date.date()} to {result.end_date.date()}
- Strategy: {result.strategy_name}
- Total Return: {result.total_return:+.2f}%
- Total Trades: {result.total_trades}
- Win Rate: {result.win_rate:.1f}%
- Profit Factor: {result.profit_factor:.2f}
- Max Drawdown: {result.max_drawdown:.2f}%
- Sharpe Ratio: {result.sharpe_ratio():.2f}

MARKET CONDITIONS:
- Market Regime: {analysis.regime.regime.value}
- Price Change: {analysis.regime.price_change_pct:+.1f}%
- Average Signal Score: {analysis.regime.avg_score:.1f}
- Bullish Periods: {analysis.regime.bullish_periods_pct:.0f}%
- Bearish Periods: {analysis.regime.bearish_periods_pct:.0f}%

COMPARISON:
- Buy & Hold Return: {analysis.buy_hold.buy_hold_return:+.2f}%
- Strategy Outperformance: {analysis.buy_hold.outperformance:+.2f}%

TRADE STATISTICS:
- Trades per Month: {analysis.trades.avg_trades_per_month:.1f}
- Avg Holding Period: {analysis.trades.avg_holding_period_days:.1f} days
- Max Win Streak: {analysis.trades.win_streak_max}
- Max Lose Streak: {analysis.trades.lose_streak_max}

RULE-BASED ASSESSMENT:
- Performance Grade: {analysis.grade.value}
- Strategy Fit Score: {analysis.strategy_fit_score:.0f}/100

ISSUES IDENTIFIED:
{chr(10).join(f"- [{i.severity}] {i.message}" for i in analysis.issues) or "None"}
"""

    prompt = """Based on the backtest results above, provide a brief (2-3 paragraph) analysis that:

1. Explains WHY the strategy performed the way it did in plain English
2. Identifies any subtle issues not captured in the rule-based analysis
3. Suggests 1-2 specific, actionable improvements

Be direct and practical. Avoid generic advice. Focus on insights specific to this data.

Keep your response under 200 words."""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\n{prompt}",
                }
            ],
        )

        return message.content[0].text

    except Exception as e:
        # Silently fail - LLM analysis is optional
        return None


def format_llm_commentary(commentary: str) -> str:
    """Format LLM commentary for display.

    Args:
        commentary: Raw LLM response.

    Returns:
        Formatted string for CLI display.
    """
    # Clean up and format
    lines = commentary.strip().split("\n")
    formatted = []

    for line in lines:
        line = line.strip()
        if line:
            formatted.append(line)

    return "\n\n".join(formatted)

