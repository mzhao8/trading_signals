"""Backtest analysis and strategy evaluation."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from .backtest import BacktestResult, STRATEGY_PRESETS


class MarketRegime(str, Enum):
    """Market regime classification."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"


class PerformanceGrade(str, Enum):
    """Overall performance assessment."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class BuyAndHoldComparison:
    """Buy and hold benchmark comparison."""

    buy_hold_return: float
    strategy_return: float
    outperformance: float  # strategy - buy_hold
    beat_benchmark: bool


@dataclass
class RegimeAnalysis:
    """Market regime analysis for the backtest period."""

    regime: MarketRegime
    avg_score: float
    score_volatility: float
    bullish_periods_pct: float  # % of time score > 0
    bearish_periods_pct: float  # % of time score < 0
    price_change_pct: float  # Overall price change


@dataclass
class TradeAnalysis:
    """Analysis of trading activity."""

    total_trades: int
    avg_trades_per_month: float
    trade_frequency: str  # "very_low", "low", "moderate", "high", "very_high"
    avg_holding_period_days: float
    win_streak_max: int
    lose_streak_max: int


@dataclass
class Issue:
    """An identified issue with the strategy."""

    severity: str  # "critical", "warning", "info"
    category: str  # "frequency", "regime", "risk", "threshold"
    message: str
    suggestion: Optional[str] = None


@dataclass
class Recommendation:
    """A recommendation for improving the strategy."""

    priority: int  # 1 = highest
    action: str
    rationale: str
    cli_example: Optional[str] = None


@dataclass
class BacktestAnalysis:
    """Complete analysis of a backtest result."""

    # Overall assessment
    grade: PerformanceGrade
    grade_rationale: str

    # Comparisons
    buy_hold: BuyAndHoldComparison

    # Market analysis
    regime: RegimeAnalysis

    # Trade analysis
    trades: TradeAnalysis

    # Issues and recommendations
    issues: list[Issue]
    recommendations: list[Recommendation]

    # Strategy fit score (0-100)
    strategy_fit_score: float
    strategy_fit_rationale: str


def calculate_buy_and_hold(df: pd.DataFrame) -> float:
    """Calculate buy and hold return for the period.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        Return as percentage.
    """
    if len(df) < 2:
        return 0.0

    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]

    return ((end_price - start_price) / start_price) * 100


def detect_market_regime(
    df: pd.DataFrame, scores: pd.Series
) -> RegimeAnalysis:
    """Detect the market regime during the backtest period.

    Args:
        df: DataFrame with OHLCV data.
        scores: Series of signal scores.

    Returns:
        RegimeAnalysis with regime classification.
    """
    # Price change
    price_change = calculate_buy_and_hold(df)

    # Score statistics
    avg_score = scores.mean()
    score_std = scores.std()
    bullish_pct = (scores > 0).sum() / len(scores) * 100
    bearish_pct = (scores < 0).sum() / len(scores) * 100

    # Classify regime based on price change and score
    if price_change > 50:
        regime = MarketRegime.STRONG_BULL
    elif price_change > 15:
        regime = MarketRegime.BULL
    elif price_change > -15:
        regime = MarketRegime.SIDEWAYS
    elif price_change > -50:
        regime = MarketRegime.BEAR
    else:
        regime = MarketRegime.STRONG_BEAR

    return RegimeAnalysis(
        regime=regime,
        avg_score=avg_score,
        score_volatility=score_std,
        bullish_periods_pct=bullish_pct,
        bearish_periods_pct=bearish_pct,
        price_change_pct=price_change,
    )


def analyze_trades(result: BacktestResult) -> TradeAnalysis:
    """Analyze trading activity and patterns.

    Args:
        result: Backtest result.

    Returns:
        TradeAnalysis with trade statistics.
    """
    trades = result.trades
    total_trades = len(trades)

    # Calculate period in months
    days = (result.end_date - result.start_date).days
    months = max(days / 30, 1)
    trades_per_month = total_trades / months

    # Classify frequency
    if trades_per_month < 0.5:
        frequency = "very_low"
    elif trades_per_month < 2:
        frequency = "low"
    elif trades_per_month < 5:
        frequency = "moderate"
    elif trades_per_month < 10:
        frequency = "high"
    else:
        frequency = "very_high"

    # Average holding period
    if trades:
        holding_periods = []
        for t in trades:
            if t.exit_time and t.entry_time:
                holding_periods.append((t.exit_time - t.entry_time).days)
        avg_holding = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    else:
        avg_holding = 0

    # Win/lose streaks
    win_streak = 0
    lose_streak = 0
    max_win = 0
    max_lose = 0

    for t in trades:
        if t.pnl and t.pnl > 0:
            win_streak += 1
            lose_streak = 0
            max_win = max(max_win, win_streak)
        elif t.pnl and t.pnl < 0:
            lose_streak += 1
            win_streak = 0
            max_lose = max(max_lose, lose_streak)

    return TradeAnalysis(
        total_trades=total_trades,
        avg_trades_per_month=trades_per_month,
        trade_frequency=frequency,
        avg_holding_period_days=avg_holding,
        win_streak_max=max_win,
        lose_streak_max=max_lose,
    )


def identify_issues(
    result: BacktestResult,
    buy_hold: BuyAndHoldComparison,
    regime: RegimeAnalysis,
    trades: TradeAnalysis,
    strategy_mode: str,
) -> list[Issue]:
    """Identify issues with the strategy performance.

    Args:
        result: Backtest result.
        buy_hold: Buy and hold comparison.
        regime: Market regime analysis.
        trades: Trade analysis.
        strategy_mode: The trading mode used.

    Returns:
        List of identified issues.
    """
    issues = []

    # Issue: No trades
    if trades.total_trades == 0:
        issues.append(
            Issue(
                severity="critical",
                category="frequency",
                message="No trades executed during the backtest period",
                suggestion="Lower entry thresholds or use a less strict strategy preset",
            )
        )

    # Issue: Very low trade frequency
    elif trades.trade_frequency == "very_low":
        issues.append(
            Issue(
                severity="warning",
                category="frequency",
                message=f"Very low trade frequency ({trades.avg_trades_per_month:.1f} trades/month)",
                suggestion="Consider lowering entry thresholds to capture more opportunities",
            )
        )

    # Issue: Strategy mode vs market regime mismatch
    if strategy_mode == "long_only" and regime.regime in (
        MarketRegime.BEAR,
        MarketRegime.STRONG_BEAR,
    ):
        issues.append(
            Issue(
                severity="critical",
                category="regime",
                message=f"Long-only strategy in a {regime.regime.value} market (price changed {regime.price_change_pct:+.1f}%)",
                suggestion="Consider short_moderate_signals or bidirectional strategy for bearish periods",
            )
        )

    if strategy_mode == "short_only" and regime.regime in (
        MarketRegime.BULL,
        MarketRegime.STRONG_BULL,
    ):
        issues.append(
            Issue(
                severity="critical",
                category="regime",
                message=f"Short-only strategy in a {regime.regime.value} market (price changed {regime.price_change_pct:+.1f}%)",
                suggestion="Consider long_moderate_signals or bidirectional strategy for bullish periods",
            )
        )

    # Issue: Significantly underperformed buy and hold
    if buy_hold.outperformance < -20:
        issues.append(
            Issue(
                severity="warning",
                category="performance",
                message=f"Strategy underperformed buy-and-hold by {abs(buy_hold.outperformance):.1f}%",
                suggestion="Review entry/exit thresholds or consider a different strategy approach",
            )
        )

    # Issue: Low win rate with no edge
    if result.total_trades > 5 and result.win_rate < 40:
        issues.append(
            Issue(
                severity="warning",
                category="risk",
                message=f"Low win rate ({result.win_rate:.1f}%) suggests poor signal accuracy",
                suggestion="Signals may be lagging; consider tighter exit conditions",
            )
        )

    # Issue: High drawdown
    if result.max_drawdown > 20:
        issues.append(
            Issue(
                severity="warning",
                category="risk",
                message=f"High maximum drawdown ({result.max_drawdown:.1f}%)",
                suggestion="Consider reducing position size or adding stop-loss logic",
            )
        )

    # Issue: Negative Sharpe ratio
    sharpe = result.sharpe_ratio()
    if sharpe < 0 and trades.total_trades > 3:
        issues.append(
            Issue(
                severity="warning",
                category="risk",
                message=f"Negative Sharpe ratio ({sharpe:.2f}) indicates poor risk-adjusted returns",
                suggestion="Strategy is not compensating for risk taken",
            )
        )

    # Issue: Signals predominantly one direction
    if regime.bullish_periods_pct > 80:
        issues.append(
            Issue(
                severity="info",
                category="signal",
                message=f"Signals were bullish {regime.bullish_periods_pct:.0f}% of the time",
                suggestion="Indicator scoring may be biased; review indicator weights",
            )
        )
    elif regime.bearish_periods_pct > 80:
        issues.append(
            Issue(
                severity="info",
                category="signal",
                message=f"Signals were bearish {regime.bearish_periods_pct:.0f}% of the time",
                suggestion="This matches a bear market; signals are working as expected",
            )
        )

    return issues


def generate_recommendations(
    result: BacktestResult,
    buy_hold: BuyAndHoldComparison,
    regime: RegimeAnalysis,
    trades: TradeAnalysis,
    issues: list[Issue],
    strategy_name: str,
) -> list[Recommendation]:
    """Generate actionable recommendations.

    Args:
        result: Backtest result.
        buy_hold: Buy and hold comparison.
        regime: Market regime analysis.
        trades: Trade analysis.
        issues: Identified issues.
        strategy_name: Name of the strategy used.

    Returns:
        List of recommendations sorted by priority.
    """
    recommendations = []
    symbol = result.symbol
    timeframe = result.timeframe

    # Recommendation: Try different strategy for regime
    if regime.regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR):
        recommendations.append(
            Recommendation(
                priority=1,
                action="Try a short or bidirectional strategy",
                rationale=f"Market was bearish ({regime.price_change_pct:+.1f}%); short strategies would capitalize on this",
                cli_example=f"python -m src.product.cli backtest {symbol} -t {timeframe} --strategy short_moderate_signals",
            )
        )
    elif regime.regime in (MarketRegime.BULL, MarketRegime.STRONG_BULL):
        recommendations.append(
            Recommendation(
                priority=1,
                action="Long strategies should perform well here",
                rationale=f"Market was bullish ({regime.price_change_pct:+.1f}%); ensure thresholds aren't too strict",
                cli_example=f"python -m src.product.cli backtest {symbol} -t {timeframe} --strategy long_any_signals",
            )
        )

    # Recommendation: Lower thresholds if no trades
    if trades.total_trades == 0:
        recommendations.append(
            Recommendation(
                priority=1,
                action="Lower entry thresholds",
                rationale=f"Scores ranged from {regime.avg_score - regime.score_volatility*2:.0f} to {regime.avg_score + regime.score_volatility*2:.0f}; current thresholds may be unreachable",
                cli_example=f"python -m src.product.cli backtest {symbol} -t {timeframe} --long-entry 20 --long-exit -20",
            )
        )

    # Recommendation: Compare strategies
    if trades.total_trades < 5:
        recommendations.append(
            Recommendation(
                priority=2,
                action="Compare multiple strategies",
                rationale="Low trade count makes it hard to evaluate; compare several approaches",
                cli_example=f"python -m src.product.cli backtest {symbol} -t {timeframe} --compare long_any_signals,trend_following,bidirectional_moderate",
            )
        )

    # Recommendation: Try trend following in sideways market
    if regime.regime == MarketRegime.SIDEWAYS:
        recommendations.append(
            Recommendation(
                priority=2,
                action="Consider trend_following or mean_reversion",
                rationale="Sideways markets can work with both approaches; test to see which fits better",
                cli_example=f"python -m src.product.cli backtest {symbol} -t {timeframe} --compare trend_following,mean_reversion",
            )
        )

    # Recommendation: Review with chart
    recommendations.append(
        Recommendation(
            priority=3,
            action="Visualize signals on price chart",
            rationale="See how signals aligned with price action",
            cli_example=f"python -m src.product.cli chart {symbol} -t {timeframe}",
        )
    )

    # Sort by priority
    recommendations.sort(key=lambda r: r.priority)

    return recommendations


def calculate_strategy_fit(
    result: BacktestResult,
    regime: RegimeAnalysis,
    strategy_mode: str,
) -> tuple[float, str]:
    """Calculate how well the strategy fits the market conditions.

    Args:
        result: Backtest result.
        regime: Market regime analysis.
        strategy_mode: The trading mode used.

    Returns:
        Tuple of (score 0-100, rationale string).
    """
    score = 50.0  # Start neutral
    reasons = []

    # Mode vs regime alignment
    if strategy_mode == "long_only":
        if regime.regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
            score += 30
            reasons.append("Long strategy aligns with bullish market")
        elif regime.regime == MarketRegime.SIDEWAYS:
            score += 0
            reasons.append("Long strategy in sideways market is neutral")
        else:
            score -= 30
            reasons.append("Long strategy misaligned with bearish market")

    elif strategy_mode == "short_only":
        if regime.regime in (MarketRegime.STRONG_BEAR, MarketRegime.BEAR):
            score += 30
            reasons.append("Short strategy aligns with bearish market")
        elif regime.regime == MarketRegime.SIDEWAYS:
            score += 0
            reasons.append("Short strategy in sideways market is neutral")
        else:
            score -= 30
            reasons.append("Short strategy misaligned with bullish market")

    elif strategy_mode == "bidirectional":
        score += 10
        reasons.append("Bidirectional adapts to any market")

    # Trade frequency adjustment
    if result.total_trades == 0:
        score -= 20
        reasons.append("No trades executed")
    elif result.total_trades < 3:
        score -= 10
        reasons.append("Very few trades for statistical significance")

    # Performance adjustment
    if result.total_return > 0:
        score += min(20, result.total_return / 2)
        reasons.append("Positive returns")
    else:
        score -= min(20, abs(result.total_return) / 2)
        reasons.append("Negative returns")

    # Clamp score
    score = max(0, min(100, score))

    return score, "; ".join(reasons)


def calculate_grade(
    result: BacktestResult,
    buy_hold: BuyAndHoldComparison,
    trades: TradeAnalysis,
) -> tuple[PerformanceGrade, str]:
    """Calculate overall performance grade.

    Args:
        result: Backtest result.
        buy_hold: Buy and hold comparison.
        trades: Trade analysis.

    Returns:
        Tuple of (grade, rationale).
    """
    # Scoring system
    points = 0
    reasons = []

    # Beat buy and hold?
    if buy_hold.beat_benchmark:
        points += 30
        reasons.append(f"beat buy-and-hold by {buy_hold.outperformance:.1f}%")
    elif buy_hold.outperformance > -10:
        points += 15
        reasons.append("close to buy-and-hold")
    else:
        reasons.append(f"underperformed buy-and-hold by {abs(buy_hold.outperformance):.1f}%")

    # Positive return?
    if result.total_return > 20:
        points += 25
        reasons.append(f"strong return ({result.total_return:.1f}%)")
    elif result.total_return > 0:
        points += 15
        reasons.append(f"positive return ({result.total_return:.1f}%)")
    elif result.total_return > -10:
        points += 5
        reasons.append("limited losses")
    else:
        reasons.append(f"significant losses ({result.total_return:.1f}%)")

    # Good win rate?
    if result.total_trades > 3:
        if result.win_rate >= 60:
            points += 20
            reasons.append(f"high win rate ({result.win_rate:.0f}%)")
        elif result.win_rate >= 50:
            points += 10
            reasons.append(f"decent win rate ({result.win_rate:.0f}%)")

    # Risk metrics
    sharpe = result.sharpe_ratio()
    if sharpe > 1:
        points += 15
        reasons.append(f"good risk-adjusted returns (Sharpe {sharpe:.2f})")
    elif sharpe > 0:
        points += 5

    # Low drawdown?
    if result.max_drawdown < 10:
        points += 10
        reasons.append("controlled drawdown")
    elif result.max_drawdown > 25:
        reasons.append("high drawdown risk")

    # Determine grade
    if points >= 80:
        grade = PerformanceGrade.EXCELLENT
    elif points >= 60:
        grade = PerformanceGrade.GOOD
    elif points >= 40:
        grade = PerformanceGrade.FAIR
    elif points >= 20:
        grade = PerformanceGrade.POOR
    else:
        grade = PerformanceGrade.VERY_POOR

    rationale = "; ".join(reasons) if reasons else "insufficient data"

    return grade, rationale.capitalize()


def analyze_backtest(
    result: BacktestResult,
    df: pd.DataFrame,
    scores: pd.Series,
    strategy_mode: str = "bidirectional",
) -> BacktestAnalysis:
    """Perform comprehensive analysis of backtest results.

    Args:
        result: Backtest result to analyze.
        df: Original OHLCV DataFrame.
        scores: Signal scores for each period.
        strategy_mode: The trading mode used.

    Returns:
        Complete BacktestAnalysis.
    """
    # Buy and hold comparison
    buy_hold_return = calculate_buy_and_hold(df)
    buy_hold = BuyAndHoldComparison(
        buy_hold_return=buy_hold_return,
        strategy_return=result.total_return,
        outperformance=result.total_return - buy_hold_return,
        beat_benchmark=result.total_return > buy_hold_return,
    )

    # Market regime
    regime = detect_market_regime(df, scores)

    # Trade analysis
    trades = analyze_trades(result)

    # Issues
    issues = identify_issues(result, buy_hold, regime, trades, strategy_mode)

    # Recommendations
    recommendations = generate_recommendations(
        result, buy_hold, regime, trades, issues, result.strategy_name
    )

    # Strategy fit
    fit_score, fit_rationale = calculate_strategy_fit(result, regime, strategy_mode)

    # Overall grade
    grade, grade_rationale = calculate_grade(result, buy_hold, trades)

    return BacktestAnalysis(
        grade=grade,
        grade_rationale=grade_rationale,
        buy_hold=buy_hold,
        regime=regime,
        trades=trades,
        issues=issues,
        recommendations=recommendations,
        strategy_fit_score=fit_score,
        strategy_fit_rationale=fit_rationale,
    )

