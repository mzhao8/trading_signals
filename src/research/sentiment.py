"""AI-powered sentiment analysis using LLM."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.shared.config import get_config


@dataclass
class NewsItem:
    """Represents a news article."""

    title: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    content: Optional[str] = None


@dataclass
class SentimentResult:
    """Result of sentiment analysis.

    Attributes:
        score: Sentiment score from -1 (bearish) to +1 (bullish).
        label: Sentiment label (bullish/bearish/neutral).
        summary: Brief summary of the news.
        key_points: List of key bullish/bearish points.
        confidence: Confidence in the analysis (0 to 1).
    """

    score: float
    label: str
    summary: str
    key_points: list[str]
    confidence: float

    def to_signal_score(self) -> float:
        """Convert sentiment to signal score (-100 to 100)."""
        return self.score * 100 * self.confidence


class CryptoPanicClient:
    """Client for fetching news from CryptoPanic API."""

    BASE_URL = "https://cryptopanic.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the CryptoPanic client.

        Args:
            api_key: API key for CryptoPanic. If None, uses public feed (limited).
        """
        self.api_key = api_key
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def fetch_news(
        self,
        currencies: Optional[list[str]] = None,
        kind: str = "news",  # news, media, all
        filter_type: Optional[str] = None,  # rising, hot, bullish, bearish, important, saved, lol
        limit: int = 10,
    ) -> list[NewsItem]:
        """Fetch news from CryptoPanic.

        Args:
            currencies: List of currency symbols to filter by (e.g., ['BTC', 'ETH']).
            kind: Type of content (news, media, all).
            filter_type: Optional filter (rising, hot, etc.).
            limit: Maximum number of items to fetch.

        Returns:
            List of NewsItem objects.
        """
        params = {
            "kind": kind,
            "public": "true",
        }

        if self.api_key:
            params["auth_token"] = self.api_key

        if currencies:
            params["currencies"] = ",".join(currencies)

        if filter_type:
            params["filter"] = filter_type

        try:
            response = self.client.get(f"{self.BASE_URL}/posts/", params=params)
            response.raise_for_status()
            data = response.json()

            items = []
            for post in data.get("results", [])[:limit]:
                published = datetime.fromisoformat(
                    post["published_at"].replace("Z", "+00:00")
                )
                items.append(
                    NewsItem(
                        title=post.get("title", ""),
                        source=post.get("source", {}).get("title", "Unknown"),
                        published_at=published,
                        url=post.get("url"),
                    )
                )

            return items

        except Exception as e:
            print(f"Failed to fetch news: {e}")
            return []


class SentimentAnalyzer:
    """Analyzes news sentiment using Claude/Anthropic API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the sentiment analyzer.

        Args:
            api_key: Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
        """
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url="https://api.anthropic.com",
                timeout=60.0,
                headers={
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def analyze(
        self,
        news_items: list[NewsItem],
        symbol: str,
    ) -> Optional[SentimentResult]:
        """Analyze sentiment of news items using Claude.

        Args:
            news_items: List of news items to analyze.
            symbol: Cryptocurrency symbol for context.

        Returns:
            SentimentResult or None if analysis fails.
        """
        if not self.api_key:
            print("No Anthropic API key configured. Skipping sentiment analysis.")
            return None

        if not news_items:
            return SentimentResult(
                score=0.0,
                label="neutral",
                summary="No recent news available.",
                key_points=[],
                confidence=0.0,
            )

        # Format news for prompt
        news_text = "\n".join(
            f"- [{item.source}] {item.title}" for item in news_items[:10]
        )

        prompt = f"""Analyze the sentiment of these recent cryptocurrency news headlines for {symbol}:

{news_text}

Provide your analysis in JSON format with these fields:
- score: float from -1 (very bearish) to +1 (very bullish)
- label: "bullish", "bearish", or "neutral"
- summary: one sentence summary of overall sentiment
- key_points: list of 2-3 key bullish or bearish factors
- confidence: float from 0 to 1 indicating confidence in the analysis

Only respond with valid JSON, no other text."""

        try:
            response = self.client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract text content
            text = data.get("content", [{}])[0].get("text", "{}")

            # Parse JSON response
            result = json.loads(text)

            return SentimentResult(
                score=float(result.get("score", 0)),
                label=result.get("label", "neutral"),
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
                confidence=float(result.get("confidence", 0.5)),
            )

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            return None
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return None


def get_symbol_sentiment(
    symbol: str,
    api_key: Optional[str] = None,
) -> Optional[SentimentResult]:
    """Convenience function to get sentiment for a symbol.

    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH').
        api_key: Optional Anthropic API key.

    Returns:
        SentimentResult or None.
    """
    # Strip USDT suffix if present
    base_symbol = symbol.replace("USDT", "").replace("USD", "")

    # Fetch news
    news_client = CryptoPanicClient()
    try:
        news = news_client.fetch_news(currencies=[base_symbol], limit=10)
    finally:
        news_client.close()

    if not news:
        return None

    # Analyze sentiment
    analyzer = SentimentAnalyzer(api_key=api_key)
    try:
        result = analyzer.analyze(news, base_symbol)
    finally:
        analyzer.close()

    return result

