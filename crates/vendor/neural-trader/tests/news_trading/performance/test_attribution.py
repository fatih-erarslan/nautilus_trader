"""Tests for trade attribution system."""

import pytest
from datetime import datetime, timedelta

from src.news_trading.performance.attribution import (
    TradeAttributor,
    SentimentAccuracyTracker,
)
from src.news_trading.performance.models import TradeResult, TradeStatus


class TestTradeAttribution:
    """Test trade attribution to news events."""

    def test_trade_attribution(self):
        """Test attribution of trades to news events."""
        attributor = TradeAttributor()

        trade = TradeResult(
            trade_id="trade-123",
            signal_id="signal-456",
            asset="BTC",
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 14, 0),
            entry_price=45000,
            exit_price=46000,
            position_size=0.1,
            pnl=100.0,
            status=TradeStatus.CLOSED,
            news_events=["news-001", "news-002"],
        )

        news_metadata = {
            "news-001": {
                "source": "reuters",
                "sentiment": 0.8,
                "published": datetime(2024, 1, 15, 9, 30),
                "impact_score": 0.9,
            },
            "news-002": {
                "source": "bloomberg",
                "sentiment": 0.6,
                "published": datetime(2024, 1, 15, 9, 45),
                "impact_score": 0.7,
            },
        }

        attribution = attributor.attribute_trade(trade, news_metadata)

        assert "reuters" in attribution.source_contributions
        assert attribution.source_contributions["reuters"] > 0
        assert attribution.primary_catalyst == "news-001"
        
        # Check that contributions sum to 1
        total = sum(attribution.source_contributions.values())
        assert pytest.approx(total, 0.01) == 1.0

    def test_time_proximity_weighting(self):
        """Test that news closer to trade time has higher weight."""
        attributor = TradeAttributor()

        trade = TradeResult(
            trade_id="trade-123",
            signal_id="signal-456",
            asset="ETH",
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_time=datetime(2024, 1, 15, 11, 0),
            entry_price=2500,
            exit_price=2550,
            position_size=1,
            pnl=50.0,
            status=TradeStatus.CLOSED,
            news_events=["news-001", "news-002", "news-003"],
        )

        news_metadata = {
            "news-001": {
                "source": "reuters",
                "sentiment": 0.7,
                "published": datetime(2024, 1, 15, 9, 55),  # 5 min before
                "impact_score": 0.8,
            },
            "news-002": {
                "source": "reuters",
                "sentiment": 0.7,
                "published": datetime(2024, 1, 15, 8, 0),  # 2 hours before
                "impact_score": 0.8,
            },
            "news-003": {
                "source": "reuters",
                "sentiment": 0.7,
                "published": datetime(2024, 1, 15, 6, 0),  # 4 hours before
                "impact_score": 0.8,
            },
        }

        attribution = attributor.attribute_trade(trade, news_metadata)

        # News closer in time should have higher weight
        assert attribution.news_weights["news-001"] > attribution.news_weights["news-002"]
        assert attribution.news_weights["news-002"] > attribution.news_weights["news-003"]

    def test_empty_news_events(self):
        """Test attribution with no news events."""
        attributor = TradeAttributor()

        trade = TradeResult(
            trade_id="trade-123",
            signal_id="signal-456",
            asset="BTC",
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            entry_price=50000,
            exit_price=51000,
            position_size=0.1,
            pnl=100.0,
            status=TradeStatus.CLOSED,
            news_events=[],
        )

        attribution = attributor.attribute_trade(trade, {})

        assert len(attribution.source_contributions) == 0
        assert attribution.primary_catalyst is None


class TestSentimentAccuracyTracking:
    """Test sentiment prediction accuracy tracking."""

    def test_sentiment_accuracy_tracking(self):
        """Test tracking of sentiment prediction accuracy."""
        tracker = SentimentAccuracyTracker()

        # Record predictions and outcomes
        predictions = [
            {
                "news_id": "news-001",
                "predicted_sentiment": 0.8,
                "predicted_impact": "bullish",
                "actual_price_change": 0.05,  # 5% increase
            },
            {
                "news_id": "news-002",
                "predicted_sentiment": -0.6,
                "predicted_impact": "bearish",
                "actual_price_change": -0.03,  # 3% decrease
            },
            {
                "news_id": "news-003",
                "predicted_sentiment": 0.2,
                "predicted_impact": "neutral",
                "actual_price_change": 0.01,  # 1% increase
            },
        ]

        for pred in predictions:
            tracker.record_prediction(**pred)

        accuracy = tracker.calculate_accuracy()
        
        assert accuracy["direction_accuracy"] > 0
        assert accuracy["magnitude_mae"] >= 0
        assert accuracy["total_predictions"] == 3
        assert accuracy["correct_directions"] >= 2  # At least 2 correct

    def test_accuracy_by_sentiment_range(self):
        """Test accuracy breakdown by sentiment ranges."""
        tracker = SentimentAccuracyTracker()

        # Add predictions with various sentiment levels
        for i in range(20):
            sentiment = (i - 10) / 10  # -1 to 1
            # Simulate correlation between sentiment and price change
            price_change = sentiment * 0.05 + (i % 3 - 1) * 0.01
            
            tracker.record_prediction(
                news_id=f"news-{i}",
                predicted_sentiment=sentiment,
                predicted_impact="bullish" if sentiment > 0.3 else "bearish" if sentiment < -0.3 else "neutral",
                actual_price_change=price_change,
            )

        accuracy_by_range = tracker.get_accuracy_by_sentiment_range()
        
        assert "strong_positive" in accuracy_by_range
        assert "strong_negative" in accuracy_by_range
        assert "neutral" in accuracy_by_range

    def test_model_comparison(self):
        """Test comparing accuracy between different models."""
        tracker = SentimentAccuracyTracker()

        # Record predictions from different models
        models = ["finbert", "roberta", "gpt"]
        
        for model in models:
            for i in range(10):
                tracker.record_prediction(
                    news_id=f"{model}-news-{i}",
                    predicted_sentiment=0.5 + (i % 3 - 1) * 0.2,
                    predicted_impact="bullish",
                    actual_price_change=0.03 + (i % 3 - 1) * 0.01,
                    model_name=model,
                )

        model_comparison = tracker.compare_models()
        
        assert all(model in model_comparison for model in models)
        assert all("accuracy" in model_comparison[model] for model in models)