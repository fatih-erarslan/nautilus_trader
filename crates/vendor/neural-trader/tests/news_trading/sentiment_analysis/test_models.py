"""
Tests for sentiment analysis data models
"""
import pytest
from decimal import Decimal
from datetime import datetime

from src.news_trading.sentiment_analysis.models import (
    SentimentDirection, MarketImpact, SentimentResult, SentimentBreakdown
)


class TestSentimentDirection:
    """Test SentimentDirection enum"""
    
    def test_sentiment_direction_values(self):
        """Test that SentimentDirection has correct values"""
        assert SentimentDirection.BULLISH.value == "bullish"
        assert SentimentDirection.BEARISH.value == "bearish"
        assert SentimentDirection.NEUTRAL.value == "neutral"
    
    def test_sentiment_direction_from_score(self):
        """Test creating direction from sentiment score"""
        assert SentimentDirection.from_score(0.7) == SentimentDirection.BULLISH
        assert SentimentDirection.from_score(-0.7) == SentimentDirection.BEARISH
        assert SentimentDirection.from_score(0.1) == SentimentDirection.NEUTRAL


class TestMarketImpact:
    """Test MarketImpact data model"""
    
    def test_market_impact_creation(self):
        """Test creating MarketImpact instance"""
        impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.75,
            timeframe="short-term",
            affected_assets=["BTC", "ETH"],
            confidence=0.85,
            volatility_expected="high",
            catalysts=["regulatory approval", "institutional adoption"]
        )
        
        assert impact.direction == SentimentDirection.BULLISH
        assert impact.magnitude == 0.75
        assert impact.timeframe == "short-term"
        assert "BTC" in impact.affected_assets
        assert impact.confidence == 0.85
        assert impact.volatility_expected == "high"
        assert "regulatory approval" in impact.catalysts
    
    def test_market_impact_validation(self):
        """Test MarketImpact validation"""
        # Test invalid magnitude
        with pytest.raises(ValueError, match="Magnitude must be between 0 and 1"):
            MarketImpact(
                direction=SentimentDirection.BULLISH,
                magnitude=1.5,
                timeframe="short-term",
                affected_assets=["BTC"]
            )
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            MarketImpact(
                direction=SentimentDirection.BULLISH,
                magnitude=0.5,
                timeframe="short-term",
                affected_assets=["BTC"],
                confidence=-0.1
            )
    
    def test_market_impact_is_significant(self):
        """Test significance check"""
        high_impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.8,
            timeframe="short-term",
            affected_assets=["BTC"],
            confidence=0.9
        )
        assert high_impact.is_significant()
        
        low_impact = MarketImpact(
            direction=SentimentDirection.NEUTRAL,
            magnitude=0.2,
            timeframe="short-term",
            affected_assets=["BTC"],
            confidence=0.5
        )
        assert not low_impact.is_significant()


class TestSentimentBreakdown:
    """Test SentimentBreakdown model"""
    
    def test_sentiment_breakdown_creation(self):
        """Test creating SentimentBreakdown"""
        breakdown = SentimentBreakdown(
            headline=0.8,
            content=0.7,
            entities=0.75,
            tone=0.6,
            language_intensity=0.85
        )
        
        assert breakdown.headline == 0.8
        assert breakdown.content == 0.7
        assert breakdown.entities == 0.75
        assert breakdown.tone == 0.6
        assert breakdown.language_intensity == 0.85
    
    def test_sentiment_breakdown_average(self):
        """Test calculating average sentiment"""
        breakdown = SentimentBreakdown(
            headline=0.8,
            content=0.6,
            entities=0.7,
            tone=0.7,
            language_intensity=0.8
        )
        
        assert breakdown.average == 0.72  # (0.8 + 0.6 + 0.7 + 0.7 + 0.8) / 5
    
    def test_sentiment_breakdown_validation(self):
        """Test validation of sentiment scores"""
        with pytest.raises(ValueError, match="scores must be between -1 and 1"):
            SentimentBreakdown(
                headline=1.5,
                content=0.5,
                entities=0.5
            )


class TestSentimentResult:
    """Test SentimentResult data model"""
    
    def test_sentiment_result_creation(self):
        """Test creating SentimentResult instance"""
        market_impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.6,
            timeframe="short-term",
            affected_assets=["BTC"]
        )
        
        breakdown = SentimentBreakdown(
            headline=0.8,
            content=0.7,
            entities=0.75
        )
        
        result = SentimentResult(
            article_id="test-123",
            overall_sentiment=0.75,
            confidence=0.85,
            market_impact=market_impact,
            sentiment_breakdown=breakdown,
            reasoning="Positive regulatory news typically drives market up",
            model_scores={"finbert": 0.8, "llm": 0.7},
            processing_time=0.5,
            source_metadata={
                "source": "reuters",
                "published_at": datetime.now(),
                "author": "John Doe"
            }
        )
        
        assert result.article_id == "test-123"
        assert result.overall_sentiment == 0.75
        assert result.confidence == 0.85
        assert result.market_impact.direction == SentimentDirection.BULLISH
        assert result.sentiment_breakdown.headline == 0.8
        assert result.reasoning == "Positive regulatory news typically drives market up"
        assert result.model_scores["finbert"] == 0.8
        assert result.processing_time == 0.5
        assert result.source_metadata["source"] == "reuters"
    
    def test_sentiment_result_validation(self):
        """Test SentimentResult validation"""
        market_impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.6,
            timeframe="short-term",
            affected_assets=["BTC"]
        )
        
        # Test invalid sentiment score
        with pytest.raises(ValueError, match="Overall sentiment must be between -1 and 1"):
            SentimentResult(
                article_id="test",
                overall_sentiment=1.5,
                confidence=0.8,
                market_impact=market_impact,
                sentiment_breakdown=SentimentBreakdown(),
                reasoning="test"
            )
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SentimentResult(
                article_id="test",
                overall_sentiment=0.5,
                confidence=1.5,
                market_impact=market_impact,
                sentiment_breakdown=SentimentBreakdown(),
                reasoning="test"
            )
    
    def test_sentiment_result_is_tradeable(self):
        """Test tradeable signal check"""
        market_impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.8,
            timeframe="short-term",
            affected_assets=["BTC"],
            confidence=0.9
        )
        
        # High confidence, significant impact - tradeable
        result = SentimentResult(
            article_id="test",
            overall_sentiment=0.8,
            confidence=0.9,
            market_impact=market_impact,
            sentiment_breakdown=SentimentBreakdown(headline=0.8),
            reasoning="Strong bullish signal"
        )
        assert result.is_tradeable()
        
        # Low confidence - not tradeable
        result_low_conf = SentimentResult(
            article_id="test",
            overall_sentiment=0.8,
            confidence=0.4,
            market_impact=market_impact,
            sentiment_breakdown=SentimentBreakdown(headline=0.8),
            reasoning="Low confidence signal"
        )
        assert not result_low_conf.is_tradeable()
    
    def test_sentiment_result_to_dict(self):
        """Test converting to dictionary"""
        market_impact = MarketImpact(
            direction=SentimentDirection.BULLISH,
            magnitude=0.6,
            timeframe="short-term",
            affected_assets=["BTC"]
        )
        
        result = SentimentResult(
            article_id="test-123",
            overall_sentiment=0.75,
            confidence=0.85,
            market_impact=market_impact,
            sentiment_breakdown=SentimentBreakdown(headline=0.8),
            reasoning="Test reasoning"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["article_id"] == "test-123"
        assert result_dict["overall_sentiment"] == 0.75
        assert result_dict["confidence"] == 0.85
        assert result_dict["market_impact"]["direction"] == "bullish"
        assert result_dict["sentiment_breakdown"]["headline"] == 0.8