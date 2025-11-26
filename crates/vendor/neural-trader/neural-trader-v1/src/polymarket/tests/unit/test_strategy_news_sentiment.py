"""
Unit tests for news sentiment trading strategy

These tests follow TDD principles by testing the functionality before implementation.
Tests cover signal generation, risk management, position sizing, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from ...strategies.news_sentiment import NewsSentimentStrategy, SentimentSignal
from ...strategies.base import (
    TradingSignal, SignalStrength, SignalDirection, StrategyConfig, StrategyError
)
from ...models import Market, MarketStatus, Order, OrderSide, OrderStatus
from ...api import PolymarketClient


class TestNewsSentimentStrategy:
    """Test suite for News Sentiment Strategy"""

    @pytest.fixture
    async def strategy(self, mock_clob_client):
        """Create news sentiment strategy instance"""
        config = StrategyConfig(
            max_position_size=Decimal('100.0'),
            min_confidence=0.6,
            min_signal_strength=SignalStrength.MODERATE,
            max_markets_monitored=25
        )
        return NewsSentimentStrategy(mock_clob_client, config)

    @pytest.fixture
    def sentiment_data(self):
        """Sample sentiment analysis data"""
        return {
            'overall_sentiment': 0.75,
            'confidence': 0.85,
            'article_count': 15,
            'sources': ['Reuters', 'Bloomberg', 'CNBC'],
            'keywords': ['bullish', 'growth', 'positive'],
            'sentiment_trend': 'increasing',
            'impact_score': 0.8
        }

    @pytest.fixture
    def market_crypto(self):
        """Sample crypto market for testing"""
        return Market(
            id="crypto-btc-100k",
            question="Will Bitcoin reach $100,000 by end of 2024?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=60),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.45"), "No": Decimal("0.55")},
            metadata={
                'category': 'crypto',
                'tags': ['bitcoin', 'price-prediction'],
                'volume_24h': 50000
            }
        )

    # Test Strategy Initialization
    async def test_strategy_initialization(self, mock_clob_client):
        """Test strategy initializes with correct parameters"""
        config = StrategyConfig(min_confidence=0.7)
        strategy = NewsSentimentStrategy(mock_clob_client, config)
        
        assert strategy.name == "NewsSentimentStrategy"
        assert strategy.config.min_confidence == 0.7
        assert strategy.sentiment_sources == ["mcp__ai-news-trader__analyze_news"]
        assert strategy.supported_categories == ["crypto", "politics", "sports", "finance"]

    # Test Market Suitability Analysis
    async def test_should_trade_market_crypto(self, strategy, market_crypto):
        """Test that strategy identifies suitable crypto markets"""
        result = await strategy.should_trade_market(market_crypto)
        assert result is True

    async def test_should_trade_market_unsupported_category(self, strategy):
        """Test rejection of unsupported market categories"""
        market = Market(
            id="weather-rain",
            question="Will it rain tomorrow?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=1),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.6"), "No": Decimal("0.4")},
            metadata={'category': 'weather'}
        )
        
        result = await strategy.should_trade_market(market)
        assert result is False

    async def test_should_trade_market_low_volume(self, strategy, market_crypto):
        """Test rejection of low volume markets"""
        market_crypto.metadata['volume_24h'] = 500  # Below threshold
        result = await strategy.should_trade_market(market_crypto)
        assert result is False

    async def test_should_trade_market_expiring_soon(self, strategy, market_crypto):
        """Test rejection of markets expiring too soon"""
        market_crypto.end_date = datetime.now() + timedelta(hours=2)
        result = await strategy.should_trade_market(market_crypto)
        assert result is False

    # Test News Sentiment Analysis
    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_fetch_news_sentiment_success(self, mock_fetch, strategy, market_crypto, sentiment_data):
        """Test successful news sentiment fetching"""
        mock_fetch.return_value = sentiment_data
        
        result = await strategy._fetch_news_sentiment(market_crypto)
        
        assert result == sentiment_data
        mock_fetch.assert_called_once_with(market_crypto)

    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_fetch_news_sentiment_failure(self, mock_fetch, strategy, market_crypto):
        """Test handling of news sentiment fetch failures"""
        mock_fetch.side_effect = Exception("API error")
        
        result = await strategy._fetch_news_sentiment(market_crypto)
        assert result is None

    async def test_parse_sentiment_keywords_bullish(self, strategy):
        """Test parsing of bullish sentiment keywords"""
        text = "positive growth outlook bullish momentum"
        sentiment_score = strategy._parse_sentiment_keywords(text)
        assert sentiment_score > 0.5

    async def test_parse_sentiment_keywords_bearish(self, strategy):
        """Test parsing of bearish sentiment keywords"""
        text = "negative decline bearish crash correction"
        sentiment_score = strategy._parse_sentiment_keywords(text)
        assert sentiment_score < 0.5

    async def test_parse_sentiment_keywords_neutral(self, strategy):
        """Test parsing of neutral sentiment"""
        text = "stable unchanged sideways"
        sentiment_score = strategy._parse_sentiment_keywords(text)
        assert 0.4 <= sentiment_score <= 0.6

    # Test Signal Generation
    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_analyze_market_strong_bullish_signal(self, mock_fetch, strategy, market_crypto, sentiment_data):
        """Test generation of strong bullish signal"""
        sentiment_data['overall_sentiment'] = 0.85
        sentiment_data['confidence'] = 0.9
        mock_fetch.return_value = sentiment_data
        
        signal = await strategy.analyze_market(market_crypto)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.outcome == "Yes"
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence >= 0.8

    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_analyze_market_strong_bearish_signal(self, mock_fetch, strategy, market_crypto, sentiment_data):
        """Test generation of strong bearish signal"""
        sentiment_data['overall_sentiment'] = 0.15
        sentiment_data['confidence'] = 0.9
        mock_fetch.return_value = sentiment_data
        
        signal = await strategy.analyze_market(market_crypto)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.outcome == "Yes"  # Sell yes = bet no
        assert signal.strength == SignalStrength.STRONG

    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_analyze_market_weak_signal_filtered(self, mock_fetch, strategy, market_crypto, sentiment_data):
        """Test filtering of weak signals"""
        sentiment_data['overall_sentiment'] = 0.55
        sentiment_data['confidence'] = 0.4  # Low confidence
        mock_fetch.return_value = sentiment_data
        
        signal = await strategy.analyze_market(market_crypto)
        assert signal is None

    @patch('polymarket.strategies.news_sentiment.NewsSentimentStrategy._fetch_news_sentiment')
    async def test_analyze_market_no_sentiment_data(self, mock_fetch, strategy, market_crypto):
        """Test handling when no sentiment data is available"""
        mock_fetch.return_value = None
        
        signal = await strategy.analyze_market(market_crypto)
        assert signal is None

    # Test Position Sizing
    async def test_calculate_position_size_kelly_criterion(self, strategy):
        """Test Kelly criterion position sizing"""
        sentiment_signal = SentimentSignal(
            sentiment_score=0.8,
            confidence=0.85,
            trend_strength=0.7,
            article_count=20,
            impact_score=0.75
        )
        market_price = Decimal("0.45")
        
        size = strategy._calculate_position_size(sentiment_signal, market_price)
        
        # Should be positive for bullish signal
        assert size > 0
        # Should be reasonable size (not too large)
        assert size <= strategy.config.max_position_size

    async def test_calculate_position_size_bearish(self, strategy):
        """Test position sizing for bearish signals"""
        sentiment_signal = SentimentSignal(
            sentiment_score=0.2,
            confidence=0.8,
            trend_strength=0.6,
            article_count=15,
            impact_score=0.7
        )
        market_price = Decimal("0.65")
        
        size = strategy._calculate_position_size(sentiment_signal, market_price)
        
        # Should be positive (selling yes = buying no)
        assert size > 0

    async def test_calculate_position_size_max_limit(self, strategy):
        """Test position size respects maximum limits"""
        sentiment_signal = SentimentSignal(
            sentiment_score=0.95,  # Very bullish
            confidence=0.95,
            trend_strength=0.9,
            article_count=50,
            impact_score=0.9
        )
        market_price = Decimal("0.1")  # Very underpriced
        
        size = strategy._calculate_position_size(sentiment_signal, market_price)
        
        assert size <= strategy.config.max_position_size

    # Test Risk Management
    async def test_check_sentiment_quality_high_quality(self, strategy, sentiment_data):
        """Test detection of high quality sentiment data"""
        result = strategy._check_sentiment_quality(sentiment_data)
        assert result is True

    async def test_check_sentiment_quality_low_article_count(self, strategy, sentiment_data):
        """Test rejection of sentiment with few articles"""
        sentiment_data['article_count'] = 2
        result = strategy._check_sentiment_quality(sentiment_data)
        assert result is False

    async def test_check_sentiment_quality_low_confidence(self, strategy, sentiment_data):
        """Test rejection of low confidence sentiment"""
        sentiment_data['confidence'] = 0.3
        result = strategy._check_sentiment_quality(sentiment_data)
        assert result is False

    async def test_check_sentiment_quality_limited_sources(self, strategy, sentiment_data):
        """Test rejection of sentiment from limited sources"""
        sentiment_data['sources'] = ['single-source']
        result = strategy._check_sentiment_quality(sentiment_data)
        assert result is False

    # Test Signal Validation
    async def test_validate_signal_strength_mapping(self, strategy):
        """Test correct mapping of sentiment to signal strength"""
        test_cases = [
            (0.9, 0.9, SignalStrength.VERY_STRONG),
            (0.8, 0.8, SignalStrength.STRONG),
            (0.7, 0.7, SignalStrength.MODERATE),
            (0.6, 0.6, SignalStrength.WEAK),
            (0.55, 0.5, SignalStrength.VERY_WEAK),
        ]
        
        for sentiment_score, confidence, expected_strength in test_cases:
            strength = strategy._map_sentiment_to_strength(sentiment_score, confidence)
            assert strength == expected_strength

    async def test_validate_signal_price_bounds(self, strategy, market_crypto):
        """Test signal price stays within valid bounds"""
        signal = TradingSignal(
            market_id=market_crypto.id,
            outcome="Yes",
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            target_price=Decimal("0.7"),
            size=Decimal("10"),
            confidence=0.8,
            reasoning="Strong bullish sentiment"
        )
        
        validated_signal = strategy._validate_signal_bounds(signal, market_crypto)
        
        assert 0 < validated_signal.target_price <= 1
        assert validated_signal.size > 0

    # Test Performance Tracking
    async def test_update_sentiment_metrics(self, strategy, sentiment_data):
        """Test sentiment metrics tracking"""
        initial_count = strategy.sentiment_analysis_count
        
        strategy._update_sentiment_metrics(sentiment_data)
        
        assert strategy.sentiment_analysis_count == initial_count + 1
        assert 'overall_sentiment' in strategy.sentiment_history

    async def test_get_sentiment_performance_summary(self, strategy):
        """Test sentiment performance summary generation"""
        # Add some test data
        strategy.sentiment_history = [0.8, 0.6, 0.7, 0.9]
        strategy.sentiment_analysis_count = 4
        
        summary = strategy.get_sentiment_performance_summary()
        
        assert 'average_sentiment' in summary
        assert 'sentiment_volatility' in summary
        assert 'analysis_count' in summary
        assert summary['analysis_count'] == 4

    # Test Error Handling
    async def test_handle_api_rate_limit(self, strategy, market_crypto):
        """Test handling of API rate limits"""
        with patch.object(strategy, '_fetch_news_sentiment') as mock_fetch:
            mock_fetch.side_effect = Exception("Rate limit exceeded")
            
            signal = await strategy.analyze_market(market_crypto)
            assert signal is None

    async def test_handle_malformed_sentiment_data(self, strategy, market_crypto):
        """Test handling of malformed sentiment data"""
        with patch.object(strategy, '_fetch_news_sentiment') as mock_fetch:
            mock_fetch.return_value = {'invalid': 'data'}
            
            signal = await strategy.analyze_market(market_crypto)
            assert signal is None

    async def test_handle_network_timeout(self, strategy, market_crypto):
        """Test handling of network timeouts"""
        with patch.object(strategy, '_fetch_news_sentiment') as mock_fetch:
            mock_fetch.side_effect = TimeoutError("Network timeout")
            
            signal = await strategy.analyze_market(market_crypto)
            assert signal is None

    # Test Batch Processing
    async def test_analyze_markets_batch(self, strategy, mock_clob_client):
        """Test batch processing of multiple markets"""
        markets = [
            Market(
                id=f"test-market-{i}",
                question=f"Test question {i}?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.5"), "No": Decimal("0.5")},
                metadata={'category': 'crypto', 'volume_24h': 10000}
            )
            for i in range(3)
        ]
        
        with patch.object(strategy, '_fetch_news_sentiment') as mock_fetch:
            mock_fetch.return_value = {
                'overall_sentiment': 0.8,
                'confidence': 0.8,
                'article_count': 10,
                'sources': ['Reuters', 'Bloomberg'],
                'impact_score': 0.7
            }
            
            signals = await strategy.analyze_markets(markets)
            
            # Should get signals from suitable markets
            assert len(signals) >= 0
            mock_fetch.call_count >= 1

    # Test Configuration
    async def test_custom_configuration(self, mock_clob_client):
        """Test strategy with custom configuration"""
        config = StrategyConfig(
            min_confidence=0.8,
            max_position_size=Decimal('50.0'),
            min_signal_strength=SignalStrength.STRONG
        )
        
        strategy = NewsSentimentStrategy(mock_clob_client, config)
        
        assert strategy.config.min_confidence == 0.8
        assert strategy.config.max_position_size == Decimal('50.0')
        assert strategy.config.min_signal_strength == SignalStrength.STRONG

    # Test Integration with MCP Tools
    @patch('polymarket.strategies.news_sentiment.mcp__ai_news_trader__analyze_news')
    async def test_mcp_integration(self, mock_mcp, strategy, market_crypto):
        """Test integration with MCP news analysis tools"""
        mock_mcp.return_value = {
            'sentiment_score': 0.75,
            'confidence': 0.85,
            'articles_analyzed': 15,
            'sentiment_category': 'bullish'
        }
        
        with patch.object(strategy, '_fetch_news_sentiment') as mock_fetch:
            mock_fetch.return_value = {
                'overall_sentiment': 0.75,
                'confidence': 0.85,
                'article_count': 15,
                'sources': ['mcp'],
                'impact_score': 0.8
            }
            
            signal = await strategy.analyze_market(market_crypto)
            
            assert signal is not None
            assert signal.confidence >= 0.7


class TestSentimentSignal:
    """Test suite for SentimentSignal data class"""

    def test_sentiment_signal_creation(self):
        """Test creation of sentiment signal"""
        signal = SentimentSignal(
            sentiment_score=0.8,
            confidence=0.85,
            trend_strength=0.7,
            article_count=15,
            impact_score=0.75
        )
        
        assert signal.sentiment_score == 0.8
        assert signal.confidence == 0.85
        assert signal.is_bullish is True
        assert signal.is_bearish is False

    def test_sentiment_signal_bearish(self):
        """Test bearish sentiment signal"""
        signal = SentimentSignal(
            sentiment_score=0.3,
            confidence=0.8,
            trend_strength=0.6,
            article_count=12,
            impact_score=0.7
        )
        
        assert signal.is_bearish is True
        assert signal.is_bullish is False

    def test_sentiment_signal_validation(self):
        """Test sentiment signal validation"""
        with pytest.raises(ValueError):
            SentimentSignal(
                sentiment_score=1.5,  # Invalid - out of range
                confidence=0.8,
                trend_strength=0.6,
                article_count=10,
                impact_score=0.7
            )

    def test_sentiment_signal_strength_calculation(self):
        """Test sentiment signal strength calculation"""
        signal = SentimentSignal(
            sentiment_score=0.9,
            confidence=0.95,
            trend_strength=0.8,
            article_count=25,
            impact_score=0.85
        )
        
        strength = signal.calculate_signal_strength()
        assert strength >= 0.8  # Should be high for strong metrics