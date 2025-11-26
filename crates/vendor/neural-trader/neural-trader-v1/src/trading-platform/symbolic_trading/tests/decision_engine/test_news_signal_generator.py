"""
Tests for News Signal Generator
Following TDD - Red-Green-Refactor approach
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from src.trading.decision_engine.models import SignalType, RiskLevel, TradingStrategy, AssetType


@pytest.mark.asyncio
async def test_news_signal_generator_initialization():
    """Test NewsSignalGenerator initialization"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    # Default initialization
    generator = NewsSignalGenerator()
    assert generator.config is not None
    assert "sentiment_thresholds" in generator.config
    assert "position_size_base" in generator.config
    
    # Custom configuration
    custom_config = {
        "sentiment_thresholds": {
            "bullish": 0.5,
            "bearish": -0.5
        },
        "position_size_base": 0.02
    }
    generator = NewsSignalGenerator(config=custom_config)
    assert generator.config["sentiment_thresholds"]["bullish"] == 0.5
    assert generator.config["position_size_base"] == 0.02


@pytest.mark.asyncio
async def test_sentiment_to_signal_conversion():
    """Test converting sentiment to trading signal"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    
    # Bullish sentiment
    sentiment_data = {
        "asset": "BTC",
        "sentiment_score": 0.8,
        "confidence": 0.85,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.7,
            "timeframe": "short-term"
        },
        "source_events": ["news-001", "news-002"]
    }
    
    # Mock market data
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 50000,
        "atr": 1500,  # Average True Range
        "volume_24h": 1000000
    })
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal is not None
    assert signal.signal_type == SignalType.BUY
    assert signal.asset == "BTC"
    assert signal.strength > 0.7
    assert signal.confidence == 0.85
    assert signal.position_size > 0
    assert signal.entry_price == 50000
    assert signal.stop_loss < signal.entry_price
    assert signal.take_profit > signal.entry_price
    assert len(signal.source_events) == 2


@pytest.mark.asyncio
async def test_bearish_signal_generation():
    """Test bearish signal generation"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    
    sentiment_data = {
        "asset": "ETH",
        "sentiment_score": -0.7,
        "confidence": 0.8,
        "market_impact": {
            "direction": "bearish",
            "magnitude": 0.6,
            "timeframe": "medium-term"
        },
        "source_events": ["news-003"]
    }
    
    # Mock market data
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 3000,
        "atr": 150,
        "volume_24h": 500000
    })
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal.signal_type == SignalType.SELL
    assert signal.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
    assert signal.stop_loss > signal.entry_price  # Stop loss above entry for short
    assert signal.take_profit < signal.entry_price  # Take profit below entry for short


@pytest.mark.asyncio
async def test_neutral_sentiment_no_signal():
    """Test that neutral sentiment generates HOLD signal"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    
    sentiment_data = {
        "asset": "ADA",
        "sentiment_score": 0.1,  # Nearly neutral
        "confidence": 0.9,
        "market_impact": {
            "direction": "neutral",
            "magnitude": 0.2,
            "timeframe": "short-term"
        }
    }
    
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 1.0,
        "atr": 0.05,
        "volume_24h": 100000
    })
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal.signal_type == SignalType.HOLD
    assert signal.position_size == 0  # No new position for HOLD


@pytest.mark.asyncio
async def test_position_size_calculation():
    """Test position size calculation based on confidence and risk"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 100,
        "atr": 5,
        "volume_24h": 10000
    })
    
    # High confidence, low risk
    sentiment_data_1 = {
        "asset": "TEST1",
        "sentiment_score": 0.9,
        "confidence": 0.95,
        "market_impact": {"magnitude": 0.8}
    }
    signal_1 = await generator.generate_signal(sentiment_data_1)
    
    # Low confidence, high risk
    sentiment_data_2 = {
        "asset": "TEST2",
        "sentiment_score": 0.9,
        "confidence": 0.5,
        "market_impact": {"magnitude": 0.8}
    }
    signal_2 = await generator.generate_signal(sentiment_data_2)
    
    # High confidence position should be larger
    assert signal_1.position_size > signal_2.position_size


@pytest.mark.asyncio
async def test_risk_level_assessment():
    """Test risk level assessment based on various factors"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 100,
        "atr": 10,  # High volatility (10% ATR)
        "volume_24h": 10000
    })
    
    # High volatility scenario
    sentiment_data = {
        "asset": "VOL",
        "sentiment_score": 0.8,
        "confidence": 0.7,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.9,
            "volatility_expected": "high"
        }
    }
    
    signal = await generator.generate_signal(sentiment_data)
    
    assert signal.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]


@pytest.mark.asyncio
async def test_stop_loss_take_profit_calculation():
    """Test stop loss and take profit calculation"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    
    # Test data with specific ATR
    sentiment_data = {
        "asset": "BTC",
        "sentiment_score": 0.7,
        "confidence": 0.8,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.6,
            "timeframe": "short-term"
        }
    }
    
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 50000,
        "atr": 1000,  # 2% ATR
        "volume_24h": 1000000
    })
    
    signal = await generator.generate_signal(sentiment_data)
    
    # For short-term bullish signal, expect:
    # - Stop loss: 1.5-2 ATR below entry
    # - Take profit: 2-3 ATR above entry
    expected_stop_loss = 50000 - (1.5 * 1000)  # 48500
    expected_take_profit = 50000 + (2.5 * 1000)  # 52500
    
    assert abs(signal.stop_loss - expected_stop_loss) < 500
    assert abs(signal.take_profit - expected_take_profit) < 500


@pytest.mark.asyncio
async def test_strategy_selection():
    """Test appropriate strategy selection based on conditions"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 100,
        "atr": 5,
        "volume_24h": 10000
    })
    
    # Short-term high impact news - expect SWING or DAY_TRADE
    sentiment_data_1 = {
        "asset": "TEST1",
        "sentiment_score": 0.8,
        "confidence": 0.9,
        "market_impact": {
            "magnitude": 0.8,
            "timeframe": "short-term"
        }
    }
    signal_1 = await generator.generate_signal(sentiment_data_1)
    assert signal_1.strategy in [TradingStrategy.SWING, TradingStrategy.DAY_TRADE]
    
    # Long-term fundamental news - expect POSITION
    sentiment_data_2 = {
        "asset": "TEST2",
        "sentiment_score": 0.7,
        "confidence": 0.85,
        "market_impact": {
            "magnitude": 0.6,
            "timeframe": "long-term"
        }
    }
    signal_2 = await generator.generate_signal(sentiment_data_2)
    assert signal_2.strategy == TradingStrategy.POSITION


@pytest.mark.asyncio
async def test_reasoning_generation():
    """Test that reasoning is properly generated"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 100,
        "atr": 5,
        "volume_24h": 10000
    })
    
    sentiment_data = {
        "asset": "BTC",
        "sentiment_score": 0.8,
        "confidence": 0.85,
        "market_impact": {
            "direction": "bullish",
            "magnitude": 0.7,
            "timeframe": "short-term",
            "catalysts": ["ETF approval", "institutional adoption"]
        },
        "source_events": ["news-001"]
    }
    
    signal = await generator.generate_signal(sentiment_data)
    
    # Check reasoning contains key information
    assert "bullish" in signal.reasoning.lower()
    assert "sentiment" in signal.reasoning.lower()
    assert str(signal.confidence) in signal.reasoning or "85%" in signal.reasoning


@pytest.mark.asyncio
async def test_asset_type_detection():
    """Test correct asset type detection"""
    from src.trading.decision_engine.news_signal_generator import NewsSignalGenerator
    
    generator = NewsSignalGenerator()
    generator._fetch_market_data = AsyncMock(return_value={
        "current_price": 100,
        "atr": 5,
        "volume_24h": 10000
    })
    
    # Crypto asset
    signal = await generator.generate_signal({
        "asset": "BTC",
        "sentiment_score": 0.7,
        "confidence": 0.8
    })
    assert signal.asset_type == AssetType.CRYPTO
    
    # Equity asset
    signal = await generator.generate_signal({
        "asset": "AAPL",
        "sentiment_score": 0.7,
        "confidence": 0.8
    })
    assert signal.asset_type == AssetType.EQUITY
    
    # Bond asset
    signal = await generator.generate_signal({
        "asset": "US10Y",
        "sentiment_score": 0.7,
        "confidence": 0.8
    })
    assert signal.asset_type == AssetType.BOND