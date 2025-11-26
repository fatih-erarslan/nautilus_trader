"""Tests for complete Trading Decision Engine following TDD."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from src.news_trading.decision_engine.engine import NewsDecisionEngine
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class TestNewsDecisionEngine:
    """Test suite for complete Trading Decision Engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization with components."""
        engine = NewsDecisionEngine(
            account_size=100000,
            max_portfolio_risk=0.2
        )
        
        assert engine.account_size == 100000
        assert engine.risk_manager is not None
        assert engine.strategies is not None
        assert len(engine.strategies) >= 3  # Swing, Momentum, Mirror
        
    @pytest.mark.asyncio
    async def test_process_sentiment_to_signal(self):
        """Test converting sentiment data to trading signal."""
        engine = NewsDecisionEngine()
        
        sentiment_data = {
            "asset": "AAPL",
            "sentiment_score": 0.8,
            "confidence": 0.85,
            "market_impact": {
                "direction": "bullish",
                "magnitude": 0.7,
                "timeframe": "short-term"
            },
            "entities": ["Apple", "iPhone", "Sales"],
            "source_events": ["news-001"]
        }
        
        signal = await engine.process_sentiment(sentiment_data)
        
        assert signal is not None
        assert signal.asset == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength > 0.7
        assert signal.position_size > 0
        
    @pytest.mark.asyncio
    async def test_bearish_sentiment_signal(self):
        """Test bearish sentiment signal generation."""
        engine = NewsDecisionEngine()
        
        sentiment_data = {
            "asset": "TSLA",
            "sentiment_score": -0.7,
            "confidence": 0.8,
            "market_impact": {
                "direction": "bearish",
                "magnitude": 0.6,
                "timeframe": "medium-term"
            }
        }
        
        signal = await engine.process_sentiment(sentiment_data)
        
        assert signal is not None
        assert signal.signal_type in [SignalType.SELL, SignalType.SHORT]
        assert signal.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        
    @pytest.mark.asyncio
    async def test_neutral_sentiment_no_signal(self):
        """Test that neutral sentiment doesn't generate signals."""
        engine = NewsDecisionEngine()
        
        sentiment_data = {
            "asset": "MSFT",
            "sentiment_score": 0.1,  # Nearly neutral
            "confidence": 0.5,       # Low confidence
            "market_impact": {
                "direction": "neutral",
                "magnitude": 0.2
            }
        }
        
        signal = await engine.process_sentiment(sentiment_data)
        
        assert signal is None  # No signal for weak/neutral sentiment
        
    @pytest.mark.asyncio
    async def test_evaluate_portfolio_rebalancing(self):
        """Test portfolio evaluation and rebalancing signals."""
        engine = NewsDecisionEngine()
        
        current_positions = {
            "BTC": {
                "size": 0.5,
                "entry_price": 40000,
                "current_price": 50000,
                "unrealized_pnl": 0.25
            },
            "ETH": {
                "size": 0.3,
                "entry_price": 3000,
                "current_price": 2500,
                "unrealized_pnl": -0.167
            },
            "AAPL": {
                "size": 0.2,
                "entry_price": 150,
                "current_price": 155,
                "unrealized_pnl": 0.033
            }
        }
        
        signals = await engine.evaluate_portfolio(current_positions)
        
        assert len(signals) > 0
        # Should suggest taking profits on BTC (25% gain)
        assert any(s.asset == "BTC" and s.signal_type == SignalType.SELL for s in signals)
        # May suggest adding to ETH (down 16.7%)
        assert any(s.asset == "ETH" for s in signals)
        
    @pytest.mark.asyncio
    async def test_process_market_data_signals(self):
        """Test generating signals from market data."""
        engine = NewsDecisionEngine()
        
        market_data = {
            "NVDA": {
                "price": 500,
                "price_change_5d": 0.15,
                "price_change_20d": 0.25,
                "volume_ratio": 2.5,
                "ma_50": 480,
                "ma_200": 450,
                "rsi": 65,
                "atr": 15
            }
        }
        
        signals = await engine.process_market_data(market_data)
        
        assert len(signals) > 0
        nvda_signal = next((s for s in signals if s.asset == "NVDA"), None)
        assert nvda_signal is not None
        assert nvda_signal.strategy in [TradingStrategy.MOMENTUM, TradingStrategy.SWING]
        
    def test_set_risk_parameters(self):
        """Test updating risk parameters."""
        engine = NewsDecisionEngine()
        
        new_params = {
            "max_position_size": 0.05,
            "max_portfolio_risk": 0.15,
            "max_correlation": 0.6
        }
        
        engine.set_risk_parameters(new_params)
        
        assert engine.risk_manager.max_position_size == 0.05
        assert engine.risk_manager.max_portfolio_risk == 0.15
        
    def test_get_active_signals(self):
        """Test retrieving active signals."""
        engine = NewsDecisionEngine()
        
        # Add some test signals
        test_signal = TradingSignal(
            id="test-001",
            timestamp=datetime.now(),
            asset="GOOGL",
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=0.8,
            confidence=0.7,
            risk_level=RiskLevel.MEDIUM,
            position_size=0.05,
            entry_price=140,
            stop_loss=135,
            take_profit=150,
            holding_period="3-7 days",
            source_events=["test"],
            reasoning="Test signal"
        )
        
        engine._active_signals.append(test_signal)
        
        active = engine.get_active_signals()
        assert len(active) == 1
        assert active[0].asset == "GOOGL"
        
    @pytest.mark.asyncio
    async def test_strategy_selection_by_market_conditions(self):
        """Test that appropriate strategy is selected based on conditions."""
        engine = NewsDecisionEngine()
        
        # High momentum conditions
        momentum_data = {
            "asset": "AMD",
            "sentiment_score": 0.9,
            "confidence": 0.9,
            "market_impact": {
                "magnitude": 0.8,
                "direction": "bullish"
            },
            "market_conditions": {
                "trend": "strong_uptrend",
                "volatility": "moderate"
            }
        }
        
        signal = await engine.process_sentiment(momentum_data)
        
        assert signal is not None
        # Should favor momentum strategy for strong trends
        assert signal.strategy == TradingStrategy.MOMENTUM
        
    @pytest.mark.asyncio
    async def test_multi_asset_support(self):
        """Test support for multiple asset types."""
        engine = NewsDecisionEngine()
        
        # Crypto sentiment
        crypto_sentiment = {
            "asset": "BTC",
            "asset_type": "crypto",
            "sentiment_score": 0.7,
            "confidence": 0.8
        }
        
        # Bond sentiment
        bond_sentiment = {
            "asset": "US10Y",
            "asset_type": "bond",
            "sentiment_score": -0.5,
            "confidence": 0.7
        }
        
        crypto_signal = await engine.process_sentiment(crypto_sentiment)
        bond_signal = await engine.process_sentiment(bond_sentiment)
        
        assert crypto_signal.asset_type == AssetType.CRYPTO
        assert bond_signal.asset_type == AssetType.BOND
        # Different risk parameters for different assets
        assert crypto_signal.stop_loss <= crypto_signal.entry_price * 0.95  # At least 5% stop for crypto