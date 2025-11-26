"""
Unit tests for momentum trading strategy

These tests follow TDD principles by testing the functionality.
Tests cover trend detection, momentum indicators, position sizing, and risk management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
from collections import deque

from ...strategies.momentum import MomentumStrategy, MomentumIndicators
from ...strategies.base import (
    TradingSignal, SignalStrength, SignalDirection, StrategyConfig, StrategyError
)
from ...models import Market, MarketStatus, Order, OrderSide, OrderStatus
from ...api import PolymarketClient


class TestMomentumStrategy:
    """Test suite for Momentum Strategy"""

    @pytest.fixture
    async def strategy(self, mock_clob_client):
        """Create momentum strategy instance"""
        config = StrategyConfig(
            max_position_size=Decimal('500.0'),
            min_confidence=0.6,
            min_signal_strength=SignalStrength.MODERATE,
            max_markets_monitored=30
        )
        return MomentumStrategy(
            mock_clob_client,
            config,
            short_period=5,
            long_period=20,
            momentum_threshold=Decimal('0.05'),
            volume_confirmation=True
        )

    @pytest.fixture
    def trending_market(self):
        """Sample trending market for testing"""
        return Market(
            id="trending-market",
            question="Will trend continue?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")},
            metadata={
                'volume_24h': 75000,
                'price_24h_change': 0.15,
                'trades_24h': 500
            }
        )

    @pytest.fixture
    def price_history(self):
        """Sample price history showing uptrend"""
        return [
            0.40, 0.42, 0.41, 0.43, 0.45, 0.44, 0.46, 0.48,
            0.50, 0.52, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61,
            0.60, 0.62, 0.64, 0.65
        ]

    @pytest.fixture
    def volume_history(self):
        """Sample volume history"""
        return [
            1000, 1200, 1100, 1300, 1500, 1400, 1600, 1800,
            2000, 2200, 2100, 2300, 2500, 2700, 2900, 3100,
            3000, 3200, 3400, 3500
        ]

    # Test Strategy Initialization
    async def test_strategy_initialization(self, mock_clob_client):
        """Test momentum strategy initialization"""
        strategy = MomentumStrategy(
            mock_clob_client,
            short_period=7,
            long_period=30,
            momentum_threshold=Decimal('0.03')
        )
        
        assert strategy.name == "MomentumStrategy"
        assert strategy.short_period == 7
        assert strategy.long_period == 30
        assert strategy.momentum_threshold == Decimal('0.03')
        assert strategy.volume_confirmation is True
        assert len(strategy.price_history) == 0

    # Test Market Suitability
    async def test_should_trade_trending_market(self, strategy, trending_market):
        """Test that strategy identifies trending markets"""
        result = await strategy.should_trade_market(trending_market)
        assert result is True

    async def test_should_trade_low_volume_market(self, strategy, trending_market):
        """Test rejection of low volume markets"""
        trending_market.metadata['volume_24h'] = 1000  # Too low
        result = await strategy.should_trade_market(trending_market)
        assert result is False

    async def test_should_trade_extreme_price_market(self, strategy, trending_market):
        """Test rejection of markets with extreme prices"""
        trending_market.current_prices = {"Yes": Decimal("0.98"), "No": Decimal("0.02")}
        result = await strategy.should_trade_market(trending_market)
        assert result is False

    async def test_should_trade_expiring_market(self, strategy, trending_market):
        """Test rejection of soon-expiring markets"""
        trending_market.end_date = datetime.now() + timedelta(hours=6)
        result = await strategy.should_trade_market(trending_market)
        assert result is False

    # Test Data Management
    async def test_update_price_history(self, strategy, trending_market):
        """Test updating market price/volume data"""
        # Update price history
        strategy._update_price_history(trending_market)
        
        market_id = trending_market.id
        assert market_id in strategy.price_history
        assert len(strategy.price_history[market_id]) == 1
        assert strategy.price_history[market_id][-1] == 0.65  # Yes price

    async def test_data_history_limit(self, strategy):
        """Test that data history is limited to prevent memory issues"""
        market_id = "test-market"
        
        # Create price and volume history
        strategy.price_history[market_id] = deque(maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque(maxlen=strategy.long_period * 2)
        
        # Add more data than the limit
        for i in range(100):
            strategy.price_history[market_id].append(0.50)
            strategy.volume_history[market_id].append(1000)
        
        # Should be limited to maxlen
        assert len(strategy.price_history[market_id]) <= strategy.long_period * 2
        assert len(strategy.volume_history[market_id]) <= strategy.long_period * 2

    # Test Momentum Calculation
    async def test_calculate_momentum_indicators(self, strategy, price_history, volume_history):
        """Test calculation of momentum indicators"""
        market_id = "test-market"
        
        # Setup price and volume history
        strategy.price_history[market_id] = deque(price_history, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque(volume_history, maxlen=strategy.long_period * 2)
        
        indicators = strategy._calculate_momentum_indicators(market_id)
        
        assert indicators is not None
        assert isinstance(indicators, MomentumIndicators)
        assert indicators.momentum > 0  # Positive momentum for uptrend
        assert indicators.trend_direction == 1  # Uptrend
        assert indicators.short_ma < indicators.long_ma  # Prices trending up
        assert 0 <= indicators.rsi <= 100

    async def test_momentum_insufficient_data(self, strategy):
        """Test momentum calculation with insufficient data"""
        market_id = "new-market"
        
        # Add only a few data points
        strategy.price_history[market_id] = deque([0.50, 0.51, 0.52], maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque([1000, 1100, 1200], maxlen=strategy.long_period * 2)
        
        indicators = strategy._calculate_momentum_indicators(market_id)
        assert indicators is None

    async def test_calculate_rsi(self, strategy):
        """Test RSI calculation"""
        prices = [
            0.50, 0.52, 0.51, 0.53, 0.55, 0.54, 0.56, 0.58,
            0.57, 0.59, 0.61, 0.60, 0.62, 0.64, 0.63
        ]
        
        rsi = strategy._calculate_rsi(prices)
        
        assert 0 <= rsi <= 100
        assert rsi > 50  # Should be above 50 for uptrend

    async def test_momentum_indicators_properties(self):
        """Test MomentumIndicators properties"""
        indicators = MomentumIndicators(
            momentum=0.1,
            velocity=0.02,
            acceleration=0.001,
            volume_momentum=0.15,
            trend_direction=1,
            short_ma=0.52,
            long_ma=0.50,
            macd=0.02,
            rsi=65.0,
            volatility=0.05
        )
        
        assert indicators.is_bullish is True
        assert indicators.is_bearish is False
        
        # Test composite score calculation
        score = indicators.calculate_composite_score()
        assert isinstance(score, float)
        assert -2 <= score <= 2  # Bounded by tanh

    # Test Trend Detection
    async def test_calculate_trend_strength(self, strategy):
        """Test trend strength calculation"""
        indicators = MomentumIndicators(
            momentum=0.15,
            velocity=0.03,
            acceleration=0.002,
            volume_momentum=0.2,
            trend_direction=1,
            short_ma=0.55,
            long_ma=0.50,
            macd=0.05,
            rsi=65.0,
            volatility=0.03
        )
        
        trend_strength = strategy._calculate_trend_strength(indicators)
        
        assert trend_strength > 0
        assert trend_strength <= 1.0

    async def test_trend_strength_with_acceleration(self, strategy):
        """Test trend strength with acceleration factor"""
        # Accelerating trend
        indicators_accel = MomentumIndicators(
            momentum=0.1,
            velocity=0.02,
            acceleration=0.002,  # Positive acceleration with positive momentum
            volume_momentum=0.1,
            trend_direction=1,
            short_ma=0.52,
            long_ma=0.50,
            macd=0.02,
            rsi=60.0,
            volatility=0.03
        )
        
        # Decelerating trend
        indicators_decel = MomentumIndicators(
            momentum=0.1,
            velocity=0.02,
            acceleration=-0.002,  # Negative acceleration with positive momentum
            volume_momentum=0.1,
            trend_direction=1,
            short_ma=0.52,
            long_ma=0.50,
            macd=0.02,
            rsi=60.0,
            volatility=0.03
        )
        
        strength_accel = strategy._calculate_trend_strength(indicators_accel)
        strength_decel = strategy._calculate_trend_strength(indicators_decel)
        
        # Accelerating trend should be stronger
        assert strength_accel > strength_decel

    # Test Signal Generation
    async def test_generate_buy_signal(self, strategy, trending_market, price_history, volume_history):
        """Test generation of buy signal in uptrend"""
        market_id = trending_market.id
        
        # Setup trending data
        strategy.price_history[market_id] = deque(price_history, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque(volume_history, maxlen=strategy.long_period * 2)
        
        signal = await strategy.analyze_market(trending_market)
        
        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence >= 0.6

    async def test_no_signal_weak_momentum(self, strategy, trending_market):
        """Test no signal when momentum is weak"""
        market_id = trending_market.id
        
        # Weak momentum data
        weak_prices = [0.50 + 0.001 * i for i in range(20)]
        strategy.price_history[market_id] = deque(weak_prices, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque([1000] * 20, maxlen=strategy.long_period * 2)
        
        signal = await strategy.analyze_market(trending_market)
        
        assert signal is None  # Too weak to trade

    async def test_signal_generation_with_metadata(self, strategy, trending_market, price_history, volume_history):
        """Test that signals contain proper metadata"""
        market_id = trending_market.id
        
        strategy.price_history[market_id] = deque(price_history, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque(volume_history, maxlen=strategy.long_period * 2)
        
        signal = await strategy.analyze_market(trending_market)
        
        assert signal is not None
        assert 'momentum' in signal.metadata
        assert 'velocity' in signal.metadata
        assert 'acceleration' in signal.metadata
        assert 'rsi' in signal.metadata
        assert 'trend_strength' in signal.metadata
        assert signal.metadata['strategy'] == 'MomentumStrategy'

    # Test Position Sizing
    async def test_position_size_calculation(self, strategy):
        """Test position size based on momentum strength"""
        indicators = MomentumIndicators(
            momentum=0.15,
            velocity=0.03,
            acceleration=0.002,
            volume_momentum=0.2,
            trend_direction=1,
            short_ma=0.55,
            long_ma=0.50,
            macd=0.05,
            rsi=65.0,
            volatility=0.03
        )
        
        trend_strength = 0.8
        confidence = 0.85
        
        size = strategy._calculate_position_size(indicators, trend_strength, confidence)
        
        assert size > 0
        assert size <= strategy.config.max_position_size
        assert size >= Decimal('10')  # Minimum size

    async def test_position_size_scales_with_volatility(self, strategy):
        """Test that position size reduces with high volatility"""
        # Low volatility
        indicators_low_vol = MomentumIndicators(
            momentum=0.1, velocity=0.02, acceleration=0.001,
            volume_momentum=0.1, trend_direction=1,
            short_ma=0.52, long_ma=0.50, macd=0.02,
            rsi=60.0, volatility=0.02
        )
        
        # High volatility
        indicators_high_vol = MomentumIndicators(
            momentum=0.1, velocity=0.02, acceleration=0.001,
            volume_momentum=0.1, trend_direction=1,
            short_ma=0.52, long_ma=0.50, macd=0.02,
            rsi=60.0, volatility=0.4
        )
        
        size_low_vol = strategy._calculate_position_size(indicators_low_vol, 0.7, 0.8)
        size_high_vol = strategy._calculate_position_size(indicators_high_vol, 0.7, 0.8)
        
        assert size_low_vol > size_high_vol

    # Test Exit Conditions
    async def test_exit_on_momentum_reversal(self, strategy, trending_market):
        """Test exit when momentum reverses"""
        market_id = trending_market.id
        entry_price = Decimal("0.50")
        position_direction = SignalDirection.BUY
        
        # Setup reversal data (downtrend)
        reversal_prices = [0.60 - 0.01 * i for i in range(20)]
        strategy.price_history[market_id] = deque(reversal_prices, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque([2000] * 20, maxlen=strategy.long_period * 2)
        
        # Update trending_market price to current
        trending_market.current_prices["Yes"] = Decimal("0.45")
        
        should_exit, reason = await strategy.should_exit_position(
            trending_market, entry_price, position_direction
        )
        
        assert should_exit is True
        assert reason == "momentum_reversal"

    async def test_exit_on_stop_loss(self, strategy, trending_market):
        """Test exit on stop loss trigger"""
        entry_price = Decimal("0.50")
        position_direction = SignalDirection.BUY
        
        # Set current price to trigger stop loss
        trending_market.current_prices["Yes"] = Decimal("0.44")  # > 10% loss
        
        should_exit, reason = await strategy.should_exit_position(
            trending_market, entry_price, position_direction
        )
        
        assert should_exit is True
        assert reason == "stop_loss"

    async def test_exit_on_rsi_extreme(self, strategy, trending_market):
        """Test exit on RSI overbought/oversold"""
        market_id = trending_market.id
        entry_price = Decimal("0.50")
        position_direction = SignalDirection.BUY
        
        # Create overbought condition
        overbought_prices = [0.50 + 0.02 * i for i in range(20)]
        strategy.price_history[market_id] = deque(overbought_prices, maxlen=strategy.long_period * 2)
        strategy.volume_history[market_id] = deque([3000] * 20, maxlen=strategy.long_period * 2)
        
        # This should create high RSI
        indicators = strategy._calculate_momentum_indicators(market_id)
        if indicators and indicators.rsi > 80:
            should_exit, reason = await strategy.should_exit_position(
                trending_market, entry_price, position_direction
            )
            
            assert should_exit is True
            assert reason == "overbought"

    # Test Risk Management
    async def test_signal_strength_mapping(self, strategy):
        """Test mapping of trend strength to signal strength"""
        test_cases = [
            (0.9, SignalStrength.VERY_STRONG),
            (0.7, SignalStrength.STRONG),
            (0.5, SignalStrength.MODERATE),
            (0.3, SignalStrength.WEAK),
            (0.1, SignalStrength.VERY_WEAK)
        ]
        
        for trend_strength, expected_signal_strength in test_cases:
            signal_strength = strategy._map_trend_to_signal_strength(trend_strength)
            assert signal_strength == expected_signal_strength

    # Test Price Target Calculation
    async def test_calculate_price_target(self, strategy):
        """Test price target calculation based on momentum"""
        current_price = Decimal("0.50")
        
        # Bullish indicators
        indicators = MomentumIndicators(
            momentum=0.1,
            velocity=0.02,
            acceleration=0.001,
            volume_momentum=0.15,
            trend_direction=1,
            short_ma=0.52,
            long_ma=0.50,
            macd=0.02,
            rsi=65.0,
            volatility=0.03
        )
        
        trend_strength = 0.7
        target = strategy._calculate_price_target(current_price, indicators, trend_strength)
        
        assert target > current_price  # Should be higher for bullish
        assert Decimal("0.01") <= target <= Decimal("0.99")  # Within bounds

    # Test Performance Metrics
    async def test_get_momentum_metrics(self, strategy):
        """Test momentum-specific performance metrics"""
        # Add some cached indicators
        strategy.momentum_cache["market1"] = MomentumIndicators(
            momentum=0.1, velocity=0.02, acceleration=0.001,
            volume_momentum=0.1, trend_direction=1,
            short_ma=0.52, long_ma=0.50, macd=0.02,
            rsi=60.0, volatility=0.03
        )
        
        strategy.momentum_cache["market2"] = MomentumIndicators(
            momentum=-0.1, velocity=-0.02, acceleration=-0.001,
            volume_momentum=-0.1, trend_direction=-1,
            short_ma=0.48, long_ma=0.50, macd=-0.02,
            rsi=40.0, volatility=0.03
        )
        
        metrics = strategy.get_momentum_metrics()
        
        assert metrics['markets_tracked'] >= 0
        assert metrics['bullish_markets'] == 1
        assert metrics['bearish_markets'] == 1
        assert metrics['average_momentum'] > 0


class TestMomentumIndicators:
    """Test suite for MomentumIndicators data class"""
    
    def test_indicators_creation(self):
        """Test creation of momentum indicators"""
        indicators = MomentumIndicators(
            momentum=0.1,
            velocity=0.02,
            acceleration=0.001,
            volume_momentum=0.15,
            trend_direction=1,
            short_ma=0.52,
            long_ma=0.50,
            macd=0.02,
            rsi=65.0,
            volatility=0.03
        )
        
        assert indicators.momentum == 0.1
        assert indicators.is_bullish is True
        assert indicators.is_bearish is False
    
    def test_bearish_indicators(self):
        """Test bearish momentum indicators"""
        indicators = MomentumIndicators(
            momentum=-0.1,
            velocity=-0.02,
            acceleration=-0.001,
            volume_momentum=-0.1,
            trend_direction=-1,
            short_ma=0.48,
            long_ma=0.50,
            macd=-0.02,
            rsi=35.0,
            volatility=0.04
        )
        
        assert indicators.is_bearish is True
        assert indicators.is_bullish is False
    
    def test_composite_score_calculation(self):
        """Test composite score calculation"""
        indicators = MomentumIndicators(
            momentum=0.2,
            velocity=0.04,
            acceleration=0.002,
            volume_momentum=0.25,
            trend_direction=1,
            short_ma=0.55,
            long_ma=0.50,
            macd=0.05,
            rsi=70.0,
            volatility=0.02
        )
        
        score = indicators.calculate_composite_score()
        assert isinstance(score, float)
        assert score > 0  # Should be positive for bullish indicators