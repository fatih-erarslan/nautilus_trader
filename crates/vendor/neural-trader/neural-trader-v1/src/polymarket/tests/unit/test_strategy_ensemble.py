"""
Unit tests for ensemble trading strategy

These tests follow TDD principles by testing the functionality.
Tests cover strategy combination, voting mechanisms, weighting, and consensus.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec
from typing import Dict, List, Any
from collections import defaultdict

from ...strategies.ensemble import EnsembleStrategy, EnsembleSignal
from ...strategies.base import (
    TradingSignal, SignalStrength, SignalDirection, StrategyConfig, StrategyError
)
from ...strategies.news_sentiment import NewsSentimentStrategy
from ...strategies.momentum import MomentumStrategy
from ...strategies.arbitrage import ArbitrageStrategy
from ...strategies.market_maker import MarketMakerStrategy
from ...models import Market, MarketStatus, Order, OrderSide, OrderStatus
from ...api import PolymarketClient


class TestEnsembleStrategy:
    """Test suite for Ensemble Strategy"""

    @pytest.fixture
    async def strategy(self, mock_clob_client):
        """Create ensemble strategy instance"""
        config = StrategyConfig(
            max_position_size=Decimal('1000.0'),
            min_confidence=0.7,
            min_signal_strength=SignalStrength.MODERATE,
            max_markets_monitored=20
        )
        
        return EnsembleStrategy(
            mock_clob_client,
            config,
            min_strategies=2,
            min_consensus=0.6,
            use_confidence_weighting=True,
            adaptive_weights=True,
            include_market_maker=False
        )

    @pytest.fixture
    def test_market(self):
        """Sample market for testing"""
        return Market(
            id="test-market",
            question="Will event happen?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.50"), "No": Decimal("0.50")},
            metadata={'volume_24h': 50000, 'category': 'politics'}
        )

    @pytest.fixture
    def diverse_signals(self):
        """Diverse signals from different strategies"""
        return [
            TradingSignal(
                market_id="test-market",
                outcome="Yes",
                direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG,
                target_price=Decimal("0.65"),
                size=Decimal("100"),
                confidence=0.85,
                reasoning="Strong bullish sentiment",
                metadata={'strategy': 'news_sentiment'}
            ),
            TradingSignal(
                market_id="test-market",
                outcome="Yes",
                direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE,
                target_price=Decimal("0.62"),
                size=Decimal("80"),
                confidence=0.75,
                reasoning="Positive momentum detected",
                metadata={'strategy': 'momentum'}
            ),
            TradingSignal(
                market_id="test-market",
                outcome="Yes",
                direction=SignalDirection.SELL,
                strength=SignalStrength.WEAK,
                target_price=Decimal("0.48"),
                size=Decimal("50"),
                confidence=0.65,
                reasoning="Arbitrage opportunity",
                metadata={'strategy': 'arbitrage'}
            )
        ]

    # Test Strategy Initialization
    async def test_strategy_initialization(self, mock_clob_client):
        """Test ensemble strategy initialization"""
        config = StrategyConfig()
        strategy = EnsembleStrategy(
            mock_clob_client,
            config,
            min_strategies=2,
            min_consensus=0.6
        )
        
        assert strategy.name == "EnsembleStrategy"
        assert len(strategy.strategies) >= 3  # At least news, momentum, arbitrage
        assert strategy.min_strategies == 2
        assert strategy.min_consensus == 0.6
        assert strategy.use_confidence_weighting is True

    async def test_strategy_initialization_with_market_maker(self, mock_clob_client):
        """Test initialization including market maker strategy"""
        strategy = EnsembleStrategy(
            mock_clob_client,
            include_market_maker=True
        )
        
        assert 'market_maker' in strategy.strategies
        assert len(strategy.strategies) == 4

    async def test_initial_strategy_weights(self, strategy):
        """Test that initial strategy weights are equal"""
        total_weight = sum(strategy.strategy_weights.values())
        assert abs(total_weight - 1.0) < 0.001
        
        # All strategies should have equal weight initially
        expected_weight = 1.0 / len(strategy.strategies)
        for weight in strategy.strategy_weights.values():
            assert abs(weight - expected_weight) < 0.001

    # Test Market Suitability
    async def test_should_trade_market_consensus(self, strategy, test_market):
        """Test market suitability based on strategy consensus"""
        with patch.object(strategy.strategies['news_sentiment'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['momentum'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['arbitrage'], 'should_trade_market', return_value=False):
            
            result = await strategy.should_trade_market(test_market)
            
            # 2 out of 3 strategies say yes, meets min_strategies requirement
            assert result is True

    async def test_should_trade_market_insufficient_consensus(self, strategy, test_market):
        """Test rejection when insufficient consensus"""
        with patch.object(strategy.strategies['news_sentiment'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['momentum'], 'should_trade_market', return_value=False), \
             patch.object(strategy.strategies['arbitrage'], 'should_trade_market', return_value=False):
            
            result = await strategy.should_trade_market(test_market)
            
            # Only 1 strategy says yes, below min_strategies
            assert result is False

    # Test Signal Collection
    async def test_collect_component_signals(self, strategy, test_market, diverse_signals):
        """Test collection of signals from sub-strategies"""
        with patch.object(strategy.strategies['news_sentiment'], 'analyze_market', return_value=diverse_signals[0]), \
             patch.object(strategy.strategies['momentum'], 'analyze_market', return_value=diverse_signals[1]), \
             patch.object(strategy.strategies['arbitrage'], 'analyze_market', return_value=diverse_signals[2]):
            
            signals = await strategy._collect_component_signals(test_market)
            
            assert len(signals) == 3
            assert all('strategy' in s.metadata for s in signals)

    async def test_collect_signals_with_errors(self, strategy, test_market):
        """Test signal collection handles strategy errors gracefully"""
        good_signal = TradingSignal(
            market_id="test", outcome="Yes", direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG, target_price=Decimal("0.60"),
            size=Decimal("100"), confidence=0.8, reasoning="Test"
        )
        
        with patch.object(strategy.strategies['news_sentiment'], 'analyze_market', return_value=good_signal), \
             patch.object(strategy.strategies['momentum'], 'analyze_market', side_effect=Exception("API Error")), \
             patch.object(strategy.strategies['arbitrage'], 'analyze_market', return_value=None):
            
            signals = await strategy._collect_component_signals(test_market)
            
            # Should get 1 valid signal despite errors
            assert len(signals) == 1

    # Test Ensemble Signal Creation
    async def test_create_ensemble_signal_consensus(self, strategy, test_market, diverse_signals):
        """Test creation of ensemble signal with consensus"""
        # 2 BUY, 1 SELL signals
        ensemble_signal = strategy._create_ensemble_signal(test_market, diverse_signals)
        
        assert ensemble_signal is not None
        assert isinstance(ensemble_signal, EnsembleSignal)
        assert ensemble_signal.direction == SignalDirection.BUY  # Majority direction
        assert ensemble_signal.consensus_score >= 0.6
        assert len(ensemble_signal.component_signals) == 3

    async def test_create_ensemble_signal_weighted(self, strategy, test_market, diverse_signals):
        """Test weighted ensemble signal creation"""
        # Adjust weights to favor certain strategies
        strategy.strategy_weights = {
            'news_sentiment': 0.5,
            'momentum': 0.3,
            'arbitrage': 0.2
        }
        
        ensemble_signal = strategy._create_ensemble_signal(test_market, diverse_signals)
        
        assert ensemble_signal is not None
        # Should be BUY due to higher weights on BUY signals
        assert ensemble_signal.direction == SignalDirection.BUY

    async def test_ensemble_signal_confidence_calculation(self, strategy, test_market):
        """Test confidence calculation for ensemble signal"""
        unanimous_signals = [
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG, target_price=Decimal("0.60"),
                size=Decimal("100"), confidence=0.9, reasoning="Test",
                metadata={'strategy': f'strategy_{i}'}
            ) for i in range(3)
        ]
        
        ensemble_signal = strategy._create_ensemble_signal(test_market, unanimous_signals)
        
        assert ensemble_signal.is_unanimous is True
        assert ensemble_signal.consensus_score == 1.0
        # Confidence should be high due to unanimity
        assert ensemble_signal.confidence >= 0.85

    # Test Signal Strength Calculation
    async def test_calculate_ensemble_strength(self, strategy):
        """Test ensemble signal strength calculation"""
        strong_signals = [
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.VERY_STRONG, target_price=Decimal("0.70"),
                size=Decimal("100"), confidence=0.9, reasoning="Test"
            ) for _ in range(2)
        ]
        
        strength = strategy._calculate_ensemble_strength(strong_signals)
        assert strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]

    # Test Position Sizing
    async def test_calculate_position_size(self, strategy, diverse_signals):
        """Test position size calculation for ensemble"""
        # Test with high confidence weighted signals
        high_conf_signals = [
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG, target_price=Decimal("0.65"),
                size=Decimal("100"), confidence=0.9, reasoning="Test"
            ) for _ in range(2)
        ]
        
        weights = strategy._get_signal_weights(high_conf_signals)
        size = strategy._weighted_average(
            high_conf_signals,
            lambda s: float(s.size),
            weights
        )
        
        assert size > 0
        assert size <= float(strategy.config.max_position_size)

    # Test Adaptive Weights
    async def test_update_strategy_performance(self, strategy):
        """Test updating strategy performance tracking"""
        strategy.update_strategy_performance('news_sentiment', 0.05)  # 5% profit
        strategy.update_strategy_performance('momentum', -0.02)  # 2% loss
        
        assert 'news_sentiment' in strategy.strategy_performance
        assert len(strategy.strategy_performance['news_sentiment']) == 1

    async def test_adaptive_weight_adjustment(self, strategy):
        """Test adaptive weight adjustment based on performance"""
        # Simulate performance history
        for _ in range(10):
            strategy.update_strategy_performance('news_sentiment', 0.03)
            strategy.update_strategy_performance('momentum', 0.01)
            strategy.update_strategy_performance('arbitrage', -0.01)
        
        old_weights = strategy.strategy_weights.copy()
        strategy._update_adaptive_weights()
        
        # News sentiment should have higher weight due to better performance
        assert strategy.strategy_weights['news_sentiment'] > old_weights['news_sentiment']
        # Arbitrage should have lower weight due to negative performance
        assert strategy.strategy_weights['arbitrage'] < old_weights['arbitrage']
        # Weights should still sum to 1
        assert abs(sum(strategy.strategy_weights.values()) - 1.0) < 0.001

    # Test Market Analysis
    async def test_analyze_market_with_consensus(self, strategy, test_market):
        """Test complete market analysis with ensemble"""
        buy_signal = TradingSignal(
            market_id=test_market.id, outcome="Yes", direction=SignalDirection.BUY,
            strength=SignalStrength.MODERATE, target_price=Decimal("0.60"),
            size=Decimal("100"), confidence=0.75, reasoning="Test buy"
        )
        
        with patch.object(strategy.strategies['news_sentiment'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['momentum'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['arbitrage'], 'should_trade_market', return_value=True), \
             patch.object(strategy.strategies['news_sentiment'], 'analyze_market', return_value=buy_signal), \
             patch.object(strategy.strategies['momentum'], 'analyze_market', return_value=buy_signal), \
             patch.object(strategy.strategies['arbitrage'], 'analyze_market', return_value=None):
            
            signal = await strategy.analyze_market(test_market)
            
            assert signal is not None
            assert isinstance(signal, EnsembleSignal)
            assert signal.direction == SignalDirection.BUY
            assert len(signal.component_signals) == 2  # Two strategies provided signals

    async def test_analyze_market_insufficient_signals(self, strategy, test_market):
        """Test handling of insufficient signals"""
        single_signal = TradingSignal(
            market_id=test_market.id, outcome="Yes", direction=SignalDirection.BUY,
            strength=SignalStrength.MODERATE, target_price=Decimal("0.55"),
            size=Decimal("50"), confidence=0.7, reasoning="Test"
        )
        
        with patch.object(strategy, '_collect_component_signals', return_value=[single_signal]):
            signal = await strategy.analyze_market(test_market)
            
            # Should return None due to insufficient signals (need min 2)
            assert signal is None

    # Test Error Handling
    async def test_handle_all_strategies_fail(self, strategy, test_market):
        """Test handling when all strategies fail"""
        with patch.object(strategy, '_collect_component_signals', return_value=[]):
            signal = await strategy.analyze_market(test_market)
            assert signal is None

    # Test Performance Metrics
    async def test_get_ensemble_metrics(self, strategy):
        """Test ensemble performance metrics calculation"""
        # Add some signal history
        for i in range(5):
            signal = EnsembleSignal(
                market_id=f"market_{i}",
                outcome="Yes",
                direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE,
                target_price=Decimal("0.55"),
                size=Decimal("100"),
                confidence=0.75,
                reasoning="Test",
                component_signals=[],
                strategy_votes={'news_sentiment': 'buy', 'momentum': 'buy'},
                consensus_score=0.8,
                disagreement_level=0.2,
                dominant_strategy='news_sentiment'
            )
            strategy._update_signal_history(signal)
        
        metrics = strategy.get_ensemble_metrics()
        
        assert metrics['total_signals'] == 5
        assert 'average_consensus' in metrics
        assert 'strategy_weights' in metrics
        assert 'dominant_strategies' in metrics

    async def test_get_performance_summary(self, strategy):
        """Test comprehensive performance summary"""
        summary = strategy.get_performance_summary()
        
        assert 'ensemble_metrics' in summary
        assert 'component_strategies' in summary
        assert summary['strategy_name'] == 'EnsembleStrategy'

    # Test EnsembleSignal properties
    def test_ensemble_signal_properties(self):
        """Test EnsembleSignal properties and methods"""
        signal = EnsembleSignal(
            market_id="test",
            outcome="Yes",
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            target_price=Decimal("0.65"),
            size=Decimal("100"),
            confidence=0.85,
            reasoning="Ensemble consensus",
            component_signals=[
                TradingSignal(
                    market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                    strength=SignalStrength.STRONG, target_price=Decimal("0.65"),
                    size=Decimal("100"), confidence=0.9, reasoning="Test",
                    metadata={'strategy': 'news_sentiment'}
                ),
                TradingSignal(
                    market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                    strength=SignalStrength.MODERATE, target_price=Decimal("0.63"),
                    size=Decimal("80"), confidence=0.8, reasoning="Test",
                    metadata={'strategy': 'momentum'}
                )
            ],
            strategy_votes={'news_sentiment': 'buy', 'momentum': 'buy'},
            consensus_score=1.0,
            disagreement_level=0.0,
            dominant_strategy='news_sentiment'
        )
        
        assert signal.is_unanimous is True
        assert signal.is_strong_consensus is True
        
        weights = signal.get_strategy_weights()
        assert 'news_sentiment' in weights
        assert 'momentum' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.001

    # Test weighted averaging
    async def test_weighted_average_calculation(self, strategy):
        """Test weighted average calculation"""
        signals = [
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG, target_price=Decimal("0.70"),
                size=Decimal("100"), confidence=0.9, reasoning="Test"
            ),
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE, target_price=Decimal("0.60"),
                size=Decimal("50"), confidence=0.7, reasoning="Test"
            )
        ]
        
        weights = [0.6, 0.4]  # Higher weight for first signal
        
        avg_price = strategy._weighted_average(
            signals,
            lambda s: float(s.target_price),
            weights
        )
        
        # Should be closer to 0.70 due to higher weight
        expected = 0.70 * 0.6 + 0.60 * 0.4
        assert abs(avg_price - expected) < 0.001

    # Test reasoning generation
    async def test_ensemble_reasoning_generation(self, strategy):
        """Test reasoning generation for ensemble signals"""
        consensus_signals = [
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG, target_price=Decimal("0.65"),
                size=Decimal("100"), confidence=0.85, reasoning="News bullish",
                metadata={'strategy': 'news_sentiment'}
            ),
            TradingSignal(
                market_id="test", outcome="Yes", direction=SignalDirection.BUY,
                strength=SignalStrength.MODERATE, target_price=Decimal("0.62"),
                size=Decimal("80"), confidence=0.75, reasoning="Momentum up",
                metadata={'strategy': 'momentum'}
            )
        ]
        
        strategy_votes = {'news_sentiment': 'buy', 'momentum': 'buy', 'arbitrage': 'hold'}
        
        reasoning = strategy._generate_ensemble_reasoning(consensus_signals, strategy_votes)
        
        assert "consensus" in reasoning.lower()
        assert "2/3" in reasoning  # 2 out of 3 strategies agree
        assert "news_sentiment" in reasoning
        assert "momentum" in reasoning