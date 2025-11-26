"""
Unit tests for arbitrage trading strategy

These tests follow TDD principles by testing the functionality before implementation.
Tests cover cross-market arbitrage, YES/NO arbitrage, execution, and risk management.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from ...strategies.arbitrage import ArbitrageStrategy, ArbitrageSignal, ArbitrageType
from ...strategies.base import (
    TradingSignal, SignalStrength, SignalDirection, StrategyConfig, StrategyError
)
from ...models import Market, MarketStatus, Order, OrderSide, OrderStatus
from ...api import PolymarketClient


class TestArbitrageStrategy:
    """Test suite for Arbitrage Strategy"""

    @pytest.fixture
    async def strategy(self, mock_clob_client):
        """Create arbitrage strategy instance"""
        config = StrategyConfig(
            max_position_size=Decimal('1000.0'),
            min_confidence=0.8,
            min_signal_strength=SignalStrength.MODERATE,
            max_markets_monitored=50
        )
        return ArbitrageStrategy(
            mock_clob_client,
            config,
            min_profit_threshold=Decimal('0.01'),  # 1% minimum profit
            transaction_cost=Decimal('0.002'),     # 0.2% transaction cost
            correlation_threshold=0.7
        )

    @pytest.fixture
    def correlated_markets(self):
        """Sample correlated markets for testing"""
        markets = [
            Market(
                id="election-winner",
                question="Will candidate A win the election?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.65"), "No": Decimal("0.35")},
                metadata={'volume_24h': 50000, 'category': 'politics'}
            ),
            Market(
                id="election-loser",
                question="Will candidate B lose the election?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.62"), "No": Decimal("0.38")},  # Should be ~0.65
                metadata={'volume_24h': 45000, 'category': 'politics'}
            )
        ]
        return markets

    @pytest.fixture
    def correlation_matrix(self):
        """Sample correlation matrix"""
        return np.array([
            [1.0, -0.95],  # Strong negative correlation
            [-0.95, 1.0]
        ])

    # Test Strategy Initialization
    async def test_strategy_initialization(self, mock_clob_client):
        """Test arbitrage strategy initialization"""
        strategy = ArbitrageStrategy(
            mock_clob_client,
            min_profit_threshold=Decimal('0.02'),
            transaction_cost=Decimal('0.001')
        )
        
        assert strategy.name == "ArbitrageStrategy"
        assert strategy.min_profit_threshold == Decimal('0.02')
        assert strategy.transaction_cost == Decimal('0.001')
        assert strategy.correlation_threshold == 0.7
        assert strategy.max_price_deviation == Decimal('0.1')

    # Test Market Suitability
    async def test_should_trade_active_markets(self, strategy, correlated_markets):
        """Test that strategy accepts active markets"""
        for market in correlated_markets:
            result = await strategy.should_trade_market(market)
            assert result is True

    async def test_should_trade_resolved_market(self, strategy):
        """Test rejection of resolved markets"""
        market = Market(
            id="resolved",
            question="Old question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() - timedelta(days=1),
            status=MarketStatus.RESOLVED,
            current_prices={"Yes": Decimal("1.0"), "No": Decimal("0.0")},
            metadata={}
        )
        
        result = await strategy.should_trade_market(market)
        assert result is False

    # Test YES/NO Arbitrage Detection
    async def test_detect_yes_no_arbitrage_profitable(self, strategy):
        """Test detection of profitable YES/NO arbitrage"""
        market = Market(
            id="arb-market",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=10),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.45"), "No": Decimal("0.50")},  # Sum = 0.95
            metadata={'volume_24h': 10000}
        )
        
        signal = await strategy._detect_yes_no_arbitrage(market)
        
        assert signal is not None
        assert signal.arbitrage_type == ArbitrageType.YES_NO
        assert signal.expected_profit > 0
        assert signal.confidence >= 0.95  # Very high confidence for YES/NO arb

    async def test_detect_yes_no_arbitrage_not_profitable(self, strategy):
        """Test no signal when YES/NO arbitrage not profitable"""
        market = Market(
            id="no-arb",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=10),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.50"), "No": Decimal("0.50")},  # Sum = 1.0
            metadata={'volume_24h': 10000}
        )
        
        signal = await strategy._detect_yes_no_arbitrage(market)
        
        assert signal is None

    async def test_detect_yes_no_arbitrage_sell_opportunity(self, strategy):
        """Test detection of YES/NO sell arbitrage (sum > 1)"""
        market = Market(
            id="sell-arb",
            question="Test question?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=10),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.55"), "No": Decimal("0.52")},  # Sum = 1.07
            metadata={'volume_24h': 20000}
        )
        
        signal = await strategy._detect_yes_no_arbitrage(market)
        
        assert signal is not None
        assert signal.arbitrage_type == ArbitrageType.YES_NO
        assert signal.trade_direction == "sell"  # Sell both YES and NO
        assert signal.expected_profit > 0

    # Test Cross-Market Arbitrage
    async def test_detect_cross_market_arbitrage(self, strategy, correlated_markets, correlation_matrix):
        """Test detection of cross-market arbitrage opportunities"""
        with patch.object(strategy, '_calculate_correlation_matrix') as mock_corr:
            mock_corr.return_value = correlation_matrix
            
            signals = await strategy._detect_cross_market_arbitrage(correlated_markets)
            
            assert len(signals) > 0
            signal = signals[0]
            assert signal.arbitrage_type == ArbitrageType.CROSS_MARKET
            assert signal.confidence > 0.7

    async def test_cross_market_no_arbitrage(self, strategy):
        """Test no signal when markets are properly priced"""
        markets = [
            Market(
                id="market1",
                question="Question 1?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.60"), "No": Decimal("0.40")},
                metadata={'volume_24h': 30000}
            ),
            Market(
                id="market2",
                question="Question 2?",
                outcomes=["Yes", "No"],
                end_date=datetime.now() + timedelta(days=30),
                status=MarketStatus.ACTIVE,
                current_prices={"Yes": Decimal("0.40"), "No": Decimal("0.60")},
                metadata={'volume_24h': 25000}
            )
        ]
        
        # Perfect negative correlation, properly priced
        correlation_matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        with patch.object(strategy, '_calculate_correlation_matrix') as mock_corr:
            mock_corr.return_value = correlation_matrix
            
            signals = await strategy._detect_cross_market_arbitrage(markets)
            
            assert len(signals) == 0

    # Test Temporal Arbitrage
    async def test_detect_temporal_arbitrage(self, strategy):
        """Test detection of temporal arbitrage (mean reversion)"""
        price_history = {
            "volatile-market": [
                (datetime.now() - timedelta(hours=i), 0.50 + 0.01 * np.sin(i))
                for i in range(24)
            ]
        }
        
        # Current price significantly deviates from mean
        current_prices = {"volatile-market": Decimal("0.65")}  # High deviation
        
        signals = await strategy._detect_temporal_arbitrage(price_history, current_prices)
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.arbitrage_type == ArbitrageType.TEMPORAL
        assert signal.trade_direction == "sell"  # Price above mean

    async def test_temporal_arbitrage_insufficient_history(self, strategy):
        """Test no temporal arbitrage with insufficient history"""
        price_history = {
            "new-market": [(datetime.now(), 0.50)]  # Only 1 data point
        }
        current_prices = {"new-market": Decimal("0.55")}
        
        signals = await strategy._detect_temporal_arbitrage(price_history, current_prices)
        
        assert len(signals) == 0

    # Test Position Sizing
    async def test_calculate_optimal_position_size(self, strategy):
        """Test Kelly criterion position sizing"""
        signal = ArbitrageSignal(
            market_a_id="market1",
            market_b_id="market2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.05'),
            confidence=0.85,
            trade_direction="buy",
            price_a=Decimal('0.45'),
            price_b=Decimal('0.38'),
            size_a=Decimal('100'),
            size_b=Decimal('100')
        )
        
        optimal_size = strategy._calculate_optimal_size(signal)
        
        assert optimal_size > 0
        assert optimal_size <= strategy.config.max_position_size
        # Kelly sizing should be conservative (quarter Kelly)
        assert optimal_size < strategy.config.max_position_size * Decimal('0.25')

    async def test_position_size_respects_liquidity(self, strategy):
        """Test position sizing respects market liquidity"""
        signal = ArbitrageSignal(
            market_a_id="illiquid",
            market_b_id="illiquid2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.10'),
            confidence=0.9,
            trade_direction="buy",
            price_a=Decimal('0.50'),
            price_b=Decimal('0.40'),
            size_a=Decimal('10'),  # Very small liquidity
            size_b=Decimal('15')
        )
        
        size = strategy._calculate_optimal_size(signal)
        
        # Should be limited by available liquidity
        assert size <= min(signal.size_a, signal.size_b)

    # Test Signal Generation
    async def test_analyze_market_generates_signals(self, strategy):
        """Test market analysis for arbitrage opportunities"""
        market = Market(
            id="arb-opp",
            question="Arbitrage opportunity?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=20),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.44"), "No": Decimal("0.50")},  # Sum = 0.94
            metadata={'volume_24h': 50000}
        )
        
        signal = await strategy.analyze_market(market)
        
        assert signal is not None
        assert isinstance(signal, ArbitrageSignal)
        assert signal.confidence >= strategy.config.min_confidence

    async def test_batch_market_analysis(self, strategy, correlated_markets):
        """Test batch analysis of multiple markets"""
        with patch.object(strategy, '_calculate_correlation_matrix') as mock_corr:
            mock_corr.return_value = np.array([[1.0, -0.9], [-0.9, 1.0]])
            
            signals = await strategy.analyze_markets(correlated_markets)
            
            assert len(signals) >= 1
            # Should detect both YES/NO and cross-market opportunities

    # Test Risk Management
    async def test_risk_limits_enforcement(self, strategy):
        """Test that risk limits are enforced"""
        signal = ArbitrageSignal(
            market_a_id="risky",
            market_b_id="risky2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.005'),  # Below threshold
            confidence=0.6,  # Below min confidence
            trade_direction="buy",
            price_a=Decimal('0.50'),
            price_b=Decimal('0.49'),
            size_a=Decimal('1000'),
            size_b=Decimal('1000')
        )
        
        # Should be filtered out due to low profit/confidence
        is_valid = strategy._validate_arbitrage_signal(signal)
        assert is_valid is False

    async def test_max_exposure_limit(self, strategy):
        """Test maximum exposure limits"""
        # Simulate existing positions
        strategy.active_positions = {
            "pos1": {"size": Decimal('400'), "market_id": "market1"},
            "pos2": {"size": Decimal('400'), "market_id": "market2"}
        }
        
        signal = ArbitrageSignal(
            market_a_id="new1",
            market_b_id="new2",
            arbitrage_type=ArbitrageType.YES_NO,
            expected_profit=Decimal('0.05'),
            confidence=0.9,
            trade_direction="buy",
            price_a=Decimal('0.45'),
            price_b=Decimal('0.45'),
            size_a=Decimal('500'),
            size_b=Decimal('500')
        )
        
        # Should reduce position size due to existing exposure
        can_trade = strategy._check_risk_limits(signal)
        assert can_trade is False  # Would exceed max position

    # Test Execution
    async def test_execute_arbitrage_yes_no(self, strategy, mock_clob_client):
        """Test execution of YES/NO arbitrage"""
        signal = ArbitrageSignal(
            market_a_id="market1",
            market_b_id="market1",  # Same market
            arbitrage_type=ArbitrageType.YES_NO,
            expected_profit=Decimal('0.05'),
            confidence=0.95,
            trade_direction="buy",
            price_a=Decimal('0.45'),  # YES price
            price_b=Decimal('0.48'),  # NO price
            size_a=Decimal('100'),
            size_b=Decimal('100')
        )
        
        with patch.object(strategy, '_place_arbitrage_orders') as mock_place:
            mock_place.return_value = [
                MagicMock(id="order1", status=OrderStatus.FILLED),
                MagicMock(id="order2", status=OrderStatus.FILLED)
            ]
            
            result = await strategy.execute_signal(signal)
            
            assert result is not None
            mock_place.assert_called_once()

    async def test_execute_arbitrage_atomic(self, strategy):
        """Test atomic execution of arbitrage trades"""
        signal = ArbitrageSignal(
            market_a_id="market1",
            market_b_id="market2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.03'),
            confidence=0.85,
            trade_direction="buy",
            price_a=Decimal('0.40'),
            price_b=Decimal('0.35'),
            size_a=Decimal('50'),
            size_b=Decimal('50')
        )
        
        with patch.object(strategy, '_place_arbitrage_orders') as mock_place:
            # Simulate one order failing
            mock_place.return_value = [
                MagicMock(id="order1", status=OrderStatus.FILLED),
                MagicMock(id="order2", status=OrderStatus.REJECTED)
            ]
            
            result = await strategy.execute_signal(signal)
            
            # Should handle partial execution
            assert result is not None
            # Should cancel the filled order or hedge

    # Test Performance Tracking
    async def test_track_arbitrage_performance(self, strategy):
        """Test tracking of arbitrage performance"""
        # Execute some arbitrage trades
        strategy._record_arbitrage_result(
            arbitrage_type=ArbitrageType.YES_NO,
            expected_profit=Decimal('0.05'),
            realized_profit=Decimal('0.048'),
            slippage=Decimal('0.002')
        )
        
        strategy._record_arbitrage_result(
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.03'),
            realized_profit=Decimal('0.025'),
            slippage=Decimal('0.005')
        )
        
        metrics = strategy.get_arbitrage_metrics()
        
        assert metrics['total_arbitrages'] == 2
        assert metrics['success_rate'] == 1.0
        assert metrics['average_slippage'] == Decimal('0.0035')
        assert 'profit_by_type' in metrics

    # Test Edge Cases
    async def test_handle_market_suspension(self, strategy):
        """Test handling of market suspension during arbitrage"""
        market = Market(
            id="suspended",
            question="Suspended market?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=10),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.40"), "No": Decimal("0.40")},
            metadata={'suspended': True}
        )
        
        signal = await strategy.analyze_market(market)
        assert signal is None

    async def test_handle_price_updates_during_execution(self, strategy):
        """Test handling of price updates during execution"""
        signal = ArbitrageSignal(
            market_a_id="volatile1",
            market_b_id="volatile2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.02'),
            confidence=0.8,
            trade_direction="buy",
            price_a=Decimal('0.50'),
            price_b=Decimal('0.45'),
            size_a=Decimal('100'),
            size_b=Decimal('100')
        )
        
        # Simulate price change during execution
        with patch.object(strategy, '_get_current_prices') as mock_prices:
            mock_prices.return_value = {
                "volatile1": Decimal('0.52'),  # Price moved against us
                "volatile2": Decimal('0.48')
            }
            
            # Should detect that arbitrage no longer exists
            is_valid = await strategy._revalidate_arbitrage(signal)
            assert is_valid is False

    async def test_correlation_calculation(self, strategy):
        """Test correlation matrix calculation"""
        markets = [
            Market(id="m1", question="Q1?", outcomes=["Yes", "No"], 
                  end_date=datetime.now() + timedelta(days=30),
                  status=MarketStatus.ACTIVE,
                  current_prices={"Yes": Decimal("0.6")},
                  metadata={}),
            Market(id="m2", question="Q2?", outcomes=["Yes", "No"],
                  end_date=datetime.now() + timedelta(days=30), 
                  status=MarketStatus.ACTIVE,
                  current_prices={"Yes": Decimal("0.4")},
                  metadata={})
        ]
        
        # Mock historical prices
        with patch.object(strategy, '_get_price_history') as mock_history:
            mock_history.return_value = {
                "m1": [0.6, 0.62, 0.58, 0.61, 0.59],
                "m2": [0.4, 0.38, 0.42, 0.39, 0.41]
            }
            
            corr_matrix = strategy._calculate_correlation_matrix(markets)
            
            assert corr_matrix.shape == (2, 2)
            assert corr_matrix[0, 0] == 1.0  # Self correlation
            assert -1 <= corr_matrix[0, 1] <= 1  # Valid correlation range


class TestArbitrageSignal:
    """Test suite for ArbitrageSignal data class"""
    
    def test_signal_creation(self):
        """Test creation of arbitrage signal"""
        signal = ArbitrageSignal(
            market_a_id="market1",
            market_b_id="market2",
            arbitrage_type=ArbitrageType.CROSS_MARKET,
            expected_profit=Decimal('0.05'),
            confidence=0.85,
            trade_direction="buy",
            price_a=Decimal('0.45'),
            price_b=Decimal('0.40'),
            size_a=Decimal('100'),
            size_b=Decimal('150')
        )
        
        assert signal.market_a_id == "market1"
        assert signal.expected_profit == Decimal('0.05')
        assert signal.is_profitable is True
        assert signal.risk_adjusted_profit == signal.expected_profit * Decimal(str(signal.confidence))
    
    def test_yes_no_signal(self):
        """Test YES/NO arbitrage signal"""
        signal = ArbitrageSignal(
            market_a_id="arb-market",
            market_b_id="arb-market",  # Same market for YES/NO
            arbitrage_type=ArbitrageType.YES_NO,
            expected_profit=Decimal('0.06'),
            confidence=0.99,
            trade_direction="buy",
            price_a=Decimal('0.44'),  # YES
            price_b=Decimal('0.50'),  # NO
            size_a=Decimal('200'),
            size_b=Decimal('200')
        )
        
        assert signal.arbitrage_type == ArbitrageType.YES_NO
        assert signal.total_cost == (signal.price_a + signal.price_b) * min(signal.size_a, signal.size_b)
        assert signal.confidence == 0.99  # Very high for YES/NO arb
    
    def test_signal_validation(self):
        """Test signal validation"""
        # Negative profit
        with pytest.raises(ValueError):
            ArbitrageSignal(
                market_a_id="m1",
                market_b_id="m2",
                arbitrage_type=ArbitrageType.CROSS_MARKET,
                expected_profit=Decimal('-0.01'),  # Invalid
                confidence=0.8,
                trade_direction="buy",
                price_a=Decimal('0.5'),
                price_b=Decimal('0.5'),
                size_a=Decimal('100'),
                size_b=Decimal('100')
            )