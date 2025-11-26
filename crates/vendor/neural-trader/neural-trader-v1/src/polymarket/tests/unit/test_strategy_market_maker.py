"""
Unit tests for market maker trading strategy

These tests follow TDD principles by testing the functionality before implementation.
Tests cover spread management, inventory risk, quote generation, and order management.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from ...strategies.market_maker import MarketMakerStrategy, MarketMakingSignal
from ...strategies.base import (
    TradingSignal, SignalStrength, SignalDirection, StrategyConfig, StrategyError
)
from ...models import Market, MarketStatus, Order, OrderSide, OrderStatus, OrderBook
from ...api import PolymarketClient


class TestMarketMakerStrategy:
    """Test suite for Market Maker Strategy"""

    @pytest.fixture
    async def strategy(self, mock_clob_client):
        """Create market maker strategy instance"""
        config = StrategyConfig(
            max_position_size=Decimal('1000.0'),
            min_confidence=0.5,
            min_signal_strength=SignalStrength.WEAK,
            max_markets_monitored=10
        )
        return MarketMakerStrategy(
            mock_clob_client, 
            config,
            spread_target=Decimal('0.02'),  # 2% spread
            max_inventory=Decimal('500.0'),
            inventory_skew_factor=Decimal('0.5')
        )

    @pytest.fixture
    def order_book(self):
        """Sample order book for testing"""
        return OrderBook(
            market_id="test-market",
            outcome_id="Yes",
            bids=[
                {"price": 0.48, "size": 100},
                {"price": 0.47, "size": 200},
                {"price": 0.46, "size": 150}
            ],
            asks=[
                {"price": 0.52, "size": 120},
                {"price": 0.53, "size": 180},
                {"price": 0.54, "size": 140}
            ],
            timestamp=datetime.now()
        )

    @pytest.fixture
    def liquid_market(self):
        """Sample liquid market for testing"""
        return Market(
            id="test-market",
            question="Will event X happen?",
            outcomes=["Yes", "No"],
            end_date=datetime.now() + timedelta(days=30),
            status=MarketStatus.ACTIVE,
            current_prices={"Yes": Decimal("0.50"), "No": Decimal("0.50")},
            metadata={
                'volume_24h': 100000,
                'liquidity': 50000,
                'spread': 0.04
            }
        )

    # Test Strategy Initialization
    async def test_strategy_initialization(self, mock_clob_client):
        """Test market maker strategy initialization"""
        strategy = MarketMakerStrategy(
            mock_clob_client,
            spread_target=Decimal('0.03'),
            max_inventory=Decimal('1000.0')
        )
        
        assert strategy.name == "MarketMakerStrategy"
        assert strategy.spread_target == Decimal('0.03')
        assert strategy.max_inventory == Decimal('1000.0')
        assert strategy.inventory_skew_factor == Decimal('0.3')  # default
        assert strategy.min_spread == Decimal('0.001')  # default
        assert strategy.max_spread == Decimal('0.1')  # default

    # Test Market Suitability
    async def test_should_trade_liquid_market(self, strategy, liquid_market):
        """Test that strategy identifies liquid markets"""
        result = await strategy.should_trade_market(liquid_market)
        assert result is True

    async def test_should_trade_illiquid_market(self, strategy, liquid_market):
        """Test rejection of illiquid markets"""
        liquid_market.metadata['volume_24h'] = 1000  # Low volume
        liquid_market.metadata['liquidity'] = 500
        
        result = await strategy.should_trade_market(liquid_market)
        assert result is False

    async def test_should_trade_expiring_market(self, strategy, liquid_market):
        """Test rejection of soon-expiring markets"""
        liquid_market.end_date = datetime.now() + timedelta(hours=2)
        
        result = await strategy.should_trade_market(liquid_market)
        assert result is False

    async def test_should_trade_inactive_market(self, strategy, liquid_market):
        """Test rejection of inactive markets"""
        liquid_market.status = MarketStatus.RESOLVED
        
        result = await strategy.should_trade_market(liquid_market)
        assert result is False

    # Test Quote Generation
    async def test_calculate_quotes_basic(self, strategy, order_book):
        """Test basic quote calculation"""
        inventory = Decimal('0')  # Neutral inventory
        
        bid_price, ask_price = strategy._calculate_quotes(order_book, inventory)
        
        # Should place quotes inside the best bid/ask
        assert bid_price > order_book.best_bid
        assert ask_price < order_book.best_ask
        # Should maintain target spread
        assert ask_price - bid_price >= strategy.spread_target

    async def test_calculate_quotes_with_inventory_skew(self, strategy, order_book):
        """Test quote adjustment based on inventory"""
        # Long inventory - should skew prices down to reduce position
        long_inventory = Decimal('300')
        bid_long, ask_long = strategy._calculate_quotes(order_book, long_inventory)
        
        # Short inventory - should skew prices up to reduce position
        short_inventory = Decimal('-300')
        bid_short, ask_short = strategy._calculate_quotes(order_book, short_inventory)
        
        # Long inventory should have lower prices than short
        assert bid_long < bid_short
        assert ask_long < ask_short

    async def test_calculate_quotes_max_inventory(self, strategy, order_book):
        """Test quote generation at max inventory"""
        # At max long inventory
        max_inventory = strategy.max_inventory
        bid_price, ask_price = strategy._calculate_quotes(order_book, max_inventory)
        
        # Should only quote on ask side (to reduce position)
        assert bid_price is None or bid_price < order_book.best_bid * Decimal('0.9')
        assert ask_price is not None

    async def test_calculate_quotes_respects_spread_limits(self, strategy, order_book):
        """Test that quotes respect min/max spread limits"""
        # Wide market
        order_book.bids = [{"price": 0.40, "size": 100}]
        order_book.asks = [{"price": 0.60, "size": 100}]
        
        bid_price, ask_price = strategy._calculate_quotes(order_book, Decimal('0'))
        
        spread = ask_price - bid_price
        assert spread >= strategy.min_spread
        assert spread <= strategy.max_spread

    # Test Order Size Calculation
    async def test_calculate_order_size_basic(self, strategy):
        """Test basic order size calculation"""
        inventory = Decimal('100')
        side = OrderSide.BUY
        
        size = strategy._calculate_order_size(inventory, side)
        
        assert size > 0
        assert size <= strategy.config.max_position_size

    async def test_calculate_order_size_reduces_with_inventory(self, strategy):
        """Test order size reduces as inventory increases"""
        # Size should decrease as inventory increases
        size_low = strategy._calculate_order_size(Decimal('100'), OrderSide.BUY)
        size_high = strategy._calculate_order_size(Decimal('400'), OrderSide.BUY)
        
        assert size_high < size_low

    async def test_calculate_order_size_zero_near_max(self, strategy):
        """Test order size goes to zero near max inventory"""
        near_max = strategy.max_inventory * Decimal('0.95')
        
        size = strategy._calculate_order_size(near_max, OrderSide.BUY)
        
        assert size == 0 or size < Decimal('10')

    # Test Signal Generation
    @patch('polymarket.strategies.market_maker.MarketMakerStrategy._get_order_book')
    async def test_analyze_market_generates_signals(self, mock_order_book, strategy, liquid_market, order_book):
        """Test market analysis generates appropriate signals"""
        mock_order_book.return_value = order_book
        strategy.inventory[liquid_market.id] = Decimal('0')
        
        signal = await strategy.analyze_market(liquid_market)
        
        assert signal is not None
        assert isinstance(signal, MarketMakingSignal)
        assert signal.bid_price is not None
        assert signal.ask_price is not None
        assert signal.bid_size > 0
        assert signal.ask_size > 0

    @patch('polymarket.strategies.market_maker.MarketMakerStrategy._get_order_book')
    async def test_analyze_market_no_signal_on_error(self, mock_order_book, strategy, liquid_market):
        """Test no signal generated on order book error"""
        mock_order_book.side_effect = Exception("API Error")
        
        signal = await strategy.analyze_market(liquid_market)
        
        assert signal is None

    # Test Inventory Management
    async def test_update_inventory_on_fill(self, strategy):
        """Test inventory updates on order fills"""
        market_id = "test-market"
        initial_inventory = Decimal('100')
        strategy.inventory[market_id] = initial_inventory
        
        # Buy fill
        strategy._update_inventory(market_id, OrderSide.BUY, Decimal('50'))
        assert strategy.inventory[market_id] == initial_inventory + Decimal('50')
        
        # Sell fill
        strategy._update_inventory(market_id, OrderSide.SELL, Decimal('30'))
        assert strategy.inventory[market_id] == initial_inventory + Decimal('50') - Decimal('30')

    async def test_get_inventory_position(self, strategy):
        """Test getting current inventory position"""
        market_id = "test-market"
        strategy.inventory[market_id] = Decimal('250')
        
        position = strategy.get_inventory_position(market_id)
        
        assert position == Decimal('250')
        
        # Non-existent market
        assert strategy.get_inventory_position("unknown") == Decimal('0')

    # Test Risk Management
    async def test_check_inventory_limits(self, strategy):
        """Test inventory limit checking"""
        market_id = "test-market"
        
        # Within limits
        strategy.inventory[market_id] = Decimal('300')
        assert strategy._check_inventory_limits(market_id, OrderSide.BUY, Decimal('100')) is True
        
        # Would exceed limit
        strategy.inventory[market_id] = Decimal('450')
        assert strategy._check_inventory_limits(market_id, OrderSide.BUY, Decimal('100')) is False
        
        # Sell reduces inventory, should be allowed
        assert strategy._check_inventory_limits(market_id, OrderSide.SELL, Decimal('100')) is True

    async def test_calculate_inventory_risk(self, strategy):
        """Test inventory risk calculation"""
        # Low risk
        risk_low = strategy._calculate_inventory_risk(Decimal('100'))
        
        # High risk
        risk_high = strategy._calculate_inventory_risk(Decimal('450'))
        
        assert risk_high > risk_low
        assert 0 <= risk_low <= 1
        assert 0 <= risk_high <= 1

    # Test Order Management
    async def test_manage_open_orders(self, strategy):
        """Test management of open orders"""
        market_id = "test-market"
        
        # Add some mock orders
        order1 = MagicMock(id="order1", market_id=market_id, status=OrderStatus.OPEN)
        order2 = MagicMock(id="order2", market_id=market_id, status=OrderStatus.FILLED)
        
        strategy.active_orders[market_id] = [order1, order2]
        
        # Should remove filled orders
        strategy._cleanup_orders(market_id)
        
        assert len(strategy.active_orders[market_id]) == 1
        assert strategy.active_orders[market_id][0].id == "order1"

    async def test_cancel_stale_orders(self, strategy):
        """Test cancellation of stale orders"""
        market_id = "test-market"
        
        # Create stale order
        stale_order = MagicMock(
            id="stale",
            market_id=market_id,
            status=OrderStatus.OPEN,
            created_at=datetime.now() - timedelta(minutes=10)
        )
        
        strategy.active_orders[market_id] = [stale_order]
        
        with patch.object(strategy.client, 'cancel_order') as mock_cancel:
            await strategy._cancel_stale_orders(market_id, max_age_seconds=300)
            
            mock_cancel.assert_called_once_with(stale_order.id)

    # Test Performance Metrics
    async def test_track_spread_capture(self, strategy):
        """Test tracking of spread capture"""
        market_id = "test-market"
        
        # Record some spread captures
        strategy._record_spread_capture(market_id, Decimal('0.02'))
        strategy._record_spread_capture(market_id, Decimal('0.025'))
        strategy._record_spread_capture(market_id, Decimal('0.015'))
        
        metrics = strategy.get_market_metrics(market_id)
        
        assert metrics['spread_captures'] == 3
        assert metrics['average_spread'] == Decimal('0.02')

    async def test_calculate_profitability(self, strategy):
        """Test profitability calculation"""
        market_id = "test-market"
        
        # Add some trades
        strategy._record_trade(market_id, OrderSide.BUY, Decimal('100'), Decimal('0.48'))
        strategy._record_trade(market_id, OrderSide.SELL, Decimal('100'), Decimal('0.52'))
        
        pnl = strategy.calculate_market_pnl(market_id, current_price=Decimal('0.50'))
        
        # Should have positive P&L from spread capture
        assert pnl > 0

    # Test Edge Cases
    async def test_handle_extreme_spreads(self, strategy, order_book):
        """Test handling of extreme spread conditions"""
        # Very wide spread
        order_book.bids = [{"price": 0.10, "size": 100}]
        order_book.asks = [{"price": 0.90, "size": 100}]
        
        bid_price, ask_price = strategy._calculate_quotes(order_book, Decimal('0'))
        
        # Should still generate reasonable quotes
        assert bid_price > Decimal('0.10')
        assert ask_price < Decimal('0.90')
        spread = ask_price - bid_price
        assert spread <= strategy.max_spread

    async def test_handle_one_sided_book(self, strategy):
        """Test handling of one-sided order books"""
        # Only bids
        order_book = OrderBook(
            market_id="test",
            outcome_id="Yes",
            bids=[{"price": 0.45, "size": 100}],
            asks=[],
            timestamp=datetime.now()
        )
        
        bid_price, ask_price = strategy._calculate_quotes(order_book, Decimal('0'))
        
        # Should generate ask based on spread target
        assert bid_price is not None
        assert ask_price is not None
        assert ask_price > bid_price

    async def test_handle_crossed_market(self, strategy):
        """Test handling of crossed markets"""
        order_book = OrderBook(
            market_id="test",
            outcome_id="Yes",
            bids=[{"price": 0.55, "size": 100}],  # Bid > Ask (crossed)
            asks=[{"price": 0.45, "size": 100}],
            timestamp=datetime.now()
        )
        
        # Should detect arbitrage opportunity
        signal = strategy._detect_arbitrage(order_book)
        assert signal is not None
        assert signal.is_arbitrage is True

    # Test Configuration
    async def test_dynamic_spread_adjustment(self, strategy):
        """Test dynamic spread adjustment based on volatility"""
        market_id = "test-market"
        
        # Low volatility
        spread_low = strategy._calculate_dynamic_spread(market_id, volatility=Decimal('0.01'))
        
        # High volatility
        spread_high = strategy._calculate_dynamic_spread(market_id, volatility=Decimal('0.05'))
        
        assert spread_high > spread_low
        assert spread_low >= strategy.min_spread
        assert spread_high <= strategy.max_spread

    async def test_adaptive_sizing(self, strategy):
        """Test adaptive order sizing based on market conditions"""
        # High liquidity market
        size_liquid = strategy._calculate_adaptive_size(
            base_size=Decimal('100'),
            liquidity=Decimal('100000'),
            volatility=Decimal('0.02')
        )
        
        # Low liquidity market
        size_illiquid = strategy._calculate_adaptive_size(
            base_size=Decimal('100'),
            liquidity=Decimal('10000'),
            volatility=Decimal('0.02')
        )
        
        assert size_illiquid < size_liquid


class TestMarketMakingSignal:
    """Test suite for MarketMakingSignal data class"""
    
    def test_signal_creation(self):
        """Test creation of market making signal"""
        signal = MarketMakingSignal(
            market_id="test-market",
            outcome="Yes",
            bid_price=Decimal('0.48'),
            ask_price=Decimal('0.52'),
            bid_size=Decimal('100'),
            ask_size=Decimal('100'),
            confidence=0.8,
            spread=Decimal('0.04'),
            inventory_risk=0.3
        )
        
        assert signal.market_id == "test-market"
        assert signal.spread == Decimal('0.04')
        assert signal.expected_profit == signal.spread * min(signal.bid_size, signal.ask_size) * Decimal(str(signal.confidence))
    
    def test_signal_validation(self):
        """Test signal validation"""
        # Invalid spread (ask < bid)
        with pytest.raises(ValueError):
            MarketMakingSignal(
                market_id="test",
                outcome="Yes",
                bid_price=Decimal('0.55'),
                ask_price=Decimal('0.45'),  # Invalid
                bid_size=Decimal('100'),
                ask_size=Decimal('100'),
                confidence=0.8,
                spread=Decimal('-0.10'),
                inventory_risk=0.3
            )
    
    def test_signal_profitability(self):
        """Test signal profitability calculation"""
        signal = MarketMakingSignal(
            market_id="test",
            outcome="Yes", 
            bid_price=Decimal('0.49'),
            ask_price=Decimal('0.51'),
            bid_size=Decimal('200'),
            ask_size=Decimal('150'),
            confidence=0.9,
            spread=Decimal('0.02'),
            inventory_risk=0.2
        )
        
        # Expected profit should be based on min size
        expected = Decimal('0.02') * Decimal('150') * Decimal('0.9')
        assert signal.expected_profit == expected