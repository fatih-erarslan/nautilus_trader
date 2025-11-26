"""Test suite for Polymarket data models."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.polymarket.models.market import Market, MarketOutcome, MarketStatus, MarketCategory
from src.polymarket.models.order import Order, OrderType, OrderSide, OrderStatus
from src.polymarket.models.position import Position, PositionStatus
from src.polymarket.models.common import TokenInfo, Resolution, TimeFrame


class TestMarketModels:
    """Test cases for market-related models."""
    
    def test_market_outcome_creation(self):
        """Test MarketOutcome model creation and validation."""
        outcome = MarketOutcome(
            id="0x456",
            name="Yes",
            price=0.65,
            liquidity=100000.0,
            volume=500000.0,
            outcome_index=0
        )
        
        assert outcome.id == "0x456"
        assert outcome.name == "Yes"
        assert outcome.price == 0.65
        assert outcome.liquidity == 100000.0
        assert outcome.outcome_index == 0
        
        # Test price validation
        with pytest.raises(ValueError):
            MarketOutcome(id="0x789", name="No", price=1.5)  # Price > 1
        
        with pytest.raises(ValueError):
            MarketOutcome(id="0x789", name="No", price=-0.1)  # Price < 0
    
    def test_market_creation(self):
        """Test Market model creation and methods."""
        outcomes = [
            MarketOutcome(id="0x456", name="Yes", price=0.70),
            MarketOutcome(id="0x789", name="No", price=0.30)
        ]
        
        market = Market(
            id="0x123",
            question="Will BTC reach $100k by 2025?",
            slug="btc-100k-2025",
            status=MarketStatus.ACTIVE,
            outcomes=outcomes,
            volume=1000000.0,
            liquidity=500000.0,
            created_at=datetime.now(timezone.utc),
            end_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            category=MarketCategory.CRYPTO,
            tags=["bitcoin", "price", "prediction"]
        )
        
        assert market.id == "0x123"
        assert market.question == "Will BTC reach $100k by 2025?"
        assert market.status == MarketStatus.ACTIVE
        assert len(market.outcomes) == 2
        assert market.category == MarketCategory.CRYPTO
        assert "bitcoin" in market.tags
        
        # Test outcome lookup
        yes_outcome = market.get_outcome_by_name("Yes")
        assert yes_outcome is not None
        assert yes_outcome.price == 0.70
        
        # Test market properties
        assert market.is_active
        assert not market.is_resolved
        assert market.total_probability == 1.0
        
        # Test spread calculation
        spread = market.get_spread()
        assert spread == 0.40  # 0.70 - 0.30
    
    def test_market_resolution(self):
        """Test market resolution functionality."""
        market = Market(
            id="0x123",
            question="Test market",
            status=MarketStatus.RESOLVED,
            resolution=Resolution(
                outcome_id="0x456",
                resolved_at=datetime.now(timezone.utc),
                resolution_source="Official announcement"
            )
        )
        
        assert market.is_resolved
        assert market.resolution.outcome_id == "0x456"
        assert market.resolution.resolution_source == "Official announcement"


class TestOrderModels:
    """Test cases for order-related models."""
    
    def test_order_creation(self):
        """Test Order model creation and validation."""
        order = Order(
            id="order_123",
            market_id="0x123",
            outcome_id="0x456",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=0.65,
            size=100,
            filled=30,
            remaining=70,
            status=OrderStatus.PARTIAL,
            created_at=datetime.now(timezone.utc)
        )
        
        assert order.id == "order_123"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.price == 0.65
        assert order.filled == 30
        assert order.remaining == 70
        assert order.status == OrderStatus.PARTIAL
        
        # Test calculated properties
        assert order.fill_percentage == 30.0
        assert order.is_active
        assert not order.is_complete
    
    def test_order_status_transitions(self):
        """Test order status transitions and validation."""
        order = Order(
            id="order_123",
            market_id="0x123",
            outcome_id="0x456",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            size=100,
            status=OrderStatus.OPEN
        )
        
        # Test status transitions
        assert order.can_cancel()
        assert not order.is_complete
        
        # Update to filled
        order.status = OrderStatus.FILLED
        order.filled = 100
        order.remaining = 0
        
        assert not order.can_cancel()
        assert order.is_complete
        assert order.fill_percentage == 100.0
    
    def test_order_validation(self):
        """Test order validation rules."""
        # Test invalid price for limit order
        with pytest.raises(ValueError):
            Order(
                id="order_123",
                market_id="0x123",
                outcome_id="0x456",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                size=100,
                price=None  # Limit order requires price
            )
        
        # Test invalid size
        with pytest.raises(ValueError):
            Order(
                id="order_123",
                market_id="0x123",
                outcome_id="0x456",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                size=-10  # Negative size
            )


class TestPositionModels:
    """Test cases for position-related models."""
    
    def test_position_creation(self):
        """Test Position model creation and calculations."""
        position = Position(
            id="pos_123",
            market_id="0x123",
            outcome_id="0x456",
            size=1000,
            avg_price=0.60,
            current_price=0.65,
            market_question="Will ETH reach $5k?",
            outcome_name="Yes",
            status=PositionStatus.OPEN
        )
        
        assert position.id == "pos_123"
        assert position.size == 1000
        assert position.avg_price == 0.60
        assert position.current_price == 0.65
        
        # Test PnL calculations
        assert position.pnl == 50.0  # (0.65 - 0.60) * 1000
        assert position.pnl_percent == pytest.approx(8.33, 0.01)  # 50 / 600
        assert position.market_value == 650.0  # 0.65 * 1000
        assert position.cost_basis == 600.0  # 0.60 * 1000
    
    def test_position_with_losses(self):
        """Test position with negative PnL."""
        position = Position(
            id="pos_456",
            market_id="0x123",
            outcome_id="0x789",
            size=500,
            avg_price=0.80,
            current_price=0.70,
            market_question="Test market",
            outcome_name="No"
        )
        
        assert position.pnl == -50.0  # (0.70 - 0.80) * 500
        assert position.pnl_percent == pytest.approx(-12.5, 0.01)
        assert position.is_profitable == False
    
    def test_closed_position(self):
        """Test closed position calculations."""
        position = Position(
            id="pos_789",
            market_id="0x123",
            outcome_id="0x456",
            size=200,
            avg_price=0.40,
            exit_price=0.60,
            current_price=0.65,  # Ignored for closed positions
            status=PositionStatus.CLOSED,
            closed_at=datetime.now(timezone.utc)
        )
        
        assert position.status == PositionStatus.CLOSED
        assert position.pnl == 40.0  # (0.60 - 0.40) * 200
        assert position.pnl_percent == 50.0  # 40 / 80
        # For closed positions, use exit price not current price
        assert position.market_value == 120.0  # 0.60 * 200


class TestCommonModels:
    """Test cases for common/shared models."""
    
    def test_token_info(self):
        """Test TokenInfo model."""
        token = TokenInfo(
            symbol="USDC",
            name="USD Coin",
            decimals=6,
            address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        )
        
        assert token.symbol == "USDC"
        assert token.decimals == 6
        
        # Test amount conversion
        raw_amount = 1000000  # 1 USDC in raw units
        human_amount = token.to_human_readable(raw_amount)
        assert human_amount == Decimal("1.0")
        
        # Test reverse conversion
        raw_amount = token.to_raw_amount(Decimal("10.5"))
        assert raw_amount == 10500000
    
    def test_resolution_model(self):
        """Test Resolution model."""
        resolution = Resolution(
            outcome_id="0x456",
            resolved_at=datetime.now(timezone.utc),
            resolution_source="Oracle consensus",
            disputed=False
        )
        
        assert resolution.outcome_id == "0x456"
        assert resolution.resolution_source == "Oracle consensus"
        assert not resolution.disputed
    
    def test_timeframe_enum(self):
        """Test TimeFrame enum functionality."""
        assert TimeFrame.HOUR.to_seconds() == 3600
        assert TimeFrame.DAY.to_seconds() == 86400
        assert TimeFrame.WEEK.to_seconds() == 604800
        assert TimeFrame.MONTH.to_seconds() == 2592000  # 30 days
        
        # Test string conversion
        assert str(TimeFrame.HOUR) == "1h"
        assert str(TimeFrame.DAY) == "1d"
        assert str(TimeFrame.WEEK) == "1w"


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_market_serialization(self):
        """Test market model JSON serialization."""
        market = Market(
            id="0x123",
            question="Test market",
            status=MarketStatus.ACTIVE,
            outcomes=[
                MarketOutcome(id="0x456", name="Yes", price=0.60),
                MarketOutcome(id="0x789", name="No", price=0.40)
            ],
            created_at=datetime.now(timezone.utc)
        )
        
        # Convert to dict
        market_dict = market.to_dict()
        assert market_dict["id"] == "0x123"
        assert market_dict["status"] == "active"
        assert len(market_dict["outcomes"]) == 2
        
        # Create from dict
        market2 = Market.from_dict(market_dict)
        assert market2.id == market.id
        assert market2.question == market.question
        assert len(market2.outcomes) == len(market.outcomes)
    
    def test_order_serialization(self):
        """Test order model JSON serialization."""
        order = Order(
            id="order_123",
            market_id="0x123",
            outcome_id="0x456",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=0.65,
            size=100,
            status=OrderStatus.OPEN
        )
        
        order_dict = order.to_dict()
        assert order_dict["side"] == "buy"
        assert order_dict["type"] == "limit"
        assert order_dict["price"] == 0.65
        
        order2 = Order.from_dict(order_dict)
        assert order2.side == OrderSide.BUY
        assert order2.type == OrderType.LIMIT