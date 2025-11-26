"""
Comprehensive unit tests for Order model.

Tests all methods, properties, validation rules, edge cases, and error conditions
to achieve 100% coverage for the Order and related models.
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional

from polymarket.models.order import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce, OrderFill, OrderBook
)


class TestOrderType:
    """Test OrderType enumeration."""
    
    @pytest.mark.unit
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
    
    @pytest.mark.unit
    def test_requires_price_property(self):
        """Test requires_price property."""
        assert OrderType.MARKET.requires_price is False
        assert OrderType.LIMIT.requires_price is True
        assert OrderType.STOP.requires_price is False
        assert OrderType.STOP_LIMIT.requires_price is True
    
    @pytest.mark.unit
    def test_requires_stop_price_property(self):
        """Test requires_stop_price property."""
        assert OrderType.MARKET.requires_stop_price is False
        assert OrderType.LIMIT.requires_stop_price is False
        assert OrderType.STOP.requires_stop_price is True
        assert OrderType.STOP_LIMIT.requires_stop_price is True


class TestOrderSide:
    """Test OrderSide enumeration."""
    
    @pytest.mark.unit
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderStatus:
    """Test OrderStatus enumeration."""
    
    @pytest.mark.unit
    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIAL.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
    
    @pytest.mark.unit
    def test_is_active_property(self):
        """Test is_active property."""
        assert OrderStatus.PENDING.is_active is False
        assert OrderStatus.OPEN.is_active is True
        assert OrderStatus.PARTIAL.is_active is True
        assert OrderStatus.FILLED.is_active is False
        assert OrderStatus.CANCELLED.is_active is False
        assert OrderStatus.REJECTED.is_active is False
        assert OrderStatus.EXPIRED.is_active is False
    
    @pytest.mark.unit
    def test_is_complete_property(self):
        """Test is_complete property."""
        assert OrderStatus.PENDING.is_complete is False
        assert OrderStatus.OPEN.is_complete is False
        assert OrderStatus.PARTIAL.is_complete is False
        assert OrderStatus.FILLED.is_complete is True
        assert OrderStatus.CANCELLED.is_complete is True
        assert OrderStatus.REJECTED.is_complete is True
        assert OrderStatus.EXPIRED.is_complete is True
    
    @pytest.mark.unit
    def test_can_cancel_property(self):
        """Test can_cancel property."""
        assert OrderStatus.PENDING.can_cancel is True
        assert OrderStatus.OPEN.can_cancel is True
        assert OrderStatus.PARTIAL.can_cancel is True
        assert OrderStatus.FILLED.can_cancel is False
        assert OrderStatus.CANCELLED.can_cancel is False
        assert OrderStatus.REJECTED.can_cancel is False
        assert OrderStatus.EXPIRED.can_cancel is False


class TestTimeInForce:
    """Test TimeInForce enumeration."""
    
    @pytest.mark.unit
    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"
        assert TimeInForce.DAY.value == "day"


class TestOrderFill:
    """Test OrderFill model."""
    
    @pytest.mark.unit
    def test_order_fill_creation(self):
        """Test creating OrderFill instance."""
        fill = OrderFill(
            id="fill_123",
            order_id="order_456",
            price=0.65,
            size=50.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
            fee=0.65,
            fee_currency="USDC"
        )
        
        assert fill.id == "fill_123"
        assert fill.order_id == "order_456"
        assert fill.price == 0.65
        assert fill.size == 50.0
        assert fill.side == OrderSide.BUY
        assert isinstance(fill.timestamp, datetime)
        assert fill.fee == 0.65
        assert fill.fee_currency == "USDC"
    
    @pytest.mark.unit
    def test_order_fill_properties(self):
        """Test OrderFill computed properties."""
        fill = OrderFill(
            id="fill_123",
            order_id="order_456",
            price=0.65,
            size=50.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
            fee=0.65
        )
        
        # Test value property
        assert fill.value == 32.5  # 0.65 * 50.0
        
        # Test net_value property
        assert fill.net_value == 31.85  # 32.5 - 0.65
    
    @pytest.mark.unit
    def test_order_fill_defaults(self):
        """Test OrderFill default values."""
        fill = OrderFill(
            id="fill_123",
            order_id="order_456",
            price=0.65,
            size=50.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert fill.fee == 0.0
        assert fill.fee_currency == "USDC"
    
    @pytest.mark.unit
    def test_order_fill_serialization(self):
        """Test OrderFill to_dict method."""
        timestamp = datetime.now(timezone.utc)
        fill = OrderFill(
            id="fill_123",
            order_id="order_456",
            price=0.65,
            size=50.0,
            side=OrderSide.BUY,
            timestamp=timestamp,
            fee=0.65,
            fee_currency="USDC"
        )
        
        fill_dict = fill.to_dict()
        
        assert isinstance(fill_dict, dict)
        assert fill_dict["id"] == "fill_123"
        assert fill_dict["order_id"] == "order_456"
        assert fill_dict["price"] == 0.65
        assert fill_dict["size"] == 50.0
        assert fill_dict["side"] == "buy"
        assert fill_dict["timestamp"] == timestamp.isoformat()
        assert fill_dict["fee"] == 0.65
        assert fill_dict["fee_currency"] == "USDC"
    
    @pytest.mark.unit
    def test_order_fill_deserialization(self):
        """Test OrderFill from_dict method."""
        fill_data = {
            "id": "fill_123",
            "order_id": "order_456",
            "price": 0.65,
            "size": 50.0,
            "side": "buy",
            "timestamp": "2024-01-01T12:00:00+00:00",
            "fee": 0.65,
            "fee_currency": "USDC"
        }
        
        fill = OrderFill.from_dict(fill_data)
        
        assert fill.id == "fill_123"
        assert fill.order_id == "order_456"
        assert fill.price == 0.65
        assert fill.size == 50.0
        assert fill.side == OrderSide.BUY
        assert isinstance(fill.timestamp, datetime)
        assert fill.fee == 0.65
        assert fill.fee_currency == "USDC"


class TestOrder:
    """Test Order model."""
    
    @pytest.mark.unit
    def test_order_creation_basic(self):
        """Test creating basic Order instance."""
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65
        )
        
        assert order.id == "order_123"
        assert order.market_id == "0x456"
        assert order.outcome_id == "Yes"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.size == 100.0
        assert order.price == 0.65
        assert order.filled == 0.0
        assert order.remaining == 100.0  # auto-calculated
        assert order.status == OrderStatus.PENDING
        assert order.time_in_force == TimeInForce.GTC
    
    @pytest.mark.unit
    def test_order_validation_required_price(self):
        """Test validation for order types requiring price."""
        # LIMIT order without price should fail
        with pytest.raises(ValueError, match="Order type limit requires a price"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                size=100.0
            )
        
        # STOP_LIMIT order without price should fail
        with pytest.raises(ValueError, match="Order type stop_limit requires a price"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.STOP_LIMIT,
                size=100.0,
                stop_price=0.70
            )
    
    @pytest.mark.unit
    def test_order_validation_required_stop_price(self):
        """Test validation for order types requiring stop price."""
        # STOP order without stop_price should fail
        with pytest.raises(ValueError, match="Order type stop requires a stop price"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.STOP,
                size=100.0
            )
        
        # STOP_LIMIT order without stop_price should fail
        with pytest.raises(ValueError, match="Order type stop_limit requires a stop price"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.STOP_LIMIT,
                size=100.0,
                price=0.65
            )
    
    @pytest.mark.unit
    def test_order_validation_size(self):
        """Test size validation."""
        # Zero size should fail
        with pytest.raises(ValueError, match="Order size must be positive"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                size=0.0
            )
        
        # Negative size should fail
        with pytest.raises(ValueError, match="Order size must be positive"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                size=-10.0
            )
    
    @pytest.mark.unit
    def test_order_validation_filled(self):
        """Test filled amount validation."""
        # Negative filled should fail
        with pytest.raises(ValueError, match="Filled amount cannot be negative"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                size=100.0,
                filled=-10.0
            )
    
    @pytest.mark.unit
    def test_order_validation_price_bounds(self):
        """Test price validation for bounds."""
        # Price too high
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                size=100.0,
                price=1.5
            )
        
        # Price zero should fail
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                size=100.0,
                price=0.0
            )
        
        # Negative price should fail
        with pytest.raises(ValueError, match="Price must be between 0 and 1"):
            Order(
                id="order_123",
                market_id="0x456",
                outcome_id="Yes",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                size=100.0,
                price=-0.1
            )
    
    @pytest.mark.unit
    def test_order_remaining_calculation(self):
        """Test automatic remaining calculation."""
        # When remaining is not provided, it should be calculated
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            filled=30.0
        )
        
        assert order.remaining == 70.0  # 100.0 - 30.0
        
        # When remaining is explicitly provided
        order_explicit = Order(
            id="order_456",
            market_id="0x789",
            outcome_id="No",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.35,
            filled=40.0,
            remaining=60.0
        )
        
        assert order_explicit.remaining == 60.0
    
    @pytest.mark.unit
    def test_order_properties(self):
        """Test Order computed properties."""
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            filled=30.0,
            status=OrderStatus.PARTIAL
        )
        
        # Test fill_percentage
        assert order.fill_percentage == 30.0  # (30/100) * 100
        
        # Test is_active
        assert order.is_active is True  # PARTIAL status is active
        
        # Test is_complete
        assert order.is_complete is False  # PARTIAL status is not complete
        
        # Test can_cancel
        assert order.can_cancel is True  # PARTIAL status can be cancelled
        
        # Test is_buy/is_sell
        assert order.is_buy is True
        assert order.is_sell is False
        
        # Test notional_value
        assert order.notional_value == 65.0  # 100.0 * 0.65
        
        # Test remaining_value
        assert order.remaining_value == 45.5  # 70.0 * 0.65
    
    @pytest.mark.unit
    def test_order_properties_edge_cases(self):
        """Test Order properties with edge cases."""
        # Order with zero size
        order_zero_size = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            size=1.0,  # Valid non-zero size to pass validation
            filled=1.0  # Fully filled
        )
        order_zero_size.size = 0.0  # Set to zero after creation for testing
        
        assert order_zero_size.fill_percentage == 0.0  # Should handle division by zero
        
        # Market order without price
        market_order = Order(
            id="order_456",
            market_id="0x789",
            outcome_id="No",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            size=50.0
        )
        
        assert market_order.price is None
        assert market_order.notional_value == 0.0
        assert market_order.remaining_value == 0.0
    
    @pytest.mark.unit
    def test_order_fill_methods(self):
        """Test order fill-related methods."""
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            status=OrderStatus.OPEN
        )
        
        # Initially no fills
        assert len(order.fills) == 0
        assert order.total_fees == 0.0
        assert order.average_fill_price is None
        
        # Add first fill
        fill1 = OrderFill(
            id="fill_1",
            order_id="order_123",
            price=0.64,
            size=30.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
            fee=0.38  # 30 * 0.64 * 0.02
        )
        
        order.add_fill(fill1)
        
        assert len(order.fills) == 1
        assert order.filled == 30.0
        assert order.remaining == 70.0
        assert order.status == OrderStatus.PARTIAL
        assert order.total_fees == 0.38
        assert order.average_fill_price == 0.64
        
        # Add second fill
        fill2 = OrderFill(
            id="fill_2",
            order_id="order_123",
            price=0.66,
            size=70.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc),
            fee=0.92  # 70 * 0.66 * 0.02
        )
        
        order.add_fill(fill2)
        
        assert len(order.fills) == 2
        assert order.filled == 100.0
        assert order.remaining == 0.0
        assert order.status == OrderStatus.FILLED
        assert order.total_fees == 1.30  # 0.38 + 0.92
        
        # Average fill price: (30 * 0.64 + 70 * 0.66) / 100 = (19.2 + 46.2) / 100 = 0.654
        assert abs(order.average_fill_price - 0.654) < 0.001
    
    @pytest.mark.unit
    def test_order_add_fill_validation(self):
        """Test add_fill validation."""
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65
        )
        
        # Fill with wrong order_id should fail
        wrong_fill = OrderFill(
            id="fill_1",
            order_id="order_456",  # Wrong order ID
            price=0.64,
            size=30.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(ValueError, match="Fill order_id order_456 doesn't match order id order_123"):
            order.add_fill(wrong_fill)
    
    @pytest.mark.unit
    def test_order_get_fills_by_side(self):
        """Test get_fills_by_side method."""
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65
        )
        
        # Add fills with different sides (shouldn't happen in practice but test the method)
        buy_fill = OrderFill(
            id="fill_buy",
            order_id="order_123",
            price=0.64,
            size=30.0,
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )
        
        sell_fill = OrderFill(
            id="fill_sell",
            order_id="order_123",
            price=0.66,
            size=20.0,
            side=OrderSide.SELL,
            timestamp=datetime.now(timezone.utc)
        )
        
        order.fills = [buy_fill, sell_fill]
        
        buy_fills = order.get_fills_by_side(OrderSide.BUY)
        sell_fills = order.get_fills_by_side(OrderSide.SELL)
        
        assert len(buy_fills) == 1
        assert len(sell_fills) == 1
        assert buy_fills[0].id == "fill_buy"
        assert sell_fills[0].id == "fill_sell"
    
    @pytest.mark.unit
    def test_order_time_methods(self):
        """Test time-related methods."""
        created_time = datetime.now(timezone.utc)
        expires_time = created_time + timedelta(hours=24)
        
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            created_at=created_time,
            expires_at=expires_time
        )
        
        # Test get_time_since_creation
        time_since = order.get_time_since_creation()
        assert time_since is not None
        assert time_since >= 0
        
        # Test is_expired
        assert order.is_expired() is False
        
        # Test with past expiry
        past_expires = created_time - timedelta(hours=1)
        order.expires_at = past_expires
        assert order.is_expired() is True
        
        # Test without created_at
        order_no_created = Order(
            id="order_456",
            market_id="0x789",
            outcome_id="No",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            size=50.0
        )
        
        assert order_no_created.get_time_since_creation() is None
        
        # Test without expires_at
        order_no_expires = Order(
            id="order_789",
            market_id="0xabc",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=75.0,
            price=0.55
        )
        
        assert order_no_expires.is_expired() is False
    
    @pytest.mark.unit
    def test_order_serialization(self):
        """Test Order to_dict method."""
        created_time = datetime.now(timezone.utc)
        updated_time = created_time + timedelta(minutes=30)
        expires_time = created_time + timedelta(hours=24)
        
        order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            filled=30.0,
            remaining=70.0,
            status=OrderStatus.PARTIAL,
            time_in_force=TimeInForce.GTC,
            created_at=created_time,
            updated_at=updated_time,
            expires_at=expires_time,
            fee_rate=0.02,
            client_order_id="client_123"
        )
        
        order_dict = order.to_dict()
        
        assert isinstance(order_dict, dict)
        assert order_dict["id"] == "order_123"
        assert order_dict["market_id"] == "0x456"
        assert order_dict["outcome_id"] == "Yes"
        assert order_dict["side"] == "buy"
        assert order_dict["type"] == "limit"
        assert order_dict["size"] == 100.0
        assert order_dict["price"] == 0.65
        assert order_dict["filled"] == 30.0
        assert order_dict["remaining"] == 70.0
        assert order_dict["status"] == "partial"
        assert order_dict["time_in_force"] == "gtc"
        assert order_dict["created_at"] == created_time.isoformat()
        assert order_dict["updated_at"] == updated_time.isoformat()
        assert order_dict["expires_at"] == expires_time.isoformat()
        assert order_dict["fills"] == []
        assert order_dict["fee_rate"] == 0.02
        assert order_dict["client_order_id"] == "client_123"
    
    @pytest.mark.unit
    def test_order_deserialization(self):
        """Test Order from_dict method."""
        order_data = {
            "id": "order_123",
            "market_id": "0x456",
            "outcome_id": "Yes",
            "side": "buy",
            "type": "limit",
            "size": 100.0,
            "price": 0.65,
            "filled": 30.0,
            "remaining": 70.0,
            "status": "partial",
            "time_in_force": "gtc",
            "created_at": "2024-01-01T12:00:00+00:00",
            "updated_at": "2024-01-01T12:30:00+00:00",
            "expires_at": "2024-01-02T12:00:00+00:00",
            "fills": [],
            "fee_rate": 0.02,
            "client_order_id": "client_123"
        }
        
        order = Order.from_dict(order_data)
        
        assert order.id == "order_123"
        assert order.market_id == "0x456"
        assert order.outcome_id == "Yes"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.size == 100.0
        assert order.price == 0.65
        assert order.filled == 30.0
        assert order.remaining == 70.0
        assert order.status == OrderStatus.PARTIAL
        assert order.time_in_force == TimeInForce.GTC
        assert isinstance(order.created_at, datetime)
        assert isinstance(order.updated_at, datetime)
        assert isinstance(order.expires_at, datetime)
        assert len(order.fills) == 0
        assert order.fee_rate == 0.02
        assert order.client_order_id == "client_123"
    
    @pytest.mark.unit
    def test_order_serialization_roundtrip(self):
        """Test serialization roundtrip (to_dict -> from_dict)."""
        original_order = Order(
            id="order_123",
            market_id="0x456",
            outcome_id="Yes",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            size=100.0,
            price=0.65,
            filled=30.0,
            status=OrderStatus.PARTIAL
        )
        
        # Serialize and deserialize
        order_dict = original_order.to_dict()
        restored_order = Order.from_dict(order_dict)
        
        # Key fields should match
        assert restored_order.id == original_order.id
        assert restored_order.market_id == original_order.market_id
        assert restored_order.outcome_id == original_order.outcome_id
        assert restored_order.side == original_order.side
        assert restored_order.type == original_order.type
        assert restored_order.size == original_order.size
        assert restored_order.price == original_order.price
        assert restored_order.filled == original_order.filled
        assert restored_order.status == original_order.status


class TestOrderBookClass:
    """Test OrderBook class from order module."""
    
    @pytest.mark.unit
    def test_order_book_creation(self):
        """Test creating OrderBook instance."""
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[{"price": 0.60, "size": 100.0}],
            asks=[{"price": 0.61, "size": 120.0}]
        )
        
        assert order_book.market_id == "0x123"
        assert order_book.outcome_id == "Yes"
        assert len(order_book.bids) == 1
        assert len(order_book.asks) == 1
        assert order_book.bids[0]["price"] == 0.60
        assert order_book.asks[0]["price"] == 0.61
    
    @pytest.mark.unit
    def test_order_book_properties(self):
        """Test OrderBook computed properties."""
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[
                {"price": 0.60, "size": 100.0},
                {"price": 0.59, "size": 150.0}
            ],
            asks=[
                {"price": 0.61, "size": 120.0},
                {"price": 0.62, "size": 180.0}
            ]
        )
        
        # Test best bid/ask
        assert order_book.best_bid == 0.60  # max of bid prices
        assert order_book.best_ask == 0.61  # min of ask prices
        
        # Test spread
        assert order_book.spread == 0.01  # 0.61 - 0.60
        
        # Test mid price
        assert order_book.mid_price == 0.605  # (0.60 + 0.61) / 2
        
        # Test total sizes
        assert order_book.total_bid_size == 250.0  # 100 + 150
        assert order_book.total_ask_size == 300.0  # 120 + 180
    
    @pytest.mark.unit
    def test_order_book_empty(self):
        """Test OrderBook with empty sides."""
        empty_order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes"
        )
        
        assert empty_order_book.best_bid is None
        assert empty_order_book.best_ask is None
        assert empty_order_book.spread is None
        assert empty_order_book.mid_price is None
        assert empty_order_book.total_bid_size == 0.0
        assert empty_order_book.total_ask_size == 0.0
    
    @pytest.mark.unit
    def test_order_book_get_depth(self):
        """Test get_depth method."""
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[
                {"price": 0.60, "size": 100.0},
                {"price": 0.59, "size": 150.0},
                {"price": 0.58, "size": 200.0}
            ],
            asks=[
                {"price": 0.62, "size": 180.0},
                {"price": 0.61, "size": 120.0},
                {"price": 0.63, "size": 220.0}
            ]
        )
        
        depth = order_book.get_depth(levels=2)
        
        # Should return top 2 levels, sorted correctly
        assert len(depth["bids"]) == 2
        assert len(depth["asks"]) == 2
        
        # Bids should be sorted descending (highest first)
        assert depth["bids"][0]["price"] == 0.60
        assert depth["bids"][1]["price"] == 0.59
        
        # Asks should be sorted ascending (lowest first)
        assert depth["asks"][0]["price"] == 0.61
        assert depth["asks"][1]["price"] == 0.62
    
    @pytest.mark.unit
    def test_order_book_price_impact(self):
        """Test get_price_impact method."""
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[
                {"price": 0.60, "size": 100.0},
                {"price": 0.59, "size": 150.0}
            ],
            asks=[
                {"price": 0.61, "size": 120.0},
                {"price": 0.62, "size": 180.0}
            ]
        )
        
        # Test buy order price impact (uses asks)
        buy_impact = order_book.get_price_impact(OrderSide.BUY, 100.0)
        # Should buy 100 at 0.61, average price = 0.61, best ask = 0.61, impact = 0
        assert buy_impact == 0.0
        
        # Test larger buy order
        large_buy_impact = order_book.get_price_impact(OrderSide.BUY, 200.0)
        # Should buy 120 at 0.61 and 80 at 0.62
        # Average price = (120 * 0.61 + 80 * 0.62) / 200 = (73.2 + 49.6) / 200 = 0.614
        # Impact = |0.614 - 0.61| / 0.61 ≈ 0.00656
        assert large_buy_impact is not None
        assert large_buy_impact > 0
        
        # Test sell order price impact (uses bids)
        sell_impact = order_book.get_price_impact(OrderSide.SELL, 50.0)
        # Should sell 50 at 0.60, average price = 0.60, best bid = 0.60, impact = 0
        assert sell_impact == 0.0
        
        # Test insufficient liquidity
        huge_order_impact = order_book.get_price_impact(OrderSide.BUY, 500.0)
        # Not enough liquidity (only 300 available)
        assert huge_order_impact is None
    
    @pytest.mark.unit
    def test_order_book_serialization(self):
        """Test OrderBook serialization."""
        timestamp = datetime.now(timezone.utc)
        order_book = OrderBook(
            market_id="0x123",
            outcome_id="Yes",
            bids=[{"price": 0.60, "size": 100.0}],
            asks=[{"price": 0.61, "size": 120.0}],
            timestamp=timestamp
        )
        
        order_book_dict = order_book.to_dict()
        
        assert isinstance(order_book_dict, dict)
        assert order_book_dict["market_id"] == "0x123"
        assert order_book_dict["outcome_id"] == "Yes"
        assert order_book_dict["bids"] == [{"price": 0.60, "size": 100.0}]
        assert order_book_dict["asks"] == [{"price": 0.61, "size": 120.0}]
        assert order_book_dict["timestamp"] == timestamp.isoformat()
        
        # Test deserialization
        restored_order_book = OrderBook.from_dict(order_book_dict)
        
        assert restored_order_book.market_id == order_book.market_id
        assert restored_order_book.outcome_id == order_book.outcome_id
        assert restored_order_book.bids == order_book.bids
        assert restored_order_book.asks == order_book.asks
        assert isinstance(restored_order_book.timestamp, datetime)