"""
Polymarket Data Models Tests

Following TDD principles - these tests validate data models, serialization,
and validation rules for Polymarket data structures.
All tests should fail initially until the data models are implemented.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
import json
from dataclasses import asdict
import uuid


class TestMarketModel:
    """Test Market data model."""
    
    @pytest.mark.unit
    def test_market_creation(self, mock_market_data):
        """Test creating a Market instance."""
        from src.polymarket.models import Market
        
        market = Market(**mock_market_data)
        
        assert market.market_id == mock_market_data["market_id"]
        assert market.question == mock_market_data["question"]
        assert market.outcomes == mock_market_data["outcomes"]
        assert market.status == mock_market_data["status"]
        assert isinstance(market.liquidity, Decimal)
        assert isinstance(market.volume, Decimal)
        assert isinstance(market.created_at, datetime)
    
    @pytest.mark.unit
    def test_market_validation(self):
        """Test Market validation rules."""
        from src.polymarket.models import Market, MarketValidationError
        
        # Test invalid status
        with pytest.raises(MarketValidationError, match="Invalid status"):
            Market(
                market_id="0x123",
                question="Test question?",
                outcomes=["Yes", "No"],
                status="invalid_status",
                liquidity="1000",
                volume="5000",
                created_at=datetime.now()
            )
        
        # Test insufficient outcomes
        with pytest.raises(MarketValidationError, match="at least 2 outcomes"):
            Market(
                market_id="0x123",
                question="Test question?",
                outcomes=["Yes"],
                status="active",
                liquidity="1000",
                volume="5000",
                created_at=datetime.now()
            )
        
        # Test invalid market ID format
        with pytest.raises(MarketValidationError, match="Invalid market_id format"):
            Market(
                market_id="invalid_id",
                question="Test question?",
                outcomes=["Yes", "No"],
                status="active",
                liquidity="1000",
                volume="5000",
                created_at=datetime.now()
            )
        
        # Test negative liquidity
        with pytest.raises(MarketValidationError, match="Liquidity must be non-negative"):
            Market(
                market_id="0x" + "a" * 40,
                question="Test question?",
                outcomes=["Yes", "No"],
                status="active",
                liquidity="-1000",
                volume="5000",
                created_at=datetime.now()
            )
    
    @pytest.mark.unit
    def test_market_serialization(self, mock_market_data):
        """Test Market serialization to/from JSON."""
        from src.polymarket.models import Market
        
        market = Market(**mock_market_data)
        
        # Test to_dict
        market_dict = market.to_dict()
        assert isinstance(market_dict, dict)
        assert market_dict["market_id"] == mock_market_data["market_id"]
        assert isinstance(market_dict["liquidity"], str)  # Decimal as string
        assert isinstance(market_dict["created_at"], str)  # DateTime as ISO string
        
        # Test from_dict
        restored_market = Market.from_dict(market_dict)
        assert restored_market.market_id == market.market_id
        assert restored_market.liquidity == market.liquidity
        assert restored_market.created_at == market.created_at
        
        # Test JSON serialization
        json_str = market.to_json()
        assert isinstance(json_str, str)
        
        # Test JSON deserialization
        market_from_json = Market.from_json(json_str)
        assert market_from_json.market_id == market.market_id
    
    @pytest.mark.unit
    def test_market_properties(self, mock_market_data):
        """Test Market computed properties."""
        from src.polymarket.models import Market
        
        market = Market(**mock_market_data)
        
        # Test is_active property
        assert market.is_active == (market.status == "active")
        
        # Test is_binary property
        assert market.is_binary == (len(market.outcomes) == 2)
        
        # Test volume_to_liquidity_ratio
        ratio = market.volume_to_liquidity_ratio
        expected_ratio = market.volume / market.liquidity
        assert ratio == expected_ratio
        
        # Test time_remaining (if end_date exists)
        if hasattr(market, 'end_date') and market.end_date:
            remaining = market.time_remaining
            if market.end_date > datetime.now():
                assert remaining > timedelta(0)
            else:
                assert remaining <= timedelta(0)
    
    @pytest.mark.unit
    def test_market_comparison(self, mock_market_data):
        """Test Market comparison methods."""
        from src.polymarket.models import Market
        
        market1 = Market(**mock_market_data)
        market2 = Market(**mock_market_data)
        
        # Same data should be equal
        assert market1 == market2
        assert hash(market1) == hash(market2)
        
        # Different market_id should not be equal
        different_data = {**mock_market_data, "market_id": "0x" + "b" * 40}
        market3 = Market(**different_data)
        assert market1 != market3
        assert hash(market1) != hash(market3)


class TestOrderModel:
    """Test Order data model."""
    
    @pytest.mark.unit
    def test_order_creation(self, mock_order_data):
        """Test creating an Order instance."""
        from src.polymarket.models import Order
        
        order = Order(**mock_order_data)
        
        assert order.order_id == mock_order_data["order_id"]
        assert order.market_id == mock_order_data["market_id"]
        assert order.side == mock_order_data["side"]
        assert order.outcome == mock_order_data["outcome"]
        assert isinstance(order.size, Decimal)
        assert isinstance(order.price, Decimal)
        assert order.type == mock_order_data["type"]
        assert order.status == mock_order_data["status"]
    
    @pytest.mark.unit
    def test_order_validation(self):
        """Test Order validation rules."""
        from src.polymarket.models import Order, OrderValidationError
        
        base_data = {
            "order_id": "order_123",
            "market_id": "0x" + "a" * 40,
            "side": "buy",
            "outcome": "Yes",
            "size": "100",
            "price": "0.65",
            "type": "limit",
            "status": "open",
            "created_at": datetime.now().isoformat()
        }
        
        # Test invalid side
        with pytest.raises(OrderValidationError, match="Invalid side"):
            Order(**{**base_data, "side": "invalid"})
        
        # Test invalid order type
        with pytest.raises(OrderValidationError, match="Invalid order type"):
            Order(**{**base_data, "type": "invalid"})
        
        # Test invalid price range
        with pytest.raises(OrderValidationError, match="Price must be between 0 and 1"):
            Order(**{**base_data, "price": "1.5"})
        
        with pytest.raises(OrderValidationError, match="Price must be between 0 and 1"):
            Order(**{**base_data, "price": "-0.1"})
        
        # Test zero or negative size
        with pytest.raises(OrderValidationError, match="Size must be positive"):
            Order(**{**base_data, "size": "0"})
        
        with pytest.raises(OrderValidationError, match="Size must be positive"):
            Order(**{**base_data, "size": "-100"})
        
        # Test invalid status
        with pytest.raises(OrderValidationError, match="Invalid status"):
            Order(**{**base_data, "status": "invalid_status"})
    
    @pytest.mark.unit
    def test_order_properties(self, mock_order_data):
        """Test Order computed properties."""
        from src.polymarket.models import Order
        
        order_data = {
            **mock_order_data,
            "filled_size": "60",
            "remaining_size": "40",
            "average_price": "0.63"
        }
        order = Order(**order_data)
        
        # Test is_open property
        assert order.is_open == (order.status == "open")
        
        # Test is_filled property
        assert order.is_filled == (order.status == "filled")
        
        # Test fill_percentage
        expected_fill = (Decimal("60") / Decimal("100")) * 100
        assert order.fill_percentage == expected_fill
        
        # Test remaining_value
        expected_value = Decimal("40") * Decimal("0.65")  # remaining_size * price
        assert order.remaining_value == expected_value
        
        # Test total_cost
        if order.side == "buy":
            expected_cost = order.size * order.price
            assert order.total_cost == expected_cost
    
    @pytest.mark.unit
    def test_order_update(self, mock_order_data):
        """Test updating order status and fill information."""
        from src.polymarket.models import Order
        
        order = Order(**mock_order_data)
        
        # Test partial fill update
        fill_update = {
            "filled_size": "30",
            "remaining_size": "70",
            "average_price": "0.64",
            "status": "partially_filled"
        }
        
        updated_order = order.update(**fill_update)
        
        assert updated_order.filled_size == Decimal("30")
        assert updated_order.remaining_size == Decimal("70")
        assert updated_order.status == "partially_filled"
        assert updated_order.order_id == order.order_id  # Unchanged fields preserved
        
        # Original order should be unchanged (immutable)
        assert order.status == "open"
        assert order.filled_size == Decimal("0")


class TestPositionModel:
    """Test Position data model."""
    
    @pytest.mark.unit
    def test_position_creation(self, mock_position_data):
        """Test creating a Position instance."""
        from src.polymarket.models import Position
        
        position = Position(**mock_position_data)
        
        assert position.market_id == mock_position_data["market_id"]
        assert position.outcome == mock_position_data["outcome"]
        assert isinstance(position.size, Decimal)
        assert isinstance(position.average_price, Decimal)
        assert isinstance(position.current_price, Decimal)
        assert isinstance(position.unrealized_pnl, Decimal)
        assert isinstance(position.realized_pnl, Decimal)
    
    @pytest.mark.unit
    def test_position_properties(self, mock_position_data):
        """Test Position computed properties."""
        from src.polymarket.models import Position
        
        position = Position(**mock_position_data)
        
        # Test total_pnl
        expected_total = position.unrealized_pnl + position.realized_pnl
        assert position.total_pnl == expected_total
        
        # Test current_value
        expected_value = position.size * position.current_price
        assert position.current_value == expected_value
        
        # Test cost_basis
        expected_basis = position.size * position.average_price
        assert position.cost_basis == expected_basis
        
        # Test pnl_percentage
        if position.cost_basis > 0:
            expected_pnl_pct = (position.total_pnl / position.cost_basis) * 100
            assert position.pnl_percentage == expected_pnl_pct
        
        # Test is_profitable
        assert position.is_profitable == (position.total_pnl > 0)
    
    @pytest.mark.unit
    def test_position_update_price(self, mock_position_data):
        """Test updating position with new market price."""
        from src.polymarket.models import Position
        
        position = Position(**mock_position_data)
        original_unrealized = position.unrealized_pnl
        
        # Update with new price
        new_price = Decimal("0.70")
        updated_position = position.update_price(new_price)
        
        assert updated_position.current_price == new_price
        assert updated_position.unrealized_pnl != original_unrealized
        
        # Verify PnL calculation
        price_diff = new_price - position.average_price
        expected_unrealized = position.size * price_diff
        assert updated_position.unrealized_pnl == expected_unrealized
    
    @pytest.mark.unit
    def test_position_add_trade(self, mock_position_data):
        """Test adding a new trade to position."""
        from src.polymarket.models import Position, Trade
        
        position = Position(**mock_position_data)
        original_size = position.size
        original_avg_price = position.average_price
        
        # Add new trade
        new_trade = Trade(
            trade_id="new_trade_123",
            market_id=position.market_id,
            outcome=position.outcome,
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.70"),
            timestamp=datetime.now()
        )
        
        updated_position = position.add_trade(new_trade)
        
        # Check size increase
        assert updated_position.size == original_size + new_trade.size
        
        # Check average price recalculation
        total_cost = (original_size * original_avg_price) + (new_trade.size * new_trade.price)
        expected_avg_price = total_cost / updated_position.size
        assert updated_position.average_price == expected_avg_price


class TestTradeModel:
    """Test Trade data model."""
    
    @pytest.mark.unit
    def test_trade_creation(self):
        """Test creating a Trade instance."""
        from src.polymarket.models import Trade
        
        trade_data = {
            "trade_id": "trade_123",
            "market_id": "0x" + "a" * 40,
            "outcome": "Yes",
            "side": "buy",
            "size": "100",
            "price": "0.65",
            "fee": "0.02",
            "timestamp": datetime.now()
        }
        
        trade = Trade(**trade_data)
        
        assert trade.trade_id == trade_data["trade_id"]
        assert isinstance(trade.size, Decimal)
        assert isinstance(trade.price, Decimal)
        assert isinstance(trade.fee, Decimal)
        assert isinstance(trade.timestamp, datetime)
    
    @pytest.mark.unit
    def test_trade_properties(self):
        """Test Trade computed properties."""
        from src.polymarket.models import Trade
        
        trade = Trade(
            trade_id="trade_123",
            market_id="0x123",
            outcome="Yes",
            side="buy",
            size=Decimal("100"),
            price=Decimal("0.65"),
            fee=Decimal("0.02"),
            timestamp=datetime.now()
        )
        
        # Test gross_amount
        expected_gross = trade.size * trade.price
        assert trade.gross_amount == expected_gross
        
        # Test net_amount (after fees)
        if trade.side == "buy":
            expected_net = expected_gross + trade.fee
        else:
            expected_net = expected_gross - trade.fee
        assert trade.net_amount == expected_net
        
        # Test is_buy/is_sell
        assert trade.is_buy == (trade.side == "buy")
        assert trade.is_sell == (trade.side == "sell")


class TestOrderBookModel:
    """Test OrderBook data model."""
    
    @pytest.mark.unit
    def test_order_book_creation(self, mock_market_data):
        """Test creating an OrderBook instance."""
        from src.polymarket.models import OrderBook, OrderBookLevel
        
        order_book_data = mock_market_data["order_book"]
        
        # Convert to OrderBookLevel objects
        bids = [OrderBookLevel(**bid) for bid in order_book_data["bids"]]
        asks = [OrderBookLevel(**ask) for ask in order_book_data["asks"]]
        
        order_book = OrderBook(
            market_id=mock_market_data["market_id"],
            outcome="Yes",
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
        
        assert order_book.market_id == mock_market_data["market_id"]
        assert len(order_book.bids) == 3
        assert len(order_book.asks) == 3
        assert all(isinstance(bid, OrderBookLevel) for bid in order_book.bids)
    
    @pytest.mark.unit
    def test_order_book_properties(self, mock_market_data):
        """Test OrderBook computed properties."""
        from src.polymarket.models import OrderBook, OrderBookLevel
        
        order_book_data = mock_market_data["order_book"]
        bids = [OrderBookLevel(**bid) for bid in order_book_data["bids"]]
        asks = [OrderBookLevel(**ask) for ask in order_book_data["asks"]]
        
        order_book = OrderBook(
            market_id=mock_market_data["market_id"],
            outcome="Yes",
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
        
        # Test best_bid/best_ask
        assert order_book.best_bid == bids[0]  # Should be highest bid
        assert order_book.best_ask == asks[0]  # Should be lowest ask
        
        # Test spread
        expected_spread = order_book.best_ask.price - order_book.best_bid.price
        assert order_book.spread == expected_spread
        
        # Test mid_price
        expected_mid = (order_book.best_bid.price + order_book.best_ask.price) / 2
        assert order_book.mid_price == expected_mid
        
        # Test total volumes
        expected_bid_volume = sum(level.size for level in bids)
        expected_ask_volume = sum(level.size for level in asks)
        assert order_book.total_bid_volume == expected_bid_volume
        assert order_book.total_ask_volume == expected_ask_volume
    
    @pytest.mark.unit
    def test_order_book_validation(self):
        """Test OrderBook validation rules."""
        from src.polymarket.models import OrderBook, OrderBookLevel, OrderBookValidationError
        
        # Test overlapping bid/ask prices (invalid)
        bids = [OrderBookLevel(price=Decimal("0.70"), size=Decimal("100"))]
        asks = [OrderBookLevel(price=Decimal("0.65"), size=Decimal("100"))]  # Lower than bid
        
        with pytest.raises(OrderBookValidationError, match="Best ask price must be higher than best bid"):
            OrderBook(
                market_id="0x123",
                outcome="Yes",
                bids=bids,
                asks=asks,
                timestamp=datetime.now()
            )
        
        # Test unsorted order levels
        unsorted_bids = [
            OrderBookLevel(price=Decimal("0.60"), size=Decimal("100")),
            OrderBookLevel(price=Decimal("0.65"), size=Decimal("100"))  # Should be first (higher)
        ]
        
        with pytest.raises(OrderBookValidationError, match="Bids must be sorted in descending price order"):
            OrderBook(
                market_id="0x123",
                outcome="Yes",
                bids=unsorted_bids,
                asks=[],
                timestamp=datetime.now()
            )


class TestPortfolioModel:
    """Test Portfolio data model."""
    
    @pytest.mark.unit
    def test_portfolio_creation(self, mock_position_data):
        """Test creating a Portfolio instance."""
        from src.polymarket.models import Portfolio, Position
        
        positions = [Position(**mock_position_data)]
        portfolio = Portfolio(
            positions=positions,
            cash_balance=Decimal("1000.00"),
            timestamp=datetime.now()
        )
        
        assert len(portfolio.positions) == 1
        assert portfolio.cash_balance == Decimal("1000.00")
        assert isinstance(portfolio.timestamp, datetime)
    
    @pytest.mark.unit
    def test_portfolio_properties(self, mock_position_data):
        """Test Portfolio computed properties."""
        from src.polymarket.models import Portfolio, Position
        
        # Create multiple positions
        position1 = Position(**mock_position_data)
        position2_data = {**mock_position_data, "market_id": "0x" + "b" * 40, "size": "200"}
        position2 = Position(**position2_data)
        
        portfolio = Portfolio(
            positions=[position1, position2],
            cash_balance=Decimal("1000.00"),
            timestamp=datetime.now()
        )
        
        # Test total_value
        expected_total = (portfolio.cash_balance + 
                         sum(pos.current_value for pos in portfolio.positions))
        assert portfolio.total_value == expected_total
        
        # Test total_pnl
        expected_pnl = sum(pos.total_pnl for pos in portfolio.positions)
        assert portfolio.total_pnl == expected_pnl
        
        # Test positions_value
        expected_positions_value = sum(pos.current_value for pos in portfolio.positions)
        assert portfolio.positions_value == expected_positions_value
        
        # Test cash_percentage
        expected_cash_pct = (portfolio.cash_balance / portfolio.total_value) * 100
        assert portfolio.cash_percentage == expected_cash_pct
    
    @pytest.mark.unit
    def test_portfolio_grouping(self, mock_position_data):
        """Test portfolio position grouping methods."""
        from src.polymarket.models import Portfolio, Position
        
        # Create positions with different markets and outcomes
        positions = [
            Position(**{**mock_position_data, "market_id": "0xAAA", "outcome": "Yes"}),
            Position(**{**mock_position_data, "market_id": "0xAAA", "outcome": "No"}),
            Position(**{**mock_position_data, "market_id": "0xBBB", "outcome": "Yes"}),
        ]
        
        portfolio = Portfolio(
            positions=positions,
            cash_balance=Decimal("1000.00"),
            timestamp=datetime.now()
        )
        
        # Test positions_by_market
        by_market = portfolio.positions_by_market
        assert len(by_market) == 2
        assert len(by_market["0xAAA"]) == 2
        assert len(by_market["0xBBB"]) == 1
        
        # Test positions_by_outcome
        by_outcome = portfolio.positions_by_outcome
        assert len(by_outcome["Yes"]) == 2
        assert len(by_outcome["No"]) == 1


class TestDataModelSerialization:
    """Test serialization across all models."""
    
    @pytest.mark.unit
    def test_model_json_roundtrip(self, mock_market_data, mock_order_data, mock_position_data):
        """Test JSON serialization roundtrip for all models."""
        from src.polymarket.models import Market, Order, Position
        
        # Test Market
        market = Market(**mock_market_data)
        market_json = market.to_json()
        restored_market = Market.from_json(market_json)
        assert market == restored_market
        
        # Test Order
        order = Order(**mock_order_data)
        order_json = order.to_json()
        restored_order = Order.from_json(order_json)
        assert order == restored_order
        
        # Test Position
        position = Position(**mock_position_data)
        position_json = position.to_json()
        restored_position = Position.from_json(position_json)
        assert position == restored_position
    
    @pytest.mark.unit
    def test_batch_serialization(self, test_data_generator):
        """Test batch serialization of multiple objects."""
        from src.polymarket.models import Market, serialize_batch, deserialize_batch
        
        # Create multiple markets
        markets = [
            Market(**test_data_generator.generate_market())
            for _ in range(5)
        ]
        
        # Serialize batch
        serialized = serialize_batch(markets)
        assert isinstance(serialized, str)
        
        # Deserialize batch
        restored_markets = deserialize_batch(serialized, Market)
        assert len(restored_markets) == 5
        assert all(isinstance(m, Market) for m in restored_markets)
        assert restored_markets == markets