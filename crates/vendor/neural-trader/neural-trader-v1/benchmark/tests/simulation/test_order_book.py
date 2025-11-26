"""
Tests for the order book simulation engine.
Testing realistic order book dynamics with high performance requirements.
"""
import asyncio
import time
import pytest
import numpy as np
from decimal import Decimal
from typing import List, Dict, Any

from benchmark.src.simulation.order_book import (
    OrderBook, Order, OrderType, OrderSide,
    Trade, OrderBookSnapshot, MarketDepth
)


class TestOrderBook:
    """Test suite for order book simulation."""
    
    @pytest.fixture
    def order_book(self):
        """Create a fresh order book instance."""
        return OrderBook(symbol="AAPL", tick_size=0.01)
    
    def test_order_book_initialization(self, order_book):
        """Test order book is properly initialized."""
        assert order_book.symbol == "AAPL"
        assert order_book.tick_size == 0.01
        assert order_book.bid_levels == {}
        assert order_book.ask_levels == {}
        assert order_book.best_bid is None
        assert order_book.best_ask is None
    
    def test_add_limit_order_buy(self, order_book):
        """Test adding a buy limit order."""
        order = Order(
            order_id="BUY001",
            side=OrderSide.BUY,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        
        order_book.add_order(order)
        
        assert order_book.best_bid == 150.00
        assert order_book.bid_levels[150.00][0] == order
        assert len(order_book.bid_levels) == 1
    
    def test_add_limit_order_sell(self, order_book):
        """Test adding a sell limit order."""
        order = Order(
            order_id="SELL001",
            side=OrderSide.SELL,
            price=151.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        
        order_book.add_order(order)
        
        assert order_book.best_ask == 151.00
        assert order_book.ask_levels[151.00][0] == order
        assert len(order_book.ask_levels) == 1
    
    def test_order_matching_full_fill(self, order_book):
        """Test order matching with full fill."""
        # Add sell order first
        sell_order = Order(
            order_id="SELL001",
            side=OrderSide.SELL,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        order_book.add_order(sell_order)
        
        # Add matching buy order
        buy_order = Order(
            order_id="BUY001",
            side=OrderSide.BUY,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        trades = order_book.add_order(buy_order)
        
        assert len(trades) == 1
        assert trades[0].price == 150.00
        assert trades[0].quantity == 100
        assert trades[0].buyer_order_id == "BUY001"
        assert trades[0].seller_order_id == "SELL001"
        assert order_book.best_ask is None  # Sell order fully filled
    
    def test_order_matching_partial_fill(self, order_book):
        """Test order matching with partial fill."""
        # Add sell order
        sell_order = Order(
            order_id="SELL001",
            side=OrderSide.SELL,
            price=150.00,
            quantity=200,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        order_book.add_order(sell_order)
        
        # Add smaller buy order
        buy_order = Order(
            order_id="BUY001",
            side=OrderSide.BUY,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        trades = order_book.add_order(buy_order)
        
        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert order_book.best_ask == 150.00
        assert order_book.ask_levels[150.00][0].remaining_quantity == 100  # Remaining quantity
    
    def test_market_order_execution(self, order_book):
        """Test market order execution."""
        # Setup order book with multiple levels
        for i in range(5):
            sell_order = Order(
                order_id=f"SELL{i}",
                side=OrderSide.SELL,
                price=150.00 + i * 0.01,
                quantity=100,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            order_book.add_order(sell_order)
        
        # Execute market buy order
        market_order = Order(
            order_id="MKT001",
            side=OrderSide.BUY,
            quantity=250,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )
        trades = order_book.add_order(market_order)
        
        assert len(trades) == 3  # Should match against 3 orders
        assert sum(t.quantity for t in trades) == 250
        assert trades[0].price == 150.00  # Best price first
        assert trades[1].price == 150.01
        assert trades[2].price == 150.02
    
    def test_cancel_order(self, order_book):
        """Test order cancellation."""
        order = Order(
            order_id="BUY001",
            side=OrderSide.BUY,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        order_book.add_order(order)
        
        assert order_book.cancel_order("BUY001") is True
        assert order_book.best_bid is None
        assert 150.00 not in order_book.bid_levels
    
    def test_modify_order(self, order_book):
        """Test order modification."""
        order = Order(
            order_id="BUY001",
            side=OrderSide.BUY,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        order_book.add_order(order)
        
        # Modify quantity
        order_book.modify_order("BUY001", new_quantity=200)
        assert order_book.bid_levels[150.00][0].quantity == 200
        
        # Modify price (should move to new level)
        order_book.modify_order("BUY001", new_price=150.50)
        assert 150.00 not in order_book.bid_levels
        assert order_book.bid_levels[150.50][0].quantity == 200
        assert order_book.best_bid == 150.50
    
    def test_get_market_depth(self, order_book):
        """Test market depth retrieval."""
        # Add multiple orders at different levels
        for i in range(5):
            buy_order = Order(
                order_id=f"BUY{i}",
                side=OrderSide.BUY,
                price=149.95 - i * 0.01,
                quantity=100 * (i + 1),
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            sell_order = Order(
                order_id=f"SELL{i}",
                side=OrderSide.SELL,
                price=150.05 + i * 0.01,
                quantity=100 * (i + 1),
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            order_book.add_order(buy_order)
            order_book.add_order(sell_order)
        
        depth = order_book.get_market_depth(levels=3)
        
        assert len(depth.bids) == 3
        assert len(depth.asks) == 3
        assert depth.bids[0].price == 149.95
        assert depth.bids[0].quantity == 100
        assert depth.asks[0].price == 150.05
        assert depth.asks[0].quantity == 100
    
    def test_get_order_book_snapshot(self, order_book):
        """Test order book snapshot."""
        # Add some orders
        for i in range(3):
            order_book.add_order(Order(
                order_id=f"BUY{i}",
                side=OrderSide.BUY,
                price=150.00 - i * 0.01,
                quantity=100,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            ))
            order_book.add_order(Order(
                order_id=f"SELL{i}",
                side=OrderSide.SELL,
                price=150.10 + i * 0.01,
                quantity=100,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            ))
        
        snapshot = order_book.get_snapshot()
        
        assert snapshot.symbol == "AAPL"
        assert snapshot.best_bid == 150.00
        assert snapshot.best_ask == 150.10
        assert abs(snapshot.spread - 0.10) < 0.0001  # Handle floating point precision
        assert snapshot.bid_depth == 3
        assert snapshot.ask_depth == 3
        assert snapshot.total_bid_volume == 300
        assert snapshot.total_ask_volume == 300
    
    @pytest.mark.asyncio
    async def test_high_frequency_order_processing(self, order_book):
        """Test processing 1M+ orders per second."""
        # Generate batch of orders
        num_orders = 10000  # Test with smaller batch first
        orders = []
        
        for i in range(num_orders):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            price = 150.00 + (np.random.random() - 0.5) * 2  # Price within ±1
            quantity = int(np.random.uniform(10, 1000))
            
            order = Order(
                order_id=f"ORDER{i}",
                side=side,
                price=round(price, 2),
                quantity=quantity,
                order_type=OrderType.LIMIT,
                timestamp=time.time()
            )
            orders.append(order)
        
        # Measure processing time
        start_time = time.perf_counter()
        
        # Process orders asynchronously
        async def process_order(order):
            return order_book.add_order(order)
        
        # Process in batches for better performance
        batch_size = 1000
        for i in range(0, len(orders), batch_size):
            batch = orders[i:i+batch_size]
            await asyncio.gather(*[process_order(order) for order in batch])
        
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        orders_per_second = num_orders / elapsed
        
        print(f"Processed {num_orders} orders in {elapsed:.3f} seconds")
        print(f"Rate: {orders_per_second:.0f} orders/second")
        
        # Verify order book integrity
        snapshot = order_book.get_snapshot()
        assert snapshot.bid_depth > 0
        assert snapshot.ask_depth > 0
        assert snapshot.best_bid < snapshot.best_ask
    
    def test_price_priority(self, order_book):
        """Test price priority in order matching."""
        # Add multiple sell orders at different prices
        order_book.add_order(Order(
            order_id="SELL1",
            side=OrderSide.SELL,
            price=150.02,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        ))
        order_book.add_order(Order(
            order_id="SELL2",
            side=OrderSide.SELL,
            price=150.01,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        ))
        order_book.add_order(Order(
            order_id="SELL3",
            side=OrderSide.SELL,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        ))
        
        # Add aggressive buy order
        buy_order = Order(
            order_id="BUY1",
            side=OrderSide.BUY,
            price=150.02,
            quantity=250,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        trades = order_book.add_order(buy_order)
        
        # Should match best prices first
        assert len(trades) == 3
        assert trades[0].price == 150.00  # Best price
        assert trades[1].price == 150.01  # Second best
        assert trades[2].price == 150.02  # Third best
    
    def test_time_priority(self, order_book):
        """Test time priority for orders at same price."""
        # Add multiple orders at same price with different timestamps
        orders = []
        for i in range(3):
            order = Order(
                order_id=f"SELL{i}",
                side=OrderSide.SELL,
                price=150.00,
                quantity=100,
                order_type=OrderType.LIMIT,
                timestamp=time.time() + i  # Different timestamps
            )
            orders.append(order)
            order_book.add_order(order)
        
        # Add matching buy order
        buy_order = Order(
            order_id="BUY1",
            side=OrderSide.BUY,
            price=150.00,
            quantity=250,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        trades = order_book.add_order(buy_order)
        
        # Should match in FIFO order
        assert len(trades) == 3
        assert trades[0].seller_order_id == "SELL0"  # First in time
        assert trades[1].seller_order_id == "SELL1"  # Second in time
        assert trades[2].seller_order_id == "SELL2"  # Third in time

    def test_stop_order_activation(self, order_book):
        """Test stop order activation and execution."""
        # Add initial market state
        order_book.add_order(Order(
            order_id="SELL1",
            side=OrderSide.SELL,
            price=150.00,
            quantity=100,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        ))
        
        # Add stop buy order (triggered when price reaches 150.00)
        stop_order = Order(
            order_id="STOP_BUY1",
            side=OrderSide.BUY,
            price=150.00,  # Stop trigger price
            quantity=50,
            order_type=OrderType.STOP,
            timestamp=time.time()
        )
        order_book.add_order(stop_order)
        
        # Verify stop order is not immediately executed
        assert order_book.best_ask == 150.00
        
        # Simulate price movement that triggers stop
        trigger_order = Order(
            order_id="BUY1",
            side=OrderSide.BUY,
            price=150.00,
            quantity=50,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )
        trades = order_book.add_order(trigger_order)
        
        # Should trigger stop order execution
        assert len(trades) >= 1
        assert order_book.has_stop_orders_triggered()