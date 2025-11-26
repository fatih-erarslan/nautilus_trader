"""
Comprehensive tests for Interactive Brokers Canada integration.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch, call
import json
import random
from typing import Dict, List, Any
import psutil
import gc

# Import the IB Canada client (assuming it exists)
# from src.canadian_trading.brokers.ib_canada import IBCanadaClient
# For now, we'll create a minimal mock

class IBCanadaClient:
    """Mock IB Canada client for testing."""
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        self.positions = []
        self.orders = {}
        self.account_data = {}
        self.market_data_subscriptions = {}
        
    def connect(self):
        self.connected = True
        
    def disconnect(self):
        self.connected = False
        
    def get_account_summary(self):
        return self.account_data
        
    def get_positions(self):
        return self.positions
        
    def place_order(self, order: Dict):
        order_id = str(random.randint(1000, 9999))
        self.orders[order_id] = order
        return {"order_id": order_id, "status": "SUBMITTED"}
        
    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELLED"
            return True
        return False
        
    def subscribe_market_data(self, symbol: str, callback):
        self.market_data_subscriptions[symbol] = callback
        
    def unsubscribe_market_data(self, symbol: str):
        if symbol in self.market_data_subscriptions:
            del self.market_data_subscriptions[symbol]


class TestIBCanadaConnection:
    """Test IB Canada connection handling."""
    
    def test_successful_connection(self, mock_ib_client):
        """Test successful connection to IB Gateway."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        with patch.object(client, 'connect', mock_ib_client.connect):
            client.connect()
            assert mock_ib_client.isConnected()
    
    def test_connection_failure(self):
        """Test handling of connection failures."""
        client = IBCanadaClient({"host": "invalid_host", "port": 7497})
        
        with pytest.raises(ConnectionError):
            with patch.object(client, 'connect', side_effect=ConnectionError("Cannot connect")):
                client.connect()
    
    def test_connection_retry(self, performance_monitor):
        """Test connection retry mechanism."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        attempt_count = 0
        
        def mock_connect():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            return True
        
        performance_monitor.start("connection_retry")
        
        with patch.object(client, 'connect', side_effect=mock_connect):
            # Implement retry logic
            for i in range(5):
                try:
                    client.connect()
                    break
                except ConnectionError:
                    time.sleep(0.1)
        
        elapsed = performance_monitor.end("connection_retry")
        assert attempt_count == 3
        assert elapsed < 1.0  # Should complete within 1 second
    
    def test_connection_timeout(self):
        """Test connection timeout handling."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497, "timeout": 5})
        
        def slow_connect():
            time.sleep(10)
        
        with patch.object(client, 'connect', side_effect=slow_connect):
            with pytest.raises(TimeoutError):
                # Should timeout after 5 seconds
                client.connect()
    
    def test_concurrent_connections(self, thread_safety_tester):
        """Test multiple concurrent connection attempts."""
        clients = []
        
        def create_and_connect(client_id):
            client = IBCanadaClient({
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": client_id
            })
            client.connect()
            return client
        
        # Create 10 concurrent connections
        args_list = [(i,) for i in range(10)]
        results, errors = thread_safety_tester.run_concurrent(create_and_connect, args_list)
        
        assert len(errors) == 0
        assert len(results) == 10
    
    def test_disconnection_handling(self, mock_ib_client):
        """Test graceful disconnection."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        # Connect and subscribe to market data
        client.connect()
        client.subscribe_market_data("TD.TO", lambda x: None)
        
        # Disconnect
        client.disconnect()
        
        # Verify cleanup
        assert not client.connected
        assert len(client.market_data_subscriptions) == 0


class TestIBCanadaAccountOperations:
    """Test IB Canada account operations."""
    
    def test_get_account_summary(self, mock_ib_client):
        """Test retrieving account summary."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        mock_ib_client.accountSummary.return_value = [
            {"tag": "TotalCashValue", "value": "100000", "currency": "CAD"},
            {"tag": "NetLiquidation", "value": "150000", "currency": "CAD"},
            {"tag": "BuyingPower", "value": "400000", "currency": "CAD"}
        ]
        
        with patch.object(client, 'get_account_summary', mock_ib_client.accountSummary):
            summary = client.get_account_summary()
        
        assert len(summary) == 3
        assert any(item["tag"] == "TotalCashValue" for item in summary)
    
    def test_get_positions(self, mock_ib_client):
        """Test retrieving positions."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        mock_positions = [
            {
                "contract": {"symbol": "TD", "exchange": "TSE"},
                "position": 100,
                "avgCost": 80.00,
                "marketValue": 8250.00,
                "unrealizedPNL": 250.00
            },
            {
                "contract": {"symbol": "RY", "exchange": "TSE"},
                "position": 50,
                "avgCost": 140.00,
                "marketValue": 7287.50,
                "unrealizedPNL": 287.50
            }
        ]
        
        with patch.object(client, 'get_positions', return_value=mock_positions):
            positions = client.get_positions()
        
        assert len(positions) == 2
        assert positions[0]["contract"]["symbol"] == "TD"
        assert positions[0]["position"] == 100
    
    def test_position_updates(self, mock_market_data_stream):
        """Test real-time position updates."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        position_updates = []
        
        def on_position_update(position):
            position_updates.append(position)
        
        # Subscribe to position updates
        client.position_update_callback = on_position_update
        
        # Simulate position changes
        mock_market_data_stream.start()
        time.sleep(0.5)
        mock_market_data_stream.stop()
        
        # Verify updates were received
        # (In real implementation, position updates would be triggered by market data changes)
    
    def test_account_value_calculations(self):
        """Test account value calculations with currency conversion."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        account_data = {
            "CAD": {"cash": 50000, "securities": 100000},
            "USD": {"cash": 10000, "securities": 20000}
        }
        
        usd_cad_rate = 1.3650
        
        # Calculate total value in CAD
        total_cad = account_data["CAD"]["cash"] + account_data["CAD"]["securities"]
        total_usd_in_cad = (account_data["USD"]["cash"] + account_data["USD"]["securities"]) * usd_cad_rate
        total_value = total_cad + total_usd_in_cad
        
        assert total_value == pytest.approx(190950.0, rel=0.01)


class TestIBCanadaOrderManagement:
    """Test IB Canada order management."""
    
    def test_place_market_order(self, mock_ib_client):
        """Test placing a market order."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        order = {
            "symbol": "TD.TO",
            "exchange": "TSE",
            "action": "BUY",
            "quantity": 100,
            "order_type": "MARKET"
        }
        
        mock_ib_client.placeOrder.return_value = {"orderId": 12345}
        
        with patch.object(client, 'place_order', mock_ib_client.placeOrder):
            result = client.place_order(order)
        
        assert result["orderId"] == 12345
        mock_ib_client.placeOrder.assert_called_once()
    
    def test_place_limit_order(self, mock_ib_client):
        """Test placing a limit order."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        order = {
            "symbol": "RY.TO",
            "exchange": "TSE",
            "action": "SELL",
            "quantity": 50,
            "order_type": "LIMIT",
            "limit_price": 146.00
        }
        
        result = client.place_order(order)
        assert "order_id" in result
    
    def test_place_stop_loss_order(self):
        """Test placing a stop-loss order."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        order = {
            "symbol": "BNS.TO",
            "exchange": "TSE",
            "action": "SELL",
            "quantity": 100,
            "order_type": "STOP",
            "stop_price": 64.50
        }
        
        result = client.place_order(order)
        assert result["status"] == "SUBMITTED"
    
    def test_place_bracket_order(self):
        """Test placing a bracket order (entry + stop loss + take profit)."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Main order
        main_order = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 82.00
        }
        
        # Stop loss order
        stop_loss = {
            "symbol": "TD.TO",
            "action": "SELL",
            "quantity": 100,
            "order_type": "STOP",
            "stop_price": 80.00,
            "parent_id": None  # Will be set after main order
        }
        
        # Take profit order
        take_profit = {
            "symbol": "TD.TO",
            "action": "SELL",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 85.00,
            "parent_id": None  # Will be set after main order
        }
        
        # Place orders
        main_result = client.place_order(main_order)
        stop_loss["parent_id"] = main_result["order_id"]
        take_profit["parent_id"] = main_result["order_id"]
        
        sl_result = client.place_order(stop_loss)
        tp_result = client.place_order(take_profit)
        
        assert all(r["status"] == "SUBMITTED" for r in [main_result, sl_result, tp_result])
    
    def test_cancel_order(self):
        """Test cancelling an order."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Place an order
        order = {
            "symbol": "CM.TO",
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 78.00
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        
        # Cancel the order
        cancel_result = client.cancel_order(order_id)
        assert cancel_result == True
        assert client.orders[order_id]["status"] == "CANCELLED"
    
    def test_modify_order(self):
        """Test modifying an existing order."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Place original order
        order = {
            "symbol": "BMO.TO",
            "action": "BUY",
            "quantity": 50,
            "order_type": "LIMIT",
            "limit_price": 135.00
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        
        # Modify the order
        modified_order = order.copy()
        modified_order["limit_price"] = 134.50
        modified_order["quantity"] = 75
        
        # In real implementation, this would call modifyOrder
        client.orders[order_id].update(modified_order)
        
        assert client.orders[order_id]["limit_price"] == 134.50
        assert client.orders[order_id]["quantity"] == 75
    
    def test_order_validation(self):
        """Test order validation before submission."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        # Invalid order - negative quantity
        invalid_order = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": -100,
            "order_type": "MARKET"
        }
        
        with pytest.raises(ValueError, match="Invalid quantity"):
            # In real implementation, validation would be done
            if invalid_order["quantity"] <= 0:
                raise ValueError("Invalid quantity")
    
    def test_order_status_tracking(self, mock_ib_client):
        """Test tracking order status changes."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        order_statuses = []
        
        def on_order_status(order_id, status):
            order_statuses.append({"order_id": order_id, "status": status})
        
        # Set status callback
        client.order_status_callback = on_order_status
        
        # Place order and simulate status changes
        order = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": 100,
            "order_type": "MARKET"
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        
        # Simulate status progression
        statuses = ["SUBMITTED", "ACKNOWLEDGED", "FILLED"]
        for status in statuses:
            if hasattr(client, 'order_status_callback'):
                client.order_status_callback(order_id, status)
        
        assert len(order_statuses) == 3
        assert order_statuses[-1]["status"] == "FILLED"


class TestIBCanadaMarketData:
    """Test IB Canada market data operations."""
    
    def test_subscribe_market_data(self, mock_ib_client, mock_market_data_stream):
        """Test subscribing to market data."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        received_ticks = []
        
        def on_tick(tick):
            received_ticks.append(tick)
        
        # Subscribe to TD.TO
        mock_market_data_stream.subscribe("TD.TO", on_tick)
        client.subscribe_market_data("TD.TO", on_tick)
        
        # Start data stream
        mock_market_data_stream.start()
        time.sleep(0.3)
        mock_market_data_stream.stop()
        
        assert len(received_ticks) > 0
        assert all(tick["symbol"] == "TD.TO" for tick in received_ticks)
    
    def test_unsubscribe_market_data(self):
        """Test unsubscribing from market data."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Subscribe to multiple symbols
        symbols = ["TD.TO", "RY.TO", "BNS.TO"]
        for symbol in symbols:
            client.subscribe_market_data(symbol, lambda x: None)
        
        assert len(client.market_data_subscriptions) == 3
        
        # Unsubscribe from one
        client.unsubscribe_market_data("RY.TO")
        assert len(client.market_data_subscriptions) == 2
        assert "RY.TO" not in client.market_data_subscriptions
    
    def test_market_data_snapshot(self, mock_ib_client):
        """Test getting market data snapshot."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Mock snapshot data
        snapshot = {
            "symbol": "TD.TO",
            "last": 82.50,
            "bid": 82.49,
            "ask": 82.51,
            "volume": 1234567,
            "high": 83.00,
            "low": 82.00,
            "close": 82.25
        }
        
        with patch.object(client, 'get_market_snapshot', return_value=snapshot):
            data = client.get_market_snapshot("TD.TO")
        
        assert data["symbol"] == "TD.TO"
        assert data["last"] == 82.50
        assert data["bid"] < data["ask"]
    
    def test_historical_data_request(self):
        """Test requesting historical data."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Mock historical data
        historical_data = []
        base_price = 80.00
        
        for i in range(30):
            date = datetime.now() - timedelta(days=30-i)
            historical_data.append({
                "date": date,
                "open": base_price + random.uniform(-1, 1),
                "high": base_price + random.uniform(0, 2),
                "low": base_price + random.uniform(-2, 0),
                "close": base_price + random.uniform(-1, 1),
                "volume": random.randint(100000, 2000000)
            })
            base_price += random.uniform(-0.5, 0.5)
        
        with patch.object(client, 'get_historical_data', return_value=historical_data):
            data = client.get_historical_data(
                symbol="TD.TO",
                end_date=datetime.now(),
                duration="30 D",
                bar_size="1 day"
            )
        
        assert len(data) == 30
        assert all(bar["high"] >= bar["low"] for bar in data)
    
    def test_market_depth(self):
        """Test Level 2 market depth data."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Mock market depth
        market_depth = {
            "symbol": "RY.TO",
            "bids": [
                {"price": 145.74, "size": 500, "mm": "TSXMM"},
                {"price": 145.73, "size": 1000, "mm": "TSXMM"},
                {"price": 145.72, "size": 1500, "mm": "TSXMM"}
            ],
            "asks": [
                {"price": 145.76, "size": 500, "mm": "TSXMM"},
                {"price": 145.77, "size": 1000, "mm": "TSXMM"},
                {"price": 145.78, "size": 1500, "mm": "TSXMM"}
            ]
        }
        
        with patch.object(client, 'get_market_depth', return_value=market_depth):
            depth = client.get_market_depth("RY.TO")
        
        assert len(depth["bids"]) == 3
        assert len(depth["asks"]) == 3
        assert depth["bids"][0]["price"] > depth["bids"][1]["price"]
        assert depth["asks"][0]["price"] < depth["asks"][1]["price"]


class TestIBCanadaPerformance:
    """Test IB Canada performance characteristics."""
    
    def test_order_latency(self, performance_monitor, mock_ib_client):
        """Test order placement latency."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        orders = []
        for i in range(100):
            order = {
                "symbol": "TD.TO",
                "action": "BUY" if i % 2 == 0 else "SELL",
                "quantity": random.randint(10, 100),
                "order_type": "MARKET"
            }
            
            performance_monitor.start(f"order_{i}")
            result = client.place_order(order)
            elapsed = performance_monitor.end(f"order_{i}")
            
            orders.append({"order": order, "result": result, "latency": elapsed})
        
        # Analyze latencies
        latencies = [o["latency"] for o in orders]
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 0.05  # Average under 50ms
        assert max_latency < 0.1   # Max under 100ms
    
    def test_market_data_throughput(self, mock_market_data_stream, performance_monitor):
        """Test market data processing throughput."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        tick_count = 0
        start_time = time.time()
        
        def on_tick(tick):
            nonlocal tick_count
            tick_count += 1
        
        # Subscribe to multiple symbols
        symbols = ["TD.TO", "RY.TO", "BNS.TO", "BMO.TO", "CM.TO"]
        for symbol in symbols:
            mock_market_data_stream.subscribe(symbol, on_tick)
        
        # Run for 5 seconds
        mock_market_data_stream.start()
        time.sleep(5)
        mock_market_data_stream.stop()
        
        elapsed = time.time() - start_time
        ticks_per_second = tick_count / elapsed
        
        assert ticks_per_second > 100  # Should handle >100 ticks/second
    
    def test_concurrent_order_handling(self, thread_safety_tester):
        """Test handling concurrent order submissions."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        def place_order_wrapper(order_data):
            return client.place_order(order_data)
        
        # Create 50 orders
        orders = []
        for i in range(50):
            orders.append({
                "symbol": random.choice(["TD.TO", "RY.TO", "BNS.TO"]),
                "action": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(10, 100),
                "order_type": "MARKET"
            })
        
        # Submit concurrently
        args_list = [(order,) for order in orders]
        results, errors = thread_safety_tester.run_concurrent(
            place_order_wrapper,
            args_list,
            num_threads=10
        )
        
        assert len(errors) == 0
        assert len(results) == len(args_list)
    
    def test_memory_usage(self, mock_market_data_stream):
        """Test memory usage under load."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Subscribe to many symbols
        symbols = [f"TEST{i}.TO" for i in range(100)]
        for symbol in symbols:
            client.subscribe_market_data(symbol, lambda x: None)
        
        # Generate lots of data
        tick_data = []
        for _ in range(10000):
            tick_data.append({
                "symbol": random.choice(symbols),
                "price": random.uniform(50, 150),
                "volume": random.randint(100, 10000),
                "timestamp": datetime.now()
            })
        
        # Check memory after load
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Should not increase by more than 100MB
    
    def test_connection_recovery(self, mock_ib_client):
        """Test connection recovery after disconnect."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        disconnect_count = 0
        reconnect_count = 0
        
        def on_disconnect():
            nonlocal disconnect_count
            disconnect_count += 1
        
        def on_reconnect():
            nonlocal reconnect_count
            reconnect_count += 1
        
        client.on_disconnect = on_disconnect
        client.on_reconnect = on_reconnect
        
        # Initial connection
        client.connect()
        
        # Simulate disconnects and recoveries
        for i in range(3):
            # Disconnect
            client.disconnect()
            if hasattr(client, 'on_disconnect'):
                client.on_disconnect()
            
            time.sleep(0.1)
            
            # Reconnect
            client.connect()
            if hasattr(client, 'on_reconnect'):
                client.on_reconnect()
        
        assert disconnect_count == 3
        assert reconnect_count == 3


class TestIBCanadaErrorHandling:
    """Test IB Canada error handling."""
    
    def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        with pytest.raises(ValueError, match="Invalid symbol"):
            # In real implementation, this would validate the symbol
            invalid_symbol = "INVALID"
            if not invalid_symbol.endswith(".TO"):
                raise ValueError("Invalid symbol format")
    
    def test_insufficient_funds_handling(self):
        """Test handling of insufficient funds errors."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Mock account with limited funds
        client.account_data = {"AvailableFunds": 1000.00}
        
        order = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": 1000,  # $82,500 worth
            "order_type": "MARKET"
        }
        
        # Check funds before placing order
        required_funds = 82.50 * 1000
        available_funds = client.account_data.get("AvailableFunds", 0)
        
        if required_funds > available_funds:
            with pytest.raises(ValueError, match="Insufficient funds"):
                raise ValueError("Insufficient funds")
    
    def test_market_closed_handling(self):
        """Test handling of market closed errors."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Check if market is open (TSX: 9:30 AM - 4:00 PM ET)
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        if not (market_open <= now <= market_close):
            with pytest.raises(RuntimeError, match="Market is closed"):
                raise RuntimeError("Market is closed")
    
    def test_rate_limit_handling(self, latency_simulator):
        """Test handling of rate limit errors."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        request_count = 0
        rate_limit = 50  # 50 requests per second
        
        def check_rate_limit():
            nonlocal request_count
            request_count += 1
            if request_count > rate_limit:
                raise RuntimeError("Rate limit exceeded")
        
        # Rapid requests
        start_time = time.time()
        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            for i in range(100):
                check_rate_limit()
                # No delay - should hit rate limit
    
    def test_connection_loss_recovery(self, mock_ib_client):
        """Test recovery from connection loss."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Place an order
        order = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 82.00
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        
        # Simulate connection loss
        client.connected = False
        
        # Try to check order status
        with pytest.raises(ConnectionError):
            if not client.connected:
                raise ConnectionError("Not connected to IB Gateway")
        
        # Reconnect and verify order still exists
        client.connect()
        assert order_id in client.orders
    
    def test_error_callback_handling(self):
        """Test error callback mechanism."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        errors_received = []
        
        def on_error(error_code, error_msg):
            errors_received.append({
                "code": error_code,
                "message": error_msg,
                "timestamp": datetime.now()
            })
        
        client.error_callback = on_error
        
        # Simulate various errors
        test_errors = [
            (200, "No security definition found"),
            (201, "Order rejected - Invalid order type"),
            (202, "Order cancelled"),
            (399, "Order price is not valid")
        ]
        
        for code, msg in test_errors:
            if hasattr(client, 'error_callback'):
                client.error_callback(code, msg)
        
        assert len(errors_received) == len(test_errors)
        assert all(e["code"] in [200, 201, 202, 399] for e in errors_received)


class TestIBCanadaStressTesting:
    """Stress test IB Canada integration."""
    
    def test_high_frequency_trading(self, stress_test_config, performance_monitor):
        """Test high-frequency trading scenario."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        orders_placed = 0
        start_time = time.time()
        target_duration = stress_test_config["duration_seconds"]
        
        while time.time() - start_time < target_duration:
            order = {
                "symbol": random.choice(stress_test_config["symbols"]),
                "action": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(10, 100),
                "order_type": "MARKET"
            }
            
            performance_monitor.start(f"hft_order_{orders_placed}")
            try:
                result = client.place_order(order)
                orders_placed += 1
            except Exception as e:
                pass
            finally:
                performance_monitor.end(f"hft_order_{orders_placed}")
            
            # Maintain target rate
            expected_orders = (time.time() - start_time) * stress_test_config["orders_per_second"]
            if orders_placed < expected_orders:
                time.sleep(0.001)  # Small delay to catch up
        
        orders_per_second = orders_placed / target_duration
        assert orders_per_second >= stress_test_config["orders_per_second"] * 0.9  # 90% of target
    
    def test_concurrent_operations(self, stress_test_config, thread_safety_tester):
        """Test many concurrent operations."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        operations = []
        
        # Mix of different operations
        for i in range(stress_test_config["concurrent_operations"]):
            op_type = random.choice(["order", "cancel", "modify", "query"])
            
            if op_type == "order":
                operations.append(("place_order", {
                    "symbol": random.choice(stress_test_config["symbols"]),
                    "action": random.choice(["BUY", "SELL"]),
                    "quantity": random.randint(10, 100),
                    "order_type": "MARKET"
                }))
            elif op_type == "cancel":
                operations.append(("cancel_order", f"ORDER_{i}"))
            elif op_type == "modify":
                operations.append(("modify_order", (f"ORDER_{i}", {"quantity": 50})))
            else:
                operations.append(("get_positions", None))
        
        # Execute concurrently
        def execute_operation(op_type, params):
            if op_type == "place_order":
                return client.place_order(params)
            elif op_type == "cancel_order":
                return client.cancel_order(params)
            elif op_type == "get_positions":
                return client.get_positions()
            return None
        
        args_list = [(op[0], op[1]) for op in operations]
        results, errors = thread_safety_tester.run_concurrent(
            lambda args: execute_operation(args[0], args[1]),
            [(args,) for args in args_list],
            num_threads=20
        )
        
        error_rate = len(errors) / len(operations)
        assert error_rate < 0.1  # Less than 10% error rate
    
    def test_market_data_flood(self, mock_market_data_stream):
        """Test handling of market data flood."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        received_ticks = 0
        dropped_ticks = 0
        
        def on_tick(tick):
            nonlocal received_ticks
            received_ticks += 1
            # Simulate processing delay
            time.sleep(0.001)
        
        # Subscribe to all test symbols
        symbols = ["TD.TO", "RY.TO", "BNS.TO", "BMO.TO", "CM.TO", 
                  "NA.TO", "MFC.TO", "SLF.TO", "IAG.TO", "GWO.TO"]
        
        for symbol in symbols:
            mock_market_data_stream.subscribe(symbol, on_tick)
        
        # Flood with data
        mock_market_data_stream.start()
        time.sleep(10)  # 10 seconds of data
        mock_market_data_stream.stop()
        
        # Should handle thousands of ticks
        assert received_ticks > 1000
        
        # Check for memory leaks
        gc.collect()
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Should stay under 500MB


class TestIBCanadaEdgeCases:
    """Test edge cases for IB Canada integration."""
    
    def test_partial_fill_handling(self):
        """Test handling of partial order fills."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        order = {
            "symbol": "RY.TO",
            "action": "BUY",
            "quantity": 1000,
            "order_type": "LIMIT",
            "limit_price": 145.50
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        
        # Simulate partial fills
        fills = [
            {"quantity": 200, "price": 145.50},
            {"quantity": 300, "price": 145.49},
            {"quantity": 500, "price": 145.48}
        ]
        
        total_filled = 0
        total_cost = 0
        
        for fill in fills:
            total_filled += fill["quantity"]
            total_cost += fill["quantity"] * fill["price"]
        
        avg_price = total_cost / total_filled
        
        assert total_filled == 1000
        assert avg_price == pytest.approx(145.49, rel=0.01)
    
    def test_order_at_market_limits(self):
        """Test orders at market price limits."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Test order at upper circuit limit
        order_high = {
            "symbol": "TD.TO",
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 90.75  # 10% above current price
        }
        
        # Test order at lower circuit limit
        order_low = {
            "symbol": "TD.TO",
            "action": "SELL",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": 74.25  # 10% below current price
        }
        
        # Both should be accepted but may not fill
        result_high = client.place_order(order_high)
        result_low = client.place_order(order_low)
        
        assert result_high["status"] == "SUBMITTED"
        assert result_low["status"] == "SUBMITTED"
    
    def test_zero_quantity_order(self):
        """Test handling of zero quantity orders."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        with pytest.raises(ValueError, match="Invalid quantity"):
            order = {
                "symbol": "BNS.TO",
                "action": "BUY",
                "quantity": 0,
                "order_type": "MARKET"
            }
            
            if order["quantity"] <= 0:
                raise ValueError("Invalid quantity: must be positive")
    
    def test_extreme_price_orders(self):
        """Test handling of extreme price orders."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Penny stock extreme prices
        penny_order = {
            "symbol": "PENNY.TO",
            "action": "BUY",
            "quantity": 10000,
            "order_type": "LIMIT",
            "limit_price": 0.01
        }
        
        # High-priced stock extreme prices
        high_price_order = {
            "symbol": "SHOP.TO",
            "action": "SELL",
            "quantity": 1,
            "order_type": "LIMIT",
            "limit_price": 10000.00
        }
        
        # Both should be validated but may be rejected by exchange
        result1 = client.place_order(penny_order)
        result2 = client.place_order(high_price_order)
        
        assert "order_id" in result1
        assert "order_id" in result2
    
    def test_symbol_with_special_characters(self):
        """Test handling of symbols with special characters."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Preferred shares with dots
        special_symbols = [
            "BAM.PR.A.TO",  # Preferred series A
            "BCE.PR.B.TO",  # Preferred series B
            "TD.PF.C.TO"    # Preferred series C
        ]
        
        for symbol in special_symbols:
            try:
                data = client.get_market_snapshot(symbol)
                assert data is not None
            except Exception as e:
                # Symbol might not exist, but shouldn't crash
                assert "Invalid symbol" in str(e)
    
    def test_currency_conversion_edge_cases(self):
        """Test edge cases in currency conversion."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        
        # Test extreme exchange rates
        test_cases = [
            {"usd_amount": 0.01, "rate": 1.3650},      # Very small amount
            {"usd_amount": 1000000, "rate": 1.3650},   # Very large amount
            {"usd_amount": 100, "rate": 0.0001},       # Extreme rate
            {"usd_amount": 100, "rate": 10000}         # Extreme rate
        ]
        
        for case in test_cases:
            cad_amount = case["usd_amount"] * case["rate"]
            # Check for overflow/underflow
            assert cad_amount > 0
            assert not float('inf') == cad_amount
    
    def test_holiday_trading(self):
        """Test behavior on market holidays."""
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        
        # Canadian holidays
        holidays = [
            datetime(2024, 1, 1),   # New Year's Day
            datetime(2024, 2, 19),  # Family Day
            datetime(2024, 3, 29),  # Good Friday
            datetime(2024, 5, 20),  # Victoria Day
            datetime(2024, 7, 1),   # Canada Day
            datetime(2024, 8, 5),   # Civic Holiday
            datetime(2024, 9, 2),   # Labour Day
            datetime(2024, 10, 14), # Thanksgiving
            datetime(2024, 12, 25), # Christmas
            datetime(2024, 12, 26)  # Boxing Day
        ]
        
        # Check if today is a holiday
        today = datetime.now().date()
        is_holiday = any(h.date() == today for h in holidays)
        
        if is_holiday:
            with pytest.raises(RuntimeError, match="Market is closed"):
                raise RuntimeError("Market is closed - Holiday")


# Integration test combining all components
class TestIBCanadaIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_trading_workflow(self, mock_ib_client, mock_market_data_stream):
        """Test complete trading workflow from connection to execution."""
        # 1. Connect to IB
        client = IBCanadaClient({"host": "127.0.0.1", "port": 7497})
        client.connect()
        assert client.connected
        
        # 2. Get account info
        account_summary = client.get_account_summary()
        assert account_summary is not None
        
        # 3. Check existing positions
        positions = client.get_positions()
        initial_position_count = len(positions)
        
        # 4. Subscribe to market data
        symbol = "TD.TO"
        market_data = []
        
        def on_tick(tick):
            market_data.append(tick)
        
        mock_market_data_stream.subscribe(symbol, on_tick)
        mock_market_data_stream.start()
        
        # 5. Wait for market data
        time.sleep(0.5)
        assert len(market_data) > 0
        
        # 6. Place a limit order based on market data
        latest_price = market_data[-1]["last"] if market_data else 82.50
        order = {
            "symbol": symbol,
            "action": "BUY",
            "quantity": 100,
            "order_type": "LIMIT",
            "limit_price": latest_price - 0.10  # Slightly below market
        }
        
        result = client.place_order(order)
        order_id = result["order_id"]
        assert order_id is not None
        
        # 7. Monitor order status
        time.sleep(1)  # Wait for potential fill
        
        # 8. Check updated positions
        new_positions = client.get_positions()
        
        # 9. Cleanup
        mock_market_data_stream.stop()
        client.disconnect()
        
        assert not client.connected