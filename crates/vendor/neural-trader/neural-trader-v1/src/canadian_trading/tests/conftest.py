"""
Pytest fixtures and configuration for Canadian trading tests.
"""

import pytest
import asyncio
import os
import json
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import threading
from decimal import Decimal

# Test environment configuration
TEST_CONFIG = {
    "ib": {
        "host": "127.0.0.1",
        "port": 7497,  # Paper trading port
        "client_id": 999,
        "account": "DU123456"
    },
    "questrade": {
        "refresh_token": "test_refresh_token",
        "api_key": "test_api_key",
        "account_number": "12345678"
    },
    "oanda": {
        "api_key": "test_oanda_key",
        "account_id": "101-001-1234567-001",
        "environment": "practice"
    }
}

# Sample market data
SAMPLE_QUOTES = {
    "TD.TO": {
        "symbol": "TD.TO",
        "last": 82.50,
        "bid": 82.49,
        "ask": 82.51,
        "volume": 1234567,
        "high": 83.00,
        "low": 82.00,
        "timestamp": datetime.now()
    },
    "RY.TO": {
        "symbol": "RY.TO", 
        "last": 145.75,
        "bid": 145.74,
        "ask": 145.76,
        "volume": 987654,
        "high": 146.50,
        "low": 145.00,
        "timestamp": datetime.now()
    },
    "USD/CAD": {
        "symbol": "USD/CAD",
        "bid": 1.3650,
        "ask": 1.3652,
        "volume": 10000000,
        "timestamp": datetime.now()
    }
}

SAMPLE_POSITIONS = [
    {
        "symbol": "TD.TO",
        "quantity": 100,
        "average_cost": 80.00,
        "current_price": 82.50,
        "market_value": 8250.00,
        "unrealized_pnl": 250.00
    },
    {
        "symbol": "RY.TO",
        "quantity": 50,
        "average_cost": 140.00,
        "current_price": 145.75,
        "market_value": 7287.50,
        "unrealized_pnl": 287.50
    }
]

SAMPLE_ORDERS = [
    {
        "order_id": "TEST001",
        "symbol": "TD.TO",
        "side": "BUY",
        "quantity": 100,
        "order_type": "LIMIT",
        "limit_price": 82.00,
        "status": "FILLED",
        "filled_quantity": 100,
        "average_fill_price": 82.00,
        "timestamp": datetime.now()
    },
    {
        "order_id": "TEST002",
        "symbol": "RY.TO",
        "side": "SELL",
        "quantity": 25,
        "order_type": "MARKET",
        "status": "PENDING",
        "filled_quantity": 0,
        "timestamp": datetime.now()
    }
]

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_ib_client():
    """Mock Interactive Brokers client."""
    client = MagicMock()
    
    # Mock connection
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock()
    client.isConnected = MagicMock(return_value=True)
    
    # Mock account data
    client.accountSummary = MagicMock(return_value=[
        {"tag": "TotalCashValue", "value": "100000", "currency": "CAD"},
        {"tag": "NetLiquidation", "value": "150000", "currency": "CAD"}
    ])
    
    # Mock positions
    client.positions = MagicMock(return_value=[
        {
            "contract": {"symbol": "TD", "exchange": "TSE"},
            "position": 100,
            "avgCost": 80.00
        }
    ])
    
    # Mock order placement
    client.placeOrder = MagicMock(return_value={"orderId": 12345})
    
    # Mock market data
    client.reqMktData = MagicMock()
    client.cancelMktData = MagicMock()
    
    return client

@pytest.fixture
def mock_questrade_client():
    """Mock Questrade client."""
    client = MagicMock()
    
    # Mock authentication
    client.refresh_access_token = AsyncMock(return_value={
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "expires_in": 1800
    })
    
    # Mock account data
    client.get_accounts = AsyncMock(return_value={
        "accounts": [{
            "number": "12345678",
            "type": "Margin",
            "status": "Active"
        }]
    })
    
    client.get_balances = AsyncMock(return_value={
        "combinedBalances": [{
            "currency": "CAD",
            "cash": 50000,
            "marketValue": 100000,
            "totalEquity": 150000
        }]
    })
    
    # Mock positions
    client.get_positions = AsyncMock(return_value={
        "positions": [{
            "symbol": "TD.TO",
            "symbolId": 38960,
            "openQuantity": 100,
            "averageEntryPrice": 80.00,
            "currentMarketValue": 8250.00
        }]
    })
    
    # Mock quotes
    client.get_quote = AsyncMock(side_effect=lambda symbol: {
        "quotes": [{
            "symbol": symbol,
            "lastTradePrice": SAMPLE_QUOTES.get(symbol, {}).get("last", 100.00),
            "bidPrice": SAMPLE_QUOTES.get(symbol, {}).get("bid", 99.99),
            "askPrice": SAMPLE_QUOTES.get(symbol, {}).get("ask", 100.01),
            "volume": SAMPLE_QUOTES.get(symbol, {}).get("volume", 1000000)
        }]
    })
    
    # Mock order placement
    client.place_order = AsyncMock(return_value={
        "orderId": 987654321,
        "orderNumber": "TEST12345"
    })
    
    return client

@pytest.fixture
def mock_oanda_client():
    """Mock OANDA client."""
    client = MagicMock()
    
    # Mock account data
    client.get_account_summary = AsyncMock(return_value={
        "account": {
            "id": "101-001-1234567-001",
            "balance": 100000,
            "currency": "CAD",
            "marginUsed": 5000,
            "marginAvailable": 95000,
            "unrealizedPL": 1250.50,
            "NAV": 101250.50
        }
    })
    
    # Mock positions
    client.get_open_positions = AsyncMock(return_value={
        "positions": [{
            "instrument": "USD_CAD",
            "long": {
                "units": 10000,
                "averagePrice": 1.3600,
                "pl": 50.00,
                "unrealizedPL": 50.00
            }
        }]
    })
    
    # Mock pricing
    client.get_prices = AsyncMock(return_value={
        "prices": [{
            "instrument": "USD_CAD",
            "bids": [{"price": 1.3650}],
            "asks": [{"price": 1.3652}],
            "time": datetime.now().isoformat()
        }]
    })
    
    # Mock order placement
    client.create_order = AsyncMock(return_value={
        "orderCreateTransaction": {
            "id": "12345",
            "type": "MARKET_ORDER",
            "instrument": "USD_CAD",
            "units": 1000
        }
    })
    
    return client

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for caching tests."""
    redis_mock = MagicMock()
    redis_data = {}
    
    def mock_get(key):
        return json.dumps(redis_data.get(key)) if key in redis_data else None
    
    def mock_set(key, value, ex=None):
        redis_data[key] = json.loads(value) if isinstance(value, str) else value
        return True
    
    def mock_delete(key):
        if key in redis_data:
            del redis_data[key]
        return True
    
    redis_mock.get = MagicMock(side_effect=mock_get)
    redis_mock.set = MagicMock(side_effect=mock_set)
    redis_mock.delete = MagicMock(side_effect=mock_delete)
    redis_mock.exists = MagicMock(side_effect=lambda key: key in redis_data)
    
    return redis_mock

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    class PerformanceMonitor:
        def __init__(self):
            self.timings = {}
            self.start_times = {}
        
        def start(self, operation: str):
            self.start_times[operation] = time.time()
        
        def end(self, operation: str):
            if operation in self.start_times:
                elapsed = time.time() - self.start_times[operation]
                if operation not in self.timings:
                    self.timings[operation] = []
                self.timings[operation].append(elapsed)
                del self.start_times[operation]
                return elapsed
            return None
        
        def get_stats(self, operation: str):
            if operation not in self.timings or not self.timings[operation]:
                return {}
            
            times = self.timings[operation]
            return {
                "count": len(times),
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "total": sum(times)
            }
        
        def assert_performance(self, operation: str, max_time: float):
            stats = self.get_stats(operation)
            if stats:
                assert stats["avg"] < max_time, \
                    f"{operation} average time {stats['avg']:.3f}s exceeds {max_time}s"
    
    return PerformanceMonitor()

@pytest.fixture
def mock_market_data_stream():
    """Mock market data streaming."""
    class MockMarketDataStream:
        def __init__(self):
            self.subscribers = {}
            self.is_running = False
            self._thread = None
        
        def subscribe(self, symbol: str, callback):
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(callback)
        
        def unsubscribe(self, symbol: str, callback):
            if symbol in self.subscribers:
                self.subscribers[symbol].remove(callback)
        
        def start(self):
            self.is_running = True
            self._thread = threading.Thread(target=self._generate_data)
            self._thread.start()
        
        def stop(self):
            self.is_running = False
            if self._thread:
                self._thread.join()
        
        def _generate_data(self):
            while self.is_running:
                for symbol, callbacks in self.subscribers.items():
                    if symbol in SAMPLE_QUOTES:
                        # Generate slightly random price
                        base_price = SAMPLE_QUOTES[symbol].get("last", 100)
                        price = base_price * (1 + random.uniform(-0.001, 0.001))
                        
                        data = {
                            "symbol": symbol,
                            "last": price,
                            "bid": price - 0.01,
                            "ask": price + 0.01,
                            "volume": random.randint(100000, 1000000),
                            "timestamp": datetime.now()
                        }
                        
                        for callback in callbacks:
                            try:
                                callback(data)
                            except Exception:
                                pass
                
                time.sleep(0.1)  # 100ms updates
    
    return MockMarketDataStream()

@pytest.fixture
def stress_test_config():
    """Configuration for stress testing."""
    return {
        "concurrent_operations": 100,
        "duration_seconds": 10,
        "orders_per_second": 50,
        "symbols": ["TD.TO", "RY.TO", "BNS.TO", "BMO.TO", "CM.TO"],
        "max_response_time": 0.1,  # 100ms
        "memory_limit_mb": 500,
        "cpu_limit_percent": 80
    }

@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "network_timeout": lambda: time.sleep(10),
        "invalid_credentials": {"error": "Invalid API credentials"},
        "rate_limit": {"error": "Rate limit exceeded", "retry_after": 60},
        "insufficient_funds": {"error": "Insufficient funds", "available": 100, "required": 1000},
        "market_closed": {"error": "Market is closed"},
        "invalid_symbol": {"error": "Symbol not found"},
        "order_rejected": {"error": "Order rejected", "reason": "Price out of range"},
        "connection_lost": ConnectionError("Connection lost"),
        "server_error": {"error": "Internal server error", "code": 500}
    }

@pytest.fixture
def mock_websocket():
    """Mock WebSocket for streaming tests."""
    class MockWebSocket:
        def __init__(self):
            self.messages = []
            self.is_connected = False
            self.callbacks = {}
        
        async def connect(self, url: str):
            self.is_connected = True
            return True
        
        async def disconnect(self):
            self.is_connected = False
        
        async def send(self, message: str):
            self.messages.append(message)
        
        def on_message(self, callback):
            self.callbacks["message"] = callback
        
        def on_error(self, callback):
            self.callbacks["error"] = callback
        
        def on_close(self, callback):
            self.callbacks["close"] = callback
        
        async def simulate_message(self, data: dict):
            if "message" in self.callbacks:
                await self.callbacks["message"](json.dumps(data))
        
        async def simulate_error(self, error: str):
            if "error" in self.callbacks:
                await self.callbacks["error"](error)
    
    return MockWebSocket()

@pytest.fixture
def thread_safety_tester():
    """Test thread safety of operations."""
    class ThreadSafetyTester:
        def __init__(self):
            self.errors = []
            self.results = []
            self.lock = threading.Lock()
        
        def run_concurrent(self, func, args_list, num_threads=10):
            threads = []
            
            def wrapper(args):
                try:
                    result = func(*args)
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    with self.lock:
                        self.errors.append(e)
            
            for args in args_list[:num_threads]:
                thread = threading.Thread(target=wrapper, args=(args,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            return self.results, self.errors
        
        async def run_async_concurrent(self, coro_func, args_list):
            tasks = []
            for args in args_list:
                task = asyncio.create_task(coro_func(*args))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            errors = [r for r in results if isinstance(r, Exception)]
            successes = [r for r in results if not isinstance(r, Exception)]
            
            return successes, errors
    
    return ThreadSafetyTester()

# Test data generators
@pytest.fixture
def order_generator():
    """Generate random test orders."""
    def generate(count: int = 1):
        orders = []
        for i in range(count):
            order = {
                "symbol": random.choice(["TD.TO", "RY.TO", "BNS.TO"]),
                "side": random.choice(["BUY", "SELL"]),
                "quantity": random.randint(10, 1000),
                "order_type": random.choice(["MARKET", "LIMIT", "STOP"]),
                "price": round(random.uniform(50, 150), 2),
                "client_order_id": f"TEST_{i}_{int(time.time())}"
            }
            orders.append(order)
        return orders if count > 1 else orders[0]
    
    return generate

@pytest.fixture
def latency_simulator():
    """Simulate network latency."""
    class LatencySimulator:
        def __init__(self):
            self.base_latency = 0.01  # 10ms base
            self.jitter = 0.005  # 5ms jitter
        
        async def add_latency(self):
            delay = self.base_latency + random.uniform(-self.jitter, self.jitter)
            await asyncio.sleep(delay)
        
        def add_sync_latency(self):
            delay = self.base_latency + random.uniform(-self.jitter, self.jitter)
            time.sleep(delay)
        
        def simulate_slow_network(self):
            self.base_latency = 0.5  # 500ms
            self.jitter = 0.1  # 100ms
        
        def simulate_fast_network(self):
            self.base_latency = 0.001  # 1ms
            self.jitter = 0.0005  # 0.5ms
    
    return LatencySimulator()