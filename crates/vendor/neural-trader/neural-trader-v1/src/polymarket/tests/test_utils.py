"""
Test Utilities and Helpers

This module provides comprehensive test utilities including mock data generators,
test fixtures, performance monitoring, and integration test helpers.
"""

import asyncio
import json
import random
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock
import numpy as np
import pandas as pd

from polymarket.models import (
    Market, MarketStatus, Order, OrderSide, OrderStatus, OrderType,
    Position, Trade, OrderBook
)
from polymarket.api import PolymarketClient, CLOBClient, GammaClient
from polymarket.strategies.base import TradingSignal, SignalStrength, SignalDirection


class MockDataGenerator:
    """Generate realistic mock data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_market(self, **kwargs) -> Market:
        """Generate a realistic market."""
        defaults = {
            "id": f"market_{random.randint(1000, 9999)}",
            "question": f"Will {random.choice(['BTC', 'ETH', 'SOL'])} reach ${random.randint(10000, 100000)}?",
            "slug": f"crypto-price-{random.randint(1000, 9999)}",
            "category": random.choice(["Crypto", "Politics", "Sports", "Entertainment"]),
            "outcomes": ["Yes", "No"],
            "outcome_prices": self._generate_outcome_prices(),
            "volume_24h": Decimal(str(random.uniform(1000, 1000000))),
            "liquidity": Decimal(str(random.uniform(5000, 500000))),
            "num_traders": random.randint(10, 5000),
            "status": MarketStatus.ACTIVE,
            "created_at": datetime.now() - timedelta(days=random.randint(1, 90)),
            "close_time": datetime.now() + timedelta(days=random.randint(1, 180))
        }
        defaults.update(kwargs)
        return Market(**defaults)
    
    def generate_markets(self, count: int, **kwargs) -> List[Market]:
        """Generate multiple markets."""
        return [self.generate_market(**kwargs) for _ in range(count)]
    
    def generate_order(self, market_id: str, **kwargs) -> Order:
        """Generate a realistic order."""
        defaults = {
            "id": f"order_{random.randint(10000, 99999)}",
            "market_id": market_id,
            "outcome": random.choice(["Yes", "No"]),
            "side": random.choice([OrderSide.BUY, OrderSide.SELL]),
            "order_type": random.choice([OrderType.LIMIT, OrderType.MARKET]),
            "size": Decimal(str(random.uniform(1, 1000))),
            "price": Decimal(str(random.uniform(0.1, 0.9))),
            "status": random.choice([OrderStatus.OPEN, OrderStatus.FILLED, OrderStatus.CANCELLED]),
            "created_at": datetime.now() - timedelta(minutes=random.randint(1, 60)),
            "filled_size": Decimal("0")
        }
        defaults.update(kwargs)
        return Order(**defaults)
    
    def generate_position(self, market_id: str, **kwargs) -> Position:
        """Generate a realistic position."""
        avg_price = Decimal(str(random.uniform(0.2, 0.8)))
        current_price = avg_price * Decimal(str(random.uniform(0.8, 1.2)))
        size = Decimal(str(random.uniform(10, 1000)))
        
        defaults = {
            "market_id": market_id,
            "outcome": random.choice(["Yes", "No"]),
            "size": size,
            "average_price": avg_price,
            "current_price": current_price,
            "pnl": (current_price - avg_price) * size,
            "pnl_percentage": float((current_price - avg_price) / avg_price)
        }
        defaults.update(kwargs)
        return Position(**defaults)
    
    def generate_orderbook(self, market_id: str, spread: float = 0.02) -> OrderBook:
        """Generate realistic orderbook data."""
        mid_price = random.uniform(0.3, 0.7)
        
        bids = []
        asks = []
        
        # Generate bid levels
        for i in range(10):
            price = mid_price - (i + 1) * spread / 20
            size = random.uniform(100, 5000) * (1 - i * 0.08)  # Decreasing size
            bids.append({
                "price": str(round(price, 4)),
                "size": str(round(size, 2))
            })
        
        # Generate ask levels
        for i in range(10):
            price = mid_price + (i + 1) * spread / 20
            size = random.uniform(100, 5000) * (1 - i * 0.08)  # Decreasing size
            asks.append({
                "price": str(round(price, 4)),
                "size": str(round(size, 2))
            })
        
        return OrderBook(
            market_id=market_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
    
    def generate_price_history(self, days: int = 30, initial_price: float = 0.5) -> pd.DataFrame:
        """Generate realistic price history."""
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=days * 24,  # Hourly data
            freq='H'
        )
        
        # Generate correlated price movements
        returns = np.random.normal(0, 0.02, len(timestamps))
        returns[0] = 0
        
        # Add trend
        trend = np.random.choice([-0.0001, 0, 0.0001])
        returns += trend
        
        # Add volatility clustering
        volatility = np.ones(len(returns))
        for i in range(1, len(returns)):
            volatility[i] = 0.94 * volatility[i-1] + 0.06 * abs(returns[i-1])
        returns *= volatility
        
        # Calculate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        prices = np.clip(prices, 0.01, 0.99)  # Keep in [0.01, 0.99]
        
        # Generate volume (correlated with price changes)
        volumes = 10000 * (1 + 10 * np.abs(returns)) * np.random.uniform(0.5, 1.5, len(returns))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'returns': returns
        })
    
    def generate_trading_signal(self, market_id: str, **kwargs) -> TradingSignal:
        """Generate a realistic trading signal."""
        defaults = {
            "market_id": market_id,
            "direction": random.choice([SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]),
            "strength": random.choice(list(SignalStrength)),
            "confidence": random.uniform(0.5, 0.95),
            "expected_return": random.uniform(-0.1, 0.3),
            "risk_score": random.uniform(0.1, 0.9),
            "metadata": {
                "momentum_score": random.uniform(-1, 1),
                "sentiment_score": random.uniform(-1, 1),
                "volume_ratio": random.uniform(0.5, 3.0)
            }
        }
        defaults.update(kwargs)
        return TradingSignal(**defaults)
    
    def _generate_outcome_prices(self) -> List[Decimal]:
        """Generate realistic outcome prices that sum to ~1."""
        if random.random() < 0.8:  # Binary market
            yes_price = random.uniform(0.1, 0.9)
            return [Decimal(str(yes_price)), Decimal(str(1 - yes_price))]
        else:  # Multi-outcome market
            n_outcomes = random.randint(3, 5)
            prices = [random.random() for _ in range(n_outcomes)]
            total = sum(prices)
            return [Decimal(str(p / total)) for p in prices]


class TestFixtures:
    """Common test fixtures for integration tests."""
    
    @staticmethod
    @asynccontextmanager
    async def mock_api_client() -> AsyncGenerator[PolymarketClient, None]:
        """Create a mock API client with realistic behavior."""
        client = AsyncMock(spec=PolymarketClient)
        generator = MockDataGenerator()
        
        # Configure mock responses
        client.get_markets.return_value = generator.generate_markets(20)
        client.get_market.side_effect = lambda id: generator.generate_market(id=id)
        client.get_orderbook.side_effect = lambda id: generator.generate_orderbook(id)
        client.get_positions.return_value = []
        client.get_orders.return_value = []
        
        # Order placement
        async def place_order(**kwargs):
            return generator.generate_order(
                market_id=kwargs.get("market_id"),
                side=kwargs.get("side"),
                size=kwargs.get("size"),
                price=kwargs.get("price"),
                status=OrderStatus.OPEN
            )
        client.place_order.side_effect = place_order
        
        yield client
        
        # Cleanup
        await client.close()
    
    @staticmethod
    @contextmanager
    def capture_performance_metrics() -> Generator[Dict[str, Any], None, None]:
        """Context manager to capture performance metrics."""
        metrics = {
            "start_time": time.time(),
            "operations": [],
            "memory_snapshots": []
        }
        
        yield metrics
        
        metrics["end_time"] = time.time()
        metrics["total_duration"] = metrics["end_time"] - metrics["start_time"]
        
        # Calculate statistics
        if metrics["operations"]:
            durations = [op["duration"] for op in metrics["operations"]]
            metrics["stats"] = {
                "mean_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "p95_duration": np.percentile(durations, 95),
                "p99_duration": np.percentile(durations, 99),
                "total_operations": len(metrics["operations"])
            }


class MockWebSocketServer:
    """Mock WebSocket server for testing streaming functionality."""
    
    def __init__(self):
        self.connections = []
        self.message_queue = asyncio.Queue()
        self.running = False
    
    async def start(self, port: int = 8765):
        """Start mock WebSocket server."""
        self.running = True
        
        async def handler(websocket, path):
            self.connections.append(websocket)
            try:
                async for message in websocket:
                    # Echo messages back
                    await websocket.send(message)
                    
                    # Send queued messages
                    while not self.message_queue.empty():
                        queued_msg = await self.message_queue.get()
                        await websocket.send(json.dumps(queued_msg))
            finally:
                self.connections.remove(websocket)
        
        # Note: In real tests, use websockets.serve
        # This is a simplified version for illustration
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        msg_str = json.dumps(message)
        for conn in self.connections:
            await conn.send(msg_str)
    
    async def stop(self):
        """Stop the server."""
        self.running = False
        for conn in self.connections:
            await conn.close()


class TestScenarioRunner:
    """Run complex test scenarios."""
    
    def __init__(self, api_client: PolymarketClient):
        self.client = api_client
        self.generator = MockDataGenerator()
        self.results = []
    
    async def run_trading_scenario(self, duration_seconds: int = 60):
        """Run a complete trading scenario."""
        scenario_results = {
            "start_time": datetime.now(),
            "trades": [],
            "signals": [],
            "errors": [],
            "metrics": {}
        }
        
        start = time.time()
        
        while time.time() - start < duration_seconds:
            try:
                # Get markets
                markets = await self.client.get_markets(limit=10)
                
                for market in markets[:3]:  # Analyze top 3
                    # Generate signal
                    signal = self.generator.generate_trading_signal(market.id)
                    scenario_results["signals"].append(signal)
                    
                    # Execute trade if strong signal
                    if signal.strength >= SignalStrength.STRONG and signal.confidence > 0.8:
                        order = await self.client.place_order(
                            market_id=market.id,
                            outcome="Yes" if signal.direction == SignalDirection.BUY else "No",
                            side=OrderSide.BUY if signal.direction == SignalDirection.BUY else OrderSide.SELL,
                            size=Decimal("100"),
                            price=market.outcome_prices[0]
                        )
                        scenario_results["trades"].append({
                            "signal": signal,
                            "order": order,
                            "timestamp": datetime.now()
                        })
                
                await asyncio.sleep(5)  # Wait before next iteration
                
            except Exception as e:
                scenario_results["errors"].append({
                    "error": str(e),
                    "timestamp": datetime.now()
                })
        
        # Calculate metrics
        scenario_results["end_time"] = datetime.now()
        scenario_results["duration"] = scenario_results["end_time"] - scenario_results["start_time"]
        scenario_results["metrics"] = {
            "total_signals": len(scenario_results["signals"]),
            "total_trades": len(scenario_results["trades"]),
            "error_rate": len(scenario_results["errors"]) / max(len(scenario_results["signals"]), 1),
            "signals_per_minute": len(scenario_results["signals"]) / (duration_seconds / 60)
        }
        
        self.results.append(scenario_results)
        return scenario_results
    
    async def run_stress_test(self, concurrent_operations: int = 100):
        """Run stress test with concurrent operations."""
        stress_results = {
            "start_time": datetime.now(),
            "successful_operations": 0,
            "failed_operations": 0,
            "response_times": []
        }
        
        async def single_operation():
            start = time.time()
            try:
                markets = await self.client.get_markets(limit=5)
                if markets:
                    orderbook = await self.client.get_orderbook(markets[0].id)
                stress_results["successful_operations"] += 1
            except Exception:
                stress_results["failed_operations"] += 1
            finally:
                stress_results["response_times"].append(time.time() - start)
        
        # Run concurrent operations
        tasks = [single_operation() for _ in range(concurrent_operations)]
        await asyncio.gather(*tasks)
        
        # Calculate statistics
        stress_results["end_time"] = datetime.now()
        stress_results["duration"] = stress_results["end_time"] - stress_results["start_time"]
        stress_results["success_rate"] = stress_results["successful_operations"] / concurrent_operations
        stress_results["avg_response_time"] = np.mean(stress_results["response_times"])
        stress_results["p95_response_time"] = np.percentile(stress_results["response_times"], 95)
        
        return stress_results


def assert_market_valid(market: Market):
    """Assert that a market object is valid."""
    assert market.id
    assert market.question
    assert len(market.outcomes) >= 2
    assert len(market.outcome_prices) == len(market.outcomes)
    assert all(0 <= p <= 1 for p in market.outcome_prices)
    assert abs(sum(market.outcome_prices) - 1.0) < 0.1  # Prices should sum to ~1
    assert market.volume_24h >= 0
    assert market.liquidity >= 0


def assert_order_valid(order: Order):
    """Assert that an order object is valid."""
    assert order.id
    assert order.market_id
    assert order.outcome in ["Yes", "No"] or order.outcome.isdigit()
    assert order.side in [OrderSide.BUY, OrderSide.SELL]
    assert order.size > 0
    assert 0 < order.price < 1 if order.order_type == OrderType.LIMIT else True
    assert order.status in OrderStatus.__members__.values()


def assert_signal_valid(signal: TradingSignal):
    """Assert that a trading signal is valid."""
    assert signal.market_id
    assert signal.direction in SignalDirection.__members__.values()
    assert signal.strength in SignalStrength.__members__.values()
    assert 0 <= signal.confidence <= 1
    assert -1 <= signal.expected_return <= 10  # Reasonable return range
    assert 0 <= signal.risk_score <= 1


async def wait_for_condition(
    condition_func: callable,
    timeout: float = 10.0,
    interval: float = 0.1
) -> bool:
    """Wait for a condition to become true."""
    start = time.time()
    
    while time.time() - start < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


def generate_test_report(test_results: Dict[str, Any]) -> str:
    """Generate a comprehensive test report."""
    report = ["# Polymarket Integration Test Report", ""]
    report.append(f"Generated at: {datetime.now().isoformat()}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    total_tests = sum(len(v) for v in test_results.values() if isinstance(v, list))
    passed_tests = sum(len([t for t in v if t.get("status") == "passed"]) 
                      for v in test_results.values() if isinstance(v, list))
    
    report.append(f"- Total Tests: {total_tests}")
    report.append(f"- Passed: {passed_tests}")
    report.append(f"- Failed: {total_tests - passed_tests}")
    report.append(f"- Success Rate: {(passed_tests/total_tests*100):.1f}%")
    report.append("")
    
    # Detailed Results
    report.append("## Detailed Results")
    
    for category, results in test_results.items():
        report.append(f"### {category}")
        
        if isinstance(results, list):
            for test in results:
                status_icon = "✅" if test.get("status") == "passed" else "❌"
                report.append(f"- {status_icon} {test.get('name', 'Unknown')}: {test.get('duration', 0):.2f}s")
                if test.get("error"):
                    report.append(f"  - Error: {test['error']}")
        elif isinstance(results, dict):
            for key, value in results.items():
                report.append(f"- {key}: {value}")
        
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    generator = MockDataGenerator(seed=42)
    
    # Generate sample data
    market = generator.generate_market()
    print(f"Generated market: {market.question}")
    
    orderbook = generator.generate_orderbook(market.id)
    print(f"Orderbook spread: {float(orderbook.asks[0]['price']) - float(orderbook.bids[0]['price']):.4f}")
    
    signal = generator.generate_trading_signal(market.id)
    print(f"Signal: {signal.direction.value} with {signal.confidence:.2%} confidence")