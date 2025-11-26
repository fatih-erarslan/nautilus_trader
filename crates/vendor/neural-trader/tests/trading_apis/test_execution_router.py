"""
Unit tests for Execution Router component
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import numpy as np

from src.trading_apis.orchestrator.execution_router import (
    ExecutionRouter, OrderType, ExecutionStrategy, OrderSlice,
    ExecutionResult, MarketDepth
)
from src.trading_apis.orchestrator.api_selector import APISelector


class MockAPI:
    """Mock API for testing"""
    
    def __init__(self, name: str, latency_ms: float = 1.0, fail_rate: float = 0.0):
        self.name = name
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self.call_count = 0
    
    async def get_order_book(self, symbol: str):
        """Mock order book fetch"""
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)
        
        if np.random.random() < self.fail_rate:
            raise Exception(f"API {self.name} failed")
        
        return {
            'bids': [
                {'price': 100.0, 'quantity': 1000},
                {'price': 99.5, 'quantity': 2000},
                {'price': 99.0, 'quantity': 1500}
            ],
            'asks': [
                {'price': 100.5, 'quantity': 1000},
                {'price': 101.0, 'quantity': 2000},
                {'price': 101.5, 'quantity': 1500}
            ]
        }
    
    async def place_order(self, **kwargs):
        """Mock order placement"""
        self.call_count += 1
        await asyncio.sleep(self.latency_ms / 1000)
        
        if np.random.random() < self.fail_rate:
            raise Exception(f"Order failed on {self.name}")
        
        return {
            'order_id': f'{self.name}_order_{self.call_count}',
            'status': 'filled',
            'filled_quantity': kwargs.get('quantity', 0),
            'avg_fill_price': kwargs.get('price', 100.0),
            'fees': kwargs.get('quantity', 0) * 0.001
        }


class TestOrderSlice:
    """Test OrderSlice dataclass"""
    
    def test_order_slice_creation(self):
        """Test order slice creation"""
        slice_order = OrderSlice(
            api="test_api",
            symbol="BTCUSD",
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        assert slice_order.api == "test_api"
        assert slice_order.symbol == "BTCUSD"
        assert slice_order.quantity == 100.0
        assert slice_order.order_type == OrderType.MARKET
        assert slice_order.time_in_force == "IOC"


class TestExecutionResult:
    """Test ExecutionResult dataclass"""
    
    def test_execution_result_creation(self):
        """Test execution result creation"""
        result = ExecutionResult(
            order_id="test_order",
            api="test_api",
            status="filled",
            filled_quantity=100.0,
            avg_fill_price=100.5,
            latency_us=1000.0,
            timestamp=datetime.now()
        )
        
        assert result.order_id == "test_order"
        assert result.api == "test_api"
        assert result.status == "filled"
        assert result.filled_quantity == 100.0
        assert result.avg_fill_price == 100.5
        assert result.latency_us == 1000.0


class TestMarketDepth:
    """Test MarketDepth dataclass"""
    
    def test_market_depth_creation(self):
        """Test market depth creation"""
        depth = MarketDepth(
            api="test_api",
            symbol="BTCUSD",
            bids=[(100.0, 1000), (99.5, 2000)],
            asks=[(100.5, 1000), (101.0, 2000)],
            timestamp=datetime.now(),
            latency_us=500.0
        )
        
        assert depth.api == "test_api"
        assert depth.symbol == "BTCUSD"
        assert len(depth.bids) == 2
        assert len(depth.asks) == 2
        assert depth.latency_us == 500.0


class TestExecutionRouter:
    """Test ExecutionRouter functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.apis = {
            "fast_api": MockAPI("fast_api", latency_ms=0.5),
            "medium_api": MockAPI("medium_api", latency_ms=2.0),
            "slow_api": MockAPI("slow_api", latency_ms=10.0)
        }
        
        self.api_selector = APISelector(list(self.apis.keys()))
        self.router = ExecutionRouter(self.api_selector, self.apis)
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.router.api_selector == self.api_selector
        assert self.router.apis == self.apis
        assert self.router.max_slice_ratio == 0.3
        assert self.router.min_slice_size == 100.0
    
    def test_calculate_liquidity_score(self):
        """Test liquidity score calculation"""
        depth = MarketDepth(
            api="test_api",
            symbol="BTCUSD",
            bids=[(100.0, 1000), (99.5, 2000)],
            asks=[(100.5, 1000), (101.0, 2000)],
            timestamp=datetime.now(),
            latency_us=500.0
        )
        
        # Test with small order
        score = self.router._calculate_liquidity_score(depth, 500.0, OrderType.MARKET)
        assert score > 0.8  # Should be high for small order
        
        # Test with large order
        score = self.router._calculate_liquidity_score(depth, 5000.0, OrderType.MARKET)
        assert score < 0.8  # Should be lower for large order
    
    def test_calculate_available_liquidity(self):
        """Test available liquidity calculation"""
        depth = MarketDepth(
            api="test_api",
            symbol="BTCUSD",
            bids=[(100.0, 1000), (99.5, 2000)],
            asks=[(100.5, 1000), (101.0, 2000)],
            timestamp=datetime.now(),
            latency_us=500.0
        )
        
        # Market order - should sum top levels
        liquidity = self.router._calculate_available_liquidity(depth, OrderType.MARKET, None)
        assert liquidity == 3000.0  # Top 3 levels
        
        # Limit order at good price
        liquidity = self.router._calculate_available_liquidity(depth, OrderType.LIMIT, 101.0)
        assert liquidity == 1000.0  # Only first level
    
    def test_strategy_to_priority(self):
        """Test strategy to priority conversion"""
        assert self.router._strategy_to_priority(ExecutionStrategy.AGGRESSIVE) == "latency"
        assert self.router._strategy_to_priority(ExecutionStrategy.PASSIVE) == "cost"
        assert self.router._strategy_to_priority(ExecutionStrategy.BALANCED) == "balanced"
    
    @pytest.mark.asyncio
    async def test_fetch_single_depth(self):
        """Test single depth fetch"""
        api = self.apis["fast_api"]
        depth = await self.router._fetch_single_depth("fast_api", api, "BTCUSD")
        
        assert depth.api == "fast_api"
        assert depth.symbol == "BTCUSD"
        assert len(depth.bids) > 0
        assert len(depth.asks) > 0
        assert depth.latency_us > 0
    
    @pytest.mark.asyncio
    async def test_fetch_market_depths(self):
        """Test market depth fetching from all APIs"""
        depths = await self.router._fetch_market_depths("BTCUSD")
        
        assert len(depths) == 3
        assert "fast_api" in depths
        assert "medium_api" in depths
        assert "slow_api" in depths
        
        # Check that all have data
        for depth in depths.values():
            assert depth.symbol == "BTCUSD"
            assert len(depth.bids) > 0
            assert len(depth.asks) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_market_depths_with_cache(self):
        """Test market depth caching"""
        # First fetch
        depths1 = await self.router._fetch_market_depths("BTCUSD")
        
        # Second fetch should use cache
        depths2 = await self.router._fetch_market_depths("BTCUSD")
        
        # Should be same objects (from cache)
        for api in self.apis.keys():
            assert depths1[api] is depths2[api]
    
    def test_score_apis_for_order(self):
        """Test API scoring for order"""
        # Create mock market depths
        depths = {}
        for api_name in self.apis.keys():
            depths[api_name] = MarketDepth(
                api=api_name,
                symbol="BTCUSD",
                bids=[(100.0, 1000), (99.5, 2000)],
                asks=[(100.5, 1000), (101.0, 2000)],
                timestamp=datetime.now(),
                latency_us=500.0 if api_name == "fast_api" else 2000.0
            )
        
        scores = self.router._score_apis_for_order(
            "BTCUSD", 1000.0, OrderType.MARKET, ExecutionStrategy.AGGRESSIVE, depths, 0.8
        )
        
        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        
        # Fast API should have higher score for aggressive strategy
        assert scores["fast_api"] > scores["slow_api"]
    
    def test_calculate_routing_simple(self):
        """Test simple routing calculation"""
        # Create mock market depths
        depths = {}
        for api_name in self.apis.keys():
            depths[api_name] = MarketDepth(
                api=api_name,
                symbol="BTCUSD",
                bids=[(100.0, 1000), (99.5, 2000)],
                asks=[(100.5, 1000), (101.0, 2000)],
                timestamp=datetime.now(),
                latency_us=500.0
            )
        
        routing = self.router._calculate_routing(
            "BTCUSD", 1000.0, OrderType.MARKET, ExecutionStrategy.BALANCED, None, depths, 0.5
        )
        
        assert len(routing) > 0
        assert all(isinstance(slice_order, OrderSlice) for slice_order in routing)
        
        # Total quantity should match
        total_quantity = sum(slice_order.quantity for slice_order in routing)
        assert abs(total_quantity - 1000.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_execute_slice(self):
        """Test single slice execution"""
        slice_order = OrderSlice(
            api="fast_api",
            symbol="BTCUSD",
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        result = await self.router._execute_slice(slice_order)
        
        assert isinstance(result, ExecutionResult)
        assert result.api == "fast_api"
        assert result.filled_quantity == 100.0
        assert result.latency_us > 0
    
    @pytest.mark.asyncio
    async def test_execute_slice_failure(self):
        """Test slice execution with failure"""
        # Use API with 100% fail rate
        failing_api = MockAPI("failing_api", fail_rate=1.0)
        self.apis["failing_api"] = failing_api
        
        slice_order = OrderSlice(
            api="failing_api",
            symbol="BTCUSD",
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        
        result = await self.router._execute_slice(slice_order)
        
        assert result.status == "failed"
        assert result.filled_quantity == 0
    
    @pytest.mark.asyncio
    async def test_execute_order_simple(self):
        """Test simple order execution"""
        results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=1000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.BALANCED
        )
        
        assert len(results) > 0
        assert all(isinstance(result, ExecutionResult) for result in results)
        
        # Check total filled quantity
        total_filled = sum(result.filled_quantity for result in results)
        assert total_filled > 0
    
    @pytest.mark.asyncio
    async def test_execute_order_aggressive(self):
        """Test aggressive order execution"""
        results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=500.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.AGGRESSIVE,
            urgency=1.0
        )
        
        assert len(results) > 0
        
        # Should prefer fast API for aggressive strategy
        api_names = [result.api for result in results]
        assert "fast_api" in api_names
    
    @pytest.mark.asyncio
    async def test_execute_order_passive(self):
        """Test passive order execution"""
        results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=500.0,
            order_type=OrderType.LIMIT,
            strategy=ExecutionStrategy.PASSIVE,
            price=100.0
        )
        
        assert len(results) > 0
        
        # Should distribute across multiple APIs
        api_names = set(result.api for result in results)
        assert len(api_names) > 1
    
    @pytest.mark.asyncio
    async def test_execute_order_with_failures(self):
        """Test order execution with some API failures"""
        # Make one API fail
        self.apis["slow_api"] = MockAPI("slow_api", fail_rate=1.0)
        
        results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=1000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.BALANCED
        )
        
        # Should still have some successful results
        successful_results = [r for r in results if r.filled_quantity > 0]
        assert len(successful_results) > 0
        
        # Should not use the failing API
        api_names = [result.api for result in successful_results]
        assert "slow_api" not in api_names
    
    def test_get_execution_analytics(self):
        """Test execution analytics generation"""
        # Add some mock execution history
        self.router.execution_history = [
            ExecutionResult(
                order_id="order1",
                api="fast_api",
                status="filled",
                filled_quantity=100.0,
                avg_fill_price=100.0,
                latency_us=500.0,
                timestamp=datetime.now(),
                fees=0.1
            ),
            ExecutionResult(
                order_id="order2",
                api="fast_api",
                status="filled",
                filled_quantity=200.0,
                avg_fill_price=101.0,
                latency_us=600.0,
                timestamp=datetime.now(),
                fees=0.2
            ),
            ExecutionResult(
                order_id="order3",
                api="medium_api",
                status="failed",
                filled_quantity=0.0,
                avg_fill_price=0.0,
                latency_us=5000.0,
                timestamp=datetime.now()
            )
        ]
        
        analytics = self.router.get_execution_analytics()
        
        assert "api_statistics" in analytics
        assert "total_executions" in analytics
        assert "avg_latency_us" in analytics
        assert "fill_rate" in analytics
        
        # Check API statistics
        fast_api_stats = analytics["api_statistics"]["fast_api"]
        assert fast_api_stats["total_orders"] == 2
        assert fast_api_stats["filled_orders"] == 2
        assert fast_api_stats["total_volume"] == 300.0 * 100.0 + 200.0 * 101.0
        
        medium_api_stats = analytics["api_statistics"]["medium_api"]
        assert medium_api_stats["total_orders"] == 1
        assert medium_api_stats["filled_orders"] == 0
    
    @pytest.mark.asyncio
    async def test_handle_failed_executions(self):
        """Test handling of failed executions"""
        # Create failed results
        failed_results = [Exception("API failed")]
        successful_results = [
            ExecutionResult(
                order_id="order1",
                api="fast_api",
                status="filled",
                filled_quantity=500.0,
                avg_fill_price=100.0,
                latency_us=500.0,
                timestamp=datetime.now()
            )
        ]
        
        # Should reallocate unfilled quantity
        await self.router._handle_failed_executions(
            failed_results, successful_results, "BTCUSD", 1000.0
        )
        
        # Check that new results were added
        assert len(successful_results) > 1


@pytest.mark.asyncio
class TestExecutionRouterIntegration:
    """Integration tests for ExecutionRouter"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.apis = {
            "primary": MockAPI("primary", latency_ms=1.0),
            "secondary": MockAPI("secondary", latency_ms=3.0),
            "backup": MockAPI("backup", latency_ms=10.0)
        }
        
        self.api_selector = APISelector(list(self.apis.keys()))
        self.router = ExecutionRouter(self.api_selector, self.apis)
    
    async def test_full_order_lifecycle(self):
        """Test complete order lifecycle"""
        # Place large order that will be split
        results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=5000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.BALANCED,
            urgency=0.7
        )
        
        # Verify results
        assert len(results) > 0
        
        total_filled = sum(result.filled_quantity for result in results)
        assert total_filled > 0
        
        # Check that multiple APIs were used
        api_names = set(result.api for result in results)
        assert len(api_names) > 1
        
        # Verify all results have proper fields
        for result in results:
            assert result.order_id
            assert result.api in self.apis
            assert result.latency_us > 0
            assert result.timestamp
    
    async def test_market_impact_scenarios(self):
        """Test various market impact scenarios"""
        # Small order - should use single API
        small_results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=50.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.STEALTH
        )
        
        # Large order - should split across APIs
        large_results = await self.router.execute_order(
            symbol="BTCUSD",
            quantity=10000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.STEALTH
        )
        
        # Large order should use more APIs
        small_apis = set(r.api for r in small_results)
        large_apis = set(r.api for r in large_results)
        
        assert len(large_apis) >= len(small_apis)
    
    async def test_concurrent_orders(self):
        """Test concurrent order execution"""
        # Submit multiple orders concurrently
        tasks = [
            self.router.execute_order("BTCUSD", 1000.0, OrderType.MARKET),
            self.router.execute_order("ETHUSD", 2000.0, OrderType.MARKET),
            self.router.execute_order("ADAUSD", 5000.0, OrderType.MARKET)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All orders should complete
        assert len(results) == 3
        assert all(len(order_results) > 0 for order_results in results)
    
    async def test_performance_under_load(self):
        """Test performance under high load"""
        # Submit many small orders
        num_orders = 50
        tasks = []
        
        for i in range(num_orders):
            task = self.router.execute_order(
                symbol="BTCUSD",
                quantity=100.0,
                order_type=OrderType.MARKET,
                strategy=ExecutionStrategy.AGGRESSIVE
            )
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Check performance
        total_time = end_time - start_time
        orders_per_second = num_orders / total_time
        
        # Should handle at least 10 orders per second
        assert orders_per_second > 10
        
        # All orders should complete
        assert len(results) == num_orders
        assert all(len(order_results) > 0 for order_results in results)