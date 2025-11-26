"""
Integration tests for the complete orchestrator system

Tests the integration between:
- API Selector
- Execution Router  
- Failover Manager
- Performance monitoring
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
import numpy as np
import time

from src.trading_apis.orchestrator import APISelector, ExecutionRouter, FailoverManager
from src.trading_apis.orchestrator.execution_router import OrderType, ExecutionStrategy
from tests.trading_apis.monitoring_dashboard import PerformanceMonitor
from tests.trading_apis.benchmarks import LatencyBenchmark


class IntegratedMockAPI:
    """Mock API with realistic behavior for integration testing"""
    
    def __init__(self, name: str, 
                 base_latency_ms: float = 1.0,
                 latency_variance: float = 0.5,
                 base_fail_rate: float = 0.01,
                 rate_limit: int = 1000):
        self.name = name
        self.base_latency_ms = base_latency_ms
        self.latency_variance = latency_variance
        self.base_fail_rate = base_fail_rate
        self.rate_limit = rate_limit
        self.requests_made = 0
        self.last_reset = datetime.now()
        
        # State simulation
        self.is_healthy = True
        self.load_factor = 1.0  # Multiplier for latency under load
        
    async def health_check(self):
        """Health check with realistic latency"""
        await self._simulate_request()
        return {"status": "ok", "server_time": datetime.now().isoformat()}
    
    async def get_server_time(self):
        """Get server time"""
        return await self.health_check()
    
    async def get_order_book(self, symbol: str):
        """Get order book data"""
        await self._simulate_request()
        
        # Simulate market data
        base_price = 50000.0 if symbol == "BTCUSD" else 3000.0
        spread = 0.01
        
        return {
            'bids': [
                {'price': base_price * (1 - spread * i), 'quantity': 1000 * (1 + i)}
                for i in range(1, 6)
            ],
            'asks': [
                {'price': base_price * (1 + spread * i), 'quantity': 1000 * (1 + i)}
                for i in range(1, 6)
            ]
        }
    
    async def place_order(self, **kwargs):
        """Place an order"""
        await self._simulate_request()
        
        # Simulate order execution
        return {
            'order_id': f'{self.name}_order_{int(time.time() * 1000)}',
            'status': 'filled',
            'filled_quantity': kwargs.get('quantity', 0),
            'avg_fill_price': kwargs.get('price', 50000.0),
            'fees': kwargs.get('quantity', 0) * 0.001
        }
    
    async def get_account_info(self):
        """Get account information"""
        await self._simulate_request()
        return {"balance": 10000.0, "available": 8000.0}
    
    async def _simulate_request(self):
        """Simulate realistic request processing"""
        # Check rate limits
        now = datetime.now()
        if (now - self.last_reset).total_seconds() > 60:
            self.requests_made = 0
            self.last_reset = now
        
        self.requests_made += 1
        
        if self.requests_made > self.rate_limit:
            raise Exception(f"Rate limit exceeded for {self.name}")
        
        # Simulate latency with variance and load
        latency_ms = self.base_latency_ms * self.load_factor
        latency_ms += np.random.normal(0, self.latency_variance)
        latency_ms = max(0.1, latency_ms)  # Minimum 0.1ms
        
        await asyncio.sleep(latency_ms / 1000)
        
        # Simulate failures
        fail_rate = self.base_fail_rate
        if not self.is_healthy:
            fail_rate = 0.8  # High failure rate when unhealthy
        
        if np.random.random() < fail_rate:
            raise Exception(f"Simulated failure in {self.name}")
    
    def set_healthy(self, healthy: bool):
        """Set health status"""
        self.is_healthy = healthy
    
    def set_load_factor(self, factor: float):
        """Set load factor for latency simulation"""
        self.load_factor = factor


@pytest.mark.asyncio
class TestOrchestratorIntegration:
    """Test complete orchestrator integration"""
    
    async def setup_method(self):
        """Setup integration test environment"""
        # Create mock APIs with different characteristics
        self.apis = {
            "primary": IntegratedMockAPI("primary", base_latency_ms=0.5, base_fail_rate=0.01),
            "secondary": IntegratedMockAPI("secondary", base_latency_ms=2.0, base_fail_rate=0.02),
            "backup": IntegratedMockAPI("backup", base_latency_ms=10.0, base_fail_rate=0.05)
        }
        
        # Create orchestrator components
        self.api_selector = APISelector(list(self.apis.keys()))
        self.execution_router = ExecutionRouter(self.api_selector, self.apis)
        self.failover_manager = FailoverManager(
            self.api_selector,
            self.execution_router,
            self.apis,
            health_check_interval=0.1,  # Fast for testing
            failure_threshold=2,
            recovery_threshold=2
        )
        
        # Start failover monitoring
        await self.failover_manager.start_monitoring()
    
    async def teardown_method(self):
        """Cleanup after test"""
        if self.failover_manager:
            await self.failover_manager.stop_monitoring()
    
    async def test_normal_operation_flow(self):
        """Test normal operation flow through all components"""
        # Execute a series of orders
        results = []
        
        for i in range(5):
            order_results = await self.execution_router.execute_order(
                symbol="BTCUSD",
                quantity=1000.0,
                order_type=OrderType.MARKET,
                strategy=ExecutionStrategy.BALANCED
            )
            results.extend(order_results)
        
        # Verify all orders were executed
        assert len(results) >= 5
        assert all(result.filled_quantity > 0 for result in results)
        
        # Check that API selector metrics were updated
        summary = self.api_selector.get_metrics_summary()
        assert len(summary) == 3
        assert all(summary[api]['success_count'] > 0 for api in self.apis.keys())
    
    async def test_failover_during_execution(self):
        """Test failover handling during order execution"""
        # Start with normal execution
        initial_results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=500.0,
            order_type=OrderType.MARKET
        )
        
        # Make primary API unhealthy
        self.apis["primary"].set_healthy(False)
        
        # Wait for failover detection
        await asyncio.sleep(0.5)
        
        # Execute another order - should avoid unhealthy API
        failover_results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=500.0,
            order_type=OrderType.MARKET
        )
        
        # Verify execution continued with healthy APIs
        assert len(failover_results) > 0
        used_apis = set(result.api for result in failover_results)
        assert "primary" not in used_apis or all(
            result.status != "failed" for result in failover_results 
            if result.api == "primary"
        )
    
    async def test_load_balancing_under_stress(self):
        """Test load balancing under high load"""
        # Simulate high load on primary API
        self.apis["primary"].set_load_factor(5.0)  # 5x latency
        
        # Execute multiple orders concurrently
        tasks = []
        for i in range(10):
            task = self.execution_router.execute_order(
                symbol="BTCUSD",
                quantity=200.0,
                order_type=OrderType.MARKET,
                strategy=ExecutionStrategy.AGGRESSIVE,
                urgency=0.8
            )
            tasks.append(task)
        
        # Wait for all executions
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for order_results in all_results:
            results.extend(order_results)
        
        # Verify load was distributed
        api_usage = {}
        for result in results:
            api_usage[result.api] = api_usage.get(result.api, 0) + 1
        
        # Primary should have reduced usage due to high latency
        assert len(api_usage) > 1  # Multiple APIs used
        
        # Check that some orders were routed to faster APIs
        faster_apis_used = api_usage.get("secondary", 0) + api_usage.get("backup", 0)
        assert faster_apis_used > 0
    
    async def test_recovery_after_failure(self):
        """Test recovery and rebalancing after API failure"""
        # Make secondary API fail
        self.apis["secondary"].set_healthy(False)
        
        # Wait for failover detection
        await asyncio.sleep(0.3)
        
        # Verify API is marked unhealthy
        healthy_apis = self.failover_manager.get_healthy_apis()
        assert "secondary" not in healthy_apis
        
        # Restore API health
        self.apis["secondary"].set_healthy(True)
        
        # Wait for recovery detection
        await asyncio.sleep(0.5)
        
        # Verify API is back in healthy list
        healthy_apis = self.failover_manager.get_healthy_apis()
        assert "secondary" in healthy_apis
        
        # Verify orders can be routed to recovered API
        results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=1000.0,
            order_type=OrderType.MARKET
        )
        
        used_apis = set(result.api for result in results)
        # Secondary should be available for routing again
        # (might not be used immediately but should be available)
        assert len(used_apis) >= 1
    
    async def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring"""
        # Create performance monitor
        monitor = PerformanceMonitor(self.apis, update_interval=0.1)
        await monitor.start_monitoring()
        
        try:
            # Let monitor collect baseline data
            await asyncio.sleep(0.5)
            
            # Execute some orders
            for i in range(3):
                await self.execution_router.execute_order(
                    symbol="BTCUSD",
                    quantity=500.0,
                    order_type=OrderType.MARKET
                )
                await asyncio.sleep(0.2)
            
            # Check monitor has collected data
            summary = monitor.get_performance_summary()
            assert summary['total_apis'] == 3
            assert summary['avg_latency_us'] > 0
            
            # Check individual API status
            for api_name in self.apis.keys():
                status = monitor.get_current_status()[api_name]
                assert status.requests_per_second > 0
                assert status.avg_latency_us > 0
        
        finally:
            await monitor.stop_monitoring()
    
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker behavior across components"""
        # Generate failures to trigger circuit breaker
        api_name = "backup"
        
        # Make API fail consistently
        self.apis[api_name].set_healthy(False)
        
        # Try multiple operations to trigger circuit breaker
        for i in range(5):
            try:
                await self.execution_router.execute_order(
                    symbol="BTCUSD",
                    quantity=100.0,
                    order_type=OrderType.MARKET
                )
            except:
                pass  # Expected failures
            
            await asyncio.sleep(0.1)
        
        # Wait for circuit breaker detection
        await asyncio.sleep(0.5)
        
        # Verify API is in failover state
        healthy_apis = self.failover_manager.get_healthy_apis()
        assert api_name not in healthy_apis
        
        # Verify API selector avoids circuit broken API
        selected_apis = []
        for i in range(10):
            try:
                selected = self.api_selector.select_api()
                selected_apis.append(selected)
            except:
                pass
        
        # Should mostly avoid the failed API
        failed_api_usage = selected_apis.count(api_name)
        total_selections = len(selected_apis)
        if total_selections > 0:
            usage_rate = failed_api_usage / total_selections
            assert usage_rate < 0.3  # Should use failed API less than 30% of time
    
    async def test_rate_limit_handling(self):
        """Test rate limit handling across components"""
        # Set low rate limit on primary API
        self.apis["primary"].rate_limit = 5
        
        # Execute many small orders rapidly
        tasks = []
        for i in range(20):
            task = self.execution_router.execute_order(
                symbol="BTCUSD",
                quantity=50.0,
                order_type=OrderType.MARKET,
                strategy=ExecutionStrategy.AGGRESSIVE
            )
            tasks.append(task)
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed, some should be routed to other APIs
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0
        
        # Check that traffic was distributed to avoid rate limits
        all_executions = []
        for result_set in successful_results:
            if isinstance(result_set, list):
                all_executions.extend(result_set)
        
        api_usage = {}
        for execution in all_executions:
            api_usage[execution.api] = api_usage.get(execution.api, 0) + 1
        
        # Should have used multiple APIs
        assert len(api_usage) > 1
    
    async def test_benchmark_integration(self):
        """Test integration with benchmarking tools"""
        # Create benchmark
        benchmark = LatencyBenchmark(self.apis)
        
        # Run quick benchmark
        health_stats = await benchmark.benchmark_health_checks(
            duration_seconds=5,
            requests_per_second=2
        )
        
        # Verify benchmark collected data
        assert len(health_stats) == 3
        for api_name, stats in health_stats.items():
            assert stats.total_requests > 0
            assert stats.avg_latency_us > 0
            assert 0 <= stats.success_rate <= 1
        
        # Verify benchmark integrated with API selector metrics
        selector_summary = self.api_selector.get_metrics_summary()
        for api_name in self.apis.keys():
            assert selector_summary[api_name]['success_count'] > 0
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Start with all systems healthy
        initial_summary = self.api_selector.get_metrics_summary()
        assert len(initial_summary) == 3
        
        # 2. Execute orders under normal conditions
        normal_results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=2000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.BALANCED
        )
        assert len(normal_results) > 0
        assert all(r.filled_quantity > 0 for r in normal_results)
        
        # 3. Introduce system stress
        self.apis["primary"].set_load_factor(3.0)
        self.apis["secondary"].set_healthy(False)
        
        # 4. Wait for system adaptation
        await asyncio.sleep(0.5)
        
        # 5. Execute orders under stress
        stress_results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=1500.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.AGGRESSIVE
        )
        assert len(stress_results) > 0
        
        # 6. Verify system adapted by using healthy APIs
        stress_apis = set(r.api for r in stress_results)
        # Should primarily use backup API since primary is slow and secondary is down
        assert "backup" in stress_apis
        
        # 7. Restore system health
        self.apis["primary"].set_load_factor(1.0)
        self.apis["secondary"].set_healthy(True)
        
        # 8. Wait for recovery
        await asyncio.sleep(0.5)
        
        # 9. Execute final orders
        recovery_results = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=1000.0,
            order_type=OrderType.MARKET,
            strategy=ExecutionStrategy.BALANCED
        )
        assert len(recovery_results) > 0
        
        # 10. Verify system returned to balanced operation
        recovery_apis = set(r.api for r in recovery_results)
        assert len(recovery_apis) >= 2  # Should use multiple APIs again
        
        # 11. Check final health status
        final_healthy = self.failover_manager.get_healthy_apis()
        assert len(final_healthy) >= 2  # Most APIs should be healthy
        
        # 12. Verify metrics were properly tracked
        final_summary = self.api_selector.get_metrics_summary()
        for api_name in self.apis.keys():
            assert final_summary[api_name]['success_count'] > initial_summary[api_name]['success_count']
    
    async def test_concurrent_operations_stability(self):
        """Test system stability under concurrent operations"""
        # Define different types of concurrent operations
        async def health_checks():
            for _ in range(10):
                try:
                    self.api_selector.select_api()
                    await asyncio.sleep(0.05)
                except:
                    pass
        
        async def order_executions():
            for _ in range(5):
                try:
                    await self.execution_router.execute_order(
                        symbol="BTCUSD",
                        quantity=200.0,
                        order_type=OrderType.MARKET
                    )
                    await asyncio.sleep(0.1)
                except:
                    pass
        
        async def status_checks():
            for _ in range(8):
                try:
                    self.failover_manager.get_all_health_status()
                    await asyncio.sleep(0.08)
                except:
                    pass
        
        # Run all operations concurrently
        tasks = [
            health_checks(),
            order_executions(),
            status_checks(),
            health_checks(),  # Multiple instances
            order_executions()
        ]
        
        # Should complete without deadlocks or crashes
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify system is still operational
        final_status = self.failover_manager.get_all_health_status()
        assert len(final_status) == 3
        
        # Should be able to execute one more order
        final_result = await self.execution_router.execute_order(
            symbol="BTCUSD",
            quantity=100.0,
            order_type=OrderType.MARKET
        )
        assert len(final_result) > 0