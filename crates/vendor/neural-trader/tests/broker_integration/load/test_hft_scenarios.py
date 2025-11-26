"""
High-Frequency Trading (HFT) Load Testing Scenarios

This module contains comprehensive load tests for high-frequency trading scenarios,
testing system performance under extreme conditions with high order volumes and
rapid market data processing.
"""

import asyncio
import concurrent.futures
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable
import pytest
import psutil
import threading
import queue
import random

from ..unit.brokers.mock_framework import MockManager, AlpacaMock, NewsAPIMock


class HFTLoadTester:
    """High-frequency trading load testing framework"""
    
    def __init__(self, mock_manager: MockManager):
        self.mock_manager = mock_manager
        self.metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None
        self.errors = []
        self.active_threads = []
        self.stop_event = threading.Event()
    
    def start_test(self):
        """Start load test timing"""
        self.start_time = time.time()
        self.metrics.clear()
        self.errors.clear()
        self.stop_event.clear()
    
    def stop_test(self):
        """Stop load test timing"""
        self.end_time = time.time()
        self.stop_event.set()
        
        # Wait for all threads to complete
        for thread in self.active_threads:
            thread.join(timeout=5.0)
        
        self.active_threads.clear()
    
    def get_duration(self) -> float:
        """Get test duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def record_latency(self, operation: str, latency: float):
        """Record operation latency"""
        self.metrics[f"{operation}_latency"].append(latency)
    
    def record_throughput(self, operation: str, count: int):
        """Record operation throughput"""
        self.metrics[f"{operation}_throughput"].append(count)
    
    def record_error(self, operation: str, error: Exception):
        """Record operation error"""
        self.errors.append({
            "operation": operation,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance test summary"""
        summary = {
            "duration": self.get_duration(),
            "total_errors": len(self.errors),
            "operations": {}
        }
        
        for metric_name, values in self.metrics.items():
            if "_latency" in metric_name:
                operation = metric_name.replace("_latency", "")
                summary["operations"][operation] = {
                    "latency_stats": {
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "mean": statistics.mean(values) if values else 0,
                        "median": statistics.median(values) if values else 0,
                        "p95": self._percentile(values, 95) if values else 0,
                        "p99": self._percentile(values, 99) if values else 0
                    },
                    "total_operations": len(values)
                }
            elif "_throughput" in metric_name:
                operation = metric_name.replace("_throughput", "")
                if operation not in summary["operations"]:
                    summary["operations"][operation] = {}
                summary["operations"][operation]["throughput"] = sum(values) if values else 0
        
        return summary
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class TestHFTScenarios:
    """Test high-frequency trading scenarios"""
    
    @pytest.fixture(scope="function")
    def hft_setup(self):
        """Setup HFT testing environment"""
        mock_manager = MockManager()
        mock_manager.setup_broker_mocks(["alpaca"])
        mock_manager.setup_news_mock()
        
        # Configure for high performance
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        alpaca_mock.set_latency(1)  # 1ms latency
        alpaca_mock.disable_rate_limiting()
        
        load_tester = HFTLoadTester(mock_manager)
        
        yield load_tester, mock_manager
        
        mock_manager.reset_all_mocks()
    
    @pytest.mark.load
    @pytest.mark.timeout(60)
    def test_rapid_order_submission(self, hft_setup):
        """Test system under rapid order submission - Target: 1000 orders/second"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        target_orders_per_second = 1000
        test_duration = 10  # seconds
        total_target_orders = target_orders_per_second * test_duration
        
        load_tester.start_test()
        
        # Submit orders as fast as possible
        orders_submitted = 0
        batch_size = 100
        
        for batch in range(0, total_target_orders, batch_size):
            batch_start = time.time()
            
            # Submit batch of orders
            for i in range(batch_size):
                if time.time() - load_tester.start_time > test_duration:
                    break
                
                try:
                    order_start = time.time()
                    
                    order = alpaca_mock.submit_order(
                        symbol="SPY",
                        qty=100,
                        side="buy" if i % 2 == 0 else "sell",
                        order_type="market"
                    )
                    
                    order_latency = time.time() - order_start
                    load_tester.record_latency("order_submission", order_latency)
                    orders_submitted += 1
                    
                    # Simulate immediate fill for HFT
                    alpaca_mock.simulate_order_fill(
                        order["id"], 
                        400.0 + (i * 0.001), 
                        100
                    )
                    
                except Exception as e:
                    load_tester.record_error("order_submission", e)
            
            batch_duration = time.time() - batch_start
            load_tester.record_throughput("order_submission", batch_size)
            
            # Check if we're meeting throughput targets
            if batch_duration > 0:
                current_rate = batch_size / batch_duration
                if current_rate < target_orders_per_second * 0.8:  # 80% of target
                    pytest.fail(f"Order submission rate too low: {current_rate:.1f} orders/sec")
        
        load_tester.stop_test()
        
        # Analyze performance
        summary = load_tester.get_performance_summary()
        
        # Assertions
        assert orders_submitted >= total_target_orders * 0.9  # At least 90% of target
        assert summary["total_errors"] < orders_submitted * 0.01  # Less than 1% errors
        
        order_stats = summary["operations"]["order_submission"]["latency_stats"]
        assert order_stats["p95"] < 0.010  # 95th percentile < 10ms
        assert order_stats["p99"] < 0.020  # 99th percentile < 20ms
        assert order_stats["mean"] < 0.005  # Mean < 5ms
        
        actual_throughput = orders_submitted / load_tester.get_duration()
        assert actual_throughput >= target_orders_per_second * 0.8  # 80% of target
    
    @pytest.mark.load
    @pytest.mark.timeout(30)
    def test_market_data_processing_rate(self, hft_setup):
        """Test high-volume market data processing - Target: 100,000 ticks/second"""
        load_tester, mock_manager = hft_setup
        
        target_ticks_per_second = 100000
        test_duration = 5  # seconds
        total_target_ticks = target_ticks_per_second * test_duration
        
        # Simulate market data processing
        market_data_queue = queue.Queue()
        processed_ticks = 0
        
        def market_data_producer():
            """Produce market data ticks"""
            tick_count = 0
            while not load_tester.stop_event.is_set() and tick_count < total_target_ticks:
                tick = {
                    "symbol": "SPY",
                    "price": 400.0 + random.uniform(-1, 1),
                    "size": random.randint(100, 1000),
                    "timestamp": time.time()
                }
                market_data_queue.put(tick)
                tick_count += 1
                
                # Throttle to approximate target rate
                if tick_count % 1000 == 0:
                    time.sleep(0.001)
        
        def market_data_processor():
            """Process market data ticks"""
            nonlocal processed_ticks
            while not load_tester.stop_event.is_set():
                try:
                    tick = market_data_queue.get(timeout=0.1)
                    
                    process_start = time.time()
                    
                    # Simulate tick processing (price validation, aggregation, etc.)
                    processed_price = tick["price"]
                    processed_size = tick["size"]
                    processing_latency = time.time() - process_start
                    
                    load_tester.record_latency("tick_processing", processing_latency)
                    processed_ticks += 1
                    
                    market_data_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    load_tester.record_error("tick_processing", e)
        
        load_tester.start_test()
        
        # Start producer and processor threads
        producer_thread = threading.Thread(target=market_data_producer)
        processor_thread = threading.Thread(target=market_data_processor)
        
        load_tester.active_threads = [producer_thread, processor_thread]
        
        producer_thread.start()
        processor_thread.start()
        
        # Run for test duration
        time.sleep(test_duration)
        
        load_tester.stop_test()
        
        # Analyze performance
        summary = load_tester.get_performance_summary()
        
        # Assertions
        assert processed_ticks >= total_target_ticks * 0.8  # At least 80% of target
        assert summary["total_errors"] < processed_ticks * 0.001  # Less than 0.1% errors
        
        if "tick_processing" in summary["operations"]:
            tick_stats = summary["operations"]["tick_processing"]["latency_stats"]
            assert tick_stats["p95"] < 0.001  # 95th percentile < 1ms
            assert tick_stats["p99"] < 0.002  # 99th percentile < 2ms
            assert tick_stats["mean"] < 0.0005  # Mean < 0.5ms
        
        actual_throughput = processed_ticks / load_tester.get_duration()
        assert actual_throughput >= target_ticks_per_second * 0.7  # 70% of target
    
    @pytest.mark.load
    def test_concurrent_strategy_execution(self, hft_setup):
        """Test multiple strategies running concurrently"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        num_strategies = 10
        orders_per_strategy = 100
        
        strategy_results = {}
        
        def run_strategy(strategy_id: int):
            """Run individual trading strategy"""
            strategy_orders = []
            strategy_errors = []
            
            for i in range(orders_per_strategy):
                try:
                    order_start = time.time()
                    
                    # Simulate strategy logic
                    symbol = random.choice(["SPY", "QQQ", "IWM", "AAPL", "MSFT"])
                    side = random.choice(["buy", "sell"])
                    
                    order = alpaca_mock.submit_order(
                        symbol=symbol,
                        qty=100,
                        side=side,
                        order_type="market"
                    )
                    
                    order_latency = time.time() - order_start
                    load_tester.record_latency(f"strategy_{strategy_id}", order_latency)
                    
                    strategy_orders.append(order)
                    
                    # Simulate order fill
                    fill_price = 400.0 + random.uniform(-5, 5)
                    alpaca_mock.simulate_order_fill(order["id"], fill_price, 100)
                    
                    # Small delay between orders
                    time.sleep(0.01)
                    
                except Exception as e:
                    strategy_errors.append(e)
                    load_tester.record_error(f"strategy_{strategy_id}", e)
            
            strategy_results[strategy_id] = {
                "orders": strategy_orders,
                "errors": strategy_errors
            }
        
        load_tester.start_test()
        
        # Run strategies concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_strategies) as executor:
            futures = [
                executor.submit(run_strategy, strategy_id) 
                for strategy_id in range(num_strategies)
            ]
            
            # Wait for all strategies to complete
            concurrent.futures.wait(futures, timeout=60)
        
        load_tester.stop_test()
        
        # Analyze results
        total_orders = sum(len(result["orders"]) for result in strategy_results.values())
        total_errors = sum(len(result["errors"]) for result in strategy_results.values())
        
        # Assertions
        assert len(strategy_results) == num_strategies
        assert total_orders >= num_strategies * orders_per_strategy * 0.9  # 90% success
        assert total_errors < total_orders * 0.02  # Less than 2% errors
        
        # Check that all strategies executed concurrently
        assert load_tester.get_duration() < orders_per_strategy * 0.1  # Much faster than sequential
        
        # Verify no resource contention issues
        summary = load_tester.get_performance_summary()
        assert summary["total_errors"] < total_orders * 0.01  # Less than 1% total errors
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_memory_usage_under_load(self, hft_setup):
        """Test memory usage under sustained load"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = [initial_memory]
        max_memory_growth = 100  # MB
        
        def monitor_memory():
            """Monitor memory usage during test"""
            while not load_tester.stop_event.is_set():
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                time.sleep(1)
        
        load_tester.start_test()
        
        # Start memory monitoring
        memory_thread = threading.Thread(target=monitor_memory)
        memory_thread.start()
        load_tester.active_threads.append(memory_thread)
        
        # Generate sustained load
        orders_submitted = 0
        test_duration = 30  # seconds
        
        while time.time() - load_tester.start_time < test_duration:
            try:
                # Submit orders in batches
                for i in range(10):
                    order = alpaca_mock.submit_order(
                        symbol="SPY",
                        qty=100,
                        side="buy" if i % 2 == 0 else "sell",
                        order_type="market"
                    )
                    
                    # Fill immediately to avoid accumulating unfilled orders
                    alpaca_mock.simulate_order_fill(order["id"], 400.0, 100)
                    orders_submitted += 1
                
                # Small delay between batches
                time.sleep(0.1)
                
            except Exception as e:
                load_tester.record_error("sustained_load", e)
        
        load_tester.stop_test()
        
        # Analyze memory usage
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = max_memory - initial_memory
        
        # Assertions
        assert orders_submitted > 1000  # Significant load generated
        assert memory_growth < max_memory_growth  # Memory growth within limits
        
        # Check for memory leaks (final memory should be close to initial)
        final_growth = final_memory - initial_memory
        assert final_growth < max_memory_growth * 0.5  # Final growth < 50% of max allowed
        
        print(f"Memory stats: Initial={initial_memory:.1f}MB, "
              f"Max={max_memory:.1f}MB, Final={final_memory:.1f}MB, "
              f"Growth={memory_growth:.1f}MB")
    
    @pytest.mark.load
    def test_cpu_usage_under_load(self, hft_setup):
        """Test CPU usage under high load"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        cpu_samples = []
        max_cpu_usage = 80  # percent
        
        def monitor_cpu():
            """Monitor CPU usage during test"""
            while not load_tester.stop_event.is_set():
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
        
        load_tester.start_test()
        
        # Start CPU monitoring
        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()
        load_tester.active_threads.append(cpu_thread)
        
        # Generate CPU-intensive load
        def cpu_intensive_task():
            """Simulate CPU-intensive trading calculations"""
            orders_processed = 0
            while time.time() - load_tester.start_time < 10:
                # Simulate complex calculations
                for i in range(1000):
                    # Price calculations
                    price = 400.0 + (i * 0.001)
                    volatility = price * 0.02
                    
                    # Risk calculations
                    risk_factor = volatility / price
                    position_size = 100000 / (price * risk_factor)
                    
                    # Technical indicators
                    sma = sum(range(i, i+20)) / 20 if i >= 20 else price
                    
                orders_processed += 1
                
                if orders_processed % 100 == 0:
                    # Submit actual order occasionally
                    try:
                        order = alpaca_mock.submit_order(
                            symbol="SPY",
                            qty=100,
                            side="buy",
                            order_type="market"
                        )
                        alpaca_mock.simulate_order_fill(order["id"], price, 100)
                    except Exception as e:
                        load_tester.record_error("cpu_intensive", e)
        
        # Run CPU-intensive tasks in parallel
        num_workers = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(num_workers)]
            concurrent.futures.wait(futures, timeout=15)
        
        load_tester.stop_test()
        
        # Analyze CPU usage
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # Assertions
            assert avg_cpu < max_cpu_usage  # Average CPU within limits
            assert max_cpu < 95  # Never exceed 95% CPU
            
            print(f"CPU stats: Average={avg_cpu:.1f}%, Max={max_cpu:.1f}%")
        else:
            pytest.skip("No CPU samples collected")
    
    @pytest.mark.load
    def test_network_simulation_load(self, hft_setup):
        """Test system behavior under network load simulation"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        news_mock = mock_manager.get_news_mock()
        
        # Simulate varying network conditions
        network_conditions = [
            {"latency": 1, "description": "Low latency"},
            {"latency": 10, "description": "Normal latency"},
            {"latency": 50, "description": "High latency"},
            {"latency": 100, "description": "Very high latency"},
        ]
        
        results_by_condition = {}
        
        for condition in network_conditions:
            # Set network condition
            alpaca_mock.set_latency(condition["latency"])
            news_mock.set_latency(condition["latency"])
            
            condition_start = time.time()
            orders_submitted = 0
            condition_errors = 0
            
            # Test under this network condition
            for i in range(100):
                try:
                    order_start = time.time()
                    
                    order = alpaca_mock.submit_order(
                        symbol="SPY",
                        qty=100,
                        side="buy" if i % 2 == 0 else "sell",
                        order_type="market"
                    )
                    
                    order_latency = time.time() - order_start
                    load_tester.record_latency(f"network_{condition['latency']}ms", order_latency)
                    
                    alpaca_mock.simulate_order_fill(order["id"], 400.0, 100)
                    orders_submitted += 1
                    
                except Exception as e:
                    condition_errors += 1
                    load_tester.record_error(f"network_{condition['latency']}ms", e)
            
            condition_duration = time.time() - condition_start
            
            results_by_condition[condition["latency"]] = {
                "orders": orders_submitted,
                "errors": condition_errors,
                "duration": condition_duration,
                "throughput": orders_submitted / condition_duration if condition_duration > 0 else 0
            }
        
        # Analyze network impact
        for latency, results in results_by_condition.items():
            # Higher latency should still maintain reasonable throughput
            expected_min_throughput = 50 / (1 + latency / 10)  # Adaptive expectation
            assert results["throughput"] >= expected_min_throughput
            
            # Error rate should remain low regardless of latency
            error_rate = results["errors"] / (results["orders"] + results["errors"])
            assert error_rate < 0.05  # Less than 5% errors
        
        # Verify latency correlation
        latencies = list(results_by_condition.keys())
        throughputs = [results_by_condition[l]["throughput"] for l in latencies]
        
        # Generally, higher latency should result in lower throughput
        assert throughputs[0] >= throughputs[-1]  # Lowest latency >= highest latency throughput
    
    @pytest.mark.load
    @pytest.mark.parametrize("order_rate", [100, 500, 1000, 2000])
    def test_scalability_limits(self, hft_setup, order_rate):
        """Test system scalability at different order rates"""
        load_tester, mock_manager = hft_setup
        alpaca_mock = mock_manager.get_broker_mock("alpaca")
        
        test_duration = 5  # seconds
        target_orders = order_rate * test_duration
        
        load_tester.start_test()
        
        orders_submitted = 0
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            batch_start = time.time()
            batch_size = min(10, target_orders - orders_submitted)
            
            for i in range(batch_size):
                try:
                    order = alpaca_mock.submit_order(
                        symbol="SPY",
                        qty=100,
                        side="buy" if i % 2 == 0 else "sell",
                        order_type="market"
                    )
                    
                    alpaca_mock.simulate_order_fill(order["id"], 400.0, 100)
                    orders_submitted += 1
                    
                except Exception as e:
                    load_tester.record_error("scalability", e)
            
            # Control rate
            batch_duration = time.time() - batch_start
            target_batch_duration = batch_size / order_rate
            
            if batch_duration < target_batch_duration:
                time.sleep(target_batch_duration - batch_duration)
        
        load_tester.stop_test()
        
        actual_rate = orders_submitted / load_tester.get_duration()
        
        # Assertions based on order rate
        if order_rate <= 500:
            # Should easily handle low rates
            assert actual_rate >= order_rate * 0.95  # 95% of target
        elif order_rate <= 1000:
            # Should handle medium rates with some degradation
            assert actual_rate >= order_rate * 0.8  # 80% of target
        else:
            # High rates may show more degradation
            assert actual_rate >= order_rate * 0.6  # 60% of target
        
        # Error rate should remain reasonable
        summary = load_tester.get_performance_summary()
        error_rate = summary["total_errors"] / orders_submitted if orders_submitted > 0 else 0
        assert error_rate < 0.1  # Less than 10% errors even at high rates