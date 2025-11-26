"""
Performance Tests for Fantasy Collective System - Concurrent Operations
Tests system behavior under various load conditions and concurrent access patterns
"""

import pytest
import asyncio
import time
import threading
import multiprocessing
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
import json
import queue

# Import system under test
from src.syndicate.syndicate_tools import (
    create_syndicate, add_member, get_syndicate_status,
    allocate_funds, distribute_profits, process_withdrawal,
    get_member_performance, simulate_allocation
)


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.response_times = []
        self.error_counts = {"total": 0, "by_operation": {}}
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_system(self):
        """Monitor system resources"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
                self.memory_samples.append(memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
            except psutil.NoSuchProcess:
                break
    
    def record_response_time(self, operation: str, duration: float, success: bool = True):
        """Record operation response time"""
        self.response_times.append({
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })
        
        if not success:
            self.error_counts["total"] += 1
            self.error_counts["by_operation"][operation] = \
                self.error_counts["by_operation"].get(operation, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        # Response time statistics
        successful_responses = [r for r in self.response_times if r["success"]]
        response_durations = [r["duration"] for r in successful_responses]
        
        response_stats = {}
        if response_durations:
            response_stats = {
                "count": len(response_durations),
                "mean": statistics.mean(response_durations),
                "median": statistics.median(response_durations),
                "p95": statistics.quantiles(response_durations, n=20)[18] if len(response_durations) > 20 else max(response_durations, default=0),
                "p99": statistics.quantiles(response_durations, n=100)[98] if len(response_durations) > 100 else max(response_durations, default=0),
                "min": min(response_durations),
                "max": max(response_durations)
            }
        
        # Resource usage statistics
        cpu_stats = {}
        if self.cpu_samples:
            cpu_stats = {
                "mean": statistics.mean(self.cpu_samples),
                "max": max(self.cpu_samples),
                "p95": statistics.quantiles(self.cpu_samples, n=20)[18] if len(self.cpu_samples) > 20 else max(self.cpu_samples, default=0)
            }
        
        memory_stats = {}
        if self.memory_samples:
            memory_stats = {
                "mean": statistics.mean(self.memory_samples),
                "max": max(self.memory_samples),
                "p95": statistics.quantiles(self.memory_samples, n=20)[18] if len(self.memory_samples) > 20 else max(self.memory_samples, default=0)
            }
        
        return {
            "duration": total_duration,
            "operations_per_second": len(self.response_times) / total_duration if total_duration > 0 else 0,
            "success_rate": (len(successful_responses) / len(self.response_times)) if self.response_times else 0,
            "response_times": response_stats,
            "cpu_usage": cpu_stats,
            "memory_usage_mb": memory_stats,
            "error_counts": self.error_counts
        }


def timed_operation(operation_name: str, monitor: PerformanceMonitor):
    """Decorator to time operations and record performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                # Check if result indicates failure
                if isinstance(result, dict) and result.get("status") == "failed":
                    success = False
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                monitor.record_response_time(operation_name, duration, success)
        return wrapper
    return decorator


class TestConcurrentOperations:
    """Test concurrent operations and system scalability"""
    
    @pytest.mark.slow
    def test_concurrent_syndicate_creation_stress(self):
        """Stress test concurrent syndicate creation"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        @timed_operation("create_syndicate", monitor)
        def create_test_syndicate(syndicate_id):
            return create_syndicate(syndicate_id, f"Stress Test Syndicate {syndicate_id}",
                                  f"Stress test syndicate created at {datetime.now()}")
        
        try:
            # Test with increasing concurrency levels
            concurrency_levels = [5, 10, 25, 50, 100]
            results = {}
            
            for concurrency in concurrency_levels:
                print(f"Testing syndicate creation with {concurrency} concurrent operations...")
                
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    syndicate_ids = [f"stress-{concurrency}-{i}" for i in range(concurrency)]
                    futures = [executor.submit(create_test_syndicate, sid) for sid in syndicate_ids]
                    
                    concurrent_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            concurrent_results.append(result)
                        except Exception as e:
                            concurrent_results.append({"status": "failed", "error": str(e)})
                
                # Analyze results for this concurrency level
                successful = [r for r in concurrent_results if r.get("status") != "failed"]
                success_rate = len(successful) / len(concurrent_results)
                
                results[concurrency] = {
                    "success_rate": success_rate,
                    "total_operations": len(concurrent_results),
                    "successful_operations": len(successful)
                }
                
                # Assert minimum success rate
                assert success_rate >= 0.90, f"Success rate {success_rate:.2%} below threshold for {concurrency} concurrent operations"
        
        finally:
            monitor.stop_monitoring()
        
        # Analyze overall performance
        stats = monitor.get_statistics()
        
        # Performance assertions
        assert stats["success_rate"] >= 0.90, f"Overall success rate {stats['success_rate']:.2%} below threshold"
        assert stats["response_times"]["p95"] < 5.0, f"95th percentile response time {stats['response_times']['p95']:.2f}s above threshold"
        
        print(f"Stress test completed: {stats['operations_per_second']:.2f} ops/sec, {stats['success_rate']:.2%} success rate")

    @pytest.mark.slow
    def test_concurrent_member_operations_mixed_load(self):
        """Test mixed concurrent member operations"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup base syndicates
        base_syndicates = []
        for i in range(5):
            syndicate_id = f"mixed-load-{i}"
            result = create_syndicate(syndicate_id, f"Mixed Load Syndicate {i}")
            if result.get("status") != "failed":
                base_syndicates.append(syndicate_id)
        
        @timed_operation("add_member", monitor)
        def add_test_member(syndicate_id, member_idx):
            return add_member(syndicate_id, f"User {member_idx}", 
                            f"user{member_idx}@mixedload.test", 
                            "contributing_member", 10000.0 + (member_idx * 500))
        
        @timed_operation("get_status", monitor)
        def get_test_status(syndicate_id):
            return get_syndicate_status(syndicate_id)
        
        @timed_operation("update_contribution", monitor)
        def update_test_contribution(syndicate_id, member_id):
            from src.syndicate.syndicate_tools import update_member_contribution
            return update_member_contribution(syndicate_id, member_id, 5000.0)
        
        try:
            # Mixed workload simulation
            operations = []
            
            # Add members (60% of operations)
            for _ in range(120):
                syndicate_id = base_syndicates[_ % len(base_syndicates)]
                operations.append(("add_member", add_test_member, syndicate_id, _))
            
            # Status checks (30% of operations)  
            for _ in range(60):
                syndicate_id = base_syndicates[_ % len(base_syndicates)]
                operations.append(("get_status", get_test_status, syndicate_id))
            
            # Contribution updates (10% of operations)
            # Note: These will mostly fail since members might not exist yet
            for _ in range(20):
                syndicate_id = base_syndicates[_ % len(base_syndicates)]
                operations.append(("update_contribution", update_test_contribution, 
                                 syndicate_id, f"member_{_}"))
            
            # Execute mixed operations concurrently
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                for op in operations:
                    if len(op) == 4:  # add_member operation
                        _, func, syndicate_id, member_idx = op
                        futures.append(executor.submit(func, syndicate_id, member_idx))
                    elif len(op) == 3:  # other operations
                        _, func, *args = op
                        futures.append(executor.submit(func, *args))
                
                # Collect results
                results = []
                for future in as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({"status": "failed", "error": str(e)})
        
        finally:
            monitor.stop_monitoring()
        
        # Analyze performance
        stats = monitor.get_statistics()
        
        # Mixed workload should handle gracefully
        assert stats["success_rate"] >= 0.70, f"Mixed workload success rate {stats['success_rate']:.2%} below threshold"
        assert stats["response_times"]["mean"] < 2.0, f"Average response time {stats['response_times']['mean']:.2f}s above threshold"
        
        print(f"Mixed load test: {len(results)} operations, {stats['success_rate']:.2%} success rate")

    @pytest.mark.slow
    def test_fund_allocation_under_load(self):
        """Test fund allocation performance under concurrent load"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup test syndicate with members
        syndicate_id = "allocation-load-test"
        create_syndicate(syndicate_id, "Allocation Load Test")
        
        # Add multiple members
        for i in range(10):
            add_member(syndicate_id, f"Trader {i}", f"trader{i}@loadtest.com", 
                      "contributing_member", 25000.0)
        
        # Sample opportunities
        sample_opportunities = [
            {
                "sport": "NFL", "event": f"Load Test Game {i}", "bet_type": "spread",
                "selection": f"Team A -{i%10}", "odds": 1.85 + (i%10)*0.01,
                "probability": 0.52 + (i%20)*0.01, "edge": 0.02 + (i%15)*0.001,
                "confidence": 0.65 + (i%25)*0.01, "model_agreement": 0.70 + (i%30)*0.01,
                "hours_until_event": 12 + (i%36), "liquidity": 30000 + (i%10)*5000
            }
            for i in range(50)
        ]
        
        @timed_operation("allocate_funds", monitor)
        def allocate_test_funds(opportunities_subset):
            return allocate_funds(syndicate_id, opportunities_subset, "kelly_criterion")
        
        try:
            # Test allocation with different batch sizes
            batch_sizes = [1, 3, 5, 10]
            
            with ThreadPoolExecutor(max_workers=15) as executor:
                futures = []
                
                for batch_size in batch_sizes:
                    for i in range(0, len(sample_opportunities), batch_size):
                        batch = sample_opportunities[i:i+batch_size]
                        futures.append(executor.submit(allocate_test_funds, batch))
                
                # Collect results
                allocation_results = []
                for future in as_completed(futures, timeout=60):
                    try:
                        result = future.result()
                        allocation_results.append(result)
                    except Exception as e:
                        allocation_results.append({"status": "failed", "error": str(e)})
        
        finally:
            monitor.stop_monitoring()
        
        # Performance analysis
        stats = monitor.get_statistics()
        successful_allocations = [r for r in allocation_results if r.get("status") != "failed"]
        
        # Verify allocation consistency
        for result in successful_allocations:
            if "total_allocated" in result:
                assert result["total_allocated"] >= 0
                assert result["total_capital"] > 0
                # Should not over-allocate
                assert result["total_allocated"] <= result["total_capital"] * 0.25  # 25% max exposure
        
        # Performance thresholds
        assert stats["success_rate"] >= 0.85, f"Allocation success rate {stats['success_rate']:.2%} below threshold"
        assert stats["response_times"]["p95"] < 10.0, f"95th percentile allocation time {stats['response_times']['p95']:.2f}s above threshold"
        
        print(f"Fund allocation load test: {len(allocation_results)} allocations, {stats['success_rate']:.2%} success rate")

    @pytest.mark.slow
    def test_profit_distribution_concurrency(self):
        """Test profit distribution under concurrent access"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup syndicate with members
        syndicate_id = "distribution-test"
        create_syndicate(syndicate_id, "Distribution Test")
        
        # Add members with different contributions
        member_ids = []
        for i in range(20):
            result = add_member(syndicate_id, f"Investor {i}", f"investor{i}@disttest.com",
                              "contributing_member", 15000.0 + (i * 2000))
            if result.get("status") != "failed":
                member_ids.append(result["member_id"])
        
        @timed_operation("distribute_profits", monitor)
        def distribute_test_profits(profit_amount, model):
            return distribute_profits(syndicate_id, profit_amount, model)
        
        try:
            # Simulate multiple concurrent profit distributions
            distribution_scenarios = [
                (5000.0, "proportional"),
                (7500.0, "hybrid"),
                (3000.0, "performance_weighted"),
                (10000.0, "tiered"),
                (2500.0, "proportional")
            ]
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                # Submit multiple distribution requests
                for profit, model in distribution_scenarios:
                    futures.append(executor.submit(distribute_test_profits, profit, model))
                
                # Submit some duplicate scenarios to test concurrent access
                for _ in range(5):
                    profit, model = distribution_scenarios[_ % len(distribution_scenarios)]
                    futures.append(executor.submit(distribute_test_profits, profit, model))
                
                # Collect results
                distribution_results = []
                for future in as_completed(futures, timeout=30):
                    try:
                        result = future.result()
                        distribution_results.append(result)
                    except Exception as e:
                        distribution_results.append({"status": "failed", "error": str(e)})
        
        finally:
            monitor.stop_monitoring()
        
        # Analyze distribution results
        stats = monitor.get_statistics()
        successful_distributions = [r for r in distribution_results 
                                  if r.get("execution_status") != "failed" and r.get("status") != "failed"]
        
        # Verify distribution integrity
        for result in successful_distributions:
            if "total_profit" in result and "distributions" in result:
                total_distributed = sum(d.get("gross_amount", 0) for d in result["distributions"])
                profit = result["total_profit"]
                # Allow small rounding differences
                assert abs(total_distributed - profit) < 0.01, f"Distribution mismatch: {total_distributed} vs {profit}"
        
        # Performance assertions
        assert len(successful_distributions) >= len(distribution_scenarios), "Some distributions should succeed"
        assert stats["response_times"]["mean"] < 5.0, f"Average distribution time {stats['response_times']['mean']:.2f}s above threshold"
        
        print(f"Profit distribution concurrency test: {len(distribution_results)} distributions attempted")

    @pytest.mark.slow
    def test_system_stability_under_sustained_load(self):
        """Test system stability under sustained concurrent load"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup multiple syndicates for load testing
        test_syndicates = []
        for i in range(3):
            syndicate_id = f"stability-{i}"
            result = create_syndicate(syndicate_id, f"Stability Test {i}")
            if result.get("status") != "failed":
                test_syndicates.append(syndicate_id)
                
                # Add members to each syndicate
                for j in range(5):
                    add_member(syndicate_id, f"User {i}-{j}", f"user{i}_{j}@stability.test",
                              "contributing_member", 20000.0 + (j * 3000))
        
        # Define sustained load operations
        def sustained_operation_worker(worker_id: int, duration_seconds: int, operation_queue: queue.Queue):
            """Worker function for sustained load testing"""
            start_time = time.time()
            operations_completed = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Rotate through different operations
                    op_type = operations_completed % 4
                    syndicate_id = test_syndicates[operations_completed % len(test_syndicates)]
                    
                    if op_type == 0:  # Status check
                        result = get_syndicate_status(syndicate_id)
                    elif op_type == 1:  # Member list
                        from src.syndicate.syndicate_tools import get_member_list
                        result = get_member_list(syndicate_id, active_only=True)
                    elif op_type == 2:  # Allocation limits
                        from src.syndicate.syndicate_tools import get_allocation_limits
                        result = get_allocation_limits(syndicate_id)
                    else:  # Simple allocation
                        simple_opp = [{
                            "sport": "NBA", "event": f"Stability Game {operations_completed}",
                            "bet_type": "total", "selection": "Over 210", "odds": 1.90,
                            "probability": 0.52, "edge": 0.02, "confidence": 0.65,
                            "model_agreement": 0.70, "hours_until_event": 18, "liquidity": 40000
                        }]
                        result = allocate_funds(syndicate_id, simple_opp, "fixed_percentage")
                    
                    # Record result
                    success = not (isinstance(result, dict) and result.get("status") == "failed")
                    operation_queue.put({"worker_id": worker_id, "success": success, "timestamp": time.time()})
                    operations_completed += 1
                    
                    # Brief pause to simulate realistic usage
                    time.sleep(0.01)  # 10ms pause
                    
                except Exception as e:
                    operation_queue.put({"worker_id": worker_id, "success": False, "error": str(e), "timestamp": time.time()})
                
            operation_queue.put({"worker_id": worker_id, "completed": True, "operations_completed": operations_completed})
        
        try:
            # Run sustained load test
            load_duration = 60  # 60 seconds of sustained load
            num_workers = 8
            result_queue = queue.Queue()
            
            # Start worker threads
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(sustained_operation_worker, i, load_duration, result_queue)
                    for i in range(num_workers)
                ]
                
                # Monitor progress
                completed_workers = 0
                total_operations = 0
                successful_operations = 0
                start_time = time.time()
                
                while completed_workers < num_workers:
                    try:
                        result = result_queue.get(timeout=1)
                        
                        if result.get("completed"):
                            completed_workers += 1
                            total_operations += result.get("operations_completed", 0)
                        elif result.get("success") is not None:
                            if result["success"]:
                                successful_operations += 1
                    except queue.Empty:
                        continue
                
                # Wait for all workers to complete
                for future in as_completed(futures, timeout=load_duration + 10):
                    future.result()
            
            actual_duration = time.time() - start_time
            
        finally:
            monitor.stop_monitoring()
        
        # Analyze sustained load performance
        stats = monitor.get_statistics()
        
        # Calculate throughput and success rate
        throughput = total_operations / actual_duration if actual_duration > 0 else 0
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        
        # Stability assertions
        assert success_rate >= 0.80, f"Success rate under sustained load {success_rate:.2%} below threshold"
        assert throughput >= 10.0, f"Throughput {throughput:.2f} ops/sec below minimum threshold"
        assert stats["memory_usage_mb"]["max"] < 1000, f"Memory usage {stats['memory_usage_mb']['max']:.1f}MB above threshold"
        
        # System should remain stable (no significant performance degradation)
        response_times = [r["duration"] for r in monitor.response_times if r["success"]]
        if len(response_times) > 100:
            # Check if response times remain stable (no significant trend upward)
            early_times = response_times[:len(response_times)//4]
            late_times = response_times[-len(response_times)//4:]
            
            early_avg = statistics.mean(early_times)
            late_avg = statistics.mean(late_times)
            
            # Response times should not degrade significantly
            degradation_ratio = late_avg / early_avg if early_avg > 0 else 1
            assert degradation_ratio < 2.0, f"Response time degradation {degradation_ratio:.2f}x indicates instability"
        
        print(f"Sustained load test: {throughput:.2f} ops/sec, {success_rate:.2%} success rate over {actual_duration:.1f}s")

    @pytest.mark.slow
    def test_memory_leak_detection(self):
        """Test for memory leaks under repeated operations"""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Setup test syndicate
        syndicate_id = "memory-test"
        create_syndicate(syndicate_id, "Memory Leak Test")
        
        try:
            # Perform many repeated operations that could cause memory leaks
            operations_per_cycle = 100
            num_cycles = 5
            
            memory_readings = []
            
            for cycle in range(num_cycles):
                cycle_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Perform operations that create and destroy objects
                for i in range(operations_per_cycle):
                    # Add and remove members
                    member_result = add_member(syndicate_id, f"Temp User {cycle}-{i}", 
                                             f"temp{cycle}_{i}@memtest.com", "observer", 1000.0)
                    
                    # Status checks
                    get_syndicate_status(syndicate_id)
                    
                    # Allocation operations
                    temp_opp = [{
                        "sport": "NFL", "event": f"Memory Test {cycle}-{i}",
                        "bet_type": "spread", "selection": "Team -3", "odds": 1.90,
                        "probability": 0.53, "edge": 0.02, "confidence": 0.65,
                        "model_agreement": 0.75, "hours_until_event": 24, "liquidity": 50000
                    }]
                    allocate_funds(syndicate_id, temp_opp, "kelly_criterion")
                
                cycle_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(cycle_end_memory)
                
                print(f"Memory after cycle {cycle + 1}: {cycle_end_memory:.1f}MB")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Brief pause between cycles
                time.sleep(1)
        
        finally:
            monitor.stop_monitoring()
        
        # Analyze memory usage pattern
        if len(memory_readings) >= 3:
            # Check for consistent memory growth (potential leak)
            memory_growth = memory_readings[-1] - memory_readings[0]
            avg_cycle_growth = memory_growth / (len(memory_readings) - 1)
            
            # Memory growth should be minimal after initial allocation
            assert memory_growth < 100, f"Total memory growth {memory_growth:.1f}MB suggests memory leak"
            assert avg_cycle_growth < 20, f"Average cycle growth {avg_cycle_growth:.1f}MB/cycle suggests memory leak"
            
            # Check for linear growth pattern (strong indicator of leak)
            if len(memory_readings) >= 5:
                # Simple linear regression to detect trend
                x_values = list(range(len(memory_readings)))
                y_values = memory_readings
                
                n = len(x_values)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values))
                sum_x2 = sum(x * x for x in x_values)
                
                # Calculate slope
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Slope should be minimal (< 5MB per cycle)
                assert abs(slope) < 5, f"Memory usage slope {slope:.2f}MB/cycle indicates potential leak"
        
        stats = monitor.get_statistics()
        final_memory = stats["memory_usage_mb"]["max"]
        
        print(f"Memory leak test completed. Peak memory: {final_memory:.1f}MB")

    def test_deadlock_prevention(self):
        """Test system behavior under conditions that could cause deadlocks"""
        # Setup multiple syndicates
        syndicate_ids = []
        for i in range(3):
            syndicate_id = f"deadlock-test-{i}"
            result = create_syndicate(syndicate_id, f"Deadlock Test {i}")
            if result.get("status") != "failed":
                syndicate_ids.append(syndicate_id)
                add_member(syndicate_id, f"User {i}", f"user{i}@deadlock.test", 
                          "contributing_member", 30000.0)
        
        def cross_syndicate_operations(worker_id: int, results_list: List):
            """Operations that could potentially cause deadlocks"""
            try:
                for i in range(10):
                    # Operations that access multiple syndicates
                    for syndicate_id in syndicate_ids:
                        status = get_syndicate_status(syndicate_id)
                        
                        # Simulate operations that might compete for resources
                        opp = [{
                            "sport": "MLB", "event": f"Deadlock Test {worker_id}-{i}",
                            "bet_type": "moneyline", "selection": "Home Team", "odds": 1.95,
                            "probability": 0.54, "edge": 0.03, "confidence": 0.68,
                            "model_agreement": 0.72, "hours_until_event": 18, "liquidity": 35000
                        }]
                        allocate_funds(syndicate_id, opp, "kelly_criterion")
                
                results_list.append({"worker_id": worker_id, "success": True})
                
            except Exception as e:
                results_list.append({"worker_id": worker_id, "success": False, "error": str(e)})
        
        # Run concurrent cross-syndicate operations
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(cross_syndicate_operations, i, results)
                for i in range(6)
            ]
            
            # All operations should complete within reasonable time
            for future in as_completed(futures, timeout=30):
                future.result()
        
        completion_time = time.time() - start_time
        
        # Verify no deadlocks occurred (all operations completed)
        assert len(results) == 6, f"Only {len(results)}/6 workers completed - possible deadlock"
        assert completion_time < 25, f"Operations took {completion_time:.1f}s - possible deadlock/contention"
        
        # At least some operations should succeed
        successful_workers = [r for r in results if r["success"]]
        assert len(successful_workers) >= 4, f"Only {len(successful_workers)}/6 workers succeeded"
        
        print(f"Deadlock prevention test completed in {completion_time:.1f}s, {len(successful_workers)}/6 workers successful")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        "-v",
        "-s",  # Don't capture output so we can see progress
        "--tb=short",
        "-m", "slow",  # Run only slow/performance tests
        __file__
    ])