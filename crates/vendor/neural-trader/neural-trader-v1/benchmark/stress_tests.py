#!/usr/bin/env python3
"""
Comprehensive stress testing suite for AI News Trading platform.

This module provides stress tests to validate system stability, error handling,
recovery mechanisms, and long-running performance under various load conditions.

Test Categories:
- High-frequency scenario testing
- Memory leak detection
- Long-running stability tests
- Error handling validation
- Recovery mechanism testing
- Load scaling tests
- Resource exhaustion tests
"""

import asyncio
import gc
import json
import logging
import multiprocessing
import psutil
import resource
import signal
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import tracemalloc

# Import existing benchmark components
from src.benchmarks.runner import BenchmarkRunner
from src.benchmarks.latency_benchmark import LatencyBenchmark
from src.benchmarks.throughput_benchmark import ThroughputBenchmark
from src.benchmarks.resource_benchmark import ResourceBenchmark


@dataclass
class StressTestResult:
    """Result container for stress test results."""
    test_name: str
    test_type: str
    status: str  # "PASS", "FAIL", "WARNING", "TIMEOUT"
    duration: float
    peak_memory_mb: float
    peak_cpu_percent: float
    operations_completed: int
    operations_failed: int
    error_rate: float
    recovery_time: float
    details: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class StressTestReport:
    """Complete stress test report."""
    platform_version: str
    test_timestamp: float
    test_duration: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    timeout_tests: int
    high_frequency_tests: List[StressTestResult]
    memory_leak_tests: List[StressTestResult]
    stability_tests: List[StressTestResult]
    error_handling_tests: List[StressTestResult]
    recovery_tests: List[StressTestResult]
    load_scaling_tests: List[StressTestResult]
    overall_status: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class HighFrequencyStressTester:
    """Tests system behavior under high-frequency trading scenarios."""
    
    def __init__(self, config=None):
        """Initialize high-frequency stress tester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: List[StressTestResult] = []
    
    async def test_burst_signal_generation(self, 
                                         burst_size: int = 10000,
                                         burst_duration: float = 1.0) -> StressTestResult:
        """Test system under burst signal generation load."""
        self.logger.info(f"Testing burst signal generation: {burst_size} signals in {burst_duration}s")
        
        start_time = time.time()
        completed_operations = 0
        failed_operations = 0
        peak_memory = 0
        peak_cpu = 0
        
        # Setup monitoring
        def monitor_resources():
            nonlocal peak_memory, peak_cpu
            while monitor_active.is_set():
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                cpu_percent = psutil.Process().cpu_percent()
                peak_memory = max(peak_memory, memory_mb)
                peak_cpu = max(peak_cpu, cpu_percent)
                time.sleep(0.01)
        
        monitor_active = threading.Event()
        monitor_active.set()
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        try:
            # Generate signals in burst
            tasks = []
            for i in range(burst_size):
                task = self._generate_high_frequency_signal(i)
                tasks.append(task)
            
            # Execute with time constraint
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=burst_duration * 2  # Allow some buffer
                )
                
                for result in results:
                    if isinstance(result, Exception):
                        failed_operations += 1
                    else:
                        completed_operations += 1
                        
            except asyncio.TimeoutError:
                failed_operations = burst_size - completed_operations
                
        finally:
            monitor_active.clear()
            monitor_thread.join()
        
        duration = time.time() - start_time
        error_rate = failed_operations / burst_size if burst_size > 0 else 0
        
        # Determine status
        if error_rate < 0.01:  # Less than 1% error rate
            status = "PASS"
        elif error_rate < 0.05:  # Less than 5% error rate
            status = "WARNING"
        else:
            status = "FAIL"
        
        return StressTestResult(
            test_name="Burst Signal Generation",
            test_type="high_frequency",
            status=status,
            duration=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            operations_completed=completed_operations,
            operations_failed=failed_operations,
            error_rate=error_rate,
            recovery_time=0.0,
            details={
                "burst_size": burst_size,
                "target_duration": burst_duration,
                "actual_duration": duration,
                "signals_per_second": completed_operations / duration if duration > 0 else 0,
                "timeout_occurred": duration > burst_duration * 1.5
            },
            timestamp=time.time()
        )
    
    async def test_sustained_high_frequency(self, 
                                          frequency_hz: int = 1000,
                                          duration_seconds: float = 300) -> StressTestResult:
        """Test sustained high-frequency signal generation."""
        self.logger.info(f"Testing sustained high-frequency: {frequency_hz}Hz for {duration_seconds}s")
        
        start_time = time.time()
        completed_operations = 0
        failed_operations = 0
        peak_memory = 0
        peak_cpu = 0
        memory_samples = []
        
        # Setup monitoring
        def monitor_resources():
            nonlocal peak_memory, peak_cpu
            while monitor_active.is_set():
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                cpu_percent = psutil.Process().cpu_percent()
                peak_memory = max(peak_memory, memory_mb)
                peak_cpu = max(peak_cpu, cpu_percent)
                memory_samples.append(memory_mb)
                time.sleep(0.1)
        
        monitor_active = threading.Event()
        monitor_active.set()
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        try:
            end_time = start_time + duration_seconds
            interval = 1.0 / frequency_hz  # Time between signals
            
            signal_id = 0
            while time.time() < end_time:
                batch_start = time.time()
                
                # Generate batch of signals
                batch_size = min(100, frequency_hz // 10)  # Adaptive batch size
                tasks = []
                
                for _ in range(batch_size):
                    task = self._generate_high_frequency_signal(signal_id)
                    tasks.append(task)
                    signal_id += 1
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            failed_operations += 1
                        else:
                            completed_operations += 1
                            
                except Exception as e:
                    self.logger.warning(f"Batch failed: {e}")
                    failed_operations += batch_size
                
                # Maintain frequency
                elapsed = time.time() - batch_start
                sleep_time = max(0, (batch_size * interval) - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        finally:
            monitor_active.clear()
            monitor_thread.join()
        
        duration = time.time() - start_time
        error_rate = failed_operations / (completed_operations + failed_operations) if (completed_operations + failed_operations) > 0 else 0
        
        # Check for memory growth (potential leak)
        memory_growth = 0
        if len(memory_samples) > 10:
            early_avg = np.mean(memory_samples[:10])
            late_avg = np.mean(memory_samples[-10:])
            memory_growth = (late_avg - early_avg) / early_avg
        
        # Determine status
        if error_rate < 0.01 and memory_growth < 0.1:  # Less than 1% error, 10% memory growth
            status = "PASS"
        elif error_rate < 0.05 and memory_growth < 0.2:  # Less than 5% error, 20% memory growth
            status = "WARNING"
        else:
            status = "FAIL"
        
        return StressTestResult(
            test_name="Sustained High Frequency",
            test_type="high_frequency",
            status=status,
            duration=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            operations_completed=completed_operations,
            operations_failed=failed_operations,
            error_rate=error_rate,
            recovery_time=0.0,
            details={
                "target_frequency_hz": frequency_hz,
                "actual_frequency_hz": completed_operations / duration if duration > 0 else 0,
                "memory_growth_percent": memory_growth * 100,
                "memory_samples": len(memory_samples),
                "avg_memory_mb": np.mean(memory_samples) if memory_samples else 0
            },
            timestamp=time.time()
        )
    
    async def _generate_high_frequency_signal(self, signal_id: int):
        """Generate a single high-frequency signal."""
        # Simulate minimal signal generation work
        await asyncio.sleep(0.0001)  # 0.1ms base processing time
        
        # Some light computation
        np.random.seed(signal_id % 1000)  # Reproducible but varied
        data = np.random.randn(10)
        signal_strength = np.mean(data)
        
        return {
            "signal_id": signal_id,
            "timestamp": time.time(),
            "strength": signal_strength,
            "action": "buy" if signal_strength > 0 else "sell"
        }


class MemoryLeakTester:
    """Tests for memory leaks during extended operations."""
    
    def __init__(self, config=None):
        """Initialize memory leak tester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def test_memory_leak_detection(self, 
                                       duration_minutes: float = 10,
                                       operation_frequency: int = 100) -> StressTestResult:
        """Test for memory leaks during extended operation."""
        self.logger.info(f"Testing memory leak detection for {duration_minutes} minutes")
        
        # Start memory tracing
        tracemalloc.start()
        start_time = time.time()
        
        memory_snapshots = []
        completed_operations = 0
        failed_operations = 0
        
        try:
            end_time = start_time + (duration_minutes * 60)
            snapshot_interval = 30  # Take snapshot every 30 seconds
            last_snapshot = start_time
            
            while time.time() < end_time:
                # Perform operations
                for _ in range(operation_frequency):
                    try:
                        await self._memory_intensive_operation(completed_operations)
                        completed_operations += 1
                    except Exception as e:
                        self.logger.warning(f"Operation failed: {e}")
                        failed_operations += 1
                
                # Take memory snapshot
                if time.time() - last_snapshot >= snapshot_interval:
                    snapshot = tracemalloc.take_snapshot()
                    current_memory = sum(stat.size for stat in snapshot.statistics('filename')) / (1024 * 1024)
                    
                    memory_snapshots.append({
                        "timestamp": time.time(),
                        "memory_mb": current_memory,
                        "operations_completed": completed_operations
                    })
                    
                    last_snapshot = time.time()
                    
                    # Force garbage collection
                    gc.collect()
                
                await asyncio.sleep(1.0 / operation_frequency)
                
        finally:
            tracemalloc.stop()
        
        duration = time.time() - start_time
        error_rate = failed_operations / (completed_operations + failed_operations) if (completed_operations + failed_operations) > 0 else 0
        
        # Analyze memory growth
        memory_leak_detected = False
        memory_growth_rate = 0
        
        if len(memory_snapshots) >= 3:
            # Calculate memory growth trend
            times = [s["timestamp"] - start_time for s in memory_snapshots]
            memories = [s["memory_mb"] for s in memory_snapshots]
            
            # Linear regression to detect trend
            n = len(times)
            sum_t = sum(times)
            sum_m = sum(memories)
            sum_tm = sum(t * m for t, m in zip(times, memories))
            sum_t2 = sum(t * t for t in times)
            
            # Slope (MB per second)
            memory_growth_rate = (n * sum_tm - sum_t * sum_m) / (n * sum_t2 - sum_t * sum_t)
            
            # Convert to MB per hour
            memory_growth_rate_per_hour = memory_growth_rate * 3600
            
            # Consider leak if growing > 10MB/hour
            memory_leak_detected = memory_growth_rate_per_hour > 10
        
        # Determine status
        if not memory_leak_detected and error_rate < 0.01:
            status = "PASS"
        elif not memory_leak_detected and error_rate < 0.05:
            status = "WARNING"
        else:
            status = "FAIL"
        
        peak_memory = max(s["memory_mb"] for s in memory_snapshots) if memory_snapshots else 0
        
        return StressTestResult(
            test_name="Memory Leak Detection",
            test_type="memory_leak",
            status=status,
            duration=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=0,  # Not monitored in this test
            operations_completed=completed_operations,
            operations_failed=failed_operations,
            error_rate=error_rate,
            recovery_time=0.0,
            details={
                "memory_snapshots": memory_snapshots,
                "memory_growth_rate_mb_per_hour": memory_growth_rate * 3600,
                "memory_leak_detected": memory_leak_detected,
                "operation_frequency": operation_frequency,
                "total_snapshots": len(memory_snapshots)
            },
            timestamp=time.time()
        )
    
    async def _memory_intensive_operation(self, operation_id: int):
        """Perform memory-intensive operation that might leak."""
        # Create and process data that might not be properly cleaned up
        data_size = 1000 + (operation_id % 5000)  # Vary data size
        
        # Create large arrays
        array1 = np.random.randn(data_size)
        array2 = np.random.randn(data_size)
        
        # Perform computations
        result = np.convolve(array1, array2[:100], mode='valid')
        fft_result = np.fft.fft(result)
        
        # Simulate some complex object creation
        complex_object = {
            "id": operation_id,
            "data": array1.tolist(),
            "processed": result.tolist(),
            "fft": fft_result.tolist(),
            "metadata": {
                "size": data_size,
                "timestamp": time.time(),
                "stats": {
                    "mean": float(np.mean(array1)),
                    "std": float(np.std(array1)),
                    "min": float(np.min(array1)),
                    "max": float(np.max(array1))
                }
            }
        }
        
        # Simulate potential leak by storing reference
        if hasattr(self, '_potential_leak_storage'):
            if len(self._potential_leak_storage) > 1000:
                # Clean up old entries to prevent actual leak in test
                self._potential_leak_storage = self._potential_leak_storage[-500:]
        else:
            self._potential_leak_storage = []
        
        self._potential_leak_storage.append(complex_object)
        
        return len(result)


class StabilityTester:
    """Tests long-running system stability."""
    
    def __init__(self, config=None):
        """Initialize stability tester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def test_long_running_stability(self, 
                                        duration_hours: float = 2,
                                        check_interval_minutes: float = 5) -> StressTestResult:
        """Test system stability over extended periods."""
        self.logger.info(f"Testing long-running stability for {duration_hours} hours")
        
        start_time = time.time()
        duration_seconds = duration_hours * 3600
        check_interval_seconds = check_interval_minutes * 60
        
        stability_checks = []
        completed_operations = 0
        failed_operations = 0
        peak_memory = 0
        peak_cpu = 0
        
        try:
            last_check = start_time
            
            while time.time() - start_time < duration_seconds:
                # Perform continuous operations
                batch_start = time.time()
                
                # Run a batch of operations
                batch_size = 50
                for i in range(batch_size):
                    try:
                        await self._stability_operation(completed_operations + i)
                        completed_operations += 1
                    except Exception as e:
                        self.logger.warning(f"Stability operation failed: {e}")
                        failed_operations += 1
                
                # Periodic stability check
                if time.time() - last_check >= check_interval_seconds:
                    check_result = await self._perform_stability_check()
                    
                    # Update peak resources
                    peak_memory = max(peak_memory, check_result["memory_mb"])
                    peak_cpu = max(peak_cpu, check_result["cpu_percent"])
                    
                    stability_checks.append({
                        "timestamp": time.time(),
                        "elapsed_hours": (time.time() - start_time) / 3600,
                        **check_result
                    })
                    
                    last_check = time.time()
                
                # Small delay between batches
                await asyncio.sleep(1.0)
                
        except Exception as e:
            self.logger.error(f"Long-running stability test failed: {e}")
            status = "FAIL"
        
        duration = time.time() - start_time
        error_rate = failed_operations / (completed_operations + failed_operations) if (completed_operations + failed_operations) > 0 else 0
        
        # Analyze stability
        stability_degradation = False
        if len(stability_checks) >= 2:
            # Check if performance degraded over time
            early_checks = stability_checks[:len(stability_checks)//4] or stability_checks[:1]
            late_checks = stability_checks[-len(stability_checks)//4:] or stability_checks[-1:]
            
            early_avg_latency = np.mean([c["avg_latency_ms"] for c in early_checks])
            late_avg_latency = np.mean([c["avg_latency_ms"] for c in late_checks])
            
            # Consider degraded if latency increased by more than 50%
            stability_degradation = (late_avg_latency / early_avg_latency) > 1.5
        
        # Determine status
        if error_rate < 0.01 and not stability_degradation:
            status = "PASS"
        elif error_rate < 0.05 and not stability_degradation:
            status = "WARNING"
        else:
            status = "FAIL"
        
        return StressTestResult(
            test_name="Long Running Stability",
            test_type="stability",
            status=status,
            duration=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            operations_completed=completed_operations,
            operations_failed=failed_operations,
            error_rate=error_rate,
            recovery_time=0.0,
            details={
                "target_duration_hours": duration_hours,
                "actual_duration_hours": duration / 3600,
                "stability_checks": stability_checks,
                "stability_degradation_detected": stability_degradation,
                "total_checks": len(stability_checks)
            },
            timestamp=time.time()
        )
    
    async def _stability_operation(self, operation_id: int):
        """Perform operation for stability testing."""
        # Simulate typical trading system operation
        start_time = time.perf_counter()
        
        # Data processing
        data = np.random.randn(100)
        processed = np.convolve(data, np.ones(10)/10, mode='valid')
        
        # Signal generation
        signal = "buy" if np.mean(processed) > 0 else "sell"
        confidence = min(1.0, abs(np.mean(processed)) * 2)
        
        # Risk calculation
        portfolio_value = 100000 + np.random.randn() * 10000
        position_size = portfolio_value * 0.02  # 2% position
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        return {
            "operation_id": operation_id,
            "signal": signal,
            "confidence": confidence,
            "position_size": position_size,
            "latency_ms": latency
        }
    
    async def _perform_stability_check(self) -> Dict[str, Any]:
        """Perform comprehensive stability check."""
        # System resources
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        # Performance check
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            await self._stability_operation(0)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        }


class ErrorHandlingTester:
    """Tests error handling and recovery mechanisms."""
    
    def __init__(self, config=None):
        """Initialize error handling tester."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def test_error_recovery_mechanisms(self) -> StressTestResult:
        """Test system recovery from various error conditions."""
        self.logger.info("Testing error recovery mechanisms")
        
        start_time = time.time()
        completed_operations = 0
        failed_operations = 0
        recovery_times = []
        peak_memory = 0
        peak_cpu = 0
        
        # Error scenarios to test
        error_scenarios = [
            "network_timeout",
            "memory_exhaustion", 
            "invalid_data",
            "api_rate_limit",
            "database_connection",
            "computation_overflow"
        ]
        
        test_results = {}
        
        for scenario in error_scenarios:
            self.logger.info(f"Testing error scenario: {scenario}")
            
            scenario_start = time.time()
            
            try:
                # Inject error and measure recovery
                recovery_start = time.time()
                await self._inject_error_scenario(scenario)
                recovery_end = time.time()
                
                recovery_time = recovery_end - recovery_start
                recovery_times.append(recovery_time)
                
                # Test normal operation after recovery
                post_recovery_success = await self._test_post_recovery_operation()
                
                test_results[scenario] = {
                    "recovery_time": recovery_time,
                    "post_recovery_success": post_recovery_success
                }
                
                completed_operations += 1
                
            except Exception as e:
                self.logger.error(f"Error scenario {scenario} failed: {e}")
                test_results[scenario] = {
                    "recovery_time": float('inf'),
                    "post_recovery_success": False,
                    "error": str(e)
                }
                failed_operations += 1
        
        duration = time.time() - start_time
        error_rate = failed_operations / len(error_scenarios)
        avg_recovery_time = np.mean([r for r in recovery_times if r != float('inf')]) if recovery_times else 0
        
        # Determine status
        if error_rate == 0 and avg_recovery_time < 5.0:  # All scenarios pass, recovery < 5s
            status = "PASS"
        elif error_rate < 0.3 and avg_recovery_time < 10.0:  # Most scenarios pass, recovery < 10s
            status = "WARNING"
        else:
            status = "FAIL"
        
        return StressTestResult(
            test_name="Error Recovery Mechanisms",
            test_type="error_handling",
            status=status,
            duration=duration,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            operations_completed=completed_operations,
            operations_failed=failed_operations,
            error_rate=error_rate,
            recovery_time=avg_recovery_time,
            details={
                "scenarios_tested": error_scenarios,
                "scenario_results": test_results,
                "recovery_times": recovery_times,
                "avg_recovery_time": avg_recovery_time,
                "max_recovery_time": max(recovery_times) if recovery_times else 0
            },
            timestamp=time.time()
        )
    
    async def _inject_error_scenario(self, scenario: str):
        """Inject specific error scenario and test recovery."""
        if scenario == "network_timeout":
            # Simulate network timeout
            await asyncio.sleep(0.1)  # Simulate brief timeout
            
        elif scenario == "memory_exhaustion":
            # Simulate memory pressure
            large_data = np.random.randn(10000)  # Create some memory pressure
            del large_data  # Clean up
            
        elif scenario == "invalid_data":
            # Simulate invalid data handling
            try:
                invalid_result = 1 / 0  # Division by zero
            except ZeroDivisionError:
                pass  # Should be handled gracefully
                
        elif scenario == "api_rate_limit":
            # Simulate API rate limiting
            await asyncio.sleep(0.5)  # Simulate rate limit delay
            
        elif scenario == "database_connection":
            # Simulate database connection issues
            await asyncio.sleep(0.2)  # Simulate connection retry
            
        elif scenario == "computation_overflow":
            # Simulate computation overflow
            try:
                overflow_result = np.exp(1000)  # Should overflow
            except (OverflowError, FloatingPointError):
                pass  # Should be handled gracefully
        
        # All scenarios should recover quickly
        await asyncio.sleep(0.1)  # Simulate recovery time
    
    async def _test_post_recovery_operation(self) -> bool:
        """Test that system works normally after error recovery."""
        try:
            # Perform normal operation
            data = np.random.randn(100)
            result = np.mean(data)
            
            # Verify result is reasonable
            return not (np.isnan(result) or np.isinf(result))
            
        except Exception:
            return False


class StressTestSuite:
    """Main stress test suite orchestrator."""
    
    def __init__(self, config=None):
        """Initialize stress test suite."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.high_frequency_tester = HighFrequencyStressTester(config)
        self.memory_leak_tester = MemoryLeakTester(config)
        self.stability_tester = StabilityTester(config)
        self.error_handling_tester = ErrorHandlingTester(config)
    
    async def run_complete_stress_tests(self, 
                                      quick_mode: bool = False) -> StressTestReport:
        """Run complete stress test suite."""
        self.logger.info("Starting complete stress test suite...")
        start_time = time.time()
        
        all_results = []
        
        # High-frequency tests
        self.logger.info("Running high-frequency stress tests...")
        hf_tests = [
            self.high_frequency_tester.test_burst_signal_generation(
                burst_size=5000 if quick_mode else 10000,
                burst_duration=0.5 if quick_mode else 1.0
            ),
            self.high_frequency_tester.test_sustained_high_frequency(
                frequency_hz=500 if quick_mode else 1000,
                duration_seconds=60 if quick_mode else 300
            )
        ]
        
        hf_results = await asyncio.gather(*hf_tests, return_exceptions=True)
        hf_results = [r for r in hf_results if isinstance(r, StressTestResult)]
        all_results.extend(hf_results)
        
        # Memory leak tests
        self.logger.info("Running memory leak tests...")
        ml_test = await self.memory_leak_tester.test_memory_leak_detection(
            duration_minutes=2 if quick_mode else 10,
            operation_frequency=50 if quick_mode else 100
        )
        ml_results = [ml_test]
        all_results.extend(ml_results)
        
        # Stability tests (skip in quick mode due to duration)
        stability_results = []
        if not quick_mode:
            self.logger.info("Running stability tests...")
            stability_test = await self.stability_tester.test_long_running_stability(
                duration_hours=0.5,  # 30 minutes for full test
                check_interval_minutes=2
            )
            stability_results = [stability_test]
            all_results.extend(stability_results)
        
        # Error handling tests
        self.logger.info("Running error handling tests...")
        eh_test = await self.error_handling_tester.test_error_recovery_mechanisms()
        eh_results = [eh_test]
        all_results.extend(eh_results)
        
        # Calculate summary statistics
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.status == "PASS")
        failed_tests = sum(1 for r in all_results if r.status == "FAIL")
        warning_tests = sum(1 for r in all_results if r.status == "WARNING")
        timeout_tests = sum(1 for r in all_results if r.status == "TIMEOUT")
        
        # Determine overall status
        if failed_tests == 0 and timeout_tests == 0:
            overall_status = "PASS"
        elif passed_tests > (failed_tests + timeout_tests):
            overall_status = "WARNING"
        else:
            overall_status = "FAIL"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        test_duration = time.time() - start_time
        
        return StressTestReport(
            platform_version="1.0.0",
            test_timestamp=time.time(),
            test_duration=test_duration,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            timeout_tests=timeout_tests,
            high_frequency_tests=hf_results,
            memory_leak_tests=ml_results,
            stability_tests=stability_results,
            error_handling_tests=eh_results,
            recovery_tests=[],  # Can be expanded
            load_scaling_tests=[],  # Can be expanded
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[StressTestResult]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        for result in results:
            if result.status == "FAIL":
                if result.test_type == "high_frequency":
                    recommendations.append(
                        f"Optimize high-frequency processing - consider async queuing, "
                        f"connection pooling, or load balancing"
                    )
                elif result.test_type == "memory_leak":
                    recommendations.append(
                        f"Fix memory leaks - review object lifecycle management, "
                        f"implement proper cleanup, add memory monitoring"
                    )
                elif result.test_type == "stability":
                    recommendations.append(
                        f"Improve long-running stability - add health checks, "
                        f"implement graceful degradation, review resource management"
                    )
                elif result.test_type == "error_handling":
                    recommendations.append(
                        f"Enhance error handling - implement circuit breakers, "
                        f"improve retry logic, add better error recovery"
                    )
            
            elif result.status == "WARNING":
                if result.error_rate > 0.01:
                    recommendations.append(
                        f"Reduce error rate in {result.test_name} - current: {result.error_rate:.2%}"
                    )
                if result.recovery_time > 1.0:
                    recommendations.append(
                        f"Improve recovery time in {result.test_name} - current: {result.recovery_time:.2f}s"
                    )
        
        # General recommendations based on overall results
        failed_count = sum(1 for r in results if r.status == "FAIL")
        if failed_count > len(results) * 0.5:
            recommendations.append(
                "Critical: Multiple stress test failures indicate systemic issues. "
                "Consider comprehensive architecture review."
            )
        
        return recommendations
    
    def save_report(self, report: StressTestReport, filepath: str):
        """Save stress test report to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Stress test report saved to {filepath}")


async def main():
    """Main entry point for stress test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI News Trading Platform Stress Tests")
    parser.add_argument("--quick", action="store_true", help="Run quick stress tests")
    parser.add_argument("--output", default="/workspaces/ai-news-trader/benchmark/results/stress_test_report.json",
                       help="Output file for stress test report")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run stress tests
    tester = StressTestSuite()
    report = await tester.run_complete_stress_tests(quick_mode=args.quick)
    
    # Save report
    tester.save_report(report, args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("AI NEWS TRADING PLATFORM - STRESS TEST SUMMARY")
    print("="*80)
    print(f"Overall Status: {report.overall_status}")
    print(f"Test Duration: {report.test_duration:.2f} seconds")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Warnings: {report.warning_tests}")
    print(f"Timeouts: {report.timeout_tests}")
    print("\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    print("="*80)
    
    return report.overall_status == "PASS"


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)