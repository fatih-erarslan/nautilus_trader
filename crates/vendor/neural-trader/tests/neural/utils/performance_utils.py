"""
Performance Testing Utilities

Utilities for performance testing, benchmarking, and monitoring of neural forecasting components.
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import asyncio
import json
from pathlib import Path
import statistics
import warnings
from functools import wraps


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency_ms: List[float] = field(default_factory=list)
    throughput_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    error_rate: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    timeout_seconds: float = 60.0
    memory_check_interval: float = 0.1
    acceptable_variance: float = 0.2  # 20%
    target_percentiles: List[int] = field(default_factory=lambda: [50, 90, 95, 99])


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import GPUtil
                self.gpu_utils = GPUtil
            except ImportError:
                self.gpu_utils = None
                warnings.warn("GPUtil not available for GPU monitoring")
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        self.monitoring = True
        self.metrics_history.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics_history.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                warnings.warn(f"Error collecting metrics: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_mb': psutil.virtual_memory().used / (1024**2),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        # GPU metrics
        if self.gpu_available:
            metrics['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
            
            if self.gpu_utils:
                try:
                    gpus = self.gpu_utils.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        metrics['gpu_utilization_percent'] = gpu.load * 100
                        metrics['gpu_temperature'] = gpu.temperature
                except Exception:
                    pass
        
        return metrics
    
    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage from monitoring history."""
        if not self.metrics_history:
            return {}
        
        peak_metrics = {
            'peak_cpu_percent': max(m['cpu_percent'] for m in self.metrics_history),
            'peak_memory_mb': max(m['memory_mb'] for m in self.metrics_history),
            'peak_memory_percent': max(m['memory_percent'] for m in self.metrics_history)
        }
        
        if self.gpu_available:
            peak_metrics.update({
                'peak_gpu_memory_mb': max(m.get('gpu_memory_mb', 0) for m in self.metrics_history),
                'peak_gpu_utilization_percent': max(m.get('gpu_utilization_percent', 0) for m in self.metrics_history)
            })
        
        return peak_metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average resource usage from monitoring history."""
        if not self.metrics_history:
            return {}
        
        avg_metrics = {
            'avg_cpu_percent': statistics.mean(m['cpu_percent'] for m in self.metrics_history),
            'avg_memory_mb': statistics.mean(m['memory_mb'] for m in self.metrics_history),
            'avg_memory_percent': statistics.mean(m['memory_percent'] for m in self.metrics_history)
        }
        
        if self.gpu_available:
            avg_metrics.update({
                'avg_gpu_memory_mb': statistics.mean(m.get('gpu_memory_mb', 0) for m in self.metrics_history),
                'avg_gpu_utilization_percent': statistics.mean(m.get('gpu_utilization_percent', 0) for m in self.metrics_history)
            })
        
        return avg_metrics


class LatencyBenchmark:
    """Benchmark latency performance."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Benchmark a function's latency."""
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        errors = 0
        
        for i in range(self.config.benchmark_iterations):
            try:
                start_time = time.perf_counter()
                
                result = func(*args, **kwargs)
                
                # Ensure GPU operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                errors += 1
                warnings.warn(f"Error in benchmark iteration {i}: {e}")
        
        if not latencies:
            raise RuntimeError("All benchmark iterations failed")
        
        # Calculate statistics
        stats = {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'error_rate': errors / self.config.benchmark_iterations
        }
        
        # Add percentiles
        for percentile in self.config.target_percentiles:
            stats[f'p{percentile}_ms'] = np.percentile(latencies, percentile)
        
        self.results.append({
            'function': func.__name__,
            'timestamp': datetime.now(),
            'stats': stats,
            'raw_latencies': latencies
        })
        
        return stats
    
    async def benchmark_async_function(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Benchmark an async function's latency."""
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                await func(*args, **kwargs)
            except Exception:
                pass
        
        # Benchmark
        latencies = []
        errors = 0
        
        for i in range(self.config.benchmark_iterations):
            try:
                start_time = time.perf_counter()
                
                result = await func(*args, **kwargs)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                errors += 1
                warnings.warn(f"Error in async benchmark iteration {i}: {e}")
        
        if not latencies:
            raise RuntimeError("All async benchmark iterations failed")
        
        # Calculate statistics
        stats = {
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'error_rate': errors / self.config.benchmark_iterations
        }
        
        # Add percentiles
        for percentile in self.config.target_percentiles:
            stats[f'p{percentile}_ms'] = np.percentile(latencies, percentile)
        
        return stats


class ThroughputBenchmark:
    """Benchmark throughput performance."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
    
    def benchmark_throughput(self, func: Callable, data_generator: Callable, 
                           duration_seconds: float = 10.0) -> Dict[str, float]:
        """Benchmark throughput over a time period."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        operations_completed = 0
        items_processed = 0
        errors = 0
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                data = data_generator()
                func(data)
            except Exception:
                pass
        
        # Benchmark
        while time.time() < end_time:
            try:
                data = data_generator()
                
                op_start = time.time()
                result = func(data)
                op_end = time.time()
                
                operations_completed += 1
                
                # Count items processed (batch size, etc.)
                if hasattr(data, '__len__'):
                    items_processed += len(data)
                elif isinstance(data, (list, tuple)):
                    items_processed += len(data)
                else:
                    items_processed += 1
                    
            except Exception as e:
                errors += 1
                warnings.warn(f"Error in throughput benchmark: {e}")
        
        actual_duration = time.time() - start_time
        
        stats = {
            'operations_per_second': operations_completed / actual_duration,
            'items_per_second': items_processed / actual_duration,
            'total_operations': operations_completed,
            'total_items': items_processed,
            'duration_seconds': actual_duration,
            'error_rate': errors / max(operations_completed + errors, 1)
        }
        
        self.results.append({
            'function': func.__name__,
            'timestamp': datetime.now(),
            'stats': stats
        })
        
        return stats


class MemoryBenchmark:
    """Benchmark memory usage."""
    
    def __init__(self):
        self.baseline_memory = None
        self.peak_memory = None
        self.memory_timeline = []
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory.copy()
        self.memory_timeline = [self.baseline_memory]
    
    def checkpoint(self, name: str = "") -> Dict[str, float]:
        """Record memory checkpoint."""
        current_memory = self._get_current_memory()
        current_memory['checkpoint'] = name
        current_memory['timestamp'] = datetime.now()
        
        self.memory_timeline.append(current_memory)
        
        # Update peak memory
        for key in ['cpu_memory_mb', 'gpu_memory_mb']:
            if key in current_memory and key in self.peak_memory:
                self.peak_memory[key] = max(self.peak_memory[key], current_memory[key])
        
        return current_memory
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.baseline_memory:
            return {}
        
        current_memory = self._get_current_memory()
        
        stats = {
            'baseline': self.baseline_memory,
            'current': current_memory,
            'peak': self.peak_memory,
            'increase': {},
            'timeline': self.memory_timeline
        }
        
        # Calculate memory increase
        for key in ['cpu_memory_mb', 'gpu_memory_mb']:
            if key in current_memory and key in self.baseline_memory:
                stats['increase'][key] = current_memory[key] - self.baseline_memory[key]
        
        return stats
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory = {
            'cpu_memory_mb': psutil.Process().memory_info().rss / (1024**2)
        }
        
        if torch.cuda.is_available():
            memory['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024**2)
            memory['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
        else:
            memory['gpu_memory_mb'] = 0
            memory['gpu_memory_cached_mb'] = 0
        
        return memory


class StressTester:
    """Stress testing utilities."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
    
    async def concurrent_stress_test(self, func: Callable, 
                                   concurrent_requests: int = 100,
                                   duration_seconds: float = 60.0,
                                   *args, **kwargs) -> Dict[str, Any]:
        """Run concurrent stress test."""
        
        results = {
            'start_time': datetime.now(),
            'concurrent_requests': concurrent_requests,
            'duration_seconds': duration_seconds,
            'completed_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request():
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        func(*args, **kwargs)
                    
                    end_time = time.perf_counter()
                    response_time = (end_time - start_time) * 1000
                    
                    results['completed_requests'] += 1
                    results['response_times'].append(response_time)
                    
                except Exception as e:
                    results['failed_requests'] += 1
                    results['errors'].append(str(e))
        
        # Start monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Generate requests for the duration
            start_time = time.time()
            tasks = []
            
            while time.time() - start_time < duration_seconds:
                task = asyncio.create_task(bounded_request())
                tasks.append(task)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            monitoring_data = monitor.stop_monitoring()
        
        results['end_time'] = datetime.now()
        results['actual_duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # Calculate statistics
        if results['response_times']:
            results['response_time_stats'] = {
                'mean_ms': statistics.mean(results['response_times']),
                'median_ms': statistics.median(results['response_times']),
                'p95_ms': np.percentile(results['response_times'], 95),
                'p99_ms': np.percentile(results['response_times'], 99),
                'min_ms': min(results['response_times']),
                'max_ms': max(results['response_times'])
            }
        
        total_requests = results['completed_requests'] + results['failed_requests']
        results['error_rate'] = results['failed_requests'] / max(total_requests, 1)
        results['requests_per_second'] = results['completed_requests'] / results['actual_duration']
        
        # Add resource usage
        if monitoring_data:
            results['resource_usage'] = monitor.get_peak_metrics()
        
        self.results.append(results)
        return results
    
    def memory_stress_test(self, func: Callable, 
                          iterations: int = 1000,
                          memory_limit_mb: float = 1000.0) -> Dict[str, Any]:
        """Test for memory leaks and excessive usage."""
        
        memory_benchmark = MemoryBenchmark()
        memory_benchmark.start_monitoring()
        
        results = {
            'start_time': datetime.now(),
            'iterations': iterations,
            'memory_limit_mb': memory_limit_mb,
            'completed_iterations': 0,
            'memory_violations': 0,
            'memory_checkpoints': []
        }
        
        try:
            for i in range(iterations):
                # Run function
                func()
                
                results['completed_iterations'] += 1
                
                # Check memory every 100 iterations
                if i % 100 == 0:
                    memory_info = memory_benchmark.checkpoint(f"iteration_{i}")
                    results['memory_checkpoints'].append(memory_info)
                    
                    # Check memory limit
                    total_memory = memory_info.get('cpu_memory_mb', 0) + memory_info.get('gpu_memory_mb', 0)
                    if total_memory > memory_limit_mb:
                        results['memory_violations'] += 1
                        
                        if results['memory_violations'] > 5:  # Too many violations
                            warnings.warn(f"Memory limit exceeded too many times: {total_memory:.1f}MB > {memory_limit_mb}MB")
                            break
        
        except Exception as e:
            results['error'] = str(e)
        
        finally:
            results['final_memory'] = memory_benchmark.get_memory_stats()
            results['end_time'] = datetime.now()
        
        return results


# Decorators for performance testing
def benchmark_latency(config: Optional[BenchmarkConfig] = None):
    """Decorator to benchmark function latency."""
    if config is None:
        config = BenchmarkConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            benchmark = LatencyBenchmark(config)
            stats = benchmark.benchmark_function(func, *args, **kwargs)
            
            # Attach stats to function for inspection
            if not hasattr(func, '_benchmark_stats'):
                func._benchmark_stats = []
            func._benchmark_stats.append(stats)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_memory(clear_cache: bool = True):
    """Decorator to monitor memory usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            benchmark = MemoryBenchmark()
            benchmark.start_monitoring()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                memory_stats = benchmark.get_memory_stats()
                
                # Attach stats to function
                if not hasattr(func, '_memory_stats'):
                    func._memory_stats = []
                func._memory_stats.append(memory_stats)
                
                if clear_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return wrapper
    return decorator


# Context managers
@contextmanager
def performance_monitoring(monitoring_interval: float = 0.1):
    """Context manager for performance monitoring."""
    monitor = PerformanceMonitor(monitoring_interval)
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        monitoring_data = monitor.stop_monitoring()


@contextmanager
def memory_tracking():
    """Context manager for memory tracking."""
    benchmark = MemoryBenchmark()
    benchmark.start_monitoring()
    
    try:
        yield benchmark
    finally:
        pass  # Memory stats available via benchmark.get_memory_stats()


# Utility functions
def assert_performance_regression(current_stats: Dict[str, float],
                                baseline_stats: Dict[str, float],
                                tolerance_percent: float = 20.0):
    """Assert that performance hasn't regressed beyond tolerance."""
    
    for metric in ['mean_ms', 'p95_ms', 'p99_ms']:
        if metric in current_stats and metric in baseline_stats:
            current_value = current_stats[metric]
            baseline_value = baseline_stats[metric]
            
            if baseline_value > 0:
                regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                
                if regression_percent > tolerance_percent:
                    raise AssertionError(
                        f"Performance regression detected for {metric}: "
                        f"{current_value:.2f}ms vs baseline {baseline_value:.2f}ms "
                        f"({regression_percent:.1f}% increase > {tolerance_percent}% tolerance)"
                    )


def save_benchmark_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to file."""
    # Convert datetime objects to strings for JSON serialization
    def serialize_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_datetime(item) for item in obj]
        else:
            return obj
    
    serialized_results = serialize_datetime(results)
    
    with open(output_path, 'w') as f:
        json.dump(serialized_results, f, indent=2)


def load_benchmark_baseline(baseline_path: Path) -> Dict[str, Any]:
    """Load benchmark baseline from file."""
    with open(baseline_path, 'r') as f:
        return json.load(f)


# Export utilities
__all__ = [
    'PerformanceMetrics',
    'BenchmarkConfig',
    'PerformanceMonitor',
    'LatencyBenchmark',
    'ThroughputBenchmark',
    'MemoryBenchmark',
    'StressTester',
    'benchmark_latency',
    'monitor_memory',
    'performance_monitoring',
    'memory_tracking',
    'assert_performance_regression',
    'save_benchmark_results',
    'load_benchmark_baseline'
]