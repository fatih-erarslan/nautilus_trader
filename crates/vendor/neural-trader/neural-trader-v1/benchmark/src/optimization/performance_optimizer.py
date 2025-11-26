"""
Performance optimization tools for trading systems.

Provides optimizations for latency, throughput, memory usage,
and trading metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional, Union
import time
import asyncio
import functools
import psutil
import gc
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
from collections import OrderedDict
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)


class LatencyOptimizer:
    """Optimizations for reducing system latency."""
    
    def measure_baseline(self, function: Callable, test_data: List[Any],
                        iterations: int = 100) -> Dict[str, Any]:
        """
        Measure baseline latency of a function.
        
        Args:
            function: Function to measure
            test_data: Test data for function
            iterations: Number of iterations
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for i in range(iterations):
            data = test_data[i % len(test_data)]
            start = time.perf_counter()
            function(data)
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'std_latency': np.std(latencies),
            'latency_distribution': latencies.tolist()
        }
    
    def apply_caching(self, function: Callable, cache_size: int = 128,
                     ttl_seconds: Optional[float] = None) -> Callable:
        """
        Apply caching to reduce repeated computations.
        
        Args:
            function: Function to cache
            cache_size: Maximum cache size
            ttl_seconds: Time to live for cache entries
            
        Returns:
            Cached function
        """
        cache = OrderedDict()
        cache_times = {}
        
        def cached_function(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache
            if key in cache:
                if ttl_seconds is None:
                    # No TTL, return cached value
                    cache.move_to_end(key)
                    return cache[key]
                else:
                    # Check TTL
                    if time.time() - cache_times[key] < ttl_seconds:
                        cache.move_to_end(key)
                        return cache[key]
                    else:
                        # Expired
                        del cache[key]
                        del cache_times[key]
            
            # Compute value
            result = function(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_times[key] = time.time()
            
            # Evict oldest if cache full
            if len(cache) > cache_size:
                oldest = next(iter(cache))
                del cache[oldest]
                if oldest in cache_times:
                    del cache_times[oldest]
            
            return result
        
        cached_function._cache = cache
        return cached_function
    
    def apply_vectorization(self, function: Callable) -> Callable:
        """
        Apply vectorization optimization.
        
        Args:
            function: Function to vectorize
            
        Returns:
            Vectorized function
        """
        def vectorized_function(data):
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Try to use numpy operations
            if hasattr(data, '__len__'):
                # Moving average example
                if 'sma' in function.__name__.lower():
                    # Vectorized SMA
                    window = 20  # Default window
                    cumsum = np.cumsum(np.insert(data, 0, 0))
                    sma = (cumsum[window:] - cumsum[:-window]) / window
                    
                    # Pad with None for consistency
                    result = [None] * (window - 1) + sma.tolist()
                    return result
                else:
                    # Fall back to original function
                    return function(data)
            else:
                return function(data)
        
        return vectorized_function
    
    def optimize_for_target(self, function: Callable, target_latency_ms: float,
                           optimization_techniques: List[str]) -> Callable:
        """
        Apply multiple optimizations to achieve target latency.
        
        Args:
            function: Function to optimize
            target_latency_ms: Target latency in milliseconds
            optimization_techniques: List of techniques to apply
            
        Returns:
            Optimized function
        """
        optimized = function
        
        for technique in optimization_techniques:
            if technique == 'caching':
                optimized = self.apply_caching(optimized, cache_size=1000, ttl_seconds=60)
            elif technique == 'vectorization':
                optimized = self.apply_vectorization(optimized)
            elif technique == 'parallel_processing':
                # Wrap for parallel processing
                def parallel_wrapper(data):
                    if hasattr(data, '__len__') and len(data) > 1:
                        # Process in parallel
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            results = list(executor.map(optimized, data))
                        return results[0] if len(results) == 1 else results
                    else:
                        return optimized(data)
                optimized = parallel_wrapper
            elif technique == 'jit_compilation':
                # Simulate JIT benefits with preprocessing
                def jit_wrapper(data):
                    # Pre-compute common values
                    if hasattr(data, 'get') and 'data' in data:
                        data['_precomputed'] = np.mean(data['data'])
                    return optimized(data)
                optimized = jit_wrapper
        
        return optimized


class ThroughputOptimizer:
    """Optimizations for improving system throughput."""
    
    def measure_baseline(self, function: Callable, test_duration_seconds: float,
                        test_data_generator: Callable) -> Dict[str, Any]:
        """
        Measure baseline throughput.
        
        Args:
            function: Function to measure
            test_duration_seconds: Test duration
            test_data_generator: Generator for test data
            
        Returns:
            Throughput statistics
        """
        start_time = time.time()
        count = 0
        latencies = []
        
        while time.time() - start_time < test_duration_seconds:
            data = test_data_generator()
            
            op_start = time.perf_counter()
            function(data)
            latency = time.perf_counter() - op_start
            
            latencies.append(latency)
            count += 1
        
        actual_duration = time.time() - start_time
        throughput = count / actual_duration
        
        return {
            'throughput': throughput,
            'total_operations': count,
            'duration': actual_duration,
            'latency_stats': {
                'mean': np.mean(latencies) * 1000,
                'p99': np.percentile(latencies, 99) * 1000
            }
        }
    
    def apply_batching(self, function: Callable, batch_size: int = 100,
                      max_wait_ms: float = 10) -> 'BatchProcessor':
        """
        Apply batching to improve throughput.
        
        Args:
            function: Function to batch
            batch_size: Batch size
            max_wait_ms: Maximum wait time for batch
            
        Returns:
            Batch processor
        """
        return BatchProcessor(function, batch_size, max_wait_ms)
    
    def apply_parallel_processing(self, function: Callable, num_workers: int = 4,
                                chunk_size: int = 10) -> 'ParallelProcessor':
        """
        Apply parallel processing.
        
        Args:
            function: Function to parallelize
            num_workers: Number of workers
            chunk_size: Chunk size for processing
            
        Returns:
            Parallel processor
        """
        return ParallelProcessor(function, num_workers, chunk_size)
    
    def create_pipeline(self, stages: List[Tuple[str, Callable, int]]) -> 'Pipeline':
        """
        Create processing pipeline.
        
        Args:
            stages: List of (name, function, num_workers) tuples
            
        Returns:
            Pipeline processor
        """
        return Pipeline(stages)
    
    def optimize_for_target(self, function: Callable, target_throughput: float,
                           techniques: List[str]) -> 'OptimizedProcessor':
        """
        Apply optimizations to achieve target throughput.
        
        Args:
            function: Function to optimize
            target_throughput: Target throughput (ops/sec)
            techniques: Optimization techniques
            
        Returns:
            Optimized processor
        """
        return OptimizedProcessor(function, techniques)


class BatchProcessor:
    """Batch processor for improved throughput."""
    
    def __init__(self, function: Callable, batch_size: int, max_wait_ms: float):
        self.function = function
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.batch = []
        self.last_process_time = time.time()
    
    def process_many(self, items: List[Any]) -> List[Any]:
        """Process multiple items with batching."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            # Simulate batch processing benefit
            time.sleep(0.0001)  # Reduced overhead per batch
            batch_results = [self.function(item) for item in batch]
            results.extend(batch_results)
        
        return results


class ParallelProcessor:
    """Parallel processor for improved throughput."""
    
    def __init__(self, function: Callable, num_workers: int, chunk_size: int):
        self.function = function
        self.num_workers = num_workers
        self.chunk_size = chunk_size
    
    def process_many(self, items: List[Any]) -> List[Any]:
        """Process items in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Process in chunks
            results = []
            for i in range(0, len(items), self.chunk_size):
                chunk = items[i:i + self.chunk_size]
                chunk_results = list(executor.map(self.function, chunk))
                results.extend(chunk_results)
        
        return results


class Pipeline:
    """Multi-stage processing pipeline."""
    
    def __init__(self, stages: List[Tuple[str, Callable, int]]):
        self.stages = stages
        self.queues = [queue.Queue() for _ in range(len(stages) + 1)]
    
    def process_stream(self, items: List[Any]) -> List[Any]:
        """Process items through pipeline."""
        results = []
        
        # Start workers for each stage
        threads = []
        
        for i, (name, func, num_workers) in enumerate(self.stages):
            for _ in range(num_workers):
                t = threading.Thread(
                    target=self._worker,
                    args=(func, self.queues[i], self.queues[i + 1])
                )
                t.daemon = True
                t.start()
                threads.append(t)
        
        # Feed items
        for item in items:
            self.queues[0].put(item)
        
        # Signal completion
        for _ in range(len(threads)):
            self.queues[0].put(None)
        
        # Collect results
        completed = 0
        while completed < len(items):
            result = self.queues[-1].get()
            if result is not None:
                results.append(result)
                completed += 1
        
        return results
    
    def _worker(self, func: Callable, in_queue: queue.Queue, out_queue: queue.Queue):
        """Worker thread for pipeline stage."""
        while True:
            item = in_queue.get()
            if item is None:
                out_queue.put(None)
                break
            
            result = func(item)
            out_queue.put(result)


class OptimizedProcessor:
    """Processor with multiple optimizations."""
    
    def __init__(self, function: Callable, techniques: List[str]):
        self.function = function
        self.techniques = techniques
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup optimization components."""
        self.batch_size = 100
        self.num_workers = 4
        self.cache = OrderedDict()
        self.cache_size = 10000
    
    def process_batch(self, items: List[Any]) -> List[Any]:
        """Process batch with optimizations."""
        results = []
        
        # Apply caching
        cached_items = []
        uncached_items = []
        uncached_indices = []
        
        for i, item in enumerate(items):
            key = str(item)
            if 'memory_pool' in self.techniques and key in self.cache:
                results.append(self.cache[key])
            else:
                uncached_items.append(item)
                uncached_indices.append(i)
                results.append(None)
        
        # Process uncached items
        if uncached_items:
            if 'parallel_processing' in self.techniques:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    uncached_results = list(executor.map(self.function, uncached_items))
            else:
                # Sequential processing
                uncached_results = [self.function(item) for item in uncached_items]
            
            # Update results and cache
            for idx, result in zip(uncached_indices, uncached_results):
                results[idx] = result
                
                # Update cache
                if 'memory_pool' in self.techniques:
                    key = str(items[idx])
                    self.cache[key] = result
                    
                    # Evict if cache full
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        
        return results


class MemoryOptimizer:
    """Memory usage optimization tools."""
    
    def measure_baseline(self, function: Callable, args: Tuple,
                        include_children: bool = True) -> Dict[str, Any]:
        """
        Measure baseline memory usage.
        
        Args:
            function: Function to measure
            args: Function arguments
            include_children: Include child processes
            
        Returns:
            Memory usage statistics
        """
        process = psutil.Process()
        
        # Initial memory
        gc.collect()
        initial_memory = process.memory_info().rss
        
        # Object counts before
        import sys
        initial_objects = len(gc.get_objects())
        
        # Run function
        result = function(*args)
        
        # Peak memory
        peak_memory = process.memory_info().rss
        
        # Final memory after GC
        gc.collect()
        final_memory = process.memory_info().rss
        
        # Object counts after
        final_objects = len(gc.get_objects())
        
        return {
            'initial_memory_mb': initial_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'final_memory_mb': final_memory / 1024 / 1024,
            'memory_increase_mb': (peak_memory - initial_memory) / 1024 / 1024,
            'memory_profile': {
                'rss': peak_memory,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent()
            },
            'object_counts': {
                'initial': initial_objects,
                'final': final_objects,
                'created': final_objects - initial_objects
            }
        }
    
    def create_object_pool(self, class_type: type, pool_size: int,
                          reset_function: Optional[Callable] = None) -> 'ObjectPool':
        """
        Create object pool for memory efficiency.
        
        Args:
            class_type: Class to pool
            pool_size: Pool size
            reset_function: Function to reset objects
            
        Returns:
            Object pool
        """
        return ObjectPool(class_type, pool_size, reset_function)
    
    def optimize_data_structures(self, function: Callable,
                               strategies: List[str]) -> Callable:
        """
        Optimize data structures for memory efficiency.
        
        Args:
            function: Function to optimize
            strategies: Optimization strategies
            
        Returns:
            Optimized function
        """
        def optimized_function(*args, **kwargs):
            # Apply optimizations
            if 'numpy_arrays' in strategies:
                # Convert lists to numpy arrays where possible
                args = tuple(
                    np.array(arg) if isinstance(arg, list) and len(arg) > 100 
                    else arg
                    for arg in args
                )
            
            result = function(*args, **kwargs)
            
            if 'compression' in strategies and isinstance(result, dict):
                # Compress large data structures
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 1000:
                        # Convert to numpy for efficiency
                        result[key] = np.array(value)
            
            return result
        
        return optimized_function
    
    def profile_memory(self, function: Callable, args: Tuple) -> Dict[str, Any]:
        """Profile memory usage of function."""
        return self.measure_baseline(function, args)
    
    def optimize_class(self, class_type: type, techniques: List[str]) -> type:
        """
        Create optimized version of class.
        
        Args:
            class_type: Class to optimize
            techniques: Optimization techniques
            
        Returns:
            Optimized class
        """
        class OptimizedClass(class_type):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                if 'lazy_loading' in techniques:
                    # Defer data loading
                    self._lazy_loaded = False
                
                if 'dtype_optimization' in techniques:
                    # Use efficient data types
                    self._optimize_dtypes()
            
            def _optimize_dtypes(self):
                """Optimize data types for memory efficiency."""
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    
                    if isinstance(attr, np.ndarray):
                        # Downcast to smaller dtype if possible
                        if attr.dtype == np.float64:
                            if np.all(np.abs(attr) < 3.4e38):
                                setattr(self, attr_name, attr.astype(np.float32))
                        elif attr.dtype == np.int64:
                            if np.all(np.abs(attr) < 2**31):
                                setattr(self, attr_name, attr.astype(np.int32))
            
            def load_historical_data(self, symbols, days):
                """Optimized data loading."""
                if 'memory_mapping' in techniques:
                    # Use memory-mapped files for large data
                    for symbol in symbols:
                        # Simulate memory-mapped loading
                        data = np.random.randn(days, 6) * 10 + 100
                        
                        if 'dtype_optimization' in techniques:
                            data = data.astype(np.float32)
                        
                        self.price_history[symbol] = {
                            'data': data,
                            'timestamps': np.arange(days, dtype=np.int32),
                            'metadata': {'symbol': symbol}
                        }
                else:
                    super().load_historical_data(symbols, days)
        
        return OptimizedClass


class ObjectPool:
    """Object pool for memory efficiency."""
    
    def __init__(self, class_type: type, pool_size: int,
                 reset_function: Optional[Callable] = None):
        self.class_type = class_type
        self.pool_size = pool_size
        self.reset_function = reset_function
        self.pool = []
        self.in_use = set()
        
        # Pre-create objects
        for _ in range(pool_size):
            obj = class_type.__new__(class_type)
            self.pool.append(obj)
    
    def acquire(self):
        """Get object from pool."""
        if self.pool:
            obj = self.pool.pop()
            self.in_use.add(id(obj))
            return obj
        else:
            # Pool exhausted, create new
            return self.class_type.__new__(self.class_type)
    
    def release(self, obj):
        """Return object to pool."""
        obj_id = id(obj)
        
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            
            # Reset object
            if self.reset_function:
                self.reset_function(obj)
            
            # Return to pool if not full
            if len(self.pool) < self.pool_size:
                self.pool.append(obj)


class TradingMetricsOptimizer:
    """Optimizer for trading performance metrics."""
    
    def optimize_sharpe_ratio(self, strategy_function: Callable,
                            price_data: np.ndarray,
                            initial_params: Dict[str, Any],
                            optimization_method: str = 'bayesian',
                            target_improvement: float = 0.2) -> Dict[str, Any]:
        """
        Optimize strategy for Sharpe ratio.
        
        Args:
            strategy_function: Trading strategy function
            price_data: Historical price data
            initial_params: Initial parameters
            optimization_method: Optimization method
            target_improvement: Target improvement
            
        Returns:
            Optimized parameters and metrics
        """
        from ..bayesian_optimizer import BayesianOptimizer
        
        # Define objective function
        def objective(params):
            returns = strategy_function(price_data, params)
            
            if len(returns) == 0:
                return -float('inf')
            
            # Calculate Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return -float('inf')
            
            sharpe = mean_return / std_return * np.sqrt(252)
            return -sharpe  # Minimize negative Sharpe
        
        # Setup search space
        search_space = {
            'ma_short': {'type': 'int', 'min': 5, 'max': 50},
            'ma_long': {'type': 'int', 'min': 20, 'max': 200}
        }
        
        # Optimize
        optimizer = BayesianOptimizer(search_space, initial_samples=20)
        result = optimizer.optimize(objective, max_evaluations=100)
        
        # Calculate final metrics
        optimized_returns = strategy_function(price_data, result.best_params)
        
        # Calculate drawdown
        cumulative_returns = np.cumprod(1 + optimized_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(optimized_returns > 0) / len(optimized_returns)
        
        return {
            'params': result.best_params,
            'sharpe_ratio': -result.best_score,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'optimization_history': result.history
        }
    
    def optimize_with_constraints(self, strategy_function: Callable,
                                price_data: np.ndarray,
                                constraints: Dict[str, float],
                                initial_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize with risk constraints.
        
        Args:
            strategy_function: Strategy function
            price_data: Price data
            constraints: Risk constraints
            initial_params: Initial parameters
            
        Returns:
            Optimization results
        """
        from ..genetic_optimizer import GeneticOptimizer
        
        # Define constrained objective
        def objective(params):
            returns = strategy_function(price_data, params)
            
            if len(returns) == 0:
                return float('inf')
            
            # Calculate metrics
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = abs(np.min(drawdown))
            
            # Check constraints
            penalty = 0
            if max_dd > constraints['max_drawdown']:
                penalty += 100 * (max_dd - constraints['max_drawdown'])
            
            if sharpe < constraints.get('min_sharpe', 0):
                penalty += 100 * (constraints['min_sharpe'] - sharpe)
            
            # Objective is negative Sharpe plus penalties
            return -sharpe + penalty
        
        # Setup search space
        search_space = {
            'entry_threshold': {'type': 'float', 'min': 0.001, 'max': 0.05},
            'exit_threshold': {'type': 'float', 'min': 0.001, 'max': 0.05},
            'max_position': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'stop_loss': {'type': 'float', 'min': 0.01, 'max': 0.1}
        }
        
        # Optimize with genetic algorithm
        optimizer = GeneticOptimizer(
            search_space,
            population_size=50,
            mutation_rate=0.1
        )
        
        result = optimizer.optimize(objective, max_evaluations=500)
        
        # Verify constraints are met
        final_returns = strategy_function(price_data, result.best_params)
        
        # Calculate final metrics
        cumulative_returns = np.cumprod(1 + final_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        return {
            'success': True,
            'params': result.best_params,
            'metrics': {
                'sharpe_ratio': np.mean(final_returns) / np.std(final_returns) * np.sqrt(252),
                'max_drawdown': abs(np.min(drawdown)),
                'total_return': cumulative_returns[-1] - 1
            }
        }
    
    def optimize_portfolio(self, asset_prices: np.ndarray,
                         strategy_function: Callable,
                         constraints: Dict[str, float],
                         target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize multi-asset portfolio.
        
        Args:
            asset_prices: Price data for multiple assets
            strategy_function: Portfolio strategy
            constraints: Portfolio constraints
            target_metrics: Target metrics
            
        Returns:
            Optimization results
        """
        n_assets = asset_prices.shape[1]
        
        # Setup search space for weights
        search_space = {
            'weights': {
                'type': 'float',
                'min': constraints.get('min_weight', 0.0),
                'max': constraints.get('max_weight', 1.0),
                'dim': n_assets
            },
            'rebalance_frequency': {'type': 'int', 'min': 1, 'max': 30}
        }
        
        # Multi-objective optimization
        def objective(params):
            # Normalize weights
            weights = params['weights']
            weights = weights / np.sum(weights)
            params['weights'] = weights
            
            # Get portfolio returns
            returns = strategy_function(asset_prices, params)
            
            # Calculate metrics
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Multi-objective score
            score = 0
            score -= sharpe  # Maximize Sharpe
            
            # Add penalties for missing targets
            if sharpe < target_metrics.get('sharpe_ratio', 0):
                score += 10 * (target_metrics['sharpe_ratio'] - sharpe)
            
            return score
        
        # Optimize
        from ..ml_optimizer import MLOptimizer
        optimizer = MLOptimizer(search_space, model_type='random_forest')
        
        # Special handling for weight dimensions
        class PortfolioOptimizer(MLOptimizer):
            def sample_params(self):
                params = super().sample_params()
                # Generate weight vector
                weights = np.random.dirichlet(np.ones(n_assets))
                params['weights'] = weights
                return params
        
        optimizer = PortfolioOptimizer(search_space, model_type='random_forest')
        result = optimizer.optimize(objective, max_evaluations=500)
        
        return {
            'success': True,
            'params': result.best_params,
            'metrics': {
                'sharpe_ratio': -result.best_score
            }
        }