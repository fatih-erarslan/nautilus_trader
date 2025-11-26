"""
Dynamic Batch Processing Optimization for GPU Trading
Advanced batch size optimization and memory-aware processing for maximum GPU utilization.
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import logging
from dataclasses import dataclass
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import gc
from collections import deque

from .flyio_gpu_config import get_gpu_config_manager
from .mixed_precision import MixedPrecisionManager

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies."""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    MEMORY_AWARE = "memory_aware"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for batch processing."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    DYNAMIC = "dynamic"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    initial_batch_size: int = 1000
    min_batch_size: int = 100
    max_batch_size: int = 100000
    target_memory_utilization: float = 0.8
    target_gpu_utilization: float = 0.85
    adaptation_rate: float = 0.1
    performance_window: int = 10
    
    # Queue management
    max_queue_size: int = 10000
    queue_timeout: float = 1.0
    
    # Parallel processing
    num_workers: int = 4
    enable_async_processing: bool = True
    
    # Memory optimization
    enable_memory_monitoring: bool = True
    memory_check_interval: int = 100
    garbage_collection_threshold: float = 0.9


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    batch_size: int
    processing_time: float
    memory_used: float
    gpu_utilization: float
    throughput: float
    latency: float
    success_rate: float
    timestamp: float


class MemoryMonitor:
    """Monitors GPU memory usage for batch optimization."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        self.history = deque(maxlen=100)
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        used_bytes = self.memory_pool.used_bytes()
        total_bytes = self.memory_pool.total_bytes()
        
        # Get device memory info
        device = cp.cuda.Device()
        device_memory = device.mem_info
        
        info = {
            'pool_used_gb': used_bytes / (1024**3),
            'pool_total_gb': total_bytes / (1024**3),
            'pool_free_gb': (total_bytes - used_bytes) / (1024**3),
            'pool_utilization': used_bytes / max(total_bytes, 1),
            'device_free_gb': device_memory[0] / (1024**3),
            'device_total_gb': device_memory[1] / (1024**3),
            'device_utilization': 1.0 - (device_memory[0] / device_memory[1])
        }
        
        self.history.append(info)
        return info
        
    def optimize_memory(self):
        """Optimize memory usage."""
        # Free unused memory blocks
        self.memory_pool.free_all_blocks()
        self.pinned_memory_pool.free_all_blocks()
        
        # Force garbage collection
        gc.collect()
        
        logger.debug("Memory optimized")
        
    def predict_memory_usage(self, batch_size: int, data_shape: Tuple[int, ...], 
                           dtype: cp.dtype = cp.float32) -> float:
        """Predict memory usage for a given batch size."""
        elements_per_sample = np.prod(data_shape)
        total_elements = batch_size * elements_per_sample
        
        # Estimate memory usage (data + intermediate results + overhead)
        base_memory = total_elements * dtype().itemsize
        overhead_factor = 3.0  # Account for intermediate results and overhead
        
        predicted_gb = (base_memory * overhead_factor) / (1024**3)
        return predicted_gb
        
    def get_max_batch_size(self, data_shape: Tuple[int, ...], 
                          dtype: cp.dtype = cp.float32,
                          memory_fraction: float = 0.8) -> int:
        """Calculate maximum batch size based on available memory."""
        memory_info = self.get_memory_info()
        available_memory_gb = memory_info['device_free_gb'] * memory_fraction
        
        elements_per_sample = np.prod(data_shape)
        bytes_per_element = dtype().itemsize
        overhead_factor = 3.0
        
        max_elements = (available_memory_gb * (1024**3)) / (bytes_per_element * overhead_factor)
        max_batch_size = int(max_elements / elements_per_sample)
        
        return max(1, max_batch_size)


class AdaptiveBatchSizer:
    """Adaptive batch size optimization based on performance metrics."""
    
    def __init__(self, config: BatchConfig, memory_monitor: MemoryMonitor):
        """Initialize adaptive batch sizer."""
        self.config = config
        self.memory_monitor = memory_monitor
        self.current_batch_size = config.initial_batch_size
        self.performance_history = deque(maxlen=config.performance_window)
        self.adaptation_count = 0
        
    def update_batch_size(self, metrics: BatchMetrics) -> int:
        """Update batch size based on performance metrics."""
        self.performance_history.append(metrics)
        
        if len(self.performance_history) < 2:
            return self.current_batch_size
            
        # Calculate performance trends
        recent_metrics = list(self.performance_history)[-3:]
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_memory_util = np.mean([m.memory_used for m in recent_metrics])
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])
        
        # Determine adjustment
        adjustment_factor = 1.0
        
        # Strategy-specific adjustments
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            adjustment_factor = self._adaptive_adjustment(
                avg_throughput, avg_memory_util, avg_gpu_util
            )
        elif self.config.strategy == BatchStrategy.MEMORY_AWARE:
            adjustment_factor = self._memory_aware_adjustment(avg_memory_util)
        elif self.config.strategy == BatchStrategy.PERFORMANCE_OPTIMIZED:
            adjustment_factor = self._performance_optimized_adjustment(avg_throughput)
        elif self.config.strategy == BatchStrategy.LATENCY_OPTIMIZED:
            adjustment_factor = self._latency_optimized_adjustment(metrics.latency)
            
        # Apply adjustment
        new_batch_size = int(self.current_batch_size * adjustment_factor)
        new_batch_size = np.clip(new_batch_size, 
                               self.config.min_batch_size, 
                               self.config.max_batch_size)
        
        # Check memory constraints
        memory_info = self.memory_monitor.get_memory_info()
        if memory_info['device_utilization'] > self.config.target_memory_utilization:
            # Reduce batch size if memory pressure is high
            new_batch_size = int(new_batch_size * 0.8)
            
        if new_batch_size != self.current_batch_size:
            logger.debug(f"Batch size adjusted: {self.current_batch_size} -> {new_batch_size}")
            self.current_batch_size = new_batch_size
            self.adaptation_count += 1
            
        return self.current_batch_size
        
    def _adaptive_adjustment(self, throughput: float, memory_util: float, 
                           gpu_util: float) -> float:
        """Adaptive adjustment based on multiple metrics."""
        adjustment = 1.0
        
        # Throughput-based adjustment
        if throughput > 0:
            if len(self.performance_history) >= 2:
                prev_throughput = self.performance_history[-2].throughput
                if throughput > prev_throughput * 1.1:
                    adjustment *= 1.1  # Increase batch size
                elif throughput < prev_throughput * 0.9:
                    adjustment *= 0.9  # Decrease batch size
                    
        # Memory utilization adjustment
        if memory_util > self.config.target_memory_utilization:
            adjustment *= 0.8
        elif memory_util < self.config.target_memory_utilization * 0.7:
            adjustment *= 1.2
            
        # GPU utilization adjustment
        if gpu_util < self.config.target_gpu_utilization * 0.8:
            adjustment *= 1.1
        elif gpu_util > 0.95:
            adjustment *= 0.9
            
        return adjustment
        
    def _memory_aware_adjustment(self, memory_util: float) -> float:
        """Memory-aware batch size adjustment."""
        target_util = self.config.target_memory_utilization
        
        if memory_util > target_util:
            # Reduce batch size aggressively if over target
            return 0.7
        elif memory_util < target_util * 0.6:
            # Increase batch size if well under target
            return 1.3
        else:
            # Fine-tune around target
            ratio = memory_util / target_util
            return 1.0 + (1.0 - ratio) * 0.2
            
    def _performance_optimized_adjustment(self, throughput: float) -> float:
        """Performance-optimized batch size adjustment."""
        if len(self.performance_history) < 2:
            return 1.0
            
        # Look at throughput trend
        recent_throughputs = [m.throughput for m in list(self.performance_history)[-3:]]
        
        if len(recent_throughputs) >= 2:
            trend = recent_throughputs[-1] - recent_throughputs[-2]
            if trend > 0:
                return 1.1  # Increasing throughput, increase batch size
            elif trend < -throughput * 0.05:
                return 0.9  # Decreasing throughput, reduce batch size
                
        return 1.0
        
    def _latency_optimized_adjustment(self, latency: float) -> float:
        """Latency-optimized batch size adjustment."""
        if len(self.performance_history) < 2:
            return 1.0
            
        # Target latency of 100ms
        target_latency = 0.1
        
        if latency > target_latency * 1.5:
            return 0.8  # Reduce batch size for lower latency
        elif latency < target_latency * 0.5:
            return 1.2  # Increase batch size if latency is very low
            
        return 1.0


class BatchProcessor:
    """High-performance batch processor with dynamic optimization."""
    
    def __init__(self, config: Optional[BatchConfig] = None,
                 precision_manager: Optional[MixedPrecisionManager] = None):
        """Initialize batch processor."""
        self.config = config or BatchConfig()
        self.precision_manager = precision_manager
        self.memory_monitor = MemoryMonitor()
        self.batch_sizer = AdaptiveBatchSizer(self.config, self.memory_monitor)
        
        # Processing state
        self.processing_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        self.is_running = False
        self.workers = []
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.total_processed = 0
        self.total_processing_time = 0.0
        
    def start(self):
        """Start batch processing workers."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start worker threads
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started {self.config.num_workers} batch processing workers")
        
    def stop(self):
        """Stop batch processing workers."""
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        self.workers.clear()
        logger.info("Stopped batch processing workers")
        
    def process_batch(self, data: cp.ndarray, 
                     processing_func: Callable[[cp.ndarray], cp.ndarray],
                     batch_size: Optional[int] = None) -> cp.ndarray:
        """Process data in optimized batches."""
        if batch_size is None:
            batch_size = self.batch_sizer.current_batch_size
            
        start_time = time.time()
        
        # Split data into batches
        batches = self._split_into_batches(data, batch_size)
        results = []
        
        # Process each batch
        for i, batch in enumerate(batches):
            batch_start = time.time()
            
            # Memory check
            if i % self.config.memory_check_interval == 0:
                memory_info = self.memory_monitor.get_memory_info()
                if memory_info['device_utilization'] > self.config.garbage_collection_threshold:
                    self.memory_monitor.optimize_memory()
                    
            # Process batch with precision optimization
            if self.precision_manager:
                batch = self.precision_manager.convert_to_compute_precision(batch)
                
            try:
                with cp.cuda.Stream():
                    batch_result = processing_func(batch)
                    results.append(batch_result)
                    
                batch_time = time.time() - batch_start
                
                # Record metrics
                memory_info = self.memory_monitor.get_memory_info()
                metrics = BatchMetrics(
                    batch_size=len(batch),
                    processing_time=batch_time,
                    memory_used=memory_info['device_utilization'],
                    gpu_utilization=0.0,  # Would need nvidia-ml-py for actual GPU util
                    throughput=len(batch) / batch_time,
                    latency=batch_time,
                    success_rate=1.0,
                    timestamp=time.time()
                )
                
                self.metrics_history.append(metrics)
                
                # Update batch size
                if self.config.strategy != BatchStrategy.FIXED:
                    self.batch_sizer.update_batch_size(metrics)
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                # Record failure metrics
                metrics = BatchMetrics(
                    batch_size=len(batch),
                    processing_time=time.time() - batch_start,
                    memory_used=memory_info['device_utilization'],
                    gpu_utilization=0.0,
                    throughput=0.0,
                    latency=time.time() - batch_start,
                    success_rate=0.0,
                    timestamp=time.time()
                )
                self.metrics_history.append(metrics)
                raise
                
        # Concatenate results
        if results:
            final_result = cp.concatenate(results, axis=0)
        else:
            final_result = cp.array([])
            
        # Update statistics
        total_time = time.time() - start_time
        self.total_processed += len(data)
        self.total_processing_time += total_time
        
        return final_result
        
    def process_async(self, data: cp.ndarray,
                     processing_func: Callable[[cp.ndarray], cp.ndarray],
                     callback: Optional[Callable] = None) -> threading.Thread:
        """Process data asynchronously."""
        def async_process():
            try:
                result = self.process_batch(data, processing_func)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Async processing failed: {str(e)}")
                if callback:
                    callback(None)
                    
        thread = threading.Thread(target=async_process)
        thread.start()
        return thread
        
    def _split_into_batches(self, data: cp.ndarray, batch_size: int) -> List[cp.ndarray]:
        """Split data into batches."""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        return batches
        
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing batches."""
        logger.debug(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get work item
                work_item = self.processing_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                    
                data, processing_func, callback = work_item
                
                # Process data
                result = self.process_batch(data, processing_func)
                
                # Send result
                if callback:
                    callback(result)
                else:
                    self.result_queue.put(result)
                    
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                
        logger.debug(f"Worker {worker_id} stopped")
        
    def get_optimal_batch_size(self, data_shape: Tuple[int, ...], 
                             dtype: cp.dtype = cp.float32) -> int:
        """Get optimal batch size for given data characteristics."""
        # Memory-based calculation
        memory_batch_size = self.memory_monitor.get_max_batch_size(
            data_shape, dtype, self.config.target_memory_utilization
        )
        
        # Performance-based adjustment
        if self.metrics_history:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_throughput = np.mean([m.throughput for m in recent_metrics])
            
            # Adjust based on historical performance
            if avg_throughput > 0:
                performance_factor = min(avg_throughput / 1000, 2.0)  # Cap at 2x
                memory_batch_size = int(memory_batch_size * performance_factor)
                
        # Apply configuration constraints
        optimal_size = np.clip(memory_batch_size,
                             self.config.min_batch_size,
                             self.config.max_batch_size)
        
        return optimal_size
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)[-100:]
        
        stats = {
            'current_batch_size': self.batch_sizer.current_batch_size,
            'total_processed': self.total_processed,
            'total_processing_time': self.total_processing_time,
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'avg_latency': np.mean([m.latency for m in recent_metrics]),
            'avg_memory_utilization': np.mean([m.memory_used for m in recent_metrics]),
            'success_rate': np.mean([m.success_rate for m in recent_metrics]),
            'adaptations_count': self.batch_sizer.adaptation_count,
            'strategy': self.config.strategy.value,
            'num_workers': self.config.num_workers,
            'queue_size': self.processing_queue.qsize(),
            'memory_info': self.memory_monitor.get_memory_info()
        }
        
        return stats
        
    def optimize_for_workload(self, sample_data: cp.ndarray,
                            processing_func: Callable[[cp.ndarray], cp.ndarray],
                            test_sizes: List[int] = None) -> Dict[str, Any]:
        """Optimize batch processing for specific workload."""
        if test_sizes is None:
            test_sizes = [100, 500, 1000, 2000, 5000, 10000]
            
        logger.info("Optimizing batch processing for workload...")
        
        results = {}
        best_throughput = 0
        best_batch_size = self.config.initial_batch_size
        
        for batch_size in test_sizes:
            try:
                # Test with this batch size
                test_data = sample_data[:min(len(sample_data), batch_size * 3)]
                
                start_time = time.time()
                _ = self.process_batch(test_data, processing_func, batch_size)
                end_time = time.time()
                
                processing_time = end_time - start_time
                throughput = len(test_data) / processing_time
                
                memory_info = self.memory_monitor.get_memory_info()
                
                results[batch_size] = {
                    'throughput': throughput,
                    'processing_time': processing_time,
                    'memory_utilization': memory_info['device_utilization'],
                    'memory_used_gb': memory_info['pool_used_gb']
                }
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    
                logger.debug(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
                
            except Exception as e:
                logger.warning(f"Failed to test batch size {batch_size}: {str(e)}")
                results[batch_size] = {'error': str(e)}
                
        # Update configuration with optimal batch size
        self.batch_sizer.current_batch_size = best_batch_size
        self.config.initial_batch_size = best_batch_size
        
        logger.info(f"Optimal batch size: {best_batch_size} (throughput: {best_throughput:.2f})")
        
        return {
            'optimal_batch_size': best_batch_size,
            'best_throughput': best_throughput,
            'test_results': results
        }


def create_batch_processor(strategy: str = "adaptive",
                         precision_mode: Optional[str] = None) -> BatchProcessor:
    """Create optimized batch processor."""
    # Get GPU configuration
    gpu_config = get_gpu_config_manager().config
    
    # Create batch config
    batch_config = BatchConfig(
        strategy=BatchStrategy(strategy),
        initial_batch_size=gpu_config.optimal_batch_size,
        min_batch_size=gpu_config.min_batch_size,
        max_batch_size=gpu_config.max_batch_size,
        target_memory_utilization=gpu_config.max_memory_utilization,
        target_gpu_utilization=gpu_config.target_gpu_utilization
    )
    
    # Create precision manager if specified
    precision_manager = None
    if precision_mode:
        from .mixed_precision import create_mixed_precision_manager
        precision_manager = create_mixed_precision_manager(precision_mode)
    
    return BatchProcessor(batch_config, precision_manager)


if __name__ == "__main__":
    # Test batch processor
    logger.info("Testing batch processor...")
    
    # Create processor
    processor = create_batch_processor("adaptive")
    
    # Create test data
    test_data = cp.random.rand(10000, 100).astype(cp.float32)
    
    # Define test processing function
    def test_processing(batch):
        # Simple matrix operations
        return cp.matmul(batch, batch.T).sum(axis=1)
    
    # Test batch processing
    start_time = time.time()
    result = processor.process_batch(test_data, test_processing)
    end_time = time.time()
    
    logger.info(f"Processed {len(test_data)} samples in {end_time - start_time:.3f}s")
    logger.info(f"Result shape: {result.shape}")
    
    # Get performance stats
    stats = processor.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    # Optimize for workload
    optimization_results = processor.optimize_for_workload(test_data, test_processing)
    logger.info(f"Optimization results: {optimization_results}")
    
    print("Batch processor tested successfully!")