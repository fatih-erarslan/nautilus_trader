"""
Advanced GPU Memory Manager with Dynamic Batch Sizing
Provides intelligent memory management, dynamic batch optimization, and memory pool allocation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock
import psutil
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics for monitoring."""
    allocated_mb: float
    cached_mb: float
    max_allocated_mb: float
    total_mb: float
    fragmentation_ratio: float
    pool_efficiency: float
    gc_count: int
    

@dataclass
class BatchOptimizationResult:
    """Result of batch size optimization."""
    optimal_batch_size: int
    max_throughput: float
    memory_efficiency: float
    latency_p95: float
    recommendation: str


class MemoryPool:
    """Custom memory pool for efficient tensor allocation."""
    
    def __init__(self, pool_size_mb: int = 1024, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.allocated_tensors = {}
        self.free_blocks = defaultdict(list)  # size -> list of free blocks
        self.allocation_count = 0
        self.deallocation_count = 0
        self.lock = Lock()
        
        # Pre-allocate pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize memory pool with pre-allocated blocks."""
        if self.device.type == 'cuda':
            try:
                # Allocate large block and fragment it
                pool_tensor = torch.empty(self.pool_size_bytes // 4, dtype=torch.float32, device=self.device)
                
                # Create different sized blocks
                block_sizes = [4096, 16384, 65536, 262144, 1048576]  # Powers of 4
                
                offset = 0
                for size in block_sizes:
                    num_blocks = min(10, (self.pool_size_bytes // 4 - offset) // size)
                    for _ in range(num_blocks):
                        if offset + size <= pool_tensor.numel():
                            block = pool_tensor[offset:offset + size]
                            self.free_blocks[size].append(block)
                            offset += size
                
                logger.info(f"Initialized memory pool with {sum(len(blocks) for blocks in self.free_blocks.values())} blocks")
                
            except torch.cuda.OutOfMemoryError:
                logger.warning("Could not initialize full memory pool, using dynamic allocation")
    
    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor from pool."""
        size_needed = np.prod(shape)
        
        with self.lock:
            # Find suitable block
            for block_size, blocks in self.free_blocks.items():
                if block_size >= size_needed and blocks:
                    block = blocks.pop()
                    self.allocation_count += 1
                    
                    # Reshape and return
                    return block[:size_needed].view(shape).contiguous()
            
            # No suitable block found, allocate directly
            try:
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                self.allocation_count += 1
                return tensor
            except torch.cuda.OutOfMemoryError:
                # Force garbage collection and retry
                self.cleanup()
                return torch.empty(shape, dtype=dtype, device=self.device)
    
    def deallocate(self, tensor: torch.Tensor):
        """Return tensor to pool."""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        size = tensor.numel()
        
        with self.lock:
            # Find appropriate size bucket
            target_size = 1
            while target_size < size:
                target_size *= 4
            
            if len(self.free_blocks[target_size]) < 20:  # Limit pool size
                self.free_blocks[target_size].append(tensor)
                self.deallocation_count += 1
    
    def cleanup(self):
        """Clean up memory pool."""
        with self.lock:
            self.free_blocks.clear()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_blocks = sum(len(blocks) for blocks in self.free_blocks.values())
        
        return {
            'total_free_blocks': total_blocks,
            'allocations': self.allocation_count,
            'deallocations': self.deallocation_count,
            'efficiency': self.deallocation_count / max(self.allocation_count, 1),
            'block_sizes': {size: len(blocks) for size, blocks in self.free_blocks.items()}
        }


class DynamicBatchOptimizer:
    """Optimizes batch sizes dynamically based on memory and performance."""
    
    def __init__(self, device: torch.device, memory_manager: 'AdvancedMemoryManager'):
        self.device = device
        self.memory_manager = memory_manager
        self.history = deque(maxlen=100)
        self.current_optimal = 32
        self.last_optimization = 0
        self.optimization_interval = 60  # seconds
        
    def find_optimal_batch_size(self, model: torch.nn.Module, 
                               sample_input: torch.Tensor,
                               target_latency_ms: float = 10.0,
                               max_memory_usage: float = 0.8) -> BatchOptimizationResult:
        """Find optimal batch size for given constraints."""
        logger.info(f"Optimizing batch size with target latency: {target_latency_ms}ms")
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        results = []
        
        model.eval()
        
        for batch_size in batch_sizes:
            try:
                # Create test batch
                test_input = sample_input.repeat(batch_size, 1).to(self.device)
                
                # Check memory requirements
                memory_before = self.memory_manager.get_memory_stats()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Measure performance
                latencies = []
                start_total = time.perf_counter()
                
                for _ in range(20):
                    start = time.perf_counter()
                    
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    latencies.append((time.perf_counter() - start) * 1000)
                
                total_time = time.perf_counter() - start_total
                throughput = (20 * batch_size) / total_time
                
                memory_after = self.memory_manager.get_memory_stats()
                memory_used = memory_after.allocated_mb - memory_before.allocated_mb
                
                p95_latency = np.percentile(latencies, 95)
                avg_latency = np.mean(latencies)
                
                # Check constraints
                memory_ok = (memory_after.allocated_mb / memory_after.total_mb) < max_memory_usage
                latency_ok = p95_latency < target_latency_ms
                
                results.append({
                    'batch_size': batch_size,
                    'throughput': throughput,
                    'p95_latency': p95_latency,
                    'avg_latency': avg_latency,
                    'memory_mb': memory_used,
                    'memory_ok': memory_ok,
                    'latency_ok': latency_ok,
                    'valid': memory_ok and latency_ok
                })
                
                logger.debug(f"Batch {batch_size}: {throughput:.1f} ops/s, "
                           f"{p95_latency:.1f}ms p95, {memory_used:.1f}MB")
                
            except torch.cuda.OutOfMemoryError:
                logger.debug(f"OOM at batch size {batch_size}")
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                continue
        
        # Find optimal batch size
        valid_results = [r for r in results if r['valid']]
        
        if not valid_results:
            # Use largest that fits in memory
            memory_ok_results = [r for r in results if r['memory_ok']]
            if memory_ok_results:
                optimal = max(memory_ok_results, key=lambda x: x['throughput'])
                recommendation = "Using largest batch that fits in memory (latency constraint not met)"
            else:
                optimal = results[0] if results else {'batch_size': 1, 'throughput': 0}
                recommendation = "Using minimum batch size due to memory constraints"
        else:
            # Choose based on throughput among valid options
            optimal = max(valid_results, key=lambda x: x['throughput'])
            recommendation = "Optimal batch size found"
        
        self.current_optimal = optimal['batch_size']
        
        return BatchOptimizationResult(
            optimal_batch_size=optimal['batch_size'],
            max_throughput=optimal.get('throughput', 0),
            memory_efficiency=optimal.get('memory_mb', 0) / optimal['batch_size'],
            latency_p95=optimal.get('p95_latency', 0),
            recommendation=recommendation
        )
    
    def should_reoptimize(self) -> bool:
        """Check if batch size should be reoptimized."""
        current_time = time.time()
        
        # Time-based reoptimization
        if current_time - self.last_optimization > self.optimization_interval:
            return True
        
        # Performance degradation detection
        if len(self.history) >= 10:
            recent_throughput = np.mean([h['throughput'] for h in list(self.history)[-5:]])
            historical_throughput = np.mean([h['throughput'] for h in list(self.history)[-20:-5]])
            
            if recent_throughput < historical_throughput * 0.8:  # 20% degradation
                logger.info("Performance degradation detected, reoptimizing batch size")
                return True
        
        return False
    
    def record_performance(self, batch_size: int, throughput: float, latency: float, memory_mb: float):
        """Record performance metrics for future optimization."""
        self.history.append({
            'timestamp': time.time(),
            'batch_size': batch_size,
            'throughput': throughput,
            'latency': latency,
            'memory_mb': memory_mb
        })


class AdvancedMemoryManager:
    """Advanced GPU memory manager with dynamic optimization."""
    
    def __init__(self, device: Optional[torch.device] = None, pool_size_mb: int = 1024):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_pool = MemoryPool(pool_size_mb, self.device) if self.device.type == 'cuda' else None
        self.batch_optimizer = DynamicBatchOptimizer(self.device, self)
        
        # Memory monitoring
        self.allocation_history = deque(maxlen=1000)
        self.gc_threshold = 0.85  # Trigger GC at 85% memory usage
        self.last_gc_time = 0
        self.gc_interval = 30  # Minimum seconds between GC
        
        # Performance tracking
        self.peak_memory_mb = 0
        self.fragmentation_warnings = 0
        
        logger.info(f"Advanced memory manager initialized for {self.device}")
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate tensor with optimal memory management."""
        if self.memory_pool and self.device.type == 'cuda':
            tensor = self.memory_pool.allocate(shape, dtype)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        
        # Record allocation
        self.allocation_history.append({
            'timestamp': time.time(),
            'size_mb': tensor.numel() * tensor.element_size() / 1024**2,
            'shape': shape,
            'dtype': dtype
        })
        
        # Check if GC needed
        self._maybe_garbage_collect()
        
        return tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Deallocate tensor efficiently."""
        if self.memory_pool and self.device.type == 'cuda':
            self.memory_pool.deallocate(tensor)
        else:
            del tensor
    
    @contextmanager
    def managed_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32):
        """Context manager for automatic tensor cleanup."""
        tensor = self.allocate_tensor(shape, dtype)
        try:
            yield tensor
        finally:
            self.deallocate_tensor(tensor)
    
    def optimize_batch_size(self, model: torch.nn.Module, 
                           sample_input: torch.Tensor,
                           target_latency_ms: float = 10.0) -> BatchOptimizationResult:
        """Optimize batch size for given model and constraints."""
        return self.batch_optimizer.find_optimal_batch_size(
            model, sample_input, target_latency_ms
        )
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            cached = torch.cuda.memory_reserved(self.device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            
            # Calculate fragmentation
            fragmentation = (cached - allocated) / max(cached, 1)
        else:
            mem = psutil.virtual_memory()
            allocated = (mem.total - mem.available) / 1024**2
            cached = allocated  # No separate cache on CPU
            max_allocated = allocated
            total = mem.total / 1024**2
            fragmentation = 0
        
        # Pool efficiency
        pool_efficiency = 0
        if self.memory_pool:
            pool_stats = self.memory_pool.get_stats()
            pool_efficiency = pool_stats.get('efficiency', 0)
        
        return MemoryStats(
            allocated_mb=allocated,
            cached_mb=cached,
            max_allocated_mb=max_allocated,
            total_mb=total,
            fragmentation_ratio=fragmentation,
            pool_efficiency=pool_efficiency,
            gc_count=len(self.allocation_history)
        )
    
    def _maybe_garbage_collect(self):
        """Trigger garbage collection if needed."""
        stats = self.get_memory_stats()
        current_time = time.time()
        
        # Check memory pressure
        memory_pressure = stats.allocated_mb / stats.total_mb
        
        if (memory_pressure > self.gc_threshold and 
            current_time - self.last_gc_time > self.gc_interval):
            
            logger.debug(f"Triggering GC: memory usage {memory_pressure:.1%}")
            self._force_garbage_collect()
            self.last_gc_time = current_time
    
    def _force_garbage_collect(self):
        """Force garbage collection and cache cleanup."""
        gc.collect()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if self.memory_pool:
            self.memory_pool.cleanup()
    
    def defragment_memory(self):
        """Defragment GPU memory."""
        if self.device.type == 'cuda':
            logger.info("Defragmenting GPU memory...")
            
            # Force cleanup
            self._force_garbage_collect()
            
            # Reset peak memory tracking
            torch.cuda.reset_peak_memory_stats(self.device)
            
            # Recreate memory pool
            if self.memory_pool:
                pool_size = self.memory_pool.pool_size_bytes // (1024 * 1024)
                self.memory_pool = MemoryPool(pool_size, self.device)
            
            logger.info("Memory defragmentation complete")
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations."""
        stats = self.get_memory_stats()
        recommendations = []
        
        # Memory usage recommendations
        if stats.allocated_mb / stats.total_mb > 0.9:
            recommendations.append("Memory usage >90%, consider reducing batch size")
        elif stats.allocated_mb / stats.total_mb < 0.5:
            recommendations.append("Memory usage <50%, consider increasing batch size")
        
        # Fragmentation recommendations
        if stats.fragmentation_ratio > 0.3:
            recommendations.append("High memory fragmentation, consider defragmentation")
            self.fragmentation_warnings += 1
        
        # Pool efficiency recommendations
        if stats.pool_efficiency < 0.8:
            recommendations.append("Low memory pool efficiency, consider pool size adjustment")
        
        # Performance recommendations
        if len(self.allocation_history) > 500:
            recent_allocations = list(self.allocation_history)[-100:]
            allocation_rate = len(recent_allocations) / 60  # per minute
            
            if allocation_rate > 50:
                recommendations.append("High allocation rate, consider tensor reuse")
        
        return {
            'memory_stats': stats,
            'recommendations': recommendations,
            'should_defragment': stats.fragmentation_ratio > 0.4,
            'should_reoptimize_batch': self.batch_optimizer.should_reoptimize(),
            'fragmentation_warnings': self.fragmentation_warnings
        }
    
    def create_optimized_dataloader(self, dataset, 
                                   target_latency_ms: float = 10.0,
                                   **kwargs) -> torch.utils.data.DataLoader:
        """Create DataLoader with optimized batch size and memory settings."""
        # Optimize batch size if sample available
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, torch.Tensor):
                sample_input = sample.unsqueeze(0)
            else:
                sample_input = torch.tensor(sample).unsqueeze(0)
            
            if hasattr(self, 'model'):  # If model is available
                opt_result = self.optimize_batch_size(self.model, sample_input, target_latency_ms)
                kwargs['batch_size'] = opt_result.optimal_batch_size
                logger.info(f"Optimized batch size: {opt_result.optimal_batch_size}")
        
        # Set memory optimization parameters
        kwargs.setdefault('pin_memory', self.device.type == 'cuda')
        kwargs.setdefault('num_workers', min(4, psutil.cpu_count()))
        kwargs.setdefault('persistent_workers', True)
        kwargs.setdefault('drop_last', True)  # For consistent batch sizes
        
        return torch.utils.data.DataLoader(dataset, **kwargs)
    
    def monitor_memory_continuously(self, interval_seconds: int = 60):
        """Start continuous memory monitoring (for production use)."""
        import threading
        
        def monitor():
            while True:
                try:
                    recommendations = self.get_optimization_recommendations()
                    
                    if recommendations['should_defragment']:
                        logger.warning("Memory fragmentation detected, consider defragmentation")
                    
                    if recommendations['recommendations']:
                        logger.info("Memory recommendations: " + 
                                  ", ".join(recommendations['recommendations']))
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Continuous memory monitoring started")
    
    def cleanup(self):
        """Cleanup all memory resources."""
        if self.memory_pool:
            self.memory_pool.cleanup()
        
        self.allocation_history.clear()
        self._force_garbage_collect()
        
        logger.info("Memory manager cleanup complete")


# Convenience functions
def create_optimized_memory_manager(device: Optional[torch.device] = None,
                                   pool_size_mb: int = 1024) -> AdvancedMemoryManager:
    """Create optimized memory manager with default settings."""
    return AdvancedMemoryManager(device, pool_size_mb)


def auto_optimize_batch_size(model: torch.nn.Module,
                            sample_input: torch.Tensor,
                            target_latency_ms: float = 10.0) -> int:
    """Auto-optimize batch size for given model and constraints."""
    memory_manager = create_optimized_memory_manager()
    result = memory_manager.optimize_batch_size(model, sample_input, target_latency_ms)
    return result.optimal_batch_size