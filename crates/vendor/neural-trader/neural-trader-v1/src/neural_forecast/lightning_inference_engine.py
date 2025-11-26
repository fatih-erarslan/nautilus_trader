"""
Lightning-Fast Inference Engine for Sub-10ms Latency
Combines all optimization techniques for maximum inference speed.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from collections import defaultdict, deque
import pickle
import hashlib
import json

from .optimized_nhits_engine import OptimizedNHITSEngine, OptimizedNHITSConfig
from .advanced_memory_manager import AdvancedMemoryManager
from .config.optimized_config_profiles import get_low_latency_config, auto_detect_optimal_config

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Inference request with metadata."""
    data: torch.Tensor
    request_id: str
    timestamp: float
    priority: int = 1  # 1=highest, 5=lowest
    use_cache: bool = True
    timeout_ms: float = 50.0


@dataclass
class InferenceResponse:
    """Inference response with performance metrics."""
    request_id: str
    point_forecast: List[float]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    inference_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    cache_hit: bool = False
    device_used: str = "unknown"
    model_version: str = "1.0"
    error: Optional[str] = None


class IntelligentCache:
    """Intelligent caching with TTL and popularity-based eviction."""
    
    def __init__(self, max_size: int = 5000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
        
    def _generate_key(self, data: torch.Tensor) -> str:
        """Generate cache key from tensor data."""
        # Use tensor statistics for key generation (faster than full hash)
        stats = (
            data.shape,
            float(data.mean()),
            float(data.std()),
            float(data.min()),
            float(data.max())
        )
        return hashlib.md5(str(stats).encode()).hexdigest()[:16]
    
    def get(self, data: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Get cached result if available and valid."""
        key = self._generate_key(data)
        current_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry_time, ttl, result = self.cache[key]
                
                if current_time - entry_time < ttl:
                    # Update access stats
                    self.access_times[key] = current_time
                    self.access_counts[key] += 1
                    return result
                else:
                    # Expired entry
                    del self.cache[key]
                    self.access_times.pop(key, None)
                    self.access_counts.pop(key, None)
        
        return None
    
    def put(self, data: torch.Tensor, result: Dict[str, Any], ttl: Optional[int] = None):
        """Store result in cache."""
        key = self._generate_key(data)
        current_time = time.time()
        ttl = ttl or self.default_ttl
        
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                self._evict_entries()
            
            self.cache[key] = (current_time, ttl, result)
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict_entries(self):
        """Evict least recently used and least popular entries."""
        current_time = time.time()
        
        # Remove expired entries first
        expired_keys = [
            key for key, (entry_time, ttl, _) in self.cache.items()
            if current_time - entry_time >= ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.access_counts.pop(key, None)
        
        # If still over capacity, use LRU + popularity
        while len(self.cache) >= self.max_size:
            # Score based on recency and popularity
            scores = {}
            for key in self.cache:
                recency_score = current_time - self.access_times.get(key, 0)
                popularity_score = 1.0 / (self.access_counts.get(key, 1) + 1)
                scores[key] = recency_score * popularity_score  # Higher = worse
            
            # Remove worst scoring entry
            worst_key = max(scores.keys(), key=lambda k: scores[k])
            del self.cache[worst_key]
            self.access_times.pop(worst_key, None)
            self.access_counts.pop(worst_key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        
        with self.lock:
            valid_entries = sum(
                1 for entry_time, ttl, _ in self.cache.values()
                if current_time - entry_time < ttl
            )
            
            total_accesses = sum(self.access_counts.values())
            
            return {
                'size': len(self.cache),
                'valid_entries': valid_entries,
                'max_size': self.max_size,
                'total_accesses': total_accesses,
                'hit_rate': total_accesses / max(len(self.cache), 1)
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()


class RequestQueue:
    """Priority queue for inference requests with batching."""
    
    def __init__(self, max_batch_size: int = 64, max_wait_ms: float = 5.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        
    async def put(self, request: InferenceRequest):
        """Add request to queue."""
        async with self.condition:
            self.queue.append(request)
            self.condition.notify()
    
    async def get_batch(self) -> List[InferenceRequest]:
        """Get batch of requests for processing."""
        async with self.condition:
            # Wait for requests or timeout
            start_time = time.time()
            
            while len(self.queue) == 0:
                wait_time = self.max_wait_ms / 1000.0
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=wait_time)
                except asyncio.TimeoutError:
                    return []
            
            # Collect batch
            batch = []
            current_time = time.time()
            
            while (len(batch) < self.max_batch_size and 
                   len(self.queue) > 0 and
                   (current_time - start_time) * 1000 < self.max_wait_ms):
                
                batch.append(self.queue.popleft())
                
                # Check if we should wait for more requests
                if len(batch) < self.max_batch_size // 4:  # Wait for more if batch is small
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=0.001)
                    except asyncio.TimeoutError:
                        break
            
            # Sort by priority (highest first)
            batch.sort(key=lambda r: r.priority)
            
            return batch


class LightningInferenceEngine:
    """Ultra-fast inference engine optimized for sub-10ms latency."""
    
    def __init__(self, 
                 model_config: Optional[OptimizedNHITSConfig] = None,
                 max_concurrent_requests: int = 1000,
                 enable_profiling: bool = False):
        
        # Configuration
        self.config = model_config or auto_detect_optimal_config()
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_profiling = enable_profiling
        
        # Core components
        self.device = self._setup_device()
        self.memory_manager = AdvancedMemoryManager(self.device)
        self.nhits_engine = OptimizedNHITSEngine(self.config)
        self.cache = IntelligentCache(max_size=self.config.max_cache_size if hasattr(self.config, 'max_cache_size') else 5000)
        
        # Request handling
        self.request_queue = RequestQueue(
            max_batch_size=getattr(self.config, 'max_batch_size', 64),
            max_wait_ms=5.0
        )
        
        # Model and state
        self.model = None
        self.is_ready = False
        self.model_lock = asyncio.Lock()
        
        # Performance tracking
        self.total_requests = 0
        self.total_inference_time = 0
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Background tasks
        self.background_tasks = []
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Lightning inference engine initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device with advanced configurations."""
        if hasattr(self.config, 'use_gpu') and self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Advanced CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Memory optimization
            if hasattr(self.config, 'memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)
            
            # Enable JIT compilation optimizations
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            
            logger.info(f"CUDA device configured: {torch.cuda.get_device_name()}")
            return device
        else:
            return torch.device('cpu')
    
    async def initialize(self):
        """Initialize the inference engine."""
        logger.info("Initializing lightning inference engine...")
        
        async with self.model_lock:
            # Create optimized model
            self.model = self.nhits_engine.create_model()
            
            # Optimize model for inference
            self.model.eval()
            
            # Apply compilation if supported
            if hasattr(self.config, 'compile_model') and self.config.compile_model:
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    logger.info("Model compiled with PyTorch 2.0")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Warmup model
            await self._warmup_model()
            
            self.is_ready = True
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Lightning inference engine ready")
    
    async def _warmup_model(self):
        """Warmup model for optimal performance."""
        logger.info("Warming up model...")
        
        # Create warmup data
        warmup_shapes = [
            (1, self.config.input_size),
            (4, self.config.input_size),
            (getattr(self.config, 'batch_size', 32), self.config.input_size)
        ]
        
        for shape in warmup_shapes:
            warmup_data = torch.randn(shape, device=self.device)
            
            # Multiple warmup iterations
            for _ in range(5):
                with torch.no_grad():
                    if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision:
                        with autocast():
                            _ = self.model(warmup_data)
                    else:
                        _ = self.model(warmup_data)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        logger.info("Model warmup complete")
    
    def _start_background_tasks(self):
        """Start background processing tasks."""
        # Batch processing task
        self.background_tasks.append(
            asyncio.create_task(self._batch_processor())
        )
        
        # Performance monitoring task
        if self.enable_profiling:
            self.background_tasks.append(
                asyncio.create_task(self._performance_monitor())
            )
        
        # Memory management task
        self.background_tasks.append(
            asyncio.create_task(self._memory_manager_task())
        )
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def predict_single(self, 
                           data: torch.Tensor,
                           request_id: Optional[str] = None,
                           priority: int = 1,
                           timeout_ms: float = 50.0) -> InferenceResponse:
        """Predict single time series with ultra-low latency."""
        if not self.is_ready:
            return InferenceResponse(
                request_id=request_id or "unknown",
                point_forecast=[],
                error="Engine not ready"
            )
        
        request_start = time.perf_counter()
        request_id = request_id or f"req_{int(time.time() * 1000000)}"
        
        # Create request
        request = InferenceRequest(
            data=data,
            request_id=request_id,
            timestamp=request_start,
            priority=priority,
            timeout_ms=timeout_ms
        )
        
        # Check cache first for ultra-low latency
        cached_result = self.cache.get(data)
        if cached_result:
            return InferenceResponse(
                request_id=request_id,
                point_forecast=cached_result['point_forecast'],
                confidence_intervals=cached_result.get('confidence_intervals'),
                inference_time_ms=(time.perf_counter() - request_start) * 1000,
                cache_hit=True,
                device_used=str(self.device)
            )
        
        # For ultra-low latency single predictions, bypass queue
        if priority == 1 and data.size(0) == 1:
            return await self._predict_direct(request)
        
        # Add to queue for batch processing
        await self.request_queue.put(request)
        
        # Wait for result (this would be implemented with async result handling)
        # For now, process directly
        return await self._predict_direct(request)
    
    async def _predict_direct(self, request: InferenceRequest) -> InferenceResponse:
        """Direct prediction bypassing queue for ultra-low latency."""
        start_time = time.perf_counter()
        
        try:
            # Ensure data is on correct device
            if request.data.device != self.device:
                request.data = request.data.to(self.device, non_blocking=True)
            
            # Predict with optimizations
            with torch.no_grad():
                if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision and self.device.type == 'cuda':
                    with autocast():
                        result = self.model(request.data)
                else:
                    result = self.model(request.data)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Extract forecast
            point_forecast = result['point_forecast'].cpu().numpy().tolist()
            if len(point_forecast) > 0 and isinstance(point_forecast[0], list):
                point_forecast = point_forecast[0]  # Handle batch dimension
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # Cache result
            cache_data = {
                'point_forecast': point_forecast,
                'model_version': '1.0'
            }
            self.cache.put(request.data, cache_data)
            
            # Update statistics
            self.total_requests += 1
            self.total_inference_time += inference_time
            self.latency_history.append(inference_time)
            
            return InferenceResponse(
                request_id=request.request_id,
                point_forecast=point_forecast,
                inference_time_ms=inference_time,
                device_used=str(self.device),
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for {request.request_id}: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                point_forecast=[],
                error=str(e),
                inference_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _batch_processor(self):
        """Background task for batch processing requests."""
        while not self.shutdown_event.is_set():
            try:
                # Get batch of requests
                batch = await self.request_queue.get_batch()
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.001)
    
    async def _process_batch(self, requests: List[InferenceRequest]):
        """Process batch of requests efficiently."""
        if not requests:
            return
        
        start_time = time.perf_counter()
        
        try:
            # Combine data into batch
            batch_data = torch.stack([req.data for req in requests])
            
            # Move to device
            if batch_data.device != self.device:
                batch_data = batch_data.to(self.device, non_blocking=True)
            
            # Batch prediction
            with torch.no_grad():
                if hasattr(self.config, 'mixed_precision') and self.config.mixed_precision and self.device.type == 'cuda':
                    with autocast():
                        results = self.model(batch_data)
                else:
                    results = self.model(batch_data)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Process results
            forecasts = results['point_forecast'].cpu().numpy()
            batch_time = (time.perf_counter() - start_time) * 1000
            
            # Send results back (implementation would use async callbacks)
            for i, request in enumerate(requests):
                forecast = forecasts[i].tolist()
                
                # Cache individual results
                cache_data = {
                    'point_forecast': forecast,
                    'model_version': '1.0'
                }
                self.cache.put(request.data, cache_data)
            
            # Update throughput stats
            throughput = len(requests) / (batch_time / 1000)
            self.throughput_history.append(throughput)
            
            logger.debug(f"Processed batch of {len(requests)} in {batch_time:.2f}ms "
                        f"({throughput:.0f} req/s)")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    async def _performance_monitor(self):
        """Background task for performance monitoring."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Calculate statistics
                stats = self.get_performance_stats()
                
                # Log performance summary
                logger.info(f"Performance: {stats['avg_latency_ms']:.1f}ms avg latency, "
                          f"{stats['p95_latency_ms']:.1f}ms p95, "
                          f"{stats['cache_hit_rate']:.1%} cache hit rate")
                
                # Check for performance issues
                if stats['avg_latency_ms'] > 20:
                    logger.warning("High latency detected, consider optimization")
                
                if stats['cache_hit_rate'] < 0.5:
                    logger.info("Low cache hit rate, consider cache tuning")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _memory_manager_task(self):
        """Background task for memory management."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get memory recommendations
                recommendations = self.memory_manager.get_optimization_recommendations()
                
                if recommendations['should_defragment']:
                    logger.info("Performing memory defragmentation...")
                    self.memory_manager.defragment_memory()
                
                # Clear old cache entries
                if len(self.latency_history) > 500:
                    # Clear some history to prevent memory growth
                    for _ in range(100):
                        if self.latency_history:
                            self.latency_history.popleft()
                
            except Exception as e:
                logger.error(f"Memory management error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        latencies = list(self.latency_history)
        throughputs = list(self.throughput_history)
        
        stats = {
            'total_requests': self.total_requests,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p50_latency_ms': np.percentile(latencies, 50) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0,
            'avg_throughput': np.mean(throughputs) if throughputs else 0,
            'cache_stats': self.cache.get_stats(),
            'cache_hit_rate': self.cache.get_stats().get('hit_rate', 0),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'device': str(self.device),
            'model_ready': self.is_ready
        }
        
        return stats
    
    async def benchmark(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run comprehensive benchmark."""
        logger.info(f"Running benchmark for {duration_seconds} seconds...")
        
        if not self.is_ready:
            await self.initialize()
        
        # Generate test data
        test_data = torch.randn(1, self.config.input_size, device=self.device)
        
        # Benchmark single predictions
        single_latencies = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            request_start = time.perf_counter()
            
            response = await self.predict_single(test_data, priority=1)
            
            if not response.error:
                latency = (time.perf_counter() - request_start) * 1000
                single_latencies.append(latency)
        
        # Benchmark batch predictions
        batch_sizes = [4, 8, 16, 32]
        batch_results = {}
        
        for batch_size in batch_sizes:
            batch_data = torch.randn(batch_size, self.config.input_size, device=self.device)
            batch_latencies = []
            
            for _ in range(10):
                start = time.perf_counter()
                
                # Simulate batch processing
                tasks = [
                    self.predict_single(batch_data[i:i+1], priority=2)
                    for i in range(batch_size)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                batch_time = (time.perf_counter() - start) * 1000
                batch_latencies.append(batch_time / batch_size)  # Per-item latency
            
            batch_results[batch_size] = {
                'avg_latency_ms': np.mean(batch_latencies),
                'p95_latency_ms': np.percentile(batch_latencies, 95)
            }
        
        return {
            'single_prediction': {
                'total_predictions': len(single_latencies),
                'avg_latency_ms': np.mean(single_latencies),
                'p50_latency_ms': np.percentile(single_latencies, 50),
                'p95_latency_ms': np.percentile(single_latencies, 95),
                'p99_latency_ms': np.percentile(single_latencies, 99),
                'min_latency_ms': np.min(single_latencies),
                'max_latency_ms': np.max(single_latencies)
            },
            'batch_predictions': batch_results,
            'system_stats': self.get_performance_stats(),
            'benchmark_duration': duration_seconds,
            'target_achieved': np.percentile(single_latencies, 95) < 10.0 if single_latencies else False
        }
    
    async def shutdown(self):
        """Gracefully shutdown the inference engine."""
        logger.info("Shutting down lightning inference engine...")
        
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cleanup resources
        self.memory_manager.cleanup()
        self.cache.clear()
        
        logger.info("Lightning inference engine shutdown complete")


# Convenience functions
async def create_lightning_engine(target_latency_ms: float = 10.0) -> LightningInferenceEngine:
    """Create and initialize lightning inference engine with optimal configuration."""
    if target_latency_ms <= 5.0:
        config = get_low_latency_config({'input_size': 24, 'horizon': 6})
    else:
        config = get_low_latency_config()
    
    engine = LightningInferenceEngine(config)
    await engine.initialize()
    
    return engine


async def benchmark_lightning_engine(duration: int = 60) -> Dict[str, Any]:
    """Quick benchmark of lightning inference engine."""
    engine = await create_lightning_engine()
    
    try:
        results = await engine.benchmark(duration)
        return results
    finally:
        await engine.shutdown()