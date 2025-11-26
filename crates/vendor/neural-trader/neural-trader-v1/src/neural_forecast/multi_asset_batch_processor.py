"""
Multi-Asset Batch Processor for Optimized Forecasting
Handles large-scale parallel processing of multiple financial assets with advanced optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from queue import PriorityQueue, Queue
import uuid
from enum import Enum
import pickle
import json

from .lightning_inference_engine import LightningInferenceEngine, InferenceRequest, InferenceResponse
from .advanced_memory_manager import AdvancedMemoryManager
from .mixed_precision_optimizer import MixedPrecisionOptimizer

logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Processing priority levels."""
    CRITICAL = 1    # Real-time trading signals
    HIGH = 2        # Portfolio rebalancing
    NORMAL = 3      # Regular forecasting
    LOW = 4         # Research and backtesting
    BACKGROUND = 5  # Batch analytics


class AssetClass(Enum):
    """Asset class categorization for optimization."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    DERIVATIVE = "derivative"


@dataclass
class AssetMetadata:
    """Metadata for asset-specific optimization."""
    symbol: str
    asset_class: AssetClass
    market_hours: Tuple[int, int]  # Trading hours in UTC
    volatility_regime: str = "normal"  # low, normal, high, extreme
    liquidity_tier: int = 1  # 1=most liquid, 5=least liquid
    correlation_group: Optional[str] = None
    last_updated: float = 0.0


@dataclass
class BatchRequest:
    """Batch processing request."""
    request_id: str
    assets: List[str]
    data: Dict[str, torch.Tensor]
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Batch processing result."""
    request_id: str
    forecasts: Dict[str, Dict[str, Any]]
    processing_time: float
    assets_processed: int
    errors: Dict[str, str] = field(default_factory=dict)
    cache_hits: int = 0
    memory_usage_mb: float = 0.0
    throughput_per_second: float = 0.0


class AdaptiveBatchSizer:
    """Dynamically optimizes batch sizes for different asset groups."""
    
    def __init__(self, memory_manager: AdvancedMemoryManager):
        self.memory_manager = memory_manager
        self.performance_history = defaultdict(list)
        self.optimal_sizes = defaultdict(lambda: 32)
        self.last_optimization = defaultdict(float)
        
    def get_optimal_batch_size(self, 
                              asset_class: AssetClass,
                              data_shape: Tuple[int, ...],
                              target_latency_ms: float = 100.0) -> int:
        """Get optimal batch size for asset class and data shape."""
        key = (asset_class.value, data_shape)
        
        # Use cached optimal size if recent
        if time.time() - self.last_optimization[key] < 300:  # 5 minutes
            return self.optimal_sizes[key]
        
        # Test different batch sizes
        test_sizes = [8, 16, 32, 64, 128, 256]
        best_size = 32
        best_throughput = 0
        
        for batch_size in test_sizes:
            try:
                # Estimate memory requirements
                memory_estimate = self._estimate_memory_usage(batch_size, data_shape)
                memory_stats = self.memory_manager.get_memory_stats()
                
                if memory_estimate > memory_stats.total_mb * 0.8:  # Skip if likely OOM
                    continue
                
                # Estimate throughput based on historical data
                throughput = self._estimate_throughput(key, batch_size)
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_size = batch_size
                    
            except Exception as e:
                logger.debug(f"Error testing batch size {batch_size}: {e}")
                continue
        
        self.optimal_sizes[key] = best_size
        self.last_optimization[key] = time.time()
        
        return best_size
    
    def _estimate_memory_usage(self, batch_size: int, data_shape: Tuple[int, ...]) -> float:
        """Estimate memory usage for batch size and data shape."""
        elements = batch_size * np.prod(data_shape)
        bytes_per_element = 4  # float32
        overhead_factor = 3.0  # Model, gradients, optimizer states
        
        return (elements * bytes_per_element * overhead_factor) / (1024 * 1024)  # MB
    
    def _estimate_throughput(self, key: Tuple, batch_size: int) -> float:
        """Estimate throughput based on historical performance."""
        history = self.performance_history[key]
        
        if not history:
            # Use simple heuristic for new keys
            return batch_size * 10  # Rough estimate
        
        # Find similar batch sizes in history
        similar_perf = [
            (perf['throughput'], perf['batch_size']) 
            for perf in history[-20:]  # Last 20 measurements
            if abs(perf['batch_size'] - batch_size) <= 16
        ]
        
        if similar_perf:
            # Average throughput for similar batch sizes
            return np.mean([throughput for throughput, _ in similar_perf])
        
        # Linear interpolation from existing data
        if len(history) >= 2:
            sorted_history = sorted(history[-10:], key=lambda x: x['batch_size'])
            
            # Simple linear interpolation
            x_values = [h['batch_size'] for h in sorted_history]
            y_values = [h['throughput'] for h in sorted_history]
            
            return np.interp(batch_size, x_values, y_values)
        
        return batch_size * 10  # Fallback estimate
    
    def record_performance(self, 
                          asset_class: AssetClass,
                          data_shape: Tuple[int, ...],
                          batch_size: int,
                          throughput: float,
                          latency: float,
                          memory_usage: float):
        """Record performance metrics for future optimization."""
        key = (asset_class.value, data_shape)
        
        self.performance_history[key].append({
            'timestamp': time.time(),
            'batch_size': batch_size,
            'throughput': throughput,
            'latency': latency,
            'memory_usage': memory_usage
        })
        
        # Limit history size
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-50:]


class AssetGroupOptimizer:
    """Optimizes processing by grouping similar assets."""
    
    def __init__(self):
        self.asset_metadata = {}
        self.correlation_matrix = {}
        self.grouping_cache = {}
        self.cache_timeout = 3600  # 1 hour
        
    def register_asset(self, metadata: AssetMetadata):
        """Register asset metadata for optimization."""
        self.asset_metadata[metadata.symbol] = metadata
        
    def group_assets_for_processing(self, 
                                   assets: List[str],
                                   max_group_size: int = 64) -> List[List[str]]:
        """Group assets optimally for batch processing."""
        if len(assets) <= max_group_size:
            return [assets]
        
        # Check cache
        cache_key = tuple(sorted(assets))
        if (cache_key in self.grouping_cache and 
            time.time() - self.grouping_cache[cache_key]['timestamp'] < self.cache_timeout):
            return self.grouping_cache[cache_key]['groups']
        
        # Group by asset class first
        class_groups = defaultdict(list)
        for asset in assets:
            metadata = self.asset_metadata.get(asset)
            if metadata:
                class_groups[metadata.asset_class].append(asset)
            else:
                class_groups[AssetClass.EQUITY].append(asset)  # Default
        
        # Further group by correlation if available
        optimized_groups = []
        for asset_class, class_assets in class_groups.items():
            if len(class_assets) <= max_group_size:
                optimized_groups.append(class_assets)
            else:
                # Split large groups using correlation-based clustering
                sub_groups = self._cluster_by_correlation(class_assets, max_group_size)
                optimized_groups.extend(sub_groups)
        
        # Cache result
        self.grouping_cache[cache_key] = {
            'groups': optimized_groups,
            'timestamp': time.time()
        }
        
        return optimized_groups
    
    def _cluster_by_correlation(self, assets: List[str], max_size: int) -> List[List[str]]:
        """Cluster assets by correlation for efficient batch processing."""
        if len(assets) <= max_size:
            return [assets]
        
        # Simple clustering: group by correlation group if available
        correlation_groups = defaultdict(list)
        ungrouped = []
        
        for asset in assets:
            metadata = self.asset_metadata.get(asset)
            if metadata and metadata.correlation_group:
                correlation_groups[metadata.correlation_group].append(asset)
            else:
                ungrouped.append(asset)
        
        # Create groups
        groups = []
        
        # Add correlation groups
        for group_assets in correlation_groups.values():
            if len(group_assets) <= max_size:
                groups.append(group_assets)
            else:
                # Split large correlation groups
                for i in range(0, len(group_assets), max_size):
                    groups.append(group_assets[i:i + max_size])
        
        # Add ungrouped assets
        for i in range(0, len(ungrouped), max_size):
            groups.append(ungrouped[i:i + max_size])
        
        return groups
    
    def get_processing_order(self, groups: List[List[str]]) -> List[List[str]]:
        """Determine optimal processing order for asset groups."""
        # Sort by priority: liquidity, market hours, volatility
        def group_priority(group):
            priorities = []
            for asset in group:
                metadata = self.asset_metadata.get(asset)
                if metadata:
                    # Higher priority for more liquid assets
                    liquidity_score = 6 - metadata.liquidity_tier
                    
                    # Higher priority for high volatility (more urgent)
                    volatility_score = {
                        'extreme': 4, 'high': 3, 'normal': 2, 'low': 1
                    }.get(metadata.volatility_regime, 2)
                    
                    priorities.append(liquidity_score + volatility_score)
                else:
                    priorities.append(3)  # Default priority
            
            return np.mean(priorities)
        
        return sorted(groups, key=group_priority, reverse=True)


class MultiAssetBatchProcessor:
    """High-performance batch processor for multi-asset forecasting."""
    
    def __init__(self,
                 inference_engine: LightningInferenceEngine,
                 max_concurrent_batches: int = 8,
                 max_queue_size: int = 1000):
        
        self.inference_engine = inference_engine
        self.max_concurrent_batches = max_concurrent_batches
        self.max_queue_size = max_queue_size
        
        # Core components
        self.memory_manager = AdvancedMemoryManager()
        self.precision_optimizer = MixedPrecisionOptimizer()
        self.batch_sizer = AdaptiveBatchSizer(self.memory_manager)
        self.asset_optimizer = AssetGroupOptimizer()
        
        # Request handling
        self.request_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_requests = {}
        self.completed_requests = {}
        
        # Processing state
        self.is_running = False
        self.worker_tasks = []
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_assets_processed': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0
        }
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_batches)
        
        logger.info(f"Multi-asset batch processor initialized with {max_concurrent_batches} workers")
    
    async def start(self):
        """Start the batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_batches):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        # Start monitoring task
        self.worker_tasks.append(asyncio.create_task(self._monitor_loop()))
        
        logger.info(f"Started {len(self.worker_tasks)} worker tasks")
    
    async def stop(self):
        """Stop the batch processor gracefully."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Batch processor stopped")
    
    async def process_batch(self, 
                           assets: List[str],
                           data: Dict[str, torch.Tensor],
                           priority: ProcessingPriority = ProcessingPriority.NORMAL,
                           timeout_seconds: float = 30.0,
                           callback: Optional[Callable] = None) -> str:
        """Submit batch processing request."""
        request_id = str(uuid.uuid4())
        
        # Create batch request
        request = BatchRequest(
            request_id=request_id,
            assets=assets,
            data=data,
            priority=priority,
            timeout_seconds=timeout_seconds,
            callback=callback
        )
        
        # Add to queue
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.request_queue.put,
                (priority.value, time.time(), request)
            )
            
            self.active_requests[request_id] = request
            self.stats['total_requests'] += 1
            
            logger.debug(f"Batch request {request_id} queued with {len(assets)} assets")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to queue batch request: {e}")
            raise
    
    async def get_result(self, request_id: str, timeout: float = None) -> Optional[BatchResult]:
        """Get result for batch request."""
        start_time = time.time()
        
        while request_id not in self.completed_requests:
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.1)
        
        return self.completed_requests.pop(request_id, None)
    
    async def process_batch_sync(self,
                                assets: List[str],
                                data: Dict[str, torch.Tensor],
                                priority: ProcessingPriority = ProcessingPriority.NORMAL) -> BatchResult:
        """Process batch synchronously and return result."""
        request_id = await self.process_batch(assets, data, priority)
        
        result = await self.get_result(request_id, timeout=30.0)
        
        if result is None:
            raise TimeoutError(f"Batch request {request_id} timed out")
        
        return result
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop for processing batch requests."""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get request from queue
                try:
                    priority, timestamp, request = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.request_queue.get(timeout=1.0)
                    )
                except:
                    continue  # Timeout, check if still running
                
                # Process request
                result = await self._process_request(request, worker_id)
                
                # Store result
                self.completed_requests[request.request_id] = result
                self.active_requests.pop(request.request_id, None)
                
                # Call callback if provided
                if request.callback:
                    try:
                        await request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for {request.request_id}: {e}")
                
                # Update stats
                if result.errors:
                    self.stats['failed_requests'] += 1
                else:
                    self.stats['completed_requests'] += 1
                
                self.stats['total_assets_processed'] += result.assets_processed
                self.stats['total_processing_time'] += result.processing_time
                self.stats['cache_hits'] += result.cache_hits
                
                logger.debug(f"Worker {worker_id} completed request {request.request_id}")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_request(self, request: BatchRequest, worker_id: str) -> BatchResult:
        """Process a single batch request."""
        start_time = time.time()
        
        try:
            # Group assets for optimal processing
            asset_groups = self.asset_optimizer.group_assets_for_processing(
                request.assets, 
                max_group_size=64
            )
            
            # Process groups
            all_forecasts = {}
            total_cache_hits = 0
            errors = {}
            
            for group in asset_groups:
                try:
                    # Get group data
                    group_data = {asset: request.data[asset] for asset in group if asset in request.data}
                    
                    if not group_data:
                        continue
                    
                    # Process group
                    group_forecasts, cache_hits = await self._process_asset_group(
                        group, group_data, worker_id
                    )
                    
                    all_forecasts.update(group_forecasts)
                    total_cache_hits += cache_hits
                    
                except Exception as e:
                    logger.error(f"Error processing group {group}: {e}")
                    for asset in group:
                        errors[asset] = str(e)
            
            processing_time = time.time() - start_time
            throughput = len(request.assets) / processing_time if processing_time > 0 else 0
            
            # Get memory usage
            memory_stats = self.memory_manager.get_memory_stats()
            
            return BatchResult(
                request_id=request.request_id,
                forecasts=all_forecasts,
                processing_time=processing_time,
                assets_processed=len(all_forecasts),
                errors=errors,
                cache_hits=total_cache_hits,
                memory_usage_mb=memory_stats.allocated_mb,
                throughput_per_second=throughput
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed for {request.request_id}: {e}")
            
            return BatchResult(
                request_id=request.request_id,
                forecasts={},
                processing_time=time.time() - start_time,
                assets_processed=0,
                errors={asset: str(e) for asset in request.assets}
            )
    
    async def _process_asset_group(self, 
                                  assets: List[str],
                                  data: Dict[str, torch.Tensor],
                                  worker_id: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """Process a group of similar assets efficiently."""
        forecasts = {}
        cache_hits = 0
        
        # Determine optimal batch size
        if assets and assets[0] in data:
            sample_data = data[assets[0]]
            asset_class = AssetClass.EQUITY  # Default, could be determined from metadata
            
            optimal_batch_size = self.batch_sizer.get_optimal_batch_size(
                asset_class, sample_data.shape
            )
        else:
            optimal_batch_size = 32
        
        # Process in optimal batches
        for i in range(0, len(assets), optimal_batch_size):
            batch_assets = assets[i:i + optimal_batch_size]
            batch_data = []
            valid_assets = []
            
            # Prepare batch data
            for asset in batch_assets:
                if asset in data:
                    batch_data.append(data[asset])
                    valid_assets.append(asset)
            
            if not batch_data:
                continue
            
            # Stack into batch tensor
            try:
                batch_tensor = torch.stack(batch_data)
                
                # Process batch
                responses = []
                for j, asset in enumerate(valid_assets):
                    response = await self.inference_engine.predict_single(
                        batch_tensor[j:j+1],
                        request_id=f"{worker_id}_{asset}",
                        priority=1
                    )
                    responses.append(response)
                    
                    if response.cache_hit:
                        cache_hits += 1
                
                # Store results
                for asset, response in zip(valid_assets, responses):
                    if not response.error:
                        forecasts[asset] = {
                            'point_forecast': response.point_forecast,
                            'confidence_intervals': response.confidence_intervals,
                            'inference_time_ms': response.inference_time_ms,
                            'model_version': response.model_version
                        }
                
            except Exception as e:
                logger.error(f"Batch processing error for assets {valid_assets}: {e}")
                for asset in valid_assets:
                    forecasts[asset] = {'error': str(e)}
        
        return forecasts, cache_hits
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Log statistics
                if self.stats['total_requests'] > 0:
                    success_rate = self.stats['completed_requests'] / self.stats['total_requests']
                    avg_processing_time = self.stats['total_processing_time'] / max(self.stats['completed_requests'], 1)
                    cache_hit_rate = self.stats['cache_hits'] / max(self.stats['total_assets_processed'], 1)
                    
                    logger.info(f"Batch processor stats: {self.stats['completed_requests']} completed, "
                              f"{success_rate:.1%} success rate, {avg_processing_time:.2f}s avg time, "
                              f"{cache_hit_rate:.1%} cache hit rate")
                
                # Queue status
                queue_size = self.request_queue.qsize()
                active_count = len(self.active_requests)
                
                if queue_size > 100:
                    logger.warning(f"High queue size: {queue_size} pending requests")
                
                if active_count > self.max_concurrent_batches * 2:
                    logger.warning(f"High active request count: {active_count}")
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def register_asset_metadata(self, metadata: AssetMetadata):
        """Register asset metadata for optimization."""
        self.asset_optimizer.register_asset(metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        queue_size = self.request_queue.qsize()
        active_count = len(self.active_requests)
        
        stats = self.stats.copy()
        stats.update({
            'queue_size': queue_size,
            'active_requests': active_count,
            'is_running': self.is_running,
            'worker_count': len(self.worker_tasks),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'avg_processing_time': (
                self.stats['total_processing_time'] / max(self.stats['completed_requests'], 1)
            ),
            'success_rate': (
                self.stats['completed_requests'] / max(self.stats['total_requests'], 1)
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(self.stats['total_assets_processed'], 1)
            )
        })
        
        return stats
    
    async def benchmark(self, 
                       test_assets: List[str],
                       test_data: Dict[str, torch.Tensor],
                       duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark batch processing performance."""
        logger.info(f"Running batch processor benchmark for {duration_seconds} seconds...")
        
        if not self.is_running:
            await self.start()
        
        start_time = time.time()
        request_ids = []
        
        # Submit requests continuously
        while time.time() - start_time < duration_seconds:
            try:
                request_id = await self.process_batch(
                    test_assets, 
                    test_data, 
                    ProcessingPriority.NORMAL
                )
                request_ids.append(request_id)
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Benchmark request failed: {e}")
        
        # Wait for all requests to complete
        results = []
        for request_id in request_ids:
            result = await self.get_result(request_id, timeout=30.0)
            if result:
                results.append(result)
        
        # Calculate benchmark metrics
        if results:
            total_assets = sum(r.assets_processed for r in results)
            total_time = sum(r.processing_time for r in results)
            total_cache_hits = sum(r.cache_hits for r in results)
            
            avg_processing_time = total_time / len(results)
            avg_throughput = total_assets / total_time if total_time > 0 else 0
            cache_hit_rate = total_cache_hits / total_assets if total_assets > 0 else 0
            
            return {
                'total_requests': len(request_ids),
                'completed_requests': len(results),
                'total_assets_processed': total_assets,
                'avg_processing_time': avg_processing_time,
                'avg_throughput_per_second': avg_throughput,
                'cache_hit_rate': cache_hit_rate,
                'benchmark_duration': duration_seconds,
                'requests_per_second': len(results) / duration_seconds,
                'success_rate': len(results) / len(request_ids) if request_ids else 0
            }
        else:
            return {
                'error': 'No requests completed during benchmark',
                'total_requests': len(request_ids),
                'benchmark_duration': duration_seconds
            }


# Convenience functions
async def create_batch_processor(inference_engine: LightningInferenceEngine) -> MultiAssetBatchProcessor:
    """Create and start a multi-asset batch processor."""
    processor = MultiAssetBatchProcessor(inference_engine)
    await processor.start()
    return processor


async def process_assets_batch(assets: List[str],
                              data: Dict[str, torch.Tensor],
                              inference_engine: LightningInferenceEngine) -> Dict[str, Dict[str, Any]]:
    """Quick batch processing of assets."""
    processor = await create_batch_processor(inference_engine)
    
    try:
        result = await processor.process_batch_sync(assets, data)
        return result.forecasts
    finally:
        await processor.stop()