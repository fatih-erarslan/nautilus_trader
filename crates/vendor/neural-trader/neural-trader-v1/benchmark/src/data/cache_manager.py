"""High-performance caching for real-time market data.

Provides multi-tier caching with in-memory L1 cache and optional
Redis L2 cache for distributed systems.
"""

import asyncio
import time
import json
import pickle
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum, auto
import logging
import zlib

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels."""
    L1_MEMORY = auto()
    L2_REDIS = auto()
    
    
class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = auto()      # Least Recently Used
    LFU = auto()      # Least Frequently Used
    FIFO = auto()     # First In First Out
    TTL = auto()      # Time To Live based


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheConfig:
    """Cache configuration."""
    # L1 Memory cache
    l1_max_size_mb: float = 100
    l1_max_entries: int = 100000
    l1_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    l1_default_ttl: Optional[float] = 300  # 5 minutes
    
    # L2 Redis cache
    enable_l2_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_key_prefix: str = "market_data:"
    
    # General settings
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    enable_stats: bool = True
    stats_interval: float = 60.0


class CacheStatistics:
    """Track cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0
        self.operation_times: List[float] = []
        self._start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    @property
    def avg_operation_time_ms(self) -> float:
        """Average operation time in milliseconds."""
        if not self.operation_times:
            return 0
        return sum(self.operation_times) / len(self.operation_times) * 1000
    
    def record_operation(self, duration: float):
        """Record operation duration."""
        self.operation_times.append(duration)
        # Keep only last 1000 operations
        if len(self.operation_times) > 1000:
            self.operation_times = self.operation_times[-1000:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        uptime = time.time() - self._start_time
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "total_size_mb": self.total_size_bytes / 1024 / 1024,
            "avg_operation_ms": self.avg_operation_time_ms,
            "uptime_seconds": uptime,
        }


class L1MemoryCache:
    """High-performance in-memory cache."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_size_bytes = config.l1_max_size_mb * 1024 * 1024
        self.max_entries = config.l1_max_entries
        
        # Storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._size_bytes = 0
        
        # Eviction policy handlers
        self._eviction_handlers = {
            EvictionPolicy.LRU: self._evict_lru,
            EvictionPolicy.LFU: self._evict_lfu,
            EvictionPolicy.FIFO: self._evict_fifo,
            EvictionPolicy.TTL: self._evict_ttl,
        }
        
        # Statistics
        self.stats = CacheStatistics() if config.enable_stats else None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.perf_counter()
        
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                if self.stats:
                    self.stats.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._size_bytes -= entry.size_bytes
                if self.stats:
                    self.stats.misses += 1
                return None
            
            # Update access metadata
            entry.access()
            
            # Move to end for LRU
            if self.config.l1_eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            if self.stats:
                self.stats.hits += 1
                self.stats.record_operation(time.perf_counter() - start_time)
            
            # Decompress if needed
            value = entry.value
            if entry.compressed:
                value = pickle.loads(zlib.decompress(value))
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        start_time = time.perf_counter()
        
        # Serialize and optionally compress
        serialized = pickle.dumps(value)
        size_bytes = len(serialized)
        compressed = False
        
        if self.config.enable_compression and size_bytes > self.config.compression_threshold_bytes:
            compressed_data = zlib.compress(serialized)
            if len(compressed_data) < size_bytes:
                serialized = compressed_data
                size_bytes = len(compressed_data)
                compressed = True
        
        async with self._lock:
            # Check if we need to evict
            while (self._size_bytes + size_bytes > self.max_size_bytes or 
                   len(self._cache) >= self.max_entries):
                if not await self._evict():
                    break  # Cannot evict more
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=serialized if compressed else value,
                timestamp=time.time(),
                ttl=ttl or self.config.l1_default_ttl,
                size_bytes=size_bytes,
                compressed=compressed
            )
            
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._size_bytes -= old_entry.size_bytes
            
            # Add new entry
            self._cache[key] = entry
            self._size_bytes += size_bytes
            
            if self.stats:
                self.stats.total_size_bytes = self._size_bytes
                self.stats.record_operation(time.perf_counter() - start_time)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._size_bytes -= entry.size_bytes
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._size_bytes = 0
    
    async def _evict(self) -> bool:
        """Evict entry based on policy."""
        if not self._cache:
            return False
        
        handler = self._eviction_handlers[self.config.l1_eviction_policy]
        return await handler()
    
    async def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        # OrderedDict maintains order, first item is LRU
        key, entry = next(iter(self._cache.items()))
        del self._cache[key]
        self._size_bytes -= entry.size_bytes
        
        if self.stats:
            self.stats.evictions += 1
        
        return True
    
    async def _evict_lfu(self) -> bool:
        """Evict least frequently used entry."""
        if not self._cache:
            return False
        
        # Find entry with lowest access count
        min_entry = min(self._cache.items(), key=lambda x: x[1].access_count)
        key = min_entry[0]
        entry = min_entry[1]
        
        del self._cache[key]
        self._size_bytes -= entry.size_bytes
        
        if self.stats:
            self.stats.evictions += 1
        
        return True
    
    async def _evict_fifo(self) -> bool:
        """Evict first in (oldest) entry."""
        # Same as LRU for OrderedDict
        return await self._evict_lru()
    
    async def _evict_ttl(self) -> bool:
        """Evict expired entries."""
        current_time = time.time()
        evicted = False
        
        # Find and remove all expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache[key]
            del self._cache[key]
            self._size_bytes -= entry.size_bytes
            evicted = True
            
            if self.stats:
                self.stats.evictions += 1
        
        # If no expired entries, fall back to LRU
        if not evicted:
            return await self._evict_lru()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.stats:
            return {}
        
        stats = self.stats.to_dict()
        stats["entries"] = len(self._cache)
        stats["size_utilization"] = self._size_bytes / self.max_size_bytes
        
        return stats


class CacheManager:
    """Multi-tier cache manager with L1 memory and optional L2 Redis."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache = L1MemoryCache(config)
        self.l2_cache = None
        
        # Redis client (lazy loaded)
        self._redis_client = None
        
        # Write-through queue for L2
        self._l2_write_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._l2_writer_task = None
        
        # Cache invalidation callbacks
        self._invalidation_callbacks: List[Callable] = []
    
    async def start(self):
        """Start cache manager."""
        if self.config.enable_l2_redis:
            await self._setup_redis()
            self._l2_writer_task = asyncio.create_task(self._l2_writer())
    
    async def stop(self):
        """Stop cache manager."""
        if self._l2_writer_task:
            self._l2_writer_task.cancel()
        
        if self._redis_client:
            self._redis_client.close()
            await self._redis_client.wait_closed()
    
    async def _setup_redis(self):
        """Setup Redis connection."""
        try:
            import aioredis
            self._redis_client = await aioredis.create_redis_pool(
                f"redis://{self.config.redis_host}:{self.config.redis_port}",
                db=self.config.redis_db,
                password=self.config.redis_password,
                minsize=5,
                maxsize=10
            )
            logger.info("Redis L2 cache connected")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.config.enable_l2_redis = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2)."""
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 if enabled
        if self.config.enable_l2_redis and self._redis_client:
            try:
                redis_key = f"{self.config.redis_key_prefix}{key}"
                data = await self._redis_client.get(redis_key)
                
                if data:
                    value = pickle.loads(data)
                    # Promote to L1
                    await self.l1_cache.set(key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache (write-through to L2)."""
        # Set in L1
        await self.l1_cache.set(key, value, ttl)
        
        # Queue for L2 write if enabled
        if self.config.enable_l2_redis and self._redis_client:
            try:
                await self._l2_write_queue.put((key, value, ttl))
            except asyncio.QueueFull:
                logger.warning("L2 write queue full, skipping Redis write")
    
    async def delete(self, key: str):
        """Delete from all cache levels."""
        # Delete from L1
        await self.l1_cache.delete(key)
        
        # Delete from L2
        if self.config.enable_l2_redis and self._redis_client:
            try:
                redis_key = f"{self.config.redis_key_prefix}{key}"
                await self._redis_client.delete(redis_key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        # Notify invalidation callbacks
        for callback in self._invalidation_callbacks:
            try:
                await callback(key)
            except Exception as e:
                logger.error(f"Invalidation callback error: {e}")
    
    async def _l2_writer(self):
        """Background task to write to L2 cache."""
        while True:
            try:
                # Batch writes for efficiency
                batch = []
                
                # Collect up to 100 items or wait 10ms
                try:
                    for _ in range(100):
                        item = await asyncio.wait_for(
                            self._l2_write_queue.get(),
                            timeout=0.01
                        )
                        batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Write batch to Redis
                if batch and self._redis_client:
                    pipe = self._redis_client.pipeline()
                    
                    for key, value, ttl in batch:
                        redis_key = f"{self.config.redis_key_prefix}{key}"
                        data = pickle.dumps(value)
                        
                        if ttl:
                            pipe.setex(redis_key, int(ttl), data)
                        else:
                            pipe.set(redis_key, data)
                    
                    await pipe.execute()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"L2 writer error: {e}")
                await asyncio.sleep(1)
    
    def add_invalidation_callback(self, callback: Callable):
        """Add cache invalidation callback."""
        self._invalidation_callbacks.append(callback)
    
    async def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values efficiently."""
        results = {}
        
        # Get from L1
        for key in keys:
            value = await self.l1_cache.get(key)
            if value is not None:
                results[key] = value
        
        # Get missing from L2
        missing_keys = [k for k in keys if k not in results]
        if missing_keys and self.config.enable_l2_redis and self._redis_client:
            try:
                redis_keys = [f"{self.config.redis_key_prefix}{k}" for k in missing_keys]
                values = await self._redis_client.mget(*redis_keys)
                
                for key, data in zip(missing_keys, values):
                    if data:
                        value = pickle.loads(data)
                        results[key] = value
                        # Promote to L1
                        await self.l1_cache.set(key, value)
            except Exception as e:
                logger.error(f"Redis batch get error: {e}")
        
        return results
    
    async def set_batch(self, items: Dict[str, Any], ttl: Optional[float] = None):
        """Set multiple values efficiently."""
        tasks = []
        for key, value in items.items():
            tasks.append(self.set(key, value, ttl))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "l1": self.l1_cache.get_stats(),
            "l2_enabled": self.config.enable_l2_redis,
            "l2_queue_size": self._l2_write_queue.qsize() if self._l2_write_queue else 0,
        }
        
        return stats