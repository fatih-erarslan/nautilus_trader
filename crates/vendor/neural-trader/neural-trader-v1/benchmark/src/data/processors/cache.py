"""
Smart cache for efficient data storage and retrieval
"""
import logging
import time
import pickle
import zlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
from pathlib import Path

from ..realtime_manager import DataPoint, AggregatedData

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    compressed: bool = False


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0


class SmartCache:
    """Smart cache with multiple eviction strategies and persistence"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Cache configuration
        self.max_size_mb = self.config.get('max_size_mb', 100)
        self.max_entries = self.config.get('max_entries', 10000)
        self.default_ttl_seconds = self.config.get('default_ttl_seconds', 3600)  # 1 hour
        self.strategy = CacheStrategy(self.config.get('strategy', 'lru'))
        
        # Compression settings
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_threshold_bytes = self.config.get('compression_threshold_bytes', 1024)
        
        # Persistence settings
        self.enable_persistence = self.config.get('enable_persistence', False)
        self.persistence_file = self.config.get('persistence_file', 'cache.pkl')
        self.persistence_interval_seconds = self.config.get('persistence_interval_seconds', 300)  # 5 minutes
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_frequencies: Dict[str, int] = defaultdict(int)
        self._size_bytes = 0
        
        # Statistics
        self._stats = CacheStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task = None
        self._persistence_task = None
        self._running = False
        
        # Symbol-specific TTLs
        self.symbol_ttls = {
            'crypto': 30.0,      # Crypto data expires quickly
            'stock': 60.0,       # Stock data moderate TTL
            'bond': 300.0,       # Bond data longer TTL
            'treasury': 600.0,   # Treasury data longest TTL
        }
    
    async def start(self) -> None:
        """Start cache background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Load from persistence if enabled
        if self.enable_persistence:
            await self._load_from_disk()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.enable_persistence:
            self._persistence_task = asyncio.create_task(self._persistence_loop())
        
        logger.info("Smart cache started")
    
    async def stop(self) -> None:
        """Stop cache and save to disk"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._persistence_task:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
        
        # Save to disk if enabled
        if self.enable_persistence:
            await self._save_to_disk()
        
        logger.info("Smart cache stopped")
    
    def put(self, key: str, data: Any, ttl_seconds: Optional[float] = None, 
            tags: List[str] = None) -> bool:
        """Store data in cache"""
        with self._lock:
            try:
                # Calculate TTL
                if ttl_seconds is None:
                    ttl_seconds = self._calculate_ttl(key)
                
                # Serialize and optionally compress data
                serialized_data, compressed = self._serialize_data(data)
                size_bytes = len(serialized_data) if isinstance(serialized_data, bytes) else len(str(serialized_data))
                
                # Create cache entry
                entry = CacheEntry(
                    data=serialized_data,
                    timestamp=datetime.utcnow(),
                    ttl_seconds=ttl_seconds,
                    size_bytes=size_bytes,
                    tags=tags or [],
                    compressed=compressed
                )
                
                # Check if we need to evict entries
                self._ensure_capacity(size_bytes)
                
                # Store entry
                if key in self._cache:
                    # Update existing entry
                    old_entry = self._cache[key]
                    self._size_bytes -= old_entry.size_bytes
                
                self._cache[key] = entry
                self._cache.move_to_end(key)  # Move to end for LRU
                self._size_bytes += size_bytes
                
                # Update access frequency for LFU
                if self.strategy == CacheStrategy.LFU:
                    self._access_frequencies[key] += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Error storing data in cache: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache"""
        with self._lock:
            try:
                if key not in self._cache:
                    self._stats.misses += 1
                    return None
                
                entry = self._cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(key)
                    self._stats.misses += 1
                    return None
                
                # Update access information
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self._cache.move_to_end(key)
                
                # Update frequency for LFU
                if self.strategy == CacheStrategy.LFU:
                    self._access_frequencies[key] += 1
                
                self._stats.hits += 1
                
                # Deserialize data
                data = self._deserialize_data(entry.data, entry.compressed)
                return data
                
            except Exception as e:
                logger.error(f"Error retrieving data from cache: {e}")
                self._stats.misses += 1
                return None
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if self._is_expired(entry):
                self._remove_entry(key)
                return False
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_frequencies.clear()
            self._size_bytes = 0
            logger.info("Cache cleared")
    
    def clear_by_tag(self, tag: str) -> int:
        """Clear entries with specific tag"""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            return len(keys_to_remove)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._size_bytes,
                entry_count=len(self._cache)
            )
            
            total_requests = stats.hits + stats.misses
            stats.hit_rate = stats.hits / total_requests if total_requests > 0 else 0.0
            
            return stats
    
    def get_keys(self, pattern: str = None) -> List[str]:
        """Get cache keys, optionally filtered by pattern"""
        with self._lock:
            keys = list(self._cache.keys())
            
            if pattern:
                import re
                pattern_regex = re.compile(pattern)
                keys = [key for key in keys if pattern_regex.match(key)]
            
            return keys
    
    def _calculate_ttl(self, key: str) -> float:
        """Calculate TTL based on key characteristics"""
        key_lower = key.lower()
        
        # Symbol-specific TTLs
        for symbol_type, ttl in self.symbol_ttls.items():
            if symbol_type in key_lower:
                return ttl
        
        # Default TTL
        return self.default_ttl_seconds
    
    def _serialize_data(self, data: Any) -> Tuple[Union[bytes, Any], bool]:
        """Serialize and optionally compress data"""
        try:
            # Serialize data
            if isinstance(data, (DataPoint, AggregatedData)):
                # Custom serialization for data objects
                serialized = pickle.dumps(data)
            else:
                serialized = pickle.dumps(data)
            
            # Compress if enabled and data is large enough
            compressed = False
            if (self.enable_compression and 
                len(serialized) > self.compression_threshold_bytes):
                
                compressed_data = zlib.compress(serialized)
                
                # Only use compression if it actually reduces size
                if len(compressed_data) < len(serialized):
                    serialized = compressed_data
                    compressed = True
            
            return serialized, compressed
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            return data, False
    
    def _deserialize_data(self, serialized_data: Any, compressed: bool) -> Any:
        """Deserialize and decompress data"""
        try:
            if compressed:
                decompressed_data = zlib.decompress(serialized_data)
                return pickle.loads(decompressed_data)
            else:
                if isinstance(serialized_data, bytes):
                    return pickle.loads(serialized_data)
                else:
                    return serialized_data
                    
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.utcnow() - entry.timestamp).total_seconds()
        return age_seconds > entry.ttl_seconds
    
    def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry"""
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        # Check size constraint
        while (self._size_bytes + new_entry_size > max_size_bytes or 
               len(self._cache) >= self.max_entries):
            
            if not self._cache:
                break
            
            evicted_key = self._select_eviction_candidate()
            if evicted_key:
                self._remove_entry(evicted_key)
                self._stats.evictions += 1
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on strategy"""
        if not self._cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            return next(iter(self._cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_frequency = min(self._access_frequencies.values())
            for key, frequency in self._access_frequencies.items():
                if frequency == min_frequency and key in self._cache:
                    return key
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            now = datetime.utcnow()
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    return key
            
            # If no expired entries, remove oldest
            return next(iter(self._cache))
        
        elif self.strategy == CacheStrategy.SIZE:
            # Remove largest entry
            largest_key = max(self._cache.keys(), 
                            key=lambda k: self._cache[k].size_bytes)
            return largest_key
        
        # Default to LRU
        return next(iter(self._cache))
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self._cache:
            entry = self._cache[key]
            self._size_bytes -= entry.size_bytes
            del self._cache[key]
            
            if key in self._access_frequencies:
                del self._access_frequencies[key]
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                
                with self._lock:
                    # Remove expired entries
                    expired_keys = []
                    for key, entry in self._cache.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                        self._stats.evictions += 1
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _persistence_loop(self) -> None:
        """Background persistence loop"""
        while self._running:
            try:
                await asyncio.sleep(self.persistence_interval_seconds)
                await self._save_to_disk()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache persistence loop: {e}")
    
    async def _save_to_disk(self) -> None:
        """Save cache to disk"""
        try:
            cache_data = {
                'cache': dict(self._cache),
                'access_frequencies': dict(self._access_frequencies),
                'stats': self._stats,
                'timestamp': datetime.utcnow()
            }
            
            # Save in a separate thread to avoid blocking
            def save_sync():
                with open(self.persistence_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            await asyncio.get_event_loop().run_in_executor(None, save_sync)
            logger.debug(f"Cache saved to {self.persistence_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    async def _load_from_disk(self) -> None:
        """Load cache from disk"""
        try:
            persistence_path = Path(self.persistence_file)
            if not persistence_path.exists():
                return
            
            def load_sync():
                with open(self.persistence_file, 'rb') as f:
                    return pickle.load(f)
            
            cache_data = await asyncio.get_event_loop().run_in_executor(None, load_sync)
            
            with self._lock:
                self._cache = OrderedDict(cache_data['cache'])
                self._access_frequencies = defaultdict(int, cache_data['access_frequencies'])
                self._stats = cache_data.get('stats', CacheStats())
                
                # Recalculate size
                self._size_bytes = sum(entry.size_bytes for entry in self._cache.values())
            
            logger.info(f"Cache loaded from {self.persistence_file} with {len(self._cache)} entries")
            
        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")
    
    # Convenience methods for data-specific caching
    def put_data_point(self, data_point: DataPoint, ttl_seconds: Optional[float] = None) -> bool:
        """Store data point in cache"""
        key = f"dp_{data_point.symbol}_{data_point.source}_{data_point.timestamp}"
        tags = ['data_point', data_point.symbol, data_point.source]
        return self.put(key, data_point, ttl_seconds, tags)
    
    def get_data_points(self, symbol: str, source: str = None) -> List[DataPoint]:
        """Get data points for symbol"""
        pattern = f"dp_{symbol}_"
        if source:
            pattern += f"{source}_"
        
        keys = self.get_keys(pattern + ".*")
        data_points = []
        
        for key in keys:
            dp = self.get(key)
            if dp and isinstance(dp, DataPoint):
                data_points.append(dp)
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x.timestamp, reverse=True)
        return data_points
    
    def put_aggregated_data(self, aggregated_data: AggregatedData, ttl_seconds: Optional[float] = None) -> bool:
        """Store aggregated data in cache"""
        key = f"agg_{aggregated_data.symbol}_{aggregated_data.timestamp}"
        tags = ['aggregated_data', aggregated_data.symbol]
        return self.put(key, aggregated_data, ttl_seconds, tags)
    
    def get_latest_aggregated_data(self, symbol: str) -> Optional[AggregatedData]:
        """Get latest aggregated data for symbol"""
        pattern = f"agg_{symbol}_"
        keys = self.get_keys(pattern + ".*")
        
        if not keys:
            return None
        
        # Sort keys by timestamp (extract from key)
        def extract_timestamp(key):
            try:
                timestamp_str = key.split('_', 2)[2]
                return datetime.fromisoformat(timestamp_str)
            except:
                return datetime.min
        
        latest_key = max(keys, key=extract_timestamp)
        return self.get(latest_key)