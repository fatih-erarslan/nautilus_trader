import threading
import time
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
import hashlib
import json

# Type variables for generics
K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheEntry(Generic[V]):
    """
    Thread-safe cache entry with value, metadata, and expiration control.

    Attributes:
        value: The cached value
        insertion_time: When the entry was added/updated
        access_time: When the entry was last accessed
        access_count: How many times the entry has been accessed
        size_bytes: Estimated memory size of the entry
        ttl_seconds: Time-to-live in seconds (0 = no expiration)
        metadata: Optional metadata dictionary
    """

    value: V
    insertion_time: float
    access_time: float
    access_count: int
    size_bytes: int
    ttl_seconds: float
    metadata: Dict[str, Any]

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl_seconds <= 0:
            return False

        current_time = current_time or time.time()
        return current_time - self.insertion_time > self.ttl_seconds


class CircularBuffer(Generic[K, V]):
    """
    Thread-safe, memory-efficient circular buffer with advanced features.

    Provides:
    - Fixed-size buffer with automatic eviction policies
    - Thread-safe operations with fine-grained locking
    - Dictionary-like access with key expiration
    - Memory usage tracking and optimization
    - Event hooks for cache operations
    - Multiple eviction policies (LRU, LFU, TTL)

    Memory efficiency features:
    - Size-aware eviction to stay under memory limits
    - Optional compression for large values
    - Lazy cleanup of expired entries
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        eviction_policy: str = "lru",
        default_ttl: float = 0,
        compression_threshold: int = 1024 * 1024,
        compression_level: int = 1,
    ):
        """
        Initialize circular buffer with configuration.

        Args:
            max_size: Maximum number of items in buffer
            max_memory_mb: Maximum memory usage in MB
            eviction_policy: 'lru', 'lfu', or 'ttl'
            default_ttl: Default time-to-live in seconds (0 = no expiration)
            compression_threshold: Size in bytes above which to compress values
            compression_level: Compression level (0-9, higher = better compression)
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.eviction_policy = eviction_policy.lower()
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level

        # Core data structures
        self.buffer: Dict[K, CacheEntry[V]] = {}
        self.keys: List[K] = []  # Ordered for eviction policies
        self.lock = threading.RLock()

        # Performance statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "updates": 0,
            "evictions": 0,
            "expirations": 0,
            "memory_savings": 0,  # From compression
            "last_cleanup_time": time.time(),
        }

        # Event hooks
        self.on_evict: Optional[Callable[[K, V], None]] = None
        self.on_expire: Optional[Callable[[K, V], None]] = None

        # Initialize compression if available
        self.compression_available = False
        try:
            import zlib

            self.compression_available = True
        except ImportError:
            pass

    def __getitem__(self, key: K) -> V:
        """Get item from buffer with statistics tracking."""
        with self.lock:
            self._lazy_cleanup()

            if key in self.buffer:
                entry = self.buffer[key]

                # Check for expiration
                if entry.is_expired():
                    self._remove_item(key)
                    self.stats["expirations"] += 1
                    self.stats["misses"] += 1
                    if self.on_expire:
                        self.on_expire(key, entry.value)
                    raise KeyError(key)

                # Update statistics
                entry.access_time = time.time()
                entry.access_count += 1
                self.stats["hits"] += 1

                # Update key order for LRU if needed
                if self.eviction_policy == "lru":
                    self.keys.remove(key)
                    self.keys.append(key)

                return entry.value

            self.stats["misses"] += 1
            raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        """Set item in buffer with eviction if needed."""
        with self.lock:
            # Estimate memory size of new value
            value_size = self._estimate_size(value)

            # Check if we need compression
            compressed_value, compressed_size = self._maybe_compress(value, value_size)
            final_value = compressed_value if compressed_value is not None else value
            final_size = compressed_size if compressed_size > 0 else value_size

            # Create cache entry
            current_time = time.time()
            metadata = {"compressed": compressed_value is not None}

            entry = CacheEntry(
                value=final_value,
                insertion_time=current_time,
                access_time=current_time,
                access_count=0,
                size_bytes=final_size,
                ttl_seconds=self.default_ttl,
                metadata=metadata,
            )

            # Check if we're updating existing entry
            is_update = key in self.buffer

            # Prepare to add the new entry
            if is_update:
                # Remove old entry size from tracking
                old_entry = self.buffer[key]
                old_size = old_entry.size_bytes

                # Update stats
                self.stats["updates"] += 1
            else:
                # It's a new item
                self.stats["writes"] += 1
                old_size = 0

            # Ensure we have space for new/updated item
            size_delta = final_size - old_size
            if size_delta > 0:
                self._ensure_memory_available(size_delta)

            # Add/update entry
            if not is_update:
                self.keys.append(key)
            else:
                # If eviction policy is LRU, move to end
                if self.eviction_policy == "lru":
                    self.keys.remove(key)
                    self.keys.append(key)

            self.buffer[key] = entry

    def __contains__(self, key: K) -> bool:
        """Check if key is in buffer and not expired."""
        with self.lock:
            if key not in self.buffer:
                return False

            entry = self.buffer[key]
            if entry.is_expired():
                self._remove_item(key)
                self.stats["expirations"] += 1
                if self.on_expire:
                    self.on_expire(key, entry.value)
                return False

            return True

    def __len__(self) -> int:
        """Allow len(buffer) to return current item count."""
        with self.lock:
            return len(self.keys)
    def get(self, key: K, default: Any = None) -> Any:
        """Get item with fallback to default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set item with optional custom TTL."""
        with self.lock:
            # Set basic value
            self[key] = value

            # Update TTL if specified
            if ttl is not None and key in self.buffer:
                self.buffer[key].ttl_seconds = ttl

    def pop(self, key: K, default: Any = None) -> Any:
        """Remove and return item, or return default if not found."""
        with self.lock:
            if key not in self.buffer:
                return default

            entry = self.buffer[key]

            # Check for expiration
            if entry.is_expired():
                self._remove_item(key)
                self.stats["expirations"] += 1
                if self.on_expire:
                    self.on_expire(key, entry.value)
                return default

            # Remove and return value
            value = entry.value
            self._remove_item(key)

            # Decompress if needed
            if entry.metadata.get("compressed", False):
                value = self._decompress(value)

            return value

    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
            self.keys.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            # Calculate derived statistics
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            memory_usage = self._calculate_memory_usage()

            return {
                **self.stats,
                "size": len(self.keys),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "memory_usage": memory_usage,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "memory_limit_mb": self.max_memory_bytes / (1024 * 1024),
                "utilization": memory_usage / self.max_memory_bytes
                if self.max_memory_bytes > 0
                else 0,
            }

    def update_ttl(self, key: K, ttl: float) -> bool:
        """
        Update TTL for a specific key.

        Args:
            key: Key to update
            ttl: New time-to-live in seconds

        Returns:
            bool: True if updated, False if key not found
        """
        with self.lock:
            if key not in self.buffer:
                return False

            entry = self.buffer[key]

            # Check for expiration
            if entry.is_expired():
                self._remove_item(key)
                self.stats["expirations"] += 1
                if self.on_expire:
                    self.on_expire(key, entry.value)
                return False

            # Update TTL
            entry.ttl_seconds = ttl
            return True

    def get_many(self, keys: List[K], default: Any = None) -> Dict[K, Any]:
        """
        Get multiple items at once.

        Args:
            keys: List of keys to retrieve
            default: Default value for missing keys

        Returns:
            dict: Dictionary mapping keys to values
        """
        result = {}

        with self.lock:
            for key in keys:
                try:
                    result[key] = self[key]
                except KeyError:
                    result[key] = default

        return result

    def set_many(self, items: Dict[K, V], ttl: Optional[float] = None) -> None:
        """
        Set multiple items at once.

        Args:
            items: Dictionary of key-value pairs
            ttl: Optional TTL for all items
        """
        with self.lock:
            for key, value in items.items():
                self.set(key, value, ttl)

    def generate_key(self, obj: Any) -> str:
        """
        Generate stable hash key for complex objects.

        Args:
            obj: Object to hash

        Returns:
            str: Stable hash key
        """
        # Convert object to stable JSON representation
        try:
            serialized = json.dumps(obj, sort_keys=True).encode("utf-8")
        except TypeError:
            # Fallback for non-serializable objects
            serialized = str(obj).encode("utf-8")

        # Generate hash
        return hashlib.md5(serialized).hexdigest()

    def _remove_item(self, key: K) -> None:
        """Remove item from buffer and update tracking."""
        if key in self.buffer:
            del self.buffer[key]

        if key in self.keys:
            self.keys.remove(key)

    def _ensure_memory_available(self, required_bytes: int) -> None:
        """
        Ensure enough memory is available by evicting items if needed.

        Args:
            required_bytes: Number of bytes needed
        """
        # Check if we need to evict based on max_size
        while len(self.keys) >= self.max_size:
            self._evict_item()

        # Check if we need to evict based on memory usage
        current_usage = self._calculate_memory_usage()

        while current_usage + required_bytes > self.max_memory_bytes and self.keys:
            freed_bytes = self._evict_item()
            current_usage -= freed_bytes

            # Safety check - if we couldn't free any memory, break
            if freed_bytes <= 0:
                break

    def _evict_item(self) -> int:
        """
        Evict an item based on the configured policy.

        Returns:
            int: Number of bytes freed
        """
        if not self.keys:
            return 0

        # Choose key to evict based on policy
        if self.eviction_policy == "lru":
            # Least Recently Used - oldest item in self.keys
            key_to_evict = self.keys[0]
        elif self.eviction_policy == "lfu":
            # Least Frequently Used - find item with lowest access count
            min_count = float("inf")
            key_to_evict = self.keys[0]

            for key in self.keys:
                count = self.buffer[key].access_count
                if count < min_count:
                    min_count = count
                    key_to_evict = key
        elif self.eviction_policy == "ttl":
            # TTL-based - find item closest to expiration
            current_time = time.time()
            min_remaining = float("inf")
            key_to_evict = self.keys[0]

            for key in self.keys:
                entry = self.buffer[key]
                if entry.ttl_seconds <= 0:
                    continue

                remaining = entry.ttl_seconds - (current_time - entry.insertion_time)
                if remaining < min_remaining:
                    min_remaining = remaining
                    key_to_evict = key
        else:
            # Default to FIFO
            key_to_evict = self.keys[0]

        # Get the entry and size before eviction
        entry = self.buffer[key_to_evict]
        freed_bytes = entry.size_bytes

        # Call eviction hook if registered
        if self.on_evict:
            value = entry.value
            if entry.metadata.get("compressed", False):
                value = self._decompress(value)
            self.on_evict(key_to_evict, value)

        # Remove the item
        self._remove_item(key_to_evict)

        # Update stats
        self.stats["evictions"] += 1

        return freed_bytes

    def _lazy_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        current_time = time.time()

        # Only clean up if it's been at least 60 seconds since last cleanup
        if current_time - self.stats["last_cleanup_time"] < 60:
            return

        # Find and remove expired entries
        expired_keys = []
        for key in list(self.buffer.keys()):
            entry = self.buffer[key]
            if entry.is_expired(current_time):
                expired_keys.append(key)

        # Clean up expired entries
        for key in expired_keys:
            entry = self.buffer[key]
            self._remove_item(key)
            self.stats["expirations"] += 1

            # Call expiration hook if registered
            if self.on_expire:
                value = entry.value
                if entry.metadata.get("compressed", False):
                    value = self._decompress(value)
                self.on_expire(key, value)

        # Update last cleanup time
        self.stats["last_cleanup_time"] = current_time

    def _calculate_memory_usage(self) -> int:
        """Calculate total memory usage of buffer."""
        return sum(entry.size_bytes for entry in self.buffer.values())

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate memory size of an object in bytes.

        Uses sys.getsizeof with special handling for common types.
        """
        try:
            # Handle common types with special sizing
            if isinstance(obj, (str, bytes, bytearray)):
                return sys.getsizeof(obj)
            elif isinstance(obj, (int, float, bool, type(None))):
                return sys.getsizeof(obj)
            elif isinstance(obj, (list, tuple)):
                return sys.getsizeof(obj) + sum(
                    self._estimate_size(item) for item in obj
                )
            elif isinstance(obj, dict):
                return sys.getsizeof(obj) + sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            else:
                # Fallback for other types
                return sys.getsizeof(obj)
        except (TypeError, AttributeError):
            # Ultimate fallback
            return 100  # Arbitrary small size

    def _maybe_compress(self, value: V, size: int) -> Tuple[Optional[bytes], int]:
        """
        Compress value if beneficial and applicable.

        Args:
            value: Value to potentially compress
            size: Estimated size in bytes

        Returns:
            tuple: (compressed_value or None, compressed_size or 0)
        """
        # Skip if compression is not available
        if not self.compression_available:
            return None, 0

        # Skip if below threshold
        if size < self.compression_threshold:
            return None, 0

        # Skip if not a serializable type
        if not isinstance(value, (str, bytes, bytearray, list, dict, tuple)):
            return None, 0

        try:
            import zlib

            # Serialize to bytes if not already
            if isinstance(value, bytes):
                data = value
            elif isinstance(value, bytearray):
                data = bytes(value)
            elif isinstance(value, str):
                data = value.encode("utf-8")
            else:
                # Try to JSON serialize
                data = json.dumps(value).encode("utf-8")

            # Compress
            compressed = zlib.compress(data, level=self.compression_level)
            compressed_size = len(compressed)

            # Only use compression if it actually saves space
            if compressed_size < size:
                self.stats["memory_savings"] += size - compressed_size
                return compressed, compressed_size
        except (ImportError, TypeError, AttributeError, zlib.error):
            pass

        return None, 0

    def _decompress(self, value: bytes) -> Any:
        """
        Decompress a value.

        Args:
            value: Compressed value

        Returns:
            Any: Decompressed value
        """
        try:
            import zlib

            # Decompress
            decompressed = zlib.decompress(value)

            # Try to interpret as JSON
            try:
                return json.loads(decompressed)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, return as bytes or decode to string if possible
                try:
                    return decompressed.decode("utf-8")
                except UnicodeDecodeError:
                    return decompressed
        except (ImportError, zlib.error, AttributeError):
            # Fallback - return as is
            return value


# Global singleton and lock
_circular_buffer_instances: Dict[str, CircularBuffer] = {}
_buffer_lock = threading.RLock()


def get_circular_buffer(
    name: str = "default", config: Optional[Dict[str, Any]] = None, reset: bool = False
) -> CircularBuffer:
    """
    Thread-safe factory for named CircularBuffer instances.

    Args:
        name: Buffer instance name
        config: Optional configuration dictionary
        reset: Whether to force recreation of instance

    Returns:
        CircularBuffer: Named instance
    """
    global _circular_buffer_instances, _buffer_lock

    with _buffer_lock:
        if name not in _circular_buffer_instances or reset:
            # Default configuration
            default_config = {
                "max_size": 1000,
                "max_memory_mb": 100.0,
                "eviction_policy": "lru",
                "default_ttl": 0,
                "compression_threshold": 1024 * 1024,
                "compression_level": 1,
            }

            # Update with provided config
            if config:
                default_config.update(config)

            # Create instance
            _circular_buffer_instances[name] = CircularBuffer(
                max_size=default_config["max_size"],
                max_memory_mb=default_config["max_memory_mb"],
                eviction_policy=default_config["eviction_policy"],
                default_ttl=default_config["default_ttl"],
                compression_threshold=default_config["compression_threshold"],
                compression_level=default_config["compression_level"],
            )

        return _circular_buffer_instances[name]
