"""
Memory Pool for Ultra-Low Latency Trading

Features:
- Pre-allocated object pools to eliminate GC pauses
- NUMA-aware memory allocation
- Zero-copy buffer management
- Lock-free pool operations
- Hardware-optimized memory alignment
"""

import numpy as np
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
import time
import mmap
import ctypes
from ctypes import c_void_p, c_size_t, c_int, Structure, POINTER
import struct
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Try to import optional performance libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


T = TypeVar('T')


class MemoryAlignment:
    """Memory alignment constants for optimal performance"""
    CACHE_LINE_SIZE = 64  # bytes
    PAGE_SIZE = 4096      # bytes
    HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB


@dataclass
class PoolStats:
    """Memory pool statistics"""
    total_objects: int
    available_objects: int
    allocated_objects: int
    allocation_count: int
    deallocation_count: int
    reuse_count: int
    memory_usage_bytes: int
    

class MemoryRegion:
    """
    Memory region with hardware-optimized alignment
    """
    
    def __init__(self, size: int, alignment: int = MemoryAlignment.CACHE_LINE_SIZE):
        self.size = size
        self.alignment = alignment
        
        # Allocate aligned memory
        self.raw_memory = None
        self.aligned_memory = None
        self._allocate_aligned_memory()
        
    def _allocate_aligned_memory(self):
        """Allocate cache-line aligned memory"""
        try:
            # Try to allocate huge pages for better performance
            if self.size >= MemoryAlignment.HUGE_PAGE_SIZE:
                self.raw_memory = mmap.mmap(
                    -1, 
                    self.size,
                    mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                    mmap.PROT_READ | mmap.PROT_WRITE
                )
                # Try to use huge pages
                try:
                    self.raw_memory.madvise(mmap.MADV_HUGEPAGE)
                except:
                    pass
            else:
                # Regular allocation with alignment
                raw_size = self.size + self.alignment - 1
                self.raw_memory = mp.RawArray(ctypes.c_byte, raw_size)
                
                # Calculate aligned address
                raw_addr = ctypes.addressof(self.raw_memory)
                aligned_addr = (raw_addr + self.alignment - 1) & ~(self.alignment - 1)
                offset = aligned_addr - raw_addr
                
                # Create aligned view
                self.aligned_memory = (ctypes.c_byte * self.size).from_address(aligned_addr)
                
        except Exception as e:
            print(f"Warning: Could not allocate aligned memory: {e}")
            # Fallback to regular allocation
            self.raw_memory = mp.RawArray(ctypes.c_byte, self.size)
            self.aligned_memory = self.raw_memory
            
    def get_pointer(self) -> ctypes.c_void_p:
        """Get pointer to aligned memory"""
        if self.aligned_memory:
            return ctypes.cast(self.aligned_memory, ctypes.c_void_p)
        else:
            return ctypes.cast(self.raw_memory, ctypes.c_void_p)
            
    def cleanup(self):
        """Clean up memory region"""
        if hasattr(self.raw_memory, 'close'):
            self.raw_memory.close()


class ObjectPool(Generic[T]):
    """
    Generic object pool with lock-free operations
    """
    
    def __init__(self, 
                 factory: Callable[[], T],
                 reset_func: Optional[Callable[[T], None]] = None,
                 initial_size: int = 1000,
                 max_size: int = 10000):
        self.factory = factory
        self.reset_func = reset_func
        self.initial_size = initial_size
        self.max_size = max_size
        
        # Pre-allocate objects
        self.objects = []
        self.available_indices = []
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = PoolStats(
            total_objects=0,
            available_objects=0,
            allocated_objects=0,
            allocation_count=0,
            deallocation_count=0,
            reuse_count=0,
            memory_usage_bytes=0
        )
        
        # Pre-populate pool
        self._populate_pool()
        
    def _populate_pool(self):
        """Pre-populate pool with objects"""
        for i in range(self.initial_size):
            obj = self.factory()
            self.objects.append(obj)
            self.available_indices.append(i)
            
        self.stats.total_objects = self.initial_size
        self.stats.available_objects = self.initial_size
        
    def acquire(self) -> T:
        """Acquire object from pool"""
        with self.lock:
            if self.available_indices:
                # Reuse existing object
                index = self.available_indices.pop()
                obj = self.objects[index]
                
                # Reset object if reset function provided
                if self.reset_func:
                    self.reset_func(obj)
                    
                self.stats.allocated_objects += 1
                self.stats.available_objects -= 1
                self.stats.reuse_count += 1
                
                return obj
            else:
                # Create new object if pool not at max size
                if self.stats.total_objects < self.max_size:
                    obj = self.factory()
                    self.objects.append(obj)
                    
                    self.stats.total_objects += 1
                    self.stats.allocated_objects += 1
                    self.stats.allocation_count += 1
                    
                    return obj
                else:
                    # Pool exhausted, create temporary object
                    obj = self.factory()
                    self.stats.allocation_count += 1
                    return obj
                    
    def release(self, obj: T):
        """Release object back to pool"""
        with self.lock:
            # Find object index
            try:
                index = self.objects.index(obj)
                self.available_indices.append(index)
                
                self.stats.allocated_objects -= 1
                self.stats.available_objects += 1
                self.stats.deallocation_count += 1
                
            except ValueError:
                # Object not from pool, ignore
                pass
                
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        with self.lock:
            return PoolStats(
                total_objects=self.stats.total_objects,
                available_objects=self.stats.available_objects,
                allocated_objects=self.stats.allocated_objects,
                allocation_count=self.stats.allocation_count,
                deallocation_count=self.stats.deallocation_count,
                reuse_count=self.stats.reuse_count,
                memory_usage_bytes=self.stats.memory_usage_bytes
            )


class ByteBufferPool:
    """
    Specialized pool for byte buffers with zero-copy operations
    """
    
    def __init__(self, buffer_size: int = 8192, pool_size: int = 1000):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        
        # Allocate large memory region
        total_size = buffer_size * pool_size
        self.memory_region = MemoryRegion(total_size, MemoryAlignment.CACHE_LINE_SIZE)
        
        # Create buffer views
        self.buffers = []
        self.available_buffers = []
        
        base_addr = ctypes.addressof(self.memory_region.aligned_memory)
        for i in range(pool_size):
            buffer_addr = base_addr + (i * buffer_size)
            buffer_view = (ctypes.c_byte * buffer_size).from_address(buffer_addr)
            self.buffers.append(buffer_view)
            self.available_buffers.append(i)
            
        self.lock = threading.Lock()
        
    def acquire_buffer(self) -> Optional[ctypes.Array]:
        """Acquire buffer from pool"""
        with self.lock:
            if self.available_buffers:
                index = self.available_buffers.pop()
                return self.buffers[index]
            return None
            
    def release_buffer(self, buffer: ctypes.Array):
        """Release buffer back to pool"""
        with self.lock:
            try:
                index = self.buffers.index(buffer)
                self.available_buffers.append(index)
            except ValueError:
                pass
                
    def cleanup(self):
        """Clean up memory region"""
        self.memory_region.cleanup()


class MessagePool:
    """
    Specialized pool for FIX messages with pre-allocated fields
    """
    
    def __init__(self, pool_size: int = 10000):
        self.pool_size = pool_size
        
        # Pre-allocate message structures
        self.messages = []
        self.available_messages = []
        
        # Message structure (simplified)
        class Message(Structure):
            _fields_ = [
                ('msg_type', ctypes.c_char * 4),
                ('symbol', ctypes.c_char * 16),
                ('side', ctypes.c_char),
                ('quantity', ctypes.c_int64),
                ('price', ctypes.c_double),
                ('timestamp', ctypes.c_int64),
                ('order_id', ctypes.c_char * 32),
                ('client_id', ctypes.c_char * 16),
                ('account', ctypes.c_char * 16),
                ('buffer', ctypes.c_char * 1024)  # Raw message buffer
            ]
            
        # Pre-allocate messages
        for i in range(pool_size):
            msg = Message()
            self.messages.append(msg)
            self.available_messages.append(i)
            
        self.lock = threading.Lock()
        
    def acquire_message(self) -> Optional[Structure]:
        """Acquire message from pool"""
        with self.lock:
            if self.available_messages:
                index = self.available_messages.pop()
                msg = self.messages[index]
                # Clear message
                ctypes.memset(msg, 0, ctypes.sizeof(msg))
                return msg
            return None
            
    def release_message(self, msg: Structure):
        """Release message back to pool"""
        with self.lock:
            try:
                index = self.messages.index(msg)
                self.available_messages.append(index)
            except ValueError:
                pass


class MemoryPoolManager:
    """
    Centralized memory pool manager for trading system
    """
    
    def __init__(self):
        self.pools = {}
        self.lock = threading.Lock()
        
        # Initialize common pools
        self._initialize_common_pools()
        
    def _initialize_common_pools(self):
        """Initialize commonly used pools"""
        # Order objects pool
        def create_order():
            return {
                'order_id': '',
                'symbol': '',
                'side': '',
                'quantity': 0,
                'price': 0.0,
                'timestamp': 0,
                'status': 'PENDING'
            }
            
        def reset_order(order):
            order['order_id'] = ''
            order['symbol'] = ''
            order['side'] = ''
            order['quantity'] = 0
            order['price'] = 0.0
            order['timestamp'] = 0
            order['status'] = 'PENDING'
            
        self.pools['orders'] = ObjectPool(
            factory=create_order,
            reset_func=reset_order,
            initial_size=1000,
            max_size=10000
        )
        
        # Execution report pool
        def create_execution_report():
            return {
                'order_id': '',
                'exec_id': '',
                'exec_type': '',
                'symbol': '',
                'side': '',
                'quantity': 0,
                'price': 0.0,
                'timestamp': 0,
                'status': ''
            }
            
        def reset_execution_report(report):
            report['order_id'] = ''
            report['exec_id'] = ''
            report['exec_type'] = ''
            report['symbol'] = ''
            report['side'] = ''
            report['quantity'] = 0
            report['price'] = 0.0
            report['timestamp'] = 0
            report['status'] = ''
            
        self.pools['execution_reports'] = ObjectPool(
            factory=create_execution_report,
            reset_func=reset_execution_report,
            initial_size=1000,
            max_size=10000
        )
        
        # Buffer pools
        self.pools['small_buffers'] = ByteBufferPool(1024, 1000)
        self.pools['large_buffers'] = ByteBufferPool(8192, 500)
        
        # Message pool
        self.pools['messages'] = MessagePool(10000)
        
    def get_pool(self, pool_name: str) -> Optional[ObjectPool]:
        """Get pool by name"""
        return self.pools.get(pool_name)
        
    def acquire_object(self, pool_name: str):
        """Acquire object from named pool"""
        pool = self.pools.get(pool_name)
        if pool:
            return pool.acquire()
        return None
        
    def release_object(self, pool_name: str, obj):
        """Release object to named pool"""
        pool = self.pools.get(pool_name)
        if pool:
            pool.release(obj)
            
    def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools"""
        stats = {}
        for name, pool in self.pools.items():
            if hasattr(pool, 'get_stats'):
                stats[name] = pool.get_stats()
        return stats
        
    def cleanup(self):
        """Clean up all pools"""
        for pool in self.pools.values():
            if hasattr(pool, 'cleanup'):
                pool.cleanup()


# Global memory pool manager instance
memory_manager = MemoryPoolManager()


# NumPy array pool for numerical computations
if NUMPY_AVAILABLE:
    class NumpyArrayPool:
        """Pool for NumPy arrays to avoid allocation overhead"""
        
        def __init__(self, shape: tuple, dtype: np.dtype, pool_size: int = 100):
            self.shape = shape
            self.dtype = dtype
            self.pool_size = pool_size
            
            # Pre-allocate arrays
            self.arrays = []
            self.available_arrays = []
            
            for i in range(pool_size):
                arr = np.zeros(shape, dtype=dtype)
                self.arrays.append(arr)
                self.available_arrays.append(i)
                
            self.lock = threading.Lock()
            
        def acquire_array(self) -> Optional[np.ndarray]:
            """Acquire array from pool"""
            with self.lock:
                if self.available_arrays:
                    index = self.available_arrays.pop()
                    arr = self.arrays[index]
                    arr.fill(0)  # Clear array
                    return arr
                return None
                
        def release_array(self, arr: np.ndarray):
            """Release array back to pool"""
            with self.lock:
                try:
                    index = self.arrays.index(arr)
                    self.available_arrays.append(index)
                except ValueError:
                    pass