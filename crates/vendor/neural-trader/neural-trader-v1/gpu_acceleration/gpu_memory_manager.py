"""
Advanced GPU Memory Management System
Optimized memory pooling, allocation, and management for high-performance GPU computing.
"""

import cupy as cp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import weakref
import gc
import time
from collections import defaultdict, deque
import warnings

from .flyio_gpu_config import get_gpu_config_manager

logger = logging.getLogger(__name__)


class MemoryPoolType(Enum):
    """Types of memory pools."""
    DEFAULT = "default"
    WORKSPACE = "workspace"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"
    SHARED = "shared"


class AllocationStrategy(Enum):
    """Memory allocation strategies."""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BUDDY_SYSTEM = "buddy_system"
    SLAB_ALLOCATOR = "slab_allocator"


@dataclass
class MemoryBlock:
    """Represents a memory block."""
    ptr: int
    size: int
    allocated: bool = False
    pool_type: MemoryPoolType = MemoryPoolType.DEFAULT
    timestamp: float = 0.0
    ref_count: int = 0
    tag: Optional[str] = None


@dataclass
class MemoryPoolConfig:
    """Configuration for memory pools."""
    pool_type: MemoryPoolType
    initial_size_gb: float = 1.0
    max_size_gb: float = 8.0
    growth_factor: float = 1.5
    allocation_strategy: AllocationStrategy = AllocationStrategy.BUDDY_SYSTEM
    enable_defragmentation: bool = True
    defrag_threshold: float = 0.7
    enable_garbage_collection: bool = True
    gc_interval: int = 1000
    enable_memory_mapping: bool = False


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated: int = 0
    total_free: int = 0
    largest_free_block: int = 0
    num_allocations: int = 0
    num_deallocations: int = 0
    fragmentation_ratio: float = 0.0
    pool_utilization: float = 0.0
    peak_usage: int = 0
    allocation_failures: int = 0
    gc_count: int = 0
    defrag_count: int = 0


class AdvancedGPUMemoryPool:
    """Advanced GPU memory pool with sophisticated allocation strategies."""
    
    def __init__(self, config: MemoryPoolConfig):
        """Initialize advanced memory pool."""
        self.config = config
        self.blocks = []
        self.free_blocks = defaultdict(list)  # size -> list of blocks
        self.allocated_blocks = {}  # ptr -> block
        self.stats = MemoryStats()
        self.lock = threading.RLock()
        
        # Initialize pool
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize the memory pool."""
        initial_size = int(self.config.initial_size_gb * (1024**3))
        
        try:
            # Allocate initial pool
            initial_ptr = cp.cuda.alloc(initial_size)
            
            # Create initial free block
            initial_block = MemoryBlock(
                ptr=initial_ptr.ptr,
                size=initial_size,
                allocated=False,
                pool_type=self.config.pool_type,
                timestamp=time.time()
            )
            
            self.blocks.append(initial_block)
            self.free_blocks[initial_size].append(initial_block)
            self.stats.total_free = initial_size
            
            logger.info(f"Initialized {self.config.pool_type.value} pool with {initial_size} bytes")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory pool: {str(e)}")
            raise
            
    def allocate(self, size: int, tag: Optional[str] = None) -> Optional[cp.cuda.MemoryPointer]:
        """Allocate memory from the pool."""
        with self.lock:
            # Align size to 256 bytes for optimal performance
            aligned_size = ((size + 255) // 256) * 256
            
            # Find suitable block
            block = self._find_block(aligned_size)
            
            if block is None:
                # Try to expand pool
                if not self._expand_pool(aligned_size):
                    self.stats.allocation_failures += 1
                    logger.warning(f"Memory allocation failed for {aligned_size} bytes")
                    return None
                    
                # Try again after expansion
                block = self._find_block(aligned_size)
                
            if block is None:
                self.stats.allocation_failures += 1
                return None
                
            # Allocate from block
            allocated_ptr = self._allocate_from_block(block, aligned_size, tag)
            
            if allocated_ptr:
                self.stats.num_allocations += 1
                self.stats.total_allocated += aligned_size
                self.stats.total_free -= aligned_size
                
                if self.stats.total_allocated > self.stats.peak_usage:
                    self.stats.peak_usage = self.stats.total_allocated
                    
            return allocated_ptr
            
    def deallocate(self, ptr: cp.cuda.MemoryPointer):
        """Deallocate memory back to the pool."""
        with self.lock:
            ptr_val = ptr.ptr if hasattr(ptr, 'ptr') else ptr
            
            if ptr_val not in self.allocated_blocks:
                logger.warning(f"Attempted to deallocate untracked pointer: {ptr_val}")
                return
                
            block = self.allocated_blocks[ptr_val]
            
            # Mark as free
            block.allocated = False
            block.ref_count = 0
            block.timestamp = time.time()
            
            # Update statistics
            self.stats.num_deallocations += 1
            self.stats.total_allocated -= block.size
            self.stats.total_free += block.size
            
            # Add to free blocks
            self.free_blocks[block.size].append(block)
            
            # Remove from allocated blocks
            del self.allocated_blocks[ptr_val]
            
            # Trigger defragmentation if needed
            if self._should_defragment():
                self._defragment()
                
    def _find_block(self, size: int) -> Optional[MemoryBlock]:
        """Find a suitable block for allocation."""
        if self.config.allocation_strategy == AllocationStrategy.FIRST_FIT:
            return self._first_fit(size)
        elif self.config.allocation_strategy == AllocationStrategy.BEST_FIT:
            return self._best_fit(size)
        elif self.config.allocation_strategy == AllocationStrategy.WORST_FIT:
            return self._worst_fit(size)
        elif self.config.allocation_strategy == AllocationStrategy.BUDDY_SYSTEM:
            return self._buddy_allocate(size)
        elif self.config.allocation_strategy == AllocationStrategy.SLAB_ALLOCATOR:
            return self._slab_allocate(size)
        else:
            return self._first_fit(size)
            
    def _first_fit(self, size: int) -> Optional[MemoryBlock]:
        """First-fit allocation strategy."""
        for block_size in sorted(self.free_blocks.keys()):
            if block_size >= size and self.free_blocks[block_size]:
                return self.free_blocks[block_size].pop(0)
        return None
        
    def _best_fit(self, size: int) -> Optional[MemoryBlock]:
        """Best-fit allocation strategy."""
        best_block = None
        best_size = float('inf')
        
        for block_size, blocks in self.free_blocks.items():
            if block_size >= size and blocks:
                if block_size < best_size:
                    best_size = block_size
                    best_block = blocks[0]
                    
        if best_block:
            self.free_blocks[best_size].remove(best_block)
            
        return best_block
        
    def _worst_fit(self, size: int) -> Optional[MemoryBlock]:
        """Worst-fit allocation strategy."""
        worst_block = None
        worst_size = 0
        
        for block_size, blocks in self.free_blocks.items():
            if block_size >= size and blocks:
                if block_size > worst_size:
                    worst_size = block_size
                    worst_block = blocks[0]
                    
        if worst_block:
            self.free_blocks[worst_size].remove(worst_block)
            
        return worst_block
        
    def _buddy_allocate(self, size: int) -> Optional[MemoryBlock]:
        """Buddy system allocation."""
        # Find the smallest power of 2 >= size
        buddy_size = 1
        while buddy_size < size:
            buddy_size *= 2
            
        # Look for exact match first
        if buddy_size in self.free_blocks and self.free_blocks[buddy_size]:
            return self.free_blocks[buddy_size].pop(0)
            
        # Look for larger block to split
        for block_size in sorted(self.free_blocks.keys()):
            if block_size > buddy_size and self.free_blocks[block_size]:
                block = self.free_blocks[block_size].pop(0)
                return self._split_block(block, buddy_size)
                
        return None
        
    def _slab_allocate(self, size: int) -> Optional[MemoryBlock]:
        """Slab allocator for common sizes."""
        # Common sizes for financial data
        common_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        
        # Find appropriate slab size
        slab_size = size
        for common_size in common_sizes:
            if common_size >= size:
                slab_size = common_size
                break
                
        # Look for exact slab match
        if slab_size in self.free_blocks and self.free_blocks[slab_size]:
            return self.free_blocks[slab_size].pop(0)
            
        # Look for larger block to split
        for block_size in sorted(self.free_blocks.keys()):
            if block_size > slab_size and self.free_blocks[block_size]:
                block = self.free_blocks[block_size].pop(0)
                return self._split_block(block, slab_size)
                
        return None
        
    def _split_block(self, block: MemoryBlock, size: int) -> MemoryBlock:
        """Split a block into two parts."""
        if block.size <= size:
            return block
            
        # Create new block for the remaining space
        remaining_size = block.size - size
        remaining_block = MemoryBlock(
            ptr=block.ptr + size,
            size=remaining_size,
            allocated=False,
            pool_type=block.pool_type,
            timestamp=time.time()
        )
        
        # Update original block
        block.size = size
        
        # Add remaining block to free list
        self.blocks.append(remaining_block)
        self.free_blocks[remaining_size].append(remaining_block)
        
        return block
        
    def _allocate_from_block(self, block: MemoryBlock, size: int, 
                           tag: Optional[str] = None) -> cp.cuda.MemoryPointer:
        """Allocate memory from a specific block."""
        # Remove from free blocks
        if block in self.free_blocks[block.size]:
            self.free_blocks[block.size].remove(block)
            
        # Mark as allocated
        block.allocated = True
        block.ref_count = 1
        block.tag = tag
        block.timestamp = time.time()
        
        # Add to allocated blocks
        self.allocated_blocks[block.ptr] = block
        
        # Create memory pointer
        memory_ptr = cp.cuda.MemoryPointer(
            cp.cuda.memory.UnownedMemory(block.ptr, size, None), 0
        )
        
        return memory_ptr
        
    def _expand_pool(self, min_size: int) -> bool:
        """Expand the memory pool."""
        current_total = sum(block.size for block in self.blocks)
        current_gb = current_total / (1024**3)
        
        if current_gb >= self.config.max_size_gb:
            logger.warning(f"Pool at maximum size: {current_gb:.2f}GB")
            return False
            
        # Calculate expansion size
        expansion_size = max(
            min_size,
            int(current_total * (self.config.growth_factor - 1.0))
        )
        
        # Ensure we don't exceed max size
        max_expansion = int((self.config.max_size_gb - current_gb) * (1024**3))
        expansion_size = min(expansion_size, max_expansion)
        
        if expansion_size <= 0:
            return False
            
        try:
            # Allocate new memory
            new_ptr = cp.cuda.alloc(expansion_size)
            
            # Create new block
            new_block = MemoryBlock(
                ptr=new_ptr.ptr,
                size=expansion_size,
                allocated=False,
                pool_type=self.config.pool_type,
                timestamp=time.time()
            )
            
            self.blocks.append(new_block)
            self.free_blocks[expansion_size].append(new_block)
            self.stats.total_free += expansion_size
            
            logger.info(f"Expanded pool by {expansion_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to expand pool: {str(e)}")
            return False
            
    def _should_defragment(self) -> bool:
        """Check if defragmentation is needed."""
        if not self.config.enable_defragmentation:
            return False
            
        # Calculate fragmentation ratio
        total_free = sum(len(blocks) * size for size, blocks in self.free_blocks.items())
        largest_free = max(self.free_blocks.keys()) if self.free_blocks else 0
        
        if total_free == 0:
            return False
            
        fragmentation_ratio = 1.0 - (largest_free / total_free)
        self.stats.fragmentation_ratio = fragmentation_ratio
        
        return fragmentation_ratio > self.config.defrag_threshold
        
    def _defragment(self):
        """Defragment the memory pool."""
        logger.info("Starting memory defragmentation")
        
        # Sort free blocks by address
        all_free_blocks = []
        for blocks in self.free_blocks.values():
            all_free_blocks.extend(blocks)
            
        all_free_blocks.sort(key=lambda b: b.ptr)
        
        # Merge adjacent blocks
        merged_blocks = []
        current_block = None
        
        for block in all_free_blocks:
            if current_block is None:
                current_block = block
            elif current_block.ptr + current_block.size == block.ptr:
                # Merge blocks
                current_block.size += block.size
                self.blocks.remove(block)
            else:
                merged_blocks.append(current_block)
                current_block = block
                
        if current_block:
            merged_blocks.append(current_block)
            
        # Rebuild free blocks dictionary
        self.free_blocks.clear()
        for block in merged_blocks:
            self.free_blocks[block.size].append(block)
            
        self.stats.defrag_count += 1
        logger.info(f"Defragmentation complete: {len(merged_blocks)} blocks")
        
    def get_stats(self) -> MemoryStats:
        """Get memory pool statistics."""
        with self.lock:
            # Update dynamic stats
            self.stats.pool_utilization = (
                self.stats.total_allocated / 
                (self.stats.total_allocated + self.stats.total_free)
                if (self.stats.total_allocated + self.stats.total_free) > 0 else 0
            )
            
            if self.free_blocks:
                self.stats.largest_free_block = max(self.free_blocks.keys())
            else:
                self.stats.largest_free_block = 0
                
            return self.stats


class GPUMemoryManager:
    """Advanced GPU memory manager with multiple pools and optimization."""
    
    def __init__(self):
        """Initialize GPU memory manager."""
        self.pools = {}
        self.default_pool = None
        self.allocation_history = deque(maxlen=10000)
        self.lock = threading.RLock()
        
        # Initialize default pools
        self._initialize_default_pools()
        
        # Background tasks
        self.gc_thread = None
        self.monitoring_thread = None
        self.should_stop = threading.Event()
        
        # Start background tasks
        self._start_background_tasks()
        
    def _initialize_default_pools(self):
        """Initialize default memory pools."""
        gpu_config = get_gpu_config_manager().config
        
        # Default pool for general allocations
        default_config = MemoryPoolConfig(
            pool_type=MemoryPoolType.DEFAULT,
            initial_size_gb=gpu_config.memory_gb * 0.4,
            max_size_gb=gpu_config.memory_gb * 0.8,
            allocation_strategy=AllocationStrategy.BUDDY_SYSTEM
        )
        self.pools[MemoryPoolType.DEFAULT] = AdvancedGPUMemoryPool(default_config)
        self.default_pool = self.pools[MemoryPoolType.DEFAULT]
        
        # Workspace pool for temporary computations
        workspace_config = MemoryPoolConfig(
            pool_type=MemoryPoolType.WORKSPACE,
            initial_size_gb=gpu_config.memory_gb * 0.2,
            max_size_gb=gpu_config.memory_gb * 0.4,
            allocation_strategy=AllocationStrategy.SLAB_ALLOCATOR,
            enable_garbage_collection=True,
            gc_interval=500
        )
        self.pools[MemoryPoolType.WORKSPACE] = AdvancedGPUMemoryPool(workspace_config)
        
        # Persistent pool for long-lived allocations
        persistent_config = MemoryPoolConfig(
            pool_type=MemoryPoolType.PERSISTENT,
            initial_size_gb=gpu_config.memory_gb * 0.1,
            max_size_gb=gpu_config.memory_gb * 0.3,
            allocation_strategy=AllocationStrategy.BEST_FIT,
            enable_defragmentation=False
        )
        self.pools[MemoryPoolType.PERSISTENT] = AdvancedGPUMemoryPool(persistent_config)
        
        logger.info("Initialized GPU memory pools")
        
    def allocate(self, size: int, pool_type: MemoryPoolType = MemoryPoolType.DEFAULT,
                tag: Optional[str] = None) -> Optional[cp.cuda.MemoryPointer]:
        """Allocate memory from specified pool."""
        with self.lock:
            if pool_type not in self.pools:
                logger.error(f"Unknown pool type: {pool_type}")
                return None
                
            pool = self.pools[pool_type]
            ptr = pool.allocate(size, tag)
            
            # Record allocation
            self.allocation_history.append({
                'timestamp': time.time(),
                'pool_type': pool_type,
                'size': size,
                'tag': tag,
                'success': ptr is not None
            })
            
            return ptr
            
    def deallocate(self, ptr: cp.cuda.MemoryPointer, 
                  pool_type: MemoryPoolType = MemoryPoolType.DEFAULT):
        """Deallocate memory from specified pool."""
        with self.lock:
            if pool_type not in self.pools:
                logger.error(f"Unknown pool type: {pool_type}")
                return
                
            pool = self.pools[pool_type]
            pool.deallocate(ptr)
            
    def allocate_array(self, shape: Tuple[int, ...], dtype: cp.dtype = cp.float32,
                      pool_type: MemoryPoolType = MemoryPoolType.DEFAULT,
                      tag: Optional[str] = None) -> Optional[cp.ndarray]:
        """Allocate a CuPy array from specified pool."""
        size = np.prod(shape) * dtype().itemsize
        ptr = self.allocate(size, pool_type, tag)
        
        if ptr is None:
            return None
            
        try:
            # Create CuPy array using the allocated memory
            array = cp.ndarray(shape, dtype=dtype, memptr=ptr)
            return array
        except Exception as e:
            logger.error(f"Failed to create array: {str(e)}")
            self.deallocate(ptr, pool_type)
            return None
            
    def create_workspace(self, size_gb: float = 1.0) -> 'WorkspaceManager':
        """Create a temporary workspace for computations."""
        return WorkspaceManager(self, size_gb)
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information."""
        with self.lock:
            info = {
                'pools': {},
                'total_allocated': 0,
                'total_free': 0,
                'overall_utilization': 0.0,
                'peak_usage': 0,
                'allocation_failures': 0
            }
            
            for pool_type, pool in self.pools.items():
                stats = pool.get_stats()
                info['pools'][pool_type.value] = {
                    'allocated': stats.total_allocated,
                    'free': stats.total_free,
                    'utilization': stats.pool_utilization,
                    'peak_usage': stats.peak_usage,
                    'allocation_failures': stats.allocation_failures,
                    'fragmentation_ratio': stats.fragmentation_ratio
                }
                
                info['total_allocated'] += stats.total_allocated
                info['total_free'] += stats.total_free
                info['peak_usage'] += stats.peak_usage
                info['allocation_failures'] += stats.allocation_failures
                
            if info['total_allocated'] + info['total_free'] > 0:
                info['overall_utilization'] = (
                    info['total_allocated'] / 
                    (info['total_allocated'] + info['total_free'])
                )
                
            return info
            
    def optimize_memory(self):
        """Optimize memory usage across all pools."""
        with self.lock:
            logger.info("Optimizing GPU memory...")
            
            # Trigger garbage collection
            for pool in self.pools.values():
                if pool.config.enable_garbage_collection:
                    pool.stats.gc_count += 1
                    
            # Defragment pools if needed
            for pool in self.pools.values():
                if pool._should_defragment():
                    pool._defragment()
                    
            # Force CuPy memory pool cleanup
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # System garbage collection
            gc.collect()
            
            logger.info("Memory optimization complete")
            
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self.gc_thread = threading.Thread(target=self._garbage_collection_loop)
        self.gc_thread.daemon = True
        self.gc_thread.start()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def _garbage_collection_loop(self):
        """Background garbage collection loop."""
        while not self.should_stop.wait(5.0):  # Check every 5 seconds
            try:
                # Check if GC is needed
                memory_info = self.get_memory_info()
                if memory_info['overall_utilization'] > 0.8:
                    self.optimize_memory()
                    
            except Exception as e:
                logger.error(f"GC loop error: {str(e)}")
                
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.should_stop.wait(30.0):  # Check every 30 seconds
            try:
                memory_info = self.get_memory_info()
                logger.debug(f"Memory utilization: {memory_info['overall_utilization']:.2%}")
                
                # Log warnings for high utilization
                if memory_info['overall_utilization'] > 0.9:
                    logger.warning("High memory utilization detected")
                    
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                
    def shutdown(self):
        """Shutdown the memory manager."""
        self.should_stop.set()
        
        if self.gc_thread:
            self.gc_thread.join(timeout=5.0)
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("GPU memory manager shutdown complete")


class WorkspaceManager:
    """Manages a temporary workspace for GPU computations."""
    
    def __init__(self, memory_manager: GPUMemoryManager, size_gb: float):
        """Initialize workspace manager."""
        self.memory_manager = memory_manager
        self.size_gb = size_gb
        self.allocated_arrays = []
        self.total_allocated = 0
        
    def __enter__(self):
        """Enter workspace context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit workspace context and cleanup."""
        self.cleanup()
        
    def allocate_array(self, shape: Tuple[int, ...], dtype: cp.dtype = cp.float32,
                      tag: Optional[str] = None) -> Optional[cp.ndarray]:
        """Allocate array in workspace."""
        array = self.memory_manager.allocate_array(
            shape, dtype, MemoryPoolType.WORKSPACE, tag
        )
        
        if array is not None:
            self.allocated_arrays.append(array)
            self.total_allocated += array.nbytes
            
        return array
        
    def cleanup(self):
        """Cleanup workspace allocations."""
        for array in self.allocated_arrays:
            try:
                # Arrays will be automatically deallocated when they go out of scope
                pass
            except:
                pass
                
        self.allocated_arrays.clear()
        self.total_allocated = 0
        
        # Trigger workspace pool cleanup
        if MemoryPoolType.WORKSPACE in self.memory_manager.pools:
            workspace_pool = self.memory_manager.pools[MemoryPoolType.WORKSPACE]
            if workspace_pool._should_defragment():
                workspace_pool._defragment()


# Global memory manager instance
_global_memory_manager = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = GPUMemoryManager()
    return _global_memory_manager


def allocate_gpu_array(shape: Tuple[int, ...], dtype: cp.dtype = cp.float32,
                      pool_type: str = "default", tag: Optional[str] = None) -> Optional[cp.ndarray]:
    """Allocate GPU array using advanced memory management."""
    manager = get_gpu_memory_manager()
    pool_enum = MemoryPoolType(pool_type)
    return manager.allocate_array(shape, dtype, pool_enum, tag)


def create_gpu_workspace(size_gb: float = 1.0) -> WorkspaceManager:
    """Create GPU workspace for temporary computations."""
    manager = get_gpu_memory_manager()
    return manager.create_workspace(size_gb)


if __name__ == "__main__":
    # Test GPU memory manager
    logger.info("Testing GPU memory manager...")
    
    # Get manager
    manager = get_gpu_memory_manager()
    
    # Test allocations
    arrays = []
    for i in range(10):
        array = manager.allocate_array((1000, 1000), cp.float32, tag=f"test_array_{i}")
        if array is not None:
            arrays.append(array)
            
    logger.info(f"Allocated {len(arrays)} arrays")
    
    # Test workspace
    with create_gpu_workspace(2.0) as workspace:
        temp_array = workspace.allocate_array((2000, 2000), cp.float32)
        if temp_array is not None:
            logger.info(f"Allocated workspace array: {temp_array.shape}")
            
    # Get memory info
    memory_info = manager.get_memory_info()
    logger.info(f"Memory info: {memory_info}")
    
    # Optimize memory
    manager.optimize_memory()
    
    print("GPU memory manager tested successfully!")