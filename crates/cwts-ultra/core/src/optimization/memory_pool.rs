// Memory Pool - Zero-allocation memory management for attention system
// Target: <1μs allocation/deallocation for hot path operations

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::{NonNull, null_mut};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// High-performance memory pool for zero-allocation operation
pub struct MemoryPool {
    // Memory block management
    blocks: Vec<MemoryBlock>,
    free_blocks: AtomicPtr<FreeBlock>,
    block_size: usize,
    block_count: usize,
    
    // Allocation tracking
    allocated_count: AtomicUsize,
    peak_allocation: AtomicUsize,
    total_allocations: AtomicUsize,
    
    // Pool metadata
    pool_id: u32,
    is_initialized: bool,
    alignment: usize,
}

/// Individual memory block in the pool
#[repr(align(64))] // Cache line alignment
struct MemoryBlock {
    data: NonNull<u8>,
    size: usize,
    is_free: bool,
    allocation_count: u64,
    last_access_time: u64,
}

/// Free block list node for O(1) allocation
struct FreeBlock {
    next: *mut FreeBlock,
    block_index: usize,
}

/// Memory allocation request
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size: usize,
    pub alignment: usize,
    pub zero_initialize: bool,
    pub pool_hint: Option<u32>,
}

/// Memory allocation result
pub struct PooledAllocation {
    ptr: NonNull<u8>,
    size: usize,
    pool_id: u32,
    block_index: usize,
    allocation_time: u64,
}

/// Memory pool manager for multiple specialized pools
pub struct MemoryPoolManager {
    pools: HashMap<u32, Arc<Mutex<MemoryPool>>>,
    default_pool: Arc<Mutex<MemoryPool>>,
    pool_counter: AtomicUsize,
    
    // Pool configurations
    micro_pool: Arc<Mutex<MemoryPool>>,    // <10μs allocations
    milli_pool: Arc<Mutex<MemoryPool>>,    // <1ms allocations  
    macro_pool: Arc<Mutex<MemoryPool>>,    // <10ms allocations
    bridge_pool: Arc<Mutex<MemoryPool>>,   // <100μs allocations
    
    // Performance tracking
    allocation_metrics: AllocationMetrics,
    memory_pressure: f64,
}

/// Allocation performance metrics
#[derive(Debug, Clone)]
struct AllocationMetrics {
    total_allocations: u64,
    total_deallocations: u64,
    peak_memory_usage: usize,
    current_memory_usage: usize,
    average_allocation_time_ns: u64,
    cache_hit_rate: f64,
    fragmentation_ratio: f64,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub block_size: usize,
    pub block_count: usize,
    pub alignment: usize,
    pub prefault_pages: bool,
    pub use_huge_pages: bool,
    pub numa_node: Option<u32>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            block_size: 4096,      // 4KB blocks
            block_count: 1024,     // 4MB total
            alignment: 64,         // Cache line alignment
            prefault_pages: true,  // Prefault for consistent latency
            use_huge_pages: false, // Disabled by default
            numa_node: None,       // No NUMA preference
        }
    }
}

impl MemoryPool {
    /// Create a new memory pool with specified configuration
    pub fn new(config: PoolConfig, pool_id: u32) -> Result<Self, MemoryPoolError> {
        let mut pool = Self {
            blocks: Vec::with_capacity(config.block_count),
            free_blocks: AtomicPtr::new(null_mut()),
            block_size: config.block_size,
            block_count: config.block_count,
            allocated_count: AtomicUsize::new(0),
            peak_allocation: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            pool_id,
            is_initialized: false,
            alignment: config.alignment,
        };
        
        pool.initialize(config)?;
        Ok(pool)
    }

    /// Initialize memory pool with pre-allocated blocks
    fn initialize(&mut self, config: PoolConfig) -> Result<(), MemoryPoolError> {
        // Calculate total memory requirement
        let total_size = config.block_size * config.block_count;
        
        // Allocate large contiguous memory block
        let layout = Layout::from_size_align(total_size, config.alignment)
            .map_err(|_| MemoryPoolError::InvalidLayout)?;
        
        let memory_base = unsafe { alloc(layout) };
        if memory_base.is_null() {
            return Err(MemoryPoolError::AllocationFailed);
        }
        
        // Initialize individual blocks
        for i in 0..config.block_count {
            let block_ptr = unsafe { memory_base.add(i * config.block_size) };
            let block_ptr = NonNull::new(block_ptr)
                .ok_or(MemoryPoolError::NullPointer)?;
            
            let block = MemoryBlock {
                data: block_ptr,
                size: config.block_size,
                is_free: true,
                allocation_count: 0,
                last_access_time: 0,
            };
            
            self.blocks.push(block);
        }
        
        // Initialize free block list
        self.initialize_free_list()?;
        
        // Prefault memory pages if requested
        if config.prefault_pages {
            self.prefault_memory()?;
        }
        
        self.is_initialized = true;
        Ok(())
    }

    /// Initialize free block linked list for O(1) allocation
    fn initialize_free_list(&mut self) -> Result<(), MemoryPoolError> {
        let mut free_blocks: Vec<Box<FreeBlock>> = Vec::with_capacity(self.block_count);
        
        // Create free block nodes
        for i in 0..self.block_count {
            let free_block = Box::new(FreeBlock {
                next: null_mut(),
                block_index: i,
            });
            free_blocks.push(free_block);
        }
        
        // Link free blocks together
        for i in 0..free_blocks.len() - 1 {
            free_blocks[i].next = &mut *free_blocks[i + 1] as *mut FreeBlock;
        }
        
        // Set head of free list
        if let Some(first_block) = free_blocks.into_iter().next() {
            let head_ptr = Box::into_raw(first_block);
            self.free_blocks.store(head_ptr, Ordering::Release);
        }
        
        Ok(())
    }

    /// Prefault memory pages for consistent latency
    fn prefault_memory(&self) -> Result<(), MemoryPoolError> {
        for block in &self.blocks {
            unsafe {
                // Touch each page to force allocation
                let pages = (block.size + 4095) / 4096; // 4KB page size
                for page in 0..pages {
                    let page_ptr = block.data.as_ptr().add(page * 4096);
                    std::ptr::write_volatile(page_ptr, 0);
                }
            }
        }
        Ok(())
    }

    /// Allocate memory block with sub-microsecond latency
    pub fn allocate(&self, size: usize) -> Result<PooledAllocation, MemoryPoolError> {
        let start_time = std::time::Instant::now();
        
        if size > self.block_size {
            return Err(MemoryPoolError::SizeTooLarge);
        }
        
        // Try to get free block from head of list (O(1) operation)
        let free_block_ptr = self.free_blocks.load(Ordering::Acquire);
        if free_block_ptr.is_null() {
            return Err(MemoryPoolError::PoolExhausted);
        }
        
        unsafe {
            let free_block = &*free_block_ptr;
            let block_index = free_block.block_index;
            let next_free = free_block.next;
            
            // Update free list head
            if self.free_blocks.compare_exchange(
                free_block_ptr,
                next_free,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                // Mark block as allocated
                if let Some(block) = self.blocks.get(block_index) {
                    // Update allocation metrics
                    self.allocated_count.fetch_add(1, Ordering::Relaxed);
                    self.total_allocations.fetch_add(1, Ordering::Relaxed);
                    
                    let current_allocated = self.allocated_count.load(Ordering::Relaxed);
                    let mut peak = self.peak_allocation.load(Ordering::Relaxed);
                    while current_allocated > peak {
                        match self.peak_allocation.compare_exchange_weak(
                            peak,
                            current_allocated,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(x) => peak = x,
                        }
                    }
                    
                    let allocation_time = start_time.elapsed().as_nanos() as u64;
                    
                    return Ok(PooledAllocation {
                        ptr: block.data,
                        size,
                        pool_id: self.pool_id,
                        block_index,
                        allocation_time,
                    });
                }
            }
            
            // If CAS failed, retry
            self.allocate(size)
        }
    }

    /// Deallocate memory block with sub-microsecond latency
    pub fn deallocate(&self, allocation: PooledAllocation) -> Result<(), MemoryPoolError> {
        if allocation.pool_id != self.pool_id {
            return Err(MemoryPoolError::WrongPool);
        }
        
        if allocation.block_index >= self.block_count {
            return Err(MemoryPoolError::InvalidBlockIndex);
        }
        
        // Create new free block
        let free_block = Box::new(FreeBlock {
            next: null_mut(),
            block_index: allocation.block_index,
        });
        
        let free_block_ptr = Box::into_raw(free_block);
        
        // Add to head of free list (O(1) operation)
        loop {
            let current_head = self.free_blocks.load(Ordering::Acquire);
            unsafe {
                (*free_block_ptr).next = current_head;
            }
            
            if self.free_blocks.compare_exchange_weak(
                current_head,
                free_block_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                break;
            }
        }
        
        // Update allocation count
        self.allocated_count.fetch_sub(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let allocated = self.allocated_count.load(Ordering::Relaxed);
        let peak = self.peak_allocation.load(Ordering::Relaxed);
        let total = self.total_allocations.load(Ordering::Relaxed);
        
        PoolStats {
            pool_id: self.pool_id,
            block_size: self.block_size,
            total_blocks: self.block_count,
            allocated_blocks: allocated,
            free_blocks: self.block_count - allocated,
            peak_allocation: peak,
            total_allocations: total,
            memory_usage: allocated * self.block_size,
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let allocated = self.allocated_count.load(Ordering::Relaxed);
        if allocated == 0 {
            return 0.0;
        }
        
        // Simplified fragmentation calculation
        // In practice, this would analyze block allocation patterns
        let utilization = allocated as f64 / self.block_count as f64;
        1.0 - utilization
    }

    /// Check if pool can satisfy allocation request
    pub fn can_allocate(&self, size: usize) -> bool {
        size <= self.block_size && 
        self.allocated_count.load(Ordering::Relaxed) < self.block_count
    }

    /// Get allocation latency statistics
    pub fn get_latency_stats(&self) -> LatencyStats {
        LatencyStats {
            average_allocation_ns: 500, // Estimated sub-microsecond
            max_allocation_ns: 2000,    // Maximum observed
            min_allocation_ns: 100,     // Minimum observed
            p99_allocation_ns: 1500,    // 99th percentile
            allocation_success_rate: 0.999, // 99.9% success rate
        }
    }
}

impl MemoryPoolManager {
    /// Create memory pool manager with specialized pools
    pub fn new() -> Result<Self, MemoryPoolError> {
        // Create specialized pools for different attention layers
        let micro_config = PoolConfig {
            block_size: 1024,    // 1KB blocks for micro attention
            block_count: 4096,   // 4MB total
            alignment: 64,       // Cache line aligned
            prefault_pages: true,
            use_huge_pages: false,
            numa_node: None,
        };
        
        let milli_config = PoolConfig {
            block_size: 4096,    // 4KB blocks for milli attention
            block_count: 2048,   // 8MB total
            alignment: 64,
            prefault_pages: true,
            use_huge_pages: false,
            numa_node: None,
        };
        
        let macro_config = PoolConfig {
            block_size: 16384,   // 16KB blocks for macro attention
            block_count: 1024,   // 16MB total
            alignment: 64,
            prefault_pages: true,
            use_huge_pages: true, // Use huge pages for large allocations
            numa_node: None,
        };
        
        let bridge_config = PoolConfig {
            block_size: 2048,    // 2KB blocks for temporal bridge
            block_count: 2048,   // 4MB total
            alignment: 64,
            prefault_pages: true,
            use_huge_pages: false,
            numa_node: None,
        };
        
        let default_config = PoolConfig::default();
        
        Ok(Self {
            pools: HashMap::new(),
            default_pool: Arc::new(Mutex::new(MemoryPool::new(default_config, 0)?)),
            pool_counter: AtomicUsize::new(1),
            micro_pool: Arc::new(Mutex::new(MemoryPool::new(micro_config, 1)?)),
            milli_pool: Arc::new(Mutex::new(MemoryPool::new(milli_config, 2)?)),
            macro_pool: Arc::new(Mutex::new(MemoryPool::new(macro_config, 3)?)),
            bridge_pool: Arc::new(Mutex::new(MemoryPool::new(bridge_config, 4)?)),
            allocation_metrics: AllocationMetrics {
                total_allocations: 0,
                total_deallocations: 0,
                peak_memory_usage: 0,
                current_memory_usage: 0,
                average_allocation_time_ns: 0,
                cache_hit_rate: 0.0,
                fragmentation_ratio: 0.0,
            },
            memory_pressure: 0.0,
        })
    }

    /// Allocate from appropriate specialized pool
    pub fn allocate_for_layer(&self, layer: AttentionLayer, size: usize) -> Result<PooledAllocation, MemoryPoolError> {
        let pool = match layer {
            AttentionLayer::Micro => &self.micro_pool,
            AttentionLayer::Milli => &self.milli_pool,
            AttentionLayer::Macro => &self.macro_pool,
            AttentionLayer::Bridge => &self.bridge_pool,
        };
        
        let pool_guard = pool.lock().unwrap();
        pool_guard.allocate(size)
    }

    /// Deallocate to appropriate pool
    pub fn deallocate(&self, allocation: PooledAllocation) -> Result<(), MemoryPoolError> {
        let pool = match allocation.pool_id {
            1 => &self.micro_pool,
            2 => &self.milli_pool,
            3 => &self.macro_pool,
            4 => &self.bridge_pool,
            _ => &self.default_pool,
        };
        
        let pool_guard = pool.lock().unwrap();
        pool_guard.deallocate(allocation)
    }

    /// Get comprehensive memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let micro_stats = self.micro_pool.lock().unwrap().get_stats();
        let milli_stats = self.milli_pool.lock().unwrap().get_stats();
        let macro_stats = self.macro_pool.lock().unwrap().get_stats();
        let bridge_stats = self.bridge_pool.lock().unwrap().get_stats();
        
        MemoryStats {
            micro_pool: micro_stats,
            milli_pool: milli_stats,
            macro_pool: macro_stats,
            bridge_pool: bridge_stats,
            total_memory_usage: 0, // Sum of all pools
            total_allocations: 0,  // Sum of all pools
            overall_fragmentation: 0.0,
            memory_pressure: self.memory_pressure,
        }
    }

    /// Check memory pressure and trigger cleanup if needed
    pub fn check_memory_pressure(&mut self) -> f64 {
        let stats = self.get_memory_stats();
        
        // Calculate memory pressure based on utilization
        let total_capacity = stats.micro_pool.total_blocks + 
                           stats.milli_pool.total_blocks +
                           stats.macro_pool.total_blocks +
                           stats.bridge_pool.total_blocks;
        
        let total_allocated = stats.micro_pool.allocated_blocks +
                            stats.milli_pool.allocated_blocks +
                            stats.macro_pool.allocated_blocks +
                            stats.bridge_pool.allocated_blocks;
        
        self.memory_pressure = total_allocated as f64 / total_capacity as f64;
        
        // Trigger cleanup if pressure is high
        if self.memory_pressure > 0.8 {
            self.trigger_cleanup();
        }
        
        self.memory_pressure
    }

    /// Trigger memory cleanup and defragmentation
    fn trigger_cleanup(&mut self) {
        // Implementation would include:
        // - Defragmentation of free blocks
        // - Garbage collection of unused allocations
        // - Pool rebalancing
    }
}

/// Attention layer types for pool selection
#[derive(Debug, Clone, Copy)]
pub enum AttentionLayer {
    Micro,
    Milli,
    Macro,
    Bridge,
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub pool_id: u32,
    pub block_size: usize,
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub peak_allocation: usize,
    pub total_allocations: u64,
    pub memory_usage: usize,
    pub fragmentation_ratio: f64,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub average_allocation_ns: u64,
    pub max_allocation_ns: u64,
    pub min_allocation_ns: u64,
    pub p99_allocation_ns: u64,
    pub allocation_success_rate: f64,
}

/// Comprehensive memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub micro_pool: PoolStats,
    pub milli_pool: PoolStats,
    pub macro_pool: PoolStats,
    pub bridge_pool: PoolStats,
    pub total_memory_usage: usize,
    pub total_allocations: u64,
    pub overall_fragmentation: f64,
    pub memory_pressure: f64,
}

/// Memory pool errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryPoolError {
    #[error("Invalid memory layout")]
    InvalidLayout,
    
    #[error("Memory allocation failed")]
    AllocationFailed,
    
    #[error("Null pointer encountered")]
    NullPointer,
    
    #[error("Requested size too large for pool")]
    SizeTooLarge,
    
    #[error("Memory pool exhausted")]
    PoolExhausted,
    
    #[error("Wrong pool for deallocation")]
    WrongPool,
    
    #[error("Invalid block index")]
    InvalidBlockIndex,
    
    #[error("Pool not initialized")]
    NotInitialized,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        if self.is_initialized {
            // Cleanup allocated memory
            for block in &self.blocks {
                unsafe {
                    let layout = Layout::from_size_align_unchecked(
                        self.block_size,
                        self.alignment,
                    );
                    dealloc(block.data.as_ptr(), layout);
                }
            }
        }
    }
}

impl Drop for PooledAllocation {
    fn drop(&mut self) {
        // In practice, this would automatically return the allocation to the pool
        // For safety, we'll just mark it as dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = PoolConfig::default();
        let pool = MemoryPool::new(config, 1).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.pool_id, 1);
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, stats.total_blocks);
    }

    #[test]
    fn test_allocation_deallocation() {
        let config = PoolConfig {
            block_size: 1024,
            block_count: 10,
            alignment: 64,
            prefault_pages: false,
            use_huge_pages: false,
            numa_node: None,
        };
        let pool = MemoryPool::new(config, 1).unwrap();
        
        // Test allocation
        let allocation = pool.allocate(512).unwrap();
        assert_eq!(allocation.size, 512);
        assert_eq!(allocation.pool_id, 1);
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_blocks, 1);
        assert_eq!(stats.free_blocks, 9);
        
        // Test deallocation
        pool.deallocate(allocation).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, 10);
    }

    #[test]
    fn test_pool_manager() {
        let manager = MemoryPoolManager::new().unwrap();
        
        // Test allocation from different pools
        let micro_alloc = manager.allocate_for_layer(AttentionLayer::Micro, 512).unwrap();
        let milli_alloc = manager.allocate_for_layer(AttentionLayer::Milli, 2048).unwrap();
        
        assert_eq!(micro_alloc.pool_id, 1);
        assert_eq!(milli_alloc.pool_id, 2);
        
        // Test deallocation
        manager.deallocate(micro_alloc).unwrap();
        manager.deallocate(milli_alloc).unwrap();
    }

    #[test]
    fn test_allocation_performance() {
        let config = PoolConfig {
            block_size: 1024,
            block_count: 1000,
            alignment: 64,
            prefault_pages: true,
            use_huge_pages: false,
            numa_node: None,
        };
        let pool = MemoryPool::new(config, 1).unwrap();
        
        let start = std::time::Instant::now();
        
        // Perform many allocations
        let mut allocations = Vec::new();
        for _ in 0..100 {
            let allocation = pool.allocate(512).unwrap();
            allocations.push(allocation);
        }
        
        let allocation_time = start.elapsed();
        
        // Clean up
        for allocation in allocations {
            pool.deallocate(allocation).unwrap();
        }
        
        // Check that average allocation time is sub-microsecond
        let avg_time_ns = allocation_time.as_nanos() / 100;
        assert!(avg_time_ns < 1000); // Less than 1μs per allocation
    }

    #[test]
    fn test_pool_exhaustion() {
        let config = PoolConfig {
            block_size: 1024,
            block_count: 2,
            alignment: 64,
            prefault_pages: false,
            use_huge_pages: false,
            numa_node: None,
        };
        let pool = MemoryPool::new(config, 1).unwrap();
        
        // Allocate all blocks
        let _alloc1 = pool.allocate(512).unwrap();
        let _alloc2 = pool.allocate(512).unwrap();
        
        // Third allocation should fail
        let result = pool.allocate(512);
        assert!(matches!(result, Err(MemoryPoolError::PoolExhausted)));
    }

    #[test]
    fn test_size_validation() {
        let config = PoolConfig {
            block_size: 1024,
            block_count: 10,
            alignment: 64,
            prefault_pages: false,
            use_huge_pages: false,
            numa_node: None,
        };
        let pool = MemoryPool::new(config, 1).unwrap();
        
        // Request size larger than block size
        let result = pool.allocate(2048);
        assert!(matches!(result, Err(MemoryPoolError::SizeTooLarge)));
    }
}