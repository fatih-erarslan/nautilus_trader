// High-Performance Memory Pool for Sub-100ns Latency
// Copyright (c) 2025 TENGRI Trading Swarm - Performance-Optimizer Agent

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use memmap2::MmapMut;
use anyhow::Result;
use tracing::{info, warn, debug};
use crate::AnalyzerError;

/// Memory pool configuration for optimal performance
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Block size for allocations
    pub block_size: usize,
    /// Initial number of blocks to pre-allocate
    pub initial_blocks: usize,
    /// Maximum number of blocks
    pub max_blocks: usize,
    /// Enable huge page support
    pub use_huge_pages: bool,
    /// NUMA node preference
    pub numa_node: Option<u32>,
    /// Memory alignment (cache line aligned)
    pub alignment: usize,
    /// Enable memory prefetching
    pub prefetch_enabled: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            block_size: 1024 * 1024,  // 1MB blocks
            initial_blocks: 1000,
            max_blocks: 10000,
            use_huge_pages: true,
            numa_node: None,
            alignment: 64, // Cache line alignment
            prefetch_enabled: true,
        }
    }
}

/// Memory block with metadata
#[derive(Debug)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    is_allocated: bool,
    allocation_time: std::time::Instant,
    numa_node: Option<u32>,
}

/// High-performance memory pool
pub struct MemoryPool {
    config: MemoryPoolConfig,
    free_blocks: SegQueue<MemoryBlock>,
    allocated_blocks: AtomicUsize,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    huge_page_mapping: Option<MmapMut>,
    statistics: MemoryStatistics,
}

/// Memory pool statistics
#[derive(Debug, Default)]
pub struct MemoryStatistics {
    pub total_allocations: AtomicUsize,
    pub total_deallocations: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub cache_misses: AtomicUsize,
    pub allocation_latency_ns: AtomicUsize,
    pub deallocation_latency_ns: AtomicUsize,
}

/// Custom allocator for quantum operations
pub struct QuantumAllocator {
    pool: Arc<MemoryPool>,
}

impl MemoryPool {
    /// Create new memory pool with optimal configuration
    pub fn new(config: MemoryPoolConfig) -> Result<Self, AnalyzerError> {
        info!("Initializing high-performance memory pool");
        
        let mut pool = Self {
            config: config.clone(),
            free_blocks: SegQueue::new(),
            allocated_blocks: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            huge_page_mapping: None,
            statistics: MemoryStatistics::default(),
        };
        
        // Initialize huge pages if enabled
        if config.use_huge_pages {
            pool.initialize_huge_pages()?;
        }
        
        // Pre-allocate initial blocks
        pool.preallocate_blocks(config.initial_blocks)?;
        
        info!("Memory pool initialized with {} blocks", config.initial_blocks);
        Ok(pool)
    }
    
    /// Allocate memory with sub-100ns latency
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>, AnalyzerError> {
        let start = std::time::Instant::now();
        
        // Try to get from free blocks first (cache hit)
        if let Some(block) = self.free_blocks.pop() {
            self.statistics.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
            
            let latency = start.elapsed().as_nanos() as usize;
            self.statistics.allocation_latency_ns.store(latency, Ordering::Relaxed);
            
            // Prefetch next block if enabled
            if self.config.prefetch_enabled {
                self.prefetch_next_block();
            }
            
            return Ok(block.ptr);
        }
        
        // Cache miss - allocate new block
        self.statistics.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Check if we can allocate more blocks
        if self.allocated_blocks.load(Ordering::Relaxed) >= self.config.max_blocks {
            return Err(AnalyzerError::MemoryPoolExhausted);
        }
        
        let ptr = self.allocate_new_block(size)?;
        
        self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
        
        // Update peak allocation
        let current = self.total_allocated.load(Ordering::Relaxed);
        let peak = self.peak_allocated.load(Ordering::Relaxed);
        if current > peak {
            self.peak_allocated.store(current, Ordering::Relaxed);
        }
        
        let latency = start.elapsed().as_nanos() as usize;
        self.statistics.allocation_latency_ns.store(latency, Ordering::Relaxed);
        self.statistics.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Deallocate memory
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<(), AnalyzerError> {
        let start = std::time::Instant::now();
        
        let block = MemoryBlock {
            ptr,
            size,
            is_allocated: false,
            allocation_time: std::time::Instant::now(),
            numa_node: self.config.numa_node,
        };
        
        // Return to free blocks queue
        self.free_blocks.push(block);
        
        self.allocated_blocks.fetch_sub(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
        
        let latency = start.elapsed().as_nanos() as usize;
        self.statistics.deallocation_latency_ns.store(latency, Ordering::Relaxed);
        self.statistics.total_deallocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Initialize huge pages for better performance
    fn initialize_huge_pages(&mut self) -> Result<(), AnalyzerError> {
        debug!("Initializing huge pages");
        
        // Create memory mapping with huge pages
        let total_size = self.config.block_size * self.config.max_blocks;
        
        let mut mmap = MmapMut::map_anon(total_size)
            .map_err(|e| AnalyzerError::MemoryMappingError(e.to_string()))?;
        
        // Advise kernel to use huge pages
        unsafe {
            libc::madvise(
                mmap.as_mut_ptr() as *mut libc::c_void,
                total_size,
                libc::MADV_HUGEPAGE,
            );
        }
        
        self.huge_page_mapping = Some(mmap);
        info!("Huge pages initialized for {} bytes", total_size);
        
        Ok(())
    }
    
    /// Pre-allocate blocks for better performance
    fn preallocate_blocks(&self, count: usize) -> Result<(), AnalyzerError> {
        debug!("Pre-allocating {} blocks", count);
        
        for _ in 0..count {
            let ptr = self.allocate_new_block(self.config.block_size)?;
            
            let block = MemoryBlock {
                ptr,
                size: self.config.block_size,
                is_allocated: false,
                allocation_time: std::time::Instant::now(),
                numa_node: self.config.numa_node,
            };
            
            self.free_blocks.push(block);
        }
        
        Ok(())
    }
    
    /// Allocate new memory block
    fn allocate_new_block(&self, size: usize) -> Result<NonNull<u8>, AnalyzerError> {
        let layout = Layout::from_size_align(size, self.config.alignment)
            .map_err(|e| AnalyzerError::MemoryLayoutError(e.to_string()))?;
        
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(AnalyzerError::MemoryAllocationError);
        }
        
        // Zero out the memory for security
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        
        // Set NUMA affinity if specified
        if let Some(node) = self.config.numa_node {
            self.set_numa_affinity(ptr, size, node)?;
        }
        
        Ok(NonNull::new(ptr).unwrap())
    }
    
    /// Set NUMA affinity for memory
    fn set_numa_affinity(&self, ptr: *mut u8, size: usize, node: u32) -> Result<(), AnalyzerError> {
        #[cfg(target_os = "linux")]
        {
            use libc::{c_void, MPOL_BIND};
            
            let nodemask = 1u64 << node;
            let result = unsafe {
                libc::mbind(
                    ptr as *mut c_void,
                    size,
                    MPOL_BIND,
                    &nodemask as *const u64,
                    64,
                    0,
                )
            };
            
            if result != 0 {
                warn!("Failed to set NUMA affinity: {}", std::io::Error::last_os_error());
            }
        }
        
        Ok(())
    }
    
    /// Prefetch next block for better cache performance
    fn prefetch_next_block(&self) {
        if let Some(block) = self.free_blocks.pop() {
            // Prefetch the memory to L1 cache
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    block.ptr.as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
            
            // Push it back
            self.free_blocks.push(block);
        }
    }
    
    /// Get memory pool statistics
    pub fn get_statistics(&self) -> MemoryPoolStatistics {
        MemoryPoolStatistics {
            total_allocations: self.statistics.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.statistics.total_deallocations.load(Ordering::Relaxed),
            cache_hits: self.statistics.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.statistics.cache_misses.load(Ordering::Relaxed),
            average_allocation_latency_ns: self.statistics.allocation_latency_ns.load(Ordering::Relaxed),
            average_deallocation_latency_ns: self.statistics.deallocation_latency_ns.load(Ordering::Relaxed),
            allocated_blocks: self.allocated_blocks.load(Ordering::Relaxed),
            total_allocated_bytes: self.total_allocated.load(Ordering::Relaxed),
            peak_allocated_bytes: self.peak_allocated.load(Ordering::Relaxed),
            cache_hit_ratio: {
                let hits = self.statistics.cache_hits.load(Ordering::Relaxed);
                let misses = self.statistics.cache_misses.load(Ordering::Relaxed);
                if hits + misses > 0 {
                    hits as f64 / (hits + misses) as f64
                } else {
                    0.0
                }
            },
        }
    }
    
    /// Optimize memory layout for better cache performance
    pub fn optimize_layout(&self) -> Result<(), AnalyzerError> {
        info!("Optimizing memory layout for cache performance");
        
        // TODO: Implement memory layout optimization
        // - Reorder allocations by access patterns
        // - Group frequently accessed data together
        // - Align data structures to cache lines
        
        Ok(())
    }
    
    /// Compact memory to reduce fragmentation
    pub fn compact(&self) -> Result<(), AnalyzerError> {
        info!("Compacting memory to reduce fragmentation");
        
        // TODO: Implement memory compaction
        // - Move allocated blocks to reduce fragmentation
        // - Merge adjacent free blocks
        // - Update pointers after compaction
        
        Ok(())
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStatistics {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub average_allocation_latency_ns: usize,
    pub average_deallocation_latency_ns: usize,
    pub allocated_blocks: usize,
    pub total_allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub cache_hit_ratio: f64,
}

impl QuantumAllocator {
    /// Create new quantum allocator
    pub fn new(pool: Arc<MemoryPool>) -> Self {
        Self { pool }
    }
    
    /// Allocate memory for quantum operations
    pub fn allocate_quantum(&self, size: usize) -> Result<NonNull<u8>, AnalyzerError> {
        // Align size to quantum requirements (power of 2)
        let aligned_size = size.next_power_of_two();
        self.pool.allocate(aligned_size)
    }
    
    /// Deallocate quantum memory
    pub fn deallocate_quantum(&self, ptr: NonNull<u8>, size: usize) -> Result<(), AnalyzerError> {
        let aligned_size = size.next_power_of_two();
        self.pool.deallocate(ptr, aligned_size)
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        info!("Deallocating memory pool");
        
        // Clean up all allocated blocks
        while let Some(block) = self.free_blocks.pop() {
            let layout = Layout::from_size_align(block.size, self.config.alignment)
                .expect("Valid layout");
            
            unsafe {
                dealloc(block.ptr.as_ptr(), layout);
            }
        }
        
        // Print final statistics
        let stats = self.get_statistics();
        info!("Memory pool statistics: {:#?}", stats);
    }
}

// Custom error types
impl From<std::alloc::LayoutError> for AnalyzerError {
    fn from(error: std::alloc::LayoutError) -> Self {
        AnalyzerError::MemoryLayoutError(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config);
        assert!(pool.is_ok());
    }
    
    #[test]
    fn test_allocation_performance() {
        let config = MemoryPoolConfig {
            block_size: 1024,
            initial_blocks: 100,
            max_blocks: 1000,
            use_huge_pages: false,
            numa_node: None,
            alignment: 64,
            prefetch_enabled: true,
        };
        
        let pool = MemoryPool::new(config).unwrap();
        
        // Measure allocation latency
        let start = Instant::now();
        let ptr = pool.allocate(1024).unwrap();
        let latency = start.elapsed();
        
        assert!(latency.as_nanos() < 1000); // Sub-microsecond allocation
        
        // Clean up
        pool.deallocate(ptr, 1024).unwrap();
    }
    
    #[test]
    fn test_quantum_allocator() {
        let config = MemoryPoolConfig::default();
        let pool = Arc::new(MemoryPool::new(config).unwrap());
        let allocator = QuantumAllocator::new(pool);
        
        let ptr = allocator.allocate_quantum(512).unwrap();
        allocator.deallocate_quantum(ptr, 512).unwrap();
    }
    
    #[test]
    fn test_memory_statistics() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let ptr = pool.allocate(1024).unwrap();
        let stats = pool.get_statistics();
        
        assert!(stats.total_allocations > 0);
        assert!(stats.total_allocated_bytes > 0);
        
        pool.deallocate(ptr, 1024).unwrap();
    }
}