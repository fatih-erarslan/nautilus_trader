//! Memory optimization for HFT systems
//! 
//! This module implements advanced memory management techniques including:
//! - Custom allocators optimized for low latency
//! - Memory pools with different allocation strategies  
//! - Cache-friendly data layout optimizations
//! - Memory prefetching and alignment optimizations

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::ptr::{self, NonNull};
use std::mem::{self, MaybeUninit};
use std::time::Instant;
use parking_lot::{RwLock, Mutex};

use crate::error::Result;
use crate::performance::{HFTConfig, MemoryOptConfig, AllocatorType};

/// Memory optimizer for HFT systems
#[derive(Debug)]
pub struct MemoryOptimizer {
    /// Configuration
    config: MemoryOptConfig,
    
    /// Custom allocator
    allocator: Arc<dyn HFTAllocator + Send + Sync>,
    
    /// Memory pools
    pools: Arc<RwLock<Vec<Arc<MemoryPool>>>>,
    
    /// Memory layout optimizer
    layout_optimizer: Arc<LayoutOptimizer>,
    
    /// Memory prefetcher
    prefetcher: Arc<MemoryPrefetcher>,
    
    /// Memory statistics
    stats: Arc<RwLock<MemoryStats>>,
}

/// HFT allocator trait
pub trait HFTAllocator {
    /// Allocate memory with alignment
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    
    /// Deallocate memory
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);
    
    /// Reallocate memory
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8;
    
    /// Get allocation statistics
    fn stats(&self) -> AllocationStats;
    
    /// Check if allocation is from this allocator
    fn owns(&self, ptr: *const u8) -> bool;
}

/// Lock-free bump allocator for ultra-low latency
#[derive(Debug)]
pub struct LockFreeBumpAllocator {
    /// Memory arena
    arena: AtomicPtr<u8>,
    
    /// Arena size
    arena_size: usize,
    
    /// Current position in arena
    position: AtomicUsize,
    
    /// Arena end position
    arena_end: usize,
    
    /// Statistics
    stats: AtomicStats,
    
    /// Alignment
    alignment: usize,
}

/// Memory pool with different allocation strategies
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool ID
    id: usize,
    
    /// Block size
    block_size: usize,
    
    /// Number of blocks
    block_count: usize,
    
    /// Free list head
    free_head: AtomicPtr<PoolBlock>,
    
    /// Allocation strategy
    strategy: PoolStrategy,
    
    /// Pool statistics
    stats: Arc<RwLock<PoolStats>>,
    
    /// Pool memory region
    memory_region: NonNull<u8>,
    
    /// Region size
    region_size: usize,
}

/// Memory pool block
#[derive(Debug)]
struct PoolBlock {
    /// Next free block
    next: AtomicPtr<PoolBlock>,
    
    /// Block data starts here
    data: [u8; 0],
}

/// Pool allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum PoolStrategy {
    /// LIFO (Last In, First Out) - best cache locality
    LIFO,
    
    /// FIFO (First In, First Out) - predictable ordering
    FIFO,
    
    /// Best fit - minimize fragmentation
    BestFit,
    
    /// Thread-local pools
    ThreadLocal,
    
    /// NUMA-aware allocation
    NumaAware,
}

/// Layout optimizer for cache-friendly data structures
#[derive(Debug)]
pub struct LayoutOptimizer {
    /// Cache line size
    cache_line_size: usize,
    
    /// Page size
    page_size: usize,
    
    /// Alignment requirements
    alignment_reqs: Arc<RwLock<Vec<AlignmentRequirement>>>,
    
    /// Layout optimizations
    optimizations: Arc<RwLock<Vec<LayoutOptimization>>>,
}

/// Memory prefetcher for predictive loading
#[derive(Debug)]
pub struct MemoryPrefetcher {
    /// Prefetch patterns
    patterns: Arc<RwLock<Vec<PrefetchPattern>>>,
    
    /// Prefetch queue
    prefetch_queue: Arc<RwLock<Vec<PrefetchRequest>>>,
    
    /// Prefetch statistics
    stats: Arc<RwLock<PrefetchStats>>,
    
    /// Worker thread handle
    worker: Arc<RwLock<Option<std::thread::JoinHandle<()>>>>,
}

/// Alignment requirement
#[derive(Debug, Clone)]
pub struct AlignmentRequirement {
    /// Type name
    pub type_name: String,
    
    /// Required alignment (bytes)
    pub alignment: usize,
    
    /// Reason for alignment
    pub reason: AlignmentReason,
    
    /// Priority (higher = more important)
    pub priority: u32,
}

/// Reasons for alignment requirements
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentReason {
    /// Cache line alignment
    CacheLineAlignment,
    
    /// SIMD instruction requirements
    SIMDAlignment,
    
    /// Page alignment for large objects
    PageAlignment,
    
    /// False sharing avoidance
    FalseSharingAvoidance,
    
    /// Hardware prefetcher optimization
    PrefetcherOptimization,
}

/// Layout optimization
#[derive(Debug, Clone)]
pub struct LayoutOptimization {
    /// Structure name
    pub struct_name: String,
    
    /// Optimization type
    pub optimization_type: OptimizationType,
    
    /// Original layout
    pub original_layout: StructLayout,
    
    /// Optimized layout
    pub optimized_layout: StructLayout,
    
    /// Expected performance improvement
    pub improvement_estimate: f64,
    
    /// Implementation status
    pub implemented: bool,
}

/// Structure layout information
#[derive(Debug, Clone)]
pub struct StructLayout {
    /// Total size
    pub size: usize,
    
    /// Alignment
    pub alignment: usize,
    
    /// Fields
    pub fields: Vec<FieldLayout>,
    
    /// Padding bytes
    pub padding: usize,
}

/// Field layout information
#[derive(Debug, Clone)]
pub struct FieldLayout {
    /// Field name
    pub name: String,
    
    /// Offset from struct start
    pub offset: usize,
    
    /// Field size
    pub size: usize,
    
    /// Field alignment
    pub alignment: usize,
    
    /// Access frequency (relative)
    pub access_frequency: f64,
}

/// Layout optimization types
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Pack hot fields together
    HotFieldPacking,
    
    /// Separate hot and cold data
    HotColdSeparation,
    
    /// Structure-of-Arrays transformation
    SoATransformation,
    
    /// Add padding to prevent false sharing
    FalseSharingPrevention,
    
    /// Align to cache boundaries
    CacheBoundaryAlignment,
    
    /// Bit packing for small fields
    BitPacking,
}

/// Memory prefetch pattern
#[derive(Debug, Clone)]
pub struct PrefetchPattern {
    /// Pattern ID
    pub id: usize,
    
    /// Base address
    pub base_address: usize,
    
    /// Access stride
    pub stride: isize,
    
    /// Number of accesses
    pub access_count: usize,
    
    /// Prefetch distance
    pub prefetch_distance: usize,
    
    /// Pattern confidence (0.0 - 1.0)
    pub confidence: f64,
    
    /// Last used timestamp
    pub last_used: Instant,
}

/// Prefetch request
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Address to prefetch
    pub address: *const u8,
    
    /// Prefetch hint level
    pub hint_level: PrefetchHint,
    
    /// Priority
    pub priority: u32,
    
    /// Request timestamp
    pub timestamp: Instant,
}

/// Prefetch hint levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefetchHint {
    /// Load data to all cache levels
    LoadAll,
    
    /// Load to L2/L3 cache only
    LoadNonTemporal,
    
    /// Prepare for write access
    PrepareForWrite,
    
    /// Exclusive access expected
    Exclusive,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated bytes
    pub total_allocated: usize,
    
    /// Total deallocated bytes
    pub total_deallocated: usize,
    
    /// Current memory usage
    pub current_usage: usize,
    
    /// Peak memory usage
    pub peak_usage: usize,
    
    /// Number of allocations
    pub allocation_count: u64,
    
    /// Number of deallocations
    pub deallocation_count: u64,
    
    /// Pool statistics
    pub pool_stats: Vec<PoolStats>,
    
    /// Cache statistics
    pub cache_stats: CacheStats,
    
    /// Last updated
    pub last_updated: Instant,
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Pool ID
    pub pool_id: usize,
    
    /// Block size
    pub block_size: usize,
    
    /// Total blocks
    pub total_blocks: usize,
    
    /// Free blocks
    pub free_blocks: usize,
    
    /// Allocated blocks
    pub allocated_blocks: usize,
    
    /// Hit rate (successful allocations from pool)
    pub hit_rate: f64,
    
    /// Average allocation time (nanoseconds)
    pub avg_alloc_time: u64,
    
    /// Memory efficiency (useful data / total memory)
    pub memory_efficiency: f64,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// L1 cache hit rate
    pub l1_hit_rate: f64,
    
    /// L2 cache hit rate
    pub l2_hit_rate: f64,
    
    /// L3 cache hit rate
    pub l3_hit_rate: f64,
    
    /// Cache miss penalty (cycles)
    pub miss_penalty: u64,
    
    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
    
    /// Prefetch effectiveness
    pub prefetch_effectiveness: f64,
}

/// Prefetch statistics
#[derive(Debug, Clone)]
pub struct PrefetchStats {
    /// Total prefetch requests
    pub total_requests: u64,
    
    /// Successful prefetches (data was used)
    pub successful_prefetches: u64,
    
    /// Wasted prefetches (data not used)
    pub wasted_prefetches: u64,
    
    /// Prefetch accuracy
    pub accuracy: f64,
    
    /// Average prefetch distance
    pub avg_prefetch_distance: f64,
    
    /// Detected patterns
    pub pattern_count: usize,
}

/// Allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total allocations
    pub total_allocations: u64,
    
    /// Total deallocations
    pub total_deallocations: u64,
    
    /// Bytes allocated
    pub bytes_allocated: u64,
    
    /// Bytes deallocated
    pub bytes_deallocated: u64,
    
    /// Average allocation size
    pub avg_allocation_size: f64,
    
    /// Peak concurrent allocations
    pub peak_concurrent_allocations: usize,
    
    /// Allocation failures
    pub allocation_failures: u64,
    
    /// Average allocation time (nanoseconds)
    pub avg_allocation_time: u64,
}

/// Atomic statistics for lock-free updates
#[derive(Debug)]
pub struct AtomicStats {
    pub total_allocations: AtomicUsize,
    pub total_deallocations: AtomicUsize,
    pub bytes_allocated: AtomicUsize,
    pub bytes_deallocated: AtomicUsize,
    pub allocation_failures: AtomicUsize,
    pub peak_usage: AtomicUsize,
}

impl MemoryOptimizer {
    /// Create new memory optimizer
    pub async fn new(config: &HFTConfig) -> Result<Self> {
        let memory_config = &config.memory_config;
        
        // Create custom allocator based on configuration
        let allocator: Arc<dyn HFTAllocator + Send + Sync> = match memory_config.allocator_type {
            AllocatorType::LockFree => {
                Arc::new(LockFreeBumpAllocator::new(64 * 1024 * 1024)?) // 64MB arena
            }
            AllocatorType::Pool => {
                Arc::new(PoolAllocator::new(memory_config.pool_sizes.clone())?)
            }
            _ => {
                Arc::new(SystemAllocatorWrapper::new())
            }
        };
        
        // Initialize memory pools
        let mut pools = Vec::new();
        for (i, &size) in memory_config.pool_sizes.iter().enumerate() {
            let pool = Arc::new(MemoryPool::new(
                i,
                size,
                1024, // 1024 blocks per pool
                PoolStrategy::LIFO,
            )?);
            pools.push(pool);
        }
        
        let layout_optimizer = Arc::new(LayoutOptimizer::new());
        let prefetcher = Arc::new(MemoryPrefetcher::new());
        
        let stats = Arc::new(RwLock::new(MemoryStats {
            total_allocated: 0,
            total_deallocated: 0,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            pool_stats: Vec::new(),
            cache_stats: CacheStats {
                l1_hit_rate: 0.0,
                l2_hit_rate: 0.0,
                l3_hit_rate: 0.0,
                miss_penalty: 0,
                bandwidth_utilization: 0.0,
                prefetch_effectiveness: 0.0,
            },
            last_updated: Instant::now(),
        }));
        
        Ok(Self {
            config: memory_config.clone(),
            allocator,
            pools: Arc::new(RwLock::new(pools)),
            layout_optimizer,
            prefetcher,
            stats,
        })
    }
    
    /// Optimize memory allocation
    pub async fn optimize_memory_allocation(&self, _config: &MemoryOptConfig) -> Result<bool> {
        // Enable huge pages if configured
        if self.config.huge_pages {
            self.enable_huge_pages().await?;
        }
        
        // Setup memory pools
        self.setup_memory_pools().await?;
        
        // Enable memory prefetching
        if self.config.prefetching {
            self.prefetcher.start_prefetching().await?;
        }
        
        // Apply cache-friendly layouts
        if self.config.cache_line_alignment {
            self.layout_optimizer.apply_cache_optimizations().await?;
        }
        
        Ok(true)
    }
    
    /// Enable lock-free data structures
    pub async fn enable_lock_free_structures(&self) -> Result<()> {
        // This would typically modify the global allocator or provide
        // lock-free alternatives to standard data structures
        // For now, we'll just update configuration
        
        info!("Enabled lock-free data structures optimization");
        Ok(())
    }
    
    /// Enable memory prefetching
    pub async fn enable_memory_prefetching(&self) -> Result<()> {
        self.prefetcher.enable_prefetching().await?;
        info!("Enabled memory prefetching optimization");
        Ok(())
    }
    
    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        let stats = self.stats.read().clone();
        stats
    }
    
    /// Enable huge pages
    async fn enable_huge_pages(&self) -> Result<()> {
        // Platform-specific huge page configuration would go here
        #[cfg(target_os = "linux")]
        {
            // This would typically use madvise with MADV_HUGEPAGE
            info!("Would enable huge pages on Linux");
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            warn!("Huge pages not supported on this platform");
        }
        
        Ok(())
    }
    
    /// Setup memory pools
    async fn setup_memory_pools(&self) -> Result<()> {
        let pools = self.pools.read();
        info!("Setting up {} memory pools", pools.len());
        
        for pool in pools.iter() {
            pool.initialize().await?;
        }
        
        Ok(())
    }
}

impl LockFreeBumpAllocator {
    /// Create new bump allocator
    pub fn new(size: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, 64)?; // 64-byte aligned
        let arena = unsafe { std::alloc::alloc(layout) };
        
        if arena.is_null() {
            return Err("Failed to allocate arena".into());
        }
        
        Ok(Self {
            arena: AtomicPtr::new(arena),
            arena_size: size,
            position: AtomicUsize::new(0),
            arena_end: size,
            stats: AtomicStats {
                total_allocations: AtomicUsize::new(0),
                total_deallocations: AtomicUsize::new(0),
                bytes_allocated: AtomicUsize::new(0),
                bytes_deallocated: AtomicUsize::new(0),
                allocation_failures: AtomicUsize::new(0),
                peak_usage: AtomicUsize::new(0),
            },
            alignment: 64,
        })
    }
    
    /// Reset allocator (development/testing only)
    pub fn reset(&self) {
        self.position.store(0, Ordering::Relaxed);
    }
}

impl HFTAllocator for LockFreeBumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align().max(self.alignment);
        
        // Align size to alignment boundary
        let aligned_size = (size + align - 1) & !(align - 1);
        
        loop {
            let current_pos = self.position.load(Ordering::Relaxed);
            let aligned_pos = (current_pos + align - 1) & !(align - 1);
            let new_pos = aligned_pos + aligned_size;
            
            if new_pos > self.arena_end {
                // Arena exhausted
                self.stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
                return ptr::null_mut();
            }
            
            // Try to claim the space
            if self.position.compare_exchange_weak(
                current_pos,
                new_pos,
                Ordering::Relaxed,
                Ordering::Relaxed
            ).is_ok() {
                // Successfully allocated
                let arena_ptr = self.arena.load(Ordering::Relaxed);
                let result_ptr = arena_ptr.add(aligned_pos);
                
                // Update statistics
                self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_allocated.fetch_add(aligned_size, Ordering::Relaxed);
                
                let current_usage = new_pos;
                let mut peak = self.stats.peak_usage.load(Ordering::Relaxed);
                while current_usage > peak {
                    match self.stats.peak_usage.compare_exchange_weak(
                        peak,
                        current_usage,
                        Ordering::Relaxed,
                        Ordering::Relaxed
                    ) {
                        Ok(_) => break,
                        Err(x) => peak = x,
                    }
                }
                
                return result_ptr;
            }
        }
    }
    
    unsafe fn dealloc(&self, _ptr: *mut u8, layout: Layout) {
        // Bump allocator doesn't support individual deallocations
        // Just update statistics
        self.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_deallocated.fetch_add(layout.size(), Ordering::Relaxed);
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_ptr = self.alloc(new_layout);
        
        if !new_ptr.is_null() {
            ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        
        new_ptr
    }
    
    fn stats(&self) -> AllocationStats {
        let total_allocs = self.stats.total_allocations.load(Ordering::Relaxed) as u64;
        let total_deallocs = self.stats.total_deallocations.load(Ordering::Relaxed) as u64;
        let bytes_alloc = self.stats.bytes_allocated.load(Ordering::Relaxed) as u64;
        let bytes_dealloc = self.stats.bytes_deallocated.load(Ordering::Relaxed) as u64;
        
        AllocationStats {
            total_allocations: total_allocs,
            total_deallocations: total_deallocs,
            bytes_allocated: bytes_alloc,
            bytes_deallocated: bytes_dealloc,
            avg_allocation_size: if total_allocs > 0 { bytes_alloc as f64 / total_allocs as f64 } else { 0.0 },
            peak_concurrent_allocations: 0, // Not tracked in bump allocator
            allocation_failures: self.stats.allocation_failures.load(Ordering::Relaxed) as u64,
            avg_allocation_time: 0, // Would need timing measurements
        }
    }
    
    fn owns(&self, ptr: *const u8) -> bool {
        let arena_ptr = self.arena.load(Ordering::Relaxed);
        let arena_end = unsafe { arena_ptr.add(self.arena_size) };
        
        ptr >= arena_ptr as *const u8 && ptr < arena_end as *const u8
    }
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(
        id: usize,
        block_size: usize,
        block_count: usize,
        strategy: PoolStrategy,
    ) -> Result<Self> {
        // Align block size to cache line boundary
        let aligned_block_size = (block_size + 63) & !63;
        let total_size = aligned_block_size * block_count;
        
        // Allocate memory region
        let layout = Layout::from_size_align(total_size, 64)?;
        let memory = unsafe { std::alloc::alloc(layout) };
        
        if memory.is_null() {
            return Err("Failed to allocate memory pool".into());
        }
        
        let memory_region = NonNull::new(memory).unwrap();
        
        let pool = Self {
            id,
            block_size: aligned_block_size,
            block_count,
            free_head: AtomicPtr::new(ptr::null_mut()),
            strategy,
            stats: Arc::new(RwLock::new(PoolStats {
                pool_id: id,
                block_size: aligned_block_size,
                total_blocks: block_count,
                free_blocks: 0,
                allocated_blocks: 0,
                hit_rate: 0.0,
                avg_alloc_time: 0,
                memory_efficiency: 0.0,
            })),
            memory_region,
            region_size: total_size,
        };
        
        Ok(pool)
    }
    
    /// Initialize pool with free blocks
    pub async fn initialize(&self) -> Result<()> {
        let mut current_ptr = self.memory_region.as_ptr();
        let mut prev_block: *mut PoolBlock = ptr::null_mut();
        
        // Initialize free list
        for i in 0..self.block_count {
            let block = current_ptr as *mut PoolBlock;
            
            unsafe {
                (*block).next = AtomicPtr::new(prev_block);
            }
            
            prev_block = block;
            current_ptr = unsafe { current_ptr.add(self.block_size) };
        }
        
        // Set free head to last block created (first in chain)
        self.free_head.store(prev_block, Ordering::Relaxed);
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.free_blocks = self.block_count;
            stats.allocated_blocks = 0;
        }
        
        Ok(())
    }
    
    /// Allocate block from pool
    pub fn allocate(&self) -> Option<NonNull<u8>> {
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            if head.is_null() {
                // Pool exhausted
                return None;
            }
            
            let next = unsafe { (*head).next.load(Ordering::Relaxed) };
            
            if self.free_head.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed
            ).is_ok() {
                // Successfully allocated block
                let block_ptr = unsafe { (head as *mut u8).add(mem::size_of::<PoolBlock>()) };
                
                // Update statistics
                {
                    let mut stats = self.stats.write();
                    stats.free_blocks -= 1;
                    stats.allocated_blocks += 1;
                }
                
                return NonNull::new(block_ptr);
            }
        }
    }
    
    /// Deallocate block back to pool
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let block_ptr = unsafe { 
            (ptr.as_ptr() as *mut u8).sub(mem::size_of::<PoolBlock>()) as *mut PoolBlock 
        };
        
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            
            unsafe {
                (*block_ptr).next.store(head, Ordering::Relaxed);
            }
            
            if self.free_head.compare_exchange_weak(
                head,
                block_ptr,
                Ordering::Release,
                Ordering::Relaxed
            ).is_ok() {
                // Successfully deallocated
                let mut stats = self.stats.write();
                stats.free_blocks += 1;
                stats.allocated_blocks -= 1;
                break;
            }
        }
    }
}

impl LayoutOptimizer {
    /// Create new layout optimizer
    pub fn new() -> Self {
        Self {
            cache_line_size: 64, // Modern x86 CPUs
            page_size: 4096,     // Standard page size
            alignment_reqs: Arc::new(RwLock::new(Vec::new())),
            optimizations: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Apply cache optimizations
    pub async fn apply_cache_optimizations(&self) -> Result<()> {
        // Add common alignment requirements for HFT structures
        let requirements = vec![
            AlignmentRequirement {
                type_name: "OrderBook".to_string(),
                alignment: self.cache_line_size,
                reason: AlignmentReason::CacheLineAlignment,
                priority: 100,
            },
            AlignmentRequirement {
                type_name: "ConsensusMessage".to_string(),
                alignment: 32, // AVX2 alignment
                reason: AlignmentReason::SIMDAlignment,
                priority: 90,
            },
            AlignmentRequirement {
                type_name: "MemoryPool".to_string(),
                alignment: self.cache_line_size,
                reason: AlignmentReason::FalseSharingAvoidance,
                priority: 85,
            },
        ];
        
        let mut align_reqs = self.alignment_reqs.write();
        align_reqs.extend(requirements);
        
        info!("Applied cache optimization alignments");
        Ok(())
    }
}

impl MemoryPrefetcher {
    /// Create new memory prefetcher
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(Vec::new())),
            prefetch_queue: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(PrefetchStats {
                total_requests: 0,
                successful_prefetches: 0,
                wasted_prefetches: 0,
                accuracy: 0.0,
                avg_prefetch_distance: 0.0,
                pattern_count: 0,
            })),
            worker: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Start prefetching worker
    pub async fn start_prefetching(&self) -> Result<()> {
        let queue = self.prefetch_queue.clone();
        let stats = self.stats.clone();
        
        let handle = std::thread::spawn(move || {
            let mut interval = std::time::Duration::from_micros(100);
            
            loop {
                let requests = {
                    let mut q = queue.write();
                    let requests: Vec<_> = q.drain(..).collect();
                    requests
                };
                
                for request in requests {
                    unsafe {
                        Self::issue_prefetch(request.address, request.hint_level);
                    }
                    
                    // Update statistics
                    {
                        let mut s = stats.write();
                        s.total_requests += 1;
                    }
                }
                
                std::thread::sleep(interval);
            }
        });
        
        let mut worker = self.worker.write();
        *worker = Some(handle);
        
        Ok(())
    }
    
    /// Enable prefetching
    pub async fn enable_prefetching(&self) -> Result<()> {
        if self.worker.read().is_none() {
            self.start_prefetching().await?;
        }
        Ok(())
    }
    
    /// Issue hardware prefetch instruction
    unsafe fn issue_prefetch(address: *const u8, hint: PrefetchHint) {
        match hint {
            PrefetchHint::LoadAll => {
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(address as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
            PrefetchHint::LoadNonTemporal => {
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(address as *const i8, std::arch::x86_64::_MM_HINT_NTA);
            }
            PrefetchHint::PrepareForWrite => {
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(address as *const i8, std::arch::x86_64::_MM_HINT_T1);
            }
            PrefetchHint::Exclusive => {
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(address as *const i8, std::arch::x86_64::_MM_HINT_T2);
            }
        }
    }
}

/// System allocator wrapper for compatibility
#[derive(Debug)]
pub struct SystemAllocatorWrapper;

impl SystemAllocatorWrapper {
    pub fn new() -> Self {
        Self
    }
}

impl HFTAllocator for SystemAllocatorWrapper {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        System.realloc(ptr, layout, new_size)
    }
    
    fn stats(&self) -> AllocationStats {
        // System allocator doesn't provide detailed statistics
        AllocationStats {
            total_allocations: 0,
            total_deallocations: 0,
            bytes_allocated: 0,
            bytes_deallocated: 0,
            avg_allocation_size: 0.0,
            peak_concurrent_allocations: 0,
            allocation_failures: 0,
            avg_allocation_time: 0,
        }
    }
    
    fn owns(&self, _ptr: *const u8) -> bool {
        false // Can't determine ownership for system allocator
    }
}

/// Pool allocator that manages multiple memory pools
#[derive(Debug)]
pub struct PoolAllocator {
    pools: Vec<Arc<MemoryPool>>,
}

impl PoolAllocator {
    pub fn new(pool_sizes: Vec<usize>) -> Result<Self> {
        let mut pools = Vec::new();
        
        for (i, size) in pool_sizes.iter().enumerate() {
            let pool = Arc::new(MemoryPool::new(
                i,
                *size,
                1024,
                PoolStrategy::LIFO,
            )?);
            pools.push(pool);
        }
        
        Ok(Self { pools })
    }
    
    /// Find suitable pool for allocation
    fn find_pool(&self, size: usize) -> Option<&Arc<MemoryPool>> {
        self.pools.iter()
            .find(|pool| pool.block_size >= size)
    }
}

impl HFTAllocator for PoolAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if let Some(pool) = self.find_pool(layout.size()) {
            if let Some(ptr) = pool.allocate() {
                return ptr.as_ptr();
            }
        }
        
        // Fallback to system allocator
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(pool) = self.find_pool(layout.size()) {
            if let Some(non_null_ptr) = NonNull::new(ptr) {
                pool.deallocate(non_null_ptr);
                return;
            }
        }
        
        System.dealloc(ptr, layout)
    }
    
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let new_ptr = self.alloc(new_layout);
        
        if !new_ptr.is_null() {
            ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        
        new_ptr
    }
    
    fn stats(&self) -> AllocationStats {
        // Aggregate statistics from all pools
        let mut total_allocs = 0u64;
        let mut total_size = 0u64;
        
        for pool in &self.pools {
            let pool_stats = pool.stats.read();
            total_allocs += pool_stats.allocated_blocks as u64;
            total_size += (pool_stats.allocated_blocks * pool_stats.block_size) as u64;
        }
        
        AllocationStats {
            total_allocations: total_allocs,
            total_deallocations: 0, // Would need tracking
            bytes_allocated: total_size,
            bytes_deallocated: 0,
            avg_allocation_size: if total_allocs > 0 { total_size as f64 / total_allocs as f64 } else { 0.0 },
            peak_concurrent_allocations: total_allocs as usize,
            allocation_failures: 0, // Would need tracking
            avg_allocation_time: 0,
        }
    }
    
    fn owns(&self, ptr: *const u8) -> bool {
        // Check if pointer belongs to any of our pools
        for pool in &self.pools {
            let region_start = pool.memory_region.as_ptr() as *const u8;
            let region_end = unsafe { region_start.add(pool.region_size) };
            
            if ptr >= region_start && ptr < region_end {
                return true;
            }
        }
        false
    }
}

// Safety implementations
unsafe impl Send for LockFreeBumpAllocator {}
unsafe impl Sync for LockFreeBumpAllocator {}
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}
unsafe impl Send for PoolAllocator {}
unsafe impl Sync for PoolAllocator {}