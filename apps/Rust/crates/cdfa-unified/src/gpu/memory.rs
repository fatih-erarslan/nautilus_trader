//! GPU memory management for efficient buffer allocation and pooling
//!
//! This module provides sophisticated memory management for GPU operations,
//! including memory pools, allocation strategies, and automatic cleanup.

use crate::error::{CdfaError, CdfaResult};
use super::{GpuBuffer, GpuContext};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use std::time::{Instant, Duration};

/// GPU memory manager for efficient buffer allocation
pub struct GpuMemoryManager {
    pools: RwLock<HashMap<usize, MemoryPool>>,
    allocation_stats: Mutex<AllocationStats>,
    config: MemoryConfig,
    cleanup_thread: Option<std::thread::JoinHandle<()>>,
}

/// Memory pool for specific buffer sizes
struct MemoryPool {
    buffers: VecDeque<PooledBuffer>,
    size_category: usize,
    max_buffers: usize,
    allocation_count: u64,
    reuse_count: u64,
}

/// Pooled buffer with metadata
struct PooledBuffer {
    buffer: Box<dyn GpuBuffer>,
    created_at: Instant,
    last_used: Instant,
    use_count: u32,
}

/// Memory allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub peak_memory_usage: u64,
    pub current_memory_usage: u64,
    pub fragmentation_ratio: f32,
}

/// Memory configuration options
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub pool_enabled: bool,
    pub max_pool_size_mb: u64,
    pub buffer_lifetime_seconds: u64,
    pub cleanup_interval_seconds: u64,
    pub size_categories: Vec<usize>,
    pub alignment: usize,
    pub prefetch_enabled: bool,
}

/// Memory allocation request
#[derive(Debug)]
pub struct AllocationRequest {
    pub size: usize,
    pub usage_hint: BufferUsageHint,
    pub lifetime_hint: BufferLifetime,
    pub priority: AllocationPriority,
}

/// Buffer usage hints for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsageHint {
    ShortTerm,   // Used once or few times
    LongTerm,    // Reused many times
    Stream,      // Streaming data
    Persistent,  // Persistent across operations
}

/// Buffer lifetime hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLifetime {
    Immediate,   // Use and discard immediately
    Operation,   // Lives for one operation
    Session,     // Lives for entire session
    Permanent,   // Never automatically freed
}

/// Allocation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
}

impl GpuMemoryManager {
    /// Create new memory manager
    pub fn new(max_memory_bytes: u64) -> Self {
        let config = MemoryConfig {
            pool_enabled: true,
            max_pool_size_mb: max_memory_bytes / (1024 * 1024),
            buffer_lifetime_seconds: 300, // 5 minutes
            cleanup_interval_seconds: 60, // 1 minute
            size_categories: vec![
                1024,      // 1KB
                4096,      // 4KB
                16384,     // 16KB
                65536,     // 64KB
                262144,    // 256KB
                1048576,   // 1MB
                4194304,   // 4MB
                16777216,  // 16MB
                67108864,  // 64MB
            ],
            alignment: 256, // Common GPU memory alignment
            prefetch_enabled: true,
        };
        
        let mut manager = Self {
            pools: RwLock::new(HashMap::new()),
            allocation_stats: Mutex::new(AllocationStats::default()),
            config,
            cleanup_thread: None,
        };
        
        // Initialize memory pools
        manager.initialize_pools();
        
        manager
    }
    
    /// Initialize memory pools for different size categories
    fn initialize_pools(&mut self) {
        let mut pools = self.pools.write().unwrap();
        
        for &size in &self.config.size_categories {
            pools.insert(size, MemoryPool::new(size));
        }
    }
    
    /// Allocate buffer with optimal strategy
    pub fn allocate(
        &self,
        context: Arc<dyn GpuContext>,
        request: AllocationRequest,
    ) -> CdfaResult<Box<dyn GpuBuffer>> {
        let aligned_size = self.align_size(request.size);
        let size_category = self.find_size_category(aligned_size);
        
        // Try to get from pool first
        if self.config.pool_enabled {
            if let Some(buffer) = self.try_allocate_from_pool(size_category, &request) {
                self.update_stats_pool_hit();
                return Ok(buffer);
            }
        }
        
        // Allocate new buffer
        self.update_stats_pool_miss();
        let buffer = context.allocate_buffer(aligned_size)?;
        self.update_stats_allocation(aligned_size);
        
        Ok(buffer)
    }
    
    /// Return buffer to pool for reuse
    pub fn deallocate(&self, buffer: Box<dyn GpuBuffer>, usage_hint: BufferUsageHint) -> CdfaResult<()> {
        let size = buffer.size();
        let size_category = self.find_size_category(size);
        
        if self.config.pool_enabled && self.should_pool_buffer(&usage_hint) {
            self.return_to_pool(buffer, size_category)?;
        }
        
        self.update_stats_deallocation(size);
        Ok(())
    }
    
    /// Try to allocate from memory pool
    fn try_allocate_from_pool(
        &self,
        size_category: usize,
        request: &AllocationRequest,
    ) -> Option<Box<dyn GpuBuffer>> {
        let mut pools = self.pools.write().unwrap();
        
        if let Some(pool) = pools.get_mut(&size_category) {
            if let Some(mut pooled_buffer) = pool.buffers.pop_front() {
                // Update buffer metadata
                pooled_buffer.last_used = Instant::now();
                pooled_buffer.use_count += 1;
                
                // Update pool statistics
                pool.reuse_count += 1;
                
                return Some(pooled_buffer.buffer);
            }
        }
        
        None
    }
    
    /// Return buffer to appropriate pool
    fn return_to_pool(&self, buffer: Box<dyn GpuBuffer>, size_category: usize) -> CdfaResult<()> {
        let mut pools = self.pools.write().unwrap();
        
        if let Some(pool) = pools.get_mut(&size_category) {
            // Check if pool has space
            if pool.buffers.len() < pool.max_buffers {
                let pooled_buffer = PooledBuffer {
                    buffer,
                    created_at: Instant::now(),
                    last_used: Instant::now(),
                    use_count: 1,
                };
                
                pool.buffers.push_back(pooled_buffer);
            }
        }
        
        Ok(())
    }
    
    /// Find appropriate size category for allocation
    fn find_size_category(&self, size: usize) -> usize {
        for &category_size in &self.config.size_categories {
            if size <= category_size {
                return category_size;
            }
        }
        
        // If larger than all categories, use the requested size
        self.align_size(size)
    }
    
    /// Align size to GPU memory requirements
    fn align_size(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }
    
    /// Check if buffer should be pooled based on usage hint
    fn should_pool_buffer(&self, usage_hint: &BufferUsageHint) -> bool {
        match usage_hint {
            BufferUsageHint::ShortTerm => false,
            BufferUsageHint::LongTerm => true,
            BufferUsageHint::Stream => false,
            BufferUsageHint::Persistent => true,
        }
    }
    
    /// Clean up expired buffers from pools
    pub fn cleanup_expired_buffers(&self) -> CdfaResult<()> {
        let mut pools = self.pools.write().unwrap();
        let lifetime_threshold = Duration::from_secs(self.config.buffer_lifetime_seconds);
        let now = Instant::now();
        
        for pool in pools.values_mut() {
            pool.buffers.retain(|buffer| {
                now.duration_since(buffer.last_used) < lifetime_threshold
            });
        }
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> AllocationStats {
        self.allocation_stats.lock().unwrap().clone()
    }
    
    /// Update allocation statistics
    fn update_stats_allocation(&self, size: usize) {
        let mut stats = self.allocation_stats.lock().unwrap();
        stats.total_allocations += 1;
        stats.current_memory_usage += size as u64;
        stats.peak_memory_usage = stats.peak_memory_usage.max(stats.current_memory_usage);
    }
    
    /// Update deallocation statistics
    fn update_stats_deallocation(&self, size: usize) {
        let mut stats = self.allocation_stats.lock().unwrap();
        stats.total_deallocations += 1;
        stats.current_memory_usage = stats.current_memory_usage.saturating_sub(size as u64);
    }
    
    /// Update pool hit statistics
    fn update_stats_pool_hit(&self) {
        self.allocation_stats.lock().unwrap().pool_hits += 1;
    }
    
    /// Update pool miss statistics
    fn update_stats_pool_miss(&self) {
        self.allocation_stats.lock().unwrap().pool_misses += 1;
    }
    
    /// Calculate memory fragmentation ratio
    pub fn calculate_fragmentation(&self) -> f32 {
        let pools = self.pools.read().unwrap();
        let mut total_pooled = 0u64;
        let mut total_wasted = 0u64;
        
        for pool in pools.values() {
            for buffer in &pool.buffers {
                total_pooled += buffer.buffer.size() as u64;
            }
            
            // Calculate waste due to size category rounding
            total_wasted += pool.buffers.len() as u64 * (pool.size_category as u64);
        }
        
        if total_pooled == 0 {
            0.0
        } else {
            total_wasted as f32 / total_pooled as f32
        }
    }
    
    /// Prefetch buffers for anticipated operations
    pub fn prefetch_buffers(
        &self,
        context: Arc<dyn GpuContext>,
        predicted_sizes: &[usize],
    ) -> CdfaResult<()> {
        if !self.config.prefetch_enabled {
            return Ok(());
        }
        
        for &size in predicted_sizes {
            let size_category = self.find_size_category(size);
            
            // Check if pool needs more buffers
            let pools = self.pools.read().unwrap();
            if let Some(pool) = pools.get(&size_category) {
                if pool.buffers.len() < pool.max_buffers / 2 {
                    // Pre-allocate some buffers
                    drop(pools);
                    
                    let buffer = context.allocate_buffer(size_category)?;
                    self.return_to_pool(buffer, size_category)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get detailed pool information
    pub fn get_pool_info(&self) -> HashMap<usize, PoolInfo> {
        let pools = self.pools.read().unwrap();
        let mut info = HashMap::new();
        
        for (&size, pool) in pools.iter() {
            info.insert(size, PoolInfo {
                size_category: size,
                buffer_count: pool.buffers.len(),
                max_buffers: pool.max_buffers,
                allocation_count: pool.allocation_count,
                reuse_count: pool.reuse_count,
                hit_ratio: if pool.allocation_count > 0 {
                    pool.reuse_count as f32 / pool.allocation_count as f32
                } else {
                    0.0
                },
            });
        }
        
        info
    }
}

/// Pool information for monitoring
#[derive(Debug, Clone)]
pub struct PoolInfo {
    pub size_category: usize,
    pub buffer_count: usize,
    pub max_buffers: usize,
    pub allocation_count: u64,
    pub reuse_count: u64,
    pub hit_ratio: f32,
}

impl MemoryPool {
    fn new(size_category: usize) -> Self {
        let max_buffers = match size_category {
            s if s <= 4096 => 100,      // Small buffers: many
            s if s <= 65536 => 50,      // Medium buffers: moderate
            s if s <= 1048576 => 20,    // Large buffers: few
            _ => 5,                     // Very large buffers: very few
        };
        
        Self {
            buffers: VecDeque::new(),
            size_category,
            max_buffers,
            allocation_count: 0,
            reuse_count: 0,
        }
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            pool_hits: 0,
            pool_misses: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_enabled: true,
            max_pool_size_mb: 1024, // 1GB
            buffer_lifetime_seconds: 300,
            cleanup_interval_seconds: 60,
            size_categories: vec![1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
            alignment: 256,
            prefetch_enabled: true,
        }
    }
}

/// Smart pointer for GPU buffers with automatic pooling
pub struct GpuBufferPtr {
    buffer: Option<Box<dyn GpuBuffer>>,
    manager: Arc<GpuMemoryManager>,
    usage_hint: BufferUsageHint,
}

impl GpuBufferPtr {
    pub fn new(
        buffer: Box<dyn GpuBuffer>,
        manager: Arc<GpuMemoryManager>,
        usage_hint: BufferUsageHint,
    ) -> Self {
        Self {
            buffer: Some(buffer),
            manager,
            usage_hint,
        }
    }
    
    pub fn as_ref(&self) -> Option<&dyn GpuBuffer> {
        self.buffer.as_ref().map(|b| b.as_ref())
    }
    
    pub fn as_mut(&mut self) -> Option<&mut dyn GpuBuffer> {
        self.buffer.as_mut().map(|b| b.as_mut())
    }
}

impl Drop for GpuBufferPtr {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.manager.deallocate(buffer, self.usage_hint);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager_creation() {
        let manager = GpuMemoryManager::new(1024 * 1024 * 1024); // 1GB
        let stats = manager.get_stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.current_memory_usage, 0);
    }
    
    #[test]
    fn test_size_alignment() {
        let manager = GpuMemoryManager::new(1024 * 1024 * 1024);
        assert_eq!(manager.align_size(100), 256);
        assert_eq!(manager.align_size(256), 256);
        assert_eq!(manager.align_size(300), 512);
    }
    
    #[test]
    fn test_size_category_finding() {
        let manager = GpuMemoryManager::new(1024 * 1024 * 1024);
        assert_eq!(manager.find_size_category(500), 1024);
        assert_eq!(manager.find_size_category(2000), 4096);
        assert_eq!(manager.find_size_category(100000), 262144);
    }
    
    #[test]
    fn test_fragmentation_calculation() {
        let manager = GpuMemoryManager::new(1024 * 1024 * 1024);
        let fragmentation = manager.calculate_fragmentation();
        assert!(fragmentation >= 0.0);
    }
    
    #[test]
    fn test_pool_info() {
        let manager = GpuMemoryManager::new(1024 * 1024 * 1024);
        let pool_info = manager.get_pool_info();
        assert!(!pool_info.is_empty());
        
        // Check that default size categories are present
        assert!(pool_info.contains_key(&1024));
        assert!(pool_info.contains_key(&4096));
    }
}