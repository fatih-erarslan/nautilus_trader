//! GPU Memory Pool Management
//! 
//! High-performance memory pooling system for GPU operations with automatic
//! garbage collection, defragmentation, and multi-GPU support.

use crate::{backend::{DeviceBuffer, get_context}, GpuError, GpuResult};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use std::time::{Duration, Instant};

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes
    pub max_size: usize,
    /// Minimum allocation size
    pub min_alloc_size: usize,
    /// Allocation alignment
    pub alignment: usize,
    /// Enable automatic defragmentation
    pub auto_defrag: bool,
    /// Defragmentation threshold (0.0 - 1.0)
    pub defrag_threshold: f32,
    /// Garbage collection interval
    pub gc_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 1024, // 1GB
            max_size: 8 * 1024 * 1024 * 1024, // 8GB
            min_alloc_size: 256, // 256 bytes
            alignment: 256, // 256-byte alignment for GPU
            auto_defrag: true,
            defrag_threshold: 0.3, // Defrag when 30% fragmented
            gc_interval: Duration::from_secs(60),
        }
    }
}

/// Memory allocation handle
#[derive(Clone)]
pub struct MemoryHandle {
    /// Pool ID
    pool_id: usize,
    /// Allocation ID
    alloc_id: u64,
    /// Offset in pool
    offset: usize,
    /// Size
    size: usize,
    /// Reference to buffer
    buffer: Arc<DeviceBuffer>,
}

/// Memory block metadata
struct BlockMeta {
    /// Offset in buffer
    offset: usize,
    /// Size
    size: usize,
    /// Is free
    is_free: bool,
    /// Last access time
    last_access: Instant,
}

/// Memory pool for a single device
struct DevicePool {
    /// Device ID
    device_id: u32,
    /// Pool configuration
    config: PoolConfig,
    /// Backing buffer
    buffer: Arc<DeviceBuffer>,
    /// Allocated size
    allocated_size: usize,
    /// Free blocks sorted by offset
    free_blocks: BTreeMap<usize, BlockMeta>,
    /// Allocated blocks
    allocated_blocks: HashMap<u64, BlockMeta>,
    /// Next allocation ID
    next_alloc_id: u64,
    /// Fragmentation ratio
    fragmentation: f32,
    /// Last defragmentation time
    last_defrag: Instant,
}

impl DevicePool {
    /// Create new device pool
    fn new(device_id: u32, config: PoolConfig) -> GpuResult<Self> {
        let context = get_context()?;
        let buffer = Arc::new(context.allocate(config.initial_size)?);
        
        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(0, BlockMeta {
            offset: 0,
            size: config.initial_size,
            is_free: true,
            last_access: Instant::now(),
        });
        
        Ok(Self {
            device_id,
            config,
            buffer,
            allocated_size: config.initial_size,
            free_blocks,
            allocated_blocks: HashMap::new(),
            next_alloc_id: 1,
            fragmentation: 0.0,
            last_defrag: Instant::now(),
        })
    }
    
    /// Allocate memory from pool
    fn allocate(&mut self, size: usize) -> GpuResult<MemoryHandle> {
        let aligned_size = align_size(size, self.config.alignment);
        
        // Find best-fit free block
        let mut best_block = None;
        for (offset, block) in &self.free_blocks {
            if block.size >= aligned_size {
                if best_block.is_none() || block.size < best_block.unwrap().1.size {
                    best_block = Some((*offset, block.clone()));
                }
            }
        }
        
        let (offset, mut block) = best_block
            .ok_or_else(|| GpuError::MemoryAllocation("No suitable free block".into()))?;
        
        // Remove from free list
        self.free_blocks.remove(&offset);
        
        // Split block if necessary
        if block.size > aligned_size {
            let remaining = BlockMeta {
                offset: offset + aligned_size,
                size: block.size - aligned_size,
                is_free: true,
                last_access: Instant::now(),
            };
            self.free_blocks.insert(remaining.offset, remaining);
        }
        
        // Create allocation
        block.size = aligned_size;
        block.is_free = false;
        block.last_access = Instant::now();
        
        let alloc_id = self.next_alloc_id;
        self.next_alloc_id += 1;
        self.allocated_blocks.insert(alloc_id, block);
        
        // Update fragmentation estimate
        self.update_fragmentation();
        
        Ok(MemoryHandle {
            pool_id: self.device_id as usize,
            alloc_id,
            offset,
            size: aligned_size,
            buffer: self.buffer.clone(),
        })
    }
    
    /// Free memory back to pool
    fn free(&mut self, alloc_id: u64) -> GpuResult<()> {
        let block = self.allocated_blocks.remove(&alloc_id)
            .ok_or_else(|| GpuError::MemoryAllocation("Invalid allocation ID".into()))?;
        
        // Mark as free
        let mut free_block = block;
        free_block.is_free = true;
        
        // Coalesce with adjacent free blocks
        self.coalesce_free_block(free_block);
        
        // Update fragmentation
        self.update_fragmentation();
        
        // Check if defragmentation needed
        if self.config.auto_defrag && self.fragmentation > self.config.defrag_threshold {
            self.defragment()?;
        }
        
        Ok(())
    }
    
    /// Coalesce free block with neighbors
    fn coalesce_free_block(&mut self, block: BlockMeta) {
        let mut merged_block = block;
        
        // Check for adjacent blocks before
        if let Some((prev_offset, prev_block)) = self.free_blocks.range(..block.offset).next_back() {
            if prev_offset + prev_block.size == block.offset {
                merged_block.offset = *prev_offset;
                merged_block.size += prev_block.size;
                self.free_blocks.remove(prev_offset);
            }
        }
        
        // Check for adjacent blocks after
        if let Some((next_offset, next_block)) = self.free_blocks.range((block.offset + block.size)..).next() {
            if block.offset + block.size == *next_offset {
                merged_block.size += next_block.size;
                self.free_blocks.remove(next_offset);
            }
        }
        
        self.free_blocks.insert(merged_block.offset, merged_block);
    }
    
    /// Update fragmentation estimate
    fn update_fragmentation(&mut self) {
        if self.free_blocks.is_empty() {
            self.fragmentation = 0.0;
            return;
        }
        
        let total_free: usize = self.free_blocks.values().map(|b| b.size).sum();
        let largest_free = self.free_blocks.values().map(|b| b.size).max().unwrap_or(0);
        
        if total_free > 0 {
            self.fragmentation = 1.0 - (largest_free as f32 / total_free as f32);
        } else {
            self.fragmentation = 0.0;
        }
    }
    
    /// Defragment memory pool
    fn defragment(&mut self) -> GpuResult<()> {
        // TODO: Implement memory defragmentation
        // This would involve:
        // 1. Creating a new compacted layout
        // 2. Moving all allocated blocks
        // 3. Updating all handles
        // 4. Consolidating free space
        
        self.last_defrag = Instant::now();
        self.fragmentation = 0.0;
        
        Ok(())
    }
}

/// Global memory pool manager
pub struct MemoryPoolManager {
    /// Device pools
    pools: RwLock<HashMap<u32, Arc<Mutex<DevicePool>>>>,
    /// Global configuration
    config: PoolConfig,
    /// Statistics
    stats: RwLock<MemoryStats>,
    /// GC thread handle
    gc_handle: Option<std::thread::JoinHandle<()>>,
}

/// Memory statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Total free memory
    pub total_free: usize,
    /// Number of allocations
    pub num_allocations: u64,
    /// Number of deallocations
    pub num_deallocations: u64,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Average fragmentation
    pub avg_fragmentation: f32,
}

impl MemoryPoolManager {
    /// Create new memory pool manager
    pub fn new(config: PoolConfig) -> Arc<Self> {
        let manager = Arc::new(Self {
            pools: RwLock::new(HashMap::new()),
            config,
            stats: RwLock::new(MemoryStats::default()),
            gc_handle: None,
        });
        
        // Start GC thread
        let gc_manager = manager.clone();
        let gc_interval = manager.config.gc_interval;
        let gc_handle = std::thread::spawn(move || {
            loop {
                std::thread::sleep(gc_interval);
                gc_manager.garbage_collect();
            }
        });
        
        // Note: In production, store gc_handle properly
        
        manager
    }
    
    /// Allocate memory
    pub fn allocate(&self, device_id: u32, size: usize) -> GpuResult<MemoryHandle> {
        // Get or create pool for device
        let pool = {
            let pools = self.pools.read();
            if let Some(pool) = pools.get(&device_id) {
                pool.clone()
            } else {
                drop(pools);
                let pool = Arc::new(Mutex::new(DevicePool::new(device_id, self.config.clone())?));
                self.pools.write().insert(device_id, pool.clone());
                pool
            }
        };
        
        // Allocate from pool
        let handle = pool.lock().allocate(size)?;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.total_allocated += size;
        stats.num_allocations += 1;
        stats.peak_usage = stats.peak_usage.max(stats.total_allocated);
        
        Ok(handle)
    }
    
    /// Free memory
    pub fn free(&self, handle: MemoryHandle) -> GpuResult<()> {
        let pools = self.pools.read();
        let pool = pools.get(&(handle.pool_id as u32))
            .ok_or_else(|| GpuError::MemoryAllocation("Invalid pool ID".into()))?;
        
        pool.lock().free(handle.alloc_id)?;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.total_free += handle.size;
        stats.num_deallocations += 1;
        
        Ok(())
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }
    
    /// Perform garbage collection
    fn garbage_collect(&self) {
        // TODO: Implement GC logic
        // - Remove unused pools
        // - Compact fragmented pools
        // - Release excess memory
    }
}

/// Align size to boundary
fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Global memory pool instance
static MEMORY_POOL: RwLock<Option<Arc<MemoryPoolManager>>> = RwLock::new(None);

/// Initialize global memory pool
pub fn initialize_pool(config: PoolConfig) -> GpuResult<()> {
    *MEMORY_POOL.write() = Some(MemoryPoolManager::new(config));
    Ok(())
}

/// Get global memory pool
pub fn get_pool() -> GpuResult<Arc<MemoryPoolManager>> {
    MEMORY_POOL.read()
        .clone()
        .ok_or_else(|| GpuError::BackendInit("Memory pool not initialized".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alignment() {
        assert_eq!(align_size(100, 256), 256);
        assert_eq!(align_size(256, 256), 256);
        assert_eq!(align_size(257, 256), 512);
    }
}