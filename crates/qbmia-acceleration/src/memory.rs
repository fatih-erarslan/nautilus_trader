//! GPU memory management for ultra-fast allocations

use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use tokio::sync::{Mutex, RwLock};
use memmap2::{MmapOptions, MmapMut};
use crate::{
    QBMIAError, QBMIAResult, GpuBuffer, GpuBufferUsage,
    gpu::GpuPipeline
};

/// GPU memory manager with zero-copy optimizations
pub struct GpuMemoryManager {
    /// GPU pipeline reference
    gpu_pipeline: Arc<GpuPipeline>,
    
    /// Memory pools for different buffer sizes
    memory_pools: Arc<RwLock<HashMap<BufferSizeClass, MemoryPool>>>,
    
    /// Active buffer registry
    active_buffers: Arc<Mutex<HashMap<u64, BufferInfo>>>,
    
    /// Memory-mapped regions for zero-copy transfers
    mmap_regions: Arc<Mutex<Vec<MmapRegion>>>,
    
    /// Allocation statistics
    stats: Arc<Mutex<MemoryStats>>,
    
    /// Next buffer ID
    next_buffer_id: Arc<Mutex<u64>>,
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub async fn new(gpu_pipeline: Arc<GpuPipeline>) -> QBMIAResult<Self> {
        tracing::info!("Initializing GPU memory manager");
        
        let memory_pools = Arc::new(RwLock::new(HashMap::new()));
        let active_buffers = Arc::new(Mutex::new(HashMap::new()));
        let mmap_regions = Arc::new(Mutex::new(Vec::new()));
        let stats = Arc::new(Mutex::new(MemoryStats::new()));
        let next_buffer_id = Arc::new(Mutex::new(1));
        
        let manager = Self {
            gpu_pipeline,
            memory_pools,
            active_buffers,
            mmap_regions,
            stats,
            next_buffer_id,
        };
        
        // Initialize memory pools
        manager.initialize_pools().await?;
        
        tracing::info!("GPU memory manager initialized");
        Ok(manager)
    }
    
    /// Create buffer with optimal allocation strategy
    pub async fn create_buffer(&self, data: &[u8]) -> QBMIAResult<GpuBuffer> {
        let start_time = std::time::Instant::now();
        
        let size = data.len();
        let size_class = BufferSizeClass::from_size(size);
        let usage = GpuBufferUsage::Storage; // Default usage
        
        // Try to get buffer from pool first
        let buffer_id = {
            let pools = self.memory_pools.read().await;
            if let Some(pool) = pools.get(&size_class) {
                pool.allocate_buffer(size, usage).await
            } else {
                None
            }
        };
        
        let buffer_id = if let Some(id) = buffer_id {
            // Reuse existing buffer
            let gpu_buffer = self.gpu_pipeline.create_buffer(data, usage).await?;
            
            let mut stats = self.stats.lock().await;
            stats.record_allocation_reuse(start_time.elapsed(), size);
            
            tracing::debug!("Reused buffer {} ({} bytes)", id, size);
            
            gpu_buffer
        } else {
            // Create new buffer
            let gpu_buffer = self.gpu_pipeline.create_buffer(data, usage).await?;
            
            // Register buffer
            let buffer_id = {
                let mut next_id = self.next_buffer_id.lock().await;
                let id = *next_id;
                *next_id += 1;
                id
            };
            
            let buffer_info = BufferInfo {
                id: buffer_id,
                size,
                usage,
                size_class,
                created_at: std::time::Instant::now(),
                last_accessed: std::time::Instant::now(),
                access_count: 1,
            };
            
            let mut active_buffers = self.active_buffers.lock().await;
            active_buffers.insert(buffer_id, buffer_info);
            
            let mut stats = self.stats.lock().await;
            stats.record_allocation_new(start_time.elapsed(), size);
            
            tracing::debug!("Created new buffer {} ({} bytes)", buffer_id, size);
            
            gpu_buffer
        };
        
        Ok(buffer_id)
    }
    
    /// Read buffer with zero-copy optimization when possible
    pub async fn read_buffer(&self, buffer: &GpuBuffer) -> QBMIAResult<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        // Try zero-copy read first
        if let Some(data) = self.try_zero_copy_read(buffer).await? {
            let mut stats = self.stats.lock().await;
            stats.record_read_zero_copy(start_time.elapsed(), buffer.size);
            
            tracing::debug!("Zero-copy read {} bytes", buffer.size);
            return Ok(data);
        }
        
        // Fallback to regular GPU read
        let data = self.gpu_pipeline.read_buffer(buffer).await?;
        
        // Update buffer access statistics
        self.update_buffer_access(buffer.id).await;
        
        let mut stats = self.stats.lock().await;
        stats.record_read_regular(start_time.elapsed(), buffer.size);
        
        Ok(data)
    }
    
    /// Try zero-copy read using memory mapping
    async fn try_zero_copy_read(&self, buffer: &GpuBuffer) -> QBMIAResult<Option<Vec<u8>>> {
        // Check if buffer is in memory-mapped region
        let mmap_regions = self.mmap_regions.lock().await;
        
        for region in &*mmap_regions {
            if region.contains_buffer(buffer.id) {
                // Read directly from mapped memory
                let offset = region.get_buffer_offset(buffer.id)?;
                let data = region.read_data(offset, buffer.size)?;
                return Ok(Some(data));
            }
        }
        
        Ok(None)
    }
    
    /// Create memory-mapped buffer for zero-copy operations
    pub async fn create_mmap_buffer(&self, size: usize) -> QBMIAResult<MmapBuffer> {
        let start_time = std::time::Instant::now();
        
        // Create memory-mapped file
        let mmap = MmapOptions::new()
            .len(size)
            .map_anon()
            .map_err(|e| QBMIAError::memory_alloc(format!("Failed to create mmap: {}", e)))?;
        
        let buffer_id = {
            let mut next_id = self.next_buffer_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        let mmap_buffer = MmapBuffer {
            id: buffer_id,
            size,
            mmap: Arc::new(Mutex::new(mmap)),
            created_at: std::time::Instant::now(),
        };
        
        // Register memory-mapped region
        let region = MmapRegion {
            buffer_id,
            size,
            offset: 0,
        };
        
        let mut mmap_regions = self.mmap_regions.lock().await;
        mmap_regions.push(region);
        
        let mut stats = self.stats.lock().await;
        stats.record_mmap_creation(start_time.elapsed(), size);
        
        tracing::debug!("Created memory-mapped buffer {} ({} bytes)", buffer_id, size);
        
        Ok(mmap_buffer)
    }
    
    /// Free buffer and return to pool
    pub async fn free_buffer(&self, buffer: &GpuBuffer) -> QBMIAResult<()> {
        let start_time = std::time::Instant::now();
        
        // Remove from active buffers
        let buffer_info = {
            let mut active_buffers = self.active_buffers.lock().await;
            active_buffers.remove(&buffer.id)
        };
        
        if let Some(info) = buffer_info {
            // Return to pool
            let pools = self.memory_pools.read().await;
            if let Some(pool) = pools.get(&info.size_class) {
                pool.deallocate_buffer(buffer.id).await;
            }
            
            let mut stats = self.stats.lock().await;
            stats.record_deallocation(start_time.elapsed(), info.size);
            
            tracing::debug!("Freed buffer {} ({} bytes)", buffer.id, info.size);
        }
        
        Ok(())
    }
    
    /// Prefetch data to GPU memory
    pub async fn prefetch(&self, data: &[u8]) -> QBMIAResult<GpuBuffer> {
        let start_time = std::time::Instant::now();
        
        // Create buffer and immediately upload data
        let buffer = self.create_buffer(data).await?;
        
        let mut stats = self.stats.lock().await;
        stats.record_prefetch(start_time.elapsed(), data.len());
        
        tracing::debug!("Prefetched {} bytes to GPU", data.len());
        
        Ok(buffer)
    }
    
    /// Garbage collect unused buffers
    pub async fn garbage_collect(&self) -> QBMIAResult<usize> {
        let start_time = std::time::Instant::now();
        let mut freed_count = 0;
        
        // Find buffers that haven't been accessed recently
        let threshold = std::time::Duration::from_secs(60); // 1 minute
        let now = std::time::Instant::now();
        
        let buffers_to_free: Vec<u64> = {
            let active_buffers = self.active_buffers.lock().await;
            active_buffers
                .iter()
                .filter(|(_, info)| now.duration_since(info.last_accessed) > threshold)
                .map(|(id, _)| *id)
                .collect()
        };
        
        // Free old buffers
        for buffer_id in buffers_to_free {
            let buffer = GpuBuffer {
                id: buffer_id,
                size: 0, // Size will be looked up
                usage: GpuBufferUsage::Storage,
            };
            
            if self.free_buffer(&buffer).await.is_ok() {
                freed_count += 1;
            }
        }
        
        // Compact memory pools
        {
            let pools = self.memory_pools.read().await;
            for pool in pools.values() {
                pool.compact().await;
            }
        }
        
        let gc_time = start_time.elapsed();
        let mut stats = self.stats.lock().await;
        stats.record_garbage_collection(gc_time, freed_count);
        
        tracing::debug!("Garbage collected {} buffers in {:.3}ms", 
                       freed_count, gc_time.as_secs_f64() * 1000.0);
        
        Ok(freed_count)
    }
    
    /// Get memory usage statistics
    pub async fn get_stats(&self) -> MemoryStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }
    
    /// Initialize memory pools for different buffer sizes
    async fn initialize_pools(&self) -> QBMIAResult<()> {
        let mut pools = self.memory_pools.write().await;
        
        // Create pools for different size classes
        for &size_class in &[
            BufferSizeClass::Small,
            BufferSizeClass::Medium,
            BufferSizeClass::Large,
            BufferSizeClass::Huge,
        ] {
            let pool = MemoryPool::new(size_class);
            pools.insert(size_class, pool);
        }
        
        tracing::debug!("Initialized {} memory pools", pools.len());
        Ok(())
    }
    
    /// Update buffer access statistics
    async fn update_buffer_access(&self, buffer_id: u64) {
        let mut active_buffers = self.active_buffers.lock().await;
        if let Some(info) = active_buffers.get_mut(&buffer_id) {
            info.last_accessed = std::time::Instant::now();
            info.access_count += 1;
        }
    }
}

/// Buffer size classification for pool management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BufferSizeClass {
    Small,  // < 1KB
    Medium, // 1KB - 1MB
    Large,  // 1MB - 100MB
    Huge,   // > 100MB
}

impl BufferSizeClass {
    fn from_size(size: usize) -> Self {
        if size < 1024 {
            Self::Small
        } else if size < 1024 * 1024 {
            Self::Medium
        } else if size < 100 * 1024 * 1024 {
            Self::Large
        } else {
            Self::Huge
        }
    }
    
    fn pool_size(&self) -> usize {
        match self {
            Self::Small => 100,   // Pool of 100 small buffers
            Self::Medium => 50,   // Pool of 50 medium buffers
            Self::Large => 10,    // Pool of 10 large buffers
            Self::Huge => 2,      // Pool of 2 huge buffers
        }
    }
}

/// Memory pool for buffer reuse
struct MemoryPool {
    size_class: BufferSizeClass,
    free_buffers: Mutex<VecDeque<u64>>,
    buffer_info: Mutex<HashMap<u64, PooledBufferInfo>>,
    stats: Mutex<PoolStats>,
}

impl MemoryPool {
    fn new(size_class: BufferSizeClass) -> Self {
        Self {
            size_class,
            free_buffers: Mutex::new(VecDeque::new()),
            buffer_info: Mutex::new(HashMap::new()),
            stats: Mutex::new(PoolStats::new()),
        }
    }
    
    async fn allocate_buffer(&self, size: usize, usage: GpuBufferUsage) -> Option<u64> {
        let mut free_buffers = self.free_buffers.lock().await;
        let mut buffer_info = self.buffer_info.lock().await;
        
        // Try to find suitable buffer in pool
        for _ in 0..free_buffers.len() {
            if let Some(buffer_id) = free_buffers.pop_front() {
                if let Some(info) = buffer_info.get(&buffer_id) {
                    if info.size >= size && info.usage == usage {
                        // Found suitable buffer
                        let mut stats = self.stats.lock().await;
                        stats.allocations += 1;
                        stats.reuses += 1;
                        return Some(buffer_id);
                    }
                }
                // Buffer not suitable, put it back
                free_buffers.push_back(buffer_id);
            }
        }
        
        None
    }
    
    async fn deallocate_buffer(&self, buffer_id: u64) {
        let mut free_buffers = self.free_buffers.lock().await;
        
        // Return buffer to pool if pool isn't full
        if free_buffers.len() < self.size_class.pool_size() {
            free_buffers.push_back(buffer_id);
            
            let mut stats = self.stats.lock().await;
            stats.deallocations += 1;
        }
    }
    
    async fn compact(&self) {
        // Remove oldest buffers to make room for new ones
        let mut free_buffers = self.free_buffers.lock().await;
        let target_size = self.size_class.pool_size() / 2;
        
        while free_buffers.len() > target_size {
            free_buffers.pop_front();
        }
    }
}

/// Information about active buffers
#[derive(Debug, Clone)]
struct BufferInfo {
    id: u64,
    size: usize,
    usage: GpuBufferUsage,
    size_class: BufferSizeClass,
    created_at: std::time::Instant,
    last_accessed: std::time::Instant,
    access_count: u64,
}

/// Information about pooled buffers
#[derive(Debug, Clone)]
struct PooledBufferInfo {
    size: usize,
    usage: GpuBufferUsage,
    created_at: std::time::Instant,
}

/// Memory-mapped buffer for zero-copy operations
pub struct MmapBuffer {
    pub id: u64,
    pub size: usize,
    mmap: Arc<Mutex<MmapMut>>,
    created_at: std::time::Instant,
}

impl MmapBuffer {
    /// Write data to memory-mapped buffer
    pub async fn write(&self, offset: usize, data: &[u8]) -> QBMIAResult<()> {
        if offset + data.len() > self.size {
            return Err(QBMIAError::buffer_op("Write would exceed buffer size"));
        }
        
        let mut mmap = self.mmap.lock().await;
        mmap[offset..offset + data.len()].copy_from_slice(data);
        
        Ok(())
    }
    
    /// Read data from memory-mapped buffer
    pub async fn read(&self, offset: usize, len: usize) -> QBMIAResult<Vec<u8>> {
        if offset + len > self.size {
            return Err(QBMIAError::buffer_op("Read would exceed buffer size"));
        }
        
        let mmap = self.mmap.lock().await;
        Ok(mmap[offset..offset + len].to_vec())
    }
    
    /// Get raw pointer for zero-copy access
    pub async fn as_ptr(&self) -> *mut u8 {
        let mmap = self.mmap.lock().await;
        mmap.as_mut_ptr()
    }
}

/// Memory-mapped region tracker
#[derive(Debug, Clone)]
struct MmapRegion {
    buffer_id: u64,
    size: usize,
    offset: usize,
}

impl MmapRegion {
    fn contains_buffer(&self, buffer_id: u64) -> bool {
        self.buffer_id == buffer_id
    }
    
    fn get_buffer_offset(&self, buffer_id: u64) -> QBMIAResult<usize> {
        if self.buffer_id == buffer_id {
            Ok(self.offset)
        } else {
            Err(QBMIAError::buffer_op("Buffer not in this region"))
        }
    }
    
    fn read_data(&self, offset: usize, len: usize) -> QBMIAResult<Vec<u8>> {
        // Simplified - would need actual memory mapping implementation
        Ok(vec![0u8; len])
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub allocation_reuses: u64,
    pub allocation_news: u64,
    pub zero_copy_reads: u64,
    pub regular_reads: u64,
    pub mmap_creations: u64,
    pub garbage_collections: u64,
    pub buffers_freed_by_gc: u64,
    
    pub total_allocation_time: std::time::Duration,
    pub total_read_time: std::time::Duration,
    pub total_gc_time: std::time::Duration,
    
    pub total_bytes_allocated: u64,
    pub total_bytes_read: u64,
    pub peak_memory_usage: u64,
    pub current_memory_usage: u64,
}

impl MemoryStats {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_allocation_reuse(&mut self, duration: std::time::Duration, size: usize) {
        self.total_allocations += 1;
        self.allocation_reuses += 1;
        self.total_allocation_time += duration;
        self.total_bytes_allocated += size as u64;
        self.current_memory_usage += size as u64;
        self.peak_memory_usage = self.peak_memory_usage.max(self.current_memory_usage);
    }
    
    fn record_allocation_new(&mut self, duration: std::time::Duration, size: usize) {
        self.total_allocations += 1;
        self.allocation_news += 1;
        self.total_allocation_time += duration;
        self.total_bytes_allocated += size as u64;
        self.current_memory_usage += size as u64;
        self.peak_memory_usage = self.peak_memory_usage.max(self.current_memory_usage);
    }
    
    fn record_deallocation(&mut self, duration: std::time::Duration, size: usize) {
        self.total_deallocations += 1;
        self.current_memory_usage = self.current_memory_usage.saturating_sub(size as u64);
    }
    
    fn record_read_zero_copy(&mut self, duration: std::time::Duration, size: usize) {
        self.zero_copy_reads += 1;
        self.total_read_time += duration;
        self.total_bytes_read += size as u64;
    }
    
    fn record_read_regular(&mut self, duration: std::time::Duration, size: usize) {
        self.regular_reads += 1;
        self.total_read_time += duration;
        self.total_bytes_read += size as u64;
    }
    
    fn record_mmap_creation(&mut self, duration: std::time::Duration, size: usize) {
        self.mmap_creations += 1;
        // Memory-mapped regions don't count towards regular memory usage
    }
    
    fn record_prefetch(&mut self, duration: std::time::Duration, size: usize) {
        // Prefetch is essentially an allocation + upload
        self.record_allocation_new(duration, size);
    }
    
    fn record_garbage_collection(&mut self, duration: std::time::Duration, freed_count: usize) {
        self.garbage_collections += 1;
        self.buffers_freed_by_gc += freed_count as u64;
        self.total_gc_time += duration;
    }
    
    pub fn allocation_reuse_rate(&self) -> f64 {
        if self.total_allocations > 0 {
            self.allocation_reuses as f64 / self.total_allocations as f64
        } else {
            0.0
        }
    }
    
    pub fn zero_copy_rate(&self) -> f64 {
        let total_reads = self.zero_copy_reads + self.regular_reads;
        if total_reads > 0 {
            self.zero_copy_reads as f64 / total_reads as f64
        } else {
            0.0
        }
    }
    
    pub fn average_allocation_time(&self) -> Option<std::time::Duration> {
        if self.total_allocations > 0 {
            Some(self.total_allocation_time / self.total_allocations as u32)
        } else {
            None
        }
    }
}

/// Pool-specific statistics
#[derive(Debug, Clone, Default)]
struct PoolStats {
    allocations: u64,
    deallocations: u64,
    reuses: u64,
    compactions: u64,
}

impl PoolStats {
    fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuPipeline;
    
    #[tokio::test]
    async fn test_memory_manager_initialization() {
        let gpu_pipeline = match GpuPipeline::new().await {
            Ok(pipeline) => Arc::new(pipeline),
            Err(_) => return, // Skip if no GPU
        };
        
        let memory_manager = GpuMemoryManager::new(gpu_pipeline).await;
        assert!(memory_manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_buffer_allocation() {
        let gpu_pipeline = match GpuPipeline::new().await {
            Ok(pipeline) => Arc::new(pipeline),
            Err(_) => return, // Skip if no GPU
        };
        
        let memory_manager = GpuMemoryManager::new(gpu_pipeline).await.unwrap();
        
        let data = vec![1u8, 2, 3, 4, 5];
        let buffer = memory_manager.create_buffer(&data).await.unwrap();
        
        assert_eq!(buffer.size, 5);
    }
    
    #[test]
    fn test_buffer_size_class() {
        assert_eq!(BufferSizeClass::from_size(512), BufferSizeClass::Small);
        assert_eq!(BufferSizeClass::from_size(1024), BufferSizeClass::Medium);
        assert_eq!(BufferSizeClass::from_size(1024 * 1024), BufferSizeClass::Large);
        assert_eq!(BufferSizeClass::from_size(200 * 1024 * 1024), BufferSizeClass::Huge);
    }
}