//! Quantum Memory Management System
//!
//! This module provides quantum memory management, including quantum state storage,
//! memory pools, garbage collection, and memory optimization for quantum operations.

use crate::quantum_state::{QuantumState, ComplexAmplitude};
use crate::error::{QuantumError, QuantumResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
// use std::alloc::Layout;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{debug, info};
use chrono::{DateTime, Utc};


/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in megabytes
    pub initial_size_mb: usize,
    /// Maximum pool size in megabytes
    pub max_size_mb: usize,
    /// Growth factor for pool expansion
    pub growth_factor: f64,
    /// Whether garbage collection is enabled
    pub enable_garbage_collection: bool,
    /// Threshold percentage for garbage collection
    pub gc_threshold_percentage: f64,
    /// Whether compression is enabled
    pub enable_compression: bool,
    /// Whether prefetching is enabled
    pub enable_prefetching: bool,
    /// Whether NUMA awareness is enabled
    pub numa_aware: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size_mb: 64,
            max_size_mb: 1024,
            growth_factor: 1.5,
            enable_garbage_collection: true,
            gc_threshold_percentage: 80.0,
            enable_compression: false,
            enable_prefetching: true,
            numa_aware: false,
        }
    }
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
    BuddySystem,
}

/// Memory block metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBlock {
    /// Unique identifier for the memory block
    pub id: String,
    /// Size of the block in bytes
    pub size_bytes: usize,
    /// Offset within the memory pool
    pub offset: usize,
    /// Whether the block is allocated
    pub is_allocated: bool,
    /// Timestamp when block was allocated
    pub allocated_at: Option<DateTime<Utc>>,
    /// Timestamp of last access
    pub last_accessed: Option<DateTime<Utc>>,
    /// Number of times this block was accessed
    pub access_count: u64,
    /// Type of data stored in this block
    pub data_type: MemoryDataType,
    /// Additional metadata for the block
    pub metadata: HashMap<String, String>,
}

/// Types of data stored in quantum memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryDataType {
    QuantumState,
    ComplexAmplitudes,
    GateMatrices,
    MeasurementResults,
    TemporaryBuffer,
    CachedResults,
    ComputationGraph,
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memory in bytes
    pub total_bytes: usize,
    /// Allocated memory in bytes
    pub allocated_bytes: usize,
    /// Free memory in bytes
    pub free_bytes: usize,
    /// Memory utilization percentage
    pub utilization_percentage: f64,
    /// Memory fragmentation percentage
    pub fragmentation_percentage: f64,
    /// Number of allocations performed
    pub allocation_count: u64,
    /// Number of deallocations performed
    pub deallocation_count: u64,
    /// Number of garbage collection runs
    pub gc_runs: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Number of prefetch hits
    pub prefetch_hits: u64,
    /// Timestamp of last garbage collection
    pub last_gc_time: Option<DateTime<Utc>>,
    /// Average allocation size in bytes
    pub average_allocation_size: f64,
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            allocated_bytes: 0,
            free_bytes: 0,
            utilization_percentage: 0.0,
            fragmentation_percentage: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
            gc_runs: 0,
            cache_hits: 0,
            cache_misses: 0,
            prefetch_hits: 0,
            last_gc_time: None,
            average_allocation_size: 0.0,
            peak_usage_bytes: 0,
            last_updated: Utc::now(),
        }
    }
}

/// Memory pool for quantum operations
#[derive(Debug)]
pub struct MemoryPool {
    pub id: String,
    pub config: MemoryPoolConfig,
    pub strategy: AllocationStrategy,
    blocks: Arc<RwLock<Vec<MemoryBlock>>>,
    free_blocks: Arc<RwLock<VecDeque<usize>>>,
    stats: Arc<RwLock<MemoryStats>>,
    memory_data: Arc<RwLock<Vec<u8>>>,
    cache: Arc<RwLock<HashMap<String, CachedEntry>>>,
    prefetch_queue: Arc<Mutex<VecDeque<String>>>,
}

#[derive(Debug, Clone)]
struct CachedEntry {
    data: Vec<u8>,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u64,
    size_bytes: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(config: MemoryPoolConfig, strategy: AllocationStrategy) -> QuantumResult<Self> {
        let id = format!("pool_{}", uuid::Uuid::new_v4());
        let initial_size_bytes = config.initial_size_mb * 1024 * 1024;
        
        let pool = Self {
            id,
            config,
            strategy,
            blocks: Arc::new(RwLock::new(Vec::new())),
            free_blocks: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            memory_data: Arc::new(RwLock::new(vec![0; initial_size_bytes])),
            cache: Arc::new(RwLock::new(HashMap::new())),
            prefetch_queue: Arc::new(Mutex::new(VecDeque::new())),
        };
        
        // Initialize the pool
        pool.initialize()?;
        
        info!("Memory pool {} created with {} MB initial size", pool.id, pool.config.initial_size_mb);
        // metrics::counter!("memory_pools_created_total", 1);
        
        Ok(pool)
    }

    /// Initialize the memory pool
    fn initialize(&self) -> QuantumResult<()> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        // Create initial free block
        let initial_block = MemoryBlock {
            id: format!("block_{}", uuid::Uuid::new_v4()),
            size_bytes: self.config.initial_size_mb * 1024 * 1024,
            offset: 0,
            is_allocated: false,
            allocated_at: None,
            last_accessed: None,
            access_count: 0,
            data_type: MemoryDataType::TemporaryBuffer,
            metadata: HashMap::new(),
        };
        
        blocks.push(initial_block);
        free_blocks.push_back(0);
        
        stats.total_bytes = self.config.initial_size_mb * 1024 * 1024;
        stats.free_bytes = stats.total_bytes;
        stats.last_updated = Utc::now();
        
        Ok(())
    }

    /// Allocate memory for quantum data
    pub fn allocate(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        let block_id = match self.strategy {
            AllocationStrategy::FirstFit => self.allocate_first_fit(size_bytes, data_type)?,
            AllocationStrategy::BestFit => self.allocate_best_fit(size_bytes, data_type)?,
            AllocationStrategy::WorstFit => self.allocate_worst_fit(size_bytes, data_type)?,
            AllocationStrategy::NextFit => self.allocate_next_fit(size_bytes, data_type)?,
            AllocationStrategy::BuddySystem => self.allocate_buddy_system(size_bytes, data_type)?,
        };
        
        self.update_allocation_stats(size_bytes);
        
        debug!("Allocated {} bytes for {:?}, block ID: {}", size_bytes, data_type, block_id);
        // metrics::counter!("memory_allocations_total", 1);
        // metrics::histogram!("memory_allocation_size_bytes", size_bytes as f64);
        
        Ok(block_id)
    }

    /// Deallocate memory block
    pub fn deallocate(&self, block_id: &str) -> QuantumResult<()> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Find the block to deallocate
        let block_index = blocks.iter().position(|b| b.id == block_id)
            .ok_or_else(|| QuantumError::memory_error("deallocation", format!("Block not found: {}", block_id)))?;
        
        let block = &mut blocks[block_index];
        
        if !block.is_allocated {
            return Err(QuantumError::memory_error("deallocation", format!("Block {} is not allocated", block_id)));
        }
        
        let size_bytes = block.size_bytes;
        block.is_allocated = false;
        block.allocated_at = None;
        block.last_accessed = None;
        block.access_count = 0;
        
        free_blocks.push_back(block_index);
        
        self.update_deallocation_stats(size_bytes);
        
        debug!("Deallocated block: {}", block_id);
        // metrics::counter!("memory_deallocations_total", 1);
        
        // Try to coalesce adjacent free blocks
        self.coalesce_free_blocks()?;
        
        Ok(())
    }

    /// Allocate using first-fit strategy
    fn allocate_first_fit(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Find first free block that can accommodate the request
        let mut selected_index = None;
        let mut free_index = None;
        
        for (i, &block_index) in free_blocks.iter().enumerate() {
            if blocks[block_index].size_bytes >= size_bytes {
                selected_index = Some(block_index);
                free_index = Some(i);
                break;
            }
        }
        
        if let (Some(block_index), Some(free_idx)) = (selected_index, free_index) {
            let block = &mut blocks[block_index];
            
            // If the block is larger than needed, split it
            if block.size_bytes > size_bytes {
                self.split_block(block_index, size_bytes, &mut blocks, &mut free_blocks)?;
            }
            
            // Allocate the block
            let block = &mut blocks[block_index];
            block.is_allocated = true;
            block.allocated_at = Some(Utc::now());
            block.data_type = data_type;
            
            free_blocks.remove(free_idx);
            
            Ok(block.id.clone())
        } else {
            // No suitable block found, try to expand the pool
            self.expand_pool(size_bytes)?;
            self.allocate_first_fit(size_bytes, data_type)
        }
    }

    /// Allocate using best-fit strategy
    fn allocate_best_fit(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Find the smallest free block that can accommodate the request
        let mut best_block_index = None;
        let mut best_free_index = None;
        let mut best_size = usize::MAX;
        
        for (i, &block_index) in free_blocks.iter().enumerate() {
            let block_size = blocks[block_index].size_bytes;
            if block_size >= size_bytes && block_size < best_size {
                best_block_index = Some(block_index);
                best_free_index = Some(i);
                best_size = block_size;
            }
        }
        
        if let (Some(block_index), Some(free_idx)) = (best_block_index, best_free_index) {
            if blocks[block_index].size_bytes > size_bytes {
                self.split_block(block_index, size_bytes, &mut blocks, &mut free_blocks)?;
            }
            
            let block = &mut blocks[block_index];
            block.is_allocated = true;
            block.allocated_at = Some(Utc::now());
            block.data_type = data_type;
            
            free_blocks.remove(free_idx);
            
            Ok(block.id.clone())
        } else {
            self.expand_pool(size_bytes)?;
            self.allocate_best_fit(size_bytes, data_type)
        }
    }

    /// Allocate using worst-fit strategy
    fn allocate_worst_fit(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Find the largest free block that can accommodate the request
        let mut worst_block_index = None;
        let mut worst_free_index = None;
        let mut worst_size = 0;
        
        for (i, &block_index) in free_blocks.iter().enumerate() {
            let block_size = blocks[block_index].size_bytes;
            if block_size >= size_bytes && block_size > worst_size {
                worst_block_index = Some(block_index);
                worst_free_index = Some(i);
                worst_size = block_size;
            }
        }
        
        if let (Some(block_index), Some(free_idx)) = (worst_block_index, worst_free_index) {
            if blocks[block_index].size_bytes > size_bytes {
                self.split_block(block_index, size_bytes, &mut blocks, &mut free_blocks)?;
            }
            
            let block = &mut blocks[block_index];
            block.is_allocated = true;
            block.allocated_at = Some(Utc::now());
            block.data_type = data_type;
            
            free_blocks.remove(free_idx);
            
            Ok(block.id.clone())
        } else {
            self.expand_pool(size_bytes)?;
            self.allocate_worst_fit(size_bytes, data_type)
        }
    }

    /// Allocate using next-fit strategy (simplified as first-fit for now)
    fn allocate_next_fit(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        // For simplicity, using first-fit strategy
        self.allocate_first_fit(size_bytes, data_type)
    }

    /// Allocate using buddy system (simplified implementation)
    fn allocate_buddy_system(&self, size_bytes: usize, data_type: MemoryDataType) -> QuantumResult<String> {
        // For simplicity, using best-fit strategy
        self.allocate_best_fit(size_bytes, data_type)
    }

    /// Split a memory block
    fn split_block(
        &self,
        block_index: usize,
        allocation_size: usize,
        blocks: &mut Vec<MemoryBlock>,
        free_blocks: &mut VecDeque<usize>,
    ) -> QuantumResult<()> {
        let original_block = &blocks[block_index];
        let remaining_size = original_block.size_bytes - allocation_size;
        
        if remaining_size > 0 {
            let new_block = MemoryBlock {
                id: format!("block_{}", uuid::Uuid::new_v4()),
                size_bytes: remaining_size,
                offset: original_block.offset + allocation_size,
                is_allocated: false,
                allocated_at: None,
                last_accessed: None,
                access_count: 0,
                data_type: MemoryDataType::TemporaryBuffer,
                metadata: HashMap::new(),
            };
            
            blocks.push(new_block);
            free_blocks.push_back(blocks.len() - 1);
        }
        
        // Update the original block size
        blocks[block_index].size_bytes = allocation_size;
        
        Ok(())
    }

    /// Coalesce adjacent free blocks
    fn coalesce_free_blocks(&self) -> QuantumResult<()> {
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Sort free blocks by offset
        let mut free_indices: Vec<usize> = free_blocks.iter().cloned().collect();
        free_indices.sort_by(|&a, &b| blocks[a].offset.cmp(&blocks[b].offset));
        
        let mut coalesced = false;
        let mut i = 0;
        
        while i < free_indices.len() - 1 {
            let current_idx = free_indices[i];
            let next_idx = free_indices[i + 1];
            
            let current_block = &blocks[current_idx];
            let next_block = &blocks[next_idx];
            
            // Check if blocks are adjacent
            if current_block.offset + current_block.size_bytes == next_block.offset {
                // Coalesce the blocks
                blocks[current_idx].size_bytes += next_block.size_bytes;
                
                // Remove the next block from free list
                free_blocks.retain(|&x| x != next_idx);
                free_indices.remove(i + 1);
                
                coalesced = true;
            } else {
                i += 1;
            }
        }
        
        if coalesced {
            debug!("Coalesced free blocks in pool {}", self.id);
        }
        
        Ok(())
    }

    /// Expand the memory pool
    fn expand_pool(&self, min_additional_size: usize) -> QuantumResult<()> {
        let mut memory_data = self.memory_data.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        
        let current_size = memory_data.len();
        let expansion_size = (min_additional_size as f64 * self.config.growth_factor) as usize;
        let new_size = current_size + expansion_size;
        
        // Check if expansion exceeds max size
        if new_size > self.config.max_size_mb * 1024 * 1024 {
            return Err(QuantumError::memory_error("expansion", format!(
                "Cannot expand pool beyond max size: {} MB",
                self.config.max_size_mb
            )));
        }
        
        // Expand the memory data
        memory_data.resize(new_size, 0);
        
        // Add new free block
        let mut blocks = self.blocks.write().unwrap();
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        let new_block = MemoryBlock {
            id: format!("block_{}", uuid::Uuid::new_v4()),
            size_bytes: expansion_size,
            offset: current_size,
            is_allocated: false,
            allocated_at: None,
            last_accessed: None,
            access_count: 0,
            data_type: MemoryDataType::TemporaryBuffer,
            metadata: HashMap::new(),
        };
        
        blocks.push(new_block);
        free_blocks.push_back(blocks.len() - 1);
        
        // Update stats
        stats.total_bytes = new_size;
        stats.free_bytes += expansion_size;
        stats.last_updated = Utc::now();
        
        info!("Expanded memory pool {} by {} MB", self.id, expansion_size / (1024 * 1024));
        // metrics::counter!("memory_pool_expansions_total", 1);
        
        Ok(())
    }

    /// Update allocation statistics
    fn update_allocation_stats(&self, size_bytes: usize) {
        let mut stats = self.stats.write().unwrap();
        
        stats.allocation_count += 1;
        stats.allocated_bytes += size_bytes;
        stats.free_bytes -= size_bytes;
        stats.utilization_percentage = (stats.allocated_bytes as f64 / stats.total_bytes as f64) * 100.0;
        
        if stats.allocated_bytes > stats.peak_usage_bytes {
            stats.peak_usage_bytes = stats.allocated_bytes;
        }
        
        stats.average_allocation_size = stats.allocated_bytes as f64 / stats.allocation_count as f64;
        stats.last_updated = Utc::now();
        
        // metrics::gauge!("memory_pool_utilization_percentage", stats.utilization_percentage, "pool_id" => self.id.clone());
        // metrics::gauge!("memory_pool_allocated_bytes", stats.allocated_bytes as f64, "pool_id" => self.id.clone());
    }

    /// Update deallocation statistics
    fn update_deallocation_stats(&self, size_bytes: usize) {
        let mut stats = self.stats.write().unwrap();
        
        stats.deallocation_count += 1;
        stats.allocated_bytes -= size_bytes;
        stats.free_bytes += size_bytes;
        stats.utilization_percentage = (stats.allocated_bytes as f64 / stats.total_bytes as f64) * 100.0;
        stats.last_updated = Utc::now();
        
        // metrics::gauge!("memory_pool_utilization_percentage", stats.utilization_percentage, "pool_id" => self.id.clone());
        // metrics::gauge!("memory_pool_allocated_bytes", stats.allocated_bytes as f64, "pool_id" => self.id.clone());
    }

    /// Run garbage collection
    pub fn garbage_collect(&self) -> QuantumResult<()> {
        if !self.config.enable_garbage_collection {
            return Ok(());
        }
        
        let stats = self.stats.read().unwrap();
        let current_utilization = stats.utilization_percentage;
        drop(stats);
        
        if current_utilization < self.config.gc_threshold_percentage {
            return Ok(());
        }
        
        info!("Running garbage collection for pool {}", self.id);
        
        // Clean up old cache entries
        self.cleanup_cache()?;
        
        // Coalesce free blocks
        self.coalesce_free_blocks()?;
        
        // Update GC stats
        let mut stats = self.stats.write().unwrap();
        stats.gc_runs += 1;
        stats.last_gc_time = Some(Utc::now());
        stats.last_updated = Utc::now();
        
        // metrics::counter!("memory_pool_gc_runs_total", 1);
        
        info!("Garbage collection completed for pool {}", self.id);
        Ok(())
    }

    /// Clean up old cache entries
    fn cleanup_cache(&self) -> QuantumResult<()> {
        let mut cache = self.cache.write().unwrap();
        let now = Utc::now();
        let cache_ttl = chrono::Duration::hours(1); // 1 hour TTL
        
        let mut to_remove = Vec::new();
        
        for (key, entry) in cache.iter() {
            if now.signed_duration_since(entry.last_accessed) > cache_ttl {
                to_remove.push(key.clone());
            }
        }
        
        for key in to_remove {
            cache.remove(&key);
        }
        
        Ok(())
    }

    /// Store data in cache
    pub fn cache_store(&self, key: String, data: Vec<u8>) -> QuantumResult<()> {
        let mut cache = self.cache.write().unwrap();
        
        let data_len = data.len();
        let entry = CachedEntry {
            data,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            size_bytes: data_len,
        };
        
        cache.insert(key, entry);
        
        // metrics::counter!("memory_cache_stores_total", 1);
        Ok(())
    }

    /// Retrieve data from cache
    pub fn cache_retrieve(&self, key: &str) -> Option<Vec<u8>> {
        let mut cache = self.cache.write().unwrap();
        
        if let Some(entry) = cache.get_mut(key) {
            entry.last_accessed = Utc::now();
            entry.access_count += 1;
            
            let mut stats = self.stats.write().unwrap();
            stats.cache_hits += 1;
            
            // metrics::counter!("memory_cache_hits_total", 1);
            Some(entry.data.clone())
        } else {
            let mut stats = self.stats.write().unwrap();
            stats.cache_misses += 1;
            
            // metrics::counter!("memory_cache_misses_total", 1);
            None
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }

    /// Get memory block information
    pub fn get_block_info(&self, block_id: &str) -> Option<MemoryBlock> {
        let blocks = self.blocks.read().unwrap();
        blocks.iter().find(|b| b.id == block_id).cloned()
    }

    /// List all allocated blocks
    pub fn list_allocated_blocks(&self) -> Vec<MemoryBlock> {
        let blocks = self.blocks.read().unwrap();
        blocks.iter().filter(|b| b.is_allocated).cloned().collect()
    }

    /// Calculate fragmentation
    pub fn calculate_fragmentation(&self) -> f64 {
        let blocks = self.blocks.read().unwrap();
        let free_blocks = self.free_blocks.read().unwrap();
        
        if free_blocks.is_empty() {
            return 0.0;
        }
        
        let total_free_size: usize = free_blocks.iter()
            .map(|&idx| blocks[idx].size_bytes)
            .sum();
        
        let largest_free_block = free_blocks.iter()
            .map(|&idx| blocks[idx].size_bytes)
            .max()
            .unwrap_or(0);
        
        if total_free_size == 0 {
            0.0
        } else {
            (1.0 - (largest_free_block as f64 / total_free_size as f64)) * 100.0
        }
    }
}

/// Quantum memory manager
#[derive(Debug)]
pub struct QuantumMemoryManager {
    pools: Arc<AsyncRwLock<HashMap<String, MemoryPool>>>,
    default_pool: Arc<RwLock<Option<String>>>,
    quantum_states: Arc<AsyncRwLock<HashMap<String, QuantumState>>>,
    allocation_tracker: Arc<RwLock<HashMap<String, String>>>, // block_id -> pool_id
}

impl QuantumMemoryManager {
    /// Create a new quantum memory manager
    pub fn new() -> QuantumResult<Self> {
        let manager = Self {
            pools: Arc::new(AsyncRwLock::new(HashMap::new())),
            default_pool: Arc::new(RwLock::new(None)),
            quantum_states: Arc::new(AsyncRwLock::new(HashMap::new())),
            allocation_tracker: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Create default pool
        manager.create_default_pool()?;
        
        info!("Quantum memory manager initialized");
        Ok(manager)
    }

    /// Create default memory pool
    fn create_default_pool(&self) -> QuantumResult<()> {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::BestFit)?;
        let pool_id = pool.id.clone();
        
        // This is a synchronous operation during initialization
        let mut pools = match self.pools.try_write() {
            Ok(guard) => guard,
            Err(_) => return Err(QuantumError::MemoryError { operation: "Lock acquisition".to_string(), message: "Failed to acquire pools lock".to_string() }),
        };
        
        pools.insert(pool_id.clone(), pool);
        
        let mut default_pool = self.default_pool.write().unwrap();
        *default_pool = Some(pool_id);
        
        Ok(())
    }

    /// Create a new memory pool
    pub async fn create_pool(&self, config: MemoryPoolConfig, strategy: AllocationStrategy) -> QuantumResult<String> {
        let pool = MemoryPool::new(config, strategy)?;
        let pool_id = pool.id.clone();
        
        let mut pools = self.pools.write().await;
        pools.insert(pool_id.clone(), pool);
        
        info!("Created memory pool: {}", pool_id);
        Ok(pool_id)
    }

    /// Remove a memory pool
    pub async fn remove_pool(&self, pool_id: &str) -> QuantumResult<()> {
        let mut pools = self.pools.write().await;
        pools.remove(pool_id);
        
        // Remove from allocation tracker
        let mut tracker = self.allocation_tracker.write().unwrap();
        tracker.retain(|_, pid| pid != pool_id);
        
        info!("Removed memory pool: {}", pool_id);
        Ok(())
    }

    /// Allocate memory for quantum state
    pub async fn allocate_quantum_state(&self, num_qubits: usize) -> QuantumResult<String> {
        let size_bytes = (1 << num_qubits) * std::mem::size_of::<ComplexAmplitude>();
        
        let pools = self.pools.read().await;
        let default_pool_id = self.default_pool.read().unwrap().clone()
            .ok_or_else(|| QuantumError::memory_error("allocation", "No default pool available"))?;
        
        let pool = pools.get(&default_pool_id)
            .ok_or_else(|| QuantumError::memory_error("allocation", "Default pool not found"))?;
        
        let block_id = pool.allocate(size_bytes, MemoryDataType::QuantumState)?;
        
        // Track allocation
        let mut tracker = self.allocation_tracker.write().unwrap();
        tracker.insert(block_id.clone(), default_pool_id);
        
        info!("Allocated quantum state storage for {} qubits", num_qubits);
        Ok(block_id)
    }

    /// Store quantum state
    pub async fn store_quantum_state(&self, state_id: String, state: QuantumState) -> QuantumResult<()> {
        let mut states = self.quantum_states.write().await;
        states.insert(state_id, state);
        
        // metrics::counter!("quantum_states_stored_total", 1);
        Ok(())
    }

    /// Retrieve quantum state
    pub async fn retrieve_quantum_state(&self, state_id: &str) -> Option<QuantumState> {
        let states = self.quantum_states.read().await;
        states.get(state_id).cloned()
    }

    /// Deallocate memory block
    pub async fn deallocate(&self, block_id: &str) -> QuantumResult<()> {
        let tracker = self.allocation_tracker.read().unwrap();
        let pool_id = tracker.get(block_id)
            .ok_or_else(|| QuantumError::memory_error("tracking", format!("Block {} not tracked", block_id)))?;
        
        let pools = self.pools.read().await;
        let pool = pools.get(pool_id)
            .ok_or_else(|| QuantumError::memory_error("deallocation", format!("Pool {} not found", pool_id)))?;
        
        pool.deallocate(block_id)?;
        
        // Remove from tracker
        drop(tracker);
        let mut tracker = self.allocation_tracker.write().unwrap();
        tracker.remove(block_id);
        
        Ok(())
    }

    /// Get memory statistics for all pools
    pub async fn get_global_stats(&self) -> HashMap<String, MemoryStats> {
        let pools = self.pools.read().await;
        pools.iter()
            .map(|(id, pool)| (id.clone(), pool.get_stats()))
            .collect()
    }

    /// Run garbage collection on all pools
    pub async fn global_garbage_collect(&self) -> QuantumResult<()> {
        let pools = self.pools.read().await;
        
        for pool in pools.values() {
            pool.garbage_collect()?;
        }
        
        info!("Global garbage collection completed");
        Ok(())
    }

    /// Get total memory usage
    pub async fn get_total_memory_usage(&self) -> (usize, usize) {
        let pools = self.pools.read().await;
        let mut total_allocated = 0;
        let mut total_capacity = 0;
        
        for pool in pools.values() {
            let stats = pool.get_stats();
            total_allocated += stats.allocated_bytes;
            total_capacity += stats.total_bytes;
        }
        
        (total_allocated, total_capacity)
    }

    /// Optimize memory layout
    pub async fn optimize_memory_layout(&self) -> QuantumResult<()> {
        let pools = self.pools.read().await;
        
        for pool in pools.values() {
            pool.coalesce_free_blocks()?;
        }
        
        info!("Memory layout optimization completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit);
        
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_bytes, 64 * 1024 * 1024); // 64MB default
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_memory_allocation_first_fit() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        let block_id = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        assert!(!block_id.is_empty());
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_bytes, 1024);
        assert_eq!(stats.allocation_count, 1);
    }

    #[test]
    fn test_memory_allocation_best_fit() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::BestFit).unwrap();
        
        let block_id = pool.allocate(2048, MemoryDataType::ComplexAmplitudes).unwrap();
        assert!(!block_id.is_empty());
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_bytes, 2048);
    }

    #[test]
    fn test_memory_deallocation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        let block_id = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        assert!(pool.deallocate(&block_id).is_ok());
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_memory_pool_expansion() {
        let mut config = MemoryPoolConfig::default();
        config.initial_size_mb = 1; // Small initial size
        config.max_size_mb = 10;
        
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        // Allocate more than initial size
        let large_size = 2 * 1024 * 1024; // 2MB
        let block_id = pool.allocate(large_size, MemoryDataType::QuantumState);
        
        assert!(block_id.is_ok());
        let stats = pool.get_stats();
        assert!(stats.total_bytes > 1024 * 1024); // Should have expanded
    }

    #[test]
    fn test_cache_operations() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        let key = "test_key".to_string();
        let data = vec![1, 2, 3, 4, 5];
        
        // Store in cache
        assert!(pool.cache_store(key.clone(), data.clone()).is_ok());
        
        // Retrieve from cache
        let retrieved = pool.cache_retrieve(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
        
        // Cache miss
        let missing = pool.cache_retrieve("non_existent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_fragmentation_calculation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        // Initially no fragmentation
        let initial_fragmentation = pool.calculate_fragmentation();
        assert_eq!(initial_fragmentation, 0.0);
        
        // Allocate and deallocate to create fragmentation
        let block1 = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        let block2 = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        let block3 = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        
        // Deallocate middle block to create fragmentation
        pool.deallocate(&block2).unwrap();
        
        let fragmentation = pool.calculate_fragmentation();
        assert!(fragmentation >= 0.0);
    }

    #[test]
    fn test_garbage_collection() {
        let mut config = MemoryPoolConfig::default();
        config.enable_garbage_collection = true;
        config.gc_threshold_percentage = 50.0;
        
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        // Allocate some memory to trigger GC threshold
        let _block1 = pool.allocate(32 * 1024 * 1024, MemoryDataType::QuantumState).unwrap();
        
        let initial_stats = pool.get_stats();
        assert!(pool.garbage_collect().is_ok());
        
        let final_stats = pool.get_stats();
        assert_eq!(final_stats.gc_runs, initial_stats.gc_runs + 1);
    }

    #[tokio::test]
    async fn test_quantum_memory_manager() {
        let manager = QuantumMemoryManager::new().unwrap();
        
        // Create additional pool
        let config = MemoryPoolConfig::default();
        let pool_id = manager.create_pool(config, AllocationStrategy::BestFit).await.unwrap();
        
        // Allocate quantum state
        let state_id = manager.allocate_quantum_state(2).await.unwrap();
        
        // Store quantum state
        let state = QuantumState::new(2).unwrap();
        manager.store_quantum_state("test_state".to_string(), state).await.unwrap();
        
        // Retrieve quantum state
        let retrieved = manager.retrieve_quantum_state("test_state").await;
        assert!(retrieved.is_some());
        
        // Deallocate
        manager.deallocate(&state_id).await.unwrap();
        
        // Remove pool
        manager.remove_pool(&pool_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_global_stats() {
        let manager = QuantumMemoryManager::new().unwrap();
        
        let stats = manager.get_global_stats().await;
        assert!(stats.len() > 0); // Should have at least the default pool
        
        let (allocated, capacity) = manager.get_total_memory_usage().await;
        assert_eq!(allocated, 0);
        assert!(capacity > 0);
    }

    #[tokio::test]
    async fn test_global_garbage_collection() {
        let manager = QuantumMemoryManager::new().unwrap();
        
        assert!(manager.global_garbage_collect().await.is_ok());
    }

    #[tokio::test]
    async fn test_memory_optimization() {
        let manager = QuantumMemoryManager::new().unwrap();
        
        assert!(manager.optimize_memory_layout().await.is_ok());
    }

    #[test]
    fn test_allocation_strategies() {
        let config = MemoryPoolConfig::default();
        let strategies = vec![
            AllocationStrategy::FirstFit,
            AllocationStrategy::BestFit,
            AllocationStrategy::WorstFit,
            AllocationStrategy::NextFit,
            AllocationStrategy::BuddySystem,
        ];
        
        for strategy in strategies {
            let pool = MemoryPool::new(config.clone(), strategy).unwrap();
            let block_id = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
            assert!(!block_id.is_empty());
            assert!(pool.deallocate(&block_id).is_ok());
        }
    }

    #[test]
    fn test_memory_data_types() {
        let data_types = vec![
            MemoryDataType::QuantumState,
            MemoryDataType::ComplexAmplitudes,
            MemoryDataType::GateMatrices,
            MemoryDataType::MeasurementResults,
            MemoryDataType::TemporaryBuffer,
            MemoryDataType::CachedResults,
            MemoryDataType::ComputationGraph,
        ];
        
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        for data_type in data_types {
            let block_id = pool.allocate(1024, data_type).unwrap();
            let block_info = pool.get_block_info(&block_id).unwrap();
            assert_eq!(block_info.data_type, data_type);
            assert!(pool.deallocate(&block_id).is_ok());
        }
    }

    #[test]
    fn test_memory_block_metadata() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        let block_id = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        let block_info = pool.get_block_info(&block_id).unwrap();
        
        assert_eq!(block_info.size_bytes, 1024);
        assert!(block_info.is_allocated);
        assert!(block_info.allocated_at.is_some());
        assert_eq!(block_info.data_type, MemoryDataType::QuantumState);
    }

    #[test]
    fn test_list_allocated_blocks() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config, AllocationStrategy::FirstFit).unwrap();
        
        let block1 = pool.allocate(1024, MemoryDataType::QuantumState).unwrap();
        let block2 = pool.allocate(2048, MemoryDataType::ComplexAmplitudes).unwrap();
        
        let allocated_blocks = pool.list_allocated_blocks();
        assert_eq!(allocated_blocks.len(), 2);
        
        let block_ids: Vec<String> = allocated_blocks.iter().map(|b| b.id.clone()).collect();
        assert!(block_ids.contains(&block1));
        assert!(block_ids.contains(&block2));
    }

    #[test]
    fn test_memory_pool_config_defaults() {
        let config = MemoryPoolConfig::default();
        
        assert_eq!(config.initial_size_mb, 64);
        assert_eq!(config.max_size_mb, 1024);
        assert_eq!(config.growth_factor, 1.5);
        assert!(config.enable_garbage_collection);
        assert_eq!(config.gc_threshold_percentage, 80.0);
        assert!(!config.enable_compression);
        assert!(config.enable_prefetching);
        assert!(!config.numa_aware);
    }

    #[test]
    fn test_memory_stats_defaults() {
        let stats = MemoryStats::default();
        
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.allocated_bytes, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.deallocation_count, 0);
        assert_eq!(stats.gc_runs, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert!(stats.last_gc_time.is_none());
    }

    #[test]
    fn test_cached_entry() {
        let data = vec![1, 2, 3, 4, 5];
        let entry = CachedEntry {
            data: data.clone(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_count: 1,
            size_bytes: data.len(),
        };
        
        assert_eq!(entry.data, data);
        assert_eq!(entry.size_bytes, 5);
        assert_eq!(entry.access_count, 1);
    }
}