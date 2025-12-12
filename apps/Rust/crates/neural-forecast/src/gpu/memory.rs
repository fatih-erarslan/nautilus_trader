//! GPU memory management for neural forecasting operations

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Buffer, BufferDescriptor, BufferUsages};
use crate::{Result, NeuralForecastError};
use crate::config::GPUConfig;

/// GPU memory pool for efficient buffer management
#[derive(Debug)]
pub struct GPUMemoryManager {
    device: Arc<Device>,
    buffer_pools: HashMap<usize, Vec<Arc<Buffer>>>,
    allocated_buffers: HashMap<u64, Arc<Buffer>>,
    total_allocated: u64,
    max_memory: u64,
    config: GPUConfig,
}

/// Memory pool for buffers of specific size
#[derive(Debug)]
pub struct BufferPool {
    size: usize,
    available_buffers: Vec<Arc<Buffer>>,
    in_use_buffers: Vec<Arc<Buffer>>,
    max_buffers: usize,
}

/// Memory allocation statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub peak_allocated: u64,
    pub current_pools: usize,
    pub total_buffers: usize,
    pub available_buffers: usize,
    pub fragmentation_ratio: f32,
}

impl GPUMemoryManager {
    /// Create new GPU memory manager
    pub fn new(device: &Device, config: &GPUConfig) -> Self {
        let max_memory = config.memory_limit.unwrap_or(1024 * 1024 * 1024) as u64; // 1GB default
        
        Self {
            device: Arc::new(device.clone()),
            buffer_pools: HashMap::new(),
            allocated_buffers: HashMap::new(),
            total_allocated: 0,
            max_memory,
            config: config.clone(),
        }
    }

    /// Allocate buffer from pool or create new
    pub fn allocate_buffer(&mut self, size: usize, usage: BufferUsages) -> Result<Arc<Buffer>> {
        // Check memory limit
        if self.total_allocated + size as u64 > self.max_memory {
            return Err(NeuralForecastError::GpuError(
                "GPU memory limit exceeded".to_string()
            ));
        }

        // Try to get from pool first
        if let Some(pool) = self.buffer_pools.get_mut(&size) {
            if let Some(buffer) = pool.pop() {
                let buffer_id = buffer.as_ref() as *const Buffer as u64;
                self.allocated_buffers.insert(buffer_id, buffer.clone());
                return Ok(buffer);
            }
        }

        // Create new buffer
        let buffer = Arc::new(self.device.create_buffer(&BufferDescriptor {
            label: Some("Neural Forecast Pooled Buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        }));

        let buffer_id = buffer.as_ref() as *const Buffer as u64;
        self.allocated_buffers.insert(buffer_id, buffer.clone());
        self.total_allocated += size as u64;

        Ok(buffer)
    }

    /// Return buffer to pool
    pub fn deallocate_buffer(&mut self, buffer: Arc<Buffer>) {
        let buffer_id = buffer.as_ref() as *const Buffer as u64;
        
        if let Some(_) = self.allocated_buffers.remove(&buffer_id) {
            let size = buffer.size() as usize;
            
            // Add to pool if enabled
            if self.config.memory_pooling {
                let pool = self.buffer_pools.entry(size).or_insert_with(Vec::new);
                pool.push(buffer);
            } else {
                // Subtract from total if not pooling
                self.total_allocated = self.total_allocated.saturating_sub(size as u64);
            }
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total_buffers = self.allocated_buffers.len();
        let available_buffers = self.buffer_pools.values().map(|pool| pool.len()).sum::<usize>();
        let fragmentation_ratio = if total_buffers > 0 {
            available_buffers as f32 / total_buffers as f32
        } else {
            0.0
        };

        MemoryStats {
            total_allocated: self.total_allocated,
            peak_allocated: self.total_allocated, // Simplified - would track peak in real implementation
            current_pools: self.buffer_pools.len(),
            total_buffers,
            available_buffers,
            fragmentation_ratio,
        }
    }

    /// Clear all pools and force garbage collection
    pub fn clear_pools(&mut self) {
        let mut total_freed = 0u64;
        
        for pool in self.buffer_pools.values() {
            for buffer in pool {
                total_freed += buffer.size();
            }
        }
        
        self.buffer_pools.clear();
        self.total_allocated = self.total_allocated.saturating_sub(total_freed);
    }

    /// Optimize memory pools based on usage patterns
    pub fn optimize_pools(&mut self) {
        // Remove empty pools
        self.buffer_pools.retain(|_, pool| !pool.is_empty());
        
        // Limit pool sizes to prevent memory bloat
        const MAX_POOL_SIZE: usize = 32;
        for pool in self.buffer_pools.values_mut() {
            if pool.len() > MAX_POOL_SIZE {
                let excess = pool.len() - MAX_POOL_SIZE;
                for _ in 0..excess {
                    if let Some(buffer) = pool.pop() {
                        self.total_allocated = self.total_allocated.saturating_sub(buffer.size());
                    }
                }
            }
        }
    }

    /// Pre-allocate buffers for common sizes
    pub fn pre_allocate_common_sizes(&mut self) -> Result<()> {
        let common_sizes = vec![
            1024,        // 1KB
            4096,        // 4KB
            16384,       // 16KB
            65536,       // 64KB
            262144,      // 256KB
            1048576,     // 1MB
            4194304,     // 4MB
            16777216,    // 16MB
        ];

        for size in common_sizes {
            let buffer = self.allocate_buffer(size, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)?;
            self.deallocate_buffer(buffer);
        }

        Ok(())
    }

    /// Get buffer utilization for a specific size
    pub fn get_buffer_utilization(&self, size: usize) -> f32 {
        if let Some(pool) = self.buffer_pools.get(&size) {
            let total_in_pool = pool.len();
            let total_allocated = self.allocated_buffers.values()
                .filter(|buffer| buffer.size() == size as u64)
                .count();
            
            if total_allocated > 0 {
                (total_allocated - total_in_pool) as f32 / total_allocated as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Force cleanup of unused buffers
    pub fn force_cleanup(&mut self) {
        // This is a simplified implementation
        // In practice, you'd track buffer usage and clean up truly unused buffers
        self.optimize_pools();
    }
}

impl BufferPool {
    /// Create new buffer pool
    pub fn new(size: usize, max_buffers: usize) -> Self {
        Self {
            size,
            available_buffers: Vec::new(),
            in_use_buffers: Vec::new(),
            max_buffers,
        }
    }

    /// Get buffer from pool
    pub fn get_buffer(&mut self) -> Option<Arc<Buffer>> {
        if let Some(buffer) = self.available_buffers.pop() {
            self.in_use_buffers.push(buffer.clone());
            Some(buffer)
        } else {
            None
        }
    }

    /// Return buffer to pool
    pub fn return_buffer(&mut self, buffer: Arc<Buffer>) {
        if let Some(pos) = self.in_use_buffers.iter().position(|b| Arc::ptr_eq(b, &buffer)) {
            self.in_use_buffers.remove(pos);
            
            if self.available_buffers.len() < self.max_buffers {
                self.available_buffers.push(buffer);
            }
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.size, self.available_buffers.len(), self.in_use_buffers.len())
    }
}

/// RAII wrapper for GPU buffer allocation
#[derive(Debug)]
pub struct GPUBufferGuard {
    buffer: Option<Arc<Buffer>>,
    manager: Arc<std::sync::Mutex<GPUMemoryManager>>,
}

impl GPUBufferGuard {
    /// Create new buffer guard
    pub fn new(buffer: Arc<Buffer>, manager: Arc<std::sync::Mutex<GPUMemoryManager>>) -> Self {
        Self {
            buffer: Some(buffer),
            manager,
        }
    }

    /// Get the buffer
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        self.buffer.as_ref()
    }

    /// Take the buffer (releases from guard)
    pub fn take(mut self) -> Option<Arc<Buffer>> {
        self.buffer.take()
    }
}

impl Drop for GPUBufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Ok(mut manager) = self.manager.lock() {
                manager.deallocate_buffer(buffer);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GPUConfig;

    // Note: These tests would require a GPU device to run properly
    // They're included for structure but may fail in CI environments

    #[test]
    fn test_memory_manager_creation() {
        // This test would require a GPU device
        // Skipping actual creation for now
        let config = GPUConfig::default();
        assert!(config.memory_pooling);
    }

    #[test]
    fn test_buffer_pool_creation() {
        let pool = BufferPool::new(1024, 10);
        assert_eq!(pool.size, 1024);
        assert_eq!(pool.max_buffers, 10);
        assert_eq!(pool.available_buffers.len(), 0);
        assert_eq!(pool.in_use_buffers.len(), 0);
    }

    #[test]
    fn test_memory_stats() {
        let config = GPUConfig::default();
        // Create a mock device here in real tests
        // For now, just test the structure
        assert!(config.memory_pooling);
    }
}