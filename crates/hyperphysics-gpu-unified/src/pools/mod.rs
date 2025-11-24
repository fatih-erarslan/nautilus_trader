//! Compute pools for domain-specific GPU workloads

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BufferUsages};
use parking_lot::RwLock;

use crate::{GpuError, GpuResult};

/// Pool type for domain-specific compute workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolType {
    /// Physics simulations (pBit, SPH, rigid body)
    Physics,
    /// Financial computations (VaR, Monte Carlo, Greeks)
    Finance,
    /// Neural network operations (attention, backprop)
    Neural,
    /// General-purpose compute
    General,
}

impl PoolType {
    /// Get optimal workgroup size for this pool type
    pub fn optimal_workgroup_size(&self) -> u32 {
        match self {
            // Physics: 256 threads (4 wavefronts) for memory coalescing
            PoolType::Physics => 256,
            // Finance: 128 threads (2 wavefronts) for better occupancy with MC
            PoolType::Finance => 128,
            // Neural: 256 threads aligned with matrix tile sizes
            PoolType::Neural => 256,
            // General: balanced default
            PoolType::General => 256,
        }
    }

    /// Get recommended buffer alignment for this pool type
    pub fn buffer_alignment(&self) -> u64 {
        match self {
            // Physics: 256-byte alignment for vec4 arrays
            PoolType::Physics => 256,
            // Finance: 128-byte for f64 pairs
            PoolType::Finance => 128,
            // Neural: 256-byte for matrix tiles
            PoolType::Neural => 256,
            PoolType::General => 256,
        }
    }
}

/// Pooled buffer for reuse
#[derive(Debug)]
pub struct PooledBuffer {
    /// Underlying wgpu buffer
    pub buffer: Buffer,
    /// Buffer size in bytes
    pub size: u64,
    /// Whether buffer is currently in use
    pub in_use: bool,
}

/// Compute pool for managing GPU resources
pub struct ComputePool {
    /// Pool name
    name: String,
    /// Pool type
    pool_type: PoolType,
    /// Associated device
    device: Arc<Device>,
    /// Associated queue
    queue: Arc<Queue>,
    /// Buffer pool for reuse
    buffers: RwLock<Vec<PooledBuffer>>,
    /// Maximum pool size in bytes
    max_size_bytes: u64,
    /// Current allocated size
    allocated_bytes: RwLock<u64>,
}

impl ComputePool {
    /// Create a new compute pool
    pub fn new(
        name: String,
        pool_type: PoolType,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Self {
        Self {
            name,
            pool_type,
            device,
            queue,
            buffers: RwLock::new(Vec::new()),
            max_size_bytes: 512 * 1024 * 1024, // 512MB default
            allocated_bytes: RwLock::new(0),
        }
    }

    /// Set maximum pool size
    pub fn with_max_size(mut self, max_bytes: u64) -> Self {
        self.max_size_bytes = max_bytes;
        self
    }

    /// Get pool name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get pool type
    pub fn pool_type(&self) -> PoolType {
        self.pool_type
    }

    /// Acquire a buffer from the pool (or create new)
    pub fn acquire_buffer(&self, size: u64, usage: BufferUsages) -> GpuResult<Arc<Buffer>> {
        let aligned_size = self.align_size(size);

        // Try to find a suitable buffer in the pool
        {
            let mut buffers = self.buffers.write();
            for pooled in buffers.iter_mut() {
                if !pooled.in_use && pooled.size >= aligned_size {
                    pooled.in_use = true;
                    return Ok(Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{}-buffer", self.name)),
                        size: aligned_size,
                        usage,
                        mapped_at_creation: false,
                    })));
                }
            }
        }

        // Check if we can allocate more
        let current = *self.allocated_bytes.read();
        if current + aligned_size > self.max_size_bytes {
            return Err(GpuError::OutOfMemory {
                available_bytes: self.max_size_bytes - current,
                requested_bytes: aligned_size,
            });
        }

        // Create new buffer
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{}-buffer", self.name)),
            size: aligned_size,
            usage,
            mapped_at_creation: false,
        });

        // Track allocation
        *self.allocated_bytes.write() += aligned_size;

        // Add to pool for potential reuse
        self.buffers.write().push(PooledBuffer {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}-pooled", self.name)),
                size: aligned_size,
                usage,
                mapped_at_creation: false,
            }),
            size: aligned_size,
            in_use: true,
        });

        Ok(Arc::new(buffer))
    }

    /// Release a buffer back to the pool
    pub fn release_buffer(&self, size: u64) {
        let aligned_size = self.align_size(size);
        let mut buffers = self.buffers.write();
        for pooled in buffers.iter_mut() {
            if pooled.in_use && pooled.size == aligned_size {
                pooled.in_use = false;
                return;
            }
        }
    }

    /// Align size to pool's buffer alignment
    fn align_size(&self, size: u64) -> u64 {
        let alignment = self.pool_type.buffer_alignment();
        (size + alignment - 1) / alignment * alignment
    }

    /// Get current allocation statistics
    pub fn stats(&self) -> PoolStats {
        let buffers = self.buffers.read();
        let total_buffers = buffers.len();
        let in_use = buffers.iter().filter(|b| b.in_use).count();

        PoolStats {
            name: self.name.clone(),
            pool_type: self.pool_type,
            total_buffers,
            in_use_buffers: in_use,
            allocated_bytes: *self.allocated_bytes.read(),
            max_bytes: self.max_size_bytes,
        }
    }

    /// Clear unused buffers from the pool
    pub fn shrink(&self) {
        let mut buffers = self.buffers.write();
        let mut allocated = self.allocated_bytes.write();

        buffers.retain(|b| {
            if !b.in_use {
                *allocated = allocated.saturating_sub(b.size);
                false
            } else {
                true
            }
        });
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Pool name
    pub name: String,
    /// Pool type
    pub pool_type: PoolType,
    /// Total buffers in pool
    pub total_buffers: usize,
    /// Buffers currently in use
    pub in_use_buffers: usize,
    /// Total allocated bytes
    pub allocated_bytes: u64,
    /// Maximum allowed bytes
    pub max_bytes: u64,
}

impl PoolStats {
    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_buffers == 0 {
            0.0
        } else {
            self.in_use_buffers as f32 / self.total_buffers as f32 * 100.0
        }
    }

    /// Get memory pressure percentage
    pub fn memory_pressure(&self) -> f32 {
        if self.max_bytes == 0 {
            100.0
        } else {
            self.allocated_bytes as f32 / self.max_bytes as f32 * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_type_workgroup_size() {
        assert_eq!(PoolType::Physics.optimal_workgroup_size(), 256);
        assert_eq!(PoolType::Finance.optimal_workgroup_size(), 128);
        assert_eq!(PoolType::Neural.optimal_workgroup_size(), 256);
    }

    #[test]
    fn test_pool_stats_utilization() {
        let stats = PoolStats {
            name: "test".to_string(),
            pool_type: PoolType::General,
            total_buffers: 10,
            in_use_buffers: 5,
            allocated_bytes: 1024,
            max_bytes: 2048,
        };

        assert_eq!(stats.utilization(), 50.0);
        assert_eq!(stats.memory_pressure(), 50.0);
    }
}
