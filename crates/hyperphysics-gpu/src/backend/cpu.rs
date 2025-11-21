//! CPU fallback backend (no GPU acceleration)

use super::{GPUBackend, GPUCapabilities, BackendType, GPUBuffer, BufferUsage, MemoryStats};
use hyperphysics_core::{Result, EngineError};

/// CPU-only backend (fallback when no GPU available)
pub struct CPUBackend {
    capabilities: GPUCapabilities,
}

impl CPUBackend {
    pub fn new() -> Self {
        Self {
            capabilities: GPUCapabilities {
                backend: BackendType::CPU,
                device_name: "CPU (no GPU)".to_string(),
                max_buffer_size: usize::MAX as u64,
                max_workgroup_size: 1,
                supports_compute: false,
            },
        }
    }
}

/// CPU buffer implementation (uses system memory)
struct CPUBuffer {
    data: Vec<u8>,
    usage: BufferUsage,
}

impl GPUBuffer for CPUBuffer {
    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GPUBackend for CPUBackend {
    fn capabilities(&self) -> &GPUCapabilities {
        &self.capabilities
    }

    fn execute_compute(&self, _shader: &str, _workgroups: [u32; 3]) -> Result<()> {
        Err(EngineError::Simulation {
            message: format!("CPU backend does not support compute shaders"),
        })
    }
    
    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Box<dyn GPUBuffer>> {
        Ok(Box::new(CPUBuffer {
            data: vec![0u8; size as usize],
            usage,
        }))
    }
    
    fn write_buffer(&self, buffer: &mut dyn GPUBuffer, data: &[u8]) -> Result<()> {
        // This is a simplified implementation - real version would need proper downcasting
        Err(EngineError::Simulation {
            message: format!("CPU backend buffer write operation not fully implemented for buffer of size {} and data of size {}", buffer.size(), data.len()),
        })
    }
    
    fn read_buffer(&self, buffer: &dyn GPUBuffer) -> Result<Vec<u8>> {
        // This is a simplified implementation - real version would need proper downcasting
        Err(EngineError::Simulation {
            message: format!("CPU backend buffer read operation not fully implemented for buffer of size {}", buffer.size()),
        })
    }
    
    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous by nature
        Ok(())
    }
    
    fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_memory: 0, // Would query system memory
            used_memory: 0,
            free_memory: 0,
            buffer_count: 0,
        }
    }
}

impl Default for CPUBackend {
    fn default() -> Self {
        Self::new()
    }
}
