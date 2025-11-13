//! GPU backend implementations

use hyperphysics_core::Result;

pub mod cpu;

#[cfg(feature = "wgpu-backend")]
pub mod wgpu;

#[cfg(feature = "cuda-backend")]
pub mod cuda;

#[cfg(feature = "cuda-backend")]
pub mod cuda_real;

#[cfg(feature = "metal-backend")]
pub mod metal;

#[cfg(feature = "rocm-backend")]
pub mod rocm;

#[cfg(feature = "webgpu-backend")]
pub mod webgpu;

#[cfg(feature = "vulkan-backend")]
pub mod vulkan;

/// GPU backend type identifier
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// WGPU cross-platform backend (Vulkan/Metal/DX12)
    WGPU,
    /// NVIDIA CUDA
    CUDA,
    /// Apple Metal
    Metal,
    /// AMD ROCm
    ROCm,
    /// WebGPU (browser compatibility)
    WebGPU,
    /// Vulkan compute (Linux fallback)
    Vulkan,
    /// CPU fallback (no GPU)
    CPU,
}

/// GPU device capabilities
#[derive(Debug, Clone)]
pub struct GPUCapabilities {
    /// Backend type in use
    pub backend: BackendType,
    /// Device name/identifier
    pub device_name: String,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum workgroup size for compute shaders
    pub max_workgroup_size: u32,
    /// Whether compute shaders are supported
    pub supports_compute: bool,
}

/// GPU backend trait for compute operations
pub trait GPUBackend: Send + Sync {
    /// Get device capabilities
    fn capabilities(&self) -> &GPUCapabilities;

    /// Get device name
    fn device_name(&self) -> &str {
        &self.capabilities().device_name
    }

    /// Check if compute shaders are supported
    fn supports_compute(&self) -> bool {
        self.capabilities().supports_compute
    }

    /// Execute compute shader
    ///
    /// # Arguments
    /// * `shader` - WGSL shader source code
    /// * `workgroups` - Workgroup dimensions [x, y, z]
    fn execute_compute(&self, shader: &str, workgroups: [u32; 3]) -> Result<()>;
    
    /// Create buffer for GPU computation
    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Box<dyn GPUBuffer>>;
    
    /// Copy data to GPU buffer
    fn write_buffer(&self, buffer: &mut dyn GPUBuffer, data: &[u8]) -> Result<()>;
    
    /// Read data from GPU buffer
    fn read_buffer(&self, buffer: &dyn GPUBuffer) -> Result<Vec<u8>>;
    
    /// Synchronize GPU operations
    fn synchronize(&self) -> Result<()>;
    
    /// Get memory usage statistics
    fn memory_stats(&self) -> MemoryStats;
}

/// GPU buffer usage flags
#[derive(Debug, Clone, Copy)]
pub enum BufferUsage {
    Storage,
    Uniform,
    Vertex,
    Index,
    CopySrc,
    CopyDst,
}

/// GPU buffer trait
pub trait GPUBuffer: Send + Sync {
    fn size(&self) -> u64;
    fn usage(&self) -> BufferUsage;
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory: u64,
    pub used_memory: u64,
    pub free_memory: u64,
    pub buffer_count: u32,
}
