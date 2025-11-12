//! GPU backend implementations

use hyperphysics_core::Result;

pub mod cpu;

#[cfg(feature = "wgpu-backend")]
pub mod wgpu;

/// GPU backend type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// WGPU cross-platform backend (Vulkan/Metal/DX12)
    WGPU,
    /// NVIDIA CUDA (future support)
    CUDA,
    /// Apple Metal (future support)
    Metal,
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
}
