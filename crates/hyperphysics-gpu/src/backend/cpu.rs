//! CPU fallback backend (no GPU acceleration)

use crate::{GPUBackend, GPUCapabilities, BackendType};
use hyperphysics_core::Result;

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

impl GPUBackend for CPUBackend {
    fn capabilities(&self) -> &GPUCapabilities {
        &self.capabilities
    }

    fn execute_compute(&self, _shader: &str, _workgroups: [u32; 3]) -> Result<()> {
        Err(hyperphysics_core::EngineError::Configuration {
            message: "CPU backend does not support compute shaders".to_string(),
        })
    }
}

impl Default for CPUBackend {
    fn default() -> Self {
        Self::new()
    }
}
