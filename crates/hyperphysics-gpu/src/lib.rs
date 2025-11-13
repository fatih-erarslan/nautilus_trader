//! GPU Compute Backend for HyperPhysics
//!
//! Provides cross-platform GPU acceleration using WGPU with fallback to CPU.
//! Supports compute shaders for massively parallel pBit lattice evolution.

pub mod backend;

use hyperphysics_core::Result;

// Re-export types from backend module
pub use backend::{GPUBackend, BackendType, GPUCapabilities};

/// Initialize GPU backend with automatic device selection
///
/// Tries backends in order: WGPU → CPU fallback
pub fn initialize_backend() -> Result<Box<dyn GPUBackend>> {
    #[cfg(feature = "wgpu-backend")]
    {
        // Try WGPU backend first
        match backend::wgpu::WGPUBackend::new_blocking() {
            Ok(wgpu_backend) => {
                eprintln!("✓ GPU acceleration enabled: {}", wgpu_backend.device_name());
                return Ok(Box::new(wgpu_backend));
            }
            Err(e) => {
                eprintln!("⚠ WGPU backend unavailable: {}", e);
                eprintln!("  Falling back to CPU...");
            }
        }
    }

    // Fallback to CPU backend
    eprintln!("  Using CPU backend (no GPU acceleration)");
    Ok(Box::new(backend::cpu::CPUBackend::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        let backend = backend::cpu::CPUBackend::new();
        assert_eq!(backend.capabilities().backend, BackendType::CPU);
        assert!(!backend.supports_compute());
    }

    #[test]
    fn test_initialize_backend() {
        let backend = initialize_backend();
        assert!(backend.is_ok());
    }
}
