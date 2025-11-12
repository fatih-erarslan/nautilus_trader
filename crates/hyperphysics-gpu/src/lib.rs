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
    // TODO: Try WGPU backend first
    // For now, return CPU fallback
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
