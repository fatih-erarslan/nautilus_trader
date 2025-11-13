//! GPU Compute Backend for HyperPhysics
//!
//! Provides cross-platform GPU acceleration using WGPU with fallback to CPU.
//! Supports compute shaders for massively parallel pBit lattice evolution.

pub mod backend;
pub mod kernels;
pub mod executor;
pub mod scheduler;
pub mod monitoring;
pub mod rng;

use hyperphysics_core::Result;

// Re-export types from backend module
pub use backend::{GPUBackend, BackendType, GPUCapabilities};
pub use executor::GPUExecutor;
pub use scheduler::GPUScheduler;
pub use monitoring::{PerformanceMonitor, OperationMetrics, ScopedTimer};
pub use rng::{GPURng, RNGState, RNGParams};

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
/// Tries backends in order: CUDA → WGPU → CPU fallback
pub async fn initialize_backend() -> Result<Box<dyn GPUBackend>> {
    // Try CUDA backend first (highest performance)
    #[cfg(feature = "cuda-backend")]
    {
        use backend::cuda_real::create_cuda_backend;
        if let Ok(Some(cuda_backend)) = create_cuda_backend() {
            tracing::info!("Using CUDA backend: {}", cuda_backend.capabilities().device_name);
            return Ok(Box::new(cuda_backend));
        }
    }

    // Try WGPU backend second
    #[cfg(feature = "wgpu-backend")]
    {
        match backend::wgpu::WGPUBackend::new().await {
            Ok(wgpu_backend) => {
                tracing::info!("Using WGPU backend: {}", wgpu_backend.capabilities().device_name);
                return Ok(Box::new(wgpu_backend));
            }
            Err(e) => {
                tracing::warn!("WGPU initialization failed: {:?}", e);
            }
        }
    }

    // Fall back to CPU
    tracing::warn!("No GPU backends available, using CPU fallback");
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

    #[tokio::test]
    async fn test_initialize_backend() {
        let backend = initialize_backend().await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_wgpu_backend_init() {
        if let Ok(backend) = backend::wgpu::WGPUBackend::new().await {
            assert_eq!(backend.capabilities().backend, BackendType::WGPU);
            assert!(backend.capabilities().supports_compute);
        }
    }
}
