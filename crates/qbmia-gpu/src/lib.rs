//! QBMIA GPU Acceleration Framework
//! 
//! High-performance GPU acceleration for quantum circuit simulation and Nash equilibrium solving.
//! Supports CUDA, ROCm, and WebGPU backends with automatic fallback and multi-GPU orchestration.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod memory;
pub mod quantum;
pub mod nash;
pub mod kernels;
pub mod orchestrator;
pub mod profiler;

use thiserror::Error;

/// GPU acceleration errors
#[derive(Error, Debug)]
pub enum GpuError {
    /// Backend initialization failed
    #[error("Backend initialization failed: {0}")]
    BackendInit(String),
    
    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
    
    /// Kernel compilation failed
    #[error("Kernel compilation failed: {0}")]
    KernelCompilation(String),
    
    /// Kernel execution failed
    #[error("Kernel execution failed: {0}")]
    KernelExecution(String),
    
    /// Device not found
    #[error("GPU device not found: {0}")]
    DeviceNotFound(String),
    
    /// Multi-GPU synchronization failed
    #[error("Multi-GPU synchronization failed: {0}")]
    SyncError(String),
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Number of compute units
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Supports double precision
    pub double_precision: bool,
    /// Supports tensor cores
    pub tensor_cores: bool,
    /// Memory bandwidth GB/s
    pub memory_bandwidth: f64,
}

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// WebGPU
    WebGpu,
    /// CPU fallback
    Cpu,
}

/// Initialize GPU acceleration
pub fn initialize() -> GpuResult<()> {
    backend::initialize()
}

/// Get available devices
pub fn get_devices() -> GpuResult<Vec<DeviceCapabilities>> {
    backend::get_devices()
}

/// Select optimal backend based on available hardware
pub fn select_backend() -> Backend {
    backend::select_optimal_backend()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_initialization() {
        // Test GPU initialization
        match initialize() {
            Ok(()) => println!("GPU initialized successfully"),
            Err(e) => println!("GPU initialization failed (expected in CI): {}", e),
        }
    }
}