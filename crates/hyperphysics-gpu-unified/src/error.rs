//! Error types for GPU unified layer

use thiserror::Error;

/// GPU operation errors
#[derive(Error, Debug)]
pub enum GpuError {
    /// Failed to initialize GPU device
    #[error("GPU initialization failed: {0}")]
    InitializationFailed(String),

    /// No suitable GPU adapter found
    #[error("No GPU adapter found matching criteria: {0}")]
    NoAdapterFound(String),

    /// Buffer allocation failed
    #[error("Buffer allocation failed: {message} (requested: {requested_bytes} bytes)")]
    AllocationFailed {
        /// Error message
        message: String,
        /// Requested size in bytes
        requested_bytes: u64,
    },

    /// Shader compilation failed
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    /// Pipeline creation failed
    #[error("Pipeline creation failed: {0}")]
    PipelineCreationFailed(String),

    /// Buffer mapping failed
    #[error("Buffer mapping failed: {0}")]
    BufferMappingFailed(String),

    /// GPU out of memory
    #[error("GPU out of memory: {available_bytes} available, {requested_bytes} requested")]
    OutOfMemory {
        /// Available VRAM in bytes
        available_bytes: u64,
        /// Requested allocation in bytes
        requested_bytes: u64,
    },

    /// Workload routing failed
    #[error("Workload routing failed: {0}")]
    RoutingFailed(String),

    /// Synchronization error
    #[error("GPU synchronization error: {0}")]
    SyncError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Kernel not found
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Burn framework error
    #[cfg(feature = "burn")]
    #[error("Burn framework error: {0}")]
    BurnError(String),
}

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        GpuError::InitializationFailed(err.to_string())
    }
}
