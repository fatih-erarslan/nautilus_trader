//! Error types for quantum device management

use thiserror::Error;

/// Quantum device management errors
#[derive(Error, Debug)]
pub enum QuantumDeviceError {
    #[error("Device not found: {device_id}")]
    DeviceNotFound { device_id: String },
    
    #[error("Device unavailable: {device_id}")]
    DeviceUnavailable { device_id: String },
    
    #[error("Task execution failed: {reason}")]
    TaskExecutionFailed { reason: String },
    
    #[error("No suitable device found for task: {task_id}")]
    NoSuitableDevice { task_id: String },
    
    #[error("Task timeout: {task_id}")]
    TaskTimeout { task_id: String },
    
    #[error("Device registration failed: {reason}")]
    DeviceRegistrationFailed { reason: String },
    
    #[error("Nash solver error: {reason}")]
    NashSolverError { reason: String },
    
    #[error("Quantum circuit error: {reason}")]
    QuantumCircuitError { reason: String },
    
    #[error("Configuration error: {reason}")]
    ConfigurationError { reason: String },
    
    #[error("Monitoring error: {reason}")]
    MonitoringError { reason: String },
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
}

/// Result type for quantum device operations
pub type QuantumDeviceResult<T> = Result<T, QuantumDeviceError>;