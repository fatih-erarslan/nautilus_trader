//! Error types for quantum backend

use thiserror::Error;

pub type Result<T> = std::result::Result<T, QuantumBackendError>;

#[derive(Error, Debug)]
pub enum QuantumBackendError {
    #[error("PennyLane error: {0}")]
    PennyLane(String),
    
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    
    #[error("Python error: {0}")]
    Python(#[from] pyo3::PyErr),
    
    #[error("Circuit optimization error: {0}")]
    Optimization(String),
    
    #[error("Quantum solver error: {0}")]
    Solver(String),
    
    #[error("Invalid quantum state: {0}")]
    InvalidState(String),
    
    #[error("Device not available: {0}")]
    DeviceNotAvailable(String),
    
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
}

impl From<String> for QuantumBackendError {
    fn from(s: String) -> Self {
        QuantumBackendError::Generic(anyhow::anyhow!(s))
    }
}