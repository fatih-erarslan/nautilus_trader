//! Error types for the whale defense ML system

use thiserror::Error;

/// Main error type for the whale defense ML system
#[derive(Error, Debug)]
pub enum WhaleMLError {
    /// Model initialization error
    #[error("Model initialization failed: {0}")]
    ModelInit(String),
    
    /// Inference error
    #[error("Inference failed: {0}")]
    Inference(String),
    
    /// Data preprocessing error
    #[error("Data preprocessing failed: {0}")]
    Preprocessing(String),
    
    /// Invalid input dimensions
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: String, actual: String },
    
    /// Missing model weights
    #[error("Model weights not loaded")]
    MissingWeights,
    
    /// Performance target not met
    #[error("Performance target not met: {0}μs > 500μs")]
    PerformanceViolation(u64),
    
    /// Candle framework error
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, WhaleMLError>;