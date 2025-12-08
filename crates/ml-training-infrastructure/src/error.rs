//! Error types for ML training infrastructure

use thiserror::Error;

/// Result type for training operations
pub type Result<T> = std::result::Result<T, TrainingError>;

/// Training infrastructure errors
#[derive(Error, Debug)]
pub enum TrainingError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Data loading error
    #[error("Data loading error: {0}")]
    DataLoading(String),
    
    /// Model creation error
    #[error("Model creation error: {0}")]
    ModelCreation(String),
    
    /// Training error
    #[error("Training error: {0}")]
    Training(String),
    
    /// Validation error
    #[error("Validation error: {0}")]
    Validation(String),
    
    /// Optimization error
    #[error("Optimization error: {0}")]
    Optimization(String),
    
    /// Persistence error
    #[error("Persistence error: {0}")]
    Persistence(String),
    
    /// Experiment tracking error
    #[error("Experiment tracking error: {0}")]
    ExperimentTracking(String),
    
    /// Deployment error
    #[error("Deployment error: {0}")]
    Deployment(String),
    
    /// GPU error
    #[error("GPU error: {0}")]
    GPU(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// Other error
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}