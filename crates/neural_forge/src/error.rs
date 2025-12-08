//! Error handling for Neural Forge

use thiserror::Error;

pub type Result<T> = std::result::Result<T, NeuralForgeError>;

#[derive(Error, Debug)]
pub enum NeuralForgeError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    #[error("Model error: {message}")]
    Model { message: String },
    
    #[error("Training error: {message}")]
    Training { message: String },
    
    #[error("Data error: {message}")]
    Data { message: String },
    
    #[error("Calibration error: {message}")]
    Calibration { message: String },
    
    #[error("Backend error: {message}")]
    Backend { message: String },
    
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    #[error("Distributed training error: {message}")]
    Distributed { message: String },
    
    #[error("CUDA error: {message}")]
    Cuda { message: String },
    
    #[error("Custom error: {message}")]
    Custom { message: String },
}

impl NeuralForgeError {
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config { message: message.into() }
    }
    
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model { message: message.into() }
    }
    
    pub fn training(message: impl Into<String>) -> Self {
        Self::Training { message: message.into() }
    }
    
    pub fn data(message: impl Into<String>) -> Self {
        Self::Data { message: message.into() }
    }
    
    pub fn calibration(message: impl Into<String>) -> Self {
        Self::Calibration { message: message.into() }
    }
    
    pub fn backend(message: impl Into<String>) -> Self {
        Self::Backend { message: message.into() }
    }
    
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation { message: message.into() }
    }
    
    pub fn distributed(message: impl Into<String>) -> Self {
        Self::Distributed { message: message.into() }
    }
    
    pub fn cuda(message: impl Into<String>) -> Self {
        Self::Cuda { message: message.into() }
    }
    
    pub fn custom(message: impl Into<String>) -> Self {
        Self::Custom { message: message.into() }
    }
}