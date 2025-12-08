//! Error types for PADS connector

use thiserror::Error;

/// PADS error type
#[derive(Error, Debug)]
pub enum PadsError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Scale management error: {0}")]
    ScaleManagement(String),
    
    #[error("Decision routing error: {0}")]
    DecisionRouting(String),
    
    #[error("Communication error: {0}")]
    Communication(String),
    
    #[error("Resilience error: {0}")]
    Resilience(String),
    
    #[error("Integration error: {0}")]
    Integration(String),
    
    #[error("Monitoring error: {0}")]
    Monitoring(String),
    
    #[error("Scale transition error: {0}")]
    ScaleTransition(String),
    
    #[error("Adaptive capacity error: {0}")]
    AdaptiveCapacity(String),
    
    #[error("Cross-scale effect error: {0}")]
    CrossScaleEffect(String),
    
    #[error("Panarchy cycle error: {0}")]
    PanarchyCycle(String),
    
    #[error("Resource allocation error: {0}")]
    ResourceAllocation(String),
    
    #[error("Performance error: {0}")]
    Performance(String),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Channel send error")]
    ChannelSend,
    
    #[error("Channel receive error")]
    ChannelReceive,
    
    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Result type for PADS operations
pub type Result<T> = std::result::Result<T, PadsError>;

impl PadsError {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    
    /// Create a scale management error
    pub fn scale(msg: impl Into<String>) -> Self {
        Self::ScaleManagement(msg.into())
    }
    
    /// Create a decision routing error
    pub fn routing(msg: impl Into<String>) -> Self {
        Self::DecisionRouting(msg.into())
    }
    
    /// Create a communication error
    pub fn comm(msg: impl Into<String>) -> Self {
        Self::Communication(msg.into())
    }
    
    /// Create a resilience error
    pub fn resilience(msg: impl Into<String>) -> Self {
        Self::Resilience(msg.into())
    }
    
    /// Create an integration error
    pub fn integration(msg: impl Into<String>) -> Self {
        Self::Integration(msg.into())
    }
    
    /// Create a monitoring error
    pub fn monitor(msg: impl Into<String>) -> Self {
        Self::Monitoring(msg.into())
    }
    
    /// Create a scale transition error
    pub fn transition(msg: impl Into<String>) -> Self {
        Self::ScaleTransition(msg.into())
    }
    
    /// Create an adaptive capacity error
    pub fn capacity(msg: impl Into<String>) -> Self {
        Self::AdaptiveCapacity(msg.into())
    }
    
    /// Create a cross-scale effect error
    pub fn cross_scale(msg: impl Into<String>) -> Self {
        Self::CrossScaleEffect(msg.into())
    }
    
    /// Create a panarchy cycle error
    pub fn cycle(msg: impl Into<String>) -> Self {
        Self::PanarchyCycle(msg.into())
    }
    
    /// Create a resource allocation error
    pub fn resource(msg: impl Into<String>) -> Self {
        Self::ResourceAllocation(msg.into())
    }
    
    /// Create a performance error
    pub fn perf(msg: impl Into<String>) -> Self {
        Self::Performance(msg.into())
    }
    
    /// Create a timeout error
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }
}