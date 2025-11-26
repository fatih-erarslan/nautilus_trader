//! Error types for the HyperPhysics-Nautilus integration.

use thiserror::Error;

/// Integration error type
#[derive(Error, Debug)]
pub enum IntegrationError {
    /// Type conversion error
    #[error("Type conversion error: {0}")]
    Conversion(String),

    /// Data validation error
    #[error("Data validation error: {0}")]
    Validation(String),

    /// Pipeline execution error
    #[error("Pipeline execution error: {0}")]
    Pipeline(String),

    /// Order submission error
    #[error("Order submission error: {0}")]
    OrderSubmission(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Consensus not reached
    #[error("Consensus not reached: confidence {confidence:.2} below threshold {threshold:.2}")]
    ConsensusNotReached {
        /// Actual confidence
        confidence: f64,
        /// Required threshold
        threshold: f64,
    },

    /// Physics simulation error
    #[error("Physics simulation error: {0}")]
    PhysicsSimulation(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    Optimization(String),

    /// Message bus error
    #[error("Message bus error: {0}")]
    MessageBus(String),

    /// Venue adapter error
    #[error("Venue adapter error: {0}")]
    VenueAdapter(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Any other error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for integration operations
pub type Result<T> = std::result::Result<T, IntegrationError>;

impl From<hyperphysics_hft_ecosystem::EcosystemError> for IntegrationError {
    fn from(err: hyperphysics_hft_ecosystem::EcosystemError) -> Self {
        IntegrationError::Pipeline(err.to_string())
    }
}
