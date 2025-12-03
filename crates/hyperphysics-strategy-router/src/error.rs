//! Error types for strategy router operations

use thiserror::Error;

/// Result type for router operations
pub type Result<T> = std::result::Result<T, RouterError>;

/// Errors that can occur in strategy routing
#[derive(Debug, Error, Clone)]
pub enum RouterError {
    /// No experts available
    #[error("No experts available for routing")]
    NoExperts,

    /// Expert not found
    #[error("Expert not found: {id}")]
    ExpertNotFound {
        /// Expert identifier
        id: usize,
    },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Temperature must be positive
    #[error("Temperature must be positive, got {0}")]
    InvalidTemperature(f64),

    /// Top-K exceeds number of experts
    #[error("Top-K ({k}) exceeds number of experts ({n})")]
    TopKExceedsExperts {
        /// Requested top-K
        k: usize,
        /// Available experts
        n: usize,
    },

    /// Capacity exceeded
    #[error("Expert {expert_id} capacity exceeded: {load}/{capacity}")]
    CapacityExceeded {
        /// Expert identifier
        expert_id: usize,
        /// Current load
        load: f64,
        /// Maximum capacity
        capacity: f64,
    },

    /// Computation error
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// pBit error
    #[error("pBit error: {0}")]
    PBitError(String),
}

#[cfg(feature = "pbit")]
impl From<hyperphysics_pbit::PBitError> for RouterError {
    fn from(err: hyperphysics_pbit::PBitError) -> Self {
        RouterError::PBitError(err.to_string())
    }
}

impl From<hyperphysics_lorentz::LorentzError> for RouterError {
    fn from(err: hyperphysics_lorentz::LorentzError) -> Self {
        RouterError::ComputationError(err.to_string())
    }
}
