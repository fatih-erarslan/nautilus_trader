//! # Cortical Bus Error Types
//!
//! Unified error handling integrating errors from:
//! - Spike routing (ring buffer overflow/underflow)
//! - pBit dynamics (via hyperphysics-pbit)
//! - Similarity search (via hyperphysics-similarity)
//! - GPU operations (via hyperphysics-gpu-unified)

use thiserror::Error;

/// Unified cortical bus error type.
#[derive(Debug, Error)]
pub enum CorticalError {
    // ========================================================================
    // Spike Routing Errors
    // ========================================================================
    
    /// Spike queue is full (backpressure).
    #[error("Spike queue {queue_id} is full - {pending} spikes pending")]
    QueueFull {
        /// Queue identifier
        queue_id: usize,
        /// Number of pending spikes
        pending: usize,
    },

    /// Spike queue is empty.
    #[error("Spike queue {0} is empty")]
    QueueEmpty(usize),

    /// Invalid spike data.
    #[error("Invalid spike: {0}")]
    InvalidSpike(String),

    // ========================================================================
    // pBit Errors (forwarded from hyperphysics-pbit)
    // ========================================================================
    
    /// pBit dynamics error.
    #[cfg(feature = "pbit")]
    #[error("pBit error: {0}")]
    PBitError(#[from] hyperphysics_pbit::PBitError),

    /// pBit feature not enabled.
    #[cfg(not(feature = "pbit"))]
    #[error("pBit feature not enabled - rebuild with --features pbit")]
    PBitNotEnabled,

    // ========================================================================
    // Similarity Search Errors (forwarded from hyperphysics-similarity)
    // ========================================================================
    
    /// Similarity search error.
    #[cfg(feature = "similarity")]
    #[error("Similarity search error: {0}")]
    SimilarityError(#[from] hyperphysics_similarity::HybridError),

    /// Similarity feature not enabled.
    #[cfg(not(feature = "similarity"))]
    #[error("Similarity feature not enabled - rebuild with --features similarity")]
    SimilarityNotEnabled,

    // ========================================================================
    // GPU Errors (forwarded from hyperphysics-gpu-unified)
    // ========================================================================
    
    /// GPU error.
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    GpuError(#[from] hyperphysics_gpu_unified::GpuError),

    /// GPU feature not enabled.
    #[cfg(not(feature = "gpu"))]
    #[error("GPU feature not enabled - rebuild with --features gpu")]
    GpuNotEnabled,

    // ========================================================================
    // General Errors
    // ========================================================================
    
    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Capacity exceeded.
    #[error("Capacity exceeded: max {max}, requested {requested}")]
    CapacityExceeded {
        /// Maximum capacity
        max: usize,
        /// Requested capacity
        requested: usize,
    },

    /// Resource not initialized.
    #[error("Resource not initialized: {0}")]
    NotInitialized(&'static str),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for cortical bus operations.
pub type Result<T> = std::result::Result<T, CorticalError>;

impl CorticalError {
    /// Check if this error is recoverable (can retry).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            CorticalError::QueueFull { .. } | CorticalError::QueueEmpty(_)
        )
    }

    /// Check if this error requires feature enablement.
    pub fn requires_feature(&self) -> Option<&'static str> {
        match self {
            #[cfg(not(feature = "pbit"))]
            CorticalError::PBitNotEnabled => Some("pbit"),
            #[cfg(not(feature = "similarity"))]
            CorticalError::SimilarityNotEnabled => Some("similarity"),
            #[cfg(not(feature = "gpu"))]
            CorticalError::GpuNotEnabled => Some("gpu"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CorticalError::QueueFull { queue_id: 5, pending: 100 };
        assert!(err.to_string().contains("queue 5"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = CorticalError::DimensionMismatch { expected: 128, actual: 64 };
        assert!(err.to_string().contains("128"));
        assert!(!err.is_recoverable());
    }
}
