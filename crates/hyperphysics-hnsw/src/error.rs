//! # Error Types for HyperPhysics HNSW
//!
//! Comprehensive error handling for the Processing layer.
//!
//! Errors are categorized by their relationship to the triangular architecture:
//!
//! - **Processing Errors**: Failures in the hot query path
//! - **Acquisition Errors**: Failures receiving from the Acquisition layer
//! - **Evolution Errors**: Failures applying Evolution layer optimizations
//! - **Constraint Errors**: Violations of Processing layer constraints

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Result type for HNSW operations.
pub type Result<T> = std::result::Result<T, HnswError>;

/// Errors that can occur in the HyperPhysics HNSW module.
#[derive(Error, Debug)]
pub enum HnswError {
    // ========================================================================
    // Processing Layer Errors (Hot Path)
    // ========================================================================
    
    /// Vector dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensionality (set on first insert or in config).
        expected: usize,
        /// Actual dimensionality of the provided vector.
        actual: usize,
    },
    
    /// Query returned no results (index may be empty).
    #[error("No results found for query")]
    NoResults,
    
    /// Index capacity exceeded.
    #[error("Index capacity exceeded: max {max}, attempted {attempted}")]
    CapacityExceeded {
        /// Maximum allowed capacity.
        max: usize,
        /// Attempted size.
        attempted: usize,
    },
    
    /// Invalid vector data (NaN, Inf, or out of bounds for hyperbolic space).
    #[error("Invalid vector: {reason}")]
    InvalidVector {
        /// Description of the invalidity.
        reason: String,
    },
    
    // ========================================================================
    // Acquisition Bridge Errors
    // ========================================================================
    
    /// Batch insert from Acquisition layer failed.
    #[error("Acquisition batch insert failed: {reason}")]
    AcquisitionInsertFailed {
        /// Reason for failure.
        reason: String,
        /// Number of patterns that were successfully inserted before failure.
        inserted_count: usize,
    },
    
    /// Pattern promotion rejected due to Processing constraints.
    #[error("Pattern rejected: {reason}")]
    PatternRejected {
        /// Reason for rejection.
        reason: String,
    },
    
    // ========================================================================
    // Evolution Bridge Errors
    // ========================================================================
    
    /// Parameter update from Evolution layer failed.
    #[error("Evolution parameter update failed: {reason}")]
    EvolutionUpdateFailed {
        /// Reason for failure.
        reason: String,
    },
    
    /// Requested parameter change requires index rebuild.
    #[error("Parameter change requires index rebuild: {parameter}")]
    RebuildRequired {
        /// Parameter that requires rebuild (e.g., "M", "dimensions").
        parameter: String,
    },
    
    // ========================================================================
    // Constraint Violations
    // ========================================================================
    
    /// Processing layer constraint violated.
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    /// Query exceeded latency budget.
    #[error("Latency budget exceeded: {actual_ns}ns > {budget_ns}ns")]
    LatencyExceeded {
        /// Actual query latency.
        actual_ns: u64,
        /// Configured latency budget.
        budget_ns: u64,
    },
    
    // ========================================================================
    // I/O and Persistence Errors
    // ========================================================================
    
    /// Failed to read/write index file.
    #[error("I/O error for {path}: {source}")]
    IoError {
        /// Path to the file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    
    /// Failed to memory-map index file.
    #[error("Memory mapping failed for {path}: {reason}")]
    MmapError {
        /// Path to the file.
        path: PathBuf,
        /// Reason for failure.
        reason: String,
    },
    
    /// Index file is corrupted.
    #[error("Corrupted index file: {path}")]
    CorruptedIndex {
        /// Path to the corrupted file.
        path: PathBuf,
    },
    
    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    // ========================================================================
    // Configuration Errors
    // ========================================================================

    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    Config(#[from] crate::config::ConfigError),

    /// Simple configuration error message.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Index not initialized.
    #[error("Index not initialized: call `insert` first or load from disk")]
    NotInitialized,

    // ========================================================================
    // Concurrency Errors
    // ========================================================================

    /// Lock acquisition failed (should be rare with proper design).
    #[error("Failed to acquire lock: {reason}")]
    LockError {
        /// Reason for failure.
        reason: String,
    },

    /// Concurrent modification detected.
    #[error("Concurrent modification detected")]
    ConcurrentModification,

    // ========================================================================
    // USearch Integration Errors
    // ========================================================================

    /// USearch library error.
    #[error("USearch error: {0}")]
    USearchError(String),
}

impl HnswError {
    /// Create an invalid vector error for NaN values.
    pub fn nan_vector(index: usize) -> Self {
        Self::InvalidVector {
            reason: format!("NaN value at index {}", index),
        }
    }
    
    /// Create an invalid vector error for Inf values.
    pub fn inf_vector(index: usize) -> Self {
        Self::InvalidVector {
            reason: format!("Infinite value at index {}", index),
        }
    }
    
    /// Create an invalid vector error for out-of-bounds hyperbolic points.
    pub fn hyperbolic_boundary_violation(norm: f32) -> Self {
        Self::InvalidVector {
            reason: format!(
                "Vector norm {} >= 1.0, outside Poincaré ball",
                norm
            ),
        }
    }
    
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NoResults
                | Self::LatencyExceeded { .. }
                | Self::PatternRejected { .. }
        )
    }
    
    /// Check if this error requires operator intervention.
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::CorruptedIndex { .. }
                | Self::CapacityExceeded { .. }
                | Self::ConstraintViolation(_)
        )
    }
}

// ============================================================================
// From implementations for common error types
// ============================================================================

impl From<io::Error> for HnswError {
    fn from(err: io::Error) -> Self {
        Self::IoError {
            path: PathBuf::from("<unknown>"),
            source: err,
        }
    }
}

impl From<bincode::Error> for HnswError {
    fn from(err: bincode::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dimension_mismatch_display() {
        let err = HnswError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));
    }
    
    #[test]
    fn test_is_recoverable() {
        assert!(HnswError::NoResults.is_recoverable());
        assert!(HnswError::LatencyExceeded {
            actual_ns: 2000,
            budget_ns: 1000,
        }.is_recoverable());
        
        assert!(!HnswError::CorruptedIndex {
            path: PathBuf::from("/tmp/test"),
        }.is_recoverable());
    }
    
    #[test]
    fn test_is_critical() {
        assert!(HnswError::CorruptedIndex {
            path: PathBuf::from("/tmp/test"),
        }.is_critical());
        
        assert!(!HnswError::NoResults.is_critical());
    }
}
