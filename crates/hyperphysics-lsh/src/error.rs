//! # Error Types for HyperPhysics LSH
//!
//! Error handling for the Acquisition layer.

use thiserror::Error;

/// Result type for LSH operations.
pub type Result<T> = std::result::Result<T, LshError>;

/// Errors that can occur in the HyperPhysics LSH module.
#[derive(Error, Debug)]
pub enum LshError {
    // ========================================================================
    // Hashing Errors
    // ========================================================================
    
    /// Input dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality.
        actual: usize,
    },
    
    /// Invalid input (NaN, Inf, empty set).
    #[error("Invalid input: {reason}")]
    InvalidInput {
        /// Description of the invalidity.
        reason: String,
    },
    
    // ========================================================================
    // Index Errors
    // ========================================================================
    
    /// Index capacity exceeded.
    #[error("Index capacity exceeded: max {max}, attempted {attempted}")]
    CapacityExceeded {
        /// Maximum allowed capacity.
        max: usize,
        /// Attempted size.
        attempted: usize,
    },
    
    /// Bucket overflow (too many collisions).
    #[error("Bucket overflow in table {table_id}, bucket {bucket_id}")]
    BucketOverflow {
        /// Table index.
        table_id: usize,
        /// Bucket index.
        bucket_id: u64,
    },
    
    /// Item not found.
    #[error("Item {id} not found")]
    NotFound {
        /// ID of the missing item.
        id: u64,
    },
    
    // ========================================================================
    // Streaming Errors
    // ========================================================================
    
    /// Streaming buffer full.
    #[error("Streaming buffer full, drop rate: {drop_rate:.2}%")]
    BufferFull {
        /// Current drop rate percentage.
        drop_rate: f32,
    },
    
    /// Backpressure threshold exceeded.
    #[error("Backpressure threshold exceeded: {current} pending, max {max}")]
    Backpressure {
        /// Current pending items.
        current: usize,
        /// Maximum allowed.
        max: usize,
    },
    
    // ========================================================================
    // Configuration Errors
    // ========================================================================
    
    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    ConfigError(#[from] crate::config::ConfigError),
    
    /// Index not initialized.
    #[error("Index not initialized")]
    NotInitialized,
    
    // ========================================================================
    // Concurrency Errors
    // ========================================================================
    
    /// Lock acquisition failed.
    #[error("Failed to acquire lock: {reason}")]
    LockError {
        /// Reason for failure.
        reason: String,
    },
}

impl LshError {
    /// Create an invalid input error for NaN values.
    pub fn nan_input(index: usize) -> Self {
        Self::InvalidInput {
            reason: format!("NaN value at index {}", index),
        }
    }
    
    /// Create an invalid input error for empty sets.
    pub fn empty_set() -> Self {
        Self::InvalidInput {
            reason: "Empty set".into(),
        }
    }
    
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::BufferFull { .. }
                | Self::Backpressure { .. }
                | Self::NotFound { .. }
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = LshError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));
    }
    
    #[test]
    fn test_is_recoverable() {
        assert!(LshError::BufferFull { drop_rate: 5.0 }.is_recoverable());
        assert!(LshError::NotFound { id: 42 }.is_recoverable());
        assert!(!LshError::NotInitialized.is_recoverable());
    }
}
