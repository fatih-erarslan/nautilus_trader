//! # Unified Error Types
//!
//! Error handling for the hybrid search system.

use thiserror::Error;

/// Result type for hybrid search operations.
pub type Result<T> = std::result::Result<T, HybridError>;

/// Errors that can occur in the hybrid search system.
#[derive(Error, Debug)]
pub enum HybridError {
    /// HNSW (Processing layer) error.
    #[error("HNSW error: {0}")]
    Hnsw(#[from] hyperphysics_hnsw::HnswError),
    
    /// LSH (Acquisition layer) error.
    #[error("LSH error: {0}")]
    Lsh(#[from] hyperphysics_lsh::LshError),
    
    /// Configuration error.
    #[error("Config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    
    /// Router error.
    #[error("Router error: {reason}")]
    Router {
        /// Reason for the error.
        reason: String,
    },
    
    /// Query timeout.
    #[error("Query timeout: {elapsed_us}μs > {limit_us}μs")]
    Timeout {
        /// Elapsed time in microseconds.
        elapsed_us: u64,
        /// Timeout limit in microseconds.
        limit_us: u64,
    },
    
    /// System not initialized.
    #[error("System not initialized: {component}")]
    NotInitialized {
        /// Component that is not initialized.
        component: String,
    },
    
    /// Promotion failed.
    #[error("Pattern promotion failed: {reason}")]
    PromotionFailed {
        /// Reason for failure.
        reason: String,
    },
    
    /// Evolution update failed.
    #[error("Evolution update failed: {reason}")]
    EvolutionFailed {
        /// Reason for failure.
        reason: String,
    },
}

impl HybridError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Timeout { .. } => true,
            Self::Hnsw(e) => e.is_recoverable(),
            Self::Lsh(e) => e.is_recoverable(),
            _ => false,
        }
    }
    
    /// Check if this error requires operator intervention.
    pub fn is_critical(&self) -> bool {
        match self {
            Self::Hnsw(e) => e.is_critical(),
            Self::Config(_) => true,
            Self::NotInitialized { .. } => true,
            _ => false,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timeout_recoverable() {
        let err = HybridError::Timeout {
            elapsed_us: 15000,
            limit_us: 10000,
        };
        assert!(err.is_recoverable());
    }
    
    #[test]
    fn test_not_initialized_critical() {
        let err = HybridError::NotInitialized {
            component: "HNSW".into(),
        };
        assert!(err.is_critical());
    }
}
