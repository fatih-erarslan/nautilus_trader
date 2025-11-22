//! Consciousness systems
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod consciousness {
    //! Consciousness modules
}

/// Error type for autopoiesis-consciousness
/// 
/// Represents errors that can occur during consciousness operations,
/// including neuronal field computations, consciousness state processing,
/// quantum information processing, and consciousness emergence detection.
#[derive(Debug, thiserror::Error)]
pub enum ConsciousnessError {
    /// Processing error during consciousness operations
    /// 
    /// This variant represents errors that occur during consciousness computations,
    /// neuronal field processing, quantum state evolution, or consciousness detection tasks.
    #[error("Consciousness error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-consciousness operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `ConsciousnessError`
/// as the error type. This is used throughout the consciousness module for consistent
/// error handling in consciousness-related operations and computations.
pub type Result<T> = std::result::Result<T, ConsciousnessError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the consciousness module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_consciousness::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
