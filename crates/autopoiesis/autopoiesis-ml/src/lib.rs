//! Machine learning components
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod ml {
    //! ML modules
}

/// Error type for autopoiesis-ml
/// 
/// Represents errors that can occur during machine learning operations,
/// including model training, inference, data processing, and neural network operations.
#[derive(Debug, thiserror::Error)]
pub enum MlError {
    /// Processing error during ML operations
    /// 
    /// This variant represents errors that occur during data processing,
    /// model training, inference, or other ML computational tasks.
    #[error("ML error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-ml operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `MlError`
/// as the error type. This is used throughout the ML module for consistent
/// error handling in machine learning operations.
pub type Result<T> = std::result::Result<T, MlError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the ML module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_ml::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
