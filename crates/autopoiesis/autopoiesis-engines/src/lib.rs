//! Trading engines
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod engines {
    //! Engine modules
}

/// Error type for autopoiesis-engines
/// 
/// Represents errors that can occur during trading engine operations,
/// including engine initialization, execution, strategy processing,
/// and algorithmic trading tasks.
#[derive(Debug, thiserror::Error)]
pub enum EnginesError {
    /// Processing error during trading engine operations
    /// 
    /// This variant represents errors that occur during engine execution,
    /// strategy processing, order management, or other trading engine tasks.
    #[error("Engines error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-engines operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `EnginesError`
/// as the error type. This is used throughout the engines module for consistent
/// error handling in trading engine operations and algorithmic strategies.
pub type Result<T> = std::result::Result<T, EnginesError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the engines module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_engines::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
