//! Analysis systems
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod analysis {
    //! Analysis modules
}

/// Error type for autopoiesis-analysis
/// 
/// Represents errors that can occur during analysis operations,
/// including technical analysis, statistical computations, data processing,
/// and analytical algorithm execution.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    /// Processing error during analysis operations
    /// 
    /// This variant represents errors that occur during analytical computations,
    /// technical indicator calculations, statistical analysis, or data processing tasks.
    #[error("Analysis error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-analysis operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `AnalysisError`
/// as the error type. This is used throughout the analysis module for consistent
/// error handling in analytical operations and computational tasks.
pub type Result<T> = std::result::Result<T, AnalysisError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the analysis module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_analysis::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
