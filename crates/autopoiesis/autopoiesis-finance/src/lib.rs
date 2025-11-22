//! Financial systems
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod finance {
    //! Finance modules
}

/// Error type for autopoiesis-finance
/// 
/// Represents errors that can occur during financial operations,
/// including trading, market data processing, portfolio management,
/// and financial calculations.
#[derive(Debug, thiserror::Error)]
pub enum FinanceError {
    /// Processing error during financial operations
    /// 
    /// This variant represents errors that occur during financial calculations,
    /// trading operations, market data processing, or portfolio management tasks.
    #[error("Finance error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-finance operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `FinanceError`
/// as the error type. This is used throughout the finance module for consistent
/// error handling in financial operations and trading systems.
pub type Result<T> = std::result::Result<T, FinanceError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the finance module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_finance::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
