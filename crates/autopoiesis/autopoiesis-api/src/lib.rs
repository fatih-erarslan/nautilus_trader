//! API systems
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod api {
    //! API modules
}

/// Error type for autopoiesis-api
/// 
/// Represents errors that can occur during API operations,
/// including HTTP request processing, response generation,
/// authentication, authorization, and API service management.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// Processing error during API operations
    /// 
    /// This variant represents errors that occur during API request handling,
    /// response processing, authentication, or other API service tasks.
    #[error("API error: {0}")]
    Processing(String),
}

/// Result type for autopoiesis-api operations
/// 
/// A convenience type alias that wraps `std::result::Result` with `ApiError`
/// as the error type. This is used throughout the API module for consistent
/// error handling in HTTP operations and API service management.
pub type Result<T> = std::result::Result<T, ApiError>;

/// Prelude module for convenient imports
/// 
/// This module re-exports commonly used types and traits from the API module,
/// allowing users to import everything they need with a single `use` statement.
/// 
/// # Example
/// 
/// ```rust
/// use autopoiesis_api::prelude::*;
/// ```
pub mod prelude {
    pub use crate::*;
}
