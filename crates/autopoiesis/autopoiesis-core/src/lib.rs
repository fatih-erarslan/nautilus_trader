//! Core mathematical and system libraries
#![warn(missing_docs)]
#![allow(clippy::all)]

pub mod core {
    //! Core system components
    pub use crate::*;
}

pub mod utils {
    //! Utility functions
    pub use crate::*;
}

pub mod models {
    //! Data models
    pub use crate::*;
}

/// Error type for autopoiesis-core
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    /// Other error  
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for autopoiesis-core
pub type Result<T> = std::result::Result<T, CoreError>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{CoreError as Error, Result};
    pub use async_trait::async_trait;
    pub use serde::{Deserialize, Serialize};
    pub use tracing::{debug, error, info, trace, warn};
}
