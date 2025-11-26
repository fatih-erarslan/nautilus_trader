//! # Neural Trader Syndicate Management System
//!
//! Comprehensive syndicate management system for collaborative sports betting
//! with 100% feature parity with Python implementation.

#![warn(missing_docs)]

mod types;
mod capital;
mod members;
mod voting;
mod collaboration;

pub use types::*;
pub use capital::*;
pub use members::*;
pub use voting::*;
pub use collaboration::*;

use napi_derive::napi;

/// Initialize the syndicate module
#[napi]
pub fn init_syndicate() -> String {
    "Syndicate module initialized".to_string()
}

/// Get module version
#[napi]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_init() {
        assert!(!init_syndicate().is_empty());
    }
}
