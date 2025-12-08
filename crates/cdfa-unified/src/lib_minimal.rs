//! CDFA Unified - Minimal Working Version
//! Core mathematical functionality for financial trading

// Core mathematical types
pub mod types;
pub mod error;

// Basic precision arithmetic
pub mod precision;

// Core diversity measures
pub mod core {
    pub mod diversity {
        pub mod spearman;
        pub mod pearson;
        pub mod jensen_shannon;
    }
}

// Re-exports for easy access
pub use error::{CdfaError, Result as CdfaResult};
pub use types::{CdfaFloat, CdfaMatrix, CdfaVector};

// Core functionality tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test that the library compiles and basic types work
        let value: CdfaFloat = 1.0;
        assert_eq!(value, 1.0);
    }
}