//! Foreign Function Interface for CDFA
//! 
//! This crate provides:
//! - C API for integration with other languages
//! - Python bindings via PyO3
//! - Safe wrappers around unsafe FFI
//! - Memory management utilities

#[cfg(feature = "c-api")]
pub mod c_api;

#[cfg(feature = "python")]
pub mod python;

pub mod utils;

#[cfg(feature = "c-api")]
pub use c_api::*;

#[cfg(feature = "python")]
pub use python::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_initialization() {
        // Placeholder test
        assert!(true);
    }
}