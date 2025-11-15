//! Lattice-based cryptography operations
//!
//! Implements Module-LWE and Module-SIS operations for Dilithium.

pub mod module_lwe;
pub mod ntt;

// Re-export commonly used types
pub use module_lwe::ModuleLWE;
pub use ntt::{NTT, barrett_reduce, montgomery_reduce, poly_add, poly_sub, poly_multiply, constant_time_eq, DILITHIUM_Q, Q, N};

// Placeholder modules for future implementation
// pub mod module_sis;
// pub mod rejection_sampling;
