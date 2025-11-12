//! Formal verification for HyperPhysics
//!
//! This crate provides:
//! - Runtime verification using Z3 SMT solver
//! - Static verification via Lean 4 (FFI integration)
//! - Property-based testing with formal guarantees
//!
//! # Example
//! ```
//! use hyperphysics_verify::z3::Z3Verifier;
//!
//! # #[cfg(feature = "z3")]
//! # {
//! let verifier = Z3Verifier::new();
//! assert!(verifier.verify_probability_bounds(0.5));
//! # }
//! ```

#[cfg(feature = "z3")]
pub mod z3;

pub mod properties;
pub mod theorems;

pub use properties::*;

/// Verification result
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationResult {
    /// Property verified successfully
    Verified,
    /// Property violated with counterexample
    Violated(String),
    /// Verification inconclusive
    Unknown,
}

impl VerificationResult {
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified)
    }
}
