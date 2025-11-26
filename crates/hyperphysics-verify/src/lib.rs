//! Formal verification for HyperPhysics
//!
//! This crate provides:
//! - Runtime verification using Z3 SMT solver
//! - Static verification via Lean 4 (FFI integration)
//! - Property-based testing with formal guarantees
//!
//! # Example (requires z3 feature)
//! ```ignore
//! use hyperphysics_verify::z3::Z3Verifier;
//!
//! let verifier = Z3Verifier::new();
//! assert!(verifier.verify_probability_bounds(0.5));
//! ```
//!
//! # Property-based verification (always available)
//! ```
//! use hyperphysics_verify::{verify_probability_bounds_pure, verify_second_law};
//!
//! assert!(verify_probability_bounds_pure(0.5).is_verified());
//! assert!(verify_second_law(1.0, 0.5).is_verified());  // Entropy must increase
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
