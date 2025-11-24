//! # HyperPhysics Optimization
//!
//! Enterprise-grade bio-inspired optimization algorithms with formal verification.
//!
//! ## Features
//!
//! - **14 Bio-Inspired Algorithms**: PSO, GA, DE, GWO, WOA, ACO, BA, FA, CS, ABC, BFO, SSO, MFO, SSA
//! - **SIMD Acceleration**: Vectorized operations for HFT latency requirements
//! - **Parallel Evaluation**: Rayon-based parallel fitness computation
//! - **Formal Verification**: Mathematical proofs of convergence properties
//! - **Benchmark Suite**: CEC2017/2020 test functions for validation
//!
//! ## Enterprise Requirements
//!
//! - Convergence guarantees with formal proofs
//! - <1ms latency for 100-dimensional problems
//! - >90% test coverage with property-based testing
//! - Thread-safe population management
//! - Serializable state for checkpointing

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]

pub mod core;
pub mod algorithms;

#[cfg(feature = "benchmarks")]
pub mod benchmarks;

#[cfg(feature = "selector")]
pub mod selector;

pub mod integration;

// Re-exports for convenience
pub use crate::core::{
    Bounds, Constraint, ConstraintType, Individual, ObjectiveFunction,
    OptimizationConfig, Population, Solution,
    TerminationCriterion, TerminationReason,
};

pub use crate::algorithms::{
    Algorithm, AlgorithmConfig, AlgorithmType, ConvergenceProfile,
    ComputationalComplexity,
};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::core::*;
    pub use crate::algorithms::*;

    #[cfg(feature = "benchmarks")]
    pub use crate::benchmarks::*;

    #[cfg(feature = "selector")]
    pub use crate::selector::*;
}

/// Error types for optimization operations
#[derive(thiserror::Error, Debug)]
pub enum OptimizationError {
    /// Algorithm failed to converge
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),

    /// Invalid configuration provided
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Constraint violation detected
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Objective function evaluation failed
    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    /// Population management error
    #[error("Population error: {0}")]
    PopulationError(String),

    /// No solution found
    #[error("No solution found: {0}")]
    NoSolution(String),

    /// Formal verification failed
    #[error("Verification failure: {0}")]
    VerificationFailure(String),
}

/// Result type for optimization operations
pub type OptimizationResult<T> = Result<T, OptimizationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OptimizationError::ConvergenceFailure("max iterations reached".to_string());
        assert!(err.to_string().contains("Convergence failure"));
    }
}
