//! # HyperPhysics Reasoning Router
//!
//! Modular hierarchical reasoning system with dynamic path selection.
//!
//! ## Architecture
//!
//! ```text
//! Problem → Signature → LSH Match → Router → Backend Pool → Synthesis → Result
//!              ↓                       ↓           ↓
//!         [dim, sparse,           [Thompson    [Physics,
//!          time, struct]           Bandit]      Optimization,
//!                                               Statistical,
//!                                               Formal]
//! ```
//!
//! ## Key Features
//!
//! - **Unified Backend Trait**: Any reasoning system (physics, optimization, statistical)
//! - **LSH Similarity Routing**: Content-addressable problem matching
//! - **Thompson Sampling**: Bayesian bandit for backend selection
//! - **Parallel Racing**: Run multiple backends, first valid wins
//! - **Performance Learning**: Continuous routing improvement
//!
//! ## Usage
//!
//! ```ignore
//! use hyperphysics_reasoning_router::prelude::*;
//!
//! let mut router = ReasoningRouter::new(RouterConfig::default());
//! router.register_backend(Box::new(RapierBackendAdapter::new()));
//! router.register_backend(Box::new(PSOBackendAdapter::new()));
//!
//! let problem = Problem::new(data, ProblemType::Optimization);
//! let result = router.solve(&problem).await?;
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod backend;
pub mod lsh;
pub mod problem;
pub mod router;
pub mod selector;
pub mod synthesis;

#[cfg(feature = "hnsw")]
pub mod hnsw_index;

pub mod prelude {
    //! Convenience re-exports
    pub use crate::backend::{
        BackendCapability, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult,
    };
    pub use crate::lsh::{LSHIndex, LSHConfig};
    pub use crate::problem::{Problem, ProblemSignature, ProblemType};
    pub use crate::router::{ReasoningRouter, RouterConfig};
    pub use crate::selector::{BackendSelector, SelectionStrategy, ThompsonSampler};
    pub use crate::synthesis::{ResultSynthesizer, SynthesisStrategy};

    #[cfg(feature = "hnsw")]
    pub use crate::hnsw_index::{HnswSimilarityConfig, HnswSimilarityIndex, HnswProblemRecord};
}

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Latency tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LatencyTier {
    /// Ultra-fast: < 10μs (lookup tables, SIMD only)
    UltraFast,
    /// Fast: < 1ms (Tier 1 algorithms, deterministic physics)
    Fast,
    /// Medium: 1-10ms (Tier 2 algorithms, hybrid physics)
    Medium,
    /// Slow: 10-100ms (Tier 3 algorithms, full simulation)
    Slow,
    /// Deep: > 100ms (ensemble methods, formal verification)
    Deep,
}

impl LatencyTier {
    /// Get the maximum allowed latency for this tier
    pub fn max_latency(&self) -> Duration {
        match self {
            LatencyTier::UltraFast => Duration::from_micros(10),
            LatencyTier::Fast => Duration::from_millis(1),
            LatencyTier::Medium => Duration::from_millis(10),
            LatencyTier::Slow => Duration::from_millis(100),
            LatencyTier::Deep => Duration::from_secs(1),
        }
    }

    /// Determine tier from duration
    pub fn from_duration(duration: Duration) -> Self {
        if duration < Duration::from_micros(10) {
            LatencyTier::UltraFast
        } else if duration < Duration::from_millis(1) {
            LatencyTier::Fast
        } else if duration < Duration::from_millis(10) {
            LatencyTier::Medium
        } else if duration < Duration::from_millis(100) {
            LatencyTier::Slow
        } else {
            LatencyTier::Deep
        }
    }
}

/// Domain classification for problems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemDomain {
    /// Physics simulation
    Physics,
    /// Optimization problems
    Optimization,
    /// Statistical inference
    Statistical,
    /// Formal verification
    Verification,
    /// Financial/trading
    Financial,
    /// Control systems
    Control,
    /// Engineering applications
    Engineering,
    /// General/unknown
    General,
}

/// Backend pool category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendPool {
    /// Physics engines (Rapier, Jolt, Warp, etc.)
    Physics,
    /// Optimization algorithms (PSO, GA, ACO, etc.)
    Optimization,
    /// Statistical methods (Monte Carlo, Bayesian, Kalman)
    Statistical,
    /// Formal methods (Z3, Lean4)
    Formal,
}

/// Error types for reasoning router
#[derive(thiserror::Error, Debug)]
pub enum RouterError {
    /// No backend available for problem
    #[error("No backend available for problem type: {0:?}")]
    NoBackendAvailable(ProblemDomain),

    /// Backend execution failed
    #[error("Backend {backend_id} execution failed: {message}")]
    BackendFailed {
        /// Backend identifier
        backend_id: String,
        /// Error message
        message: String,
    },

    /// Timeout exceeded
    #[error("Execution timeout exceeded: {0:?}")]
    Timeout(Duration),

    /// Invalid problem specification
    #[error("Invalid problem: {0}")]
    InvalidProblem(String),

    /// Synthesis failed
    #[error("Result synthesis failed: {0}")]
    SynthesisFailed(String),

    /// LSH index error
    #[error("LSH index error: {0}")]
    LSHError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for router operations
pub type RouterResult<T> = Result<T, RouterError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_tier_ordering() {
        assert!(LatencyTier::UltraFast.max_latency() < LatencyTier::Fast.max_latency());
        assert!(LatencyTier::Fast.max_latency() < LatencyTier::Medium.max_latency());
        assert!(LatencyTier::Medium.max_latency() < LatencyTier::Slow.max_latency());
        assert!(LatencyTier::Slow.max_latency() < LatencyTier::Deep.max_latency());
    }

    #[test]
    fn test_latency_tier_from_duration() {
        assert_eq!(
            LatencyTier::from_duration(Duration::from_micros(5)),
            LatencyTier::UltraFast
        );
        assert_eq!(
            LatencyTier::from_duration(Duration::from_micros(500)),
            LatencyTier::Fast
        );
        assert_eq!(
            LatencyTier::from_duration(Duration::from_millis(5)),
            LatencyTier::Medium
        );
        assert_eq!(
            LatencyTier::from_duration(Duration::from_millis(50)),
            LatencyTier::Slow
        );
        assert_eq!(
            LatencyTier::from_duration(Duration::from_secs(2)),
            LatencyTier::Deep
        );
    }
}
