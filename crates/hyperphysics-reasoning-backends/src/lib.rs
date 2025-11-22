//! Backend adapters for the HyperPhysics reasoning router.
//!
//! This crate provides concrete implementations of the `ReasoningBackend` trait
//! for various physics engines, optimizers, and verification systems.
//!
//! # Backend Categories
//!
//! - **Physics**: Rapier3D, Jolt, Warp, Taichi, MuJoCo, Genesis, Avian, Chrono
//! - **Optimization**: Particle Swarm Optimization (PSO), Genetic Algorithm (GA),
//!   Ant Colony Optimization (ACO)
//! - **Statistical**: Monte Carlo, Bayesian Inference, Kalman Filtering
//! - **Formal**: Z3 SMT solver, verification proofs
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   ReasoningRouter                           │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              ReasoningBackend Trait                  │   │
//! │  └───────────────────────┬─────────────────────────────┘   │
//! │                          │                                  │
//! │  ┌───────────┬───────────┼───────────┬───────────┐        │
//! │  ▼           ▼           ▼           ▼           ▼        │
//! │ Physics   Optim      Statistical  Formal    Custom        │
//! │ Adapter   Adapter    Adapter      Adapter   Adapter       │
//! │  │           │           │           │           │         │
//! │  ▼           ▼           ▼           ▼           ▼        │
//! │ Rapier    PSO/GA     MonteCarlo   Z3       User           │
//! │ Jolt      ACO        Bayesian     Lean4    Defined        │
//! │ Warp      DE         Kalman       ...      ...            │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "optimization")]
pub mod optimization;

#[cfg(feature = "statistical")]
pub mod statistical;

#[cfg(feature = "formal")]
pub mod formal;

// Re-exports
pub use hyperphysics_reasoning_router::{
    backend::{BackendCapability, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult, ResultValue},
    problem::{Problem, ProblemData, ProblemSignature, ProblemType},
    BackendPool, LatencyTier, ProblemDomain, RouterError, RouterResult,
};

// Re-export for internal use
pub(crate) mod problem {
    pub use hyperphysics_reasoning_router::problem::*;
}

#[cfg(feature = "physics")]
pub use physics::PhysicsBackendAdapter;

#[cfg(feature = "optimization")]
pub use optimization::{GeneticAlgorithmBackend, PSOBackend};

#[cfg(feature = "statistical")]
pub use statistical::MonteCarloBackend;

/// Prelude for common imports
pub mod prelude {
    pub use super::*;
    pub use hyperphysics_reasoning_router::prelude::*;
}

/// Create a default set of backends for a full reasoning system
pub fn create_default_backends() -> Vec<std::sync::Arc<dyn ReasoningBackend>> {
    let mut backends: Vec<std::sync::Arc<dyn ReasoningBackend>> = Vec::new();

    #[cfg(feature = "optimization")]
    {
        backends.push(std::sync::Arc::new(optimization::PSOBackend::new(
            optimization::PSOConfig::default(),
        )));
        backends.push(std::sync::Arc::new(optimization::GeneticAlgorithmBackend::new(
            optimization::GAConfig::default(),
        )));
    }

    #[cfg(feature = "statistical")]
    {
        backends.push(std::sync::Arc::new(statistical::MonteCarloBackend::new(
            statistical::MonteCarloConfig::default(),
        )));
    }

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_pool_types() {
        assert_ne!(BackendPool::Physics, BackendPool::Optimization);
        assert_ne!(BackendPool::Statistical, BackendPool::Formal);
    }

    #[test]
    fn test_latency_tiers() {
        assert!(LatencyTier::UltraFast < LatencyTier::Fast);
        assert!(LatencyTier::Fast < LatencyTier::Medium);
    }
}
