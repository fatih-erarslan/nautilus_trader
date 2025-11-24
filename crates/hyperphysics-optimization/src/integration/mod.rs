//! Integration with HyperPhysics reasoning system.
//!
//! This module provides optimization backends for the hyperphysics-reasoning-router.

/// Optimization backend trait for reasoning router integration.
pub trait OptimizationBackend: Send + Sync {
    /// Get backend name.
    fn name(&self) -> &str;

    /// Check if backend is available.
    fn is_available(&self) -> bool;

    /// Get backend priority (higher = preferred).
    fn priority(&self) -> u32;
}

/// Optimization request for reasoning router.
#[derive(Debug, Clone)]
pub struct OptimizationRequest {
    /// Problem dimension.
    pub dimension: usize,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Target accuracy.
    pub target_accuracy: f64,
    /// Maximum time in milliseconds.
    pub max_time_ms: Option<u64>,
    /// Use parallel evaluation.
    pub parallel: bool,
}

impl Default for OptimizationRequest {
    fn default() -> Self {
        Self {
            dimension: 10,
            max_iterations: 1000,
            target_accuracy: 1e-6,
            max_time_ms: Some(1000),
            parallel: true,
        }
    }
}

/// Optimization response from backend.
#[derive(Debug, Clone)]
pub struct OptimizationResponse {
    /// Best solution found.
    pub solution: Vec<f64>,
    /// Best fitness value.
    pub fitness: f64,
    /// Iterations performed.
    pub iterations: u32,
    /// Time taken in microseconds.
    pub time_us: u64,
    /// Convergence status.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_request() {
        let req = OptimizationRequest::default();
        assert_eq!(req.dimension, 10);
        assert!(req.parallel);
    }
}
