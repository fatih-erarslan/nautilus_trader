//! Optimization configuration and termination criteria.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for optimization algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Maximum number of iterations/generations
    pub max_iterations: u32,
    /// Population/swarm size
    pub population_size: usize,
    /// Convergence tolerance (fitness improvement threshold)
    pub tolerance: f64,
    /// Maximum stagnation iterations before restart
    pub max_stagnation: u32,
    /// Maximum wall-clock time
    pub max_time: Option<Duration>,
    /// Target fitness value (stop if reached)
    pub target_fitness: Option<f64>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable elitism (preserve best individuals)
    pub elitism: bool,
    /// Number of elite individuals to preserve
    pub elite_count: usize,
    /// Initialization strategy
    pub initialization: InitializationStrategy,
    /// Boundary handling strategy
    pub boundary_handling: BoundaryHandling,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            population_size: 50,
            tolerance: 1e-8,
            max_stagnation: 100,
            max_time: Some(Duration::from_secs(60)),
            target_fitness: None,
            seed: None,
            elitism: true,
            elite_count: 2,
            initialization: InitializationStrategy::LatinHypercube,
            boundary_handling: BoundaryHandling::Clamp,
        }
    }
}

impl OptimizationConfig {
    /// Create configuration optimized for HFT (low latency).
    #[must_use]
    pub fn hft() -> Self {
        Self {
            max_iterations: 100,
            population_size: 20,
            tolerance: 1e-6,
            max_stagnation: 20,
            max_time: Some(Duration::from_millis(100)),
            target_fitness: None,
            seed: None,
            elitism: true,
            elite_count: 1,
            initialization: InitializationStrategy::Random,
            boundary_handling: BoundaryHandling::Clamp,
        }
    }

    /// Create configuration for high accuracy.
    #[must_use]
    pub fn high_accuracy() -> Self {
        Self {
            max_iterations: 10000,
            population_size: 100,
            tolerance: 1e-12,
            max_stagnation: 500,
            max_time: Some(Duration::from_secs(300)),
            target_fitness: None,
            seed: None,
            elitism: true,
            elite_count: 5,
            initialization: InitializationStrategy::LatinHypercube,
            boundary_handling: BoundaryHandling::Reflect,
        }
    }

    /// Builder method for max iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Builder method for population size.
    #[must_use]
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Builder method for tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Builder method for target fitness.
    #[must_use]
    pub fn with_target_fitness(mut self, target: f64) -> Self {
        self.target_fitness = Some(target);
        self
    }

    /// Builder method for seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Population initialization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitializationStrategy {
    /// Uniform random initialization
    Random,
    /// Latin Hypercube Sampling (better coverage)
    LatinHypercube,
    /// Sobol sequence (quasi-random)
    Sobol,
    /// Halton sequence (quasi-random)
    Halton,
    /// Grid-based initialization
    Grid,
    /// Centered around a known solution
    Centered,
}

/// Boundary handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryHandling {
    /// Clamp to bounds
    Clamp,
    /// Reflect back into bounds
    Reflect,
    /// Wrap around (periodic)
    Wrap,
    /// Resample randomly
    Resample,
    /// Apply penalty function
    Penalty,
}

/// Termination criterion for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationCriterion {
    /// Maximum iterations reached
    MaxIterations,
    /// Tolerance achieved
    ToleranceReached,
    /// Target fitness achieved
    TargetReached,
    /// Maximum stagnation
    Stagnation,
    /// Time limit exceeded
    TimeLimit,
    /// Manual termination
    Manual,
}

/// Reason for algorithm termination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminationReason {
    /// Primary termination criterion
    pub criterion: TerminationCriterion,
    /// Detailed message
    pub message: String,
    /// Final iteration count
    pub iterations: u32,
    /// Final best fitness
    pub best_fitness: f64,
    /// Wall-clock time in microseconds
    pub time_us: u64,
}

impl TerminationReason {
    /// Create a new termination reason.
    #[must_use]
    pub fn new(criterion: TerminationCriterion, iterations: u32, best_fitness: f64, time_us: u64) -> Self {
        let message = match criterion {
            TerminationCriterion::MaxIterations => format!("Maximum iterations ({}) reached", iterations),
            TerminationCriterion::ToleranceReached => format!("Tolerance achieved at iteration {}", iterations),
            TerminationCriterion::TargetReached => format!("Target fitness achieved: {:.2e}", best_fitness),
            TerminationCriterion::Stagnation => format!("Stagnation detected at iteration {}", iterations),
            TerminationCriterion::TimeLimit => format!("Time limit exceeded after {} μs", time_us),
            TerminationCriterion::Manual => "Manual termination".to_string(),
        };

        Self {
            criterion,
            message,
            iterations,
            best_fitness,
            time_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.population_size, 50);
        assert!(config.elitism);
    }

    #[test]
    fn test_hft_config() {
        let config = OptimizationConfig::hft();
        assert_eq!(config.max_iterations, 100);
        assert!(config.max_time.unwrap() <= Duration::from_millis(100));
    }

    #[test]
    fn test_builder_pattern() {
        let config = OptimizationConfig::default()
            .with_max_iterations(500)
            .with_population_size(100)
            .with_tolerance(1e-10)
            .with_seed(42);

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.population_size, 100);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_termination_reason() {
        let reason = TerminationReason::new(
            TerminationCriterion::ToleranceReached,
            250,
            1e-8,
            1500000,
        );

        assert_eq!(reason.criterion, TerminationCriterion::ToleranceReached);
        assert!(reason.message.contains("250"));
    }
}
