//! Natural Drift Optimizer based on Maturana & Varela's Autopoiesis Theory
//!
//! ## Theoretical Foundation
//!
//! From "The Tree of Knowledge" (Maturana & Varela, 1987):
//! - Evolution is **satisficing**, not optimizing
//! - Any viable trajectory is acceptable
//! - Conservation of adaptation, not optimization
//!
//! ## Key Concepts
//!
//! 1. **Viability Region**: System must maintain organization within bounds
//! 2. **Natural Drift**: Random perturbations that preserve viability
//! 3. **Satisficing**: Maintaining existence, not maximizing fitness
//!
//! ## References
//!
//! - Maturana, H. R., & Varela, F. J. (1987). The tree of knowledge:
//!   The biological roots of human understanding.
//! - Varela, F. J. (1979). Principles of biological autonomy.

use std::collections::VecDeque;
use nalgebra::DVector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use crate::error::{OptimizationError, OptimizationResult};

/// A viable state in the system's trajectory
#[derive(Debug, Clone)]
pub struct ViableState {
    /// State vector at this point
    pub state: DVector<f64>,
    /// Time step when this state was recorded
    pub timestamp: u64,
    /// Viability score (distance from boundary)
    pub viability_score: f64,
}

/// Result of a drift step
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// New state after drift (may be unchanged if perturbation not viable)
    pub new_state: DVector<f64>,
    /// Whether the new state is viable
    pub is_viable: bool,
    /// Viability score of the new state
    pub viability_score: f64,
    /// Length of recorded trajectory history
    pub trajectory_length: usize,
}

/// Natural Drift Optimizer following Maturana & Varela (1987)
///
/// This optimizer implements the concept of "natural drift" from autopoiesis theory:
/// - **No optimization**: System maintains viability, doesn't maximize fitness
/// - **Satisficing**: Any state within viability bounds is acceptable
/// - **Conservation of adaptation**: System conserves its organization
///
/// # Example
///
/// ```
/// use nalgebra::DVector;
/// use hyperphysics_optimization::natural_drift::NaturalDriftOptimizer;
///
/// let initial_state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
/// let viability_bounds = vec![(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)];
///
/// let mut optimizer = NaturalDriftOptimizer::new(
///     initial_state,
///     viability_bounds,
/// ).unwrap();
///
/// // Execute drift steps - system wanders within viable region
/// for _ in 0..100 {
///     let result = optimizer.drift_step();
///     assert!(result.is_viable);  // Never leaves viability region
/// }
/// ```
pub struct NaturalDriftOptimizer {
    /// Current state vector
    state: DVector<f64>,
    /// Viability bounds [min, max] for each dimension
    viability_bounds: Vec<(f64, f64)>,
    /// Historical viable trajectories (limited buffer)
    trajectory_history: VecDeque<ViableState>,
    /// Maximum trajectory history length
    max_history: usize,
    /// Perturbation scale (standard deviation)
    perturbation_scale: f64,
    /// Random number generator
    rng: ChaCha8Rng,
    /// Current time step
    timestamp: u64,
}

impl NaturalDriftOptimizer {
    /// Create a new Natural Drift Optimizer
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state vector
    /// * `viability_bounds` - Min/max bounds for each dimension
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Bounds length doesn't match state dimension
    /// - Initial state is not viable
    /// - Any bound has min >= max
    pub fn new(
        initial_state: DVector<f64>,
        viability_bounds: Vec<(f64, f64)>,
    ) -> OptimizationResult<Self> {
        // Validate dimensions
        if initial_state.len() != viability_bounds.len() {
            return Err(OptimizationError::DimensionMismatch {
                expected: viability_bounds.len(),
                got: initial_state.len(),
            });
        }

        // Validate bounds
        for (i, &(min, max)) in viability_bounds.iter().enumerate() {
            if min >= max {
                return Err(OptimizationError::InvalidBounds(
                    format!("Dimension {}: min ({}) >= max ({})", i, min, max)
                ));
            }
            if !min.is_finite() || !max.is_finite() {
                return Err(OptimizationError::NumericalError(
                    format!("Non-finite bounds at dimension {}", i)
                ));
            }
        }

        let mut optimizer = Self {
            state: initial_state.clone(),
            viability_bounds,
            trajectory_history: VecDeque::new(),
            max_history: 1000,
            perturbation_scale: 0.1,
            rng: ChaCha8Rng::from_entropy(),
            timestamp: 0,
        };

        // Validate initial state is viable
        if !optimizer.is_viable(&initial_state) {
            return Err(OptimizationError::ConstraintViolation(
                "Initial state is not within viability bounds".to_string()
            ));
        }

        // Record initial state
        let viability = optimizer.viability_score(&initial_state);
        optimizer.trajectory_history.push_back(ViableState {
            state: initial_state,
            timestamp: 0,
            viability_score: viability,
        });

        Ok(optimizer)
    }

    /// Create optimizer with custom RNG seed for reproducibility
    pub fn with_seed(
        initial_state: DVector<f64>,
        viability_bounds: Vec<(f64, f64)>,
        seed: u64,
    ) -> OptimizationResult<Self> {
        let mut optimizer = Self::new(initial_state, viability_bounds)?;
        optimizer.rng = ChaCha8Rng::seed_from_u64(seed);
        Ok(optimizer)
    }

    /// Set the perturbation scale (standard deviation of Gaussian noise)
    pub fn set_perturbation_scale(&mut self, scale: f64) -> OptimizationResult<()> {
        if scale <= 0.0 || !scale.is_finite() {
            return Err(OptimizationError::Configuration(
                format!("Invalid perturbation scale: {}", scale)
            ));
        }
        self.perturbation_scale = scale;
        Ok(())
    }

    /// Set maximum trajectory history length
    pub fn set_max_history(&mut self, max_history: usize) {
        self.max_history = max_history;
        while self.trajectory_history.len() > max_history {
            self.trajectory_history.pop_front();
        }
    }

    /// Execute one drift step - perturb and check viability
    ///
    /// Returns the new state if perturbation is viable, otherwise returns current state.
    /// This implements the core "satisficing" behavior: accept any viable state.
    pub fn drift_step(&mut self) -> DriftResult {
        self.timestamp += 1;

        // Generate perturbation
        let perturbation = self.generate_perturbation(self.perturbation_scale);
        let candidate_state = &self.state + &perturbation;

        // Check viability
        let is_viable = self.is_viable(&candidate_state);
        let viability = self.viability_score(&candidate_state);

        let new_state = if is_viable {
            // Accept perturbation - satisficing behavior
            self.state = candidate_state.clone();

            // Record in trajectory history
            self.trajectory_history.push_back(ViableState {
                state: candidate_state.clone(),
                timestamp: self.timestamp,
                viability_score: viability,
            });

            // Limit history size
            while self.trajectory_history.len() > self.max_history {
                self.trajectory_history.pop_front();
            }

            candidate_state
        } else {
            // Reject perturbation - maintain current state
            self.state.clone()
        };

        DriftResult {
            new_state,
            is_viable,
            viability_score: if is_viable { viability } else { self.viability_score(&self.state) },
            trajectory_length: self.trajectory_history.len(),
        }
    }

    /// Check if state is within viability bounds
    ///
    /// Returns true if all dimensions are within their respective bounds.
    pub fn is_viable(&self, state: &DVector<f64>) -> bool {
        if state.len() != self.viability_bounds.len() {
            return false;
        }

        state.iter().zip(&self.viability_bounds).all(|(x, &(min, max))| {
            x.is_finite() && *x >= min && *x <= max
        })
    }

    /// Calculate viability score (distance from boundary)
    ///
    /// Returns the minimum normalized distance to any boundary:
    /// - Score of 1.0: at center of viable region
    /// - Score approaching 0.0: near boundary
    /// - Negative score: outside viable region
    ///
    /// Formula: min_i(min(x_i - lower_i, upper_i - x_i) / (upper_i - lower_i))
    pub fn viability_score(&self, state: &DVector<f64>) -> f64 {
        if state.len() != self.viability_bounds.len() {
            return f64::NEG_INFINITY;
        }

        state.iter()
            .zip(&self.viability_bounds)
            .map(|(x, &(min, max))| {
                if !x.is_finite() {
                    return f64::NEG_INFINITY;
                }
                let range = max - min;
                let dist_to_lower = x - min;
                let dist_to_upper = max - x;
                dist_to_lower.min(dist_to_upper) / range
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Generate perturbation vector using Gaussian noise
    ///
    /// Each component is drawn from N(0, scale²)
    fn generate_perturbation(&mut self, scale: f64) -> DVector<f64> {
        let normal = Normal::new(0.0, scale).expect("Invalid normal distribution parameters");
        DVector::from_fn(self.state.len(), |_, _| normal.sample(&mut self.rng))
    }

    /// Find viable trajectory from current to target (if exists)
    ///
    /// Uses random walk with bias toward target. Returns path if found within max_steps.
    /// This demonstrates that any viable path is acceptable (satisficing, not optimizing).
    ///
    /// # Arguments
    ///
    /// * `target` - Target state to reach
    /// * `max_steps` - Maximum number of steps to attempt
    ///
    /// # Returns
    ///
    /// Some(path) if viable trajectory found, None otherwise.
    /// Path includes initial and final states.
    pub fn find_viable_path(
        &mut self,
        target: &DVector<f64>,
        max_steps: usize,
    ) -> Option<Vec<DVector<f64>>> {
        // Validate target
        if !self.is_viable(target) {
            return None;
        }

        let mut path = vec![self.state.clone()];
        let mut current = self.state.clone();
        let initial_distance = (&current - target).norm();

        for step in 0..max_steps {
            // Bias perturbation toward target
            let direction = target - &current;
            let direction_norm = direction.norm();

            if direction_norm < 1e-6 {
                // Reached target
                path.push(target.clone());
                return Some(path);
            }

            // Adaptive step size based on distance
            let step_scale = (direction_norm / initial_distance).min(1.0) * self.perturbation_scale;

            // Perturbation with bias toward target
            let bias_strength = 0.5; // 50% bias toward target
            let random_component = self.generate_perturbation(step_scale);
            let biased_direction = &direction * (bias_strength * step_scale / direction_norm);
            let perturbation = &random_component * (1.0 - bias_strength) + biased_direction;

            let candidate = &current + &perturbation;

            // Accept if viable
            if self.is_viable(&candidate) {
                current = candidate.clone();
                path.push(candidate);

                // Check if close enough to target
                if (&current - target).norm() < step_scale {
                    path.push(target.clone());
                    return Some(path);
                }
            }

            // Timeout if making no progress
            if step > max_steps / 2 && (&current - target).norm() > initial_distance * 0.9 {
                return None;
            }
        }

        None
    }

    /// Get current state
    pub fn current_state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Get trajectory history
    pub fn trajectory_history(&self) -> &VecDeque<ViableState> {
        &self.trajectory_history
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get viability bounds
    pub fn viability_bounds(&self) -> &[(f64, f64)] {
        &self.viability_bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_create_optimizer() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];

        let optimizer = NaturalDriftOptimizer::new(state, bounds);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0)]; // Wrong dimension

        let result = NaturalDriftOptimizer::new(state, bounds);
        assert!(matches!(result, Err(OptimizationError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_invalid_bounds() {
        let state = DVector::from_vec(vec![0.0]);
        let bounds = vec![(1.0, -1.0)]; // min > max

        let result = NaturalDriftOptimizer::new(state, bounds);
        assert!(matches!(result, Err(OptimizationError::InvalidBounds(_))));
    }

    #[test]
    fn test_initial_state_not_viable() {
        let state = DVector::from_vec(vec![2.0]); // Outside bounds
        let bounds = vec![(-1.0, 1.0)];

        let result = NaturalDriftOptimizer::new(state, bounds);
        assert!(matches!(result, Err(OptimizationError::ConstraintViolation(_))));
    }

    #[test]
    fn test_is_viable() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let optimizer = NaturalDriftOptimizer::new(state, bounds).unwrap();

        // Test viable states
        assert!(optimizer.is_viable(&DVector::from_vec(vec![0.0, 0.0])));
        assert!(optimizer.is_viable(&DVector::from_vec(vec![0.5, -0.5])));
        assert!(optimizer.is_viable(&DVector::from_vec(vec![1.0, 1.0])));
        assert!(optimizer.is_viable(&DVector::from_vec(vec![-1.0, -1.0])));

        // Test non-viable states
        assert!(!optimizer.is_viable(&DVector::from_vec(vec![1.1, 0.0])));
        assert!(!optimizer.is_viable(&DVector::from_vec(vec![0.0, -1.1])));
        assert!(!optimizer.is_viable(&DVector::from_vec(vec![2.0, 2.0])));
    }

    #[test]
    fn test_viability_score() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let optimizer = NaturalDriftOptimizer::new(state, bounds).unwrap();

        // Center should have score 0.5 (halfway from any boundary)
        let center_score = optimizer.viability_score(&DVector::from_vec(vec![0.0, 0.0]));
        assert_relative_eq!(center_score, 0.5, epsilon = 1e-10);

        // At boundary should have score 0.0
        let boundary_score = optimizer.viability_score(&DVector::from_vec(vec![1.0, 0.0]));
        assert_relative_eq!(boundary_score, 0.0, epsilon = 1e-10);

        // Outside boundary should have negative score
        let outside_score = optimizer.viability_score(&DVector::from_vec(vec![1.5, 0.0]));
        assert!(outside_score < 0.0);
    }

    #[test]
    fn test_drift_never_leaves_viable_region() {
        let state = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        // Run many drift steps
        for _ in 0..1000 {
            let result = optimizer.drift_step();

            // Verify state remains viable
            assert!(optimizer.is_viable(&result.new_state),
                "State left viable region: {:?}", result.new_state);
            assert!(result.viability_score >= 0.0 || !result.is_viable,
                "Viability score negative for viable state");
        }
    }

    #[test]
    fn test_satisficing_behavior() {
        // Test that optimizer doesn't maximize, just maintains viability
        let state = DVector::from_vec(vec![0.0]);
        let bounds = vec![(-10.0, 10.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        let mut positions = Vec::new();
        for _ in 0..100 {
            optimizer.drift_step();
            positions.push(optimizer.current_state()[0]);
        }

        // Check that we explore the space (not stuck)
        let min_pos = positions.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_pos = positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_pos - min_pos > 0.5, "System should explore space");

        // Verify all positions are viable
        assert!(positions.iter().all(|&p| p >= -10.0 && p <= 10.0));
    }

    #[test]
    fn test_find_viable_path_same_point() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state.clone(), bounds, 42).unwrap();

        let path = optimizer.find_viable_path(&state, 100);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 2); // Start and end
    }

    #[test]
    fn test_find_viable_path_target_not_viable() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        let outside_target = DVector::from_vec(vec![2.0, 2.0]);
        let path = optimizer.find_viable_path(&outside_target, 100);
        assert!(path.is_none());
    }

    #[test]
    fn test_find_viable_path_returns_valid_trajectory() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        let target = DVector::from_vec(vec![0.5, 0.5]);
        let path = optimizer.find_viable_path(&target, 1000);

        if let Some(path) = path {
            // Verify all states in path are viable
            for state in &path {
                assert!(optimizer.is_viable(state),
                    "Path contains non-viable state: {:?}", state);
            }

            // Verify path starts at current state and ends at target
            assert_relative_eq!(path[0][0], 0.0, epsilon = 1e-10);
            assert_relative_eq!(path[0][1], 0.0, epsilon = 1e-10);

            let last = path.last().unwrap();
            assert_relative_eq!(last[0], 0.5, epsilon = 0.1);
            assert_relative_eq!(last[1], 0.5, epsilon = 0.1);
        }
    }

    #[test]
    fn test_trajectory_history() {
        let state = DVector::from_vec(vec![0.0]);
        let bounds = vec![(-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        // Should start with initial state
        assert_eq!(optimizer.trajectory_history().len(), 1);

        // Run some drift steps
        for _ in 0..10 {
            optimizer.drift_step();
        }

        // History should grow (viable steps are accepted)
        let history_len = optimizer.trajectory_history().len();
        assert!(history_len > 1);
        assert!(history_len <= 11); // Initial + up to 10 viable steps
    }

    #[test]
    fn test_max_history_limit() {
        let state = DVector::from_vec(vec![0.0]);
        let bounds = vec![(-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        optimizer.set_max_history(5);

        // Run many steps
        for _ in 0..100 {
            optimizer.drift_step();
        }

        // History should be limited
        assert!(optimizer.trajectory_history().len() <= 5);
    }

    #[test]
    fn test_perturbation_scale() {
        let state = DVector::from_vec(vec![0.0]);
        let bounds = vec![(-1.0, 1.0)];
        let mut optimizer = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        // Test valid scale
        assert!(optimizer.set_perturbation_scale(0.05).is_ok());

        // Test invalid scales
        assert!(optimizer.set_perturbation_scale(0.0).is_err());
        assert!(optimizer.set_perturbation_scale(-0.1).is_err());
        assert!(optimizer.set_perturbation_scale(f64::NAN).is_err());
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let state = DVector::from_vec(vec![0.0, 0.0]);
        let bounds = vec![(-1.0, 1.0), (-1.0, 1.0)];

        let mut opt1 = NaturalDriftOptimizer::with_seed(state.clone(), bounds.clone(), 42).unwrap();
        let mut opt2 = NaturalDriftOptimizer::with_seed(state, bounds, 42).unwrap();

        // Both should produce identical sequences
        for _ in 0..10 {
            let result1 = opt1.drift_step();
            let result2 = opt2.drift_step();

            assert_relative_eq!(result1.new_state[0], result2.new_state[0], epsilon = 1e-10);
            assert_relative_eq!(result1.new_state[1], result2.new_state[1], epsilon = 1e-10);
        }
    }
}
