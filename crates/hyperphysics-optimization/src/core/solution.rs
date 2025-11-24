//! Solution representation with formal verification properties.
//!
//! # Formal Properties
//!
//! **Invariant I1**: `position.len() == dimension`
//! **Invariant I2**: `∀i: bounds[i].0 ≤ position[i] ≤ bounds[i].1` (when bounded)
//! **Invariant I3**: `fitness` is monotonically updated (for minimization)

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use uuid::Uuid;

/// Individual solution in the search space.
///
/// Represents a candidate solution with position, velocity (for PSO-like algorithms),
/// and cached fitness value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Unique identifier for tracking
    pub id: Uuid,
    /// Position in search space (decision variables)
    pub position: Array1<f64>,
    /// Velocity vector (used by PSO, BA, etc.)
    pub velocity: Option<Array1<f64>>,
    /// Cached fitness value (None if not evaluated)
    pub fitness: Option<f64>,
    /// Personal best position (for PSO-like algorithms)
    pub best_position: Option<Array1<f64>>,
    /// Personal best fitness
    pub best_fitness: Option<f64>,
    /// Number of stagnation iterations
    pub stagnation_count: u32,
    /// Generation/iteration when created
    pub birth_generation: u32,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, f64>,
}

impl Individual {
    /// Create a new individual with given position.
    ///
    /// # Arguments
    /// * `position` - Initial position in search space
    ///
    /// # Formal Postcondition
    /// `result.position.len() == position.len() ∧ result.fitness == None`
    #[must_use]
    pub fn new(position: Array1<f64>) -> Self {
        Self {
            id: Uuid::new_v4(),
            position,
            velocity: None,
            fitness: None,
            best_position: None,
            best_fitness: None,
            stagnation_count: 0,
            birth_generation: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create individual with velocity (for PSO-like algorithms).
    #[must_use]
    pub fn with_velocity(mut self, velocity: Array1<f64>) -> Self {
        debug_assert_eq!(velocity.len(), self.position.len(), "Velocity dimension mismatch");
        self.velocity = Some(velocity);
        self
    }

    /// Set the birth generation.
    #[must_use]
    pub fn with_generation(mut self, generation: u32) -> Self {
        self.birth_generation = generation;
        self
    }

    /// Get the dimensionality of the solution.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.position.len()
    }

    /// Update fitness and track personal best.
    ///
    /// # Formal Specification
    /// ```text
    /// POST: self.fitness == Some(new_fitness)
    /// POST: new_fitness < self.best_fitness.unwrap_or(∞)
    ///       ⟹ self.best_fitness == Some(new_fitness)
    ///          ∧ self.best_position == Some(self.position.clone())
    /// ```
    pub fn update_fitness(&mut self, new_fitness: f64) {
        self.fitness = Some(new_fitness);

        // Update personal best (minimization)
        let is_better = self.best_fitness.map_or(true, |best| new_fitness < best);
        if is_better {
            self.best_fitness = Some(new_fitness);
            self.best_position = Some(self.position.clone());
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }
    }

    /// Check if individual has been evaluated.
    #[must_use]
    pub fn is_evaluated(&self) -> bool {
        self.fitness.is_some()
    }

    /// Get fitness value, panics if not evaluated.
    ///
    /// # Panics
    /// Panics if the individual has not been evaluated yet.
    #[must_use]
    pub fn fitness_unchecked(&self) -> f64 {
        self.fitness.expect("Individual not evaluated")
    }

    /// Euclidean distance to another individual.
    ///
    /// # Formal Specification
    /// ```text
    /// PRE: self.dimension() == other.dimension()
    /// POST: result ≥ 0
    /// POST: result == 0 ⟺ self.position == other.position
    /// ```
    #[must_use]
    pub fn distance_to(&self, other: &Individual) -> f64 {
        debug_assert_eq!(self.dimension(), other.dimension(), "Dimension mismatch");
        self.position
            .iter()
            .zip(other.position.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Update position with bounds enforcement.
    pub fn update_position(&mut self, new_position: ArrayView1<f64>, bounds: &[(f64, f64)]) {
        debug_assert_eq!(new_position.len(), self.dimension(), "Position dimension mismatch");

        for (i, (val, (min, max))) in new_position.iter().zip(bounds.iter()).enumerate() {
            self.position[i] = val.clamp(*min, *max);
        }
    }

    /// Apply velocity update (PSO-style).
    ///
    /// # Arguments
    /// * `bounds` - Search space bounds for clamping
    /// * `max_velocity` - Maximum velocity magnitude
    pub fn apply_velocity(&mut self, bounds: &[(f64, f64)], max_velocity: f64) {
        if let Some(ref velocity) = self.velocity {
            for i in 0..self.dimension() {
                let new_pos = self.position[i] + velocity[i];
                self.position[i] = new_pos.clamp(bounds[i].0, bounds[i].1);
            }
        }
    }

    /// Clamp velocity to maximum magnitude.
    pub fn clamp_velocity(&mut self, max_velocity: f64) {
        if let Some(ref mut velocity) = self.velocity {
            for v in velocity.iter_mut() {
                *v = v.clamp(-max_velocity, max_velocity);
            }
        }
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Individual {}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Individual {
    /// Compare by fitness (lower is better for minimization).
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.fitness, other.fitness) {
            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    }
}

/// Solution wrapper for final optimization results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Decision variable values
    pub position: Vec<f64>,
    /// Objective function value
    pub fitness: f64,
    /// Constraint violations (if any)
    pub violations: Vec<f64>,
    /// Whether solution is feasible
    pub feasible: bool,
    /// Algorithm that produced this solution
    pub algorithm: String,
    /// Number of function evaluations
    pub evaluations: u64,
    /// Wall-clock time in microseconds
    pub time_us: u64,
    /// Convergence history (best fitness per iteration)
    pub convergence_history: Vec<f64>,
}

impl Solution {
    /// Create a new solution from an individual.
    #[must_use]
    pub fn from_individual(individual: &Individual, algorithm: &str, evaluations: u64, time_us: u64) -> Self {
        Self {
            position: individual.position.to_vec(),
            fitness: individual.fitness.unwrap_or(f64::INFINITY),
            violations: Vec::new(),
            feasible: true,
            algorithm: algorithm.to_string(),
            evaluations,
            time_us,
            convergence_history: Vec::new(),
        }
    }

    /// Add convergence history.
    #[must_use]
    pub fn with_history(mut self, history: Vec<f64>) -> Self {
        self.convergence_history = history;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_individual_creation() {
        let pos = array![1.0, 2.0, 3.0];
        let ind = Individual::new(pos.clone());

        assert_eq!(ind.dimension(), 3);
        assert!(!ind.is_evaluated());
        assert_eq!(ind.position, pos);
    }

    #[test]
    fn test_fitness_update() {
        let pos = array![1.0, 2.0, 3.0];
        let mut ind = Individual::new(pos);

        ind.update_fitness(10.0);
        assert_eq!(ind.fitness, Some(10.0));
        assert_eq!(ind.best_fitness, Some(10.0));

        ind.update_fitness(5.0);
        assert_eq!(ind.fitness, Some(5.0));
        assert_eq!(ind.best_fitness, Some(5.0));

        ind.update_fitness(8.0);
        assert_eq!(ind.fitness, Some(8.0));
        assert_eq!(ind.best_fitness, Some(5.0)); // Best unchanged
        assert_eq!(ind.stagnation_count, 1);
    }

    #[test]
    fn test_distance() {
        let ind1 = Individual::new(array![0.0, 0.0, 0.0]);
        let ind2 = Individual::new(array![3.0, 4.0, 0.0]);

        assert!((ind1.distance_to(&ind2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ordering() {
        let mut ind1 = Individual::new(array![1.0]);
        let mut ind2 = Individual::new(array![2.0]);

        ind1.update_fitness(10.0);
        ind2.update_fitness(5.0);

        assert!(ind2 < ind1); // Lower fitness is better
    }

    #[test]
    fn test_velocity_update() {
        let pos = array![5.0, 5.0];
        let vel = array![1.0, -1.0];
        let mut ind = Individual::new(pos).with_velocity(vel);

        let bounds = vec![(0.0, 10.0), (0.0, 10.0)];
        ind.apply_velocity(&bounds, 2.0);

        assert!((ind.position[0] - 6.0).abs() < 1e-10);
        assert!((ind.position[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounds_clamping() {
        let pos = array![5.0, 5.0];
        let vel = array![10.0, -10.0];
        let mut ind = Individual::new(pos).with_velocity(vel);

        let bounds = vec![(0.0, 10.0), (0.0, 10.0)];
        ind.apply_velocity(&bounds, 20.0);

        assert_eq!(ind.position[0], 10.0); // Clamped to max
        assert_eq!(ind.position[1], 0.0);  // Clamped to min
    }
}
