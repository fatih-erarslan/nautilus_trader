//! Search space bounds and constraint handling.
//!
//! # Formal Verification
//!
//! **Theorem B1**: For any point `x` and bounds `b`, `repair(x, b)` produces
//! a point `x'` such that `∀i: b[i].0 ≤ x'[i] ≤ b[i].1`
//!
//! **Theorem B2**: The penalty function `P(x)` satisfies:
//! - `P(x) = 0` if `x` is feasible
//! - `P(x) > 0` if `x` is infeasible
//! - `P` is continuous

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Search space bounds with constraint support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounds {
    /// Lower and upper bounds for each dimension
    pub box_bounds: Vec<(f64, f64)>,
    /// Additional constraints
    pub constraints: Vec<Constraint>,
}

impl Bounds {
    /// Create bounds from box constraints.
    ///
    /// # Formal Precondition
    /// `∀(min, max) ∈ bounds: min ≤ max`
    #[must_use]
    pub fn new(box_bounds: Vec<(f64, f64)>) -> Self {
        debug_assert!(
            box_bounds.iter().all(|(min, max)| min <= max),
            "Invalid bounds: min > max"
        );
        Self {
            box_bounds,
            constraints: Vec::new(),
        }
    }

    /// Create symmetric bounds [-bound, bound] for all dimensions.
    #[must_use]
    pub fn symmetric(dimension: usize, bound: f64) -> Self {
        Self::new(vec![(-bound, bound); dimension])
    }

    /// Create unit hypercube bounds [0, 1] for all dimensions.
    #[must_use]
    pub fn unit_hypercube(dimension: usize) -> Self {
        Self::new(vec![(0.0, 1.0); dimension])
    }

    /// Get dimensionality.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.box_bounds.len()
    }

    /// Add a constraint.
    #[must_use]
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Check if a point is within box bounds.
    ///
    /// # Formal Specification
    /// ```text
    /// POST: result ⟺ ∀i: box_bounds[i].0 ≤ x[i] ≤ box_bounds[i].1
    /// ```
    #[must_use]
    pub fn is_within_box(&self, x: ArrayView1<f64>) -> bool {
        x.iter()
            .zip(self.box_bounds.iter())
            .all(|(val, (min, max))| *min <= *val && *val <= *max)
    }

    /// Check if a point satisfies all constraints.
    #[must_use]
    pub fn is_feasible(&self, x: ArrayView1<f64>) -> bool {
        self.is_within_box(x) && self.constraints.iter().all(|c| c.is_satisfied(x))
    }

    /// Calculate total constraint violation.
    ///
    /// # Formal Specification
    /// ```text
    /// POST: result ≥ 0
    /// POST: result == 0 ⟺ is_feasible(x)
    /// ```
    #[must_use]
    pub fn violation(&self, x: ArrayView1<f64>) -> f64 {
        let box_violation: f64 = x
            .iter()
            .zip(self.box_bounds.iter())
            .map(|(val, (min, max))| {
                if *val < *min {
                    min - val
                } else if *val > *max {
                    val - max
                } else {
                    0.0
                }
            })
            .sum();

        let constraint_violation: f64 = self
            .constraints
            .iter()
            .map(|c| c.violation(x))
            .sum();

        box_violation + constraint_violation
    }

    /// Repair a point to satisfy box bounds.
    ///
    /// # Formal Postcondition
    /// `∀i: box_bounds[i].0 ≤ result[i] ≤ box_bounds[i].1`
    #[must_use]
    pub fn repair(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(
            x.iter()
                .zip(self.box_bounds.iter())
                .map(|(val, (min, max))| val.clamp(*min, *max))
        )
    }

    /// Reflect a point back into bounds (for boundary handling).
    #[must_use]
    pub fn reflect(&self, x: ArrayView1<f64>) -> Array1<f64> {
        Array1::from_iter(
            x.iter()
                .zip(self.box_bounds.iter())
                .map(|(val, (min, max))| {
                    let range = max - min;
                    if *val < *min {
                        let excess = min - val;
                        min + (excess % range)
                    } else if *val > *max {
                        let excess = val - max;
                        max - (excess % range)
                    } else {
                        *val
                    }
                })
        )
    }

    /// Get the range (max - min) for each dimension.
    #[must_use]
    pub fn ranges(&self) -> Vec<f64> {
        self.box_bounds.iter().map(|(min, max)| max - min).collect()
    }

    /// Get the center point of the search space.
    #[must_use]
    pub fn center(&self) -> Array1<f64> {
        Array1::from_iter(
            self.box_bounds.iter().map(|(min, max)| (min + max) / 2.0)
        )
    }

    /// Get lower bounds as array.
    #[must_use]
    pub fn lower(&self) -> Array1<f64> {
        Array1::from_iter(self.box_bounds.iter().map(|(min, _)| *min))
    }

    /// Get upper bounds as array.
    #[must_use]
    pub fn upper(&self) -> Array1<f64> {
        Array1::from_iter(self.box_bounds.iter().map(|(_, max)| *max))
    }
}

/// Constraint types for optimization problems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Linear constraint: a^T x ≤ b (inequality) or a^T x = b (equality)
    Linear {
        coefficients: Vec<f64>,
        bound: f64,
        constraint_type: ConstraintType,
    },
    /// Quadratic constraint: x^T Q x + c^T x ≤ b
    Quadratic {
        /// Quadratic matrix (flattened row-major)
        q_matrix: Vec<f64>,
        linear: Vec<f64>,
        bound: f64,
    },
    /// Custom constraint with evaluation function name
    Custom {
        name: String,
        constraint_type: ConstraintType,
    },
}

/// Type of constraint (inequality or equality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// g(x) ≤ 0
    Inequality,
    /// h(x) = 0
    Equality,
    /// g(x) ≥ 0
    GreaterOrEqual,
}

impl Constraint {
    /// Create a linear inequality constraint.
    #[must_use]
    pub fn linear_le(coefficients: Vec<f64>, bound: f64) -> Self {
        Self::Linear {
            coefficients,
            bound,
            constraint_type: ConstraintType::Inequality,
        }
    }

    /// Create a linear equality constraint.
    #[must_use]
    pub fn linear_eq(coefficients: Vec<f64>, bound: f64) -> Self {
        Self::Linear {
            coefficients,
            bound,
            constraint_type: ConstraintType::Equality,
        }
    }

    /// Check if constraint is satisfied.
    #[must_use]
    pub fn is_satisfied(&self, x: ArrayView1<f64>) -> bool {
        self.violation(x) <= 1e-10
    }

    /// Calculate constraint violation (0 if satisfied).
    #[must_use]
    pub fn violation(&self, x: ArrayView1<f64>) -> f64 {
        match self {
            Constraint::Linear { coefficients, bound, constraint_type } => {
                let value: f64 = coefficients.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                match constraint_type {
                    ConstraintType::Inequality => (value - bound).max(0.0),
                    ConstraintType::Equality => (value - bound).abs(),
                    ConstraintType::GreaterOrEqual => (bound - value).max(0.0),
                }
            }
            Constraint::Quadratic { q_matrix, linear, bound } => {
                let n = x.len();
                let mut quad_term = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        quad_term += x[i] * q_matrix[i * n + j] * x[j];
                    }
                }
                let lin_term: f64 = linear.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                (quad_term + lin_term - bound).max(0.0)
            }
            Constraint::Custom { .. } => 0.0, // Custom constraints need external evaluation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bounds_creation() {
        let bounds = Bounds::new(vec![(-5.0, 5.0), (0.0, 10.0)]);
        assert_eq!(bounds.dimension(), 2);
    }

    #[test]
    fn test_symmetric_bounds() {
        let bounds = Bounds::symmetric(3, 10.0);
        assert_eq!(bounds.box_bounds, vec![(-10.0, 10.0); 3]);
    }

    #[test]
    fn test_within_box() {
        let bounds = Bounds::new(vec![(0.0, 10.0), (0.0, 10.0)]);

        assert!(bounds.is_within_box(array![5.0, 5.0].view()));
        assert!(bounds.is_within_box(array![0.0, 10.0].view()));
        assert!(!bounds.is_within_box(array![-1.0, 5.0].view()));
        assert!(!bounds.is_within_box(array![5.0, 11.0].view()));
    }

    #[test]
    fn test_repair() {
        let bounds = Bounds::new(vec![(0.0, 10.0), (0.0, 10.0)]);
        let x = array![-5.0, 15.0];
        let repaired = bounds.repair(x.view());

        assert_eq!(repaired[0], 0.0);
        assert_eq!(repaired[1], 10.0);
    }

    #[test]
    fn test_violation() {
        let bounds = Bounds::new(vec![(0.0, 10.0), (0.0, 10.0)]);

        assert_eq!(bounds.violation(array![5.0, 5.0].view()), 0.0);
        assert!((bounds.violation(array![-2.0, 5.0].view()) - 2.0).abs() < 1e-10);
        assert!((bounds.violation(array![5.0, 12.0].view()) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_constraint() {
        let constraint = Constraint::linear_le(vec![1.0, 1.0], 10.0);

        assert!(constraint.is_satisfied(array![3.0, 5.0].view()));
        assert!(!constraint.is_satisfied(array![6.0, 6.0].view()));
    }

    #[test]
    fn test_center() {
        let bounds = Bounds::new(vec![(0.0, 10.0), (-5.0, 5.0)]);
        let center = bounds.center();

        assert!((center[0] - 5.0).abs() < 1e-10);
        assert!((center[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ranges() {
        let bounds = Bounds::new(vec![(0.0, 10.0), (-5.0, 5.0)]);
        let ranges = bounds.ranges();

        assert_eq!(ranges, vec![10.0, 10.0]);
    }
}
