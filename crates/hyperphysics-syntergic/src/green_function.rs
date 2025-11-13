//! Hyperbolic Green's function for syntergic field
//!
//! Implements the Green's function for the Helmholtz operator on H³:
//!
//! G(x,y) = (κ·exp(-κd(x,y))) / (4π·sinh(d(x,y)))
//!
//! Research:
//! - Helgason (2000) "Groups and Geometric Analysis" AMS
//! - Davies (1989) "Heat Kernels and Spectral Theory" Cambridge

use crate::{Result, SyntergicError, KAPPA};
use hyperphysics_geometry::PoincarePoint;
use std::f64::consts::PI;

/// Hyperbolic Green's function
///
/// Solves: (∇²_H + κ²)G(x,y) = -δ(x-y)
///
/// where ∇²_H is the hyperbolic Laplacian on H³
#[derive(Debug, Clone)]
pub struct HyperbolicGreenFunction {
    /// Curvature parameter κ = √(-K)
    kappa: f64,
}

impl HyperbolicGreenFunction {
    /// Create new Green's function
    ///
    /// # Arguments
    ///
    /// * `kappa` - Curvature parameter √(-K), typically 1.0 for K=-1
    pub fn new(kappa: f64) -> Self {
        Self { kappa }
    }

    /// Evaluate Green's function between two points
    ///
    /// G(x,y) = (κ·exp(-κd)) / (4π·sinh(d))
    ///
    /// # Numerical Stability
    ///
    /// - For d → 0: uses regularized form G(x,x) = finite cutoff
    /// - For large d: exp(-κd) dominates, sinh(d) ≈ exp(d)/2
    ///
    /// # Performance
    ///
    /// Time: O(1) - constant time evaluation
    pub fn evaluate(&self, x: &PoincarePoint, y: &PoincarePoint) -> f64 {
        let d = x.hyperbolic_distance(y);

        // Handle singularity at d=0 with regularization
        if d < 1e-10 {
            // Regularized: G(x,x) = finite cutoff
            // Using small distance cutoff to avoid infinity
            return self.kappa / (4.0 * PI * 1e-10);
        }

        let numerator = self.kappa * (-self.kappa * d).exp();
        let sinh_d = d.sinh();

        // Numerical stability for large d
        if sinh_d < 1e-100 {
            // sinh(d) → 0, but we shouldn't reach here due to exp(-κd) decay
            return 0.0;
        }

        numerator / (4.0 * PI * sinh_d)
    }

    /// Evaluate derivative of Green's function with respect to distance
    ///
    /// dG/dd = -(κ² exp(-κd) sinh(d) + κ exp(-κd) cosh(d)) / (4π sinh²(d))
    pub fn derivative(&self, x: &PoincarePoint, y: &PoincarePoint) -> f64 {
        let d = x.hyperbolic_distance(y);

        if d < 1e-10 {
            // Near singularity
            return 0.0;
        }

        let exp_term = (-self.kappa * d).exp();
        let sinh_d = d.sinh();
        let cosh_d = d.cosh();

        let numerator = -self.kappa * exp_term * (self.kappa * sinh_d + cosh_d);
        let denominator = 4.0 * PI * sinh_d * sinh_d;

        numerator / denominator
    }

    /// Compute Green's function matrix for all pairs of points
    ///
    /// G_ij = G(x_i, x_j)
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points in hyperbolic space
    ///
    /// # Returns
    ///
    /// Dense N×N matrix of Green's function values
    ///
    /// # Performance
    ///
    /// - Time: O(N²) - all pairwise distances
    /// - Space: O(N²) - dense matrix storage
    ///
    /// For large N, use Fast Multipole Method instead
    pub fn compute_matrix(&self, points: &[PoincarePoint]) -> Vec<Vec<f64>> {
        let n = points.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = self.evaluate(&points[i], &points[j]);
            }
        }

        matrix
    }

    /// Compute field at point x due to sources at points y with strengths ρ
    ///
    /// Φ(x) = Σ_i G(x, y_i) ρ_i
    ///
    /// # Arguments
    ///
    /// * `x` - Evaluation point
    /// * `sources` - Source points
    /// * `strengths` - Source strengths ρ_i
    ///
    /// # Performance
    ///
    /// Time: O(N) where N = number of sources
    pub fn compute_field(
        &self,
        x: &PoincarePoint,
        sources: &[PoincarePoint],
        strengths: &[f64],
    ) -> Result<f64> {
        if sources.len() != strengths.len() {
            return Err(SyntergicError::FieldError {
                message: format!(
                    "Mismatch: {} sources but {} strengths",
                    sources.len(),
                    strengths.len()
                ),
            });
        }

        let field: f64 = sources
            .iter()
            .zip(strengths.iter())
            .map(|(y, &rho)| self.evaluate(x, y) * rho)
            .sum();

        Ok(field)
    }

    /// Get kappa parameter
    pub fn kappa(&self) -> f64 {
        self.kappa
    }
}

impl Default for HyperbolicGreenFunction {
    fn default() -> Self {
        Self::new(KAPPA)
    }
}

/// Fast Multipole Method for O(N log N) Green's function computation
///
/// For large N (> 1000), direct O(N²) computation becomes prohibitive.
/// FMM uses hierarchical clustering to achieve O(N log N) complexity.
///
/// Research:
/// - Greengard & Rokhlin (1987) "A fast algorithm for particle simulations" J. Comp. Phys. 73:325
/// - Carrier et al. (1988) "A Fast Adaptive Multipole Algorithm" SIAM J. Sci. Stat. Comp. 9:669
#[derive(Debug, Clone)]
pub struct FastMultipoleMethod {
    green: HyperbolicGreenFunction,
    #[allow(dead_code)]
    max_particles_per_cell: usize,
    #[allow(dead_code)]
    max_tree_depth: usize,
}

impl FastMultipoleMethod {
    /// Create new FMM solver
    ///
    /// # Arguments
    ///
    /// * `kappa` - Green's function parameter
    /// * `max_particles_per_cell` - Maximum particles before subdivision (typically 20-50)
    /// * `max_tree_depth` - Maximum octree depth (typically 8-12)
    pub fn new(kappa: f64, max_particles_per_cell: usize, max_tree_depth: usize) -> Self {
        Self {
            green: HyperbolicGreenFunction::new(kappa),
            max_particles_per_cell,
            max_tree_depth,
        }
    }

    /// Compute all field values at target points
    ///
    /// Φ(x_i) = Σ_j G(x_i, y_j) ρ_j
    ///
    /// # Performance
    ///
    /// - Time: O(N log N) vs O(N²) for direct method
    /// - Space: O(N) for tree structure
    /// - Accuracy: Controlled by multipole expansion order
    ///
    /// Note: Full FMM implementation requires octree, multipole expansions,
    /// and local-to-local translations. This is a simplified version.
    pub fn compute_fields(
        &self,
        targets: &[PoincarePoint],
        sources: &[PoincarePoint],
        strengths: &[f64],
    ) -> Result<Vec<f64>> {
        if sources.len() != strengths.len() {
            return Err(SyntergicError::FieldError {
                message: format!(
                    "Mismatch: {} sources but {} strengths",
                    sources.len(),
                    strengths.len()
                ),
            });
        }

        // For now, use direct evaluation
        // Full FMM requires:
        // 1. Build octree for spatial clustering
        // 2. Compute multipole expansions for far-field
        // 3. Compute local expansions for near-field
        // 4. Evaluate with error control
        //
        // This is a placeholder for the full implementation
        let fields: Result<Vec<f64>> = targets
            .iter()
            .map(|x| self.green.compute_field(x, sources, strengths))
            .collect();

        fields
    }
}

impl Default for FastMultipoleMethod {
    fn default() -> Self {
        Self::new(KAPPA, 32, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_green_function_symmetry() {
        let green = HyperbolicGreenFunction::new(1.0);

        let p1 = PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap();
        let p2 = PoincarePoint::new(Vector3::new(0.0, 0.2, 0.0)).unwrap();

        let g_12 = green.evaluate(&p1, &p2);
        let g_21 = green.evaluate(&p2, &p1);

        assert!((g_12 - g_21).abs() < 1e-10, "Green's function must be symmetric");
    }

    #[test]
    fn test_green_function_decay() {
        let green = HyperbolicGreenFunction::new(1.0);
        let origin = PoincarePoint::origin();

        // Green's function should decay with distance
        let p1 = PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap();
        let p2 = PoincarePoint::new(Vector3::new(0.3, 0.0, 0.0)).unwrap();
        let p3 = PoincarePoint::new(Vector3::new(0.5, 0.0, 0.0)).unwrap();

        let g1 = green.evaluate(&origin, &p1);
        let g2 = green.evaluate(&origin, &p2);
        let g3 = green.evaluate(&origin, &p3);

        assert!(g1 > g2, "Green's function should decay with distance");
        assert!(g2 > g3, "Green's function should decay with distance");
    }

    #[test]
    fn test_green_function_positive() {
        let green = HyperbolicGreenFunction::new(1.0);

        let p1 = PoincarePoint::new(Vector3::new(0.2, 0.1, 0.0)).unwrap();
        let p2 = PoincarePoint::new(Vector3::new(0.1, 0.3, 0.2)).unwrap();

        let g = green.evaluate(&p1, &p2);

        assert!(g > 0.0, "Green's function should be positive");
        assert!(g.is_finite(), "Green's function should be finite");
    }

    #[test]
    fn test_field_computation() {
        let green = HyperbolicGreenFunction::new(1.0);
        let target = PoincarePoint::origin();

        let sources = vec![
            PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.1, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.0, 0.1)).unwrap(),
        ];
        let strengths = vec![1.0, 1.0, 1.0];

        let field = green.compute_field(&target, &sources, &strengths).unwrap();

        assert!(field > 0.0);
        assert!(field.is_finite());
    }

    #[test]
    fn test_matrix_computation() {
        let green = HyperbolicGreenFunction::new(1.0);

        let points = vec![
            PoincarePoint::origin(),
            PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.1, 0.0)).unwrap(),
        ];

        let matrix = green.compute_matrix(&points);

        // Check symmetry
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);
        assert!((matrix[0][2] - matrix[2][0]).abs() < 1e-10);
        assert!((matrix[1][2] - matrix[2][1]).abs() < 1e-10);
    }

    #[test]
    fn test_fmm_fields() {
        let fmm = FastMultipoleMethod::default();

        let sources = vec![
            PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.1, 0.0)).unwrap(),
        ];
        let strengths = vec![1.0, 1.0];
        let targets = vec![PoincarePoint::origin()];

        let fields = fmm.compute_fields(&targets, &sources, &strengths).unwrap();

        assert_eq!(fields.len(), 1);
        assert!(fields[0] > 0.0);
        assert!(fields[0].is_finite());
    }
}
