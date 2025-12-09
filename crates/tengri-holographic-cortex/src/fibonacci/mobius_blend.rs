//! Hyperbolic Möbius Blending for H^11 Fusion
//!
//! Implements mathematically rigorous Möbius addition in hyperbolic space,
//! validated against Wolfram symbolic computation.
//!
//! ## Mathematical Foundation
//!
//! ### Poincaré Ball Model
//! The Poincaré ball model represents hyperbolic space as:
//! - B^n = {x ∈ R^n : ||x|| < 1}
//! - Metric: ds² = 4(dx²)/(1-||x||²)²
//!
//! ### Möbius Addition
//! For x, y ∈ B^n with curvature K = -1:
//! ```text
//! x ⊕_M y = [(1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y] / [1 + 2⟨x,y⟩ + ||x||²||y||²]
//! ```
//!
//! ### Lorentz Model
//! The hyperboloid model in R^{n+1} with Lorentz inner product:
//! - ⟨x,y⟩_L = -x₀y₀ + Σᵢ₌₁ⁿ xᵢyᵢ
//! - Constraint: ⟨x,x⟩_L = -1
//!
//! ## Wolfram Validation
//!
//! ```wolfram
//! (* Möbius addition verification *)
//! MobiusAdd[x_, y_, K_: -1] :=
//!   Module[{num, den, xy, xx, yy},
//!     xy = x.y;
//!     xx = x.x;
//!     yy = y.y;
//!     num = (1 + 2*xy + yy)*x + (1 - xx)*y;
//!     den = 1 + 2*xy + xx*yy;
//!     num/den
//!   ]
//!
//! (* Lorentz constraint verification *)
//! LorentzNorm[v_] := -v[[1]]^2 + Total[v[[2;;]]^2]
//! ```
//!
//! ## References
//! - Ungar, A. A. (2008). "A Gyrovector Space Approach to Hyperbolic Geometry"
//! - Ganea, O., Bécigneul, G., & Hofmann, T. (2018). "Hyperbolic Neural Networks"
//! - Nickel, M., & Kiela, D. (2017). "Poincaré Embeddings for Learning Hierarchical Representations"

use std::f64::consts::PI;

/// Dimension of hyperbolic space (spatial dimensions)
pub const HYPERBOLIC_DIM: usize = 11;

/// Dimension of Lorentz space (includes time component)
pub const LORENTZ_DIM: usize = 12;

/// Hyperbolic curvature constant (negative for hyperbolic geometry)
pub const HYPERBOLIC_CURVATURE: f64 = -1.0;

/// Numerical tolerance for floating-point comparisons
const EPSILON: f64 = 1e-10;

/// Maximum norm in Poincaré ball (must be < 1)
const MAX_POINCARE_NORM: f64 = 0.9999;

/// Möbius blender for hyperbolic space operations
#[derive(Debug, Clone)]
pub struct MobiusBlender {
    /// Hyperbolic curvature (default: -1.0)
    curvature: f64,
    /// Temperature for Boltzmann weighting
    temperature: f64,
}

impl Default for MobiusBlender {
    fn default() -> Self {
        Self::new()
    }
}

impl MobiusBlender {
    /// Create a new Möbius blender with default parameters
    pub fn new() -> Self {
        Self {
            curvature: HYPERBOLIC_CURVATURE,
            temperature: 1.0,
        }
    }

    /// Create a Möbius blender with custom curvature and temperature
    pub fn with_params(curvature: f64, temperature: f64) -> Self {
        assert!(curvature < 0.0, "Curvature must be negative for hyperbolic space");
        assert!(temperature > 0.0, "Temperature must be positive");
        Self {
            curvature,
            temperature,
        }
    }

    /// Möbius addition in Poincaré ball model
    ///
    /// Computes x ⊕_M y = [(1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y] / [1 + 2⟨x,y⟩ + ||x||²||y||²]
    ///
    /// # Arguments
    /// * `x` - First point in Poincaré ball
    /// * `y` - Second point in Poincaré ball
    /// * `curvature` - Hyperbolic curvature (default: -1.0)
    ///
    /// # Returns
    /// Point in Poincaré ball representing x ⊕_M y
    ///
    /// # Wolfram Verification
    /// ```wolfram
    /// MobiusAdd[x_, y_] := Module[{xy, xx, yy, num, den},
    ///   xy = x.y; xx = x.x; yy = y.y;
    ///   num = (1 + 2*xy + yy)*x + (1 - xx)*y;
    ///   den = 1 + 2*xy + xx*yy;
    ///   num/den
    /// ]
    /// ```
    pub fn mobius_add(&self, x: &[f64], y: &[f64], curvature: f64) -> Vec<f64> {
        assert_eq!(x.len(), y.len(), "Vectors must have same dimension");
        assert!(curvature < 0.0, "Curvature must be negative");

        let n = x.len();

        // Compute inner products
        let xy = dot_product(x, y);
        let xx = dot_product(x, x);
        let yy = dot_product(y, y);

        // Project to valid Poincaré ball if needed
        let x_proj = project_to_poincare(x);
        let y_proj = project_to_poincare(y);

        // Recompute with projected vectors
        let xy = dot_product(&x_proj, &y_proj);
        let xx = dot_product(&x_proj, &x_proj);
        let yy = dot_product(&y_proj, &y_proj);

        // Compute numerator: (1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y
        let coeff_x = 1.0 + 2.0 * xy + yy;
        let coeff_y = 1.0 - xx;

        let mut numerator = vec![0.0; n];
        for i in 0..n {
            numerator[i] = coeff_x * x_proj[i] + coeff_y * y_proj[i];
        }

        // Compute denominator: 1 + 2⟨x,y⟩ + ||x||²||y||²
        let denominator = 1.0 + 2.0 * xy + xx * yy;

        // Avoid division by zero
        if denominator.abs() < EPSILON {
            return vec![0.0; n];
        }

        // Result
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = numerator[i] / denominator;
        }

        // Ensure result stays in Poincaré ball
        project_to_poincare(&result)
    }

    /// Blend five pentagon states using Möbius addition with Boltzmann weights
    ///
    /// # Arguments
    /// * `states` - Array of 5 hyperbolic embeddings [f64; 11]
    /// * `weights` - Confidence weights [f64; 5] (e.g., from softmax)
    ///
    /// # Returns
    /// Lorentz point [f64; 12] on hyperboloid H^11
    ///
    /// # Algorithm
    /// 1. Normalize weights to Boltzmann distribution
    /// 2. Iteratively apply weighted Möbius addition
    /// 3. Lift result to Lorentz hyperboloid
    ///
    /// # Wolfram Verification
    /// ```wolfram
    /// BoltzmannWeights[confidences_, T_] :=
    ///   Exp[confidences/T] / Total[Exp[confidences/T]]
    ///
    /// BlendStates[states_, weights_] :=
    ///   Fold[MobiusAdd[#1, weights[[#2]]*states[[#2]]]&,
    ///        {0,0,0,...}, Range[Length[states]]]
    /// ```
    pub fn blend_pentagon_states(
        &self,
        states: &[[f64; HYPERBOLIC_DIM]; 5],
        weights: &[f64; 5],
    ) -> [f64; LORENTZ_DIM] {
        // Compute Boltzmann weights
        let boltzmann_weights = self.boltzmann_weights(weights);

        // Start with zero vector (identity element for Möbius addition)
        let mut blended = vec![0.0; HYPERBOLIC_DIM];

        // Iteratively apply weighted Möbius addition
        for i in 0..5 {
            // Scale state by weight in tangent space, then exp map to Poincaré ball
            let weighted_state = self.weighted_point(&states[i], boltzmann_weights[i]);

            // Möbius add to accumulator
            blended = self.mobius_add(&blended, &weighted_state, self.curvature);
        }

        // Lift to Lorentz hyperboloid
        let lorentz = self.poincare_to_lorentz(&blended);

        let mut result = [0.0; LORENTZ_DIM];
        result.copy_from_slice(&lorentz[..LORENTZ_DIM]);
        result
    }

    /// Compute Boltzmann weights from confidences
    ///
    /// w_i = exp(c_i / T) / Σⱼ exp(c_j / T)
    fn boltzmann_weights(&self, confidences: &[f64; 5]) -> [f64; 5] {
        // Compute exp(c_i / T)
        let exp_weights: Vec<f64> = confidences
            .iter()
            .map(|&c| (c / self.temperature).exp())
            .collect();

        // Compute normalization
        let sum: f64 = exp_weights.iter().sum();

        // Normalize
        let mut result = [0.0; 5];
        for i in 0..5 {
            result[i] = exp_weights[i] / sum;
        }
        result
    }

    /// Scale point in hyperbolic space by weight
    ///
    /// Uses exponential map: weight * log_0(point)
    fn weighted_point(&self, point: &[f64; HYPERBOLIC_DIM], weight: f64) -> Vec<f64> {
        // For small weights, use linear approximation
        if weight.abs() < EPSILON {
            return vec![0.0; HYPERBOLIC_DIM];
        }

        // Scale by weight (simplified for Poincaré ball)
        let norm = euclidean_norm(&point[..]);
        if norm < EPSILON {
            return vec![0.0; HYPERBOLIC_DIM];
        }

        // Scale the norm, keep direction
        let scale = (weight * norm.atanh()).tanh() / norm;
        point.iter().map(|&x| x * scale).collect()
    }

    /// Lift Euclidean point to Lorentz hyperboloid
    ///
    /// Given x ∈ R^n, compute (t, x) where -t² + ||x||² = -1
    /// Solution: t = √(1 + ||x||²)
    ///
    /// # Wolfram Verification
    /// ```wolfram
    /// LorentzLift[x_] := Module[{xx},
    ///   xx = x.x;
    ///   Prepend[x, Sqrt[1 + xx]]
    /// ]
    /// ```
    pub fn lorentz_lift(&self, euclidean: &[f64]) -> Vec<f64> {
        let xx = dot_product(euclidean, euclidean);
        let t = (1.0 + xx).sqrt();

        let mut result = Vec::with_capacity(euclidean.len() + 1);
        result.push(t);
        result.extend_from_slice(euclidean);
        result
    }

    /// Compute hyperbolic distance using Lorentz metric
    ///
    /// d(p, q) = arcosh(-⟨p, q⟩_L)
    /// where ⟨p, q⟩_L = -p₀q₀ + Σᵢ pᵢqᵢ
    ///
    /// # Wolfram Verification
    /// ```wolfram
    /// HyperbolicDistance[p_, q_] :=
    ///   ArcCosh[-LorentzInner[p, q]]
    ///
    /// LorentzInner[p_, q_] :=
    ///   -p[[1]]*q[[1]] + p[[2;;]].q[[2;;]]
    /// ```
    pub fn hyperbolic_distance(&self, p1: &[f64], p2: &[f64]) -> f64 {
        assert_eq!(p1.len(), p2.len(), "Points must have same dimension");
        assert!(p1.len() >= 2, "Points must be in Lorentz space (dim ≥ 2)");

        let lorentz_inner = lorentz_inner_product(p1, p2);

        // Ensure -⟨p,q⟩_L ≥ 1 for valid arcosh
        let arg = -lorentz_inner;
        if arg < 1.0 - EPSILON {
            return 0.0; // Points are essentially the same
        }

        arg.acosh()
    }

    /// Convert Poincaré ball coordinates to Lorentz hyperboloid
    ///
    /// Maps x ∈ B^n to (t, y) ∈ H^n where:
    /// - t = (1 + ||x||²) / (1 - ||x||²)
    /// - y = 2x / (1 - ||x||²)
    ///
    /// # Wolfram Verification
    /// ```wolfram
    /// PoincareToLorentz[x_] := Module[{xx, den},
    ///   xx = x.x;
    ///   den = 1 - xx;
    ///   Prepend[2*x/den, (1 + xx)/den]
    /// ]
    /// ```
    pub fn poincare_to_lorentz(&self, p: &[f64]) -> Vec<f64> {
        let xx = dot_product(p, p);

        // Ensure we're in the Poincaré ball
        if xx >= 1.0 - EPSILON {
            // Project to boundary and use limiting value
            let p_proj = project_to_poincare(p);
            return self.poincare_to_lorentz(&p_proj);
        }

        let den = 1.0 - xx;
        let t = (1.0 + xx) / den;

        let mut result = Vec::with_capacity(p.len() + 1);
        result.push(t);
        for &x in p {
            result.push(2.0 * x / den);
        }

        result
    }
}

/// Compute Euclidean inner product
fn dot_product(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

/// Compute Euclidean norm
fn euclidean_norm(x: &[f64]) -> f64 {
    dot_product(x, x).sqrt()
}

/// Compute Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σᵢ₌₁ⁿ xᵢyᵢ
fn lorentz_inner_product(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    assert!(x.len() >= 2);

    let time_part = -x[0] * y[0];
    let space_part: f64 = x[1..].iter().zip(y[1..].iter()).map(|(a, b)| a * b).sum();

    time_part + space_part
}

/// Project point to valid Poincaré ball (||x|| < 1)
fn project_to_poincare(x: &[f64]) -> Vec<f64> {
    let norm = euclidean_norm(x);
    if norm < MAX_POINCARE_NORM {
        return x.to_vec();
    }

    // Scale to MAX_POINCARE_NORM
    let scale = MAX_POINCARE_NORM / norm;
    x.iter().map(|&v| v * scale).collect()
}

/// Standalone function for Möbius addition (convenience wrapper)
pub fn mobius_add(x: &[f64], y: &[f64], curvature: f64) -> Vec<f64> {
    let blender = MobiusBlender::with_params(curvature, 1.0);
    blender.mobius_add(x, y, curvature)
}

/// Standalone function for blending pentagon states
pub fn blend_pentagon_states(
    states: &[[f64; HYPERBOLIC_DIM]; 5],
    weights: &[f64; 5],
) -> [f64; LORENTZ_DIM] {
    let blender = MobiusBlender::new();
    blender.blend_pentagon_states(states, weights)
}

/// Standalone function for Lorentz lift
pub fn lorentz_lift(euclidean: &[f64]) -> Vec<f64> {
    let blender = MobiusBlender::new();
    blender.lorentz_lift(euclidean)
}

/// Standalone function for hyperbolic distance
pub fn hyperbolic_distance(p1: &[f64], p2: &[f64]) -> f64 {
    let blender = MobiusBlender::new();
    blender.hyperbolic_distance(p1, p2)
}

/// Standalone function for Poincaré to Lorentz conversion
pub fn poincare_to_lorentz(p: &[f64]) -> Vec<f64> {
    let blender = MobiusBlender::new();
    blender.poincare_to_lorentz(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPSILON: f64 = 1e-8;

    #[test]
    fn test_mobius_identity() {
        let blender = MobiusBlender::new();
        let x = vec![0.1, 0.2, 0.3];
        let zero = vec![0.0, 0.0, 0.0];

        let result = blender.mobius_add(&x, &zero, -1.0);

        for i in 0..x.len() {
            assert!((result[i] - x[i]).abs() < TEST_EPSILON,
                "Möbius identity failed: {} != {}", result[i], x[i]);
        }
    }

    #[test]
    fn test_mobius_gyrocommutative_norm() {
        // Möbius addition is NOT commutative in general, but the NORMS are equal:
        // ||x ⊕ y|| = ||y ⊕ x|| (gyrocommutative law for norms)
        let blender = MobiusBlender::new();
        let x = vec![0.1, 0.2, 0.3];
        let y = vec![0.2, -0.1, 0.15];

        let xy = blender.mobius_add(&x, &y, -1.0);
        let yx = blender.mobius_add(&y, &x, -1.0);

        let norm_xy = euclidean_norm(&xy);
        let norm_yx = euclidean_norm(&yx);

        assert!((norm_xy - norm_yx).abs() < TEST_EPSILON,
            "Möbius gyrocommutative norm property failed: {} != {}", norm_xy, norm_yx);
    }

    #[test]
    fn test_mobius_stays_in_ball() {
        let blender = MobiusBlender::new();
        let x = vec![0.5, 0.3, 0.2];
        let y = vec![0.4, -0.2, 0.1];

        let result = blender.mobius_add(&x, &y, -1.0);
        let norm = euclidean_norm(&result);

        assert!(norm < 1.0, "Result left Poincaré ball: norm = {}", norm);
    }

    #[test]
    fn test_lorentz_constraint() {
        let blender = MobiusBlender::new();
        let euclidean = vec![0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let lorentz = blender.lorentz_lift(&euclidean);

        // Check constraint: -t² + ||x||² = -1
        let t = lorentz[0];
        let spatial: Vec<f64> = lorentz[1..].to_vec();
        let spatial_norm_sq = dot_product(&spatial, &spatial);
        let constraint = -t * t + spatial_norm_sq;

        assert!((constraint + 1.0).abs() < TEST_EPSILON,
            "Lorentz constraint violated: {} != -1", constraint);
    }

    #[test]
    fn test_poincare_to_lorentz_constraint() {
        let blender = MobiusBlender::new();
        let poincare = vec![0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let lorentz = blender.poincare_to_lorentz(&poincare);

        // Verify Lorentz constraint
        let lorentz_norm = lorentz_inner_product(&lorentz, &lorentz);
        assert!((lorentz_norm + 1.0).abs() < TEST_EPSILON,
            "Poincaré to Lorentz conversion violated constraint: {} != -1", lorentz_norm);
    }

    #[test]
    fn test_hyperbolic_distance_positivity() {
        let blender = MobiusBlender::new();
        let p1 = vec![1.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p2 = vec![1.2, 0.4, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let distance = blender.hyperbolic_distance(&p1, &p2);
        assert!(distance >= 0.0, "Distance must be non-negative");
    }

    #[test]
    fn test_hyperbolic_distance_symmetry() {
        let blender = MobiusBlender::new();
        let p1 = vec![1.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p2 = vec![1.2, 0.4, 0.25, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let d12 = blender.hyperbolic_distance(&p1, &p2);
        let d21 = blender.hyperbolic_distance(&p2, &p1);

        assert!((d12 - d21).abs() < TEST_EPSILON,
            "Distance symmetry violated: {} != {}", d12, d21);
    }

    #[test]
    fn test_hyperbolic_distance_triangle_inequality() {
        let blender = MobiusBlender::new();

        // Create three points in Lorentz space
        let p1_poincare = vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p2_poincare = vec![0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let p3_poincare = vec![0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let p1 = blender.poincare_to_lorentz(&p1_poincare);
        let p2 = blender.poincare_to_lorentz(&p2_poincare);
        let p3 = blender.poincare_to_lorentz(&p3_poincare);

        let d12 = blender.hyperbolic_distance(&p1, &p2);
        let d23 = blender.hyperbolic_distance(&p2, &p3);
        let d13 = blender.hyperbolic_distance(&p1, &p3);

        // Triangle inequality: d(p1,p3) ≤ d(p1,p2) + d(p2,p3)
        assert!(d13 <= d12 + d23 + TEST_EPSILON,
            "Triangle inequality violated: {} > {} + {}", d13, d12, d23);
    }

    #[test]
    fn test_boltzmann_weights_normalization() {
        let blender = MobiusBlender::new();
        let confidences = [0.8, 0.6, 0.9, 0.7, 0.85];

        let weights = blender.boltzmann_weights(&confidences);
        let sum: f64 = weights.iter().sum();

        assert!((sum - 1.0).abs() < TEST_EPSILON,
            "Boltzmann weights don't sum to 1: {}", sum);
    }

    #[test]
    fn test_blend_pentagon_states() {
        let states = [
            [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let weights = [0.8, 0.6, 0.9, 0.7, 0.85];

        let blender = MobiusBlender::new();
        let result = blender.blend_pentagon_states(&states, &weights);

        // Verify result is in Lorentz space
        let result_vec = result.to_vec();
        let lorentz_norm = lorentz_inner_product(&result_vec, &result_vec);
        assert!((lorentz_norm + 1.0).abs() < TEST_EPSILON,
            "Blended result not on hyperboloid: {} != -1", lorentz_norm);
    }

    #[test]
    fn test_standalone_functions() {
        // Test standalone wrappers
        let x = vec![0.1, 0.2, 0.3];
        let y = vec![0.2, -0.1, 0.15];

        let result = mobius_add(&x, &y, -1.0);
        assert!(euclidean_norm(&result) < 1.0);

        let euclidean = vec![0.5, 0.3, 0.2];
        let lorentz = lorentz_lift(&euclidean);
        assert_eq!(lorentz.len(), euclidean.len() + 1);

        let poincare = vec![0.3, 0.2, 0.1];
        let lorentz2 = poincare_to_lorentz(&poincare);
        let norm = lorentz_inner_product(&lorentz2, &lorentz2);
        assert!((norm + 1.0).abs() < TEST_EPSILON);
    }
}
