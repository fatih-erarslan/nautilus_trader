//! # 11D Hyperbolic Geometry (Lorentz Model)
//!
//! Implementation of hyperbolic space H¹¹ using the Lorentz (hyperboloid) model.
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Lorentz Model
//! ```text
//! H¹¹ = { x ∈ ℝ¹² : ⟨x,x⟩_L = -1, x₀ > 0 }
//! ```
//!
//! ### Lorentz Inner Product
//! ```text
//! ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + x₁₁y₁₁
//! ```
//!
//! ### Hyperbolic Distance
//! ```text
//! d_H(x,y) = acosh(-⟨x,y⟩_L)
//! ```
//!
//! ### Lift from ℝ¹¹ to Hyperboloid
//! ```text
//! x₀ = √(1 + ||z||²)
//! ```
//!
//! ### Möbius Addition (Poincaré Ball)
//! ```text
//! x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
//! ```

use crate::constants::*;
use crate::{CortexError, Result};

/// A point on the Lorentz hyperboloid H¹¹ (12D vector)
#[derive(Debug, Clone)]
pub struct LorentzPoint11 {
    /// Coordinates: [x₀, x₁, ..., x₁₁] where x₀ is the time component
    pub coords: [f64; LORENTZ_DIM],
}

impl LorentzPoint11 {
    /// Create origin point (0, 0, ..., 0, 1) in H¹¹
    pub fn origin() -> Self {
        let mut coords = [0.0; LORENTZ_DIM];
        coords[0] = 1.0; // x₀ = 1, all spatial components = 0
        Self { coords }
    }
    
    /// Create from Euclidean coordinates (lift to hyperboloid)
    /// x₀ = √(1 + ||z||²)
    pub fn from_euclidean(z: &[f64]) -> Self {
        let mut coords = [0.0; LORENTZ_DIM];
        
        // Copy spatial components
        let n = z.len().min(HYPERBOLIC_DIM);
        for i in 0..n {
            coords[i + 1] = z[i];
        }
        
        // Compute x₀ = √(1 + ||z||²)
        let norm_sq: f64 = coords[1..].iter().map(|x| x * x).sum();
        coords[0] = (1.0 + norm_sq).sqrt();
        
        Self { coords }
    }
    
    /// Create from raw Lorentz coordinates (must satisfy constraint)
    pub fn from_lorentz(coords: [f64; LORENTZ_DIM]) -> Result<Self> {
        let point = Self { coords };
        let constraint = point.lorentz_constraint();
        
        if (constraint + 1.0).abs() > 0.01 {
            return Err(CortexError::HyperbolicError(
                format!("Point not on hyperboloid: ⟨x,x⟩_L = {} ≠ -1", constraint)
            ));
        }
        
        Ok(point)
    }
    
    /// Project point back onto hyperboloid (normalize)
    pub fn project_to_hyperboloid(&mut self) {
        // Compute current ||z||²
        let spatial_norm_sq: f64 = self.coords[1..].iter().map(|x| x * x).sum();
        
        // Set x₀ to satisfy constraint
        self.coords[0] = (1.0 + spatial_norm_sq).sqrt();
    }
    
    /// Get the Lorentz constraint ⟨x,x⟩_L (should be -1)
    pub fn lorentz_constraint(&self) -> f64 {
        lorentz_inner(&self.coords, &self.coords)
    }
    
    /// Get spatial components [x₁, ..., x₁₁]
    pub fn spatial(&self) -> &[f64] {
        &self.coords[1..]
    }
    
    /// Get time component x₀
    pub fn time(&self) -> f64 {
        self.coords[0]
    }
    
    /// Compute hyperbolic distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        hyperbolic_distance(&self.coords, &other.coords)
    }
    
    /// Project to Poincaré ball (for visualization/Möbius ops)
    pub fn to_poincare(&self) -> Vec<f64> {
        // Projection: p_i = x_i / (1 + x_0)
        let denom = 1.0 + self.coords[0];
        self.coords[1..].iter().map(|&x| x / denom).collect()
    }
}

/// Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ
pub fn lorentz_inner(x: &[f64; LORENTZ_DIM], y: &[f64; LORENTZ_DIM]) -> f64 {
    let mut result = -x[0] * y[0];
    for i in 1..LORENTZ_DIM {
        result += x[i] * y[i];
    }
    result
}

/// Hyperbolic distance: d_H(x,y) = acosh(-⟨x,y⟩_L)
pub fn hyperbolic_distance(x: &[f64; LORENTZ_DIM], y: &[f64; LORENTZ_DIM]) -> f64 {
    let inner = -lorentz_inner(x, y);
    stable_acosh(inner.max(1.0))
}

/// Möbius addition in Poincaré ball with curvature c
/// x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
pub fn mobius_add(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let n = x.len().min(y.len());
    
    // Compute dot products and norms
    let xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let x_norm_sq: f64 = x.iter().map(|&a| a * a).sum();
    let y_norm_sq: f64 = y.iter().map(|&a| a * a).sum();
    
    // Denominator: 1 + 2c⟨x,y⟩ + c²||x||²||y||²
    let denom = 1.0 + 2.0 * c * xy + c * c * x_norm_sq * y_norm_sq;
    
    // Coefficients for x and y
    let coef_x = 1.0 + 2.0 * c * xy + c * y_norm_sq;
    let coef_y = 1.0 - c * x_norm_sq;
    
    // Result
    (0..n)
        .map(|i| (coef_x * x[i] + coef_y * y[i]) / denom.max(HYPERBOLIC_EPSILON))
        .collect()
}

/// Möbius scalar multiplication in Poincaré ball
pub fn mobius_scalar(r: f64, x: &[f64], c: f64) -> Vec<f64> {
    let x_norm: f64 = x.iter().map(|&a| a * a).sum::<f64>().sqrt();
    
    if x_norm < HYPERBOLIC_EPSILON {
        return x.to_vec();
    }
    
    let sqrt_c = c.abs().sqrt();
    
    // tanh(r * artanh(√c ||x||)) / (√c ||x||) * x
    let artanh_arg = sqrt_c * x_norm;
    let artanh_val = if artanh_arg < 1.0 {
        0.5 * ((1.0 + artanh_arg) / (1.0 - artanh_arg)).ln()
    } else {
        HYPERBOLIC_MAX_DIST
    };
    
    let scale = (r * artanh_val).tanh() / (sqrt_c * x_norm);
    
    x.iter().map(|&xi| scale * xi).collect()
}

/// Exponential map: tangent space → hyperboloid
/// exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * (v / ||v||_L)
pub fn exp_map(x: &[f64; LORENTZ_DIM], v: &[f64; LORENTZ_DIM]) -> [f64; LORENTZ_DIM] {
    // Compute Lorentz norm of v (must be spacelike)
    let v_norm_sq = lorentz_inner(v, v);
    
    if v_norm_sq < HYPERBOLIC_EPSILON {
        return *x;
    }
    
    let v_norm = v_norm_sq.sqrt();
    let cosh_norm = v_norm.cosh();
    let sinh_norm = v_norm.sinh();
    
    let mut result = [0.0; LORENTZ_DIM];
    for i in 0..LORENTZ_DIM {
        result[i] = cosh_norm * x[i] + sinh_norm * v[i] / v_norm;
    }
    
    result
}

/// Logarithm map: hyperboloid → tangent space
/// log_x(y) = d(x,y) * (y + ⟨x,y⟩_L * x) / ||y + ⟨x,y⟩_L * x||_L
pub fn log_map(x: &[f64; LORENTZ_DIM], y: &[f64; LORENTZ_DIM]) -> [f64; LORENTZ_DIM] {
    let inner = lorentz_inner(x, y);
    let dist = stable_acosh(-inner);
    
    // Compute y + ⟨x,y⟩_L * x
    let mut diff = [0.0; LORENTZ_DIM];
    for i in 0..LORENTZ_DIM {
        diff[i] = y[i] + inner * x[i];
    }
    
    let diff_norm_sq = lorentz_inner(&diff, &diff);
    
    if diff_norm_sq < HYPERBOLIC_EPSILON {
        return [0.0; LORENTZ_DIM];
    }
    
    let scale = dist / diff_norm_sq.sqrt();
    
    let mut result = [0.0; LORENTZ_DIM];
    for i in 0..LORENTZ_DIM {
        result[i] = scale * diff[i];
    }
    
    result
}

/// Hyperbolic operations trait
pub trait HyperbolicOps {
    /// Compute distance to another point
    fn distance(&self, other: &Self) -> f64;
    
    /// Move along geodesic toward target
    fn move_toward(&mut self, target: &Self, t: f64);
}

impl HyperbolicOps for LorentzPoint11 {
    fn distance(&self, other: &Self) -> f64 {
        self.distance(other)
    }
    
    fn move_toward(&mut self, target: &Self, t: f64) {
        // Compute log map (tangent direction)
        let v = log_map(&self.coords, &target.coords);
        
        // Scale by t
        let mut scaled_v = [0.0; LORENTZ_DIM];
        for i in 0..LORENTZ_DIM {
            scaled_v[i] = t * v[i];
        }
        
        // Apply exp map
        self.coords = exp_map(&self.coords, &scaled_v);
        
        // Project back to hyperboloid (numerical stability)
        self.project_to_hyperboloid();
    }
}

/// Möbius blend of multiple points in Poincaré ball
/// Weighted average then project back
pub struct MobiusBlend {
    /// Curvature parameter (typically -1)
    pub curvature: f64,
}

impl MobiusBlend {
    /// Create new Möbius blender
    pub fn new(curvature: f64) -> Self {
        Self { curvature }
    }
    
    /// Blend multiple points with weights
    pub fn blend(&self, points: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
        if points.is_empty() {
            return vec![];
        }
        
        let dim = points[0].len();
        let c = self.curvature.abs();
        
        // Normalize weights
        let w_sum: f64 = weights.iter().sum();
        let normalized: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();
        
        // Start from first point, iteratively blend
        let mut result = points[0].clone();
        
        for (i, point) in points.iter().enumerate().skip(1) {
            // Möbius weighted combination
            let w = normalized[i] / (normalized[..=i].iter().sum::<f64>());
            
            // Scalar multiply current result by (1-w), point by w, then add
            let scaled_result = mobius_scalar(1.0 - w, &result, c);
            let scaled_point = mobius_scalar(w, point, c);
            result = mobius_add(&scaled_result, &scaled_point, c);
        }
        
        result
    }
    
    /// Blend and lift to Lorentz hyperboloid
    pub fn blend_to_lorentz(&self, points: &[Vec<f64>], weights: &[f64]) -> LorentzPoint11 {
        let poincare_result = self.blend(points, weights);
        LorentzPoint11::from_euclidean(&poincare_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_origin() {
        let origin = LorentzPoint11::origin();
        let constraint = origin.lorentz_constraint();
        assert!((constraint + 1.0).abs() < HYPERBOLIC_EPSILON);
    }
    
    #[test]
    fn test_from_euclidean() {
        let z = vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let point = LorentzPoint11::from_euclidean(&z);
        
        // Should satisfy constraint
        let constraint = point.lorentz_constraint();
        assert!((constraint + 1.0).abs() < 0.001, "Constraint: {}", constraint);
        
        // x₀ should be √(1 + 0.01 + 0.04 + 0.09) = √1.14
        let expected_x0 = (1.0 + 0.01 + 0.04 + 0.09_f64).sqrt();
        assert!((point.time() - expected_x0).abs() < HYPERBOLIC_EPSILON);
    }
    
    #[test]
    fn test_lorentz_inner() {
        let origin = LorentzPoint11::origin();
        let inner = lorentz_inner(&origin.coords, &origin.coords);
        assert!((inner + 1.0).abs() < HYPERBOLIC_EPSILON);
    }
    
    #[test]
    fn test_distance_to_self() {
        let point = LorentzPoint11::from_euclidean(&vec![0.5; HYPERBOLIC_DIM]);
        let dist = point.distance(&point);
        assert!(dist.abs() < 0.001);
    }
    
    #[test]
    fn test_mobius_add_identity() {
        let x = vec![0.3, 0.4];
        let zero = vec![0.0, 0.0];
        
        let result = mobius_add(&x, &zero, 1.0);
        
        assert!((result[0] - x[0]).abs() < 0.001);
        assert!((result[1] - x[1]).abs() < 0.001);
    }
    
    #[test]
    fn test_mobius_add_wolfram() {
        // Verified: Möbius({0.3,0},{0,0.4},c=1) = {0.343, 0.359}
        let x = vec![0.3, 0.0];
        let y = vec![0.0, 0.4];
        
        let result = mobius_add(&x, &y, 1.0);
        
        assert!((result[0] - 0.343).abs() < 0.01);
        assert!((result[1] - 0.359).abs() < 0.01);
        
        // Result norm should be < 1 (Poincaré ball)
        let norm: f64 = result.iter().map(|&r| r * r).sum::<f64>().sqrt();
        assert!(norm < 1.0);
    }
    
    #[test]
    fn test_mobius_blend() {
        let blender = MobiusBlend::new(-1.0);
        
        let points = vec![
            vec![0.3, 0.0],
            vec![0.0, 0.3],
        ];
        let weights = vec![0.5, 0.5];
        
        let result = blender.blend(&points, &weights);
        
        // Should be somewhere between the two points
        let norm: f64 = result.iter().map(|&r| r * r).sum::<f64>().sqrt();
        assert!(norm < 1.0);
    }
}
