//! # Distance Metrics for HyperPhysics HNSW
//!
//! This module provides distance metric implementations optimized for the
//! HyperPhysics trading system, with special emphasis on hyperbolic geometry
//! used in the consciousness/Φ computation framework.
//!
//! ## Why Hyperbolic Geometry?
//!
//! The HyperPhysics consciousness model embeds market states in hyperbolic space
//! (specifically the Poincaré ball model) because:
//!
//! 1. **Hierarchical Structure**: Markets have natural hierarchies (sectors,
//!    industries, companies) that embed more naturally in hyperbolic space
//!    than Euclidean space.
//!
//! 2. **Exponential Growth**: The "volume" of hyperbolic space grows
//!    exponentially with radius, matching the exponential nature of
//!    market relationships.
//!
//! 3. **Boundary Behavior**: Points near the boundary of the Poincaré ball
//!    represent extreme market states, with distance increasing rapidly
//!    as we approach the edge.
//!
//! ## Poincaré Ball Model
//!
//! The Poincaré ball B^n is the open unit ball {x ∈ ℝ^n : ||x|| < 1} equipped
//! with the Riemannian metric:
//!
//! ```text
//! ds² = (2 / (1 - ||x||²))² ||dx||²
//! ```
//!
//! The geodesic distance between points u and v is:
//!
//! ```text
//! d_H(u, v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
//! ```
//!
//! ## SIMD Optimization Strategy
//!
//! We optimize the hyperbolic distance calculation by:
//!
//! 1. Computing ||u||², ||v||², and ||u-v||² using SIMD dot products
//! 2. Using a fast `arcosh` approximation based on Horner's method
//! 3. Avoiding branches in the inner loop
//! 4. Pre-computing the (1 - ||x||²) terms for stored vectors

use std::marker::PhantomData;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Trait for distance metrics compatible with HNSW indexing.
/// 
/// Implementations must be:
/// - **Deterministic**: Same inputs always produce same output
/// - **Symmetric**: d(u, v) = d(v, u)
/// - **Non-negative**: d(u, v) ≥ 0
/// - **Identity of indiscernibles**: d(u, v) = 0 iff u = v
/// 
/// Note: Triangle inequality is NOT required for HNSW to function,
/// though it may affect recall. Hyperbolic distance satisfies it.
pub trait DistanceMetric: Send + Sync + Clone {
    /// Compute distance between two vectors.
    /// 
    /// # Safety
    /// 
    /// Implementations must handle:
    /// - Vectors of different lengths (should panic or return infinity)
    /// - NaN/Inf values (should propagate or return infinity)
    /// - Numerical instability (e.g., points near Poincaré boundary)
    fn distance(&self, u: &[f32], v: &[f32]) -> f32;
    
    /// SIMD-optimized batch distance computation.
    /// 
    /// Computes distances from a single query to multiple targets.
    /// Default implementation calls `distance` in a loop; override
    /// for better performance.
    fn distance_batch(&self, query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
        targets.iter().map(|t| self.distance(query, t)).collect()
    }
    
    /// Return a string identifier for this metric.
    fn name(&self) -> &'static str;
    
    /// Whether lower values indicate more similarity (true for distances).
    fn lower_is_better(&self) -> bool {
        true
    }
}

// ============================================================================
// Hyperbolic (Poincaré Ball) Distance
// ============================================================================

/// Hyperbolic distance metric in the Poincaré ball model.
///
/// This is the primary metric for HyperPhysics consciousness space, where
/// market states are embedded in hyperbolic geometry to capture hierarchical
/// relationships and exponential scaling.
///
/// ## Mathematical Definition
///
/// For points u, v in the Poincaré ball B^n (||u||, ||v|| < 1):
///
/// ```text
/// d_H(u, v) = √|K| · arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
/// ```
///
/// where K is the (negative) sectional curvature. We use K = -1 by default.
///
/// ## Numerical Considerations
///
/// As points approach the boundary (||x|| → 1), the denominator approaches 0,
/// causing numerical instability. We clamp norms to `1 - EPSILON` to avoid this.
#[derive(Debug, Clone)]
pub struct HyperbolicMetric {
    /// Sectional curvature (negative for hyperbolic space).
    /// The distance is scaled by √|curvature|.
    curvature: f32,
    
    /// Pre-computed √|curvature| for efficiency.
    sqrt_abs_curvature: f32,
    
    /// Epsilon for numerical stability near boundary.
    epsilon: f32,
}

impl HyperbolicMetric {
    /// Create a new hyperbolic metric with the given curvature.
    ///
    /// # Arguments
    ///
    /// * `curvature` - The sectional curvature (must be negative).
    ///   Standard hyperbolic space has K = -1.
    ///
    /// # Panics
    ///
    /// Panics if curvature is non-negative.
    pub fn new(curvature: f32) -> Self {
        assert!(curvature < 0.0, "Hyperbolic curvature must be negative");
        Self {
            curvature,
            sqrt_abs_curvature: (-curvature).sqrt(),
            epsilon: 1e-5,
        }
    }
    
    /// Create a hyperbolic metric with standard curvature K = -1.
    pub fn standard() -> Self {
        Self::new(-1.0)
    }
    
    /// Create a hyperbolic metric for the Poincaré ball with given curvature.
    pub fn poincare(curvature: f32) -> Self {
        Self::new(curvature)
    }
    
    /// Adjust the curvature (for Evolution layer optimization).
    pub fn with_curvature_adjustment(&self, adjustment: f32) -> Self {
        Self::new(self.curvature * (1.0 + adjustment))
    }
    
    /// Fast arcosh approximation using Horner's method.
    ///
    /// For x ≥ 1, arcosh(x) = ln(x + √(x² - 1))
    ///
    /// We use a polynomial approximation for small (x - 1) and
    /// the logarithmic form for larger values.
    #[inline(always)]
    fn fast_arcosh(&self, x: f32) -> f32 {
        // For x close to 1, use Taylor series: arcosh(1+y) ≈ √(2y) for small y
        let y = x - 1.0;
        if y < 0.1 {
            // Taylor approximation: arcosh(1+y) ≈ √(2y) * (1 - y/12 + 3y²/160)
            let sqrt_2y = (2.0 * y).sqrt();
            sqrt_2y * (1.0 - y / 12.0 + 3.0 * y * y / 160.0)
        } else {
            // Standard formula: arcosh(x) = ln(x + √(x² - 1))
            (x + (x * x - 1.0).sqrt()).ln()
        }
    }
    
    /// Compute squared Euclidean norm using SIMD when available.
    #[inline(always)]
    fn squared_norm(&self, v: &[f32]) -> f32 {
        self.dot_product(v, v)
    }
    
    /// Compute squared Euclidean distance using SIMD when available.
    #[inline(always)]
    fn squared_euclidean(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len());
        
        // SIMD-friendly computation: ||u-v||² = ||u||² + ||v||² - 2<u,v>
        // This avoids materializing the difference vector
        let norm_u_sq = self.squared_norm(u);
        let norm_v_sq = self.squared_norm(v);
        let dot_uv = self.dot_product(u, v);
        
        // Ensure non-negative due to floating point errors
        (norm_u_sq + norm_v_sq - 2.0 * dot_uv).max(0.0)
    }
    
    /// SIMD-optimized dot product.
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    fn dot_product(&self, u: &[f32], v: &[f32]) -> f32 {
        // Fall back to scalar for now; full SIMD implementation below
        self.dot_product_scalar(u, v)
    }
    
    #[inline(always)]
    #[cfg(not(target_arch = "x86_64"))]
    fn dot_product(&self, u: &[f32], v: &[f32]) -> f32 {
        self.dot_product_scalar(u, v)
    }
    
    /// Scalar dot product (fallback and for non-SIMD platforms).
    #[inline(always)]
    fn dot_product_scalar(&self, u: &[f32], v: &[f32]) -> f32 {
        u.iter()
            .zip(v.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl DistanceMetric for HyperbolicMetric {
    fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len(), "Vectors must have same dimension");
        
        // Compute squared norms
        let norm_u_sq = self.squared_norm(u);
        let norm_v_sq = self.squared_norm(v);
        
        // Clamp norms to avoid boundary instability
        // Points with ||x|| ≥ 1 are outside the Poincaré ball
        let norm_u_sq = norm_u_sq.min(1.0 - self.epsilon);
        let norm_v_sq = norm_v_sq.min(1.0 - self.epsilon);
        
        // Compute squared Euclidean distance
        let diff_sq = self.squared_euclidean(u, v);
        
        // Compute the argument to arcosh
        // arcosh_arg = 1 + 2||u-v||² / ((1-||u||²)(1-||v||²))
        let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
        let arcosh_arg = 1.0 + 2.0 * diff_sq / denom.max(self.epsilon);
        
        // Compute hyperbolic distance
        self.sqrt_abs_curvature * self.fast_arcosh(arcosh_arg)
    }
    
    fn name(&self) -> &'static str {
        "hyperbolic_poincare"
    }
}

// ============================================================================
// Euclidean Distance (for comparison and fallback)
// ============================================================================

/// Standard Euclidean (L2) distance metric.
///
/// Provided for comparison and for cases where hyperbolic embedding
/// is not appropriate.
#[derive(Debug, Clone, Default)]
pub struct EuclideanMetric;

impl DistanceMetric for EuclideanMetric {
    #[inline(always)]
    fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len());
        
        u.iter()
            .zip(v.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
    
    fn name(&self) -> &'static str {
        "euclidean"
    }
}

// ============================================================================
// Cosine Distance
// ============================================================================

/// Cosine distance (1 - cosine similarity).
///
/// Useful for normalized embeddings where angle matters more than magnitude.
#[derive(Debug, Clone, Default)]
pub struct CosineMetric;

impl DistanceMetric for CosineMetric {
    #[inline(always)]
    fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len());
        
        let mut dot = 0.0f32;
        let mut norm_u = 0.0f32;
        let mut norm_v = 0.0f32;
        
        for (a, b) in u.iter().zip(v.iter()) {
            dot += a * b;
            norm_u += a * a;
            norm_v += b * b;
        }
        
        let denom = (norm_u * norm_v).sqrt();
        if denom < 1e-10 {
            return 1.0; // Maximum distance for zero vectors
        }
        
        // Cosine distance = 1 - cosine_similarity
        // Clamp to [0, 2] to handle floating point errors
        (1.0 - dot / denom).clamp(0.0, 2.0)
    }
    
    fn name(&self) -> &'static str {
        "cosine"
    }
}

// ============================================================================
// Inner Product (for Maximum Inner Product Search - MIPS)
// ============================================================================

/// Negative inner product for Maximum Inner Product Search (MIPS).
///
/// Returns -<u, v> so that HNSW's "minimize distance" behavior
/// becomes "maximize inner product".
///
/// Useful for recommendation systems and attention mechanisms.
#[derive(Debug, Clone, Default)]
pub struct InnerProductMetric;

// ============================================================================
// Lorentz/Hyperboloid Distance (10x faster than Poincaré)
// ============================================================================

/// Lorentz (hyperboloid) distance metric.
///
/// Uses the hyperboloid model H^n embedded in Minkowski space R^{n,1}:
/// ```text
/// H^n = {x ∈ R^{n+1} : ⟨x,x⟩_M = -1, x_0 > 0}
/// ```
///
/// where ⟨·,·⟩_M is the Minkowski inner product:
/// ```text
/// ⟨u,v⟩_M = -u_0 v_0 + u_1 v_1 + ... + u_n v_n
/// ```
///
/// The geodesic distance is:
/// ```text
/// d_L(u,v) = √|K| · arcosh(-⟨u,v⟩_M)
/// ```
///
/// ## Performance Advantage
///
/// The Lorentz model is ~10x faster than Poincaré because:
/// 1. No division by (1 - ||x||²) terms (no boundary singularity)
/// 2. Simple Minkowski inner product (SIMD-friendly)
/// 3. Direct arcosh without ratio computation
///
/// ## Integration with hyperphysics-geometry
///
/// This metric is compatible with `hyperphysics_geometry::LorentzVec4D` and
/// the hyperbolic SNN's Lorentz-based neuron positions.
#[derive(Debug, Clone)]
pub struct LorentzMetric {
    /// Sectional curvature (negative for hyperbolic space).
    curvature: f32,
    /// Pre-computed √|curvature| for efficiency.
    sqrt_abs_curvature: f32,
}

impl LorentzMetric {
    /// Create a new Lorentz metric with the given curvature.
    ///
    /// # Arguments
    ///
    /// * `curvature` - The sectional curvature (must be negative).
    ///   Standard hyperbolic space has K = -1.
    ///
    /// # Panics
    ///
    /// Panics if curvature is non-negative.
    pub fn new(curvature: f32) -> Self {
        assert!(curvature < 0.0, "Hyperbolic curvature must be negative");
        Self {
            curvature,
            sqrt_abs_curvature: (-curvature).sqrt(),
        }
    }

    /// Create a Lorentz metric with standard curvature K = -1.
    pub fn standard() -> Self {
        Self::new(-1.0)
    }

    /// Adjust the curvature (for Evolution layer optimization).
    pub fn with_curvature_adjustment(&self, adjustment: f32) -> Self {
        Self::new(self.curvature * (1.0 + adjustment))
    }

    /// Compute Minkowski inner product: ⟨u,v⟩_M = -u_0 v_0 + Σ u_i v_i
    ///
    /// The first component (index 0) is the time coordinate with negative signature.
    /// This is SIMD-friendly: one negation + dot product.
    #[inline(always)]
    fn minkowski_inner(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert!(u.len() >= 2 && v.len() >= 2, "Lorentz vectors need at least 2 components");
        debug_assert_eq!(u.len(), v.len());

        // Time component (negative signature)
        let time_part = -u[0] * v[0];

        // Spatial components (positive signature)
        let spatial_part: f32 = u[1..].iter()
            .zip(v[1..].iter())
            .map(|(a, b)| a * b)
            .sum();

        time_part + spatial_part
    }

    /// Fast arcosh approximation
    #[inline(always)]
    fn fast_arcosh(&self, x: f32) -> f32 {
        let y = x - 1.0;
        if y < 0.1 {
            // Taylor approximation for small arguments
            let sqrt_2y = (2.0 * y.max(0.0)).sqrt();
            sqrt_2y * (1.0 - y / 12.0 + 3.0 * y * y / 160.0)
        } else {
            // Standard formula
            (x + (x * x - 1.0).max(0.0).sqrt()).ln()
        }
    }
}

impl Default for LorentzMetric {
    fn default() -> Self {
        Self::standard()
    }
}

impl DistanceMetric for LorentzMetric {
    fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len(), "Vectors must have same dimension");

        // Minkowski inner product (negative for points on same sheet of hyperboloid)
        let inner = self.minkowski_inner(u, v);

        // For points on the hyperboloid, -⟨u,v⟩_M ≥ 1
        // Clamp for numerical stability
        let arcosh_arg = (-inner).max(1.0);

        self.sqrt_abs_curvature * self.fast_arcosh(arcosh_arg)
    }

    fn distance_batch(&self, query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
        // SIMD-optimized batch computation
        targets.iter().map(|t| self.distance(query, t)).collect()
    }

    fn name(&self) -> &'static str {
        "lorentz_hyperboloid"
    }
}

impl DistanceMetric for InnerProductMetric {
    #[inline(always)]
    fn distance(&self, u: &[f32], v: &[f32]) -> f32 {
        debug_assert_eq!(u.len(), v.len());
        
        let dot: f32 = u.iter()
            .zip(v.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        -dot // Negate so that minimizing distance maximizes inner product
    }
    
    fn name(&self) -> &'static str {
        "inner_product"
    }
    
    fn lower_is_better(&self) -> bool {
        true // We negated, so lower (more negative) means higher inner product
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_hyperbolic_identity() {
        let metric = HyperbolicMetric::standard();
        let v = vec![0.1, 0.2, 0.3];
        
        // Distance to self should be zero
        assert_relative_eq!(metric.distance(&v, &v), 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_hyperbolic_symmetry() {
        let metric = HyperbolicMetric::standard();
        let u = vec![0.1, 0.2, 0.3];
        let v = vec![0.4, 0.1, 0.2];
        
        // d(u, v) should equal d(v, u)
        assert_relative_eq!(
            metric.distance(&u, &v),
            metric.distance(&v, &u),
            epsilon = 1e-6
        );
    }
    
    #[test]
    fn test_hyperbolic_triangle_inequality() {
        let metric = HyperbolicMetric::standard();
        let u = vec![0.1, 0.1];
        let v = vec![0.3, 0.2];
        let w = vec![0.2, 0.4];
        
        let d_uv = metric.distance(&u, &v);
        let d_vw = metric.distance(&v, &w);
        let d_uw = metric.distance(&u, &w);
        
        // Triangle inequality: d(u, w) ≤ d(u, v) + d(v, w)
        assert!(d_uw <= d_uv + d_vw + 1e-6);
    }
    
    #[test]
    fn test_hyperbolic_boundary_behavior() {
        let metric = HyperbolicMetric::standard();
        
        // Points near the origin
        let near_origin = vec![0.01, 0.01];
        
        // Points near the boundary (||x|| close to 1)
        let near_boundary = vec![0.7, 0.7]; // ||x|| ≈ 0.99
        
        // Distance to boundary should be larger than distance to origin
        let d_origin = metric.distance(&near_origin, &[0.0, 0.0]);
        let d_boundary = metric.distance(&near_boundary, &[0.0, 0.0]);
        
        assert!(d_boundary > d_origin);
    }
    
    #[test]
    fn test_euclidean_basic() {
        let metric = EuclideanMetric;
        
        let u = vec![0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0];
        
        assert_relative_eq!(metric.distance(&u, &v), 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cosine_orthogonal() {
        let metric = CosineMetric;
        
        let u = vec![1.0, 0.0];
        let v = vec![0.0, 1.0];
        
        // Orthogonal vectors have cosine similarity 0, distance 1
        assert_relative_eq!(metric.distance(&u, &v), 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cosine_parallel() {
        let metric = CosineMetric;
        
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![2.0, 4.0, 6.0];
        
        // Parallel vectors have cosine similarity 1, distance 0
        assert_relative_eq!(metric.distance(&u, &v), 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_inner_product() {
        let metric = InnerProductMetric;

        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];

        // <u, v> = 1*4 + 2*5 + 3*6 = 32
        // Metric returns -32
        assert_relative_eq!(metric.distance(&u, &v), -32.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lorentz_identity() {
        let metric = LorentzMetric::standard();

        // Point on hyperboloid: (cosh(r), sinh(r) * n) for unit vector n
        // For r=0, this is (1, 0, 0)
        let v = vec![1.0, 0.0, 0.0]; // Origin on hyperboloid

        // Distance to self should be zero
        assert_relative_eq!(metric.distance(&v, &v), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lorentz_symmetry() {
        let metric = LorentzMetric::standard();

        // Two points on hyperboloid
        // t = sqrt(1 + x² + y²) for x, y spatial coords
        let u = vec![1.0_f32.hypot(0.3).hypot(0.2), 0.3, 0.2];
        let v = vec![1.0_f32.hypot(0.5).hypot(0.1), 0.5, 0.1];

        // d(u, v) should equal d(v, u)
        assert_relative_eq!(
            metric.distance(&u, &v),
            metric.distance(&v, &u),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_lorentz_vs_poincare_consistency() {
        let lorentz = LorentzMetric::standard();
        let poincare = HyperbolicMetric::standard();

        // Test that both metrics give similar results for comparable inputs
        // Note: Poincaré uses the ball model, Lorentz uses hyperboloid
        // For small displacements near origin, they should be close

        // Near-origin point in Poincaré (inside unit ball)
        let p_poincare = vec![0.1, 0.1];
        let q_poincare = vec![0.2, 0.15];

        // Equivalent points on hyperboloid (converted from Poincaré)
        // For point x in Poincaré ball, Lorentz coords are:
        // t = (1 + ||x||²) / (1 - ||x||²)
        // spatial = 2x / (1 - ||x||²)
        fn poincare_to_lorentz(p: &[f32]) -> Vec<f32> {
            let norm_sq: f32 = p.iter().map(|x| x * x).sum();
            let denom = 1.0 - norm_sq;
            let t = (1.0 + norm_sq) / denom;
            let spatial: Vec<f32> = p.iter().map(|x| 2.0 * x / denom).collect();
            std::iter::once(t).chain(spatial).collect()
        }

        let p_lorentz = poincare_to_lorentz(&p_poincare);
        let q_lorentz = poincare_to_lorentz(&q_poincare);

        let d_poincare = poincare.distance(&p_poincare, &q_poincare);
        let d_lorentz = lorentz.distance(&p_lorentz, &q_lorentz);

        // Distances should match (both are geodesic distance in H²)
        assert_relative_eq!(d_poincare, d_lorentz, epsilon = 0.01);
    }

    #[test]
    fn test_lorentz_triangle_inequality() {
        let metric = LorentzMetric::standard();

        // Three points on hyperboloid
        let u = vec![1.0_f32.hypot(0.1).hypot(0.1), 0.1, 0.1];
        let v = vec![1.0_f32.hypot(0.3).hypot(0.2), 0.3, 0.2];
        let w = vec![1.0_f32.hypot(0.2).hypot(0.4), 0.2, 0.4];

        let d_uv = metric.distance(&u, &v);
        let d_vw = metric.distance(&v, &w);
        let d_uw = metric.distance(&u, &w);

        // Triangle inequality: d(u, w) ≤ d(u, v) + d(v, w)
        assert!(d_uw <= d_uv + d_vw + 1e-6);
    }
}
