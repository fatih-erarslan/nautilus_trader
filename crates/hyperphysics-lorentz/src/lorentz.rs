//! Lorentz/Hyperboloid model implementation
//!
//! Represents hyperbolic space as the upper sheet of a hyperboloid
//! in Minkowski space.

use crate::{LorentzError, Result, EPSILON, DEFAULT_CURVATURE};
use crate::minkowski::SimdMinkowski;
use serde::{Deserialize, Serialize};

/// Point on the Lorentz hyperboloid
///
/// Represents a point on H^n embedded in R^{n+1} Minkowski space.
/// The point satisfies ⟨x,x⟩_L = -1/|K| where K is the curvature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LorentzPoint {
    /// Coordinates in Minkowski space (time component at index 0)
    coords: Vec<f64>,
    /// Curvature of the hyperbolic space (negative)
    curvature: f64,
}

impl LorentzPoint {
    /// Create a new Lorentz point
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinates in Minkowski space (time at index 0)
    /// * `curvature` - Curvature K (must be negative)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Curvature is non-negative
    /// - Point is not on the hyperboloid
    /// - Time coordinate is not positive
    pub fn new(coords: Vec<f64>, curvature: f64) -> Result<Self> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }

        if coords.len() < 2 {
            return Err(LorentzError::EmptyInput(
                "Lorentz point must have at least 2 dimensions".to_string(),
            ));
        }

        if coords[0] <= 0.0 {
            return Err(LorentzError::NegativeTimeCoordinate(coords[0]));
        }

        let point = Self { coords, curvature };

        // Verify point is on hyperboloid
        if !point.is_on_hyperboloid(EPSILON * 100.0) {
            let norm = SimdMinkowski::norm_sq(&point.coords)?;
            let expected = -1.0 / curvature.abs();
            return Err(LorentzError::NotOnHyperboloid { norm, expected });
        }

        Ok(point)
    }

    /// Create from raw coordinates without validation
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - Point is on the hyperboloid
    /// - Time coordinate is positive
    /// - Curvature is negative
    pub fn new_unchecked(coords: Vec<f64>, curvature: f64) -> Self {
        Self { coords, curvature }
    }

    /// Create the origin point on the hyperboloid
    ///
    /// The origin is (1/√|K|, 0, 0, ..., 0) in Minkowski coordinates.
    pub fn origin(dim: usize, curvature: f64) -> Result<Self> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }

        let mut coords = vec![0.0; dim + 1];
        coords[0] = 1.0 / curvature.abs().sqrt();

        Ok(Self { coords, curvature })
    }

    /// Get coordinates
    #[inline]
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// Get time coordinate (x₀)
    #[inline]
    pub fn time_coord(&self) -> f64 {
        self.coords[0]
    }

    /// Get spatial coordinates (x₁, x₂, ..., xₙ)
    #[inline]
    pub fn spatial_coords(&self) -> &[f64] {
        &self.coords[1..]
    }

    /// Get curvature
    #[inline]
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Get dimension of hyperbolic space (n for H^n)
    #[inline]
    pub fn dim(&self) -> usize {
        self.coords.len() - 1
    }

    /// Check if point is on hyperboloid
    pub fn is_on_hyperboloid(&self, tolerance: f64) -> bool {
        SimdMinkowski::is_on_hyperboloid(&self.coords, self.curvature, tolerance)
            .unwrap_or(false)
    }

    /// Minkowski inner product with another point
    pub fn minkowski_dot(&self, other: &Self) -> Result<f64> {
        SimdMinkowski::dot(&self.coords, &other.coords)
    }

    /// Hyperbolic distance to another point
    pub fn distance(&self, other: &Self) -> Result<f64> {
        SimdMinkowski::distance(&self.coords, &other.coords, self.curvature)
    }

    /// Geodesic midpoint between two points
    pub fn midpoint(&self, other: &Self) -> Result<Self> {
        let inner = self.minkowski_dot(other)?;
        let k = self.curvature.abs();

        // Midpoint formula: m = (x + y) / √(-2⟨x,y⟩_L)
        let denom = (-2.0 * k * inner).sqrt();
        if denom < EPSILON {
            return Err(LorentzError::NumericalInstability(
                "Division by near-zero in midpoint calculation".to_string(),
            ));
        }

        let coords: Vec<f64> = self
            .coords
            .iter()
            .zip(other.coords.iter())
            .map(|(&a, &b)| (a + b) / denom)
            .collect();

        // Project back to hyperboloid to ensure numerical accuracy
        let projected = SimdMinkowski::project_to_hyperboloid(&coords, self.curvature)?;

        Ok(Self::new_unchecked(projected, self.curvature))
    }

    /// Exponential map: tangent space → hyperboloid
    pub fn exp(&self, tangent: &[f64]) -> Result<Self> {
        let coords = SimdMinkowski::exp_map(&self.coords, tangent, self.curvature)?;

        // Project to ensure on hyperboloid
        let projected = SimdMinkowski::project_to_hyperboloid(&coords, self.curvature)?;

        Ok(Self::new_unchecked(projected, self.curvature))
    }

    /// Logarithmic map: hyperboloid → tangent space
    pub fn log(&self, other: &Self) -> Result<Vec<f64>> {
        SimdMinkowski::log_map(&self.coords, &other.coords, self.curvature)
    }

    /// Parallel transport of tangent vector to another point
    pub fn parallel_transport(&self, to: &Self, tangent: &[f64]) -> Result<Vec<f64>> {
        SimdMinkowski::parallel_transport(&self.coords, &to.coords, tangent, self.curvature)
    }

    /// Project to nearest point on hyperboloid
    pub fn project(&mut self) -> Result<()> {
        self.coords = SimdMinkowski::project_to_hyperboloid(&self.coords, self.curvature)?;
        Ok(())
    }
}

/// Lorentz model operations
#[derive(Debug, Clone, Copy)]
pub struct LorentzModel {
    /// Curvature of hyperbolic space
    curvature: f64,
}

impl LorentzModel {
    /// Create new Lorentz model with given curvature
    pub fn new(curvature: f64) -> Result<Self> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }
        Ok(Self { curvature })
    }

    /// Create with default curvature (K = -1)
    pub fn default_curvature() -> Self {
        Self { curvature: DEFAULT_CURVATURE }
    }

    /// Get curvature
    #[inline]
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Create origin point
    pub fn origin(&self, dim: usize) -> Result<LorentzPoint> {
        LorentzPoint::origin(dim, self.curvature)
    }

    /// Create point from coordinates
    pub fn point(&self, coords: Vec<f64>) -> Result<LorentzPoint> {
        LorentzPoint::new(coords, self.curvature)
    }

    /// Create point from spatial coordinates (computes time coordinate)
    pub fn from_spatial(&self, spatial: &[f64]) -> Result<LorentzPoint> {
        let mut coords = vec![0.0; spatial.len() + 1];

        // Compute spatial norm squared
        let spatial_norm_sq: f64 = spatial.iter().map(|&x| x * x).sum();

        // Time coordinate: x₀ = √(1/|K| + ||x_spatial||²)
        let k = self.curvature.abs();
        coords[0] = (1.0 / k + spatial_norm_sq).sqrt();
        coords[1..].copy_from_slice(spatial);

        Ok(LorentzPoint::new_unchecked(coords, self.curvature))
    }

    /// Distance between two points
    pub fn distance(&self, a: &LorentzPoint, b: &LorentzPoint) -> Result<f64> {
        a.distance(b)
    }

    /// Geodesic interpolation between two points
    ///
    /// Returns point at parameter t ∈ [0, 1] along geodesic from a to b.
    pub fn geodesic(&self, a: &LorentzPoint, b: &LorentzPoint, t: f64) -> Result<LorentzPoint> {
        if t < 0.0 || t > 1.0 {
            return Err(LorentzError::NumericalInstability(
                format!("Geodesic parameter t must be in [0,1], got {}", t),
            ));
        }

        // Get tangent vector from a to b
        let tangent = a.log(b)?;

        // Scale by t
        let scaled_tangent: Vec<f64> = tangent.iter().map(|&v| v * t).collect();

        // Exponential map
        a.exp(&scaled_tangent)
    }

    /// Fréchet mean (centroid) of a set of points
    ///
    /// Computes the weighted mean using gradient descent.
    pub fn frechet_mean(
        &self,
        points: &[LorentzPoint],
        weights: Option<&[f64]>,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<LorentzPoint> {
        if points.is_empty() {
            return Err(LorentzError::EmptyInput("No points for Fréchet mean".to_string()));
        }

        let n = points.len();
        let dim = points[0].dim();

        // Default uniform weights
        let default_weights: Vec<f64> = vec![1.0 / n as f64; n];
        let w = weights.unwrap_or(&default_weights);

        if w.len() != n {
            return Err(LorentzError::DimensionMismatch {
                expected: n,
                actual: w.len(),
            });
        }

        // Start at first point
        let mut mean = points[0].clone();

        for _ in 0..max_iter {
            // Compute weighted sum of log maps
            let mut tangent_sum = vec![0.0; dim + 1];

            for (point, &weight) in points.iter().zip(w.iter()) {
                let tangent = mean.log(point)?;
                for (i, &t) in tangent.iter().enumerate() {
                    tangent_sum[i] += weight * t;
                }
            }

            // Check convergence
            let tangent_norm: f64 = tangent_sum.iter().map(|&x| x * x).sum();
            if tangent_norm < tolerance * tolerance {
                break;
            }

            // Move in direction of mean
            mean = mean.exp(&tangent_sum)?;
        }

        Ok(mean)
    }
}

impl Default for LorentzModel {
    fn default() -> Self {
        Self::default_curvature()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_origin() {
        let model = LorentzModel::default_curvature();
        let origin = model.origin(2).unwrap();

        assert!((origin.time_coord() - 1.0).abs() < TOL);
        assert!(origin.is_on_hyperboloid(TOL));
    }

    #[test]
    fn test_from_spatial() {
        let model = LorentzModel::default_curvature();
        let point = model.from_spatial(&[0.5, 0.5]).unwrap();

        assert!(point.is_on_hyperboloid(TOL));
        assert!(point.time_coord() > 1.0);
    }

    #[test]
    fn test_distance_to_self() {
        let model = LorentzModel::default_curvature();
        let point = model.from_spatial(&[0.3, 0.4]).unwrap();

        let dist = point.distance(&point).unwrap();
        // Distance to self should be very small
        assert!(dist < 1e-6, "Distance to self should be near zero, got {}", dist);
    }

    #[test]
    fn test_distance_symmetric() {
        let model = LorentzModel::default_curvature();
        let a = model.from_spatial(&[0.3, 0.4]).unwrap();
        let b = model.from_spatial(&[0.5, 0.1]).unwrap();

        let d1 = a.distance(&b).unwrap();
        let d2 = b.distance(&a).unwrap();

        assert!((d1 - d2).abs() < TOL);
    }

    #[test]
    fn test_midpoint() {
        let model = LorentzModel::default_curvature();
        let a = model.from_spatial(&[0.3, 0.4]).unwrap();
        let b = model.from_spatial(&[0.5, 0.1]).unwrap();

        let mid = a.midpoint(&b).unwrap();

        // Midpoint should be on hyperboloid (with tolerance for projection)
        assert!(mid.is_on_hyperboloid(1e-4));

        // Distances should be approximately equal (midpoint property)
        let d_am = a.distance(&mid).unwrap();
        let d_mb = mid.distance(&b).unwrap();

        // Allow 10% relative tolerance for midpoint distances
        let rel_diff = (d_am - d_mb).abs() / d_am.max(d_mb).max(1e-10);
        assert!(rel_diff < 0.1, "Midpoint distances differ: {} vs {}", d_am, d_mb);
    }

    #[test]
    fn test_geodesic_endpoints() {
        let model = LorentzModel::default_curvature();
        let a = model.from_spatial(&[0.2, 0.3]).unwrap();
        let b = model.from_spatial(&[0.6, 0.1]).unwrap();

        // At t=0, should be at a
        let g0 = model.geodesic(&a, &b, 0.0).unwrap();
        assert!(a.distance(&g0).unwrap() < TOL);

        // At t=1, should be at b
        let g1 = model.geodesic(&a, &b, 1.0).unwrap();
        assert!(b.distance(&g1).unwrap() < TOL * 10.0);
    }

    #[test]
    fn test_invalid_curvature() {
        let result = LorentzModel::new(1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_frechet_mean_single_point() {
        let model = LorentzModel::default_curvature();
        let point = model.from_spatial(&[0.3, 0.4]).unwrap();

        let mean = model.frechet_mean(&[point.clone()], None, 100, 1e-10).unwrap();

        assert!(point.distance(&mean).unwrap() < TOL * 10.0);
    }
}
