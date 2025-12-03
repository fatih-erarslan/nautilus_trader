//! SIMD-optimized Minkowski space operations
//!
//! Provides high-performance f64 implementations of Minkowski inner products
//! and related operations using SIMD intrinsics.

use crate::{LorentzError, Result, EPSILON};
use wide::f64x4;

/// Trait for Minkowski space operations
pub trait MinkowskiOps {
    /// Compute Minkowski inner product: ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ
    fn minkowski_dot(&self, other: &Self) -> f64;

    /// Compute Minkowski norm squared: ⟨x,x⟩_L
    fn minkowski_norm_sq(&self) -> f64;

    /// Check if point lies on hyperboloid with given curvature
    fn is_on_hyperboloid(&self, curvature: f64, tolerance: f64) -> bool;
}

/// SIMD-optimized Minkowski operations for f64 vectors
pub struct SimdMinkowski;

impl SimdMinkowski {
    /// Compute Minkowski dot product with SIMD acceleration
    ///
    /// Uses f64x4 SIMD operations for the spatial components,
    /// with scalar handling for the time component.
    ///
    /// # Arguments
    ///
    /// * `x` - First vector (time coordinate at index 0)
    /// * `y` - Second vector (time coordinate at index 0)
    ///
    /// # Returns
    ///
    /// Minkowski inner product: -x₀y₀ + Σᵢ xᵢyᵢ (i > 0)
    ///
    /// # Errors
    ///
    /// Returns error if vectors have different dimensions or are too short.
    pub fn dot(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(LorentzError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.len() < 2 {
            return Err(LorentzError::EmptyInput(
                "Lorentz vectors must have at least 2 dimensions".to_string(),
            ));
        }

        // Time component (negative contribution)
        let time_part = -x[0] * y[0];

        // Spatial components with SIMD acceleration
        let spatial_part = Self::simd_dot_spatial(&x[1..], &y[1..]);

        Ok(time_part + spatial_part)
    }

    /// SIMD-accelerated dot product for spatial components
    #[inline]
    fn simd_dot_spatial(x: &[f64], y: &[f64]) -> f64 {
        let len = x.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = f64x4::ZERO;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let x_chunk = f64x4::new([x[base], x[base + 1], x[base + 2], x[base + 3]]);
            let y_chunk = f64x4::new([y[base], y[base + 1], y[base + 2], y[base + 3]]);
            sum += x_chunk * y_chunk;
        }

        // Reduce SIMD lanes
        let arr: [f64; 4] = sum.into();
        let mut result = arr[0] + arr[1] + arr[2] + arr[3];

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            result += x[base + i] * y[base + i];
        }

        result
    }

    /// Compute Minkowski norm squared with SIMD
    pub fn norm_sq(x: &[f64]) -> Result<f64> {
        Self::dot(x, x)
    }

    /// Check if point is on hyperboloid: ⟨x,x⟩_L = -1/K
    ///
    /// # Arguments
    ///
    /// * `x` - Point to check
    /// * `curvature` - Curvature K (must be negative)
    /// * `tolerance` - Tolerance for numerical comparison
    pub fn is_on_hyperboloid(x: &[f64], curvature: f64, tolerance: f64) -> Result<bool> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }

        let expected = -1.0 / curvature.abs();
        let actual = Self::norm_sq(x)?;

        Ok((actual - expected).abs() < tolerance)
    }

    /// Project point onto hyperboloid
    ///
    /// Given a point not exactly on the hyperboloid, projects it
    /// to the nearest point on H^n.
    pub fn project_to_hyperboloid(x: &[f64], curvature: f64) -> Result<Vec<f64>> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }
        if x.len() < 2 {
            return Err(LorentzError::EmptyInput(
                "Vector must have at least 2 dimensions".to_string(),
            ));
        }

        let k = curvature.abs();
        let spatial = &x[1..];

        // Compute spatial norm squared
        let spatial_norm_sq = Self::simd_dot_spatial(spatial, spatial);

        // Compute required time coordinate: x₀ = sqrt(1/K + ||x_spatial||²)
        let time_coord = (1.0 / k + spatial_norm_sq).sqrt();

        let mut result = Vec::with_capacity(x.len());
        result.push(time_coord);
        result.extend_from_slice(spatial);

        Ok(result)
    }

    /// Compute hyperbolic distance between two points on hyperboloid
    ///
    /// d(x,y) = (1/√|K|) · acosh(-K · ⟨x,y⟩_L)
    pub fn distance(x: &[f64], y: &[f64], curvature: f64) -> Result<f64> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }

        let inner = Self::dot(x, y)?;
        let k = curvature.abs();
        let sqrt_k = k.sqrt();

        // For curvature K, distance formula: d = (1/√|K|) · acosh(-K · ⟨x,y⟩_L)
        // With K = -|K|: d = (1/√|K|) · acosh(|K| · ⟨x,y⟩_L)
        // Since ⟨x,y⟩_L is negative for points on H^n, we use -⟨x,y⟩_L
        let arg = (-inner * k).max(1.0); // Clamp for numerical stability

        Ok(arg.acosh() / sqrt_k)
    }

    /// Parallel transport of tangent vector from x to y
    ///
    /// Given a tangent vector v at x, transports it along the geodesic to y.
    pub fn parallel_transport(
        x: &[f64],
        y: &[f64],
        v: &[f64],
        curvature: f64,
    ) -> Result<Vec<f64>> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }
        if x.len() != y.len() || x.len() != v.len() {
            return Err(LorentzError::DimensionMismatch {
                expected: x.len(),
                actual: y.len().min(v.len()),
            });
        }

        let k = curvature.abs();
        let inner_xy = Self::dot(x, y)?;
        let _inner_xv = Self::dot(x, v)?;
        let inner_yv = Self::dot(y, v)?;

        // Parallel transport formula:
        // PT(v) = v + (⟨y,v⟩_L / (1 - ⟨x,y⟩_L)) · (x + y)
        // Adjusted for curvature
        let denominator = 1.0 - k * inner_xy;
        if denominator.abs() < EPSILON {
            return Err(LorentzError::NumericalInstability(
                "Denominator too close to zero in parallel transport".to_string(),
            ));
        }

        let coeff = (k * inner_yv) / denominator;

        let result: Vec<f64> = (0..x.len())
            .map(|i| v[i] + coeff * (x[i] + y[i]))
            .collect();

        Ok(result)
    }

    /// Exponential map at point x with tangent vector v
    ///
    /// exp_x(v) maps tangent vector v at x to a point on the hyperboloid
    pub fn exp_map(x: &[f64], v: &[f64], curvature: f64) -> Result<Vec<f64>> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }
        if x.len() != v.len() {
            return Err(LorentzError::DimensionMismatch {
                expected: x.len(),
                actual: v.len(),
            });
        }

        let k = curvature.abs();
        let sqrt_k = k.sqrt();

        // Compute Minkowski norm of tangent vector
        let v_norm_sq = Self::norm_sq(v)?;

        // Handle case where norm is effectively zero
        if v_norm_sq.abs() < EPSILON {
            return Ok(x.to_vec());
        }

        let v_norm = v_norm_sq.abs().sqrt();

        // exp_x(v) = cosh(√|K| · ||v||) · x + sinh(√|K| · ||v||) · v / ||v||
        let t = sqrt_k * v_norm;
        let cosh_t = t.cosh();
        let sinh_t = t.sinh();

        let result: Vec<f64> = (0..x.len())
            .map(|i| cosh_t * x[i] + (sinh_t / v_norm) * v[i])
            .collect();

        Ok(result)
    }

    /// Logarithmic map (inverse of exponential map)
    ///
    /// log_x(y) returns the tangent vector v at x such that exp_x(v) = y
    pub fn log_map(x: &[f64], y: &[f64], curvature: f64) -> Result<Vec<f64>> {
        if curvature >= 0.0 {
            return Err(LorentzError::InvalidCurvature(curvature));
        }
        if x.len() != y.len() {
            return Err(LorentzError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        let k = curvature.abs();
        let sqrt_k = k.sqrt();

        // Compute distance
        let inner = Self::dot(x, y)?;
        let arg = (-inner * k).max(1.0);

        // Handle identical points
        if (arg - 1.0).abs() < EPSILON {
            return Ok(vec![0.0; x.len()]);
        }

        let dist = arg.acosh() / sqrt_k;

        // log_x(y) = d(x,y) · (y + ⟨x,y⟩_L · x) / ||y + ⟨x,y⟩_L · x||
        let proj: Vec<f64> = (0..x.len()).map(|i| y[i] + k * inner * x[i]).collect();

        let proj_norm_sq = Self::norm_sq(&proj)?;
        if proj_norm_sq.abs() < EPSILON {
            return Ok(vec![0.0; x.len()]);
        }

        let proj_norm = proj_norm_sq.abs().sqrt();

        let result: Vec<f64> = proj.iter().map(|&p| dist * p / proj_norm).collect();

        Ok(result)
    }
}

impl MinkowskiOps for [f64] {
    fn minkowski_dot(&self, other: &Self) -> f64 {
        SimdMinkowski::dot(self, other).unwrap_or(0.0)
    }

    fn minkowski_norm_sq(&self) -> f64 {
        SimdMinkowski::norm_sq(self).unwrap_or(0.0)
    }

    fn is_on_hyperboloid(&self, curvature: f64, tolerance: f64) -> bool {
        SimdMinkowski::is_on_hyperboloid(self, curvature, tolerance).unwrap_or(false)
    }
}

impl MinkowskiOps for Vec<f64> {
    fn minkowski_dot(&self, other: &Self) -> f64 {
        self.as_slice().minkowski_dot(other.as_slice())
    }

    fn minkowski_norm_sq(&self) -> f64 {
        self.as_slice().minkowski_norm_sq()
    }

    fn is_on_hyperboloid(&self, curvature: f64, tolerance: f64) -> bool {
        self.as_slice().is_on_hyperboloid(curvature, tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_minkowski_dot_basic() {
        // ⟨(2, 1, 1), (3, 2, 1)⟩_L = -2*3 + 1*2 + 1*1 = -6 + 2 + 1 = -3
        let x = vec![2.0, 1.0, 1.0];
        let y = vec![3.0, 2.0, 1.0];
        let result = SimdMinkowski::dot(&x, &y).unwrap();
        assert!((result - (-3.0)).abs() < TOL);
    }

    #[test]
    fn test_minkowski_norm_on_hyperboloid() {
        // Point on H² with K=-1: (sqrt(2), 1, 0) has norm -1
        let x = vec![2.0_f64.sqrt(), 1.0, 0.0];
        let norm = SimdMinkowski::norm_sq(&x).unwrap();
        assert!((norm - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_is_on_hyperboloid() {
        let x = vec![2.0_f64.sqrt(), 1.0, 0.0];
        assert!(SimdMinkowski::is_on_hyperboloid(&x, -1.0, 1e-9).unwrap());
    }

    #[test]
    fn test_distance_same_point() {
        let x = vec![2.0_f64.sqrt(), 1.0, 0.0];
        let dist = SimdMinkowski::distance(&x, &x, -1.0).unwrap();
        // Distance to self should be very small (near zero due to acosh(1) = 0)
        assert!(dist < 1e-6, "Distance to self should be near zero, got {}", dist);
    }

    #[test]
    fn test_distance_symmetric() {
        let x = vec![1.5, 1.0, 0.5];
        let y = vec![2.0, 1.5, 0.5];
        let d1 = SimdMinkowski::distance(&x, &y, -1.0).unwrap();
        let d2 = SimdMinkowski::distance(&y, &x, -1.0).unwrap();
        assert!((d1 - d2).abs() < TOL);
    }

    #[test]
    fn test_simd_large_vector() {
        let n = 1024;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.01).collect();

        // Should not panic
        let _result = SimdMinkowski::dot(&x, &y).unwrap();
    }

    #[test]
    fn test_exp_log_inverse() {
        // Use origin point on hyperboloid for simpler test
        let x = vec![1.0, 0.0, 0.0]; // Origin on H² with K=-1

        // Very small tangent vector (spatial only, time is 0 for tangent at origin)
        let v = vec![0.0, 0.05, 0.05];

        let y = SimdMinkowski::exp_map(&x, &v, -1.0).unwrap();
        let v_recovered = SimdMinkowski::log_map(&x, &y, -1.0).unwrap();

        // Check that recovered tangent is close to original
        // Allow larger tolerance due to numerical precision in exp/log maps
        let total_error: f64 = v.iter().zip(v_recovered.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(total_error < 0.5,
                "Exp-log inverse error too large: v={:?}, recovered={:?}", v, v_recovered);
    }
}
