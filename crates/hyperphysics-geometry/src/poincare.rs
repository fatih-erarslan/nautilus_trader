//! Poincaré disk model for hyperbolic 3-space
//!
//! Research: Cannon et al. (1997) "Hyperbolic Geometry"

use nalgebra as na;
use serde::{Deserialize, Serialize};
use crate::{GeometryError, Result, EPSILON};

/// Point in the Poincaré disk D³
///
/// Invariant: ||coords|| < 1 (enforced at construction)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoincarePoint {
    coords: na::Vector3<f64>,
}

impl PoincarePoint {
    /// Create new point in Poincaré disk
    ///
    /// # Errors
    ///
    /// Returns `GeometryError::OutsideDisk` if ||coords|| >= 1
    pub fn new(coords: na::Vector3<f64>) -> Result<Self> {
        let norm = coords.norm();
        if norm >= 1.0 {
            return Err(GeometryError::OutsideDisk { norm });
        }
        Ok(Self { coords })
    }

    /// Create point from spherical coordinates
    ///
    /// # Arguments
    ///
    /// * `r` - Radial coordinate in [0, 1)
    /// * `theta` - Azimuthal angle in [0, 2π]
    /// * `phi` - Polar angle in [0, π]
    ///
    /// # Errors
    ///
    /// Returns error if r >= 1
    pub fn from_spherical(r: f64, theta: f64, phi: f64) -> Result<Self> {
        if r < 0.0 || r >= 1.0 {
            return Err(GeometryError::OutsideDisk { norm: r });
        }

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.sin() * theta.sin();
        let z = r * phi.cos();

        Self::new(na::Vector3::new(x, y, z))
    }

    /// Origin of Poincaré disk
    #[inline]
    pub fn origin() -> Self {
        Self {
            coords: na::Vector3::zeros(),
        }
    }

    /// Get Cartesian coordinates
    #[inline]
    pub fn coords(&self) -> na::Vector3<f64> {
        self.coords
    }

    /// Get norm squared (for efficiency)
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.coords.norm_squared()
    }

    /// Get norm
    #[inline]
    pub fn norm(&self) -> f64 {
        self.coords.norm()
    }

    /// Hyperbolic distance to another point
    ///
    /// Formula: d_H(p,q) = acosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))
    ///
    /// Uses Taylor expansion for numerical stability when points are close,
    /// and log1p for precision when argument is close to 1.
    ///
    /// # Mathematical Foundation
    ///
    /// For very small distances, uses Taylor series:
    /// d_H ≈ 2||p-q|| / sqrt((1-||p||²)(1-||q||²))
    ///
    /// For argument close to 1, uses:
    /// acosh(1 + ε) ≈ sqrt(2ε) for small ε
    pub fn hyperbolic_distance(&self, other: &Self) -> f64 {
        let p_norm_sq = self.norm_squared();
        let q_norm_sq = other.norm_squared();
        let diff = self.coords - other.coords;
        let diff_norm_sq = diff.norm_squared();

        // Multi-precision handling for numerical stability
        // Case 1: Identical or nearly identical points
        if diff_norm_sq < EPSILON.sqrt() {
            // Taylor expansion for very small distances
            // d_H ≈ 2||p-q|| / sqrt((1-||p||²)(1-||q||²))
            return 2.0 * diff_norm_sq.sqrt()
                / ((1.0 - p_norm_sq) * (1.0 - q_norm_sq)).sqrt();
        }

        // Case 2: Points near boundary require extra care
        let denominator = (1.0 - p_norm_sq) * (1.0 - q_norm_sq);
        if denominator < EPSILON {
            // Near boundary: return large but finite distance
            return 100.0; // Practical cutoff for numerical stability
        }

        let numerator = 2.0 * diff_norm_sq;
        let ratio = numerator / denominator;

        // Case 3: Small ratio (argument close to 1)
        // Use acosh(1 + x) = sqrt(2x) + O(x^(3/2)) for better precision
        if ratio < 0.01 {
            return (2.0 * ratio).sqrt();
        }

        // Case 4: General case - use standard acosh
        let argument = 1.0 + ratio;

        // acosh(x) = ln(x + sqrt(x² - 1))
        // For x close to 1, use log1p for better precision
        if (argument - 1.0).abs() < 0.1 {
            // log1p(x) gives better precision than log(1 + x)
            let sqrt_term = (argument * argument - 1.0).sqrt();
            (argument + sqrt_term - 1.0).ln_1p()
        } else {
            argument.acosh()
        }
    }

    /// Möbius addition (hyperbolic translation)
    ///
    /// Research: Ungar (2001) "Hyperbolic Trigonometry and Its Application"
    pub fn mobius_add(&self, other: &Self) -> Result<Self> {
        let p = self.coords;
        let q = other.coords;
        let p_norm_sq = p.norm_squared();
        let q_norm_sq = q.norm_squared();
        let p_dot_q = p.dot(&q);

        let numerator = (1.0 + 2.0 * p_dot_q + q_norm_sq) * p
                      + (1.0 - p_norm_sq) * q;
        let denominator = 1.0 + 2.0 * p_dot_q + p_norm_sq * q_norm_sq;

        let result = numerator / denominator;
        Self::new(result)
    }

    /// Project onto Poincaré disk if slightly outside (numerical errors)
    pub fn project_to_disk(&mut self) {
        let norm = self.norm();
        if norm >= 0.99 {
            self.coords *= 0.98 / norm;
        }
    }

    /// Conformal factor at this point
    ///
    /// λ(x) = 2 / (1 - ||x||²)
    #[inline]
    pub fn conformal_factor(&self) -> f64 {
        2.0 / (1.0 - self.norm_squared())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_origin() {
        let origin = PoincarePoint::origin();
        assert_eq!(origin.norm(), 0.0);
    }

    #[test]
    fn test_outside_disk() {
        let result = PoincarePoint::new(na::Vector3::new(1.0, 0.0, 0.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_spherical_coordinates() {
        let p = PoincarePoint::from_spherical(0.5, 0.0, PI / 2.0).unwrap();
        assert!((p.coords().x - 0.5).abs() < 1e-10);
        assert!(p.coords().y.abs() < 1e-10);
        assert!(p.coords().z.abs() < 1e-10);
    }

    #[test]
    fn test_distance_to_self() {
        let p = PoincarePoint::new(na::Vector3::new(0.5, 0.0, 0.0)).unwrap();
        assert_eq!(p.hyperbolic_distance(&p), 0.0);
    }

    #[test]
    fn test_distance_symmetry() {
        let p = PoincarePoint::new(na::Vector3::new(0.3, 0.0, 0.0)).unwrap();
        let q = PoincarePoint::new(na::Vector3::new(0.0, 0.5, 0.0)).unwrap();

        let d_pq = p.hyperbolic_distance(&q);
        let d_qp = q.hyperbolic_distance(&p);

        assert!((d_pq - d_qp).abs() < EPSILON);
    }

    #[test]
    fn test_triangle_inequality() {
        let p = PoincarePoint::new(na::Vector3::new(0.2, 0.0, 0.0)).unwrap();
        let q = PoincarePoint::new(na::Vector3::new(0.0, 0.3, 0.0)).unwrap();
        let r = PoincarePoint::new(na::Vector3::new(0.0, 0.0, 0.4)).unwrap();

        let d_pq = p.hyperbolic_distance(&q);
        let d_qr = q.hyperbolic_distance(&r);
        let d_pr = p.hyperbolic_distance(&r);

        assert!(d_pr <= d_pq + d_qr + EPSILON,
                "Triangle inequality violated: {} > {} + {}", d_pr, d_pq, d_qr);
    }

    #[test]
    fn test_mobius_identity() {
        let p = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.1)).unwrap();
        let origin = PoincarePoint::origin();

        let result = p.mobius_add(&origin).unwrap();
        assert!((result.coords() - p.coords()).norm() < EPSILON);
    }

    #[test]
    fn test_conformal_factor() {
        let origin = PoincarePoint::origin();
        assert_eq!(origin.conformal_factor(), 2.0);

        let p = PoincarePoint::new(na::Vector3::new(0.0, 0.0, 0.0)).unwrap();
        assert_eq!(p.conformal_factor(), 2.0);
    }
}
