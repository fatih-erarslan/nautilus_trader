//! Curvature tensor computations for hyperbolic space
//!
//! Research: do Carmo (1992) "Riemannian Geometry"

use nalgebra as na;
use crate::poincare::PoincarePoint;

/// Riemann curvature tensor for hyperbolic 3-space
///
/// For constant curvature K=-1: R_ijkl = K(g_ik g_jl - g_il g_jk)
pub struct CurvatureTensor;

impl CurvatureTensor {
    /// Ricci scalar curvature at a point
    ///
    /// For H³ with K=-1: R = 6K = -6
    pub fn ricci_scalar(_point: &PoincarePoint) -> f64 {
        -6.0 // Constant for H³
    }

    /// Sectional curvature in plane spanned by tangent vectors
    ///
    /// For H³: K = -1 (constant)
    pub fn sectional_curvature(
        _point: &PoincarePoint,
        _tangent1: &na::Vector3<f64>,
        _tangent2: &na::Vector3<f64>,
    ) -> f64 {
        -1.0 // Constant negative curvature
    }

    /// Ricci curvature tensor component
    ///
    /// For constant curvature: Ric_ij = (n-1)K g_ij where n=3
    /// So Ric_ij = -2 g_ij
    pub fn ricci_tensor(point: &PoincarePoint) -> na::Matrix3<f64> {
        let g = Self::metric_tensor(point);
        -2.0 * g
    }

    /// Metric tensor in Poincaré disk coordinates
    ///
    /// g_ij = λ²δ_ij where λ = 2/(1-||x||²)
    pub fn metric_tensor(point: &PoincarePoint) -> na::Matrix3<f64> {
        let lambda = point.conformal_factor();
        let lambda_sq = lambda * lambda;

        na::Matrix3::from_diagonal(&na::Vector3::new(
            lambda_sq,
            lambda_sq,
            lambda_sq,
        ))
    }

    /// Christoffel symbols of the second kind
    ///
    /// Γⁱⱼₖ for Poincaré disk model
    pub fn christoffel_symbols(point: &PoincarePoint) -> [na::Matrix3<f64>; 3] {
        let coords = point.coords();
        let norm_sq = coords.norm_squared();
        let factor = 2.0 / (1.0 - norm_sq);

        let mut symbols = [na::Matrix3::zeros(); 3];

        // Simplified: Γⁱⱼₖ = λ⁻¹ ∂λ/∂xᵏ δⁱⱼ + (symmetric terms)
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    if i == k {
                        symbols[i][(j, k)] = factor * coords[j];
                    }
                    if j == k {
                        symbols[i][(j, k)] += factor * coords[i];
                    }
                    if i == j {
                        symbols[i][(j, k)] -= factor * coords[k];
                    }
                }
            }
        }

        symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_curvature() {
        let point = PoincarePoint::new(na::Vector3::new(0.5, 0.3, 0.2)).unwrap();

        let k = CurvatureTensor::sectional_curvature(
            &point,
            &na::Vector3::new(1.0, 0.0, 0.0),
            &na::Vector3::new(0.0, 1.0, 0.0),
        );

        assert_eq!(k, -1.0, "Sectional curvature must be K=-1");
    }

    #[test]
    fn test_ricci_scalar() {
        let point = PoincarePoint::origin();
        let r = CurvatureTensor::ricci_scalar(&point);

        assert_eq!(r, -6.0, "Ricci scalar must be -6 for H³");
    }

    #[test]
    fn test_metric_positive_definite() {
        let point = PoincarePoint::new(na::Vector3::new(0.4, 0.3, 0.2)).unwrap();
        let g = CurvatureTensor::metric_tensor(&point);

        // Check diagonal entries are positive
        for i in 0..3 {
            assert!(g[(i, i)] > 0.0, "Metric must be positive definite");
        }
    }
}
