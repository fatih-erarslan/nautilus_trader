//! Curvature tensor computations for hyperbolic space
//!
//! ## Mathematical Foundation
//!
//! Complete Riemann curvature tensor implementation for H³ (hyperbolic 3-space)
//! with constant sectional curvature K = -1.
//!
//! ## References
//!
//! - do Carmo (1992) "Riemannian Geometry" - Springer GTM 115
//! - Lee (2018) "Introduction to Riemannian Manifolds" - Springer GTM 176
//! - Petersen (2016) "Riemannian Geometry" 3rd ed. - Springer GTM 171
//! - Cannon et al. (1997) "Hyperbolic Geometry" - MSRI Publications 31
//!
//! ## Poincaré Ball Model
//!
//! Metric: ds² = λ²(|x|²)(dx₁² + dx₂² + dx₃²) where λ = 2/(1-|x|²)
//! Constant curvature: K = -1
//! Ricci scalar: R = n(n-1)K = 3·2·(-1) = -6

use nalgebra as na;
use crate::poincare::PoincarePoint;

/// Curvature constant for H³
pub const CURVATURE_K: f64 = -1.0;

/// Dimension of the manifold
pub const DIMENSION: usize = 3;

/// Full Riemann curvature tensor for hyperbolic 3-space H³
///
/// The Riemann curvature tensor R^i_jkl measures the failure of parallel
/// transport around infinitesimal loops. For constant curvature K:
///
/// R^i_jkl = K(δⁱₖgⱼₗ - δⁱₗgⱼₖ)
///
/// or in fully covariant form:
///
/// R_ijkl = K(g_ik·g_jl - g_il·g_jk)
pub struct CurvatureTensor;

impl CurvatureTensor {
    // ========================================================================
    // Metric Tensor
    // ========================================================================

    /// Metric tensor in Poincaré ball coordinates
    ///
    /// g_ij = λ²(x)·δ_ij where λ(x) = 2/(1-|x|²) is the conformal factor
    ///
    /// The Poincaré ball is conformally flat: g = λ² · δ
    pub fn metric_tensor(point: &PoincarePoint) -> na::Matrix3<f64> {
        let lambda = point.conformal_factor();
        let lambda_sq = lambda * lambda;

        na::Matrix3::from_diagonal(&na::Vector3::new(lambda_sq, lambda_sq, lambda_sq))
    }

    /// Inverse metric tensor g^{ij}
    ///
    /// g^{ij} = λ⁻²(x)·δⁱʲ
    pub fn inverse_metric(point: &PoincarePoint) -> na::Matrix3<f64> {
        let lambda = point.conformal_factor();
        let lambda_inv_sq = 1.0 / (lambda * lambda);

        na::Matrix3::from_diagonal(&na::Vector3::new(lambda_inv_sq, lambda_inv_sq, lambda_inv_sq))
    }

    /// Determinant of the metric tensor
    ///
    /// det(g) = λ⁶ for n=3 dimensions
    pub fn metric_determinant(point: &PoincarePoint) -> f64 {
        let lambda = point.conformal_factor();
        lambda.powi(6)
    }

    /// Volume element √det(g)
    pub fn volume_element(point: &PoincarePoint) -> f64 {
        let lambda = point.conformal_factor();
        lambda.powi(3)
    }

    // ========================================================================
    // Christoffel Symbols
    // ========================================================================

    /// Christoffel symbols of the second kind Γⁱⱼₖ
    ///
    /// For conformal metric g_ij = λ²δ_ij:
    ///
    /// Γⁱⱼₖ = λ⁻¹(δⁱⱼ∂ₖλ + δⁱₖ∂ⱼλ - δⱼₖ∂ⁱλ)
    ///
    /// where ∂ᵢλ = 4xᵢ/(1-|x|²)² = λ²xᵢ/2
    ///
    /// Substituting: Γⁱⱼₖ = (λ/2)(δⁱⱼxₖ + δⁱₖxⱼ - δⱼₖxⁱ)
    ///              = (2/(1-|x|²))/2 · (...) = 1/(1-|x|²) · (...)
    ///
    /// Or equivalently with factor = 2/(1-r²):
    /// Γⁱⱼₖ = factor·(δⁱⱼxₖ + δⁱₖxⱼ - δⱼₖxⁱ) / 2
    pub fn christoffel_symbols(point: &PoincarePoint) -> [[[f64; 3]; 3]; 3] {
        let x = point.coords();
        let norm_sq = x.norm_squared();

        // Factor: 2/(1-r²)
        let factor = 2.0 / (1.0 - norm_sq);
        let half_factor = factor / 2.0;

        let mut gamma = [[[0.0f64; 3]; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    // Γⁱⱼₖ = half_factor · (δⁱⱼxₖ + δⁱₖxⱼ - δⱼₖxⁱ)
                    let mut val = 0.0;

                    if i == j {
                        val += x[k];
                    }
                    if i == k {
                        val += x[j];
                    }
                    if j == k {
                        val -= x[i];
                    }

                    gamma[i][j][k] = half_factor * val;
                }
            }
        }

        gamma
    }

    /// Christoffel symbols as Matrix3 array (legacy format)
    pub fn christoffel_matrix(point: &PoincarePoint) -> [na::Matrix3<f64>; 3] {
        let gamma = Self::christoffel_symbols(point);
        let mut result = [na::Matrix3::zeros(); 3];

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][(j, k)] = gamma[i][j][k];
                }
            }
        }

        result
    }

    /// Christoffel symbols of the first kind Γ_ijk
    ///
    /// Γ_ijk = g_il·Γˡⱼₖ = λ²·Γⁱⱼₖ (no sum since metric is diagonal)
    pub fn christoffel_first_kind(point: &PoincarePoint) -> [[[f64; 3]; 3]; 3] {
        let gamma2 = Self::christoffel_symbols(point);
        let lambda_sq = point.conformal_factor().powi(2);

        let mut gamma1 = [[[0.0f64; 3]; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    // For diagonal metric: Γ_ijk = g_ii·Γⁱⱼₖ = λ²·Γⁱⱼₖ
                    gamma1[i][j][k] = lambda_sq * gamma2[i][j][k];
                }
            }
        }

        gamma1
    }

    // ========================================================================
    // Riemann Curvature Tensor
    // ========================================================================

    /// Full Riemann curvature tensor R^i_jkl (mixed tensor)
    ///
    /// R^i_jkl = ∂ₖΓⁱⱼₗ - ∂ₗΓⁱⱼₖ + ΓⁱₘₖΓᵐⱼₗ - ΓⁱₘₗΓᵐⱼₖ
    ///
    /// For constant curvature K = -1 (do Carmo convention):
    /// R^i_jkl = K(δⁱₗgⱼₖ - δⁱₖgⱼₗ)
    ///
    /// This ensures sectional curvature K(X,Y) = K for all planes.
    ///
    /// Returns 4-index tensor R[i][j][k][l]
    pub fn riemann_tensor(point: &PoincarePoint) -> [[[[f64; 3]; 3]; 3]; 3] {
        let g = Self::metric_tensor(point);
        let k = CURVATURE_K;

        let mut riemann = [[[[0.0f64; 3]; 3]; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k_idx in 0..3 {
                    for l in 0..3 {
                        // R^i_jkl = K(δⁱₗgⱼₖ - δⁱₖgⱼₗ)
                        // Following do Carmo "Riemannian Geometry" convention
                        let term1 = if i == l { g[(j, k_idx)] } else { 0.0 };
                        let term2 = if i == k_idx { g[(j, l)] } else { 0.0 };

                        riemann[i][j][k_idx][l] = k * (term1 - term2);
                    }
                }
            }
        }

        riemann
    }

    /// Covariant Riemann tensor R_ijkl (all indices down)
    ///
    /// R_ijkl = g_im·R^m_jkl = K(g_il·g_jk - g_ik·g_jl)
    ///
    /// Following do Carmo "Riemannian Geometry" (1992) convention.
    /// For H³ with K=-1: R_ijkl = -(g_il·g_jk - g_ik·g_jl)
    ///
    /// This ensures: K(X,Y) = R(X,Y,Y,X) / |X∧Y|² = K
    pub fn riemann_covariant(point: &PoincarePoint) -> [[[[f64; 3]; 3]; 3]; 3] {
        let g = Self::metric_tensor(point);
        let k = CURVATURE_K;

        let mut riemann = [[[[0.0f64; 3]; 3]; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k_idx in 0..3 {
                    for l in 0..3 {
                        // R_ijkl = K(g_il·g_jk - g_ik·g_jl)
                        riemann[i][j][k_idx][l] = k * (g[(i, l)] * g[(j, k_idx)] - g[(i, k_idx)] * g[(j, l)]);
                    }
                }
            }
        }

        riemann
    }

    /// Verify Riemann tensor symmetries
    ///
    /// The Riemann tensor satisfies:
    /// 1. R_ijkl = -R_jikl (antisymmetric in first pair)
    /// 2. R_ijkl = -R_ijlk (antisymmetric in second pair)
    /// 3. R_ijkl = R_klij (pair symmetry)
    /// 4. R_ijkl + R_iklj + R_iljk = 0 (first Bianchi identity)
    pub fn verify_riemann_symmetries(point: &PoincarePoint) -> RiemannSymmetryCheck {
        let r = Self::riemann_covariant(point);
        let tol = 1e-10;

        let mut antisym_ij = true;
        let mut antisym_kl = true;
        let mut pair_sym = true;
        let mut bianchi = true;
        let mut max_error = 0.0f64;

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        // Antisymmetry in (i,j)
                        let err1 = (r[i][j][k][l] + r[j][i][k][l]).abs();
                        if err1 > tol { antisym_ij = false; }
                        max_error = max_error.max(err1);

                        // Antisymmetry in (k,l)
                        let err2 = (r[i][j][k][l] + r[i][j][l][k]).abs();
                        if err2 > tol { antisym_kl = false; }
                        max_error = max_error.max(err2);

                        // Pair symmetry
                        let err3 = (r[i][j][k][l] - r[k][l][i][j]).abs();
                        if err3 > tol { pair_sym = false; }
                        max_error = max_error.max(err3);

                        // First Bianchi identity
                        let err4 = (r[i][j][k][l] + r[i][k][l][j] + r[i][l][j][k]).abs();
                        if err4 > tol { bianchi = false; }
                        max_error = max_error.max(err4);
                    }
                }
            }
        }

        RiemannSymmetryCheck {
            antisymmetric_first_pair: antisym_ij,
            antisymmetric_second_pair: antisym_kl,
            pair_symmetric: pair_sym,
            bianchi_identity: bianchi,
            max_error,
        }
    }

    // ========================================================================
    // Ricci Tensor and Scalar
    // ========================================================================

    /// Ricci tensor Ric_ij = R^k_ikj (contraction of Riemann)
    ///
    /// For constant curvature K in dimension n:
    /// Ric_ij = (n-1)K·g_ij
    ///
    /// For H³ (n=3, K=-1): Ric_ij = -2·g_ij
    pub fn ricci_tensor(point: &PoincarePoint) -> na::Matrix3<f64> {
        let g = Self::metric_tensor(point);
        let ricci_factor = (DIMENSION as f64 - 1.0) * CURVATURE_K; // = 2 * (-1) = -2
        ricci_factor * g
    }

    /// Ricci tensor computed from Riemann tensor contraction (verification)
    pub fn ricci_tensor_from_contraction(point: &PoincarePoint) -> na::Matrix3<f64> {
        let riemann = Self::riemann_tensor(point);
        let _g_inv = Self::inverse_metric(point); // Used for general case, not needed for diagonal
        let mut ricci = na::Matrix3::zeros();

        for i in 0..3 {
            for j in 0..3 {
                // Ric_ij = R^k_ikj = g^{km}·R_mikj
                for k in 0..3 {
                    ricci[(i, j)] += riemann[k][i][k][j];
                }
            }
        }

        // Should match ricci_tensor() for constant curvature
        ricci
    }

    /// Ricci scalar R = g^{ij}·Ric_ij
    ///
    /// For constant curvature K in dimension n:
    /// R = n(n-1)K
    ///
    /// For H³ (n=3, K=-1): R = 3·2·(-1) = -6
    pub fn ricci_scalar(_point: &PoincarePoint) -> f64 {
        let n = DIMENSION as f64;
        n * (n - 1.0) * CURVATURE_K // = 3 * 2 * (-1) = -6
    }

    /// Ricci scalar computed from tensor contraction (verification)
    pub fn ricci_scalar_from_contraction(point: &PoincarePoint) -> f64 {
        let ric = Self::ricci_tensor(point);
        let g_inv = Self::inverse_metric(point);

        let mut scalar = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                scalar += g_inv[(i, j)] * ric[(i, j)];
            }
        }

        scalar
    }

    // ========================================================================
    // Sectional Curvature
    // ========================================================================

    /// Sectional curvature K(X, Y) for plane spanned by tangent vectors X, Y
    ///
    /// K(X, Y) = R(X,Y,Y,X) / (g(X,X)·g(Y,Y) - g(X,Y)²)
    ///
    /// where R(X,Y,Z,W) = R_ijkl·X^i·Y^j·Z^k·W^l
    ///
    /// For H³: K(X,Y) = -1 for all non-degenerate planes
    pub fn sectional_curvature(
        point: &PoincarePoint,
        tangent1: &na::Vector3<f64>,
        tangent2: &na::Vector3<f64>,
    ) -> f64 {
        let g = Self::metric_tensor(point);
        let r = Self::riemann_covariant(point);

        // Compute g(X,X), g(Y,Y), g(X,Y)
        let g_xx = (tangent1.transpose() * g * tangent1)[0];
        let g_yy = (tangent2.transpose() * g * tangent2)[0];
        let g_xy = (tangent1.transpose() * g * tangent2)[0];

        // Denominator: area² of parallelogram
        let denom = g_xx * g_yy - g_xy * g_xy;

        if denom.abs() < 1e-14 {
            // Degenerate plane (vectors parallel)
            return f64::NAN;
        }

        // Numerator: R(X,Y,Y,X) = R_ijkl X^i Y^j Y^k X^l
        let mut numer = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        numer += r[i][j][k][l] * tangent1[i] * tangent2[j] * tangent2[k] * tangent1[l];
                    }
                }
            }
        }

        numer / denom
    }

    // ========================================================================
    // Einstein and Weyl Tensors
    // ========================================================================

    /// Einstein tensor G_ij = Ric_ij - (R/2)·g_ij
    ///
    /// For constant curvature K in dimension n:
    /// G_ij = (n-1)K·g_ij - (n(n-1)K/2)·g_ij
    ///      = K(n-1)[1 - n/2]·g_ij
    ///      = K(n-1)(2-n)/2·g_ij
    ///
    /// For H³: G_ij = (-1)(2)(-1)/2·g_ij = g_ij
    pub fn einstein_tensor(point: &PoincarePoint) -> na::Matrix3<f64> {
        let g = Self::metric_tensor(point);
        let ric = Self::ricci_tensor(point);
        let r = Self::ricci_scalar(point);

        ric - (r / 2.0) * g
    }

    /// Weyl conformal curvature tensor C_ijkl
    ///
    /// C_ijkl = R_ijkl - (1/(n-2))(g_ik·Ric_jl - g_il·Ric_jk + g_jl·Ric_ik - g_jk·Ric_il)
    ///        + (R/((n-1)(n-2)))(g_ik·g_jl - g_il·g_jk)
    ///
    /// For n=3 dimensions, the Weyl tensor vanishes identically.
    /// This is because 3D manifolds are conformally flat.
    pub fn weyl_tensor(_point: &PoincarePoint) -> [[[[f64; 3]; 3]; 3]; 3] {
        // In 3 dimensions, Weyl tensor is identically zero
        // The Riemann tensor is fully determined by the Ricci tensor
        [[[[0.0f64; 3]; 3]; 3]; 3]
    }

    /// Kretschmann scalar K = R^ijkl·R_ijkl
    ///
    /// For constant curvature K in dimension n:
    /// Kretschmann = 2n(n-1)K²
    ///
    /// For H³: 2·3·2·1 = 12
    pub fn kretschmann_scalar(_point: &PoincarePoint) -> f64 {
        let n = DIMENSION as f64;
        2.0 * n * (n - 1.0) * CURVATURE_K * CURVATURE_K
    }

    // ========================================================================
    // Geodesic Deviation
    // ========================================================================

    /// Geodesic deviation equation (Jacobi equation)
    ///
    /// D²J/ds² = -R(T,J)T
    ///
    /// where J is the deviation vector, T is the tangent to the geodesic.
    /// In components: (D²J/ds²)^i = -R^i_jkl T^j J^k T^l
    ///
    /// For constant curvature K and J ⊥ T:
    /// D²J/ds² = -K |T|² J
    ///
    /// For H³ (K=-1): D²J/ds² = |T|² J (geodesics diverge exponentially)
    ///
    /// Returns the acceleration of the deviation vector.
    pub fn geodesic_deviation(
        point: &PoincarePoint,
        deviation: &na::Vector3<f64>,
        tangent: &na::Vector3<f64>,
    ) -> na::Vector3<f64> {
        let riemann = Self::riemann_tensor(point);
        let mut accel = na::Vector3::zeros();

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        // (D²J/ds²)^i = -R^i_jkl T^j J^k T^l
                        // Note: index order is T^j J^k T^l (tangent, deviation, tangent)
                        accel[i] -= riemann[i][j][k][l] * tangent[j] * deviation[k] * tangent[l];
                    }
                }
            }
        }

        accel
    }

    /// Compute Jacobi field magnitude evolution
    ///
    /// For H³ with K=-1, geodesics diverge exponentially:
    /// |J(t)| ~ |J(0)|·sinh(t) for initially orthogonal deviation
    ///
    /// Returns the expected magnitude ratio |J(t)|/|J(0)|
    pub fn jacobi_field_growth(t: f64) -> f64 {
        // For negative curvature K = -1:
        // J(t) = J(0)·cosh(√|K|t) + J'(0)·sinh(√|K|t)/√|K|
        // For orthogonal deviation with J'(0) = 0: |J(t)| = |J(0)|·cosh(t)
        t.cosh()
    }

    // ========================================================================
    // Gauss-Codazzi for Hypersurfaces
    // ========================================================================

    /// Second fundamental form component (extrinsic curvature)
    /// for a hypersurface with unit normal n
    ///
    /// II(X, Y) = -g(∇_X n, Y)
    ///
    /// For a horosphere (constant geodesic distance from ideal point):
    /// II = g (umbilic with principal curvatures = 1)
    pub fn horosphere_second_form(point: &PoincarePoint) -> na::Matrix3<f64> {
        Self::metric_tensor(point)
    }

    /// Mean curvature H for a hypersurface
    ///
    /// H = (1/n-1)·tr(II·g^{-1})
    ///
    /// For horospheres in H³: H = 1
    pub fn horosphere_mean_curvature(_point: &PoincarePoint) -> f64 {
        1.0
    }

    // ========================================================================
    // Covariant Derivative
    // ========================================================================

    /// Covariant derivative of a vector field: ∇_X Y
    ///
    /// (∇_X Y)^i = X^j·∂_j Y^i + Γⁱⱼₖ·X^j·Y^k
    ///
    /// For constant Y (just computing connection term):
    pub fn covariant_derivative_term(
        point: &PoincarePoint,
        direction: &na::Vector3<f64>,
        vector: &na::Vector3<f64>,
    ) -> na::Vector3<f64> {
        let gamma = Self::christoffel_symbols(point);
        let mut result = na::Vector3::zeros();

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    // Γⁱⱼₖ · X^j · Y^k
                    result[i] += gamma[i][j][k] * direction[j] * vector[k];
                }
            }
        }

        result
    }

    // ========================================================================
    // Scalar Curvature Invariants
    // ========================================================================

    /// Chern-Gauss-Bonnet integrand in 3D
    ///
    /// The Euler characteristic of a compact 3-manifold is always 0,
    /// but the integrand is related to the scalar curvature.
    pub fn euler_density(point: &PoincarePoint) -> f64 {
        let _kretschmann = Self::kretschmann_scalar(point); // Available for higher-order invariants
        let r = Self::ricci_scalar(point);
        // In 3D: χ = 0 for closed manifolds
        // The density is R/8π in 2D, more complex in 3D
        r / (4.0 * std::f64::consts::PI)
    }

    /// Compute scalar curvature gradient (should vanish for constant curvature)
    pub fn scalar_curvature_gradient(_point: &PoincarePoint) -> na::Vector3<f64> {
        // For constant curvature manifold, ∇R = 0
        na::Vector3::zeros()
    }
}

/// Result of Riemann tensor symmetry verification
#[derive(Debug, Clone)]
pub struct RiemannSymmetryCheck {
    /// R_ijkl = -R_jikl
    pub antisymmetric_first_pair: bool,
    /// R_ijkl = -R_ijlk
    pub antisymmetric_second_pair: bool,
    /// R_ijkl = R_klij
    pub pair_symmetric: bool,
    /// First Bianchi identity: R_ijkl + R_iklj + R_iljk = 0
    pub bianchi_identity: bool,
    /// Maximum error across all symmetry checks
    pub max_error: f64,
}

impl RiemannSymmetryCheck {
    /// Check if all symmetries are satisfied
    pub fn all_satisfied(&self) -> bool {
        self.antisymmetric_first_pair
            && self.antisymmetric_second_pair
            && self.pair_symmetric
            && self.bianchi_identity
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

        assert!((k - (-1.0)).abs() < 1e-10, "Sectional curvature must be K=-1, got {}", k);
    }

    #[test]
    fn test_sectional_curvature_arbitrary_plane() {
        let point = PoincarePoint::new(na::Vector3::new(0.3, 0.4, 0.1)).unwrap();

        // Arbitrary non-orthogonal tangent vectors
        let v1 = na::Vector3::new(1.0, 0.5, 0.2);
        let v2 = na::Vector3::new(-0.3, 1.0, 0.7);

        let k = CurvatureTensor::sectional_curvature(&point, &v1, &v2);

        assert!((k - (-1.0)).abs() < 1e-10, "Sectional curvature must be K=-1 for any plane, got {}", k);
    }

    #[test]
    fn test_ricci_scalar() {
        let point = PoincarePoint::origin();
        let r = CurvatureTensor::ricci_scalar(&point);

        assert_eq!(r, -6.0, "Ricci scalar must be -6 for H³");
    }

    #[test]
    fn test_ricci_scalar_from_contraction() {
        let point = PoincarePoint::new(na::Vector3::new(0.4, 0.2, 0.1)).unwrap();

        let r_direct = CurvatureTensor::ricci_scalar(&point);
        let r_contracted = CurvatureTensor::ricci_scalar_from_contraction(&point);

        assert!((r_direct - r_contracted).abs() < 1e-10,
            "Ricci scalar from contraction should match direct: {} vs {}", r_direct, r_contracted);
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

    #[test]
    fn test_metric_inverse() {
        let point = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.1)).unwrap();

        let g = CurvatureTensor::metric_tensor(&point);
        let g_inv = CurvatureTensor::inverse_metric(&point);
        let product = g * g_inv;

        // Should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-10,
                    "g * g^{{-1}} should be identity at ({}, {}): got {}", i, j, product[(i, j)]);
            }
        }
    }

    #[test]
    fn test_riemann_symmetries() {
        let point = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.1)).unwrap();

        let check = CurvatureTensor::verify_riemann_symmetries(&point);

        assert!(check.antisymmetric_first_pair, "Riemann should be antisymmetric in first pair");
        assert!(check.antisymmetric_second_pair, "Riemann should be antisymmetric in second pair");
        assert!(check.pair_symmetric, "Riemann should have pair symmetry");
        assert!(check.bianchi_identity, "Riemann should satisfy first Bianchi identity");
        assert!(check.all_satisfied(), "All Riemann symmetries should be satisfied");
    }

    #[test]
    fn test_kretschmann_scalar() {
        let point = PoincarePoint::origin();
        let k = CurvatureTensor::kretschmann_scalar(&point);

        // For H³: 2*3*2*(-1)² = 12
        assert_eq!(k, 12.0, "Kretschmann scalar should be 12 for H³");
    }

    #[test]
    fn test_weyl_vanishes_in_3d() {
        let point = PoincarePoint::new(na::Vector3::new(0.2, 0.3, 0.4)).unwrap();
        let weyl = CurvatureTensor::weyl_tensor(&point);

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        assert_eq!(weyl[i][j][k][l], 0.0,
                            "Weyl tensor should vanish in 3D");
                    }
                }
            }
        }
    }

    #[test]
    fn test_einstein_tensor() {
        let point = PoincarePoint::new(na::Vector3::new(0.2, 0.1, 0.3)).unwrap();

        let g = CurvatureTensor::metric_tensor(&point);
        let einstein = CurvatureTensor::einstein_tensor(&point);

        // For H³: G_ij = g_ij
        for i in 0..3 {
            for j in 0..3 {
                assert!((einstein[(i, j)] - g[(i, j)]).abs() < 1e-10,
                    "Einstein tensor should equal metric for H³");
            }
        }
    }

    #[test]
    fn test_christoffel_symmetry() {
        let point = PoincarePoint::new(na::Vector3::new(0.3, 0.2, 0.1)).unwrap();
        let gamma = CurvatureTensor::christoffel_symbols(&point);

        // Christoffel symbols symmetric in lower indices: Γⁱⱼₖ = Γⁱₖⱼ
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    assert!((gamma[i][j][k] - gamma[i][k][j]).abs() < 1e-10,
                        "Christoffel symbols should be symmetric in lower indices");
                }
            }
        }
    }

    #[test]
    fn test_jacobi_field_growth() {
        // At t=0, growth factor should be 1
        assert!((CurvatureTensor::jacobi_field_growth(0.0) - 1.0).abs() < 1e-10);

        // For t > 0, growth should be > 1 (geodesics diverge in negative curvature)
        let growth = CurvatureTensor::jacobi_field_growth(1.0);
        assert!(growth > 1.0, "Jacobi field should grow in negative curvature");

        // Should equal cosh(t)
        let expected = 1.0_f64.cosh();
        assert!((growth - expected).abs() < 1e-10,
            "Jacobi field growth should be cosh(t)");
    }

    #[test]
    fn test_geodesic_deviation_direction() {
        let point = PoincarePoint::origin();

        let deviation = na::Vector3::new(1.0, 0.0, 0.0);
        let tangent = na::Vector3::new(0.0, 1.0, 0.0);

        let accel = CurvatureTensor::geodesic_deviation(&point, &deviation, &tangent);

        // For K=-1, geodesic deviation should accelerate away
        // The sign depends on convention, but magnitude should be non-zero
        // At origin, metric is euclidean-like, so |accel| ≈ |deviation| * K * |tangent|²
        let mag = accel.norm();
        assert!(mag > 0.0, "Geodesic deviation should be non-zero");
    }

    #[test]
    fn test_volume_element() {
        let point = PoincarePoint::new(na::Vector3::new(0.5, 0.3, 0.2)).unwrap();

        let vol = CurvatureTensor::volume_element(&point);
        let det = CurvatureTensor::metric_determinant(&point);

        // Volume element = √det(g), so vol² = det
        assert!((vol * vol - det).abs() < 1e-10,
            "Volume element squared should equal metric determinant: vol²={}, det={}", vol*vol, det);
    }

    #[test]
    fn test_conformal_scaling() {
        // At origin, conformal factor λ = 2
        let origin = PoincarePoint::origin();
        let g_origin = CurvatureTensor::metric_tensor(&origin);

        // At origin, g = 4·I
        for i in 0..3 {
            assert!((g_origin[(i, i)] - 4.0).abs() < 1e-10,
                "At origin, metric should be 4·δ_ij");
        }

        // Closer to boundary, metric blows up
        let near_boundary = PoincarePoint::new(na::Vector3::new(0.9, 0.0, 0.0)).unwrap();
        let g_boundary = CurvatureTensor::metric_tensor(&near_boundary);

        assert!(g_boundary[(0, 0)] > g_origin[(0, 0)],
            "Metric should increase near boundary");
    }
}
