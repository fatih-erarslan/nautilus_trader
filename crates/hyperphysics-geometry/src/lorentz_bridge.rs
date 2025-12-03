//! Bridge between Poincaré and Lorentz models
//!
//! Provides seamless conversion between PoincarePoint and LorentzPoint
//! for use cases requiring the hyperboloid representation.
//!
//! # Mathematical Foundation
//!
//! The Poincaré ball and Lorentz hyperboloid are isometrically equivalent
//! representations of hyperbolic space. This bridge enables:
//!
//! - SIMD-accelerated distance computation via Minkowski inner products
//! - Exponential/logarithmic maps in the hyperboloid
//! - Parallel transport of tangent vectors
//!
//! Reference: Nickel & Kiela (2018) "Poincaré Embeddings for Learning
//! Hierarchical Representations" NIPS

#[cfg(feature = "lorentz")]
use hyperphysics_lorentz::{
    poincare_to_lorentz, lorentz_to_poincare,
    LorentzPoint, LorentzModel,
};
use crate::{PoincarePoint, GeometryError, Result, CURVATURE};

/// Extension trait for PoincarePoint to convert to Lorentz representation
#[cfg(feature = "lorentz")]
pub trait ToLorentz {
    /// Convert to Lorentz hyperboloid coordinates
    fn to_lorentz(&self) -> Result<LorentzPoint>;

    /// Convert to raw Lorentz vector
    fn to_lorentz_coords(&self) -> Result<Vec<f64>>;
}

#[cfg(feature = "lorentz")]
impl ToLorentz for PoincarePoint {
    fn to_lorentz(&self) -> Result<LorentzPoint> {
        let coords = self.coords();
        let poincare_vec: Vec<f64> = vec![coords.x, coords.y, coords.z];

        let lorentz_coords = poincare_to_lorentz(&poincare_vec, CURVATURE)
            .map_err(|e| GeometryError::GeodesicFailure {
                reason: format!("Poincaré to Lorentz conversion failed: {}", e)
            })?;

        Ok(LorentzPoint::new_unchecked(lorentz_coords, CURVATURE))
    }

    fn to_lorentz_coords(&self) -> Result<Vec<f64>> {
        let coords = self.coords();
        let poincare_vec: Vec<f64> = vec![coords.x, coords.y, coords.z];

        poincare_to_lorentz(&poincare_vec, CURVATURE)
            .map_err(|e| GeometryError::GeodesicFailure {
                reason: format!("Poincaré to Lorentz conversion failed: {}", e)
            })
    }
}

/// Extension trait for LorentzPoint to convert to Poincaré representation
#[cfg(feature = "lorentz")]
pub trait ToPoincare {
    /// Convert to Poincaré ball coordinates
    fn to_poincare(&self) -> Result<PoincarePoint>;
}

#[cfg(feature = "lorentz")]
impl ToPoincare for LorentzPoint {
    fn to_poincare(&self) -> Result<PoincarePoint> {
        let poincare_vec = lorentz_to_poincare(self.coords(), CURVATURE)
            .map_err(|e| GeometryError::GeodesicFailure {
                reason: format!("Lorentz to Poincaré conversion failed: {}", e)
            })?;

        if poincare_vec.len() < 3 {
            return Err(GeometryError::GeodesicFailure {
                reason: "Converted Poincaré vector has insufficient dimensions".to_string()
            });
        }

        PoincarePoint::new(nalgebra::Vector3::new(
            poincare_vec[0],
            poincare_vec[1],
            poincare_vec[2],
        ))
    }
}

/// Compute hyperbolic distance using SIMD Minkowski operations
///
/// This is significantly faster for batch distance computations
/// compared to the standard Poincaré distance formula.
#[cfg(feature = "lorentz")]
pub fn simd_hyperbolic_distance(p: &PoincarePoint, q: &PoincarePoint) -> Result<f64> {
    use hyperphysics_lorentz::SimdMinkowski;

    let p_lorentz = p.to_lorentz_coords()?;
    let q_lorentz = q.to_lorentz_coords()?;

    SimdMinkowski::distance(&p_lorentz, &q_lorentz, CURVATURE)
        .map_err(|_| GeometryError::NumericalInstability)
}

/// Batch convert multiple Poincaré points to Lorentz
#[cfg(feature = "lorentz")]
pub fn batch_to_lorentz(points: &[PoincarePoint]) -> Result<Vec<LorentzPoint>> {
    points.iter()
        .map(|p| p.to_lorentz())
        .collect()
}

/// Batch convert multiple Lorentz points to Poincaré
#[cfg(feature = "lorentz")]
pub fn batch_to_poincare(points: &[LorentzPoint]) -> Result<Vec<PoincarePoint>> {
    points.iter()
        .map(|p| p.to_poincare())
        .collect()
}

/// Compute pairwise distances using SIMD Minkowski operations
///
/// More efficient for large point sets compared to individual distance calls.
#[cfg(feature = "lorentz")]
pub fn pairwise_distances_simd(points: &[PoincarePoint]) -> Result<Vec<Vec<f64>>> {
    use hyperphysics_lorentz::SimdMinkowski;

    let lorentz_points: Vec<Vec<f64>> = points.iter()
        .map(|p| p.to_lorentz_coords())
        .collect::<Result<Vec<_>>>()?;

    let n = lorentz_points.len();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = SimdMinkowski::distance(&lorentz_points[i], &lorentz_points[j], CURVATURE)
                .map_err(|_| GeometryError::NumericalInstability)?;
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    Ok(distances)
}

/// Create a Lorentz model with standard curvature K=-1
#[cfg(feature = "lorentz")]
pub fn lorentz_model() -> LorentzModel {
    LorentzModel::default_curvature()
}

#[cfg(all(test, feature = "lorentz"))]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_roundtrip_conversion() {
        let p = PoincarePoint::new(Vector3::new(0.3, 0.4, 0.0)).unwrap();

        let lorentz = p.to_lorentz().unwrap();
        let recovered = lorentz.to_poincare().unwrap();

        let diff = (p.coords() - recovered.coords()).norm();
        assert!(diff < TOL, "Roundtrip error: {}", diff);
    }

    #[test]
    fn test_origin_conversion() {
        let origin = PoincarePoint::origin();
        let lorentz = origin.to_lorentz().unwrap();

        // Origin in Poincaré (0,0,0) → Lorentz (1,0,0,0)
        assert!((lorentz.time_coord() - 1.0).abs() < TOL);
        assert!(lorentz.spatial_coords().iter().all(|&x| x.abs() < TOL));
    }

    #[test]
    fn test_simd_distance_matches_poincare() {
        let p = PoincarePoint::new(Vector3::new(0.2, 0.3, 0.1)).unwrap();
        let q = PoincarePoint::new(Vector3::new(0.4, 0.1, 0.2)).unwrap();

        let poincare_dist = p.hyperbolic_distance(&q);
        let simd_dist = simd_hyperbolic_distance(&p, &q).unwrap();

        // Allow 1% relative tolerance due to different computation paths
        let rel_diff = (poincare_dist - simd_dist).abs() / poincare_dist.max(1e-10);
        assert!(rel_diff < 0.01, "Distance mismatch: Poincaré={}, SIMD={}", poincare_dist, simd_dist);
    }

    #[test]
    fn test_batch_conversion() {
        let points = vec![
            PoincarePoint::new(Vector3::new(0.1, 0.2, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.3, 0.0, 0.1)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.4, 0.2)).unwrap(),
        ];

        let lorentz_points = batch_to_lorentz(&points).unwrap();
        let recovered = batch_to_poincare(&lorentz_points).unwrap();

        for (orig, rec) in points.iter().zip(recovered.iter()) {
            let diff = (orig.coords() - rec.coords()).norm();
            assert!(diff < TOL, "Batch roundtrip error: {}", diff);
        }
    }

    #[test]
    fn test_pairwise_distances() {
        let points = vec![
            PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.2, 0.0)).unwrap(),
            PoincarePoint::new(Vector3::new(0.0, 0.0, 0.3)).unwrap(),
        ];

        let distances = pairwise_distances_simd(&points).unwrap();

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[i][j] - distances[j][i]).abs() < TOL);
            }
        }

        // Check diagonal is zero
        for i in 0..3 {
            assert!(distances[i][i].abs() < TOL);
        }
    }
}
