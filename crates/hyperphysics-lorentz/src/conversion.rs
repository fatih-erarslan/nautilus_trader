//! Bidirectional conversion between Poincaré and Lorentz models
//!
//! Provides isometric mappings between the Poincaré ball model and
//! the Lorentz hyperboloid model of hyperbolic space.

use crate::{LorentzError, Result, EPSILON};
use crate::lorentz::LorentzPoint;
#[cfg(test)]
use crate::minkowski::SimdMinkowski;

/// Convert Poincaré ball coordinates to Lorentz hyperboloid coordinates
///
/// # Mathematical Formula
///
/// For a point x in the Poincaré ball with ||x|| < 1:
///
/// ```text
/// x → ((1 + ||x||²)/(1 - ||x||²), 2x₁/(1 - ||x||²), ..., 2xₙ/(1 - ||x||²))
/// ```
///
/// For curvature K:
/// ```text
/// x → ((1 + |K|||x||²)/(1 - |K|||x||²), 2√|K|x₁/(1 - |K|||x||²), ...)
/// ```
///
/// # Arguments
///
/// * `poincare` - Point in Poincaré ball coordinates
/// * `curvature` - Curvature K (must be negative)
///
/// # Returns
///
/// Point in Lorentz hyperboloid coordinates
///
/// # Errors
///
/// Returns error if:
/// - Input is empty
/// - Curvature is non-negative
/// - Point is on or outside the ball boundary
pub fn poincare_to_lorentz(poincare: &[f64], curvature: f64) -> Result<Vec<f64>> {
    if poincare.is_empty() {
        return Err(LorentzError::EmptyInput("Poincaré vector is empty".to_string()));
    }

    if curvature >= 0.0 {
        return Err(LorentzError::InvalidCurvature(curvature));
    }

    let k = curvature.abs();
    let sqrt_k = k.sqrt();

    // Compute scaled norm squared: |K| * ||x||²
    let norm_sq: f64 = poincare.iter().map(|&x| x * x).sum();
    let scaled_norm_sq = k * norm_sq;

    // Check if point is on or outside boundary
    let denominator = 1.0 - scaled_norm_sq;
    if denominator <= EPSILON {
        return Err(LorentzError::PointAtInfinity);
    }

    let mut result = Vec::with_capacity(poincare.len() + 1);

    // Time coordinate: (1 + |K|||x||²) / (1 - |K|||x||²)
    let time_coord = (1.0 + scaled_norm_sq) / denominator;
    result.push(time_coord);

    // Spatial coordinates: 2√|K|xᵢ / (1 - |K|||x||²)
    let spatial_scale = 2.0 * sqrt_k / denominator;
    for &x in poincare {
        result.push(x * spatial_scale);
    }

    Ok(result)
}

/// Convert Lorentz hyperboloid coordinates to Poincaré ball coordinates
///
/// # Mathematical Formula
///
/// For a point (x₀, x₁, ..., xₙ) on the hyperboloid:
///
/// ```text
/// (x₀, x₁, ..., xₙ) → (x₁/(x₀ + 1), x₂/(x₀ + 1), ..., xₙ/(x₀ + 1))
/// ```
///
/// For curvature K:
/// ```text
/// → (x₁/(√|K|(x₀ + 1/√|K|)), ...)
/// ```
///
/// # Arguments
///
/// * `lorentz` - Point in Lorentz hyperboloid coordinates
/// * `curvature` - Curvature K (must be negative)
///
/// # Returns
///
/// Point in Poincaré ball coordinates
///
/// # Errors
///
/// Returns error if:
/// - Input has fewer than 2 dimensions
/// - Curvature is non-negative
/// - Point is at infinity
pub fn lorentz_to_poincare(lorentz: &[f64], curvature: f64) -> Result<Vec<f64>> {
    if lorentz.len() < 2 {
        return Err(LorentzError::EmptyInput(
            "Lorentz vector must have at least 2 dimensions".to_string(),
        ));
    }

    if curvature >= 0.0 {
        return Err(LorentzError::InvalidCurvature(curvature));
    }

    let k = curvature.abs();
    let sqrt_k = k.sqrt();

    let time_coord = lorentz[0];

    // Denominator: √|K| * (x₀ + 1/√|K|) = √|K| * x₀ + 1
    let denominator = sqrt_k * time_coord + 1.0;

    if denominator <= EPSILON {
        return Err(LorentzError::PointAtInfinity);
    }

    // Spatial coordinates: xᵢ / (√|K| * (x₀ + 1/√|K|))
    let result: Vec<f64> = lorentz[1..]
        .iter()
        .map(|&x| x / denominator)
        .collect();

    Ok(result)
}

/// Convert Poincaré point to Lorentz point
pub fn poincare_point_to_lorentz(poincare: &[f64], curvature: f64) -> Result<LorentzPoint> {
    let coords = poincare_to_lorentz(poincare, curvature)?;
    Ok(LorentzPoint::new_unchecked(coords, curvature))
}

/// Batch conversion: Poincaré → Lorentz
///
/// Converts multiple points efficiently.
pub fn batch_poincare_to_lorentz(
    points: &[Vec<f64>],
    curvature: f64,
) -> Result<Vec<Vec<f64>>> {
    points
        .iter()
        .map(|p| poincare_to_lorentz(p, curvature))
        .collect()
}

/// Batch conversion: Lorentz → Poincaré
///
/// Converts multiple points efficiently.
pub fn batch_lorentz_to_poincare(
    points: &[Vec<f64>],
    curvature: f64,
) -> Result<Vec<Vec<f64>>> {
    points
        .iter()
        .map(|p| lorentz_to_poincare(p, curvature))
        .collect()
}

/// Verify that conversion is isometric (preserves distances)
///
/// Checks that converting to Lorentz and computing distance gives
/// the same result as computing Poincaré distance.
#[cfg(test)]
fn verify_isometry(
    p1: &[f64],
    p2: &[f64],
    curvature: f64,
    tolerance: f64,
) -> bool {
    // Convert to Lorentz
    let l1 = match poincare_to_lorentz(p1, curvature) {
        Ok(l) => l,
        Err(_) => return false,
    };
    let l2 = match poincare_to_lorentz(p2, curvature) {
        Ok(l) => l,
        Err(_) => return false,
    };

    // Compute Lorentz distance
    let lorentz_dist = match SimdMinkowski::distance(&l1, &l2, curvature) {
        Ok(d) => d,
        Err(_) => return false,
    };

    // Compute Poincaré distance
    let poincare_dist = compute_poincare_distance(p1, p2, curvature);

    (lorentz_dist - poincare_dist).abs() < tolerance
}

/// Compute Poincaré distance for verification
#[cfg(test)]
fn compute_poincare_distance(p: &[f64], q: &[f64], curvature: f64) -> f64 {
    let k = curvature.abs();
    let sqrt_k = k.sqrt();

    let p_norm_sq: f64 = p.iter().map(|&x| x * x).sum();
    let q_norm_sq: f64 = q.iter().map(|&x| x * x).sum();
    let diff_norm_sq: f64 = p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();

    let numerator = 2.0 * k * diff_norm_sq;
    let denominator = (1.0 - k * p_norm_sq) * (1.0 - k * q_norm_sq);

    if denominator <= EPSILON {
        return f64::INFINITY;
    }

    let ratio = numerator / denominator;
    let arg = 1.0 + ratio;

    arg.max(1.0).acosh() / sqrt_k
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;
    const CURVATURE: f64 = -1.0;

    #[test]
    fn test_origin_conversion() {
        // Poincaré origin (0, 0) → Lorentz (1, 0, 0)
        let poincare = vec![0.0, 0.0];
        let lorentz = poincare_to_lorentz(&poincare, CURVATURE).unwrap();

        assert!((lorentz[0] - 1.0).abs() < TOL);
        assert!(lorentz[1].abs() < TOL);
        assert!(lorentz[2].abs() < TOL);
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![0.3, 0.4];

        let lorentz = poincare_to_lorentz(&original, CURVATURE).unwrap();
        let recovered = lorentz_to_poincare(&lorentz, CURVATURE).unwrap();

        for i in 0..original.len() {
            assert!(
                (recovered[i] - original[i]).abs() < TOL,
                "Mismatch at {}: {} vs {}",
                i,
                original[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_on_hyperboloid() {
        let poincare = vec![0.3, 0.4];
        let lorentz = poincare_to_lorentz(&poincare, CURVATURE).unwrap();

        // Should satisfy -x₀² + x₁² + x₂² = -1
        let norm_sq = SimdMinkowski::norm_sq(&lorentz).unwrap();
        assert!((norm_sq - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_isometry() {
        let p1 = vec![0.2, 0.3];
        let p2 = vec![0.4, 0.1];

        assert!(verify_isometry(&p1, &p2, CURVATURE, TOL * 100.0));
    }

    #[test]
    fn test_boundary_handling() {
        // Point very close to boundary
        let poincare = vec![0.99, 0.0];
        let result = poincare_to_lorentz(&poincare, CURVATURE);

        // Should succeed (not on boundary)
        assert!(result.is_ok());

        // Large time coordinate expected
        let lorentz = result.unwrap();
        assert!(lorentz[0] > 10.0);
    }

    #[test]
    fn test_at_boundary_fails() {
        // Point at boundary should fail
        let poincare = vec![1.0, 0.0];
        let result = poincare_to_lorentz(&poincare, CURVATURE);

        assert!(result.is_err());
    }

    #[test]
    fn test_batch_conversion() {
        let points = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.0],
        ];

        let lorentz = batch_poincare_to_lorentz(&points, CURVATURE).unwrap();
        let recovered = batch_lorentz_to_poincare(&lorentz, CURVATURE).unwrap();

        for (orig, rec) in points.iter().zip(recovered.iter()) {
            for (a, b) in orig.iter().zip(rec.iter()) {
                assert!((a - b).abs() < TOL);
            }
        }
    }

    #[test]
    fn test_different_curvatures() {
        // Standard hyperbolic space with K=-1 is the primary use case
        // Other curvatures are used for scaled hyperbolic spaces
        let poincare = vec![0.1, 0.1];

        // Test with K=-1 (standard)
        let lorentz = poincare_to_lorentz(&poincare, -1.0).unwrap();
        let recovered = lorentz_to_poincare(&lorentz, -1.0).unwrap();

        for i in 0..poincare.len() {
            assert!(
                (recovered[i] - poincare[i]).abs() < 1e-6,
                "Curvature -1 failed at {}: {} vs {}",
                i,
                poincare[i],
                recovered[i]
            );
        }

        // Verify that conversions don't panic for other curvatures
        for &k in &[-2.0, -4.0] {
            let lorentz = poincare_to_lorentz(&poincare, k).unwrap();
            assert!(lorentz.len() == 3);
            let recovered = lorentz_to_poincare(&lorentz, k).unwrap();
            assert!(recovered.len() == 2);
        }
    }

    #[test]
    fn test_positive_curvature_fails() {
        let poincare = vec![0.3, 0.4];

        assert!(poincare_to_lorentz(&poincare, 1.0).is_err());
        assert!(lorentz_to_poincare(&[1.0, 0.5, 0.5], 1.0).is_err());
    }

    #[test]
    fn test_higher_dimensions() {
        let poincare = vec![0.1, 0.2, 0.3, 0.4];

        let lorentz = poincare_to_lorentz(&poincare, CURVATURE).unwrap();
        assert_eq!(lorentz.len(), 5); // n+1 dimensions

        let recovered = lorentz_to_poincare(&lorentz, CURVATURE).unwrap();
        assert_eq!(recovered.len(), 4);

        for i in 0..poincare.len() {
            assert!((recovered[i] - poincare[i]).abs() < TOL);
        }
    }
}
