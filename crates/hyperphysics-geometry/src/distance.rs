//! Hyperbolic distance calculations and metrics

use crate::poincare::PoincarePoint;
use nalgebra as na;

/// Hyperbolic distance calculator with optimizations
pub struct HyperbolicDistance;

impl HyperbolicDistance {
    /// Calculate distance between two points
    #[inline]
    pub fn distance(p: &PoincarePoint, q: &PoincarePoint) -> f64 {
        p.hyperbolic_distance(q)
    }

    /// Calculate distances from one point to many (vectorized)
    pub fn distances_from(
        origin: &PoincarePoint,
        targets: &[PoincarePoint],
    ) -> Vec<f64> {
        targets
            .iter()
            .map(|target| origin.hyperbolic_distance(target))
            .collect()
    }

    /// Calculate pairwise distance matrix
    pub fn distance_matrix(points: &[PoincarePoint]) -> na::DMatrix<f64> {
        let n = points.len();
        let mut matrix = na::DMatrix::zeros(n, n);

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = points[i].hyperbolic_distance(&points[j]);
                matrix[(i, j)] = dist;
                matrix[(j, i)] = dist;
            }
        }

        matrix
    }

    /// Find k-nearest neighbors in hyperbolic space
    pub fn k_nearest(
        origin: &PoincarePoint,
        candidates: &[PoincarePoint],
        k: usize,
    ) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, p)| (i, origin.hyperbolic_distance(p)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_symmetry() {
        let p = PoincarePoint::new(na::Vector3::new(0.3, 0.0, 0.0)).unwrap();
        let q = PoincarePoint::new(na::Vector3::new(0.0, 0.4, 0.0)).unwrap();

        let d1 = HyperbolicDistance::distance(&p, &q);
        let d2 = HyperbolicDistance::distance(&q, &p);

        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix() {
        let points = vec![
            PoincarePoint::new(na::Vector3::new(0.1, 0.0, 0.0)).unwrap(),
            PoincarePoint::new(na::Vector3::new(0.0, 0.2, 0.0)).unwrap(),
            PoincarePoint::new(na::Vector3::new(0.0, 0.0, 0.3)).unwrap(),
        ];

        let matrix = HyperbolicDistance::distance_matrix(&points);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-10);
            }
        }

        // Check diagonal is zero
        for i in 0..3 {
            assert_eq!(matrix[(i, i)], 0.0);
        }
    }
}
