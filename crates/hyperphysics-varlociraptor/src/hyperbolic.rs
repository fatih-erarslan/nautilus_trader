//! Hyperbolic variant space for genomic clustering
//!
//! Maps genetic variants to hyperbolic geometry for clustering and visualization

use anyhow::Result;
use hyperphysics_geometry::hyperbolic::{HyperbolicPoint, PoincareModel};
use nalgebra::DVector;

use crate::vcf::VcfVariant;

/// Hyperbolic variant space
#[derive(Debug)]
pub struct HyperbolicVariantSpace {
    /// Poincare disk model
    model: PoincareModel,

    /// Dimension of hyperbolic space
    dimension: usize,
}

impl HyperbolicVariantSpace {
    /// Create new hyperbolic variant space
    pub fn new(dimension: usize) -> Self {
        Self {
            model: PoincareModel::new(dimension),
            dimension,
        }
    }

    /// Map variant to hyperbolic space
    ///
    /// Uses variant features (VAF, depth, quality, position) to create
    /// hyperbolic embedding that preserves variant relationships
    pub fn map_variant(&self, variant: &VcfVariant) -> Result<HyperbolicPoint> {
        // Extract variant features
        let features = self.extract_features(variant)?;

        // Normalize to unit ball
        let norm = features.norm();
        let normalized = if norm > 0.0 {
            features / norm * 0.95 // Scale to avoid boundary
        } else {
            features
        };

        Ok(HyperbolicPoint::from_poincare(&normalized))
    }

    /// Extract features from variant for embedding
    fn extract_features(&self, variant: &VcfVariant) -> Result<DVector<f64>> {
        let mut features = Vec::new();

        // Feature 1: Variant allele frequency (if available)
        if let Some(vaf) = variant.get_vaf(0) {
            features.push(vaf);
        } else {
            features.push(0.5); // Default
        }

        // Feature 2: Quality score (normalized)
        if let Some(qual) = variant.qual {
            features.push((qual / 100.0).min(1.0));
        } else {
            features.push(0.5);
        }

        // Feature 3: Coverage depth (normalized)
        if let Some(depth) = variant.get_depth(0) {
            features.push((depth as f64 / 100.0).min(1.0));
        } else {
            features.push(0.5);
        }

        // Feature 4: Reference allele length
        features.push((variant.ref_allele.len() as f64 / 100.0).min(1.0));

        // Feature 5: Alternate allele length
        if !variant.alt_alleles.is_empty() {
            features.push((variant.alt_alleles[0].len() as f64 / 100.0).min(1.0));
        } else {
            features.push(0.5);
        }

        // Pad or truncate to match dimension
        while features.len() < self.dimension {
            features.push(0.0);
        }
        features.truncate(self.dimension);

        Ok(DVector::from_vec(features))
    }

    /// Compute hyperbolic distance between two variants
    pub fn variant_distance(&self, v1: &VcfVariant, v2: &VcfVariant) -> Result<f64> {
        let p1 = self.map_variant(v1)?;
        let p2 = self.map_variant(v2)?;
        Ok(self.model.distance(&p1, &p2))
    }

    /// Cluster variants in hyperbolic space
    pub fn cluster_variants(&self, variants: &[VcfVariant], k: usize) -> Result<Vec<usize>> {
        // Map all variants to hyperbolic space
        let points: Vec<HyperbolicPoint> = variants
            .iter()
            .map(|v| self.map_variant(v))
            .collect::<Result<Vec<_>>>()?;

        // Use hyperbolic k-means clustering
        let clusters = self.hyperbolic_kmeans(&points, k)?;

        Ok(clusters)
    }

    /// Hyperbolic k-means clustering
    fn hyperbolic_kmeans(&self, points: &[HyperbolicPoint], k: usize) -> Result<Vec<usize>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let n = points.len();
        let mut assignments = vec![0; n];

        // Initialize centroids (simple: take first k points)
        let mut centroids: Vec<HyperbolicPoint> = points.iter().take(k).cloned().collect();

        // Iterate until convergence
        for _ in 0..100 {
            let mut changed = false;

            // Assign points to nearest centroid
            for (i, point) in points.iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = self.model.distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids (hyperbolic mean)
            for j in 0..k {
                let cluster_points: Vec<&HyperbolicPoint> = points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| assignments[*i] == j)
                    .map(|(_, p)| p)
                    .collect();

                if !cluster_points.is_empty() {
                    centroids[j] = self.hyperbolic_mean(&cluster_points);
                }
            }
        }

        Ok(assignments)
    }

    /// Compute hyperbolic mean (Frechet mean)
    fn hyperbolic_mean(&self, points: &[&HyperbolicPoint]) -> HyperbolicPoint {
        if points.is_empty() {
            return HyperbolicPoint::origin(self.dimension);
        }

        // Start with first point
        let mut mean = points[0].clone();

        // Iterative Frechet mean computation
        for _ in 0..10 {
            let mut sum = DVector::zeros(self.dimension);

            for point in points {
                let tangent = self.model.log_map(&mean, point);
                sum += tangent;
            }

            let update = sum / (points.len() as f64);
            mean = self.model.exp_map(&mean, &update);
        }

        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcf::VcfVariant;

    #[test]
    fn test_hyperbolic_space_creation() {
        let space = HyperbolicVariantSpace::new(5);
        assert_eq!(space.dimension, 5);
    }

    #[test]
    fn test_variant_mapping() {
        let space = HyperbolicVariantSpace::new(5);
        let line = "chr1\t12345\t.\tA\tG\t30.0\tPASS\tDP=50\tGT:DP:AF\t0/1:50:0.3";
        let variant = VcfVariant::from_vcf_line(line).unwrap();

        let result = space.map_variant(&variant);
        assert!(result.is_ok());
    }
}
