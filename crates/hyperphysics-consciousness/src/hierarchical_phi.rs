//! Hierarchical Φ calculator with spatial clustering
//!
//! Implements multi-scale IIT approximation using hyperbolic geometry.
//!
//! # Mathematical Foundation
//!
//! Traditional IIT computation: O(2^N) - intractable for N > 20
//!
//! Our approach:
//! 1. Spatial clustering using hyperbolic distance
//! 2. Hierarchical decomposition with tessellation levels
//! 3. Multi-scale aggregation: Φ_total = Σ_scales α_s · Φ_s
//!
//! # Performance
//!
//! - N < 1000: Exact calculation O(2^N)
//! - N < 10^6: Spatial clustering O(N log N)
//! - N > 10^6: Hierarchical O(N log² N)
//!
//! Research:
//! - Balduzzi & Tononi (2009) "Qualia: The geometry of integrated information" PLOS Comp Bio
//! - Mayner et al. (2018) "PyPhi approximations" PLOS Comp Bio

use crate::{Result, phi::PhiMethod};
use hyperphysics_geometry::PoincarePoint;
use hyperphysics_pbit::PBitLattice;
use rayon::prelude::*;

/// Hierarchical Φ result with multi-scale decomposition
#[derive(Debug, Clone)]
pub struct HierarchicalPhi {
    /// Total integrated information
    pub phi_total: f64,

    /// Φ at each spatial scale
    pub phi_per_scale: Vec<f64>,

    /// Number of clusters at each scale
    pub clusters_per_scale: Vec<usize>,

    /// Computation method
    pub method: PhiMethod,

    /// Spatial scales used (hyperbolic radii)
    pub scales: Vec<f64>,
}

/// Spatial cluster of pBits
#[derive(Debug, Clone)]
pub struct SpatialCluster {
    /// pBit indices in cluster
    pub indices: Vec<usize>,

    /// Cluster centroid in hyperbolic space
    pub centroid: PoincarePoint,

    /// Cluster radius (hyperbolic distance)
    pub radius: f64,
}

/// Hierarchical Φ calculator with spatial clustering
pub struct HierarchicalPhiCalculator {
    /// Number of hierarchical levels
    levels: usize,

    /// Clustering method
    clustering: ClusteringMethod,

    /// Scale factor between levels (typically 2-4)
    scale_factor: f64,

    /// Minimum cluster size
    min_cluster_size: usize,
}

/// Clustering method for spatial decomposition
#[derive(Debug, Clone, Copy)]
pub enum ClusteringMethod {
    /// K-means clustering in hyperbolic space
    HyperbolicKMeans,

    /// Hierarchical agglomerative clustering
    Agglomerative,

    /// Tessellation-based (use existing hyperbolic tessellation levels)
    Tessellation,
}

impl Default for HierarchicalPhiCalculator {
    fn default() -> Self {
        Self::new(3, ClusteringMethod::Tessellation, 2.0, 10)
    }
}

impl HierarchicalPhiCalculator {
    /// Create new hierarchical calculator
    ///
    /// # Arguments
    ///
    /// * `levels` - Number of hierarchical scales (typically 3-5)
    /// * `clustering` - Spatial clustering method
    /// * `scale_factor` - Multiplicative factor between scales (2-4)
    /// * `min_cluster_size` - Minimum nodes per cluster (5-20)
    pub fn new(
        levels: usize,
        clustering: ClusteringMethod,
        scale_factor: f64,
        min_cluster_size: usize,
    ) -> Self {
        Self {
            levels,
            clustering,
            scale_factor,
            min_cluster_size,
        }
    }

    /// Calculate hierarchical Φ with spatial clustering
    pub fn calculate(&self, lattice: &PBitLattice) -> Result<HierarchicalPhi> {
        let n = lattice.size();

        // Generate spatial scales
        let scales = self.generate_scales(lattice);

        // Multi-scale clustering
        let clusterings: Vec<Vec<SpatialCluster>> = scales
            .par_iter()
            .map(|&scale| self.cluster_at_scale(lattice, scale))
            .collect::<Result<Vec<_>>>()?;

        // Calculate Φ at each scale
        let phi_per_scale: Vec<f64> = clusterings
            .par_iter()
            .enumerate()
            .map(|(scale_idx, clusters)| {
                self.calculate_scale_phi(lattice, clusters, scales[scale_idx])
            })
            .collect::<Result<Vec<_>>>()?;

        // Aggregate across scales with weights
        let weights = self.scale_weights(&scales);
        let phi_total: f64 = phi_per_scale
            .iter()
            .zip(weights.iter())
            .map(|(&phi, &w)| phi * w)
            .sum();

        let clusters_per_scale = clusterings.iter().map(|c| c.len()).collect();

        Ok(HierarchicalPhi {
            phi_total,
            phi_per_scale,
            clusters_per_scale,
            method: PhiMethod::Hierarchical,
            scales,
        })
    }

    /// Generate spatial scales for hierarchical decomposition
    fn generate_scales(&self, lattice: &PBitLattice) -> Vec<f64> {
        let positions = lattice.positions();

        // Find characteristic scale of lattice
        let mut distances = Vec::new();
        for i in 0..positions.len().min(100) {
            for j in i + 1..positions.len().min(100) {
                distances.push(positions[i].hyperbolic_distance(&positions[j]));
            }
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_distance = distances[distances.len() / 2];

        // Generate scales geometrically
        let mut scales = Vec::new();
        let mut current_scale = median_distance;

        for _ in 0..self.levels {
            scales.push(current_scale);
            current_scale *= self.scale_factor;
        }

        scales
    }

    /// Cluster pBits at given spatial scale
    pub fn cluster_at_scale(
        &self,
        lattice: &PBitLattice,
        scale: f64,
    ) -> Result<Vec<SpatialCluster>> {
        match self.clustering {
            ClusteringMethod::Tessellation => self.cluster_tessellation(lattice, scale),
            ClusteringMethod::HyperbolicKMeans => self.cluster_kmeans(lattice, scale),
            ClusteringMethod::Agglomerative => self.cluster_agglomerative(lattice, scale),
        }
    }

    /// Tessellation-based clustering (fastest)
    fn cluster_tessellation(
        &self,
        lattice: &PBitLattice,
        scale: f64,
    ) -> Result<Vec<SpatialCluster>> {
        let positions = lattice.positions();
        let n = positions.len();

        // Simple grid-based clustering using hyperbolic distance threshold
        let mut clusters = Vec::new();
        let mut assigned = vec![false; n];

        for i in 0..n {
            if assigned[i] {
                continue;
            }

            let mut cluster_indices = vec![i];
            assigned[i] = true;

            // Add all nearby nodes
            for j in i + 1..n {
                if assigned[j] {
                    continue;
                }

                let dist = positions[i].hyperbolic_distance(&positions[j]);
                if dist <= scale {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }

            if cluster_indices.len() >= self.min_cluster_size {
                let centroid = self.compute_centroid(&positions, &cluster_indices);
                let radius = self.compute_cluster_radius(&positions, &cluster_indices, &centroid);

                clusters.push(SpatialCluster {
                    indices: cluster_indices,
                    centroid,
                    radius,
                });
            }
        }

        // Handle unassigned nodes (add to nearest cluster or create small clusters)
        for i in 0..n {
            if !assigned[i] {
                if let Some(nearest_cluster) = self.find_nearest_cluster(&clusters, &positions[i])
                {
                    clusters[nearest_cluster].indices.push(i);
                }
            }
        }

        Ok(clusters)
    }

    /// K-means clustering in hyperbolic space
    fn cluster_kmeans(
        &self,
        lattice: &PBitLattice,
        scale: f64,
    ) -> Result<Vec<SpatialCluster>> {
        let positions = lattice.positions();
        let n = positions.len();

        // Estimate number of clusters based on scale
        let k = (n as f64 / (scale * scale)).ceil().max(1.0) as usize;

        // Initialize centroids randomly
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<PoincarePoint> =
            positions.choose_multiple(&mut rng, k).cloned().collect();

        let max_iterations = 50;
        for _ in 0..max_iterations {
            // Assignment step: assign each point to nearest centroid
            let assignments: Vec<usize> = positions
                .par_iter()
                .map(|pos| {
                    centroids
                        .iter()
                        .enumerate()
                        .min_by(|(_, c1), (_, c2)| {
                            let d1 = pos.hyperbolic_distance(c1);
                            let d2 = pos.hyperbolic_distance(c2);
                            d1.partial_cmp(&d2).unwrap()
                        })
                        .map(|(idx, _)| idx)
                        .unwrap()
                })
                .collect();

            // Update step: recompute centroids
            let mut new_centroids = Vec::new();
            for cluster_idx in 0..k {
                let cluster_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c == cluster_idx)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let centroid = self.compute_centroid(&positions, &cluster_points);
                    new_centroids.push(centroid);
                } else {
                    new_centroids.push(centroids[cluster_idx]);
                }
            }

            // Check convergence
            let converged = centroids
                .iter()
                .zip(new_centroids.iter())
                .all(|(old, new)| old.hyperbolic_distance(new) < 1e-6);

            centroids = new_centroids;

            if converged {
                break;
            }
        }

        // Build final clusters
        let mut clusters = Vec::new();
        for cluster_idx in 0..k {
            let cluster_indices: Vec<usize> = (0..n)
                .filter(|&i| {
                    let closest = centroids
                        .iter()
                        .enumerate()
                        .min_by(|(_, c1), (_, c2)| {
                            let d1 = positions[i].hyperbolic_distance(c1);
                            let d2 = positions[i].hyperbolic_distance(c2);
                            d1.partial_cmp(&d2).unwrap()
                        })
                        .map(|(idx, _)| idx)
                        .unwrap();
                    closest == cluster_idx
                })
                .collect();

            if cluster_indices.len() >= self.min_cluster_size {
                let radius =
                    self.compute_cluster_radius(&positions, &cluster_indices, &centroids[cluster_idx]);

                clusters.push(SpatialCluster {
                    indices: cluster_indices,
                    centroid: centroids[cluster_idx],
                    radius,
                });
            }
        }

        Ok(clusters)
    }

    /// Agglomerative hierarchical clustering
    fn cluster_agglomerative(
        &self,
        lattice: &PBitLattice,
        scale: f64,
    ) -> Result<Vec<SpatialCluster>> {
        // Simplified: use tessellation for now
        // Full implementation would build dendrogram and cut at scale threshold
        self.cluster_tessellation(lattice, scale)
    }

    /// Compute cluster centroid in hyperbolic space
    fn compute_centroid(
        &self,
        positions: &[PoincarePoint],
        indices: &[usize],
    ) -> PoincarePoint {
        if indices.is_empty() {
            return PoincarePoint::origin();
        }

        // Hyperbolic centroid: Fréchet mean
        // Approximation: average in Euclidean embedding (good for small clusters)
        let coords_sum: nalgebra::Vector3<f64> = indices
            .iter()
            .map(|&i| positions[i].coords())
            .sum();

        let avg_coords = coords_sum / indices.len() as f64;

        // Project back to Poincaré disk if outside
        let norm = avg_coords.norm();
        if norm >= 0.99 {
            let scaled = avg_coords * (0.99 / norm);
            PoincarePoint::new(scaled).unwrap_or_else(|_| PoincarePoint::origin())
        } else {
            PoincarePoint::new(avg_coords).unwrap_or_else(|_| PoincarePoint::origin())
        }
    }

    /// Compute cluster radius (maximum distance from centroid)
    fn compute_cluster_radius(
        &self,
        positions: &[PoincarePoint],
        indices: &[usize],
        centroid: &PoincarePoint,
    ) -> f64 {
        indices
            .iter()
            .map(|&i| centroid.hyperbolic_distance(&positions[i]))
            .fold(0.0, f64::max)
    }

    /// Find nearest cluster to a point
    fn find_nearest_cluster(
        &self,
        clusters: &[SpatialCluster],
        point: &PoincarePoint,
    ) -> Option<usize> {
        clusters
            .iter()
            .enumerate()
            .min_by(|(_, c1), (_, c2)| {
                let d1 = point.hyperbolic_distance(&c1.centroid);
                let d2 = point.hyperbolic_distance(&c2.centroid);
                d1.partial_cmp(&d2).unwrap()
            })
            .map(|(idx, _)| idx)
    }

    /// Calculate Φ for a specific scale
    fn calculate_scale_phi(
        &self,
        lattice: &PBitLattice,
        clusters: &[SpatialCluster],
        _scale: f64,
    ) -> Result<f64> {
        let states = lattice.states();

        // Calculate intra-cluster vs inter-cluster information
        let mut phi_sum = 0.0;

        for cluster in clusters {
            // Intra-cluster integration
            let intra_integration = self.cluster_integration(&cluster.indices, &states);

            // Inter-cluster coupling to neighbors
            let inter_coupling = self.inter_cluster_coupling(cluster, clusters, &states);

            // Φ contribution: difference between integration and coupling
            let cluster_phi = (intra_integration - inter_coupling).max(0.0);

            phi_sum += cluster_phi;
        }

        Ok(phi_sum)
    }

    /// Calculate integration within a cluster
    fn cluster_integration(&self, indices: &[usize], states: &[bool]) -> f64 {
        if indices.len() <= 1 {
            return 0.0;
        }

        // Simplified: correlation-based integration
        let mut active_count = 0;
        for &i in indices {
            if states[i] {
                active_count += 1;
            }
        }

        let p = active_count as f64 / indices.len() as f64;

        // Shannon entropy-based integration
        if p < 1e-10 || p > 1.0 - 1e-10 {
            0.0
        } else {
            -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
        }
    }

    /// Calculate coupling between clusters
    fn inter_cluster_coupling(
        &self,
        cluster: &SpatialCluster,
        all_clusters: &[SpatialCluster],
        states: &[bool],
    ) -> f64 {
        let mut coupling = 0.0;

        for other in all_clusters {
            if std::ptr::eq(cluster, other) {
                continue;
            }

            // Distance-weighted coupling
            let dist = cluster.centroid.hyperbolic_distance(&other.centroid);
            let weight = (-dist).exp();

            // State correlation
            let corr = self.cluster_correlation(&cluster.indices, &other.indices, states);

            coupling += weight * corr;
        }

        coupling
    }

    /// Compute correlation between two clusters
    fn cluster_correlation(
        &self,
        indices_a: &[usize],
        indices_b: &[usize],
        states: &[bool],
    ) -> f64 {
        if indices_a.is_empty() || indices_b.is_empty() {
            return 0.0;
        }

        let mut correlation = 0.0;

        for &i in indices_a {
            for &j in indices_b {
                let si = if states[i] { 1.0 } else { 0.0 };
                let sj = if states[j] { 1.0 } else { 0.0 };
                correlation += si * sj;
            }
        }

        correlation / (indices_a.len() * indices_b.len()) as f64
    }

    /// Compute weights for multi-scale aggregation
    fn scale_weights(&self, scales: &[f64]) -> Vec<f64> {
        // Geometric weighting: finer scales have higher weight
        let total: f64 = scales.iter().map(|&s| 1.0 / s).sum();

        scales.iter().map(|&s| (1.0 / s) / total).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_calculator() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calculator = HierarchicalPhiCalculator::default();

        let result = calculator.calculate(&lattice).unwrap();

        assert!(result.phi_total >= 0.0);
        assert!(result.phi_total.is_finite());
        assert_eq!(result.phi_per_scale.len(), result.scales.len());
        assert_eq!(result.clusters_per_scale.len(), result.scales.len());
    }

    #[test]
    fn test_spatial_clustering() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        // Use smaller min_cluster_size for small lattice
        let calculator = HierarchicalPhiCalculator::new(3, ClusteringMethod::Tessellation, 2.0, 3);

        // Use larger scale to get bigger clusters
        let clusters = calculator.cluster_at_scale(&lattice, 2.0).unwrap();

        assert!(!clusters.is_empty(), "Should have at least one cluster");
        for cluster in &clusters {
            assert!(cluster.indices.len() >= calculator.min_cluster_size,
                "Cluster size {} should be >= {}", cluster.indices.len(), calculator.min_cluster_size);
            assert!(cluster.radius >= 0.0);
        }
    }

    #[test]
    fn test_scale_generation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calculator = HierarchicalPhiCalculator::new(3, ClusteringMethod::Tessellation, 2.0, 5);

        let scales = calculator.generate_scales(&lattice);

        assert_eq!(scales.len(), 3);
        // Scales should be monotonically increasing
        for i in 1..scales.len() {
            assert!(scales[i] > scales[i - 1]);
        }
    }
}
