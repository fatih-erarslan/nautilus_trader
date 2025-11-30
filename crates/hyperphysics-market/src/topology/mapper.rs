//! Market topology mapper using Topological Data Analysis (TDA)
//!
//! Maps market data structures to hyperbolic topological spaces for analysis.
//!
//! # Scientific Foundation
//!
//! This implementation is based on the following peer-reviewed research:
//!
//! 1. **Carlsson, G. (2009).** "Topology and data." *Bulletin of the American Mathematical Society*, 46(2), 255-308.
//!    - Foundational theory of topological data analysis and persistent homology
//!
//! 2. **Ghrist, R. (2008).** "Barcodes: The persistent topology of data." *Bulletin of the American Mathematical Society*, 45(1), 61-75.
//!    - Theory of persistence barcodes for detecting topological features
//!
//! 3. **Gidea, M., & Katz, Y. (2018).** "Topological data analysis of financial time series: Landscapes of crashes."
//!    *Physica A: Statistical Mechanics and its Applications*, 491, 820-834.
//!    - Application of TDA to financial market crash detection
//!
//! 4. **Cannon, J. W., Floyd, W. J., Kenyon, R., & Parry, W. R. (1997).** "Hyperbolic geometry."
//!    *Flavors of Geometry*, 31, 59-115.
//!    - Mathematical foundation for hyperbolic geometry used in tessellation
//!
//! # Implementation
//!
//! The mapper performs the following transformations:
//!
//! 1. **Feature Extraction**: Extract multi-dimensional features from OHLCV bars
//! 2. **Normalization**: Scale features to appropriate ranges for hyperbolic space
//! 3. **Hyperbolic Projection**: Map normalized features to Poincaré disk using {7,3} tessellation
//! 4. **Vietoris-Rips Complex**: Construct simplicial complex for topological analysis
//! 5. **Persistent Homology**: Compute persistence diagrams to identify regime changes

use crate::data::Bar;
use crate::error::{MarketError, MarketResult};
use hyperphysics_geometry::{
    poincare::PoincarePoint,
    tessellation_73::HeptagonalTessellation,
};
use nalgebra as na;
use std::collections::HashMap;

/// Metric type for distance computation in feature space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Hyperbolic distance in Poincaré disk
    Hyperbolic,
}

/// Normalization strategy for feature scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationType {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Z-score standardization (mean=0, std=1)
    ZScore,
    /// Robust scaling using median and IQR
    Robust,
}

/// Configuration for topology mapper
#[derive(Debug, Clone)]
pub struct TopologyConfig {
    /// Dimension of feature space (must be <= 3 for Poincaré projection)
    pub dimension: usize,
    /// Metric for distance computation
    pub metric_type: MetricType,
    /// Normalization method
    pub normalization: NormalizationType,
    /// Maximum filtration radius for Vietoris-Rips complex
    pub max_filtration_radius: f64,
    /// Number of tessellation layers for hyperbolic mapping
    pub tessellation_depth: usize,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            dimension: 3,
            metric_type: MetricType::Hyperbolic,
            normalization: NormalizationType::Robust,
            max_filtration_radius: 1.0,
            tessellation_depth: 2,
        }
    }
}

/// Feature vector extracted from a market bar
#[derive(Debug, Clone)]
pub struct BarFeatures {
    /// Original bar data
    pub bar: Bar,
    /// Extracted feature values
    pub features: Vec<f64>,
    /// Normalized features (after scaling)
    pub normalized: Vec<f64>,
}

/// Simplicial complex edge in Vietoris-Rips construction
#[derive(Debug, Clone)]
pub struct SimplexEdge {
    /// Index of first vertex
    pub v1: usize,
    /// Index of second vertex
    pub v2: usize,
    /// Distance between vertices (birth time in filtration)
    pub distance: f64,
}

/// Persistence pair representing a topological feature
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Dimension of the feature (0=component, 1=loop, 2=void)
    pub dimension: usize,
    /// Birth time (when feature appears)
    pub birth: f64,
    /// Death time (when feature disappears)
    pub death: f64,
    /// Persistence (death - birth)
    pub persistence: f64,
}

impl PersistencePair {
    /// Check if this is a significant feature (persistence above threshold)
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence > threshold
    }
}

/// Topological invariants extracted from persistent homology
#[derive(Debug, Clone)]
pub struct TopologicalInvariants {
    /// 0-dimensional persistence pairs (connected components)
    pub h0_pairs: Vec<PersistencePair>,
    /// 1-dimensional persistence pairs (loops/cycles)
    pub h1_pairs: Vec<PersistencePair>,
    /// Total persistence (sum of all persistence values)
    pub total_persistence: f64,
    /// Number of significant features
    pub num_significant_features: usize,
}

/// Mapper for converting market data to topological representations
///
/// This implements the TDA pipeline described in Carlsson (2009) and Ghrist (2008),
/// with application to financial markets following Gidea & Katz (2018).
pub struct MarketTopologyMapper {
    /// Configuration for topology mapping
    config: TopologyConfig,
    /// Underlying {7,3} hyperbolic tessellation
    tessellation: Option<HeptagonalTessellation>,
}

impl MarketTopologyMapper {
    /// Create new topology mapper with default configuration
    pub fn new() -> Self {
        Self {
            config: TopologyConfig::default(),
            tessellation: None,
        }
    }

    /// Create new topology mapper with custom configuration
    pub fn with_config(config: TopologyConfig) -> Self {
        Self {
            config,
            tessellation: None,
        }
    }

    /// Initialize the hyperbolic tessellation for mapping
    ///
    /// This creates the {7,3} tessellation substrate following Cannon et al. (1997)
    pub fn initialize_tessellation(&mut self) -> MarketResult<()> {
        let tess = HeptagonalTessellation::new_with_algebraic(self.config.tessellation_depth)
            .map_err(|e| MarketError::ConfigError(format!("Tessellation initialization failed: {}", e)))?;

        self.tessellation = Some(tess);
        Ok(())
    }

    /// Extract features from a market bar
    ///
    /// Features extracted (Gidea & Katz, 2018):
    /// 1. Log return: ln(close/open)
    /// 2. Volatility proxy: (high - low) / close
    /// 3. Volume change: volume (normalized separately)
    pub fn extract_features(&self, bar: &Bar) -> Vec<f64> {
        let log_return = (bar.close / bar.open).ln();
        let volatility = (bar.high - bar.low) / bar.close;
        let volume = bar.volume as f64;

        vec![log_return, volatility, volume]
    }

    /// Normalize features using configured normalization strategy
    ///
    /// # Arguments
    /// * `features` - Raw feature vectors
    ///
    /// # Returns
    /// Normalized feature vectors in [0, 1] range for Poincaré disk mapping
    pub fn normalize_features(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if features.is_empty() {
            return Vec::new();
        }

        let dim = features[0].len();
        let mut normalized = vec![vec![0.0; dim]; features.len()];

        // Normalize each dimension independently
        for d in 0..dim {
            let values: Vec<f64> = features.iter().map(|f| f[d]).collect();

            match self.config.normalization {
                NormalizationType::MinMax => {
                    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    if range > 1e-10 {
                        for (i, &val) in values.iter().enumerate() {
                            // Scale to [0, 0.95] to stay inside Poincaré disk
                            normalized[i][d] = 0.95 * (val - min) / range;
                        }
                    }
                }
                NormalizationType::ZScore => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter()
                        .map(|&v| (v - mean).powi(2))
                        .sum::<f64>() / values.len() as f64;
                    let std = variance.sqrt();

                    if std > 1e-10 {
                        for (i, &val) in values.iter().enumerate() {
                            // Transform z-scores to [0, 0.95] using tanh
                            let z = (val - mean) / std;
                            normalized[i][d] = 0.475 * (z.tanh() + 1.0);
                        }
                    }
                }
                NormalizationType::Robust => {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let median = sorted[sorted.len() / 2];
                    let q1 = sorted[sorted.len() / 4];
                    let q3 = sorted[3 * sorted.len() / 4];
                    let iqr = q3 - q1;

                    if iqr > 1e-10 {
                        for (i, &val) in values.iter().enumerate() {
                            // Robust scaling to [0, 0.95]
                            let scaled = (val - median) / iqr;
                            normalized[i][d] = 0.475 * (scaled.tanh() + 1.0);
                        }
                    }
                }
            }
        }

        normalized
    }

    /// Map price bars to point cloud in hyperbolic space
    ///
    /// Implementation follows:
    /// 1. Feature extraction (Gidea & Katz, 2018)
    /// 2. Normalization to [0, 1]³
    /// 3. Projection to Poincaré disk (Cannon et al., 1997)
    ///
    /// # Arguments
    /// * `bars` - OHLCV bar data to map
    ///
    /// # Returns
    /// Point cloud in Poincaré disk coordinates
    pub fn map_bars_to_point_cloud(&self, bars: &[Bar]) -> MarketResult<Vec<PoincarePoint>> {
        if bars.is_empty() {
            return Ok(Vec::new());
        }

        // Extract raw features
        let raw_features: Vec<Vec<f64>> = bars.iter()
            .map(|bar| self.extract_features(bar))
            .collect();

        // Normalize features
        let normalized = self.normalize_features(&raw_features);

        // Project to Poincaré disk
        let mut points = Vec::new();
        for norm_features in normalized {
            // Ensure we have exactly 3 dimensions for Poincaré disk
            let coords = match norm_features.len() {
                1 => na::Vector3::new(norm_features[0], 0.0, 0.0),
                2 => na::Vector3::new(norm_features[0], norm_features[1], 0.0),
                3 => na::Vector3::new(norm_features[0], norm_features[1], norm_features[2]),
                n => {
                    // Project higher dimensions using PCA-like approach (take first 3)
                    na::Vector3::new(
                        norm_features[0],
                        if n > 1 { norm_features[1] } else { 0.0 },
                        if n > 2 { norm_features[2] } else { 0.0 },
                    )
                }
            };

            // Ensure point is inside Poincaré disk
            let norm = coords.norm();
            let safe_coords = if norm >= 0.99 {
                coords * 0.95 / norm
            } else {
                coords
            };

            let point = PoincarePoint::new(safe_coords)
                .map_err(|e| MarketError::DataIntegrityError(format!("Point outside Poincaré disk: {}", e)))?;

            points.push(point);
        }

        Ok(points)
    }

    /// Compute distance matrix for point cloud
    fn compute_distance_matrix(&self, points: &[PoincarePoint]) -> Vec<Vec<f64>> {
        let n = points.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = match self.config.metric_type {
                    MetricType::Hyperbolic => points[i].hyperbolic_distance(&points[j]),
                    MetricType::Euclidean => points[i].distance(&points[j]),
                    MetricType::Manhattan => {
                        let diff = points[i].coords() - points[j].coords();
                        diff.x.abs() + diff.y.abs() + diff.z.abs()
                    }
                };
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Build Vietoris-Rips complex edges from distance matrix
    ///
    /// Following Carlsson (2009), constructs the 1-skeleton of the Vietoris-Rips complex
    /// by connecting points within specified radius.
    fn build_vietoris_rips_edges(&self, distances: &[Vec<f64>]) -> Vec<SimplexEdge> {
        let mut edges = Vec::new();
        let n = distances.len();

        for i in 0..n {
            for j in (i + 1)..n {
                if distances[i][j] <= self.config.max_filtration_radius {
                    edges.push(SimplexEdge {
                        v1: i,
                        v2: j,
                        distance: distances[i][j],
                    });
                }
            }
        }

        // Sort edges by distance (filtration parameter)
        edges.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        edges
    }

    /// Compute 0-dimensional persistent homology (connected components)
    ///
    /// Uses Union-Find algorithm to track component births and deaths
    fn compute_h0_persistence(&self, edges: &[SimplexEdge], num_points: usize) -> Vec<PersistencePair> {
        // Union-Find data structure
        let mut parent: Vec<usize> = (0..num_points).collect();
        let mut rank = vec![0; num_points];
        let birth_time = vec![0.0; num_points]; // All components born at time 0

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        let mut pairs = Vec::new();

        for edge in edges {
            let root1 = find(&mut parent, edge.v1);
            let root2 = find(&mut parent, edge.v2);

            if root1 != root2 {
                // Merge components: the younger one dies
                let (dying_root, _surviving_root) = if birth_time[root1] > birth_time[root2] {
                    (root1, root2)
                } else {
                    (root2, root1)
                };

                pairs.push(PersistencePair {
                    dimension: 0,
                    birth: birth_time[dying_root],
                    death: edge.distance,
                    persistence: edge.distance - birth_time[dying_root],
                });

                // Union by rank
                if rank[root1] < rank[root2] {
                    parent[root1] = root2;
                } else if rank[root1] > rank[root2] {
                    parent[root2] = root1;
                } else {
                    parent[root2] = root1;
                    rank[root1] += 1;
                }
            }
        }

        pairs
    }

    /// Compute 1-dimensional persistent homology (loops/cycles)
    ///
    /// Simplified implementation: detects when cycles form in the graph
    fn compute_h1_persistence(&self, edges: &[SimplexEdge], num_points: usize) -> Vec<PersistencePair> {
        let mut graph: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut pairs = Vec::new();

        for edge in edges {
            // Check if adding this edge creates a cycle
            if graph.contains_key(&edge.v1) && graph.contains_key(&edge.v2) {
                // BFS to check for existing path
                let has_path = self.has_path(&graph, edge.v1, edge.v2, num_points);

                if has_path {
                    // Cycle detected: a 1-dimensional feature is born
                    pairs.push(PersistencePair {
                        dimension: 1,
                        birth: edge.distance,
                        death: self.config.max_filtration_radius, // Lives until end of filtration
                        persistence: self.config.max_filtration_radius - edge.distance,
                    });
                }
            }

            // Add edge to graph
            graph.entry(edge.v1).or_insert_with(Vec::new).push(edge.v2);
            graph.entry(edge.v2).or_insert_with(Vec::new).push(edge.v1);
        }

        pairs
    }

    /// BFS helper to check for path between vertices
    fn has_path(&self, graph: &HashMap<usize, Vec<usize>>, start: usize, end: usize, num_points: usize) -> bool {
        let mut visited = vec![false; num_points];
        let mut queue = vec![start];
        visited[start] = true;

        while let Some(current) = queue.pop() {
            if current == end {
                return true;
            }

            if let Some(neighbors) = graph.get(&current) {
                for &neighbor in neighbors {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }
        }

        false
    }

    /// Compute persistent homology of price action using Vietoris-Rips complex
    ///
    /// Implementation based on:
    /// - Carlsson (2009): Vietoris-Rips complex construction
    /// - Ghrist (2008): Persistence computation
    /// - Gidea & Katz (2018): Financial market application
    ///
    /// # Arguments
    /// * `bars` - Market bar data to analyze
    ///
    /// # Returns
    /// Topological invariants including persistence pairs
    pub fn compute_persistence(&self, bars: &[Bar]) -> MarketResult<TopologicalInvariants> {
        if bars.is_empty() {
            return Ok(TopologicalInvariants {
                h0_pairs: Vec::new(),
                h1_pairs: Vec::new(),
                total_persistence: 0.0,
                num_significant_features: 0,
            });
        }

        // Map bars to point cloud
        let points = self.map_bars_to_point_cloud(bars)?;

        // Compute distance matrix
        let distances = self.compute_distance_matrix(&points);

        // Build Vietoris-Rips complex (1-skeleton)
        let edges = self.build_vietoris_rips_edges(&distances);

        // Compute persistent homology
        let h0_pairs = self.compute_h0_persistence(&edges, points.len());
        let h1_pairs = self.compute_h1_persistence(&edges, points.len());

        // Compute total persistence
        let total_persistence: f64 = h0_pairs.iter().chain(h1_pairs.iter())
            .map(|p| p.persistence)
            .sum();

        // Count significant features (persistence > 10% of max filtration)
        let significance_threshold = self.config.max_filtration_radius * 0.1;
        let num_significant_features = h0_pairs.iter().chain(h1_pairs.iter())
            .filter(|p| p.is_significant(significance_threshold))
            .count();

        Ok(TopologicalInvariants {
            h0_pairs,
            h1_pairs,
            total_persistence,
            num_significant_features,
        })
    }
}

impl Default for MarketTopologyMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bars(n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                Bar::new(
                    "TEST".to_string(),
                    Utc::now(),
                    100.0 + t,
                    105.0 + t,
                    95.0 + t,
                    102.0 + t,
                    1000 + i as u64,
                )
            })
            .collect()
    }

    #[test]
    fn test_mapper_creation() {
        let mapper = MarketTopologyMapper::new();
        assert!(mapper.tessellation.is_none());
        assert_eq!(mapper.config.dimension, 3);
        assert_eq!(mapper.config.metric_type, MetricType::Hyperbolic);
    }

    #[test]
    fn test_custom_config() {
        let config = TopologyConfig {
            dimension: 2,
            metric_type: MetricType::Euclidean,
            normalization: NormalizationType::MinMax,
            max_filtration_radius: 0.5,
            tessellation_depth: 1,
        };

        let mapper = MarketTopologyMapper::with_config(config);
        assert_eq!(mapper.config.dimension, 2);
        assert_eq!(mapper.config.metric_type, MetricType::Euclidean);
    }

    #[test]
    fn test_tessellation_initialization() {
        let mut mapper = MarketTopologyMapper::new();
        let result = mapper.initialize_tessellation();
        assert!(result.is_ok());
        assert!(mapper.tessellation.is_some());

        let tess = mapper.tessellation.as_ref().unwrap();
        assert!(tess.num_tiles() >= 1); // At least central tile
    }

    #[test]
    fn test_feature_extraction() {
        let mapper = MarketTopologyMapper::new();
        let bar = Bar::new(
            "TEST".to_string(),
            Utc::now(),
            100.0,
            110.0,
            95.0,
            105.0,
            5000,
        );

        let features = mapper.extract_features(&bar);
        assert_eq!(features.len(), 3);

        // Log return: ln(105/100)
        assert!((features[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);

        // Volatility: (110 - 95) / 105
        assert!((features[1] - (110.0 - 95.0) / 105.0).abs() < 1e-10);

        // Volume
        assert_eq!(features[2], 5000.0);
    }

    #[test]
    fn test_minmax_normalization() {
        let config = TopologyConfig {
            normalization: NormalizationType::MinMax,
            ..Default::default()
        };
        let mapper = MarketTopologyMapper::with_config(config);

        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let normalized = mapper.normalize_features(&features);

        // First dimension: min=1, max=7, range=6
        // Normalized values should be in [0, 0.95]
        assert!(normalized[0][0] < normalized[1][0]);
        assert!(normalized[1][0] < normalized[2][0]);
        assert!(normalized[2][0] <= 0.95);
    }

    #[test]
    fn test_zscore_normalization() {
        let config = TopologyConfig {
            normalization: NormalizationType::ZScore,
            ..Default::default()
        };
        let mapper = MarketTopologyMapper::with_config(config);

        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ];

        let normalized = mapper.normalize_features(&features);

        // Z-scores transformed via tanh should be in [0, 0.95]
        for norm in &normalized {
            assert!(norm[0] >= 0.0 && norm[0] <= 0.95);
        }
    }

    #[test]
    fn test_robust_normalization() {
        let mapper = MarketTopologyMapper::new(); // Uses Robust by default

        let features = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![100.0], // Outlier
        ];

        let normalized = mapper.normalize_features(&features);

        // Robust scaling should handle outliers better
        for norm in &normalized {
            assert!(norm[0] >= 0.0 && norm[0] <= 0.95);
        }
    }

    #[test]
    fn test_point_cloud_mapping() {
        let mapper = MarketTopologyMapper::new();
        let bars = create_test_bars(5);

        let result = mapper.map_bars_to_point_cloud(&bars);
        assert!(result.is_ok());

        let points = result.unwrap();
        assert_eq!(points.len(), 5);

        // All points should be inside Poincaré disk
        for point in &points {
            assert!(point.norm() < 1.0);
        }
    }

    #[test]
    fn test_empty_bars_handling() {
        let mapper = MarketTopologyMapper::new();
        let bars: Vec<Bar> = Vec::new();

        let result = mapper.map_bars_to_point_cloud(&bars);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_distance_matrix_computation() {
        let mapper = MarketTopologyMapper::new();
        let bars = create_test_bars(3);

        let points = mapper.map_bars_to_point_cloud(&bars).unwrap();
        let distances = mapper.compute_distance_matrix(&points);

        // Distance matrix should be symmetric
        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0].len(), 3);

        // Diagonal should be zero
        for i in 0..3 {
            assert_eq!(distances[i][i], 0.0);
        }

        // Symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[i][j] - distances[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_vietoris_rips_construction() {
        let mapper = MarketTopologyMapper::new();
        let bars = create_test_bars(4);

        let points = mapper.map_bars_to_point_cloud(&bars).unwrap();
        let distances = mapper.compute_distance_matrix(&points);
        let edges = mapper.build_vietoris_rips_edges(&distances);

        // Edges should be sorted by distance
        for i in 1..edges.len() {
            assert!(edges[i].distance >= edges[i - 1].distance);
        }

        // All edges should be within max filtration radius
        for edge in &edges {
            assert!(edge.distance <= mapper.config.max_filtration_radius);
        }
    }

    #[test]
    fn test_h0_persistence_computation() {
        let mapper = MarketTopologyMapper::new();

        // Create simple edge list for testing
        let edges = vec![
            SimplexEdge { v1: 0, v2: 1, distance: 0.1 },
            SimplexEdge { v1: 1, v2: 2, distance: 0.2 },
            SimplexEdge { v1: 0, v2: 3, distance: 0.3 },
        ];

        let pairs = mapper.compute_h0_persistence(&edges, 4);

        // Should have 3 death events (merging 4 components to 1)
        assert_eq!(pairs.len(), 3);

        // All should be 0-dimensional
        for pair in &pairs {
            assert_eq!(pair.dimension, 0);
            assert!(pair.persistence >= 0.0);
        }
    }

    #[test]
    fn test_h1_persistence_computation() {
        let mapper = MarketTopologyMapper::new();

        // Create triangle (cycle)
        let edges = vec![
            SimplexEdge { v1: 0, v2: 1, distance: 0.1 },
            SimplexEdge { v1: 1, v2: 2, distance: 0.2 },
            SimplexEdge { v1: 2, v2: 0, distance: 0.3 }, // Closes the loop
        ];

        let pairs = mapper.compute_h1_persistence(&edges, 3);

        // Should detect at least one 1-dimensional feature (the loop)
        assert!(!pairs.is_empty());

        // All should be 1-dimensional
        for pair in &pairs {
            assert_eq!(pair.dimension, 1);
        }
    }

    #[test]
    fn test_persistence_computation_integration() {
        let mapper = MarketTopologyMapper::new();
        let bars = create_test_bars(10);

        let result = mapper.compute_persistence(&bars);
        assert!(result.is_ok());

        let invariants = result.unwrap();

        // Should have some 0-dimensional features
        assert!(!invariants.h0_pairs.is_empty());

        // Total persistence should be non-negative
        assert!(invariants.total_persistence >= 0.0);
    }

    #[test]
    fn test_persistence_pair_significance() {
        let pair = PersistencePair {
            dimension: 0,
            birth: 0.0,
            death: 0.5,
            persistence: 0.5,
        };

        assert!(pair.is_significant(0.1));
        assert!(!pair.is_significant(0.6));
    }

    #[test]
    fn test_metric_types() {
        let bars = create_test_bars(3);

        // Test Euclidean metric
        let config_euclidean = TopologyConfig {
            metric_type: MetricType::Euclidean,
            ..Default::default()
        };
        let mapper_euclidean = MarketTopologyMapper::with_config(config_euclidean);
        let points = mapper_euclidean.map_bars_to_point_cloud(&bars).unwrap();
        let dist_euclidean = mapper_euclidean.compute_distance_matrix(&points);

        // Test Manhattan metric
        let config_manhattan = TopologyConfig {
            metric_type: MetricType::Manhattan,
            ..Default::default()
        };
        let mapper_manhattan = MarketTopologyMapper::with_config(config_manhattan);
        let dist_manhattan = mapper_manhattan.compute_distance_matrix(&points);

        // Manhattan distance should generally be >= Euclidean
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(dist_manhattan[i][j] >= dist_euclidean[i][j] - 1e-10);
                }
            }
        }

        // Test Hyperbolic metric
        let config_hyperbolic = TopologyConfig {
            metric_type: MetricType::Hyperbolic,
            ..Default::default()
        };
        let mapper_hyperbolic = MarketTopologyMapper::with_config(config_hyperbolic);
        let dist_hyperbolic = mapper_hyperbolic.compute_distance_matrix(&points);

        // All distances should be non-negative
        for i in 0..3 {
            for j in 0..3 {
                assert!(dist_hyperbolic[i][j] >= 0.0);
            }
        }
    }

    #[test]
    fn test_real_market_data_pattern() {
        let mapper = MarketTopologyMapper::new();

        // Simulate market crash pattern (high volatility)
        let crash_bars: Vec<Bar> = (0..20)
            .map(|i| {
                let volatility = if i < 10 { 2.0 } else { 20.0 }; // Spike
                Bar::new(
                    "CRASH".to_string(),
                    Utc::now(),
                    100.0,
                    100.0 + volatility,
                    100.0 - volatility,
                    100.0 + (i as f64 - 10.0),
                    1000 * (i as u64 + 1),
                )
            })
            .collect();

        let result = mapper.compute_persistence(&crash_bars);
        assert!(result.is_ok());

        let invariants = result.unwrap();

        // Should detect topological changes
        assert!(invariants.total_persistence > 0.0);
        assert!(invariants.num_significant_features > 0);
    }

    #[test]
    fn test_stable_market_pattern() {
        let mapper = MarketTopologyMapper::new();

        // Simulate stable market (low volatility)
        let stable_bars: Vec<Bar> = (0..20)
            .map(|i| {
                Bar::new(
                    "STABLE".to_string(),
                    Utc::now(),
                    100.0,
                    100.5,
                    99.5,
                    100.0 + (i as f64 * 0.1),
                    1000,
                )
            })
            .collect();

        let result = mapper.compute_persistence(&stable_bars);
        assert!(result.is_ok());

        // Stable markets should have less complex topology
        let invariants = result.unwrap();
        assert!(invariants.h0_pairs.len() > 0);
    }
}
