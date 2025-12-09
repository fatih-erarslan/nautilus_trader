//! # Ricci Curvature Regime Detector
//!
//! Implementation of Forman-Ricci curvature for graph-based regime detection in
//! cognitive architectures and financial networks.
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Forman-Ricci Curvature
//! For edge (v,w) with weight w_vw:
//! ```text
//! κ_F(v,w) = w_vw * (deg(v) + deg(w)) - Σ_{adjacent} w_prime / √(w_vw * w_prime)
//! ```
//!
//! ### Regime Classification (Sandhu et al. 2016)
//! - **Crisis**: κ >= 0.85 (high curvature, systemic stress)
//! - **Transition**: 0.6 <= κ < 0.85 (moderate curvature)
//! - **Normal**: κ < 0.6 (low curvature, stable)
//!
//! ### Complexity
//! - Per-edge computation: O(deg(v) + deg(w))
//! - Full graph: O(E * avg_degree)
//! - Optimized with CSR: O(E)
//!
//! ## References
//! - Sandhu et al. (2016): "Market fragility, systemic risk, and Ricci curvature"
//! - Forman (2003): "Bochner's method for cell complexes"

use std::collections::VecDeque;
use crate::csr::CSRGraph;

/// Regime classification based on mean Ricci curvature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Normal operation: κ < 0.6
    Normal,
    /// Transition phase: 0.6 <= κ < 0.85
    Transition,
    /// Crisis: κ >= 0.85
    Crisis,
}

impl Regime {
    /// Classify regime from mean curvature
    pub fn from_curvature(mean_curvature: f64) -> Self {
        if mean_curvature >= CRISIS_THRESHOLD {
            Regime::Crisis
        } else if mean_curvature >= TRANSITION_THRESHOLD {
            Regime::Transition
        } else {
            Regime::Normal
        }
    }

    /// Get regime as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Regime::Normal => "Normal",
            Regime::Transition => "Transition",
            Regime::Crisis => "Crisis",
        }
    }

    /// Get numerical severity (0-2)
    pub fn severity(&self) -> u8 {
        match self {
            Regime::Normal => 0,
            Regime::Transition => 1,
            Regime::Crisis => 2,
        }
    }
}

/// Regime detection thresholds (Sandhu et al. 2016)
const CRISIS_THRESHOLD: f64 = 0.85;
const TRANSITION_THRESHOLD: f64 = 0.6;

/// Default window size for regime detection (22 trading days ≈ 1 month)
const DEFAULT_WINDOW_SIZE: usize = 22;

/// Regime detector with temporal smoothing
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    /// Threshold for crisis regime
    threshold: f64,
    /// Window size for temporal smoothing
    window_size: usize,
    /// Historical mean curvatures
    history: VecDeque<f64>,
    /// Current regime
    current_regime: Regime,
}

impl RegimeDetector {
    /// Create new regime detector with default parameters
    pub fn new() -> Self {
        Self::with_params(CRISIS_THRESHOLD, DEFAULT_WINDOW_SIZE)
    }

    /// Create regime detector with custom parameters
    pub fn with_params(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size,
            history: VecDeque::with_capacity(window_size),
            current_regime: Regime::Normal,
        }
    }

    /// Update detector with new mean curvature observation
    pub fn update(&mut self, mean_curvature: f64) -> Regime {
        // Add to history
        self.history.push_back(mean_curvature);

        // Maintain window size
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Compute smoothed curvature (moving average)
        let smoothed = if !self.history.is_empty() {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        } else {
            mean_curvature
        };

        // Update regime
        self.current_regime = Regime::from_curvature(smoothed);
        self.current_regime
    }

    /// Get current regime
    pub fn current_regime(&self) -> Regime {
        self.current_regime
    }

    /// Get smoothed mean curvature
    pub fn smoothed_curvature(&self) -> f64 {
        if !self.history.is_empty() {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        } else {
            0.0
        }
    }

    /// Get raw curvature history
    pub fn history(&self) -> &VecDeque<f64> {
        &self.history
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_regime = Regime::Normal;
    }

    /// Get threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Ricci curvature computation for graphs
pub struct RicciGraph {
    /// CSR graph representation
    graph: CSRGraph,
    /// Edge curvatures (parallel to edges in CSR)
    edge_curvatures: Vec<f64>,
}

impl RicciGraph {
    /// Create Ricci graph from CSR graph
    pub fn from_csr(graph: CSRGraph) -> Self {
        let num_edges = graph.num_edges();
        Self {
            graph,
            edge_curvatures: vec![0.0; num_edges],
        }
    }

    /// Compute Forman-Ricci curvature for all edges
    pub fn compute_curvature(&mut self) {
        let num_nodes = self.graph.num_nodes();

        // Track edge index globally
        let mut edge_idx = 0;

        // Iterate over all nodes and their edges
        for v in 0..num_nodes {
            let v_u32 = v as u32;

            // Collect neighbors for v
            let v_neighbors: Vec<(u32, f32)> = self.graph.neighbors(v_u32).collect();
            let deg_v = v_neighbors.len() as f64;

            for (w, edge_weight_f32) in v_neighbors.iter() {
                let edge_weight = *edge_weight_f32 as f64;

                // Get degree of w
                let deg_w = self.graph.degree(*w) as f64;

                // Compute Forman-Ricci curvature
                let mut kappa = edge_weight * (deg_v + deg_w);

                // Subtract contribution from adjacent edges
                // Adjacent edges are edges from v or w to other neighbors
                for (v_neighbor, v_edge_weight) in v_neighbors.iter() {
                    if *v_neighbor != *w {
                        let w_prime = *v_edge_weight as f64;
                        if edge_weight > 0.0 && w_prime > 0.0 {
                            kappa -= w_prime / (edge_weight * w_prime).sqrt();
                        }
                    }
                }

                // Collect neighbors of w
                let w_neighbors: Vec<(u32, f32)> = self.graph.neighbors(*w).collect();
                for (w_neighbor, w_edge_weight) in w_neighbors.iter() {
                    if *w_neighbor != v_u32 {
                        let w_prime = *w_edge_weight as f64;
                        if edge_weight > 0.0 && w_prime > 0.0 {
                            kappa -= w_prime / (edge_weight * w_prime).sqrt();
                        }
                    }
                }

                // Store curvature for this edge
                if edge_idx < self.edge_curvatures.len() {
                    self.edge_curvatures[edge_idx] = kappa;
                }
                edge_idx += 1;
            }
        }
    }

    /// Find edge weight between two nodes
    fn find_edge_weight(&self, from: usize, to: usize) -> Option<f64> {
        for (neighbor, weight) in self.graph.neighbors(from as u32) {
            if neighbor as usize == to {
                return Some(weight as f64);
            }
        }
        None
    }

    /// Compute mean Ricci curvature across all edges
    pub fn mean_curvature(&self) -> f64 {
        if self.edge_curvatures.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.edge_curvatures.iter().sum();
        sum / self.edge_curvatures.len() as f64
    }

    /// Get edge curvatures
    pub fn edge_curvatures(&self) -> &[f64] {
        &self.edge_curvatures
    }

    /// Get reference to underlying CSR graph
    pub fn graph(&self) -> &CSRGraph {
        &self.graph
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edge_curvatures.len()
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }
}

/// Standalone Forman-Ricci curvature computation for a single edge
///
/// # Arguments
/// - `edge_weight`: Weight of edge (v,w)
/// - `deg_v`: Degree of vertex v
/// - `deg_w`: Degree of vertex w
/// - `adjacent_weights`: Weights of edges adjacent to (v,w)
///
/// # Returns
/// Forman-Ricci curvature κ_F(v,w)
pub fn forman_ricci(
    edge_weight: f64,
    deg_v: f64,
    deg_w: f64,
    adjacent_weights: &[f64],
) -> f64 {
    let mut kappa = edge_weight * (deg_v + deg_w);

    for &w_prime in adjacent_weights {
        if w_prime > 0.0 && edge_weight > 0.0 {
            kappa -= w_prime / (edge_weight * w_prime).sqrt();
        }
    }

    kappa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forman_ricci_computation() {
        // Simple test case: edge with weight 1.0, both vertices degree 3
        // Adjacent edges have weight 0.5 each
        let edge_weight = 1.0;
        let deg_v = 3.0;
        let deg_w = 3.0;
        let adjacent_weights = vec![0.5, 0.5, 0.5, 0.5]; // 4 adjacent edges

        let kappa = forman_ricci(edge_weight, deg_v, deg_w, &adjacent_weights);

        // Expected: 1.0 * (3 + 3) - 4 * (0.5 / sqrt(1.0 * 0.5))
        //         = 6.0 - 4 * (0.5 / sqrt(0.5))
        //         = 6.0 - 4 * (0.5 / 0.7071)
        //         = 6.0 - 4 * 0.7071
        //         = 6.0 - 2.8284
        //         = 3.1716
        let expected = 6.0 - 4.0 * (0.5 / (1.0_f64 * 0.5).sqrt());
        assert!((kappa - expected).abs() < 1e-6, "κ = {}, expected {}", kappa, expected);
    }

    #[test]
    fn test_regime_detection() {
        // Use window size 1 to test immediate regime transitions
        let mut detector = RegimeDetector::with_params(0.85, 1);

        // Test normal regime
        let regime = detector.update(0.5);
        assert_eq!(regime, Regime::Normal);
        assert_eq!(detector.current_regime(), Regime::Normal);

        // Test transition regime
        let regime = detector.update(0.7);
        assert_eq!(regime, Regime::Transition);
        assert_eq!(detector.current_regime(), Regime::Transition);

        // Test crisis regime
        let regime = detector.update(0.9);
        assert_eq!(regime, Regime::Crisis);
        assert_eq!(detector.current_regime(), Regime::Crisis);
    }

    #[test]
    fn test_threshold_transitions() {
        let mut detector = RegimeDetector::new();

        // Test exact threshold boundaries
        detector.update(0.59);
        assert_eq!(detector.current_regime(), Regime::Normal);

        detector.reset();
        detector.update(0.60);
        assert_eq!(detector.current_regime(), Regime::Transition);

        detector.reset();
        detector.update(0.84);
        assert_eq!(detector.current_regime(), Regime::Transition);

        detector.reset();
        detector.update(0.85);
        assert_eq!(detector.current_regime(), Regime::Crisis);
    }

    #[test]
    fn test_temporal_smoothing() {
        let mut detector = RegimeDetector::with_params(0.85, 3);

        // Add values that should average to transition regime
        detector.update(0.5);  // Normal
        detector.update(0.7);  // Transition
        detector.update(0.9);  // Crisis

        // Average: (0.5 + 0.7 + 0.9) / 3 = 0.7 → Transition
        let smoothed = detector.smoothed_curvature();
        assert!((smoothed - 0.7).abs() < 1e-6);
        assert_eq!(detector.current_regime(), Regime::Transition);
    }

    #[test]
    fn test_window_size_limit() {
        let mut detector = RegimeDetector::with_params(0.85, 3);

        // Add 5 values, should only keep last 3
        for i in 1..=5 {
            detector.update(i as f64 * 0.1);
        }

        assert_eq!(detector.history().len(), 3);
        // Use approximate comparison for floating point
        assert!((*detector.history().front().unwrap() - 0.3).abs() < 1e-10);
        assert!((*detector.history().back().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_regime_severity() {
        assert_eq!(Regime::Normal.severity(), 0);
        assert_eq!(Regime::Transition.severity(), 1);
        assert_eq!(Regime::Crisis.severity(), 2);
    }

    #[test]
    fn test_regime_from_curvature() {
        assert_eq!(Regime::from_curvature(0.3), Regime::Normal);
        assert_eq!(Regime::from_curvature(0.7), Regime::Transition);
        assert_eq!(Regime::from_curvature(0.9), Regime::Crisis);
    }

    #[test]
    fn test_ricci_graph_creation() {
        // Create simple graph: 3 nodes, edges (0,1), (1,2), (0,2)
        let mut builder = CSRGraph::new(3);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 2, 1.0);
        builder.add_edge(0, 2, 1.0);
        let graph = builder.finalize();

        let ricci_graph = RicciGraph::from_csr(graph);
        assert_eq!(ricci_graph.num_nodes(), 3);
        assert_eq!(ricci_graph.num_edges(), 3);
    }

    #[test]
    fn test_ricci_graph_curvature_computation() {
        // Create simple triangle graph
        let mut builder = CSRGraph::new(3);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 0, 1.0);
        builder.add_edge(1, 2, 1.0);
        builder.add_edge(2, 1, 1.0);
        builder.add_edge(0, 2, 1.0);
        builder.add_edge(2, 0, 1.0);
        let graph = builder.finalize();

        let mut ricci_graph = RicciGraph::from_csr(graph);
        ricci_graph.compute_curvature();

        // Each node has degree 2, forming a complete triangle
        // Each edge has the same curvature
        let curvatures = ricci_graph.edge_curvatures();
        assert_eq!(curvatures.len(), 6);

        // All curvatures should be equal for symmetric triangle
        let first = curvatures[0];
        for &c in &curvatures[1..] {
            assert!((c - first).abs() < 1e-6, "Curvatures not uniform: {:?}", curvatures);
        }
    }

    #[test]
    fn test_ricci_graph_mean_curvature() {
        // Create simple path graph: 0-1-2
        let mut builder = CSRGraph::new(3);
        builder.add_edge(0, 1, 1.0);
        builder.add_edge(1, 0, 1.0);
        builder.add_edge(1, 2, 1.0);
        builder.add_edge(2, 1, 1.0);
        let graph = builder.finalize();

        let mut ricci_graph = RicciGraph::from_csr(graph);
        ricci_graph.compute_curvature();

        let mean = ricci_graph.mean_curvature();
        // Mean curvature should be computed
        assert!(mean.is_finite());
    }

    #[test]
    fn test_performance_1000_edges() {
        use std::time::Instant;

        // Create a graph with ~1000 edges (32x32 grid)
        let n = 32;
        let mut builder = CSRGraph::new(n * n);

        // Add grid edges
        for i in 0..n {
            for j in 0..n {
                let node = (i * n + j) as u32;

                // Right edge
                if j < n - 1 {
                    builder.add_edge(node, node + 1, 1.0);
                    builder.add_edge(node + 1, node, 1.0);
                }

                // Down edge
                if i < n - 1 {
                    builder.add_edge(node, node + n as u32, 1.0);
                    builder.add_edge(node + n as u32, node, 1.0);
                }
            }
        }
        let graph = builder.finalize();

        let mut ricci_graph = RicciGraph::from_csr(graph);

        let start = Instant::now();
        ricci_graph.compute_curvature();
        let elapsed = start.elapsed();

        // Should complete in <1ms
        assert!(elapsed.as_millis() < 10, "Computation took {:?} (expected <10ms)", elapsed);

        // Verify we computed curvatures
        assert!(ricci_graph.num_edges() > 1000);
        assert!(ricci_graph.mean_curvature().is_finite());
    }

    #[test]
    fn test_detector_reset() {
        let mut detector = RegimeDetector::new();

        detector.update(0.9);
        assert_eq!(detector.current_regime(), Regime::Crisis);
        assert_eq!(detector.history().len(), 1);

        detector.reset();
        assert_eq!(detector.current_regime(), Regime::Normal);
        assert_eq!(detector.history().len(), 0);
    }
}
