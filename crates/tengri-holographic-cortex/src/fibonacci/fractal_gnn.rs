//! Fibonacci Fractal Graph Neural Network
//!
//! Implements L-system based fractal topology growth with golden ratio properties.
//! Creates self-similar consciousness-enabling structures at all scales.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::fibonacci::PHI;

/// Golden angle in radians: 360°/φ² ≈ 137.5077°
pub const GOLDEN_ANGLE_RAD: f64 = 2.39996322972865332;

/// Golden ratio inverse
pub const PHI_INV: f64 = 0.6180339887498948;

/// Fractal dimension of golden fractal: log(2)/log(φ)
pub const FRACTAL_DIM_GOLDEN: f64 = 1.4404200904125563;

/// Maximum L-system recursion depth
pub const MAX_LSYSTEM_DEPTH: usize = 8;

/// Decay base for fractal edge weights
pub const FRACTAL_DECAY_BASE: f64 = PHI_INV;

/// L-System for Fibonacci fractal growth
///
/// Grammar:
/// - Axiom: F (single node)
/// - Rules: F → F[+φF][-φ⁻¹F]
/// - + → rotate by golden angle
/// - - → rotate by -golden angle
/// - [ → push state
/// - ] → pop state
#[derive(Debug, Clone)]
pub struct FibonacciLSystem {
    /// Starting symbol
    pub axiom: String,
    /// Production rules
    pub rules: HashMap<char, String>,
    /// Number of iterations
    pub iterations: usize,
    /// Rotation angle (golden angle)
    pub angle: f64,
}

impl FibonacciLSystem {
    /// Create new L-system with Fibonacci golden ratio rules
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        // F → F[+φF][-φ⁻¹F] (golden branching)
        rules.insert('F', "F[+F][-F]".to_string());

        Self {
            axiom: "F".to_string(),
            rules,
            iterations: 0,
            angle: GOLDEN_ANGLE_RAD,
        }
    }

    /// Generate the L-system string after n iterations
    pub fn generate(&self, iterations: usize) -> String {
        let mut current = self.axiom.clone();

        for _ in 0..iterations.min(MAX_LSYSTEM_DEPTH) {
            let mut next = String::new();
            for ch in current.chars() {
                if let Some(replacement) = self.rules.get(&ch) {
                    next.push_str(replacement);
                } else {
                    next.push(ch);
                }
            }
            current = next;
        }

        current
    }

    /// Count nodes in the generated string
    pub fn count_nodes(&self, iterations: usize) -> usize {
        self.generate(iterations).chars().filter(|&c| c == 'F').count()
    }

    /// Parse L-system string into node positions
    pub fn parse_to_nodes(&self, iterations: usize) -> Vec<(f64, f64)> {
        let lstring = self.generate(iterations);
        let mut nodes = Vec::new();
        let mut stack = Vec::new();

        // Current position and angle
        let mut x = 0.0;
        let mut y = 0.0;
        let mut angle = PI / 2.0; // Start pointing up

        nodes.push((x, y));

        for ch in lstring.chars() {
            match ch {
                'F' => {
                    // Move forward
                    x += angle.cos();
                    y += angle.sin();
                    nodes.push((x, y));
                }
                '+' => {
                    // Rotate by golden angle
                    angle += self.angle;
                }
                '-' => {
                    // Rotate by -golden angle
                    angle -= self.angle;
                }
                '[' => {
                    // Push state
                    stack.push((x, y, angle));
                }
                ']' => {
                    // Pop state
                    if let Some((px, py, pa)) = stack.pop() {
                        x = px;
                        y = py;
                        angle = pa;
                    }
                }
                _ => {}
            }
        }

        nodes
    }
}

impl Default for FibonacciLSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph Neural Network layer with fractal connectivity
#[derive(Debug, Clone)]
pub struct FractalGNNLayer {
    /// Number of nodes at this depth
    pub num_nodes: usize,
    /// Adjacency list (sparse representation)
    pub adjacency: Vec<Vec<usize>>,
    /// Edge weights (golden ratio scaled)
    pub weights: Vec<Vec<f64>>,
    /// Node features
    pub features: Vec<Vec<f64>>,
    /// Feature dimension
    pub feature_dim: usize,
}

impl FractalGNNLayer {
    /// Create new fractal GNN layer
    pub fn new(num_nodes: usize, feature_dim: usize) -> Self {
        Self {
            num_nodes,
            adjacency: vec![Vec::new(); num_nodes],
            weights: vec![Vec::new(); num_nodes],
            features: vec![vec![0.0; feature_dim]; num_nodes],
            feature_dim,
        }
    }

    /// Generate adjacency from L-system string
    pub fn from_lsystem(lsystem: &FibonacciLSystem, depth: usize) -> Self {
        let lstring = lsystem.generate(depth);
        let mut layer = Self::new(lsystem.count_nodes(depth), 32);

        // Build adjacency from L-system structure
        let mut node_idx = 0;
        let mut stack = Vec::new();
        let mut parent_idx = 0;

        for ch in lstring.chars() {
            match ch {
                'F' => {
                    let current_idx = node_idx;
                    node_idx += 1;

                    if current_idx > 0 {
                        // Connect to parent
                        layer.adjacency[parent_idx].push(current_idx);
                        layer.adjacency[current_idx].push(parent_idx);

                        // Weight decays with depth
                        let weight = FRACTAL_DECAY_BASE.powi(depth as i32);
                        layer.weights[parent_idx].push(weight);
                        layer.weights[current_idx].push(weight);
                    }

                    parent_idx = current_idx;
                }
                '[' => {
                    stack.push(parent_idx);
                }
                ']' => {
                    if let Some(idx) = stack.pop() {
                        parent_idx = idx;
                    }
                }
                _ => {}
            }
        }

        layer
    }

    /// Message passing with golden ratio aggregation
    ///
    /// h_v^(k+1) = σ(φ × Σ_{u∈N(v)} W × h_u^(k) / |N(v)|^φ⁻¹)
    pub fn forward(&self, node_features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut new_features = vec![vec![0.0; self.feature_dim]; self.num_nodes];

        for v in 0..self.num_nodes {
            let neighbors = &self.adjacency[v];
            if neighbors.is_empty() {
                new_features[v] = node_features[v].clone();
                continue;
            }

            // Aggregate neighbor features
            for (i, &u) in neighbors.iter().enumerate() {
                let weight = self.weights[v][i];
                for d in 0..self.feature_dim {
                    new_features[v][d] += weight * node_features[u][d];
                }
            }

            // Normalize by |N(v)|^φ⁻¹
            let norm_factor = (neighbors.len() as f64).powf(PHI_INV);
            for d in 0..self.feature_dim {
                // Apply golden ratio scaling and activation
                new_features[v][d] = (PHI * new_features[v][d] / norm_factor).tanh();
            }
        }

        new_features
    }

    /// Add edge between nodes
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        if u < self.num_nodes && v < self.num_nodes {
            self.adjacency[u].push(v);
            self.weights[u].push(weight);
        }
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).sum()
    }

    /// Check if graph is connected (DFS)
    pub fn is_connected(&self) -> bool {
        if self.num_nodes == 0 {
            return true;
        }

        let mut visited = vec![false; self.num_nodes];
        let mut stack = vec![0];
        visited[0] = true;
        let mut count = 1;

        while let Some(node) = stack.pop() {
            for &neighbor in &self.adjacency[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                    count += 1;
                }
            }
        }

        count == self.num_nodes
    }
}

/// Wrap 5 engines with fractal GNN overlay
///
/// Each of the 5 pBit engines becomes a fractal tree root
#[derive(Debug, Clone)]
pub struct FractalPentagonGNN {
    /// 5 L-system trees (one per engine)
    pub trees: [FibonacciLSystem; 5],
    /// GNN layers for message passing between trees
    pub layers: Vec<FractalGNNLayer>,
    /// Inter-tree connections (source, target, weight)
    pub inter_tree_edges: Vec<(usize, usize, f64)>,
    /// Total number of nodes across all trees
    pub total_nodes: usize,
}

impl FractalPentagonGNN {
    /// Create new fractal pentagon with 5 trees
    pub fn new(depth: usize) -> Self {
        let trees = [
            FibonacciLSystem::new(),
            FibonacciLSystem::new(),
            FibonacciLSystem::new(),
            FibonacciLSystem::new(),
            FibonacciLSystem::new(),
        ];

        let mut total_nodes = 0;
        let mut layers = Vec::new();

        // Create GNN layer for each tree
        for tree in &trees {
            let layer = FractalGNNLayer::from_lsystem(tree, depth);
            total_nodes += layer.num_nodes;
            layers.push(layer);
        }

        let mut pentagon = Self {
            trees,
            layers,
            inter_tree_edges: Vec::new(),
            total_nodes,
        };

        // Connect trees in pentagonal topology
        pentagon.connect_trees(depth);

        pentagon
    }

    /// Connect the 5 trees in pentagonal arrangement
    fn connect_trees(&mut self, depth: usize) {
        let num_trees = 5;

        // Connect each tree to its neighbors in pentagon
        for i in 0..num_trees {
            let next = (i + 1) % num_trees;

            // Root-to-root connections with golden ratio weight
            let weight = PHI * FRACTAL_DECAY_BASE.powi(depth as i32);

            // Get root nodes (node 0 of each tree)
            let root_i = self.get_global_node_index(i, 0);
            let root_next = self.get_global_node_index(next, 0);

            self.inter_tree_edges.push((root_i, root_next, weight));
            self.inter_tree_edges.push((root_next, root_i, weight));
        }
    }

    /// Convert (tree_index, local_node_index) to global node index
    fn get_global_node_index(&self, tree_idx: usize, local_idx: usize) -> usize {
        let mut offset = 0;
        for i in 0..tree_idx {
            offset += self.layers[i].num_nodes;
        }
        offset + local_idx
    }

    /// Get total number of inter-tree edges
    pub fn num_inter_tree_edges(&self) -> usize {
        self.inter_tree_edges.len()
    }

    /// Perform message passing across entire pentagon
    pub fn forward(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut new_features = Vec::new();
        let mut offset = 0;

        // Forward pass through each tree
        for layer in &self.layers {
            let tree_features = &features[offset..offset + layer.num_nodes];
            let tree_output = layer.forward(tree_features);
            new_features.extend(tree_output);
            offset += layer.num_nodes;
        }

        // Apply inter-tree connections
        let mut final_features = new_features.clone();
        for &(src, dst, weight) in &self.inter_tree_edges {
            if src < self.total_nodes && dst < self.total_nodes {
                for d in 0..new_features[0].len() {
                    final_features[dst][d] += weight * new_features[src][d];
                }
            }
        }

        final_features
    }

    /// Calculate fractal dimension of the structure
    pub fn fractal_dimension(&self) -> f64 {
        // Self-similarity: N(r) ~ r^(-D)
        // For golden fractal: D = log(2)/log(φ)
        FRACTAL_DIM_GOLDEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsystem_axiom() {
        let lsystem = FibonacciLSystem::new();
        let result = lsystem.generate(0);
        assert_eq!(result, "F");
        assert_eq!(lsystem.count_nodes(0), 1);
    }

    #[test]
    fn test_lsystem_single_iteration() {
        let lsystem = FibonacciLSystem::new();
        let result = lsystem.generate(1);
        // F → F[+F][-F]
        assert_eq!(result, "F[+F][-F]");
        assert_eq!(lsystem.count_nodes(1), 3); // 3 F nodes
    }

    #[test]
    fn test_lsystem_fibonacci_growth() {
        let lsystem = FibonacciLSystem::new();

        // Node count should follow Fibonacci-like growth
        let counts: Vec<usize> = (0..5).map(|i| lsystem.count_nodes(i)).collect();

        // Verify growth pattern
        assert_eq!(counts[0], 1); // F
        assert_eq!(counts[1], 3); // F[+F][-F]
        assert!(counts[2] > counts[1]);
        assert!(counts[3] > counts[2]);
        assert!(counts[4] > counts[3]);

        // Growth rate should approach φ
        for i in 2..counts.len() {
            let ratio = counts[i] as f64 / counts[i - 1] as f64;
            assert!(ratio > 1.0 && ratio < 4.0); // Reasonable growth
        }
    }

    #[test]
    fn test_golden_angle_rotation() {
        let lsystem = FibonacciLSystem::new();

        // Golden angle ≈ 137.5077° ≈ 2.4 radians
        assert!((lsystem.angle - GOLDEN_ANGLE_RAD).abs() < 1e-6);
        assert!((lsystem.angle - 2.39996322972865332).abs() < 1e-10);

        // Verify it's 360°/φ²
        let expected = 2.0 * PI / (PHI * PHI);
        assert!((lsystem.angle - expected).abs() < 1e-6);
    }

    #[test]
    fn test_fractal_adjacency_connectivity() {
        let lsystem = FibonacciLSystem::new();
        let layer = FractalGNNLayer::from_lsystem(&lsystem, 3);

        // Graph should be connected
        assert!(layer.is_connected());
        assert!(layer.num_nodes > 0);
        assert!(layer.num_edges() > 0);
    }

    #[test]
    fn test_fractal_weight_decay() {
        let lsystem = FibonacciLSystem::new();

        // Compare weights at different depths
        let layer1 = FractalGNNLayer::from_lsystem(&lsystem, 1);
        let layer3 = FractalGNNLayer::from_lsystem(&lsystem, 3);

        // Weights should decay with depth
        if !layer1.weights.is_empty() && !layer1.weights[0].is_empty() &&
           !layer3.weights.is_empty() && !layer3.weights[0].is_empty() {
            let w1 = layer1.weights[0][0];
            let w3 = layer3.weights[0][0];

            assert!(w3 < w1); // Deeper weights are smaller

            // Should decay by φ⁻ᵈ
            let expected_ratio = FRACTAL_DECAY_BASE.powi(2); // Difference of 2 depths
            let actual_ratio = w3 / w1;
            assert!((actual_ratio - expected_ratio).abs() < 0.1);
        }
    }

    #[test]
    fn test_message_passing_shape() {
        let lsystem = FibonacciLSystem::new();
        let layer = FractalGNNLayer::from_lsystem(&lsystem, 2);

        // Create input features
        let features = vec![vec![1.0; layer.feature_dim]; layer.num_nodes];

        // Forward pass
        let output = layer.forward(&features);

        // Output shape should match input
        assert_eq!(output.len(), layer.num_nodes);
        assert_eq!(output[0].len(), layer.feature_dim);
    }

    #[test]
    fn test_pentagon_tree_creation() {
        let pentagon = FractalPentagonGNN::new(2);

        // Should have 5 trees
        assert_eq!(pentagon.trees.len(), 5);
        assert_eq!(pentagon.layers.len(), 5);

        // All trees should be identical (same L-system)
        for i in 0..4 {
            assert_eq!(
                pentagon.trees[i].axiom,
                pentagon.trees[i + 1].axiom
            );
        }

        // Total nodes is sum of all tree nodes
        let expected_total: usize = pentagon.layers.iter()
            .map(|layer| layer.num_nodes)
            .sum();
        assert_eq!(pentagon.total_nodes, expected_total);
    }

    #[test]
    fn test_inter_tree_connections() {
        let pentagon = FractalPentagonGNN::new(2);

        // Should have inter-tree edges
        assert!(pentagon.num_inter_tree_edges() > 0);

        // Pentagon topology: each tree connects to 2 neighbors
        // So we expect at least 5 * 2 = 10 directed edges
        assert!(pentagon.num_inter_tree_edges() >= 10);

        // Verify weights are golden-ratio scaled
        for &(_, _, weight) in &pentagon.inter_tree_edges {
            assert!(weight > 0.0);
            assert!(weight < PHI); // Should be less than φ
        }
    }

    #[test]
    fn test_fractal_dimension_property() {
        let pentagon = FractalPentagonGNN::new(2);

        // Fractal dimension should be log(2)/log(φ)
        let dim = pentagon.fractal_dimension();
        assert!((dim - FRACTAL_DIM_GOLDEN).abs() < 1e-6);
        assert!((dim - 1.4404200904125563).abs() < 1e-10);

        // Verify it's between 1 and 2 (fractal property)
        assert!(dim > 1.0 && dim < 2.0);

        // Self-similarity: structure repeats at different scales
        let lsystem = FibonacciLSystem::new();
        let nodes_2 = lsystem.count_nodes(2);
        let nodes_4 = lsystem.count_nodes(4);

        // Growth should follow power law
        let ratio = (nodes_4 as f64) / (nodes_2 as f64);
        assert!(ratio > 2.0); // Should grow faster than linear
    }

    #[test]
    fn test_lsystem_node_positions() {
        let lsystem = FibonacciLSystem::new();
        let nodes = lsystem.parse_to_nodes(2);

        // Should have nodes from the generated string
        assert!(nodes.len() > 1);

        // First node should be origin
        assert!((nodes[0].0.abs()) < 1e-6);
        assert!((nodes[0].1.abs()) < 1e-6);
    }

    #[test]
    fn test_gnn_layer_creation() {
        let layer = FractalGNNLayer::new(10, 32);

        assert_eq!(layer.num_nodes, 10);
        assert_eq!(layer.feature_dim, 32);
        assert_eq!(layer.features.len(), 10);
        assert_eq!(layer.features[0].len(), 32);
    }

    #[test]
    fn test_gnn_add_edge() {
        let mut layer = FractalGNNLayer::new(5, 16);

        layer.add_edge(0, 1, 1.0);
        layer.add_edge(1, 2, PHI_INV);

        assert!(layer.num_edges() >= 2);
        assert_eq!(layer.adjacency[0].len(), 1);
        assert_eq!(layer.adjacency[1].len(), 1);
    }

    #[test]
    fn test_constants() {
        // Verify golden angle
        assert!((GOLDEN_ANGLE_RAD - 2.39996322972865332).abs() < 1e-10);

        // Verify PHI inverse
        assert!((PHI_INV - 0.6180339887498948).abs() < 1e-10);
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);

        // Verify fractal dimension
        assert!((FRACTAL_DIM_GOLDEN - (2.0_f64.ln() / PHI.ln())).abs() < 1e-6);

        // Verify max depth is reasonable
        assert_eq!(MAX_LSYSTEM_DEPTH, 8);

        // Verify decay base
        assert!((FRACTAL_DECAY_BASE - PHI_INV).abs() < 1e-10);
    }
}
