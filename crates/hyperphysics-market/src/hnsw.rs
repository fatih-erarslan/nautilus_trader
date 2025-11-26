//! HNSW (Hierarchical Navigable Small World) Index Implementation
//!
//! This module implements a production-ready HNSW index for fast approximate
//! nearest neighbor search with SIMD acceleration using simsimd.
//!
//! # References
//!
//! - Malkov, Y. A., & Yashunin, D. A. (2020). "Efficient and robust approximate
//!   nearest neighbor search using Hierarchical Navigable Small World graphs."
//!   IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824-836.
//!   DOI: 10.1109/TPAMI.2018.2889473
//!
//! - Original paper: https://arxiv.org/abs/1603.09320
//!
//! # Algorithm Overview
//!
//! HNSW builds a multi-layer graph structure where:
//! - Layer 0 contains all vectors (densest layer)
//! - Higher layers contain exponentially fewer vectors
//! - Each node connects to M nearest neighbors in each layer
//! - Search navigates from top layer to bottom, refining candidates
//!
//! # Performance Characteristics
//!
//! - **Construction**: O(M * log(N) * D) where N=vectors, D=dimensions, M=connections
//! - **Search**: O(log(N) * D) average case
//! - **Memory**: O(N * M * log(N))
//! - **Target Latency**: <0.5ms for 1M vectors @ ef=50
//! - **Target Recall**: 95%+ @ ef=50
//!
//! # Example
//!
//! ```rust
//! use hyperphysics_market::hnsw::HNSWIndex;
//!
//! // Create index for 128-dimensional vectors
//! let mut index = HNSWIndex::new(128, 16, 200);
//!
//! // Insert vectors
//! let vector1 = vec![0.5; 128];
//! let vector2 = vec![0.3; 128];
//! index.insert(vector1);
//! index.insert(vector2);
//!
//! // Search for 10 nearest neighbors
//! let query = vec![0.4; 128];
//! let results = index.search(&query, 10, 50);
//! ```

use simsimd::SpatialSimilarity;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use parking_lot::RwLock;
use std::sync::Arc;

/// HNSW Index for fast approximate nearest neighbor search.
///
/// This implementation uses SIMD-accelerated distance calculations via simsimd
/// for optimal performance on modern CPU architectures (AVX2, AVX-512, NEON).
///
/// # Thread Safety
///
/// The index uses `Arc<RwLock<>>` for interior mutability, allowing concurrent
/// reads and exclusive writes. Search operations can run in parallel.
#[derive(Clone)]
pub struct HNSWIndex {
    /// Nodes in the graph (vectors with metadata)
    nodes: Arc<RwLock<Vec<HNSWNode>>>,
    /// Layers (layer 0 is densest, contains all nodes)
    layers: Arc<RwLock<Vec<HNSWLayer>>>,
    /// Entry point node ID (highest layer node)
    entry_point: Arc<RwLock<Option<usize>>>,
    /// Max connections per node per layer (M parameter)
    m: usize,
    /// Max connections for layer 0 (M_max_0 = 2*M)
    m_max_0: usize,
    /// Level generation factor: m_L = 1/ln(M)
    ml: f64,
    /// Search expansion factor during construction (ef_construction)
    ef_construction: usize,
    /// Dimension of vectors
    dim: usize,
    /// Random number generator for level selection
    rng: Arc<RwLock<ChaCha8Rng>>,
}

/// Node in the HNSW graph containing a vector and its metadata
#[derive(Clone, Debug)]
pub struct HNSWNode {
    /// Unique node identifier
    pub id: usize,
    /// Feature vector (dimension = index.dim)
    pub vector: Vec<f32>,
    /// Highest layer this node appears in
    pub level: usize,
}

/// Layer in the HNSW graph with adjacency lists
#[derive(Clone, Debug)]
pub struct HNSWLayer {
    /// Adjacency list: node_id -> Vec<neighbor_id>
    /// neighbors[i] contains the IDs of nodes connected to node i
    pub connections: Vec<Vec<usize>>,
}

/// Search result containing node ID and distance to query
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Node identifier
    pub id: usize,
    /// L2 squared distance to query vector
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl HNSWIndex {
    /// Create a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension of vectors to index
    /// * `m` - Maximum number of connections per node per layer (typical: 12-48)
    /// * `ef_construction` - Size of dynamic candidate list during construction (typical: 100-500)
    ///
    /// # Recommended Parameters
    ///
    /// - **High precision**: M=48, ef_construction=500
    /// - **Balanced**: M=16, ef_construction=200
    /// - **Speed-optimized**: M=12, ef_construction=100
    ///
    /// # Example
    ///
    /// ```rust
    /// use hyperphysics_market::hnsw::HNSWIndex;
    ///
    /// // Balanced configuration for 128-d vectors
    /// let index = HNSWIndex::new(128, 16, 200);
    /// ```
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert!(m > 0, "M must be positive");
        assert!(ef_construction >= m, "ef_construction must be >= M");

        let m_max_0 = m * 2; // Layer 0 has more connections
        let ml = 1.0 / (m as f64).ln(); // Level generation multiplier

        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            layers: Arc::new(RwLock::new(Vec::new())),
            entry_point: Arc::new(RwLock::new(None)),
            m,
            m_max_0,
            ml,
            ef_construction,
            dim,
            rng: Arc::new(RwLock::new(ChaCha8Rng::from_entropy())),
        }
    }

    /// Insert a vector into the index.
    ///
    /// This implements Algorithm 1 from Malkov & Yashunin (2020).
    ///
    /// # Arguments
    ///
    /// * `vector` - Feature vector to insert (must have dimension = self.dim)
    ///
    /// # Returns
    ///
    /// The unique identifier assigned to the inserted vector.
    ///
    /// # Panics
    ///
    /// Panics if vector dimension doesn't match index dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hyperphysics_market::hnsw::HNSWIndex;
    ///
    /// let mut index = HNSWIndex::new(128, 16, 200);
    /// let vector = vec![0.5; 128];
    /// let id = index.insert(vector);
    /// println!("Inserted vector with ID: {}", id);
    /// ```
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        assert_eq!(vector.len(), self.dim,
            "Vector dimension {} doesn't match index dimension {}",
            vector.len(), self.dim);

        // Step 1: Determine insertion level using exponential decay
        let level = self.random_level();

        let mut nodes = self.nodes.write();
        let mut layers = self.layers.write();
        let mut entry_point = self.entry_point.write();

        let node_id = nodes.len();

        // Step 2: Create new node
        let node = HNSWNode {
            id: node_id,
            vector: vector.clone(),
            level,
        };
        nodes.push(node);

        // Step 3: Initialize layers if needed
        while layers.len() <= level {
            layers.push(HNSWLayer {
                connections: vec![Vec::new(); nodes.len()],
            });
        }

        // Ensure all layers have space for new node
        for layer in layers.iter_mut() {
            layer.connections.push(Vec::new());
        }

        // Step 4: Handle first insertion
        if entry_point.is_none() {
            *entry_point = Some(node_id);
            return node_id;
        }

        let ep = entry_point.unwrap();

        // Step 5: Search for nearest neighbors at each layer
        // Start from top layer and work down to layer 0
        let mut curr_nearest = vec![SearchResult { id: ep, distance: Self::distance_simd_static(&nodes[ep].vector, &vector) }];

        // Greedy search on layers above insertion level
        for lc in (level + 1..=nodes[ep].level).rev() {
            curr_nearest = self.search_layer_internal(&nodes, &layers, &vector, curr_nearest[0].id, 1, lc);
        }

        // Step 6: Insert node and connect to neighbors at each layer from level down to 0
        for lc in (0..=level).rev() {
            let m_max = if lc == 0 { self.m_max_0 } else { self.m };

            // Find ef_construction nearest neighbors
            let candidates = self.search_layer_internal(&nodes, &layers, &vector, curr_nearest[0].id, self.ef_construction, lc);

            // Select M best neighbors using heuristic (Algorithm 4)
            let neighbors = self.select_neighbors_heuristic(&nodes, &candidates, m_max, lc, &vector);

            // Add bidirectional connections
            for &neighbor_id in &neighbors {
                layers[lc].connections[node_id].push(neighbor_id);
                layers[lc].connections[neighbor_id].push(node_id);

                // Prune neighbor's connections if needed
                let neighbor_m_max = if lc == 0 { self.m_max_0 } else { self.m };
                if layers[lc].connections[neighbor_id].len() > neighbor_m_max {
                    // Prune to M_max best connections
                    let neighbor_vec = &nodes[neighbor_id].vector;
                    let mut neighbor_candidates: Vec<SearchResult> = layers[lc].connections[neighbor_id]
                        .iter()
                        .map(|&id| SearchResult {
                            id,
                            distance: Self::distance_simd_static(&nodes[id].vector, neighbor_vec),
                        })
                        .collect();

                    let pruned = self.select_neighbors_heuristic(&nodes, &neighbor_candidates, neighbor_m_max, lc, neighbor_vec);
                    layers[lc].connections[neighbor_id] = pruned;
                }
            }

            curr_nearest = candidates;
        }

        // Update entry point if new node is at higher level
        if level > nodes[ep].level {
            *entry_point = Some(node_id);
        }

        node_id
    }

    /// Search for k nearest neighbors.
    ///
    /// This implements Algorithm 2 from Malkov & Yashunin (2020).
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (must have dimension = self.dim)
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Size of dynamic candidate list (must be >= k, typical: 50-400)
    ///
    /// # Returns
    ///
    /// Vector of search results sorted by distance (closest first).
    ///
    /// # Panics
    ///
    /// Panics if query dimension doesn't match or ef < k.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hyperphysics_market::hnsw::HNSWIndex;
    ///
    /// let mut index = HNSWIndex::new(128, 16, 200);
    /// index.insert(vec![0.5; 128]);
    /// index.insert(vec![0.3; 128]);
    ///
    /// let query = vec![0.4; 128];
    /// let results = index.search(&query, 10, 50);
    /// ```
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim,
            "Query dimension {} doesn't match index dimension {}",
            query.len(), self.dim);
        assert!(ef >= k, "ef ({}) must be >= k ({})", ef, k);

        let nodes = self.nodes.read();
        let layers = self.layers.read();
        let entry_point = self.entry_point.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.unwrap();
        let ep_level = nodes[ep].level;

        // Step 1: Greedy search from top layer to layer 1
        let mut curr_nearest = vec![SearchResult {
            id: ep,
            distance: Self::distance_simd_static(&nodes[ep].vector, query),
        }];

        for lc in (1..=ep_level).rev() {
            curr_nearest = self.search_layer_internal(&nodes, &layers, query, curr_nearest[0].id, 1, lc);
        }

        // Step 2: Search layer 0 with ef candidates
        let mut results = self.search_layer_internal(&nodes, &layers, query, curr_nearest[0].id, ef, 0);

        // Step 3: Return k nearest
        results.truncate(k);
        results
    }

    /// SIMD-accelerated distance calculation using simsimd.
    ///
    /// Uses L2 squared distance (Euclidean distance without square root).
    /// This is faster and maintains the same ordering as L2 distance.
    ///
    /// # Performance
    ///
    /// - AVX2: ~10-20x faster than scalar
    /// - AVX-512: ~20-40x faster than scalar
    /// - NEON (ARM): ~4-8x faster than scalar
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// L2 squared distance between vectors.
    #[inline(always)]
    fn distance_simd(a: &[f32], b: &[f32]) -> f32 {
        // simsimd returns f64, convert to f32
        SpatialSimilarity::l2sq(a, b)
            .map(|d| d as f32)
            .unwrap_or(f32::MAX)
    }

    /// Static version of distance_simd for use without self reference
    #[inline(always)]
    fn distance_simd_static(a: &[f32], b: &[f32]) -> f32 {
        // simsimd returns f64, convert to f32
        SpatialSimilarity::l2sq(a, b)
            .map(|d| d as f32)
            .unwrap_or(f32::MAX)
    }

    /// Generate random level for new node using exponential decay.
    ///
    /// Implements the level selection algorithm from the paper:
    /// level = floor(-ln(uniform(0,1)) * m_L)
    ///
    /// This creates a geometric distribution where each level has
    /// exponentially fewer nodes than the level below.
    fn random_level(&self) -> usize {
        let mut rng = self.rng.write();
        let uniform: f64 = rng.gen();
        let level = (-uniform.ln() * self.ml).floor() as usize;
        level.min(16) // Cap at 16 layers for practical reasons
    }

    /// Search a specific layer for nearest neighbors.
    ///
    /// Internal implementation used by both insert and search operations.
    /// Uses greedy best-first search with a priority queue.
    fn search_layer_internal(
        &self,
        nodes: &[HNSWNode],
        layers: &[HNSWLayer],
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<SearchResult> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut nearest = BinaryHeap::new();

        let entry_dist = Self::distance_simd_static(&nodes[entry_point].vector, query);
        let entry_result = SearchResult {
            id: entry_point,
            distance: entry_dist,
        };

        candidates.push(std::cmp::Reverse(entry_result.clone()));
        nearest.push(entry_result.clone());
        visited.insert(entry_point);

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            // If current is farther than worst result in nearest, we're done
            if let Some(worst) = nearest.peek() {
                if current.distance > worst.distance {
                    break;
                }
            }

            // Explore neighbors
            if layer < layers.len() && current.id < layers[layer].connections.len() {
                for &neighbor_id in &layers[layer].connections[current.id] {
                    if visited.insert(neighbor_id) {
                        let neighbor_dist = Self::distance_simd_static(&nodes[neighbor_id].vector, query);
                        let neighbor_result = SearchResult {
                            id: neighbor_id,
                            distance: neighbor_dist,
                        };

                        // Add to candidates if better than worst in nearest or nearest not full
                        if nearest.len() < ef || neighbor_dist < nearest.peek().unwrap().distance {
                            candidates.push(std::cmp::Reverse(neighbor_result.clone()));
                            nearest.push(neighbor_result);

                            // Prune nearest if too large
                            if nearest.len() > ef {
                                nearest.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert heap to sorted vector (closest first)
        let mut results: Vec<SearchResult> = nearest.into_iter().collect();
        results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        results
    }

    /// Select M best neighbors using heuristic from Algorithm 4.
    ///
    /// This implements Algorithm 4 from Malkov & Yashunin (2020):
    /// A greedy heuristic that prefers diverse neighbors to create better
    /// graph connectivity and search performance.
    ///
    /// The heuristic works by:
    /// 1. Starting with the nearest candidate
    /// 2. For each subsequent candidate, check if it's closer to the base vector
    ///    than to any already-selected neighbor
    /// 3. This creates diverse connections that improve navigability
    ///
    /// # References
    ///
    /// Algorithm 4 ("SELECT-NEIGHBORS-HEURISTIC") from the HNSW paper.
    fn select_neighbors_heuristic(
        &self,
        nodes: &[HNSWNode],
        candidates: &[SearchResult],
        m: usize,
        _layer: usize,
        base_vector: &[f32],
    ) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|r| r.id).collect();
        }

        // Recalculate distances relative to base_vector for robustness
        // This ensures correctness even if candidates were computed relative to different query
        let mut sorted: Vec<SearchResult> = candidates.iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: Self::distance_simd_static(&nodes[c.id].vector, base_vector),
            })
            .collect();

        // Sort by distance to base_vector (ascending)
        sorted.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        // Apply diversity heuristic: prefer candidates that are closer to base_vector
        // than to already-selected neighbors
        let mut selected = Vec::with_capacity(m);
        let mut discarded = Vec::new();

        // Always take the closest candidate first
        if !sorted.is_empty() {
            selected.push(sorted[0].id);
        }

        // For remaining candidates, apply diversity check
        for candidate in sorted.iter().skip(1) {
            if selected.len() >= m {
                break;
            }

            // Check if candidate is closer to base_vector than to any selected neighbor
            let dist_to_base = candidate.distance;
            let mut is_diverse = true;

            for &selected_id in &selected {
                // Calculate distance from candidate to selected neighbor
                let dist_to_neighbor = Self::distance_simd_static(
                    &nodes[candidate.id].vector,
                    &nodes[selected_id].vector
                );

                // If candidate is closer to a neighbor than to base, it's not diverse
                if dist_to_neighbor < dist_to_base {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(candidate.id);
            } else {
                discarded.push(candidate.id);
            }
        }

        // If we don't have M neighbors yet, fill from discarded (nearest first)
        if selected.len() < m {
            let needed = m - selected.len();
            selected.extend(discarded.iter().take(needed));
        }

        selected
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Get statistics about the index structure
    pub fn stats(&self) -> HNSWStats {
        let nodes = self.nodes.read();
        let layers = self.layers.read();

        let mut level_counts = vec![0; layers.len()];
        let mut total_connections = 0;

        for node in nodes.iter() {
            for level in 0..=node.level {
                level_counts[level] += 1;
            }
        }

        for (level, layer) in layers.iter().enumerate() {
            for connections in &layer.connections {
                total_connections += connections.len();
            }
        }

        HNSWStats {
            total_nodes: nodes.len(),
            total_layers: layers.len(),
            nodes_per_layer: level_counts,
            total_connections,
            avg_connections_per_node: if nodes.is_empty() {
                0.0
            } else {
                total_connections as f64 / nodes.len() as f64
            },
        }
    }
}

/// Statistics about the HNSW index structure
#[derive(Debug, Clone)]
pub struct HNSWStats {
    /// Total number of nodes in the index
    pub total_nodes: usize,
    /// Number of layers in the index
    pub total_layers: usize,
    /// Number of nodes at each layer
    pub nodes_per_layer: Vec<usize>,
    /// Total number of edges in all layers
    pub total_connections: usize,
    /// Average number of connections per node
    pub avg_connections_per_node: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hnsw_creation() {
        let index = HNSWIndex::new(128, 16, 200);
        assert_eq!(index.dim, 128);
        assert_eq!(index.m, 16);
        assert_eq!(index.m_max_0, 32);
        assert_eq!(index.ef_construction, 200);
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_insertion() {
        let mut index = HNSWIndex::new(4, 16, 200);
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let id = index.insert(vector);
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_multiple_insertions() {
        let mut index = HNSWIndex::new(3, 16, 200);
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        for (i, vec) in vectors.iter().enumerate() {
            let id = index.insert(vec.clone());
            assert_eq!(id, i);
        }

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_search_exact_match() {
        let mut index = HNSWIndex::new(3, 16, 200);
        let vector = vec![1.0, 2.0, 3.0];
        index.insert(vector.clone());

        let results = index.search(&vector, 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert_relative_eq!(results[0].distance, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_search_nearest_neighbors() {
        let mut index = HNSWIndex::new(2, 16, 200);

        // Insert points in a grid
        index.insert(vec![0.0, 0.0]); // id=0
        index.insert(vec![1.0, 0.0]); // id=1
        index.insert(vec![0.0, 1.0]); // id=2
        index.insert(vec![1.0, 1.0]); // id=3

        // Query near (0.1, 0.1) - closest should be (0, 0)
        let query = vec![0.1, 0.1];
        let results = index.search(&query, 2, 10);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // (0, 0) is closest
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn test_simd_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // L2^2 = (1-5)^2 + (2-6)^2 + (3-7)^2 + (4-8)^2 = 16 + 16 + 16 + 16 = 64
        let dist = HNSWIndex::distance_simd(&a, &b);
        assert_relative_eq!(dist, 64.0, epsilon = 1e-4);
    }

    #[test]
    fn test_index_stats() {
        let mut index = HNSWIndex::new(4, 16, 200);

        for i in 0..100 {
            index.insert(vec![i as f32, i as f32, i as f32, i as f32]);
        }

        let stats = index.stats();
        assert_eq!(stats.total_nodes, 100);
        assert!(stats.total_layers > 0);
        assert!(stats.avg_connections_per_node > 0.0);
    }

    #[test]
    #[should_panic(expected = "Dimension must be positive")]
    fn test_zero_dimension_panics() {
        HNSWIndex::new(0, 16, 200);
    }

    #[test]
    #[should_panic(expected = "ef_construction must be >= M")]
    fn test_invalid_ef_construction_panics() {
        HNSWIndex::new(128, 16, 10);
    }

    #[test]
    #[should_panic(expected = "doesn't match index dimension")]
    fn test_wrong_dimension_insert_panics() {
        let mut index = HNSWIndex::new(4, 16, 200);
        index.insert(vec![1.0, 2.0]); // Wrong dimension
    }

    #[test]
    #[should_panic(expected = "doesn't match index dimension")]
    fn test_wrong_dimension_search_panics() {
        let mut index = HNSWIndex::new(4, 16, 200);
        index.insert(vec![1.0, 2.0, 3.0, 4.0]);
        index.search(&[1.0, 2.0], 1, 10); // Wrong dimension
    }
}
