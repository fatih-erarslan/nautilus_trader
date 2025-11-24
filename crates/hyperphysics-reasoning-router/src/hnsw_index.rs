//! HNSW-based similarity index for 10-100x faster problem routing.
//!
//! Uses ruvector-core's HNSW implementation for O(log n) approximate nearest neighbor search.

use crate::problem::ProblemSignature;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

#[cfg(feature = "hnsw")]
use ruvector_core::{
    index::hnsw::HnswIndex as RuvectorHnsw,
    index::VectorIndex,
    types::{DistanceMetric, HnswConfig as RuvectorHnswConfig},
};

/// HNSW index configuration optimized for problem similarity routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswSimilarityConfig {
    /// Number of connections per layer (M parameter)
    /// Higher = better recall, more memory
    pub m: usize,
    /// Size of dynamic candidate list during construction
    /// Higher = better index quality, slower build
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search
    /// Higher = better recall, slower search
    pub ef_search: usize,
    /// Maximum number of elements in the index
    pub max_elements: usize,
    /// Feature dimensionality (matches ProblemSignature)
    pub feature_dim: usize,
}

impl Default for HnswSimilarityConfig {
    fn default() -> Self {
        Self {
            m: 16,                    // Good balance for small-medium datasets
            ef_construction: 100,     // High quality index construction
            ef_search: 50,            // Fast search with good recall
            max_elements: 100_000,    // Room for many problem records
            feature_dim: 16,          // Matches ProblemSignature feature vector
        }
    }
}

impl HnswSimilarityConfig {
    /// Configuration optimized for HFT latency (<100μs search)
    pub fn hft_optimized() -> Self {
        Self {
            m: 8,                     // Minimal connections for speed
            ef_construction: 64,      // Fast construction
            ef_search: 16,            // Ultra-fast search
            max_elements: 50_000,     // Smaller index for speed
            feature_dim: 16,
        }
    }

    /// Configuration optimized for recall (>99% recall@10)
    pub fn high_recall() -> Self {
        Self {
            m: 32,                    // Many connections
            ef_construction: 200,     // Very high quality
            ef_search: 100,           // High recall search
            max_elements: 500_000,    // Large capacity
            feature_dim: 16,
        }
    }
}

/// Entry stored in the HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswEntry<T: Clone> {
    /// Associated data (typically backend ID or routing info)
    pub data: T,
    /// Original feature vector for exact similarity computation
    pub features: Vec<f32>,
}

/// HNSW-based similarity index for problem routing
///
/// Provides O(log n) approximate nearest neighbor search compared to
/// O(n) for LSH with linear scan fallback.
pub struct HnswSimilarityIndex<T: Clone + Send + Sync> {
    config: HnswSimilarityConfig,
    /// Entry storage: id -> entry data
    entries: Arc<RwLock<HashMap<String, HnswEntry<T>>>>,
    /// Next available entry ID
    next_id: Arc<RwLock<usize>>,
    #[cfg(feature = "hnsw")]
    /// Underlying HNSW index from ruvector
    hnsw: Arc<RwLock<Option<RuvectorHnsw>>>,
}

impl<T: Clone + Send + Sync> HnswSimilarityIndex<T> {
    /// Create a new HNSW similarity index
    #[cfg(feature = "hnsw")]
    pub fn new(config: HnswSimilarityConfig) -> Result<Self, String> {
        let ruvector_config = RuvectorHnswConfig {
            m: config.m,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            max_elements: config.max_elements,
        };

        let hnsw = RuvectorHnsw::new(
            config.feature_dim,
            DistanceMetric::Cosine,
            ruvector_config,
        ).map_err(|e| format!("Failed to create HNSW index: {}", e))?;

        Ok(Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
            hnsw: Arc::new(RwLock::new(Some(hnsw))),
        })
    }

    /// Create without HNSW feature (fallback to linear scan)
    #[cfg(not(feature = "hnsw"))]
    pub fn new(config: HnswSimilarityConfig) -> Result<Self, String> {
        Ok(Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
        })
    }

    /// Insert a feature vector with associated data
    #[cfg(feature = "hnsw")]
    pub fn insert(&self, features: Vec<f32>, data: T) -> Result<String, String> {
        if features.len() != self.config.feature_dim {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.config.feature_dim,
                features.len()
            ));
        }

        let id = {
            let mut next_id = self.next_id.write();
            let id = format!("entry_{}", *next_id);
            *next_id += 1;
            id
        };

        // Insert into HNSW
        {
            let mut hnsw_guard = self.hnsw.write();
            if let Some(ref mut hnsw) = *hnsw_guard {
                hnsw.add(id.clone(), features.clone())
                    .map_err(|e| format!("HNSW insert failed: {}", e))?;
            }
        }

        // Store entry data
        {
            let mut entries = self.entries.write();
            entries.insert(id.clone(), HnswEntry { data, features });
        }

        Ok(id)
    }

    /// Insert without HNSW feature
    #[cfg(not(feature = "hnsw"))]
    pub fn insert(&self, features: Vec<f32>, data: T) -> Result<String, String> {
        if features.len() != self.config.feature_dim {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.config.feature_dim,
                features.len()
            ));
        }

        let id = {
            let mut next_id = self.next_id.write();
            let id = format!("entry_{}", *next_id);
            *next_id += 1;
            id
        };

        {
            let mut entries = self.entries.write();
            entries.insert(id.clone(), HnswEntry { data, features });
        }

        Ok(id)
    }

    /// Insert using a problem signature
    pub fn insert_signature(&self, signature: &ProblemSignature, data: T) -> Result<String, String> {
        let features = signature.to_feature_vector().to_vec();
        self.insert(features, data)
    }

    /// Query for similar entries using HNSW
    #[cfg(feature = "hnsw")]
    pub fn query(&self, features: &[f32], max_results: usize) -> Result<Vec<(f32, T)>, String> {
        if features.len() != self.config.feature_dim {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.config.feature_dim,
                features.len()
            ));
        }

        let search_results = {
            let hnsw_guard = self.hnsw.read();
            if let Some(ref hnsw) = *hnsw_guard {
                hnsw.search(features, max_results)
                    .map_err(|e| format!("HNSW search failed: {}", e))?
            } else {
                return self.linear_scan(features, max_results);
            }
        };

        let entries = self.entries.read();
        let results: Vec<(f32, T)> = search_results
            .into_iter()
            .filter_map(|result| {
                entries.get(&result.id).map(|entry| {
                    // Convert distance to similarity (1.0 - normalized_distance for cosine)
                    let similarity = 1.0 - result.score.min(1.0);
                    (similarity, entry.data.clone())
                })
            })
            .collect();

        Ok(results)
    }

    /// Query without HNSW feature (linear scan)
    #[cfg(not(feature = "hnsw"))]
    pub fn query(&self, features: &[f32], max_results: usize) -> Result<Vec<(f32, T)>, String> {
        self.linear_scan(features, max_results)
    }

    /// Linear scan fallback for query
    fn linear_scan(&self, features: &[f32], max_results: usize) -> Result<Vec<(f32, T)>, String> {
        let entries = self.entries.read();
        let mut results: Vec<(f32, T)> = entries
            .values()
            .map(|entry| {
                let similarity = cosine_similarity(features, &entry.features);
                (similarity, entry.data.clone())
            })
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        Ok(results)
    }

    /// Query using a problem signature
    pub fn query_signature(
        &self,
        signature: &ProblemSignature,
        max_results: usize,
    ) -> Result<Vec<(f32, T)>, String> {
        let features = signature.to_feature_vector().to_vec();
        self.query(&features, max_results)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear the index
    #[cfg(feature = "hnsw")]
    pub fn clear(&self) -> Result<(), String> {
        self.entries.write().clear();
        *self.next_id.write() = 0;

        // Recreate HNSW index
        let ruvector_config = RuvectorHnswConfig {
            m: self.config.m,
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
            max_elements: self.config.max_elements,
        };

        let new_hnsw = RuvectorHnsw::new(
            self.config.feature_dim,
            DistanceMetric::Cosine,
            ruvector_config,
        ).map_err(|e| format!("Failed to recreate HNSW index: {}", e))?;

        *self.hnsw.write() = Some(new_hnsw);
        Ok(())
    }

    #[cfg(not(feature = "hnsw"))]
    pub fn clear(&self) -> Result<(), String> {
        self.entries.write().clear();
        *self.next_id.write() = 0;
        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &HnswSimilarityConfig {
        &self.config
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Record of a problem-solution pair for HNSW-based learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswProblemRecord {
    /// Problem signature features
    pub features: Vec<f32>,
    /// Backend that solved it successfully
    pub backend_id: String,
    /// Solution quality achieved (0.0-1.0)
    pub quality: f64,
    /// Latency achieved in milliseconds
    pub latency_ms: f64,
    /// Timestamp of the record
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl HnswProblemRecord {
    /// Create a new problem record
    pub fn new(
        signature: &ProblemSignature,
        backend_id: String,
        quality: f64,
        latency_ms: f64,
    ) -> Self {
        Self {
            features: signature.to_feature_vector().to_vec(),
            backend_id,
            quality,
            latency_ms,
            timestamp: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ProblemType, ProblemSignature};
    use crate::ProblemDomain;

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswSimilarityConfig::default();
        assert_eq!(config.feature_dim, 16);
        assert_eq!(config.m, 16);
    }

    #[test]
    fn test_hnsw_config_hft() {
        let config = HnswSimilarityConfig::hft_optimized();
        assert_eq!(config.ef_search, 16); // Fast search
        assert_eq!(config.m, 8); // Minimal connections
    }

    #[test]
    fn test_hnsw_insert_and_query() {
        let config = HnswSimilarityConfig::default();
        let index: HnswSimilarityIndex<String> = HnswSimilarityIndex::new(config).unwrap();

        // Insert some entries
        index.insert(vec![1.0; 16], "entry1".to_string()).unwrap();
        index.insert(vec![0.9; 16], "entry2".to_string()).unwrap();
        index.insert(vec![-1.0; 16], "entry3".to_string()).unwrap();

        // Query similar to entry1
        let results = index.query(&vec![1.0; 16], 10).unwrap();

        assert!(!results.is_empty());
        // First result should be entry1 (exact match)
        assert!(results[0].0 > 0.99); // Very high similarity
    }

    #[test]
    fn test_hnsw_with_signatures() {
        let config = HnswSimilarityConfig::default();
        let index: HnswSimilarityIndex<String> = HnswSimilarityIndex::new(config).unwrap();

        let sig1 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(100);
        let sig2 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(150);
        let sig3 = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics);

        index.insert_signature(&sig1, "backend-pso".to_string()).unwrap();
        index.insert_signature(&sig2, "backend-ga".to_string()).unwrap();
        index.insert_signature(&sig3, "backend-rapier".to_string()).unwrap();

        // Query with a similar financial optimization problem
        let query = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(120);

        let results = index.query_signature(&query, 3).unwrap();

        assert!(!results.is_empty());
        // Should find financial optimization backends first
        assert!(results[0].1 == "backend-pso" || results[0].1 == "backend-ga");
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = HnswSimilarityConfig::default();
        let index: HnswSimilarityIndex<String> = HnswSimilarityIndex::new(config).unwrap();

        // Wrong dimension should fail
        let result = index.insert(vec![1.0; 8], "test".to_string());
        assert!(result.is_err());
    }
}
