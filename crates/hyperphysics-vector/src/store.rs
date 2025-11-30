//! Vector store implementation wrapping ruvector-core

use crate::error::{Result, VectorError};
use ruvector_core::{VectorDB, VectorEntry, SearchQuery, SearchResult, DistanceMetric};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info};

/// Configuration for HyperPhysics vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperVectorConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// Storage path for persistent storage
    pub storage_path: PathBuf,
    /// Distance metric for similarity
    pub distance_metric: DistanceMetric,
    /// HNSW M parameter (max connections per node)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter (default search quality)
    pub hnsw_ef_search: usize,
    /// Enable quantization for memory efficiency
    pub enable_quantization: bool,
    /// Number of bits for quantization (4, 8, or 16)
    pub quantization_bits: u8,
}

impl Default for HyperVectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            storage_path: PathBuf::from("hyperphysics_vectors.db"),
            distance_metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            enable_quantization: false,
            quantization_bits: 8,
        }
    }
}

/// High-performance vector store for HyperPhysics
pub struct HyperVectorStore {
    /// Underlying vector database
    db: Arc<VectorDB>,
    /// Configuration
    config: HyperVectorConfig,
    /// Statistics cache
    stats: Arc<RwLock<StoreStats>>,
}

/// Store statistics
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    /// Total vectors stored
    pub total_vectors: usize,
    /// Total searches performed
    pub total_searches: usize,
    /// Average search latency in microseconds
    pub avg_search_latency_us: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl HyperVectorStore {
    /// Create a new vector store with the given configuration
    pub fn new(config: HyperVectorConfig) -> Result<Self> {
        info!(
            dimensions = config.dimensions,
            path = %config.storage_path.display(),
            "Initializing HyperPhysics vector store"
        );

        let db_options = ruvector_core::types::DbOptions {
            dimensions: config.dimensions,
            storage_path: config.storage_path.to_string_lossy().to_string(),
            distance_metric: config.distance_metric.clone(),
            hnsw_m: config.hnsw_m,
            hnsw_ef_construction: config.hnsw_ef_construction,
            ..Default::default()
        };

        let db = VectorDB::new(db_options)?;

        Ok(Self {
            db: Arc::new(db),
            config,
            stats: Arc::new(RwLock::new(StoreStats::default())),
        })
    }

    /// Insert a vector with metadata
    pub fn insert(
        &self,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        self.validate_dimensions(&vector)?;

        let entry = VectorEntry {
            id: None, // Auto-generate
            vector,
            metadata,
        };

        let id = self.db.insert(entry)?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_vectors += 1;
        }

        debug!(id = %id, "Inserted vector");
        Ok(id.to_string())
    }

    /// Insert a vector with a specific ID
    pub fn insert_with_id(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        self.validate_dimensions(&vector)?;

        let entry = VectorEntry {
            id: Some(id.to_string()),
            vector,
            metadata,
        };

        let result_id = self.db.insert(entry)?;

        {
            let mut stats = self.stats.write();
            stats.total_vectors += 1;
        }

        Ok(result_id.to_string())
    }

    /// Batch insert multiple vectors
    pub fn insert_batch(
        &self,
        entries: Vec<(Vec<f32>, Option<HashMap<String, serde_json::Value>>)>,
    ) -> Result<Vec<String>> {
        let vector_entries: Vec<VectorEntry> = entries
            .into_iter()
            .map(|(vector, metadata)| {
                self.validate_dimensions(&vector)?;
                Ok(VectorEntry {
                    id: None,
                    vector,
                    metadata,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let ids = self.db.insert_batch(vector_entries)?;

        {
            let mut stats = self.stats.write();
            stats.total_vectors += ids.len();
        }

        Ok(ids.into_iter().map(|id| id.to_string()).collect())
    }

    /// Search for similar vectors
    pub fn search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<SearchResult>> {
        self.validate_dimensions(&query_vector)?;

        let start = std::time::Instant::now();

        let query = SearchQuery {
            vector: query_vector,
            k,
            filter,
            ef_search: Some(self.config.hnsw_ef_search),
        };

        let results = self.db.search(query)?;

        let latency = start.elapsed().as_micros() as f64;

        // Update stats with exponential moving average
        {
            let mut stats = self.stats.write();
            stats.total_searches += 1;
            let alpha = 0.1;
            stats.avg_search_latency_us = alpha * latency + (1.0 - alpha) * stats.avg_search_latency_us;
        }

        debug!(k = k, results = results.len(), latency_us = latency, "Search completed");
        Ok(results)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>> {
        self.db.get(id).map_err(VectorError::from)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let deleted = self.db.delete(id)?;

        if deleted {
            let mut stats = self.stats.write();
            stats.total_vectors = stats.total_vectors.saturating_sub(1);
        }

        Ok(deleted)
    }

    /// Get store statistics
    pub fn stats(&self) -> StoreStats {
        self.stats.read().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &HyperVectorConfig {
        &self.config
    }

    /// Validate vector dimensions
    fn validate_dimensions(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.config.dimensions,
                actual: vector.len(),
            });
        }
        Ok(())
    }
}

impl Clone for HyperVectorStore {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_store() -> Result<HyperVectorStore> {
        let dir = tempdir().unwrap();
        let config = HyperVectorConfig {
            dimensions: 4,
            storage_path: dir.path().join("test.db"),
            ..Default::default()
        };
        HyperVectorStore::new(config)
    }

    #[test]
    fn test_insert_and_search() -> Result<()> {
        let store = create_test_store()?;

        // Insert test vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.9, 0.1, 0.0, 0.0];

        store.insert(v1, None)?;
        store.insert(v2, None)?;
        store.insert(v3, None)?;

        // Search for vectors similar to v1
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = store.search(query, 2, None)?;

        assert_eq!(results.len(), 2);
        // First result should be most similar to query

        Ok(())
    }

    #[test]
    fn test_dimension_validation() {
        let store = create_test_store().unwrap();

        // Wrong dimensions should fail
        let wrong_dim = vec![1.0, 0.0, 0.0]; // 3 instead of 4
        let result = store.insert(wrong_dim, None);

        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_batch_insert() -> Result<()> {
        let store = create_test_store()?;

        let entries = vec![
            (vec![1.0, 0.0, 0.0, 0.0], None),
            (vec![0.0, 1.0, 0.0, 0.0], None),
            (vec![0.0, 0.0, 1.0, 0.0], None),
        ];

        let ids = store.insert_batch(entries)?;
        assert_eq!(ids.len(), 3);

        let stats = store.stats();
        assert_eq!(stats.total_vectors, 3);

        Ok(())
    }
}
