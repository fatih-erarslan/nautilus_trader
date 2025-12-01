//! # Similarity Search Bridge
//!
//! Integration bridge between the cortical bus and `hyperphysics-similarity`.
//!
//! Provides content-addressable memory for the cortical bus via:
//! - **HNSW (Processing)**: Sub-microsecond hot path queries
//! - **LSH (Acquisition)**: Streaming pattern ingestion
//! - **Hybrid Routing**: Automatic query routing based on freshness
//!
//! ## Architecture
//!
//! ```text
//! Cortical Bus
//!      │
//!      ▼
//! ┌─────────────────────────────────────────────────┐
//! │              Similarity Bridge                   │
//! │                                                  │
//! │  ┌──────────────┐      ┌──────────────────────┐ │
//! │  │ Hot Queries  │      │ Pattern Ingestion    │ │
//! │  │ (HNSW)       │      │ (LSH Streaming)      │ │
//! │  │ <1µs         │      │ ~100µs               │ │
//! │  └──────┬───────┘      └──────────┬───────────┘ │
//! │         │                          │            │
//! │         ▼                          ▼            │
//! │  ┌─────────────────────────────────────────────┐│
//! │  │           HybridIndex                        ││
//! │  │   HNSW ←──promotes── LSH                    ││
//! │  └─────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────┘
//! ```

use hyperphysics_similarity::{
    HybridIndex, SearchConfig, SearchMode, SearchResult,
    HOT_PATH_LATENCY_NS, STREAMING_LATENCY_NS,
};
use hyperphysics_hnsw::IndexConfig as HnswConfig;
use hyperphysics_lsh::LshConfig;

use crate::spike::Spike;
use crate::error::{CorticalError, Result};

/// Configuration for similarity bridge.
#[derive(Debug, Clone)]
pub struct SimilarityBridgeConfig {
    /// Embedding dimension.
    pub dim: usize,
    /// HNSW M parameter (connections per node).
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter.
    pub hnsw_ef_search: usize,
    /// Number of LSH tables.
    pub lsh_num_tables: usize,
    /// Number of hash functions per table.
    pub lsh_num_hashes: usize,
    /// Similarity threshold for promotion from LSH to HNSW.
    pub promotion_threshold: f32,
}

impl Default for SimilarityBridgeConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
            lsh_num_tables: 8,
            lsh_num_hashes: 16,
            promotion_threshold: 0.8,
        }
    }
}

/// Bridge between cortical bus and similarity search.
///
/// Provides content-addressable memory with:
/// - Sub-microsecond hot path queries via HNSW
/// - Streaming pattern ingestion via LSH
/// - Automatic promotion of frequently accessed patterns
pub struct SimilarityBridge {
    /// Hybrid index (HNSW + LSH).
    index: HybridIndex,
    /// Configuration.
    config: SimilarityBridgeConfig,
    /// Query statistics.
    stats: SimilarityStats,
}

/// Statistics for similarity operations.
#[derive(Debug, Default, Clone)]
pub struct SimilarityStats {
    /// Total queries performed.
    pub total_queries: u64,
    /// Queries that hit HNSW (hot path).
    pub hnsw_hits: u64,
    /// Queries that fell back to LSH.
    pub lsh_fallbacks: u64,
    /// Patterns ingested.
    pub patterns_ingested: u64,
    /// Patterns promoted from LSH to HNSW.
    pub patterns_promoted: u64,
}

impl SimilarityBridge {
    /// Create a new similarity bridge.
    pub fn new(config: SimilarityBridgeConfig) -> Result<Self> {
        let search_config = SearchConfig::default();
        let index = HybridIndex::new(search_config)
            .map_err(|e| CorticalError::ConfigError(format!("Similarity index: {}", e)))?;

        Ok(Self {
            index,
            config,
            stats: SimilarityStats::default(),
        })
    }

    /// Search for k nearest neighbors.
    ///
    /// Automatically routes to HNSW for hot queries, LSH for cold queries.
    /// Target latency: <1µs for hot path.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.stats.total_queries += 1;

        // Try hot path first (HNSW)
        match self.index.search_hot(query, k) {
            Ok(results) if !results.is_empty() => {
                self.stats.hnsw_hits += 1;
                Ok(results)
            }
            Ok(_) | Err(_) => {
                // Fall back to LSH
                self.stats.lsh_fallbacks += 1;
                self.index
                    .search_cold(query, k)
                    .map_err(CorticalError::from)
            }
        }
    }

    /// Search with explicit mode selection.
    pub fn search_mode(
        &mut self,
        query: &[f32],
        k: usize,
        mode: SearchMode,
    ) -> Result<Vec<SearchResult>> {
        self.stats.total_queries += 1;

        match mode {
            SearchMode::Hot => {
                self.stats.hnsw_hits += 1;
                self.index.search_hot(query, k).map_err(CorticalError::from)
            }
            SearchMode::Cold => {
                self.stats.lsh_fallbacks += 1;
                self.index.search_cold(query, k).map_err(CorticalError::from)
            }
            SearchMode::Auto => self.search(query, k),
        }
    }

    /// Ingest a new pattern via streaming LSH.
    ///
    /// Patterns are initially stored in LSH, then promoted to HNSW
    /// based on access frequency.
    pub fn ingest(&mut self, id: u64, embedding: &[f32]) -> Result<()> {
        self.stats.patterns_ingested += 1;
        self.index
            .stream_ingest(id, embedding)
            .map_err(CorticalError::from)
    }

    /// Ingest a batch of patterns.
    pub fn ingest_batch(&mut self, patterns: &[(u64, Vec<f32>)]) -> Result<usize> {
        let mut count = 0;
        for (id, embedding) in patterns {
            self.ingest(*id, embedding)?;
            count += 1;
        }
        Ok(count)
    }

    /// Convert spike pattern to embedding for storage/query.
    ///
    /// Uses spike metadata to create a searchable embedding.
    pub fn spike_to_embedding(&self, spikes: &[Spike]) -> Vec<f32> {
        // Simple embedding: histogram of spike sources modulated by strength
        let mut embedding = vec![0.0f32; self.config.dim];
        
        for spike in spikes {
            let idx = (spike.source_id as usize) % self.config.dim;
            embedding[idx] += spike.normalized_strength();
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Ingest a spike pattern.
    pub fn ingest_spike_pattern(&mut self, id: u64, spikes: &[Spike]) -> Result<()> {
        let embedding = self.spike_to_embedding(spikes);
        self.ingest(id, &embedding)
    }

    /// Query with a spike pattern.
    pub fn query_spike_pattern(&mut self, spikes: &[Spike], k: usize) -> Result<Vec<SearchResult>> {
        let embedding = self.spike_to_embedding(spikes);
        self.search(&embedding, k)
    }

    /// Get number of patterns in index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &SimilarityStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SimilarityStats::default();
    }

    /// Get hit rate (HNSW hits / total queries).
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_queries > 0 {
            self.stats.hnsw_hits as f64 / self.stats.total_queries as f64
        } else {
            0.0
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &SimilarityBridgeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_bridge_creation() {
        let config = SimilarityBridgeConfig::default();
        let bridge = SimilarityBridge::new(config).unwrap();
        assert!(bridge.is_empty());
    }

    #[test]
    fn test_spike_to_embedding() {
        let config = SimilarityBridgeConfig::default();
        let bridge = SimilarityBridge::new(config).unwrap();

        let spikes = vec![
            Spike::excitatory(10, 100, 0),
            Spike::inhibitory(20, 101, 0),
        ];

        let embedding = bridge.spike_to_embedding(&spikes);
        assert_eq!(embedding.len(), 128);
        
        // Should be normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pattern_ingestion() {
        let config = SimilarityBridgeConfig::default();
        let mut bridge = SimilarityBridge::new(config).unwrap();

        let embedding = vec![0.1f32; 128];
        bridge.ingest(1, &embedding).unwrap();

        assert_eq!(bridge.stats().patterns_ingested, 1);
    }
}
