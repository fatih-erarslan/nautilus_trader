//! # Search Router and Hybrid Index Manager
//!
//! Unified interface for HNSW and LSH, implementing the triangular
//! constraint architecture.
//!
//! ## Routing Logic
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        QUERY ROUTER                             │
//! │                                                                 │
//! │  ┌─────────────┐                                               │
//! │  │   Query     │                                               │
//! │  │   Input     │                                               │
//! │  └──────┬──────┘                                               │
//! │         │                                                       │
//! │         ▼                                                       │
//! │  ┌──────────────┐                                              │
//! │  │ Mode Select  │                                              │
//! │  │  - Hot       │ ◄─── Sub-μs latency critical                 │
//! │  │  - Stream    │ ◄─── Real-time ingestion                     │
//! │  │  - Whale     │ ◄─── Set similarity (MinHash)                │
//! │  │  - Hybrid    │ ◄─── LSH filter → HNSW refine                │
//! │  └──────┬───────┘                                              │
//! │         │                                                       │
//! │    ┌────┴────┬───────────┬──────────┐                          │
//! │    ▼         ▼           ▼          ▼                          │
//! │  ┌────┐   ┌─────┐   ┌────────┐  ┌────────┐                     │
//! │  │HNSW│   │ LSH │   │MinHash │  │ Both   │                     │
//! │  │Hot │   │Strm │   │ Whale  │  │ Hybrid │                     │
//! │  └─┬──┘   └──┬──┘   └───┬────┘  └───┬────┘                     │
//! │    │         │          │           │                           │
//! │    └─────────┴──────────┴───────────┘                          │
//! │                    │                                            │
//! │                    ▼                                            │
//! │             ┌──────────────┐                                    │
//! │             │   Results    │                                    │
//! │             │   Merger     │                                    │
//! │             └──────────────┘                                    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use hyperphysics_hnsw::{HotIndex, HyperbolicMetric, SearchResult as HnswResult};
use hyperphysics_lsh::{LshResult, StreamingLshIndex};

use crate::config::SearchConfig;
use crate::error::{HybridError, Result};
use crate::PerformanceMetrics;

// ============================================================================
// Types
// ============================================================================

/// Search mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Hot path: direct HNSW query for sub-μs latency.
    Hot,
    
    /// Streaming: LSH for O(1) ingestion.
    Streaming,
    
    /// Whale detection: MinHash for set similarity.
    Whale,
    
    /// Hybrid: LSH filter followed by HNSW refinement.
    Hybrid,
    
    /// Auto: router decides based on query characteristics.
    Auto,
}

/// Unified search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Item ID.
    pub id: u64,
    
    /// Distance or similarity score.
    pub score: f32,
    
    /// Source of the result.
    pub source: ResultSource,
    
    /// Rank in result list.
    pub rank: usize,
}

/// Source of a search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultSource {
    /// From HNSW hot index.
    Hnsw,
    
    /// From LSH index.
    Lsh,
    
    /// From hybrid (LSH filtered, HNSW refined).
    Hybrid,
}

// ============================================================================
// Statistics
// ============================================================================

/// Router statistics.
#[derive(Debug, Default)]
pub struct RouterStats {
    /// Hot queries (HNSW).
    pub hot_queries: AtomicU64,
    
    /// Streaming inserts (LSH).
    pub streaming_inserts: AtomicU64,
    
    /// Whale detection queries (MinHash).
    pub whale_queries: AtomicU64,
    
    /// Hybrid queries.
    pub hybrid_queries: AtomicU64,
    
    /// Patterns promoted from LSH to HNSW.
    pub patterns_promoted: AtomicU64,
    
    /// Total query time (nanoseconds).
    pub total_query_time_ns: AtomicU64,
    
    /// Queries exceeding timeout.
    pub timeouts: AtomicU64,
}

// ============================================================================
// Hybrid Index
// ============================================================================

/// Hybrid index combining HNSW (Processing) and LSH (Acquisition).
///
/// This is the main entry point for the similarity search system.
pub struct HybridIndex {
    /// Configuration.
    config: SearchConfig,
    
    /// HNSW hot index (Processing layer).
    hnsw: RwLock<Option<HotIndex<HyperbolicMetric>>>,
    
    /// LSH streaming index (Acquisition layer).
    lsh: RwLock<Option<StreamingLshIndex>>,
    
    /// Router statistics.
    stats: RouterStats,
    
    /// Vector dimensionality (set on first operation).
    dimensions: AtomicU64,
}

impl HybridIndex {
    /// Create a new hybrid index.
    pub fn new(config: SearchConfig) -> Result<Self> {
        config.validate()?;
        
        Ok(Self {
            config,
            hnsw: RwLock::new(None),
            lsh: RwLock::new(None),
            stats: RouterStats::default(),
            dimensions: AtomicU64::new(0),
        })
    }
    
    /// Initialize the HNSW index with the given dimensionality.
    pub fn init_hnsw(&self, dimensions: usize) -> Result<()> {
        let hnsw_config = self.config.hnsw.clone().with_dimensions(dimensions);
        // HyperbolicMetric::poincare(curvature) - using negative curvature for Poincaré ball model
        let index = HotIndex::new(hnsw_config, HyperbolicMetric::poincare(-1.0))?;
        
        *self.hnsw.write() = Some(index);
        self.dimensions.store(dimensions as u64, Ordering::Relaxed);
        
        tracing::info!(dimensions, "HNSW index initialized");
        Ok(())
    }
    
    /// Initialize the LSH index.
    pub fn init_lsh(&self) -> Result<()> {
        let index = StreamingLshIndex::new(self.config.lsh.clone())?;
        *self.lsh.write() = Some(index);
        
        tracing::info!("LSH index initialized");
        Ok(())
    }
    
    /// Search using the specified mode.
    pub fn search(&self, query: &[f32], k: usize, mode: SearchMode) -> Result<Vec<SearchResult>> {
        let start = Instant::now();
        
        let effective_mode = if mode == SearchMode::Auto {
            self.auto_select_mode(query)
        } else {
            mode
        };
        
        let results = match effective_mode {
            SearchMode::Hot => self.search_hot(query, k)?,
            SearchMode::Streaming => {
                return Err(HybridError::Router {
                    reason: "Streaming mode is for ingestion, not queries".into(),
                });
            }
            SearchMode::Whale => {
                return Err(HybridError::Router {
                    reason: "Whale mode requires set input, use search_whale()".into(),
                });
            }
            SearchMode::Hybrid => self.search_hybrid(query, k)?,
            SearchMode::Auto => unreachable!(),
        };
        
        // Check timeout
        let elapsed_us = start.elapsed().as_micros() as u64;
        if elapsed_us > self.config.router.query_timeout_us {
            self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
            return Err(HybridError::Timeout {
                elapsed_us,
                limit_us: self.config.router.query_timeout_us,
            });
        }
        
        self.stats.total_query_time_ns.fetch_add(
            start.elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
        
        Ok(results)
    }
    
    /// Hot path search (HNSW only).
    ///
    /// Target latency: <1μs.
    pub fn search_hot(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.stats.hot_queries.fetch_add(1, Ordering::Relaxed);
        
        let hnsw_guard = self.hnsw.read();
        let hnsw = hnsw_guard.as_ref().ok_or(HybridError::NotInitialized {
            component: "HNSW".into(),
        })?;
        
        let hnsw_results = hnsw.search(query, k)?;
        
        Ok(hnsw_results
            .into_iter()
            .enumerate()
            .map(|(rank, r)| SearchResult {
                id: r.id,
                score: r.distance,
                source: ResultSource::Hnsw,
                rank,
            })
            .collect())
    }
    
    /// Hybrid search: LSH filter → HNSW refinement.
    pub fn search_hybrid(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.stats.hybrid_queries.fetch_add(1, Ordering::Relaxed);
        
        // Get candidates from LSH
        let lsh_guard = self.lsh.read();
        let lsh = lsh_guard.as_ref().ok_or(HybridError::NotInitialized {
            component: "LSH".into(),
        })?;
        
        let candidate_limit = self.config.router.lsh_candidate_limit;
        let lsh_results = lsh.query(query, candidate_limit)?;
        
        // If few candidates, return LSH results directly
        if lsh_results.len() <= k {
            return Ok(lsh_results
                .into_iter()
                .enumerate()
                .map(|(rank, r)| SearchResult {
                    id: r.id,
                    score: r.estimated_similarity,
                    source: ResultSource::Lsh,
                    rank,
                })
                .collect());
        }
        
        // Refine with HNSW (using candidate IDs for re-ranking)
        // For now, just return top-k from LSH sorted by estimated similarity
        let mut results: Vec<_> = lsh_results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.estimated_similarity,
                source: ResultSource::Hybrid,
                rank: 0,
            })
            .collect();
        
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        results.truncate(k);
        
        // Update ranks
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i;
        }
        
        Ok(results)
    }
    
    /// Stream ingest a vector (goes through LSH Acquisition layer).
    ///
    /// Target latency: <500ns.
    pub fn stream_ingest(&self, vector: Vec<f32>) -> Result<()> {
        self.stats.streaming_inserts.fetch_add(1, Ordering::Relaxed);
        
        let lsh_guard = self.lsh.read();
        let lsh = lsh_guard.as_ref().ok_or(HybridError::NotInitialized {
            component: "LSH".into(),
        })?;
        
        lsh.stream_insert(vector)?;
        Ok(())
    }
    
    /// Insert directly into HNSW hot index.
    pub fn insert_hot(&self, vector: &[f32]) -> Result<u64> {
        let hnsw_guard = self.hnsw.read();
        let hnsw = hnsw_guard.as_ref().ok_or(HybridError::NotInitialized {
            component: "HNSW".into(),
        })?;
        
        let result = hnsw.insert(vector)?;
        Ok(result.id)
    }
    
    /// Promote patterns from LSH to HNSW.
    ///
    /// This is the Acquisition → Processing edge in the triangle.
    pub fn promote_patterns(&self, count: usize) -> Result<usize> {
        let lsh_guard = self.lsh.read();
        let lsh = lsh_guard.as_ref().ok_or(HybridError::NotInitialized {
            component: "LSH".into(),
        })?;
        
        let mut hnsw_guard = self.hnsw.write();
        let hnsw = hnsw_guard.as_mut().ok_or(HybridError::NotInitialized {
            component: "HNSW".into(),
        })?;
        
        // Get high-collision patterns from LSH
        let threshold = self.config.evolution.target_recall;
        
        // Simple promotion: get recent items from LSH
        let mut promoted = 0;
        for i in 0..count.min(lsh.len()) {
            if let Some(item) = lsh.get(i as u64) {
                if hnsw.insert(&item.vector).is_ok() {
                    promoted += 1;
                }
            }
        }
        
        self.stats.patterns_promoted.fetch_add(promoted as u64, Ordering::Relaxed);
        
        Ok(promoted)
    }
    
    /// Get performance metrics.
    pub fn metrics(&self) -> PerformanceMetrics {
        let hot_queries = self.stats.hot_queries.load(Ordering::Relaxed);
        let total_ns = self.stats.total_query_time_ns.load(Ordering::Relaxed);
        
        let avg_latency_ns = if hot_queries > 0 {
            total_ns as f64 / hot_queries as f64
        } else {
            0.0
        };
        
        let hot_pattern_count = self.hnsw.read()
            .as_ref()
            .map(|h| h.len())
            .unwrap_or(0);
        
        PerformanceMetrics {
            avg_latency_ns,
            p99_latency_ns: avg_latency_ns * 2.0, // Estimate
            throughput_qps: if avg_latency_ns > 0.0 {
                1_000_000_000.0 / avg_latency_ns
            } else {
                0.0
            },
            estimated_recall: None,
            hot_pattern_count,
            slow_query_percentage: 0.0, // TODO: implement
        }
    }
    
    /// Get router statistics.
    pub fn stats(&self) -> &RouterStats {
        &self.stats
    }
    
    /// Automatically select search mode based on query characteristics.
    fn auto_select_mode(&self, _query: &[f32]) -> SearchMode {
        // For now, default to hot path if HNSW is initialized and has data
        let hnsw_ready = self.hnsw.read()
            .as_ref()
            .map(|h| h.len() > 0)
            .unwrap_or(false);
        
        if hnsw_ready {
            SearchMode::Hot
        } else {
            SearchMode::Hybrid
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn test_config() -> SearchConfig {
        SearchConfig::trading()
    }
    
    #[test]
    fn test_create_hybrid_index() {
        let index = HybridIndex::new(test_config()).unwrap();
        assert!(index.hnsw.read().is_none()); // Not initialized yet
    }
    
    #[test]
    fn test_init_components() {
        let index = HybridIndex::new(test_config()).unwrap();
        
        index.init_hnsw(64).unwrap();
        index.init_lsh().unwrap();
        
        assert!(index.hnsw.read().is_some());
        assert!(index.lsh.read().is_some());
    }
    
    #[test]
    fn test_search_not_initialized() {
        let index = HybridIndex::new(test_config()).unwrap();
        
        let query = vec![1.0f32; 64];
        let result = index.search(&query, 10, SearchMode::Hot);
        
        assert!(matches!(result, Err(HybridError::NotInitialized { .. })));
    }
    
    #[test]
    fn test_streaming_mode_rejected_for_query() {
        let index = HybridIndex::new(test_config()).unwrap();
        
        let query = vec![1.0f32; 64];
        let result = index.search(&query, 10, SearchMode::Streaming);
        
        assert!(matches!(result, Err(HybridError::Router { .. })));
    }
}
