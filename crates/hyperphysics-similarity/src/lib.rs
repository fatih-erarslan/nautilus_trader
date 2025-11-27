//! # HyperPhysics Similarity - Unified Search Integration
//!
//! This crate integrates HNSW (Processing) and LSH (Acquisition) into a
//! unified similarity search system following the triangular constraint model.
//!
//! ## Triangular Architecture
//!
//! ```text
//!                           ACQUISITION (LSH)
//!                         ╱ Streaming Ingestion ╲
//!                        ╱   O(1) Insertion      ╲
//!                       ╱  Whale Detection        ╲
//!                      ╱                           ╲
//!                     ╱ promotes        constrains  ╲
//!                    ╱  patterns        evolution    ╲
//!                   ╱                                 ╲
//!   EVOLUTION ◄──────────────────────────────────────► PROCESSING
//!   (pBit/FPGA)        tunes parameters              (HNSW Hot)
//!   Thermodynamic         ◄──────►                   Sub-μs Queries
//!   Optimization       provides feedback             
//! ```
//!
//! ## Usage Pattern
//!
//! ```ignore
//! use hyperphysics_similarity::{HybridIndex, SearchConfig};
//!
//! // Create hybrid index
//! let config = SearchConfig::default();
//! let index = HybridIndex::new(config)?;
//!
//! // Streaming ingestion goes through LSH (Acquisition)
//! index.stream_ingest(market_tick);
//!
//! // Hot queries go through HNSW (Processing)
//! let results = index.search_hot(&query, k)?;
//!
//! // Whale detection uses MinHash in LSH
//! let whale_candidates = index.detect_whales(&transaction_set)?;
//! ```
//!
//! ## Component Responsibilities
//!
//! | Component | Timescale | Responsibility |
//! |-----------|-----------|----------------|
//! | HNSW | Microseconds | Hot path queries |
//! | LSH Streaming | Milliseconds | Pattern ingestion |
//! | LSH MinHash | Milliseconds | Whale detection |
//! | Evolution | Seconds-Hours | Parameter optimization |

#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod router;

pub use config::SearchConfig;
pub use error::{HybridError, Result};
pub use router::{HybridIndex, SearchMode, SearchResult};

// Re-export from component crates
pub use hyperphysics_hnsw::{HotIndex, HyperbolicMetric, IndexConfig};
pub use hyperphysics_lsh::{LshConfig, MinHash, SimHash, StreamingLshIndex};

// ============================================================================
// Architecture Constants
// ============================================================================

/// Hot path query latency budget (1 microsecond).
pub const HOT_PATH_LATENCY_NS: u64 = 1_000;

/// Streaming ingestion latency budget (100 microseconds).
/// LSH streaming ingestion is fast but not as fast as HNSW hot queries.
pub const STREAMING_LATENCY_NS: u64 = 100_000;

/// Pattern promotion interval (milliseconds).
pub const PROMOTION_INTERVAL_MS: u64 = 100;

/// Evolution feedback interval (seconds).
pub const EVOLUTION_INTERVAL_SEC: u64 = 60;

// ============================================================================
// Triangular Integration Traits
// ============================================================================

/// Interface between Acquisition and Processing layers.
///
/// Acquisition promotes high-quality patterns to Processing for hot queries.
pub trait AcquisitionToProcessing {
    /// Pattern type being promoted.
    type Pattern;
    
    /// ID assigned by Processing layer.
    type ProcessingId;
    
    /// Promote a batch of patterns from Acquisition to Processing.
    ///
    /// Returns IDs for successfully promoted patterns.
    fn promote_patterns(
        &mut self,
        patterns: &[Self::Pattern],
    ) -> Result<Vec<Self::ProcessingId>>;
    
    /// Get patterns pending promotion.
    fn pending_promotions(&self) -> usize;
    
    /// Get promotion throughput (patterns per second).
    fn promotion_throughput(&self) -> f64;
}

/// Interface between Processing and Evolution layers.
///
/// Processing reports performance metrics; Evolution tunes parameters.
pub trait ProcessingToEvolution {
    /// Report query latency to Evolution for optimization.
    fn report_latency(&mut self, latency_ns: u64);
    
    /// Report recall metric (fraction of true neighbors found).
    fn report_recall(&mut self, recall: f32);
    
    /// Receive parameter update from Evolution.
    fn apply_evolution_params(&mut self, ef_search: usize) -> Result<()>;
    
    /// Get current performance metrics for Evolution.
    fn get_metrics(&self) -> PerformanceMetrics;
}

/// Interface between Evolution and Acquisition layers.
///
/// Evolution guides which patterns to sample; Acquisition provides candidates.
pub trait EvolutionToAcquisition {
    /// Pattern type.
    type Pattern;
    
    /// Request patterns for thermodynamic sampling.
    fn sample_for_evolution(&self, temperature: f32, count: usize) -> Vec<Self::Pattern>;
    
    /// Report which patterns improved system performance.
    fn report_beneficial_patterns(&mut self, patterns: &[Self::Pattern]);
    
    /// Adjust hash parameters based on evolution feedback.
    fn adjust_acquisition_params(&mut self, adjustment: f32) -> Result<()>;
}

/// Performance metrics for the Processing layer.
#[derive(Clone, Debug, Default)]
pub struct PerformanceMetrics {
    /// Average query latency in nanoseconds.
    pub avg_latency_ns: f64,
    
    /// 99th percentile latency in nanoseconds.
    pub p99_latency_ns: f64,
    
    /// Queries per second.
    pub throughput_qps: f64,
    
    /// Estimated recall (requires ground truth).
    pub estimated_recall: Option<f32>,
    
    /// Total patterns in hot index.
    pub hot_pattern_count: usize,
    
    /// Percentage of queries exceeding latency budget.
    pub slow_query_percentage: f32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_latency_constants_hierarchy() {
        // Hot path should be fastest
        assert!(HOT_PATH_LATENCY_NS <= STREAMING_LATENCY_NS);
        
        // Promotion should be slower than individual operations
        assert!(PROMOTION_INTERVAL_MS > 0);
    }
}
