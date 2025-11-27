//! # HyperPhysics LSH - Acquisition Layer
//!
//! Zero-allocation Locality Sensitive Hashing for specialized use cases where
//! LSH provides irreplaceable value over HNSW.
//!
//! ## Triangular Architecture Role: ACQUISITION
//!
//! ```text
//!                           ACQUISITION (LSH)
//!                         ╱ Pattern Ingestion  ╲
//!                        ╱   O(1) Insertion     ╲
//!                       ╱  Streaming MinHash     ╲
//!                      ╱                          ╲
//!                     ╱ constrains    constrains   ╲
//!                    ╱   what can      what is      ╲
//!                   ╱    evolve        useful        ╲
//!                  ╱                                  ╲
//!   EVOLUTION ◄────────────────────────────────────────► PROCESSING
//!   (LSH pBit)         constrains each other           (HNSW Hot)
//!   Thermodynamic                                      Sub-μs Queries
//!   Optimization                                       
//! ```
//!
//! ## Irreplaceable LSH Use Cases
//!
//! This crate implements LSH ONLY for scenarios where it provides unique value:
//!
//! ### 1. Streaming Data Ingestion
//! - **O(1) insertion** without index rebuild
//! - HNSW requires O(log n) graph updates per insertion
//! - Critical for real-time market data (1000+ ticks/second)
//!
//! ### 2. Whale Transaction Detection (MinHash)
//! - **Variable-cardinality set similarity** (Jaccard index)
//! - HNSW requires fixed-dimension vectors
//! - Whale transactions have variable feature counts
//!
//! ### 3. pBit Probabilistic Sampling
//! - **Hash collision probability matches pBit activation**
//! - HNSW is deterministic, incompatible with thermodynamic memory
//! - Enables temperature-controlled pattern selection
//!
//! ### 4. FPGA Hardware Acceleration (Future)
//! - **Hash-then-lookup maps to pipeline stages**
//! - Graph traversal has data-dependent control flow
//! - LSH parallelizes naturally on streaming architectures
//!
//! ## Performance Targets
//!
//! | Operation | Target | Allocation |
//! |-----------|--------|------------|
//! | Hash computation | <100ns | Zero |
//! | Single insertion | <500ns | Zero |
//! | Streaming insertion | <200ns | Zero |
//! | Query (when appropriate) | <5μs | Minimal |
//!
//! ## Design Principles
//!
//! 1. **Zero-allocation hot path**: ArrayVec, pre-allocated buffers
//! 2. **Lock-free concurrency**: crossbeam for streaming inserts
//! 3. **SIMD-parallel hashing**: Process multiple hash functions simultaneously
//! 4. **Triangular integration**: Explicit interfaces to Processing and Evolution

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod config;
pub mod error;
pub mod hash;
pub mod index;

pub use config::LshConfig;
pub use error::{LshError, Result};
pub use hash::{HashFamily, MinHash, SimHash, SrpHash};
pub use index::{LshResult, StreamingLshIndex};

// ============================================================================
// Performance Constants
// ============================================================================

/// Maximum hash computation latency in nanoseconds.
pub const HASH_LATENCY_BUDGET_NS: u64 = 100;

/// Maximum single insertion latency in nanoseconds.
pub const INSERT_LATENCY_BUDGET_NS: u64 = 500;

/// Maximum streaming insertion latency in nanoseconds.
pub const STREAMING_INSERT_LATENCY_BUDGET_NS: u64 = 200;

/// Maximum query latency in nanoseconds (only for LSH-appropriate queries).
pub const QUERY_LATENCY_BUDGET_NS: u64 = 5_000;

// ============================================================================
// Default LSH Parameters
// ============================================================================

/// Default number of hash tables.
pub const DEFAULT_NUM_TABLES: usize = 8;

/// Default number of hash functions per table.
pub const DEFAULT_HASHES_PER_TABLE: usize = 4;

/// Default bucket capacity before splitting.
pub const DEFAULT_BUCKET_CAPACITY: usize = 100;

/// Maximum dimensions supported (for fixed-size arrays).
pub const MAX_DIMENSIONS: usize = 1024;

/// Maximum hash signature length in bits.
pub const MAX_SIGNATURE_BITS: usize = 256;

// ============================================================================
// Triangular Architecture Traits
// ============================================================================

/// Constraint that the Acquisition layer places on Processing.
///
/// The Acquisition layer controls what patterns are promoted to the
/// hot index, thereby constraining what the Processing layer operates on.
pub trait AcquisitionConstraint {
    /// Pattern type being promoted.
    type Pattern;
    
    /// Check if a pattern should be promoted to the Processing layer.
    ///
    /// This implements the "constrains what is useful" edge in the triangle.
    fn should_promote(&self, pattern: &Self::Pattern) -> bool;
    
    /// Get the promotion threshold (patterns with similarity above this
    /// are candidates for promotion).
    fn promotion_threshold(&self) -> f32;
    
    /// Get current ingestion rate (patterns per second).
    fn ingestion_rate(&self) -> f64;
}

/// Receiver for patterns promoted from Acquisition to Processing.
pub trait ProcessingReceiver {
    /// Pattern type being received.
    type Pattern;
    
    /// ID type for indexed patterns.
    type PatternId;
    
    /// Receive a batch of promoted patterns.
    ///
    /// Returns the IDs assigned to successfully indexed patterns.
    fn receive_promoted(
        &mut self,
        patterns: &[Self::Pattern],
    ) -> std::result::Result<Vec<Self::PatternId>, String>;
    
    /// Report back which patterns were useful (appeared in query results).
    ///
    /// This feedback helps Acquisition tune its promotion threshold.
    fn useful_patterns(&self) -> &[Self::PatternId];
}

/// Constraint that the Acquisition layer places on Evolution.
///
/// The Acquisition layer controls what patterns are available for
/// thermodynamic optimization.
pub trait EvolutionSource {
    /// Pattern type for evolution.
    type Pattern;
    
    /// Hash signature type.
    type Signature;
    
    /// Get patterns with high collision probability (candidates for evolution).
    fn high_collision_patterns(&self, threshold: f32) -> Vec<Self::Pattern>;
    
    /// Get the hash signature for temperature-controlled selection.
    fn pattern_signature(&self, pattern: &Self::Pattern) -> Self::Signature;
    
    /// Sample patterns according to Boltzmann distribution.
    ///
    /// Higher temperature = more random selection.
    /// Lower temperature = prefer high-collision patterns.
    fn boltzmann_sample(&self, temperature: f32, count: usize) -> Vec<Self::Pattern>;
}

/// Receiver for evolution feedback.
pub trait EvolutionFeedback {
    /// Pattern type.
    type Pattern;
    
    /// Receive feedback on which patterns improved system performance.
    fn receive_evolution_feedback(
        &mut self,
        improved: &[Self::Pattern],
        degraded: &[Self::Pattern],
    );
    
    /// Adjust hash function parameters based on evolution results.
    fn adjust_hash_parameters(&mut self, delta: f32);
}

// ============================================================================
// Feature-gated modules
// ============================================================================

#[cfg(feature = "pbit")]
pub mod pbit;

#[cfg(feature = "fpga")]
pub mod fpga;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constants() {
        // Verify performance budget hierarchy
        assert!(HASH_LATENCY_BUDGET_NS < INSERT_LATENCY_BUDGET_NS);
        assert!(STREAMING_INSERT_LATENCY_BUDGET_NS < INSERT_LATENCY_BUDGET_NS);
        assert!(INSERT_LATENCY_BUDGET_NS < QUERY_LATENCY_BUDGET_NS);
    }
    
    #[test]
    fn test_signature_fits_in_cache_line() {
        // 256 bits = 32 bytes, fits in half a cache line
        assert!(MAX_SIGNATURE_BITS / 8 <= 64);
    }
}
