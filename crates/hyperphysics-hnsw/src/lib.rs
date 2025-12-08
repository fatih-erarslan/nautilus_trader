//! # HyperPhysics HNSW - The Processing Layer
//!
//! This crate implements the **Processing** component of the triangular constraint
//! architecture inspired by Christiansen & Chater's model of language creation.
//!
//! ## Architectural Role
//!
//! In the triangular model, Processing operates at the fastest timescale (microseconds)
//! and serves as the real-time query engine. It constrains both Acquisition (what
//! patterns are worth learning) and Evolution (what optimizations matter).
//!
//! ```text
//!                    ┌──────────────┐
//!                    │  Acquisition │
//!                    │   (LSH)      │
//!                    └──────┬───────┘
//!                           │
//!         "fits what is     │      "processing constrains
//!          learned to       │       what can be acquired"
//!          processing"      │
//!                           ▼
//!     ┌──────────┐    ╔══════════════╗    
//!     │Evolution │◄───║  PROCESSING  ║◄── YOU ARE HERE
//!     │  (LSH)   │    ║   (HNSW)     ║
//!     └──────────┘    ╚══════════════╝
//!           │               │
//!           └───────────────┘
//!        "processing constrains
//!         what can evolve"
//! ```
//!
//! ## Key Constraints Enforced
//!
//! 1. **Latency Budget**: <1μs per query (enforced via benchmarks)
//! 2. **Zero Allocations**: No heap allocations in query hot path
//! 3. **Lock-Free Reads**: Concurrent queries never block each other
//! 4. **Hyperbolic Geometry**: Native support for Poincaré ball model
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_hnsw::{HotIndex, HyperbolicMetric, IndexConfig};
//!
//! // Create index with hyperbolic distance metric
//! let config = IndexConfig::default()
//!     .with_dimensions(128)
//!     .with_metric(HyperbolicMetric::poincare(curvature: -1.0));
//!
//! let index = HotIndex::new(config)?;
//!
//! // Insert vectors (from Acquisition layer)
//! index.insert(id, &embedding)?;
//!
//! // Query in sub-microsecond time
//! let neighbors = index.search(&query, k: 5)?;
//! ```
//!
//! ## Timescale Interactions
//!
//! | From Layer | To Processing | Mechanism |
//! |------------|---------------|-----------|
//! | Acquisition | Pattern promotion | Batch insert from streaming LSH |
//! | Evolution | Parameter tuning | Thermodynamic optimization of M, ef |
//! | Processing | Self | Query result caching |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Module Declarations
// ============================================================================

pub mod metric;
pub mod index;
pub mod config;
pub mod error;

#[cfg(feature = "gpu")]
pub mod gpu;

// ============================================================================
// Public Re-exports
// ============================================================================

pub use metric::{DistanceMetric, HyperbolicMetric, EuclideanMetric, CosineMetric, LorentzMetric, InnerProductMetric};
pub use index::{HotIndex, SearchResult, InsertResult};
pub use config::IndexConfig;
pub use error::{HnswError, Result};

// ============================================================================
// Crate-Level Constants
// ============================================================================

/// Maximum latency budget for a single query (nanoseconds).
/// Queries exceeding this trigger a warning in debug builds.
pub const QUERY_LATENCY_BUDGET_NS: u64 = 1_000; // 1 microsecond

/// Default number of connections per node (M parameter).
/// Higher values improve recall but increase memory and latency.
pub const DEFAULT_M: usize = 16;

/// Default number of connections for layer 0 (M0 = 2*M typically).
pub const DEFAULT_M0: usize = 32;

/// Default expansion factor during construction.
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Default expansion factor during search.
pub const DEFAULT_EF_SEARCH: usize = 64;

// ============================================================================
// Constraint Verification (Compile-Time Where Possible)
// ============================================================================

/// Marker trait for types that satisfy the Processing layer constraints.
/// 
/// Any index implementation must satisfy:
/// 1. Query latency < `QUERY_LATENCY_BUDGET_NS`
/// 2. Zero heap allocations in query path
/// 3. Thread-safe concurrent reads
pub trait ProcessingConstraint: Send + Sync {
    /// Verify this implementation satisfies Processing constraints.
    /// Called during index construction in debug builds.
    #[cfg(debug_assertions)]
    fn verify_constraints(&self) -> Result<()>;
}

// ============================================================================
// Timescale Bridge Traits
// ============================================================================

/// Trait for receiving patterns from the Acquisition layer.
/// 
/// The Acquisition layer (LSH Streaming) operates at a slower timescale,
/// accumulating patterns that are periodically promoted to Processing.
pub trait AcquisitionReceiver {
    /// Batch insert patterns received from Acquisition layer.
    /// 
    /// This operation may be slower than single inserts as it involves
    /// graph rebalancing. Called asynchronously, never on hot path.
    fn receive_from_acquisition(&mut self, patterns: &[(u64, Vec<f32>)]) -> Result<usize>;
}

/// Trait for receiving optimization parameters from the Evolution layer.
/// 
/// The Evolution layer (LSH pBit/Hardware) operates at the slowest timescale,
/// discovering optimal parameters through thermodynamic exploration.
pub trait EvolutionReceiver {
    /// Apply optimized parameters from Evolution layer.
    /// 
    /// Parameters include M, ef_search, and potentially learned
    /// distance metric adjustments.
    fn receive_from_evolution(&mut self, params: EvolutionParams) -> Result<()>;
}

/// Parameters discovered by the Evolution layer through thermodynamic optimization.
#[derive(Debug, Clone)]
pub struct EvolutionParams {
    /// Optimized number of connections per node.
    pub m: Option<usize>,
    
    /// Optimized expansion factor for search.
    pub ef_search: Option<usize>,
    
    /// Learned curvature adjustment for hyperbolic metric.
    pub curvature_adjustment: Option<f32>,
    
    /// Temperature parameter from pBit annealing.
    pub temperature: f32,
}

impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            m: None,
            ef_search: None,
            curvature_adjustment: None,
            temperature: 1.0,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_are_reasonable() {
        // Verify our latency budget is sub-microsecond
        assert!(QUERY_LATENCY_BUDGET_NS <= 1_000);
        
        // Verify M parameters follow the 2x rule
        assert_eq!(DEFAULT_M0, DEFAULT_M * 2);
        
        // Verify ef_construction > ef_search (standard practice)
        assert!(DEFAULT_EF_CONSTRUCTION > DEFAULT_EF_SEARCH);
    }
}
