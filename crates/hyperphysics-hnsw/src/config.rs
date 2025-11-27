//! # Index Configuration
//!
//! Configuration for the HotIndex, allowing tuning of HNSW parameters
//! and integration with the triangular constraint model.
//!
//! ## Parameter Relationships
//!
//! The HNSW parameters interact with the triangular architecture:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    PARAMETER CONSTRAINTS                         │
//! │                                                                  │
//! │  ACQUISITION → PROCESSING:                                       │
//! │    • Pattern promotion rate constrained by insert latency        │
//! │    • Batch size limited by memory budget                         │
//! │                                                                  │
//! │  EVOLUTION → PROCESSING:                                         │
//! │    • ef_search tuned by thermodynamic optimization               │
//! │    • M changes require index rebuild (slow path)                 │
//! │                                                                  │
//! │  PROCESSING → ACQUISITION:                                       │
//! │    • Query latency budget constrains what patterns are useful    │
//! │    • Recall requirements affect pattern quality threshold        │
//! │                                                                  │
//! │  PROCESSING → EVOLUTION:                                         │
//! │    • Latency measurements guide parameter search                 │
//! │    • Recall feedback shapes optimization objective               │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for the HotIndex.
///
/// This structure defines all tunable parameters for the HNSW index,
/// with defaults optimized for HyperPhysics trading workloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Vector dimensionality (must match all inserted vectors).
    /// If None, will be inferred from first insertion.
    pub dimensions: Option<usize>,
    
    /// Maximum number of connections per node (M parameter).
    /// Higher values improve recall but increase memory and latency.
    /// Default: 16 (good balance for 100K-1M vectors).
    pub m: Option<usize>,
    
    /// Maximum connections for layer 0 (typically 2*M).
    /// Default: 32.
    pub m0: Option<usize>,
    
    /// Expansion factor during construction.
    /// Higher values build better graphs but take longer.
    /// Default: 200.
    pub ef_construction: Option<usize>,
    
    /// Expansion factor during search.
    /// Can be tuned at runtime by Evolution layer.
    /// Default: 64.
    pub ef_search: Option<usize>,
    
    /// Initial capacity (number of vectors to pre-allocate).
    /// Default: 10,000.
    pub initial_capacity: Option<usize>,
    
    /// Maximum capacity (hard limit on index size).
    /// Default: None (unlimited).
    pub max_capacity: Option<usize>,
    
    /// Enable memory-mapped storage for persistence.
    /// Default: false.
    pub mmap_enabled: bool,
    
    /// Path for memory-mapped file (required if mmap_enabled).
    pub mmap_path: Option<String>,
    
    /// Query latency budget in nanoseconds.
    /// Queries exceeding this trigger warnings.
    /// Default: 1,000 (1 microsecond).
    pub latency_budget_ns: Option<u64>,
    
    /// Enable collection of detailed statistics.
    /// Has minor performance overhead.
    /// Default: true in debug, false in release.
    pub collect_stats: bool,
    
    /// Seed for random number generation (for reproducibility).
    /// Default: None (use system entropy).
    pub seed: Option<u64>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimensions: None,
            m: Some(crate::DEFAULT_M),
            m0: Some(crate::DEFAULT_M0),
            ef_construction: Some(crate::DEFAULT_EF_CONSTRUCTION),
            ef_search: Some(crate::DEFAULT_EF_SEARCH),
            initial_capacity: Some(10_000),
            max_capacity: None,
            mmap_enabled: false,
            mmap_path: None,
            latency_budget_ns: Some(crate::QUERY_LATENCY_BUDGET_NS),
            collect_stats: cfg!(debug_assertions),
            seed: None,
        }
    }
}

impl IndexConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the vector dimensionality.
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
    
    /// Set the M parameter (connections per node).
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self.m0 = Some(m * 2); // Maintain 2x relationship
        self
    }
    
    /// Set the ef_construction parameter.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = Some(ef);
        self
    }
    
    /// Set the ef_search parameter.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }
    
    /// Set the initial capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = Some(capacity);
        self
    }
    
    /// Set the maximum capacity.
    pub fn with_max_capacity(mut self, max: usize) -> Self {
        self.max_capacity = Some(max);
        self
    }
    
    /// Enable memory-mapped storage.
    pub fn with_mmap<S: Into<String>>(mut self, path: S) -> Self {
        self.mmap_enabled = true;
        self.mmap_path = Some(path.into());
        self
    }
    
    /// Set the latency budget.
    pub fn with_latency_budget_ns(mut self, ns: u64) -> Self {
        self.latency_budget_ns = Some(ns);
        self
    }
    
    /// Enable or disable statistics collection.
    pub fn with_stats(mut self, enabled: bool) -> Self {
        self.collect_stats = enabled;
        self
    }
    
    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    /// Create a configuration optimized for low latency.
    /// 
    /// Uses smaller M and ef values for faster queries at the cost of recall.
    pub fn low_latency() -> Self {
        Self {
            m: Some(8),
            m0: Some(16),
            ef_construction: Some(100),
            ef_search: Some(32),
            latency_budget_ns: Some(500), // 500ns target
            ..Default::default()
        }
    }
    
    /// Create a configuration optimized for high recall.
    /// 
    /// Uses larger M and ef values for better accuracy at the cost of latency.
    pub fn high_recall() -> Self {
        Self {
            m: Some(32),
            m0: Some(64),
            ef_construction: Some(400),
            ef_search: Some(128),
            latency_budget_ns: Some(5_000), // 5μs acceptable
            ..Default::default()
        }
    }
    
    /// Create a configuration for billion-scale datasets.
    /// 
    /// Optimized for memory efficiency with mmap support.
    pub fn billion_scale<S: Into<String>>(mmap_path: S) -> Self {
        Self {
            m: Some(16),
            m0: Some(32),
            ef_construction: Some(200),
            ef_search: Some(64),
            initial_capacity: Some(1_000_000),
            mmap_enabled: true,
            mmap_path: Some(mmap_path.into()),
            ..Default::default()
        }
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // M must be positive
        if let Some(m) = self.m {
            if m == 0 {
                return Err(ConfigError::InvalidParameter("m must be > 0".into()));
            }
        }
        
        // ef_search should be >= M for good recall
        if let (Some(m), Some(ef)) = (self.m, self.ef_search) {
            if ef < m {
                return Err(ConfigError::InvalidParameter(
                    format!("ef_search ({}) should be >= M ({}) for good recall", ef, m)
                ));
            }
        }
        
        // mmap_path required if mmap_enabled
        if self.mmap_enabled && self.mmap_path.is_none() {
            return Err(ConfigError::MissingField("mmap_path required when mmap_enabled".into()));
        }
        
        Ok(())
    }
}

/// Configuration validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    /// A parameter has an invalid value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// A required field is missing.
    #[error("Missing required field: {0}")]
    MissingField(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = IndexConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.m, Some(16));
        assert_eq!(config.m0, Some(32));
    }
    
    #[test]
    fn test_builder_pattern() {
        let config = IndexConfig::new()
            .with_dimensions(128)
            .with_m(32)
            .with_ef_search(128)
            .with_capacity(100_000);
        
        assert!(config.validate().is_ok());
        assert_eq!(config.dimensions, Some(128));
        assert_eq!(config.m, Some(32));
        assert_eq!(config.m0, Some(64)); // Auto-set to 2*M
    }
    
    #[test]
    fn test_low_latency_preset() {
        let config = IndexConfig::low_latency();
        assert!(config.validate().is_ok());
        assert_eq!(config.m, Some(8));
        assert_eq!(config.latency_budget_ns, Some(500));
    }
    
    #[test]
    fn test_validation_ef_search() {
        let config = IndexConfig::new()
            .with_m(32)
            .with_ef_search(16); // Too small
        
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_validation_mmap() {
        let mut config = IndexConfig::default();
        config.mmap_enabled = true;
        config.mmap_path = None;
        
        assert!(config.validate().is_err());
    }
}
