//! # Unified Search Configuration
//!
//! Configuration for the hybrid HNSW + LSH system.

use serde::{Deserialize, Serialize};

use hyperphysics_hnsw::IndexConfig as HnswConfig;
use hyperphysics_lsh::LshConfig;

/// Unified configuration for the hybrid search system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// HNSW (Processing layer) configuration.
    pub hnsw: HnswConfig,
    
    /// LSH (Acquisition layer) configuration.
    pub lsh: LshConfig,
    
    /// Router configuration.
    pub router: RouterConfig,
    
    /// Evolution configuration.
    pub evolution: EvolutionConfig,
}

/// Router configuration for directing queries to appropriate index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Enable automatic routing based on query characteristics.
    pub auto_route: bool,
    
    /// Threshold for routing to LSH (below this, use HNSW).
    /// Based on expected selectivity.
    pub lsh_threshold: f32,
    
    /// Maximum candidates to retrieve from LSH before HNSW refinement.
    pub lsh_candidate_limit: usize,
    
    /// Enable parallel query execution.
    pub parallel_query: bool,
    
    /// Timeout for queries in microseconds.
    pub query_timeout_us: u64,
}

/// Evolution layer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Enable automatic parameter tuning.
    pub auto_tune: bool,
    
    /// Feedback interval in milliseconds.
    pub feedback_interval_ms: u64,
    
    /// Target latency for optimization (nanoseconds).
    pub target_latency_ns: u64,
    
    /// Target recall for optimization.
    pub target_recall: f32,
    
    /// Learning rate for parameter updates.
    pub learning_rate: f32,
    
    /// Temperature for Boltzmann sampling (pBit integration).
    pub temperature: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            hnsw: HnswConfig::default(),
            lsh: LshConfig::default(),
            router: RouterConfig::default(),
            evolution: EvolutionConfig::default(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            auto_route: true,
            lsh_threshold: 0.3,
            lsh_candidate_limit: 1000,
            parallel_query: false,
            query_timeout_us: 10_000, // 10ms
        }
    }
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            auto_tune: true,
            feedback_interval_ms: 1000,
            target_latency_ns: crate::HOT_PATH_LATENCY_NS,
            target_recall: 0.95,
            learning_rate: 0.01,
            temperature: 1.0,
        }
    }
}

impl SearchConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Configure for low-latency trading.
    pub fn trading() -> Self {
        Self {
            hnsw: HnswConfig::low_latency(),
            lsh: LshConfig::simhash(128, 64).with_tables(8),
            router: RouterConfig {
                auto_route: true,
                lsh_threshold: 0.2,
                lsh_candidate_limit: 500,
                parallel_query: false,
                query_timeout_us: 5_000, // 5ms hard limit
            },
            evolution: EvolutionConfig {
                target_latency_ns: 500, // 500ns target
                target_recall: 0.9,
                ..Default::default()
            },
        }
    }
    
    /// Configure for high-recall research.
    pub fn research() -> Self {
        Self {
            hnsw: HnswConfig::high_recall(),
            lsh: LshConfig::simhash(256, 128).with_tables(16),
            router: RouterConfig {
                auto_route: true,
                lsh_threshold: 0.5,
                lsh_candidate_limit: 5000,
                parallel_query: true,
                query_timeout_us: 100_000, // 100ms acceptable
            },
            evolution: EvolutionConfig {
                target_latency_ns: 10_000, // 10μs acceptable
                target_recall: 0.99,
                ..Default::default()
            },
        }
    }
    
    /// Configure for whale detection.
    pub fn whale_detection() -> Self {
        Self {
            hnsw: HnswConfig::default(),
            lsh: LshConfig::minhash(256).with_tables(16),
            router: RouterConfig {
                auto_route: false, // Always use LSH for whale detection
                lsh_threshold: 0.0,
                lsh_candidate_limit: 10_000,
                parallel_query: true,
                query_timeout_us: 50_000,
            },
            evolution: EvolutionConfig::default(),
        }
    }
    
    /// Set HNSW configuration.
    pub fn with_hnsw(mut self, hnsw: HnswConfig) -> Self {
        self.hnsw = hnsw;
        self
    }
    
    /// Set LSH configuration.
    pub fn with_lsh(mut self, lsh: LshConfig) -> Self {
        self.lsh = lsh;
        self
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.hnsw.validate().map_err(|e| ConfigError::HnswConfig(e.to_string()))?;
        self.lsh.validate().map_err(|e| ConfigError::LshConfig(e.to_string()))?;
        
        if self.router.query_timeout_us == 0 {
            return Err(ConfigError::InvalidParameter(
                "query_timeout_us must be > 0".into()
            ));
        }
        
        if self.evolution.target_recall <= 0.0 || self.evolution.target_recall > 1.0 {
            return Err(ConfigError::InvalidParameter(
                "target_recall must be in (0, 1]".into()
            ));
        }
        
        Ok(())
    }
}

/// Configuration validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    /// HNSW configuration error.
    #[error("HNSW config error: {0}")]
    HnswConfig(String),
    
    /// LSH configuration error.
    #[error("LSH config error: {0}")]
    LshConfig(String),
    
    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = SearchConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_trading_config() {
        let config = SearchConfig::trading();
        assert!(config.validate().is_ok());
        assert!(config.evolution.target_latency_ns < 1000);
    }
    
    #[test]
    fn test_research_config() {
        let config = SearchConfig::research();
        assert!(config.validate().is_ok());
        assert!(config.evolution.target_recall > 0.95);
    }
    
    #[test]
    fn test_whale_detection_config() {
        let config = SearchConfig::whale_detection();
        assert!(config.validate().is_ok());
        assert!(!config.router.auto_route);
    }
}
