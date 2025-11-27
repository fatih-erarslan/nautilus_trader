//! # LSH Configuration
//!
//! Configuration for the StreamingLshIndex and hash families.

use serde::{Deserialize, Serialize};

/// Configuration for the LSH index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshConfig {
    /// Hash family to use.
    pub hash_family: HashFamilyConfig,
    
    /// Number of hash tables (L parameter).
    pub num_tables: usize,
    
    /// Hash functions per table (k parameter).
    pub hashes_per_table: usize,
    
    /// Maximum items per bucket before splitting.
    pub bucket_capacity: usize,
    
    /// Initial number of buckets per table.
    pub initial_buckets: usize,
    
    /// Random seed for reproducibility.
    pub seed: u64,
    
    /// Enable zero-allocation mode (pre-allocate all buffers).
    pub zero_alloc: bool,
    
    /// Maximum number of items in the index.
    pub max_capacity: Option<usize>,
    
    /// Enable statistics collection.
    pub collect_stats: bool,
}

/// Hash family configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashFamilyConfig {
    /// SimHash for cosine similarity.
    SimHash {
        /// Vector dimensionality.
        dimensions: usize,
        /// Signature bits.
        num_bits: usize,
    },
    
    /// MinHash for Jaccard similarity.
    MinHash {
        /// Number of hash functions.
        num_hashes: usize,
    },
    
    /// Signed Random Projections.
    Srp {
        /// Vector dimensionality.
        dimensions: usize,
        /// Signature bits.
        num_bits: usize,
    },
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            hash_family: HashFamilyConfig::SimHash {
                dimensions: 128,
                num_bits: 64,
            },
            num_tables: crate::DEFAULT_NUM_TABLES,
            hashes_per_table: crate::DEFAULT_HASHES_PER_TABLE,
            bucket_capacity: crate::DEFAULT_BUCKET_CAPACITY,
            initial_buckets: 1024,
            seed: 42,
            zero_alloc: true,
            max_capacity: None,
            collect_stats: cfg!(debug_assertions),
        }
    }
}

impl LshConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Configure for SimHash (cosine similarity).
    pub fn simhash(dimensions: usize, num_bits: usize) -> Self {
        Self {
            hash_family: HashFamilyConfig::SimHash { dimensions, num_bits },
            ..Default::default()
        }
    }
    
    /// Configure for MinHash (Jaccard similarity).
    pub fn minhash(num_hashes: usize) -> Self {
        Self {
            hash_family: HashFamilyConfig::MinHash { num_hashes },
            ..Default::default()
        }
    }
    
    /// Configure for SRP (angular similarity).
    pub fn srp(dimensions: usize, num_bits: usize) -> Self {
        Self {
            hash_family: HashFamilyConfig::Srp { dimensions, num_bits },
            ..Default::default()
        }
    }
    
    /// Set the number of tables.
    pub fn with_tables(mut self, num_tables: usize) -> Self {
        self.num_tables = num_tables;
        self
    }
    
    /// Set hash functions per table.
    pub fn with_hashes_per_table(mut self, k: usize) -> Self {
        self.hashes_per_table = k;
        self
    }
    
    /// Set bucket capacity.
    pub fn with_bucket_capacity(mut self, capacity: usize) -> Self {
        self.bucket_capacity = capacity;
        self
    }
    
    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    
    /// Set maximum capacity.
    pub fn with_max_capacity(mut self, max: usize) -> Self {
        self.max_capacity = Some(max);
        self
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_tables == 0 {
            return Err(ConfigError::InvalidParameter(
                "num_tables must be > 0".into()
            ));
        }
        
        if self.hashes_per_table == 0 {
            return Err(ConfigError::InvalidParameter(
                "hashes_per_table must be > 0".into()
            ));
        }
        
        if self.bucket_capacity == 0 {
            return Err(ConfigError::InvalidParameter(
                "bucket_capacity must be > 0".into()
            ));
        }
        
        match &self.hash_family {
            HashFamilyConfig::SimHash { dimensions, num_bits } => {
                if *dimensions == 0 || *dimensions > crate::MAX_DIMENSIONS {
                    return Err(ConfigError::InvalidParameter(
                        format!("dimensions must be 1..{}", crate::MAX_DIMENSIONS)
                    ));
                }
                if *num_bits == 0 || *num_bits > crate::MAX_SIGNATURE_BITS {
                    return Err(ConfigError::InvalidParameter(
                        format!("num_bits must be 1..{}", crate::MAX_SIGNATURE_BITS)
                    ));
                }
            }
            HashFamilyConfig::MinHash { num_hashes } => {
                if *num_hashes == 0 || *num_hashes > 256 {
                    return Err(ConfigError::InvalidParameter(
                        "num_hashes must be 1..256".into()
                    ));
                }
            }
            HashFamilyConfig::Srp { dimensions, num_bits } => {
                if *dimensions == 0 || *dimensions > crate::MAX_DIMENSIONS {
                    return Err(ConfigError::InvalidParameter(
                        format!("dimensions must be 1..{}", crate::MAX_DIMENSIONS)
                    ));
                }
                if *num_bits == 0 || *num_bits > crate::MAX_SIGNATURE_BITS {
                    return Err(ConfigError::InvalidParameter(
                        format!("num_bits must be 1..{}", crate::MAX_SIGNATURE_BITS)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate expected collision probability for given similarity.
    pub fn expected_collision_prob(&self, similarity: f32) -> f32 {
        // p = probability of single hash collision
        let p = similarity; // Simplified; actual varies by hash family
        
        // With k hashes per table (AND), collision prob = p^k
        let p_table = p.powi(self.hashes_per_table as i32);
        
        // With L tables (OR), at least one collision prob = 1 - (1-p^k)^L
        1.0 - (1.0 - p_table).powi(self.num_tables as i32)
    }
}

/// Configuration validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    /// A parameter has an invalid value.
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
        let config = LshConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_simhash_config() {
        let config = LshConfig::simhash(128, 64)
            .with_tables(16)
            .with_hashes_per_table(4);
        
        assert!(config.validate().is_ok());
        assert_eq!(config.num_tables, 16);
    }
    
    #[test]
    fn test_minhash_config() {
        let config = LshConfig::minhash(128);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_invalid_dimensions() {
        let config = LshConfig::simhash(0, 64);
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_collision_probability() {
        let config = LshConfig::default()
            .with_tables(8)
            .with_hashes_per_table(4);
        
        // High similarity should have high collision prob
        let high_sim = config.expected_collision_prob(0.9);
        let low_sim = config.expected_collision_prob(0.1);
        
        assert!(high_sim > low_sim);
        assert!(high_sim > 0.5);
        assert!(low_sim < 0.01);
    }
}
