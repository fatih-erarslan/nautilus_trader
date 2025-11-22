//! Locality-Sensitive Hashing for problem similarity routing.
//!
//! Uses random hyperplane LSH for cosine similarity.

use crate::problem::ProblemSignature;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LSH configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSHConfig {
    /// Number of hash functions per band
    pub num_hash_functions: usize,
    /// Number of bands (for AND-OR amplification)
    pub num_bands: usize,
    /// Feature dimensionality
    pub feature_dim: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            num_hash_functions: 4,  // Hash functions per band
            num_bands: 8,           // Total bands
            feature_dim: 16,        // Matches ProblemSignature feature vector
            seed: 42,
        }
    }
}

impl LSHConfig {
    /// Total number of hash bits
    pub fn total_hash_bits(&self) -> usize {
        self.num_hash_functions * self.num_bands
    }
}

/// LSH hash value
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LSHHash {
    /// Hash bits per band
    pub bands: Vec<u64>,
}

impl LSHHash {
    /// Check if any band matches (OR condition across bands)
    pub fn any_band_matches(&self, other: &LSHHash) -> bool {
        self.bands.iter().zip(other.bands.iter()).any(|(a, b)| a == b)
    }

    /// Count matching bands
    pub fn matching_bands(&self, other: &LSHHash) -> usize {
        self.bands.iter().zip(other.bands.iter()).filter(|(a, b)| a == b).count()
    }
}

/// Entry in the LSH index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSHEntry<T: Clone> {
    /// Original feature vector
    pub features: Vec<f32>,
    /// Associated data
    pub data: T,
    /// Precomputed hash
    pub hash: LSHHash,
}

/// LSH Index for fast approximate nearest neighbor search
pub struct LSHIndex<T: Clone + Send + Sync> {
    config: LSHConfig,
    /// Random hyperplanes for hashing
    hyperplanes: Vec<Vec<f32>>,
    /// Buckets: band_index -> hash_value -> entries
    buckets: Vec<HashMap<u64, Vec<LSHEntry<T>>>>,
    /// All entries for linear scan fallback
    entries: Vec<LSHEntry<T>>,
}

impl<T: Clone + Send + Sync> LSHIndex<T> {
    /// Create a new LSH index
    pub fn new(config: LSHConfig) -> Self {
        let mut rng = rand::thread_rng();
        let total_hyperplanes = config.num_hash_functions * config.num_bands;

        // Generate random hyperplanes
        let hyperplanes: Vec<Vec<f32>> = (0..total_hyperplanes)
            .map(|_| {
                (0..config.feature_dim)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        // Initialize buckets for each band
        let buckets = (0..config.num_bands).map(|_| HashMap::new()).collect();

        Self {
            config,
            hyperplanes,
            buckets,
            entries: Vec::new(),
        }
    }

    /// Compute LSH hash for a feature vector
    pub fn compute_hash(&self, features: &[f32]) -> LSHHash {
        let mut bands = Vec::with_capacity(self.config.num_bands);

        for band_idx in 0..self.config.num_bands {
            let mut band_hash: u64 = 0;

            for h_idx in 0..self.config.num_hash_functions {
                let hyperplane_idx = band_idx * self.config.num_hash_functions + h_idx;
                let hyperplane = &self.hyperplanes[hyperplane_idx];

                // Compute dot product
                let dot: f32 = features
                    .iter()
                    .zip(hyperplane.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                // Hash bit is 1 if dot product >= 0
                if dot >= 0.0 {
                    band_hash |= 1 << h_idx;
                }
            }

            bands.push(band_hash);
        }

        LSHHash { bands }
    }

    /// Insert an entry into the index
    pub fn insert(&mut self, features: Vec<f32>, data: T) {
        let hash = self.compute_hash(&features);

        // Insert into buckets
        for (band_idx, &band_hash) in hash.bands.iter().enumerate() {
            self.buckets[band_idx]
                .entry(band_hash)
                .or_insert_with(Vec::new)
                .push(LSHEntry {
                    features: features.clone(),
                    data: data.clone(),
                    hash: hash.clone(),
                });
        }

        // Store in entries list
        self.entries.push(LSHEntry {
            features,
            data,
            hash,
        });
    }

    /// Insert a problem signature
    pub fn insert_signature(&mut self, signature: &ProblemSignature, data: T) {
        let features = signature.to_feature_vector().to_vec();
        self.insert(features, data);
    }

    /// Query for similar entries
    ///
    /// Returns entries that hash to the same bucket in at least one band.
    pub fn query(&self, features: &[f32], max_results: usize) -> Vec<(f32, &T)> {
        let query_hash = self.compute_hash(features);
        let mut candidates: HashMap<usize, &LSHEntry<T>> = HashMap::new();

        // Collect candidates from all matching buckets
        for (band_idx, &band_hash) in query_hash.bands.iter().enumerate() {
            if let Some(bucket) = self.buckets[band_idx].get(&band_hash) {
                for entry in bucket {
                    // Use pointer as unique identifier
                    let ptr = entry as *const _ as usize;
                    candidates.insert(ptr, entry);
                }
            }
        }

        // If no candidates, fall back to linear scan
        if candidates.is_empty() && !self.entries.is_empty() {
            for entry in &self.entries {
                let ptr = entry as *const _ as usize;
                candidates.insert(ptr, entry);
            }
        }

        // Compute exact cosine similarity and sort
        let mut results: Vec<(f32, &T)> = candidates
            .values()
            .map(|entry| {
                let similarity = cosine_similarity(features, &entry.features);
                (similarity, &entry.data)
            })
            .collect();

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(max_results);

        results
    }

    /// Query using a problem signature
    pub fn query_signature(
        &self,
        signature: &ProblemSignature,
        max_results: usize,
    ) -> Vec<(f32, &T)> {
        let features = signature.to_feature_vector().to_vec();
        self.query(&features, max_results)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.entries.clear();
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

/// Record of a problem-solution pair for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSolutionRecord {
    /// Problem signature features
    pub features: Vec<f32>,
    /// Backend that solved it successfully
    pub backend_id: String,
    /// Solution quality achieved
    pub quality: f64,
    /// Latency achieved
    pub latency_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ProblemType, ProblemSignature};
    use crate::ProblemDomain;

    #[test]
    fn test_lsh_config_default() {
        let config = LSHConfig::default();
        assert_eq!(config.feature_dim, 16);
        assert_eq!(config.total_hash_bits(), 32);
    }

    #[test]
    fn test_lsh_hash_consistency() {
        let config = LSHConfig::default();
        let index: LSHIndex<String> = LSHIndex::new(config);

        let features = vec![1.0; 16];
        let hash1 = index.compute_hash(&features);
        let hash2 = index.compute_hash(&features);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_lsh_insert_and_query() {
        let config = LSHConfig::default();
        let mut index: LSHIndex<String> = LSHIndex::new(config);

        // Insert some entries
        index.insert(vec![1.0; 16], "entry1".to_string());
        index.insert(vec![0.9; 16], "entry2".to_string());
        index.insert(vec![-1.0; 16], "entry3".to_string());

        // Query similar to entry1
        let results = index.query(&vec![1.0; 16], 10);

        assert!(!results.is_empty());
        assert_eq!(results[0].1, "entry1");
        assert!(results[0].0 > 0.99); // Very high similarity
    }

    #[test]
    fn test_lsh_with_signatures() {
        let config = LSHConfig::default();
        let mut index: LSHIndex<String> = LSHIndex::new(config);

        let sig1 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(100);
        let sig2 = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(150);
        let sig3 = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics);

        index.insert_signature(&sig1, "backend-pso".to_string());
        index.insert_signature(&sig2, "backend-ga".to_string());
        index.insert_signature(&sig3, "backend-rapier".to_string());

        // Query with a similar financial optimization problem
        let query = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial)
            .with_dimensionality(120);

        let results = index.query_signature(&query, 3);

        // Should find the financial optimization backends first
        assert!(!results.is_empty());
        assert!(results[0].1 == "backend-pso" || results[0].1 == "backend-ga");
    }

    #[test]
    fn test_band_matching() {
        let hash1 = LSHHash {
            bands: vec![1, 2, 3, 4],
        };
        let hash2 = LSHHash {
            bands: vec![1, 5, 6, 7],
        };
        let hash3 = LSHHash {
            bands: vec![10, 11, 12, 13],
        };

        assert!(hash1.any_band_matches(&hash2)); // First band matches
        assert!(!hash1.any_band_matches(&hash3)); // No bands match
        assert_eq!(hash1.matching_bands(&hash2), 1);
        assert_eq!(hash1.matching_bands(&hash1), 4); // Self match
    }
}
