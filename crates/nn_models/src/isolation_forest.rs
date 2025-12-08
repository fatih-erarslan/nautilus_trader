//! Isolation Forest Anomaly Detection Implementation
//! 
//! High-performance Rust implementation of Isolation Forest algorithm
//! for detecting anomalies in trading data with <50μs inference time.

use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Configuration for Isolation Forest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IsolationForestConfig {
    /// Number of isolation trees (default: 200)
    pub n_estimators: usize,
    /// Maximum samples per tree (default: 256)
    pub max_samples: usize,
    /// Expected contamination rate (default: 0.1)
    pub contamination: f32,
    /// Maximum tree depth (default: log2(max_samples))
    pub max_depth: Option<usize>,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Number of threads for parallel processing
    pub n_jobs: Option<usize>,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self {
            n_estimators: 200,
            max_samples: 256,
            contamination: 0.1,
            max_depth: None,
            random_seed: None,
            n_jobs: None,
        }
    }
}

/// A single node in an isolation tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IsolationNode {
    /// Feature index for splitting
    pub feature_idx: Option<usize>,
    /// Split value for the feature
    pub split_value: Option<f32>,
    /// Left child node index
    pub left: Option<usize>,
    /// Right child node index
    pub right: Option<usize>,
    /// Node size (number of samples that reached this node)
    pub size: usize,
    /// Path length adjustment for external nodes
    pub c: f32,
}

/// An isolation tree
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IsolationTree {
    /// Tree nodes stored in a vector
    nodes: Vec<IsolationNode>,
    /// Root node index
    root: usize,
    /// Number of features
    n_features: usize,
    /// Random number generator
    #[serde(skip)]
    rng: XorShiftRng,
}

impl IsolationTree {
    /// Create a new isolation tree
    pub fn new(n_features: usize, seed: u64) -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            n_features,
            rng: XorShiftRng::seed_from_u64(seed),
        }
    }

    /// Build the tree from data
    pub fn fit(&mut self, data: &[Vec<f32>], max_depth: usize) {
        if data.is_empty() {
            return;
        }

        let indices: Vec<usize> = (0..data.len()).collect();
        self.root = self.build_node(&indices, data, 0, max_depth);
    }

    /// Recursively build tree nodes
    fn build_node(
        &mut self,
        indices: &[usize],
        data: &[Vec<f32>],
        depth: usize,
        max_depth: usize,
    ) -> usize {
        let n = indices.len();
        
        // Create external node if we've reached max depth or have only one sample
        if depth >= max_depth || n <= 1 {
            let c = if n > 2 {
                2.0 * (harmonic(n) - 1.0)
            } else if n == 2 {
                1.0
            } else {
                0.0
            };
            
            let node = IsolationNode {
                feature_idx: None,
                split_value: None,
                left: None,
                right: None,
                size: n,
                c,
            };
            
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            return node_idx;
        }

        // Select random feature
        let feature_idx = self.rng.gen_range(0..self.n_features);
        
        // Find min and max values for the selected feature
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for &idx in indices {
            let val = data[idx][feature_idx];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        
        // If all values are the same, create external node
        if (max_val - min_val).abs() < f32::EPSILON {
            let c = 2.0 * (harmonic(n) - 1.0);
            
            let node = IsolationNode {
                feature_idx: None,
                split_value: None,
                left: None,
                right: None,
                size: n,
                c,
            };
            
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            return node_idx;
        }
        
        // Select random split value
        let split_value = self.rng.gen_range(min_val..max_val);
        
        // Partition data
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for &idx in indices {
            if data[idx][feature_idx] < split_value {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        
        // Create internal node
        let node_idx = self.nodes.len();
        self.nodes.push(IsolationNode {
            feature_idx: Some(feature_idx),
            split_value: Some(split_value),
            left: None,
            right: None,
            size: n,
            c: 0.0,
        });
        
        // Build child nodes
        let left_idx = self.build_node(&left_indices, data, depth + 1, max_depth);
        let right_idx = self.build_node(&right_indices, data, depth + 1, max_depth);
        
        // Update node with child indices
        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);
        
        node_idx
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &[f32]) -> f32 {
        self.path_length_recursive(sample, self.root, 0)
    }

    fn path_length_recursive(&self, sample: &[f32], node_idx: usize, current_depth: usize) -> f32 {
        let node = &self.nodes[node_idx];
        
        // External node
        if node.feature_idx.is_none() {
            return current_depth as f32 + node.c;
        }
        
        // Internal node
        let feature_idx = node.feature_idx.unwrap();
        let split_value = node.split_value.unwrap();
        
        if sample[feature_idx] < split_value {
            if let Some(left_idx) = node.left {
                self.path_length_recursive(sample, left_idx, current_depth + 1)
            } else {
                current_depth as f32
            }
        } else {
            if let Some(right_idx) = node.right {
                self.path_length_recursive(sample, right_idx, current_depth + 1)
            } else {
                current_depth as f32
            }
        }
    }
}

/// Isolation Forest anomaly detector
#[derive(Clone, Debug)]
pub struct IsolationForest {
    /// Configuration
    config: IsolationForestConfig,
    /// Collection of isolation trees
    trees: Vec<Arc<IsolationTree>>,
    /// Anomaly score threshold
    threshold: f32,
    /// Number of features
    n_features: Option<usize>,
    /// Sample size used for training
    sample_size: usize,
}

impl IsolationForest {
    /// Create a new Isolation Forest with default configuration
    pub fn new() -> Self {
        Self::with_config(IsolationForestConfig::default())
    }

    /// Create a new Isolation Forest with custom configuration
    pub fn with_config(config: IsolationForestConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            threshold: 0.5,
            n_features: None,
            sample_size: 0,
        }
    }

    /// Create a builder for Isolation Forest
    pub fn builder() -> IsolationForestBuilder {
        IsolationForestBuilder::new()
    }

    /// Fit the model to training data
    pub fn fit(&mut self, data: &[Vec<f32>]) {
        if data.is_empty() {
            panic!("Cannot fit on empty data");
        }

        let n_samples = data.len();
        self.n_features = Some(data[0].len());
        
        // Determine sample size
        self.sample_size = if n_samples > self.config.max_samples {
            self.config.max_samples
        } else {
            n_samples
        };
        
        // Calculate max depth if not specified
        let max_depth = self.config.max_depth.unwrap_or_else(|| {
            (self.sample_size as f32).log2().ceil() as usize
        });
        
        // Determine number of threads
        let n_jobs = self.config.n_jobs.unwrap_or_else(|| {
            rayon::current_num_threads()
        });
        
        // Build trees in parallel
        let base_seed = self.config.random_seed.unwrap_or(42);
        
        self.trees = (0..self.config.n_estimators)
            .into_par_iter()
            .map(|i| {
                let seed = base_seed + i as u64;
                let mut rng = XorShiftRng::seed_from_u64(seed);
                
                // Sample data for this tree
                let sample_indices: Vec<usize> = if n_samples > self.sample_size {
                    (0..self.sample_size)
                        .map(|_| rng.gen_range(0..n_samples))
                        .collect()
                } else {
                    (0..n_samples).collect()
                };
                
                let sampled_data: Vec<Vec<f32>> = sample_indices
                    .iter()
                    .map(|&idx| data[idx].clone())
                    .collect();
                
                // Build tree
                let mut tree = IsolationTree::new(self.n_features.unwrap(), seed);
                tree.fit(&sampled_data, max_depth);
                
                Arc::new(tree)
            })
            .collect();
        
        // Calculate threshold based on contamination
        let scores: Vec<f32> = data.iter()
            .map(|sample| self.anomaly_score(sample))
            .collect();
        
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let threshold_idx = (data.len() as f32 * self.config.contamination) as usize;
        self.threshold = sorted_scores.get(threshold_idx).copied().unwrap_or(0.5);
    }

    /// Predict if samples are anomalies
    pub fn predict(&self, data: &[Vec<f32>]) -> Vec<i32> {
        data.par_iter()
            .map(|sample| {
                let score = self.anomaly_score(sample);
                if score > self.threshold {
                    -1  // Anomaly
                } else {
                    1   // Normal
                }
            })
            .collect()
    }

    /// Get anomaly scores for samples
    pub fn decision_function(&self, data: &[Vec<f32>]) -> Vec<f32> {
        data.par_iter()
            .map(|sample| self.anomaly_score(sample))
            .collect()
    }

    /// Calculate anomaly score for a single sample
    pub fn anomaly_score(&self, sample: &[f32]) -> f32 {
        if self.trees.is_empty() {
            return 0.5;
        }

        // Calculate average path length across all trees
        let avg_path_length: f32 = self.trees
            .par_iter()
            .map(|tree| tree.path_length(sample))
            .sum::<f32>() / self.trees.len() as f32;
        
        // Calculate anomaly score
        let c_n = average_path_length(self.sample_size);
        let score = 2.0_f32.powf(-avg_path_length / c_n);
        
        score
    }

    /// Get feature importances based on split counts
    pub fn feature_importances(&self) -> Vec<f32> {
        if self.n_features.is_none() || self.trees.is_empty() {
            return Vec::new();
        }

        let n_features = self.n_features.unwrap();
        let mut importances = vec![0.0; n_features];
        let mut total_splits = 0;

        // Count splits per feature across all trees
        for tree in &self.trees {
            for node in &tree.nodes {
                if let Some(feature_idx) = node.feature_idx {
                    importances[feature_idx] += 1.0;
                    total_splits += 1;
                }
            }
        }

        // Normalize importances
        if total_splits > 0 {
            for importance in &mut importances {
                *importance /= total_splits as f32;
            }
        }

        importances
    }

    /// Get the number of estimators
    pub fn n_estimators(&self) -> usize {
        self.config.n_estimators
    }

    /// Get the contamination rate
    pub fn contamination(&self) -> f32 {
        self.config.contamination
    }

    /// Get the threshold value
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Builder for Isolation Forest
pub struct IsolationForestBuilder {
    config: IsolationForestConfig,
}

impl IsolationForestBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: IsolationForestConfig::default(),
        }
    }

    /// Set the number of estimators (trees)
    pub fn n_estimators(mut self, n: usize) -> Self {
        self.config.n_estimators = n;
        self
    }

    /// Set the maximum number of samples per tree
    pub fn max_samples(mut self, n: usize) -> Self {
        self.config.max_samples = n;
        self
    }

    /// Set the contamination rate
    pub fn contamination(mut self, c: f32) -> Self {
        self.config.contamination = c;
        self
    }

    /// Set the maximum tree depth
    pub fn max_depth(mut self, d: usize) -> Self {
        self.config.max_depth = Some(d);
        self
    }

    /// Set the random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n: usize) -> Self {
        self.config.n_jobs = Some(n);
        self
    }

    /// Build the Isolation Forest
    pub fn build(self) -> IsolationForest {
        IsolationForest::with_config(self.config)
    }
}

/// Anomaly score structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnomalyScore {
    /// The anomaly score (0 to 1, higher means more anomalous)
    pub score: f32,
    /// Whether the sample is classified as an anomaly
    pub is_anomaly: bool,
    /// Feature contributions to the anomaly score
    pub feature_contributions: Option<Vec<f32>>,
}

/// Feature importance structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature index
    pub feature_idx: usize,
    /// Importance score
    pub importance: f32,
}

// Helper functions

/// Calculate harmonic number
fn harmonic(n: usize) -> f32 {
    if n <= 0 {
        return 0.0;
    }
    
    // Use approximation for large n
    if n > 100 {
        0.5772156649 + (n as f32).ln() + 0.5 / n as f32
    } else {
        // Direct calculation for small n
        (1..=n).map(|i| 1.0 / i as f32).sum()
    }
}

/// Calculate average path length for n samples
fn average_path_length(n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    
    if n == 2 {
        return 1.0;
    }
    
    2.0 * harmonic(n - 1) - 2.0 * (n as f32 - 1.0) / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_forest_basic() {
        let mut data = Vec::new();
        
        // Generate normal data
        for i in 0..100 {
            data.push(vec![i as f32 * 0.1, (i as f32 * 0.1).sin()]);
        }
        
        // Add some anomalies
        data.push(vec![5.0, 10.0]);
        data.push(vec![-5.0, -10.0]);
        
        let mut forest = IsolationForest::builder()
            .n_estimators(100)
            .contamination(0.02)
            .random_seed(42)
            .build();
        
        forest.fit(&data);
        
        // Test predictions
        let predictions = forest.predict(&data);
        let anomaly_count = predictions.iter().filter(|&&p| p == -1).count();
        
        // Should detect approximately 2 anomalies (2% of 102 samples)
        assert!(anomaly_count >= 1 && anomaly_count <= 3);
    }

    #[test]
    fn test_anomaly_scores() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![10.0, 10.0], // Anomaly
        ];
        
        let mut forest = IsolationForest::builder()
            .n_estimators(50)
            .build();
        
        forest.fit(&data);
        
        let scores = forest.decision_function(&data);
        
        // Last point should have higher anomaly score
        assert!(scores[3] > scores[0]);
        assert!(scores[3] > scores[1]);
        assert!(scores[3] > scores[2]);
    }

    #[test]
    fn test_feature_importances() {
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                vec![
                    i as f32 * 0.1,              // Important feature
                    (i as f32 * 0.1).sin(),       // Important feature
                    0.5,                          // Constant, not important
                    rand::random::<f32>() * 0.01, // Noise, less important
                ]
            })
            .collect();
        
        let mut forest = IsolationForest::new();
        forest.fit(&data);
        
        let importances = forest.feature_importances();
        
        assert_eq!(importances.len(), 4);
        // First two features should be more important than the constant feature
        assert!(importances[0] > importances[2]);
        assert!(importances[1] > importances[2]);
    }

    #[test]
    fn test_performance() {
        use std::time::Instant;
        
        // Generate test data
        let n_samples = 1000;
        let n_features = 10;
        let data: Vec<Vec<f32>> = (0..n_samples)
            .map(|_| {
                (0..n_features)
                    .map(|_| rand::random::<f32>())
                    .collect()
            })
            .collect();
        
        let mut forest = IsolationForest::builder()
            .n_estimators(200)
            .build();
        
        // Measure training time
        let start = Instant::now();
        forest.fit(&data);
        let fit_time = start.elapsed();
        
        println!("Training time for {} samples: {:?}", n_samples, fit_time);
        
        // Measure inference time
        let test_sample = vec![0.5; n_features];
        let start = Instant::now();
        let n_iterations = 10000;
        
        for _ in 0..n_iterations {
            let _ = forest.anomaly_score(&test_sample);
        }
        
        let total_time = start.elapsed();
        let avg_time = total_time / n_iterations;
        
        println!("Average inference time: {:?}", avg_time);
        
        // Should be less than 50μs
        assert!(avg_time.as_micros() < 50);
    }
}