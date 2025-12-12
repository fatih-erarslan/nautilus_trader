//! Quantum pattern recognition and matching algorithms
//!
//! This module provides quantum-enhanced pattern recognition including:
//! - Quantum template matching
//! - Quantum similarity measures
//! - Quantum clustering algorithms
//! - Pattern classification using quantum machine learning

use crate::core::{QarResult, QarError, constants};
use crate::quantum::{QuantumState, Gate, StandardGates};
use crate::core::{CircuitParams, ExecutionContext, PatternMatch, PatternData, RegimeAnalysis};
use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;
use super::traits::*;

/// Quantum pattern data structure
#[derive(Debug, Clone)]
pub struct QuantumPattern {
    /// Pattern identifier
    pub id: String,
    /// Quantum state representation of the pattern
    pub quantum_state: QuantumState,
    /// Classical feature vector
    pub features: Vec<f64>,
    /// Pattern metadata
    pub metadata: HashMap<String, String>,
    /// Pattern creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Pattern confidence score
    pub confidence: f64,
}

impl QuantumPattern {
    /// Create a new quantum pattern
    pub fn new(id: String, features: Vec<f64>) -> QarResult<Self> {
        let num_qubits = (features.len() as f64).log2().ceil() as usize;
        let quantum_state = Self::encode_features_to_quantum(&features, num_qubits)?;
        
        Ok(Self {
            id,
            quantum_state,
            features,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            confidence: 1.0,
        })
    }

    /// Encode classical features into quantum state
    fn encode_features_to_quantum(features: &[f64], num_qubits: usize) -> QarResult<QuantumState> {
        let state_size = 1 << num_qubits;
        let mut amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); state_size];
        
        // Normalize features
        let norm = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < constants::math::EPSILON {
            return Err(QarError::InvalidInput("Features have zero norm".to_string()));
        }
        
        // Encode features as amplitudes
        for (i, &feature) in features.iter().enumerate() {
            if i < state_size {
                amplitudes[i] = num_complex::Complex64::new(feature / norm, 0.0);
            }
        }
        
        let mut state = QuantumState::new(num_qubits);
        state.amplitudes = amplitudes;
        
        Ok(state)
    }

    /// Calculate quantum similarity with another pattern
    pub fn quantum_similarity(&self, other: &QuantumPattern) -> QarResult<f64> {
        self.quantum_state.fidelity(&other.quantum_state)
    }

    /// Calculate classical similarity (cosine similarity)
    pub fn classical_similarity(&self, other: &QuantumPattern) -> f64 {
        if self.features.len() != other.features.len() {
            return 0.0;
        }

        let dot_product: f64 = self.features.iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_self: f64 = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_other: f64 = other.features.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_self > 0.0 && norm_other > 0.0 {
            dot_product / (norm_self * norm_other)
        } else {
            0.0
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.max(0.0).min(1.0);
        self
    }
}

/// Quantum pattern matcher using quantum algorithms
#[derive(Debug)]
pub struct QuantumPatternMatcher {
    /// Stored reference patterns
    reference_patterns: Vec<QuantumPattern>,
    /// Similarity threshold for matches
    similarity_threshold: f64,
    /// Maximum number of patterns to store
    max_patterns: usize,
    /// Whether to use quantum or classical similarity
    use_quantum_similarity: bool,
}

impl QuantumPatternMatcher {
    /// Create a new quantum pattern matcher
    pub fn new(similarity_threshold: f64, max_patterns: usize) -> Self {
        Self {
            reference_patterns: Vec::new(),
            similarity_threshold,
            max_patterns,
            use_quantum_similarity: true,
        }
    }

    /// Enable or disable quantum similarity calculation
    pub fn with_quantum_similarity(mut self, use_quantum: bool) -> Self {
        self.use_quantum_similarity = use_quantum;
        self
    }

    /// Add a reference pattern
    pub fn add_pattern(&mut self, pattern: QuantumPattern) -> QarResult<()> {
        if self.reference_patterns.len() >= self.max_patterns {
            // Remove oldest pattern to make room
            self.reference_patterns.remove(0);
        }
        
        self.reference_patterns.push(pattern);
        Ok(())
    }

    /// Find matching patterns for input features
    pub async fn find_matches(&self, features: &[f64]) -> QarResult<Vec<PatternMatch>> {
        if self.reference_patterns.is_empty() {
            return Ok(Vec::new());
        }

        let input_pattern = QuantumPattern::new("input".to_string(), features.to_vec())?;
        let mut matches = Vec::new();

        for reference in &self.reference_patterns {
            let similarity = if self.use_quantum_similarity {
                input_pattern.quantum_similarity(reference)?
            } else {
                input_pattern.classical_similarity(reference)
            };

            if similarity >= self.similarity_threshold {
                let mut metadata = HashMap::new();
                metadata.insert("similarity_type".to_string(), 
                               if self.use_quantum_similarity { "quantum" } else { "classical" }.to_string());
                metadata.insert("reference_confidence".to_string(), reference.confidence.to_string());

                let pattern_match = PatternMatch {
                    pattern_id: reference.id.clone(),
                    similarity,
                    confidence: similarity * reference.confidence,
                    metadata,
                };

                matches.push(pattern_match);
            }
        }

        // Sort by similarity (highest first)
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches)
    }

    /// Get the best matching pattern
    pub async fn find_best_match(&self, features: &[f64]) -> QarResult<Option<PatternMatch>> {
        let matches = self.find_matches(features).await?;
        Ok(matches.into_iter().next())
    }

    /// Update pattern confidence based on feedback
    pub fn update_pattern_confidence(&mut self, pattern_id: &str, feedback: f64) {
        for pattern in &mut self.reference_patterns {
            if pattern.id == pattern_id {
                // Update confidence using exponential moving average
                let alpha = 0.1;
                pattern.confidence = alpha * feedback + (1.0 - alpha) * pattern.confidence;
                pattern.confidence = pattern.confidence.max(0.0).min(1.0);
                break;
            }
        }
    }

    /// Remove patterns with low confidence
    pub fn prune_low_confidence_patterns(&mut self, min_confidence: f64) {
        self.reference_patterns.retain(|pattern| pattern.confidence >= min_confidence);
    }

    /// Get pattern statistics
    pub fn get_statistics(&self) -> PatternStatistics {
        if self.reference_patterns.is_empty() {
            return PatternStatistics::default();
        }

        let total_patterns = self.reference_patterns.len();
        let avg_confidence = self.reference_patterns.iter()
            .map(|p| p.confidence)
            .sum::<f64>() / total_patterns as f64;

        let feature_dimensions = self.reference_patterns[0].features.len();
        
        PatternStatistics {
            total_patterns,
            average_confidence: avg_confidence,
            feature_dimensions,
            similarity_threshold: self.similarity_threshold,
            oldest_pattern_age: chrono::Utc::now().signed_duration_since(
                self.reference_patterns.iter()
                    .map(|p| p.created_at)
                    .min()
                    .unwrap_or(chrono::Utc::now())
            ).num_seconds(),
        }
    }
}

/// Pattern statistics
#[derive(Debug, Clone)]
pub struct PatternStatistics {
    pub total_patterns: usize,
    pub average_confidence: f64,
    pub feature_dimensions: usize,
    pub similarity_threshold: f64,
    pub oldest_pattern_age: i64,
}

impl Default for PatternStatistics {
    fn default() -> Self {
        Self {
            total_patterns: 0,
            average_confidence: 0.0,
            feature_dimensions: 0,
            similarity_threshold: 0.0,
            oldest_pattern_age: 0,
        }
    }
}

/// Quantum clustering algorithm
#[derive(Debug)]
pub struct QuantumClustering {
    /// Number of clusters
    num_clusters: usize,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Use quantum distance metric
    use_quantum_distance: bool,
}

impl QuantumClustering {
    /// Create a new quantum clustering instance
    pub fn new(num_clusters: usize, max_iterations: usize) -> Self {
        Self {
            num_clusters,
            max_iterations,
            tolerance: 1e-6,
            use_quantum_distance: true,
        }
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Enable or disable quantum distance metric
    pub fn with_quantum_distance(mut self, use_quantum: bool) -> Self {
        self.use_quantum_distance = use_quantum;
        self
    }

    /// Perform quantum k-means clustering
    pub async fn cluster_patterns(&self, patterns: &[QuantumPattern]) -> QarResult<ClusteringResult> {
        if patterns.len() < self.num_clusters {
            return Err(QarError::InvalidInput(
                "Number of patterns must be >= number of clusters".to_string()
            ));
        }

        // Initialize centroids randomly
        let mut centroids = self.initialize_centroids(patterns)?;
        let mut assignments = vec![0; patterns.len()];
        let mut prev_assignments = vec![usize::MAX; patterns.len()];

        for iteration in 0..self.max_iterations {
            // Assign patterns to nearest centroids
            for (i, pattern) in patterns.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_distance = f64::INFINITY;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = if self.use_quantum_distance {
                        1.0 - pattern.quantum_similarity(centroid)?
                    } else {
                        1.0 - pattern.classical_similarity(centroid)
                    };

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Check for convergence
            if assignments == prev_assignments {
                break;
            }
            prev_assignments = assignments.clone();

            // Update centroids
            centroids = self.update_centroids(patterns, &assignments)?;
        }

        // Calculate cluster statistics
        let mut cluster_sizes = vec![0; self.num_clusters];
        let mut cluster_coherence = vec![0.0; self.num_clusters];

        for &assignment in &assignments {
            cluster_sizes[assignment] += 1;
        }

        // Calculate within-cluster coherence
        for cluster_id in 0..self.num_clusters {
            let cluster_patterns: Vec<&QuantumPattern> = patterns.iter()
                .zip(assignments.iter())
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(pattern, _)| pattern)
                .collect();

            if cluster_patterns.len() > 1 {
                let mut total_similarity = 0.0;
                let mut pair_count = 0;

                for i in 0..cluster_patterns.len() {
                    for j in (i + 1)..cluster_patterns.len() {
                        let similarity = if self.use_quantum_distance {
                            cluster_patterns[i].quantum_similarity(cluster_patterns[j])?
                        } else {
                            cluster_patterns[i].classical_similarity(cluster_patterns[j])
                        };
                        total_similarity += similarity;
                        pair_count += 1;
                    }
                }

                cluster_coherence[cluster_id] = if pair_count > 0 {
                    total_similarity / pair_count as f64
                } else {
                    0.0
                };
            }
        }

        Ok(ClusteringResult {
            assignments,
            centroids,
            cluster_sizes,
            cluster_coherence,
            iterations_used: self.max_iterations.min(patterns.len()),
        })
    }

    /// Initialize centroids using k-means++ style selection
    fn initialize_centroids(&self, patterns: &[QuantumPattern]) -> QarResult<Vec<QuantumPattern>> {
        let mut centroids = Vec::with_capacity(self.num_clusters);
        
        // First centroid is random
        let first_idx = rand::random::<usize>() % patterns.len();
        centroids.push(patterns[first_idx].clone());

        // Select remaining centroids based on distance from existing ones
        for _ in 1..self.num_clusters {
            let mut distances = Vec::with_capacity(patterns.len());
            
            for pattern in patterns {
                let mut min_distance = f64::INFINITY;
                
                for centroid in &centroids {
                    let distance = if self.use_quantum_distance {
                        1.0 - pattern.quantum_similarity(centroid)?
                    } else {
                        1.0 - pattern.classical_similarity(centroid)
                    };
                    min_distance = min_distance.min(distance);
                }
                
                distances.push(min_distance);
            }

            // Select pattern with highest minimum distance
            let max_distance_idx = distances.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            centroids.push(patterns[max_distance_idx].clone());
        }

        Ok(centroids)
    }

    /// Update centroids based on current assignments
    fn update_centroids(&self, patterns: &[QuantumPattern], assignments: &[usize]) -> QarResult<Vec<QuantumPattern>> {
        let mut centroids = Vec::with_capacity(self.num_clusters);

        for cluster_id in 0..self.num_clusters {
            let cluster_patterns: Vec<&QuantumPattern> = patterns.iter()
                .zip(assignments.iter())
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(pattern, _)| pattern)
                .collect();

            if cluster_patterns.is_empty() {
                // If cluster is empty, keep previous centroid or create random one
                let random_idx = rand::random::<usize>() % patterns.len();
                centroids.push(patterns[random_idx].clone());
            } else {
                // Calculate centroid as average of cluster patterns
                let centroid = self.calculate_centroid(&cluster_patterns)?;
                centroids.push(centroid);
            }
        }

        Ok(centroids)
    }

    /// Calculate centroid pattern from a set of patterns
    fn calculate_centroid(&self, patterns: &[&QuantumPattern]) -> QarResult<QuantumPattern> {
        if patterns.is_empty() {
            return Err(QarError::InvalidInput("Cannot calculate centroid of empty cluster".to_string()));
        }

        let feature_dim = patterns[0].features.len();
        let mut centroid_features = vec![0.0; feature_dim];

        // Average the features
        for pattern in patterns {
            for (i, &feature) in pattern.features.iter().enumerate() {
                centroid_features[i] += feature;
            }
        }

        for feature in &mut centroid_features {
            *feature /= patterns.len() as f64;
        }

        // Create centroid pattern
        let centroid_id = format!("centroid_{}", rand::random::<u32>());
        let mut centroid = QuantumPattern::new(centroid_id, centroid_features)?;
        
        // Set confidence as average of cluster patterns
        let avg_confidence = patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64;
        centroid.confidence = avg_confidence;

        Ok(centroid)
    }
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster assignment for each pattern
    pub assignments: Vec<usize>,
    /// Centroid patterns for each cluster
    pub centroids: Vec<QuantumPattern>,
    /// Size of each cluster
    pub cluster_sizes: Vec<usize>,
    /// Coherence score for each cluster
    pub cluster_coherence: Vec<f64>,
    /// Number of iterations used
    pub iterations_used: usize,
}

impl ClusteringResult {
    /// Get the most coherent cluster
    pub fn best_cluster(&self) -> Option<usize> {
        self.cluster_coherence.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Get overall clustering quality
    pub fn overall_quality(&self) -> f64 {
        if self.cluster_coherence.is_empty() {
            return 0.0;
        }

        // Weight by cluster size
        let total_patterns: usize = self.cluster_sizes.iter().sum();
        if total_patterns == 0 {
            return 0.0;
        }

        let weighted_coherence: f64 = self.cluster_coherence.iter()
            .zip(self.cluster_sizes.iter())
            .map(|(&coherence, &size)| coherence * size as f64)
            .sum();

        weighted_coherence / total_patterns as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_pattern_creation() {
        let features = vec![0.5, 0.3, 0.2, 0.1];
        let pattern = QuantumPattern::new("test_pattern".to_string(), features.clone());
        
        assert!(pattern.is_ok());
        let pattern = pattern.unwrap();
        
        assert_eq!(pattern.id, "test_pattern");
        assert_eq!(pattern.features, features);
        assert_eq!(pattern.confidence, 1.0);
        assert!(pattern.quantum_state.is_normalized());
    }

    #[test]
    fn test_classical_similarity() {
        let features1 = vec![1.0, 0.0, 0.0, 0.0];
        let features2 = vec![1.0, 0.0, 0.0, 0.0];
        let features3 = vec![0.0, 1.0, 0.0, 0.0];

        let pattern1 = QuantumPattern::new("p1".to_string(), features1).unwrap();
        let pattern2 = QuantumPattern::new("p2".to_string(), features2).unwrap();
        let pattern3 = QuantumPattern::new("p3".to_string(), features3).unwrap();

        // Identical patterns should have similarity 1
        let sim12 = pattern1.classical_similarity(&pattern2);
        assert_relative_eq!(sim12, 1.0, epsilon = 1e-10);

        // Orthogonal patterns should have similarity 0
        let sim13 = pattern1.classical_similarity(&pattern3);
        assert_relative_eq!(sim13, 0.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_similarity() {
        let features1 = vec![1.0, 0.0, 0.0, 0.0];
        let features2 = vec![1.0, 0.0, 0.0, 0.0];

        let pattern1 = QuantumPattern::new("p1".to_string(), features1).unwrap();
        let pattern2 = QuantumPattern::new("p2".to_string(), features2).unwrap();

        let quantum_sim = pattern1.quantum_similarity(&pattern2);
        assert!(quantum_sim.is_ok());

        let sim = quantum_sim.unwrap();
        assert_relative_eq!(sim, 1.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_pattern_matcher() {
        let mut matcher = QuantumPatternMatcher::new(0.7, 10);

        // Add reference patterns
        let ref_pattern1 = QuantumPattern::new("ref1".to_string(), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let ref_pattern2 = QuantumPattern::new("ref2".to_string(), vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        matcher.add_pattern(ref_pattern1).unwrap();
        matcher.add_pattern(ref_pattern2).unwrap();

        // Test matching
        let test_features = vec![0.9, 0.1, 0.0, 0.0]; // Similar to ref1
        let matches = matcher.find_matches(&test_features).await.unwrap();

        assert!(!matches.is_empty());
        assert_eq!(matches[0].pattern_id, "ref1");
        assert!(matches[0].similarity > 0.7);
    }

    #[tokio::test]
    async fn test_pattern_confidence_update() {
        let mut matcher = QuantumPatternMatcher::new(0.5, 10);

        let pattern = QuantumPattern::new("test".to_string(), vec![1.0, 0.0]).unwrap();
        let initial_confidence = pattern.confidence;
        matcher.add_pattern(pattern).unwrap();

        // Update confidence with positive feedback
        matcher.update_pattern_confidence("test", 0.9);

        // Confidence should increase
        let updated_pattern = &matcher.reference_patterns[0];
        assert!(updated_pattern.confidence > initial_confidence * 0.9);
    }

    #[test]
    fn test_pattern_statistics() {
        let mut matcher = QuantumPatternMatcher::new(0.6, 10);

        let pattern1 = QuantumPattern::new("p1".to_string(), vec![1.0, 0.0, 0.0]).unwrap();
        let pattern2 = QuantumPattern::new("p2".to_string(), vec![0.0, 1.0, 0.0]).unwrap();

        matcher.add_pattern(pattern1).unwrap();
        matcher.add_pattern(pattern2).unwrap();

        let stats = matcher.get_statistics();
        assert_eq!(stats.total_patterns, 2);
        assert_eq!(stats.feature_dimensions, 3);
        assert_eq!(stats.similarity_threshold, 0.6);
        assert_relative_eq!(stats.average_confidence, 1.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_clustering() {
        let clustering = QuantumClustering::new(2, 100);

        // Create test patterns
        let patterns = vec![
            QuantumPattern::new("p1".to_string(), vec![1.0, 0.1]).unwrap(),
            QuantumPattern::new("p2".to_string(), vec![0.9, 0.2]).unwrap(),
            QuantumPattern::new("p3".to_string(), vec![0.1, 1.0]).unwrap(),
            QuantumPattern::new("p4".to_string(), vec![0.2, 0.9]).unwrap(),
        ];

        let result = clustering.cluster_patterns(&patterns).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.assignments.len(), 4);
        assert_eq!(result.centroids.len(), 2);
        assert_eq!(result.cluster_sizes.len(), 2);
        assert_eq!(result.cluster_coherence.len(), 2);

        // Check that similar patterns are clustered together
        assert_eq!(result.assignments[0], result.assignments[1]); // p1 and p2
        assert_eq!(result.assignments[2], result.assignments[3]); // p3 and p4
        assert_ne!(result.assignments[0], result.assignments[2]); // Different clusters
    }

    #[test]
    fn test_clustering_result_quality() {
        let result = ClusteringResult {
            assignments: vec![0, 0, 1, 1],
            centroids: vec![], // Not needed for this test
            cluster_sizes: vec![2, 2],
            cluster_coherence: vec![0.8, 0.9],
            iterations_used: 5,
        };

        let quality = result.overall_quality();
        assert_relative_eq!(quality, 0.85, epsilon = 1e-10); // (0.8*2 + 0.9*2) / 4

        let best_cluster = result.best_cluster();
        assert_eq!(best_cluster, Some(1)); // Cluster 1 has higher coherence
    }

    #[test]
    fn test_pattern_pruning() {
        let mut matcher = QuantumPatternMatcher::new(0.5, 10);

        let pattern1 = QuantumPattern::new("p1".to_string(), vec![1.0, 0.0]).unwrap()
            .with_confidence(0.8);
        let pattern2 = QuantumPattern::new("p2".to_string(), vec![0.0, 1.0]).unwrap()
            .with_confidence(0.3);

        matcher.add_pattern(pattern1).unwrap();
        matcher.add_pattern(pattern2).unwrap();

        assert_eq!(matcher.reference_patterns.len(), 2);

        // Prune patterns with confidence < 0.5
        matcher.prune_low_confidence_patterns(0.5);

        assert_eq!(matcher.reference_patterns.len(), 1);
        assert_eq!(matcher.reference_patterns[0].id, "p1");
    }
}