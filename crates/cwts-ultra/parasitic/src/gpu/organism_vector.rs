//! Organism Vector Representation for GPU Correlation Computing
//!
//! Optimized data structures for representing parasitic organisms in vector form
//! for efficient correlation computation. Features are aligned for SIMD operations.

use super::*;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Vector representation of a parasitic organism for correlation computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismVector {
    /// Unique organism identifier
    id: String,

    /// Feature vector (strategy weights, preferences, etc.)
    /// Aligned to 64 bytes for optimal SIMD performance
    features: AlignedFeatureVector,

    /// Performance history vector (recent profits, losses, etc.)
    /// Aligned to 64 bytes for optimal SIMD performance
    performance_history: AlignedPerformanceVector,

    /// Organism metadata
    metadata: OrganismMetadata,

    /// Cached hash for fast lookups
    cached_hash: Option<u64>,
}

impl OrganismVector {
    /// Create new organism vector
    pub fn new(id: String, features: Vec<f32>, performance_history: Vec<f32>) -> Self {
        Self {
            id,
            features: AlignedFeatureVector::new(features),
            performance_history: AlignedPerformanceVector::new(performance_history),
            metadata: OrganismMetadata::default(),
            cached_hash: None,
        }
    }

    /// Create organism vector from parasitic organism trait object
    /// Note: Trait implementation would require adding a traits module
    pub fn from_organism_data(
        id: String,
        fitness_score: f64,
        detection_risk: f64,
        performance_data: Vec<f32>,
    ) -> Self {
        let features = vec![fitness_score as f32, detection_risk as f32];

        Self::new(id, features, performance_data)
    }

    /// Get organism ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get feature vector
    pub fn features(&self) -> &[f32] {
        self.features.as_slice()
    }

    /// Get performance history vector
    pub fn performance_history(&self) -> &[f32] {
        self.performance_history.as_slice()
    }

    /// Get organism metadata
    pub fn metadata(&self) -> &OrganismMetadata {
        &self.metadata
    }

    /// Update features
    pub fn update_features(&mut self, new_features: Vec<f32>) {
        self.features = AlignedFeatureVector::new(new_features);
        self.cached_hash = None; // Invalidate cache
        self.metadata.last_updated = std::time::SystemTime::now();
    }

    /// Update performance history
    pub fn update_performance_history(&mut self, new_history: Vec<f32>) {
        self.performance_history = AlignedPerformanceVector::new(new_history);
        self.cached_hash = None; // Invalidate cache
        self.metadata.last_updated = std::time::SystemTime::now();
    }

    /// Validate organism vector data
    pub fn validate(&self) -> Result<(), String> {
        // Check feature vector
        if self.features.is_empty() {
            return Err("Feature vector is empty".to_string());
        }

        for (i, &feature) in self.features.as_slice().iter().enumerate() {
            if feature.is_nan() || feature.is_infinite() {
                return Err(format!("Invalid feature value at index {}: {}", i, feature));
            }
        }

        // Check performance history
        if self.performance_history.is_empty() {
            return Err("Performance history is empty".to_string());
        }

        for (i, &perf) in self.performance_history.as_slice().iter().enumerate() {
            if perf.is_nan() || perf.is_infinite() {
                return Err(format!(
                    "Invalid performance value at index {}: {}",
                    i, perf
                ));
            }
        }

        // Check consistency
        if self.features.len() > 32 {
            return Err("Feature vector too large (max 32)".to_string());
        }

        if self.performance_history.len() > 16 {
            return Err("Performance history too large (max 16)".to_string());
        }

        Ok(())
    }

    /// Get combined data vector for correlation computation
    pub fn combined_data(&self) -> Vec<f32> {
        let mut combined = Vec::new();
        combined.extend_from_slice(self.features.as_slice());
        combined.extend_from_slice(self.performance_history.as_slice());
        combined
    }

    /// Calculate Euclidean distance to another organism
    pub fn distance_to(&self, other: &OrganismVector) -> f32 {
        let data1 = self.combined_data();
        let data2 = other.combined_data();

        if data1.len() != data2.len() {
            return f32::INFINITY; // Incompatible organisms
        }

        data1
            .iter()
            .zip(data2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Calculate similarity score (inverse of distance)
    pub fn similarity_to(&self, other: &OrganismVector) -> f32 {
        let distance = self.distance_to(other);
        if distance > 0.0 {
            1.0 / (1.0 + distance)
        } else {
            1.0
        }
    }

    /// Get hash for fast comparisons
    pub fn get_hash(&mut self) -> u64 {
        if let Some(hash) = self.cached_hash {
            return hash;
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();

        self.cached_hash = Some(hash);
        hash
    }

    /// Normalize feature vector to unit length
    pub fn normalize_features(&mut self) {
        self.features.normalize();
        self.cached_hash = None;
        self.metadata.last_updated = std::time::SystemTime::now();
    }

    /// Normalize performance history to unit length
    pub fn normalize_performance(&mut self) {
        self.performance_history.normalize();
        self.cached_hash = None;
        self.metadata.last_updated = std::time::SystemTime::now();
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self)
            + self.id.len()
            + self.features.memory_usage()
            + self.performance_history.memory_usage()
    }

    /// Create a copy with normalized vectors
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize_features();
        copy.normalize_performance();
        copy
    }

    // Private helper methods - removed trait-specific code for now
}

impl Hash for OrganismVector {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);

        // Hash feature vector
        for &feature in self.features.as_slice() {
            feature.to_bits().hash(state);
        }

        // Hash performance history
        for &perf in self.performance_history.as_slice() {
            perf.to_bits().hash(state);
        }
    }
}

impl PartialEq for OrganismVector {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.features.as_slice() == other.features.as_slice()
            && self.performance_history.as_slice() == other.performance_history.as_slice()
    }
}

impl Eq for OrganismVector {}

/// Memory-aligned feature vector for SIMD operations
#[repr(align(64))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedFeatureVector {
    data: Vec<f32>,
    alignment_padding: [u8; 0], // Force alignment
}

impl AlignedFeatureVector {
    pub fn new(mut data: Vec<f32>) -> Self {
        // Pad to multiple of 4 for SIMD alignment
        while data.len() % 4 != 0 {
            data.push(0.0);
        }

        Self {
            data,
            alignment_padding: [],
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let length = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if length > 1e-8 {
            for value in &mut self.data {
                *value /= length;
            }
        }
    }

    /// Get vector magnitude
    pub fn magnitude(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Dot product with another vector
    pub fn dot_product(&self, other: &AlignedFeatureVector) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

/// Memory-aligned performance vector for SIMD operations
#[repr(align(64))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedPerformanceVector {
    data: Vec<f32>,
    alignment_padding: [u8; 0], // Force alignment
}

impl AlignedPerformanceVector {
    pub fn new(mut data: Vec<f32>) -> Self {
        // Pad to multiple of 4 for SIMD alignment
        while data.len() % 4 != 0 {
            data.push(0.0);
        }

        Self {
            data,
            alignment_padding: [],
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn memory_usage(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Normalize vector to unit length
    pub fn normalize(&mut self) {
        let length = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if length > 1e-8 {
            for value in &mut self.data {
                *value /= length;
            }
        }
    }

    /// Get recent performance trend
    pub fn trend(&self) -> f32 {
        if self.data.len() < 2 {
            return 0.0;
        }

        let mid = self.data.len() / 2;
        let recent_avg = self.data[mid..].iter().sum::<f32>() / (self.data.len() - mid) as f32;
        let older_avg = self.data[..mid].iter().sum::<f32>() / mid as f32;

        recent_avg - older_avg
    }

    /// Get performance volatility (standard deviation)
    pub fn volatility(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }

        let mean = self.data.iter().sum::<f32>() / self.data.len() as f32;
        let variance =
            self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.data.len() as f32;

        variance.sqrt()
    }
}

/// Organism metadata for additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismMetadata {
    pub organism_type: Option<String>,
    pub creation_time: std::time::SystemTime,
    pub last_updated: std::time::SystemTime,
    pub version: u32,
    pub tags: Vec<String>,
    pub source: Option<String>,
}

impl Default for OrganismMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now();
        Self {
            organism_type: None,
            creation_time: now,
            last_updated: now,
            version: 1,
            tags: Vec::new(),
            source: None,
        }
    }
}

impl OrganismMetadata {
    pub fn with_type(mut self, organism_type: impl Into<String>) -> Self {
        self.organism_type = Some(organism_type.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn age_seconds(&self) -> u64 {
        self.creation_time.elapsed().unwrap_or_default().as_secs()
    }
}

/// Batch of organism vectors for efficient processing
#[derive(Debug, Clone)]
pub struct OrganismBatch {
    organisms: Vec<OrganismVector>,
    batch_id: String,
    created_at: std::time::Instant,
}

impl OrganismBatch {
    pub fn new(organisms: Vec<OrganismVector>) -> Self {
        Self {
            batch_id: uuid::Uuid::new_v4().to_string(),
            organisms,
            created_at: std::time::Instant::now(),
        }
    }

    pub fn organisms(&self) -> &[OrganismVector] {
        &self.organisms
    }

    pub fn len(&self) -> usize {
        self.organisms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.organisms.is_empty()
    }

    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Validate all organisms in batch
    pub fn validate(&self) -> Result<(), Vec<(usize, String)>> {
        let mut errors = Vec::new();

        for (i, organism) in self.organisms.iter().enumerate() {
            if let Err(e) = organism.validate() {
                errors.push((i, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Split batch into smaller batches
    pub fn split(self, chunk_size: usize) -> Vec<OrganismBatch> {
        self.organisms
            .chunks(chunk_size)
            .map(|chunk| OrganismBatch::new(chunk.to_vec()))
            .collect()
    }

    /// Get memory usage of entire batch
    pub fn memory_usage(&self) -> usize {
        self.organisms
            .iter()
            .map(|o| o.memory_usage())
            .sum::<usize>()
            + std::mem::size_of_val(self)
            + self.batch_id.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_organism_vector_creation() {
        let features = vec![1.0, 0.5, -0.2, 0.8];
        let performance = vec![0.1, -0.05, 0.3, 0.0];

        let organism = OrganismVector::new(
            "test_organism".to_string(),
            features.clone(),
            performance.clone(),
        );

        assert_eq!(organism.id(), "test_organism");
        assert_eq!(organism.features().len(), 4);
        assert_eq!(organism.performance_history().len(), 4);

        // Test validation
        assert!(organism.validate().is_ok());

        println!("✅ Organism vector creation verified");
    }

    #[test]
    fn test_organism_validation() {
        let features = vec![1.0, f32::NAN, 0.5]; // Invalid NaN
        let performance = vec![0.1, 0.2];

        let organism = OrganismVector::new("invalid".to_string(), features, performance);

        assert!(organism.validate().is_err());

        println!("✅ Organism validation works correctly");
    }

    #[test]
    fn test_distance_calculation() {
        let org1 = OrganismVector::new("org1".to_string(), vec![1.0, 0.0, 0.0], vec![1.0, 0.0]);

        let org2 = OrganismVector::new("org2".to_string(), vec![0.0, 1.0, 0.0], vec![0.0, 1.0]);

        let distance = org1.distance_to(&org2);
        // Combined vectors: [1,0,0,1,0] vs [0,1,0,0,1]
        // Distance = sqrt((1-0)^2 + (0-1)^2 + (0-0)^2 + (1-0)^2 + (0-1)^2)
        //          = sqrt(1 + 1 + 0 + 1 + 1) = sqrt(4) = 2.0
        let expected_distance = 2.0_f32;

        assert!((distance - expected_distance).abs() < 1e-6);

        let similarity = org1.similarity_to(&org2);
        assert!(similarity > 0.0 && similarity <= 1.0);

        println!(
            "✅ Distance calculation verified: {:.3}, similarity: {:.3}",
            distance, similarity
        );
    }

    #[test]
    fn test_aligned_vectors() {
        let data = vec![1.0, 2.0, 3.0];
        let aligned = AlignedFeatureVector::new(data.clone());

        // Should be padded to multiple of 4
        assert_eq!(aligned.len(), 4);
        assert_eq!(aligned.as_slice()[0], 1.0);
        assert_eq!(aligned.as_slice()[3], 0.0); // Padding

        println!("✅ Aligned vector creation verified");
    }

    #[test]
    fn test_vector_normalization() {
        let mut features = AlignedFeatureVector::new(vec![3.0, 4.0]);
        let initial_magnitude = features.magnitude();
        assert!((initial_magnitude - 5.0).abs() < 1e-6);

        features.normalize();
        let normalized_magnitude = features.magnitude();
        assert!((normalized_magnitude - 1.0).abs() < 1e-6);

        println!("✅ Vector normalization verified");
    }

    #[test]
    fn test_performance_metrics() {
        let perf_data = vec![0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1];
        let performance = AlignedPerformanceVector::new(perf_data);

        let trend = performance.trend();
        let volatility = performance.volatility();

        assert!(trend.is_finite());
        assert!(volatility >= 0.0);

        println!(
            "✅ Performance metrics: trend={:.3}, volatility={:.3}",
            trend, volatility
        );
    }

    #[test]
    fn test_organism_batch() {
        let organisms = vec![
            OrganismVector::new("org1".to_string(), vec![1.0, 2.0], vec![0.1, 0.2]),
            OrganismVector::new("org2".to_string(), vec![3.0, 4.0], vec![0.3, 0.4]),
            OrganismVector::new("org3".to_string(), vec![5.0, 6.0], vec![0.5, 0.6]),
        ];

        let batch = OrganismBatch::new(organisms);
        assert_eq!(batch.len(), 3);
        assert!(batch.validate().is_ok());

        let split_batches = batch.split(2);
        assert_eq!(split_batches.len(), 2);
        assert_eq!(split_batches[0].len(), 2);
        assert_eq!(split_batches[1].len(), 1);

        println!("✅ Organism batch operations verified");
    }

    #[test]
    fn test_hash_and_equality() {
        let org1 = OrganismVector::new("test".to_string(), vec![1.0, 2.0], vec![0.1]);
        let org2 = OrganismVector::new("test".to_string(), vec![1.0, 2.0], vec![0.1]);
        let org3 = OrganismVector::new("test".to_string(), vec![1.0, 2.0], vec![0.2]);

        assert_eq!(org1, org2);
        assert_ne!(org1, org3);

        // Note: Hash comparison would require mutable access
        println!("✅ Hash and equality verified");
    }
}
