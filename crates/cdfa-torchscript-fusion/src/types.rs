//! Types and data structures for TorchScript Fusion operations
//!
//! This module defines the core types used throughout the fusion system,
//! including fusion types, results, and configuration parameters.

use crate::error::{FusionError, Result};
use candle_core::Tensor;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Types of fusion algorithms available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionType {
    /// Score-based fusion: Confidence-weighted linear combination
    Score,
    /// Rank-based fusion: Ordering-based combination
    Rank,
    /// Hybrid fusion: Adaptive combination of score and rank methods
    Hybrid,
    /// Weighted fusion: Diversity-aware weighted combination
    Weighted,
    /// Layered fusion: Hierarchical fusion with sub-grouping
    Layered,
    /// Adaptive fusion: Dynamic method selection based on signal properties
    Adaptive,
}

impl FusionType {
    /// Get all available fusion types
    pub fn all() -> Vec<FusionType> {
        vec![
            FusionType::Score,
            FusionType::Rank,
            FusionType::Hybrid,
            FusionType::Weighted,
            FusionType::Layered,
            FusionType::Adaptive,
        ]
    }

    /// Get the fusion type as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            FusionType::Score => "score",
            FusionType::Rank => "rank",
            FusionType::Hybrid => "hybrid",
            FusionType::Weighted => "weighted",
            FusionType::Layered => "layered",
            FusionType::Adaptive => "adaptive",
        }
    }

    /// Check if this fusion type requires diversity calculation
    pub fn requires_diversity(&self) -> bool {
        matches!(self, FusionType::Weighted | FusionType::Adaptive)
    }

    /// Check if this fusion type uses ranking
    pub fn uses_ranking(&self) -> bool {
        matches!(self, FusionType::Rank | FusionType::Hybrid | FusionType::Adaptive)
    }

    /// Get the computational complexity category
    pub fn complexity(&self) -> ComputationalComplexity {
        match self {
            FusionType::Score => ComputationalComplexity::Low,
            FusionType::Rank => ComputationalComplexity::Medium,
            FusionType::Hybrid => ComputationalComplexity::Medium,
            FusionType::Weighted => ComputationalComplexity::High,
            FusionType::Layered => ComputationalComplexity::Medium,
            FusionType::Adaptive => ComputationalComplexity::High,
        }
    }
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for FusionType {
    type Err = FusionError;

    fn from_str(s: &str) -> Result<Self> {
        let s_lower = s.to_lowercase();
        match s_lower.as_str() {
            "score" => Ok(FusionType::Score),
            "rank" => Ok(FusionType::Rank),
            "hybrid" => Ok(FusionType::Hybrid),
            "weighted" => Ok(FusionType::Weighted),
            "layered" => Ok(FusionType::Layered),
            "adaptive" => Ok(FusionType::Adaptive),
            _ => Err(FusionError::unsupported_fusion_type(s)),
        }
    }
}

/// Computational complexity categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputationalComplexity {
    /// Low complexity: O(n)
    Low,
    /// Medium complexity: O(n log n) to O(n²)
    Medium,
    /// High complexity: O(n²) to O(n³)
    High,
}

impl fmt::Display for ComputationalComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputationalComplexity::Low => write!(f, "Low"),
            ComputationalComplexity::Medium => write!(f, "Medium"),
            ComputationalComplexity::High => write!(f, "High"),
        }
    }
}

/// Result of a fusion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// The fused signal values
    pub fused_signal: Array1<f32>,
    /// Confidence values for the fused signal
    pub confidence: Array1<f32>,
    /// Individual signal weights used in fusion
    pub weights: Vec<Array1<f32>>,
    /// Type of fusion used
    pub fusion_type: FusionType,
    /// Additional metadata
    pub metadata: FusionMetadata,
}

impl FusionResult {
    /// Create a new fusion result
    pub fn new(
        fused_signal: Array1<f32>,
        confidence: Array1<f32>,
        weights: Vec<Array1<f32>>,
        fusion_type: FusionType,
    ) -> Self {
        Self {
            fused_signal,
            confidence,
            weights,
            fusion_type,
            metadata: FusionMetadata::default(),
        }
    }

    /// Get the number of signals that were fused
    pub fn num_signals(&self) -> usize {
        self.weights.len()
    }

    /// Get the sequence length
    pub fn sequence_length(&self) -> usize {
        self.fused_signal.len()
    }

    /// Get the average confidence
    pub fn average_confidence(&self) -> f32 {
        self.confidence.mean().unwrap_or(0.0)
    }

    /// Get the signal-to-noise ratio (if available)
    pub fn signal_to_noise_ratio(&self) -> Option<f32> {
        self.metadata.signal_to_noise_ratio
    }

    /// Validate the result for consistency
    pub fn validate(&self) -> Result<()> {
        // Check that all arrays have the same length
        let seq_len = self.fused_signal.len();
        if self.confidence.len() != seq_len {
            return Err(FusionError::dimension_mismatch(
                format!("fused_signal length: {}", seq_len),
                format!("confidence length: {}", self.confidence.len()),
            ));
        }

        // Check that all weights have the correct length
        for (i, weight) in self.weights.iter().enumerate() {
            if weight.len() != seq_len {
                return Err(FusionError::dimension_mismatch(
                    format!("sequence length: {}", seq_len),
                    format!("weight[{}] length: {}", i, weight.len()),
                ));
            }
        }

        // Check that weights sum to approximately 1.0 at each time step
        for t in 0..seq_len {
            let weight_sum: f32 = self.weights.iter().map(|w| w[t]).sum();
            if (weight_sum - 1.0).abs() > 1e-3 {
                return Err(FusionError::numerical(format!(
                    "Weights at time step {} sum to {} instead of 1.0",
                    t, weight_sum
                )));
            }
        }

        // Check that all values are finite
        for val in self.fused_signal.iter() {
            if !val.is_finite() {
                return Err(FusionError::numerical("Non-finite value in fused signal"));
            }
        }

        for val in self.confidence.iter() {
            if !val.is_finite() || *val < 0.0 {
                return Err(FusionError::numerical("Invalid confidence value"));
            }
        }

        Ok(())
    }
}

/// Metadata associated with a fusion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetadata {
    /// Timestamp when the fusion was performed
    pub timestamp: Option<u64>,
    /// Inference time in microseconds
    pub inference_time_us: Option<u64>,
    /// Device used for computation
    pub device: Option<String>,
    /// Model compilation time in microseconds
    pub compilation_time_us: Option<u64>,
    /// Number of signals fused
    pub num_signals: Option<usize>,
    /// Sequence length
    pub sequence_length: Option<usize>,
    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: Option<f32>,
    /// Diversity score
    pub diversity_score: Option<f32>,
    /// Agreement score
    pub agreement_score: Option<f32>,
    /// Additional custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for FusionMetadata {
    fn default() -> Self {
        Self {
            timestamp: None,
            inference_time_us: None,
            device: None,
            compilation_time_us: None,
            num_signals: None,
            sequence_length: None,
            signal_to_noise_ratio: None,
            diversity_score: None,
            agreement_score: None,
            custom: HashMap::new(),
        }
    }
}

impl FusionMetadata {
    /// Set the timestamp to current time
    pub fn set_timestamp(&mut self) {
        self.timestamp = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        );
    }

    /// Add custom metadata
    pub fn add_custom(&mut self, key: String, value: serde_json::Value) {
        self.custom.insert(key, value);
    }

    /// Get custom metadata
    pub fn get_custom(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom.get(key)
    }
}

/// Parameters for fusion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionParams {
    /// Minimum weight threshold
    pub min_weight: f32,
    /// Score-rank mixing parameter for hybrid fusion (0.0 = all rank, 1.0 = all score)
    pub score_alpha: f32,
    /// Confidence weighting factor
    pub confidence_factor: f32,
    /// Diversity weighting factor
    pub diversity_factor: f32,
    /// Whether to use nonlinear weighting
    pub use_nonlinear_weighting: bool,
    /// Nonlinear weighting exponent
    pub nonlinear_exponent: f32,
    /// Chunk size for processing large tensors
    pub chunk_size: usize,
    /// Whether to enable numerical stability checks
    pub enable_stability_checks: bool,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            min_weight: crate::DEFAULT_MIN_WEIGHT,
            score_alpha: crate::DEFAULT_SCORE_ALPHA,
            confidence_factor: crate::DEFAULT_CONFIDENCE_FACTOR,
            diversity_factor: crate::DEFAULT_DIVERSITY_FACTOR,
            use_nonlinear_weighting: true,
            nonlinear_exponent: crate::DEFAULT_NONLINEAR_EXPONENT,
            chunk_size: crate::DEFAULT_CHUNK_SIZE,
            enable_stability_checks: true,
        }
    }
}

impl FusionParams {
    /// Create a new FusionParams with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum weight
    pub fn with_min_weight(mut self, min_weight: f32) -> Self {
        self.min_weight = min_weight;
        self
    }

    /// Set score alpha for hybrid fusion
    pub fn with_score_alpha(mut self, score_alpha: f32) -> Self {
        self.score_alpha = score_alpha;
        self
    }

    /// Set confidence factor
    pub fn with_confidence_factor(mut self, confidence_factor: f32) -> Self {
        self.confidence_factor = confidence_factor;
        self
    }

    /// Set diversity factor
    pub fn with_diversity_factor(mut self, diversity_factor: f32) -> Self {
        self.diversity_factor = diversity_factor;
        self
    }

    /// Enable or disable nonlinear weighting
    pub fn with_nonlinear_weighting(mut self, use_nonlinear: bool) -> Self {
        self.use_nonlinear_weighting = use_nonlinear;
        self
    }

    /// Set nonlinear exponent
    pub fn with_nonlinear_exponent(mut self, exponent: f32) -> Self {
        self.nonlinear_exponent = exponent;
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Enable or disable stability checks
    pub fn with_stability_checks(mut self, enable: bool) -> Self {
        self.enable_stability_checks = enable;
        self
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        if self.min_weight < 0.0 || self.min_weight > 1.0 {
            return Err(FusionError::invalid_input(
                "min_weight must be between 0.0 and 1.0",
            ));
        }

        if self.score_alpha < 0.0 || self.score_alpha > 1.0 {
            return Err(FusionError::invalid_input(
                "score_alpha must be between 0.0 and 1.0",
            ));
        }

        if self.confidence_factor < 0.0 || self.confidence_factor > 1.0 {
            return Err(FusionError::invalid_input(
                "confidence_factor must be between 0.0 and 1.0",
            ));
        }

        if self.diversity_factor < 0.0 || self.diversity_factor > 1.0 {
            return Err(FusionError::invalid_input(
                "diversity_factor must be between 0.0 and 1.0",
            ));
        }

        if (self.confidence_factor + self.diversity_factor - 1.0).abs() > 1e-6 {
            return Err(FusionError::invalid_input(
                "confidence_factor + diversity_factor must equal 1.0",
            ));
        }

        if self.nonlinear_exponent <= 0.0 {
            return Err(FusionError::invalid_input(
                "nonlinear_exponent must be positive",
            ));
        }

        if self.chunk_size == 0 {
            return Err(FusionError::invalid_input("chunk_size must be positive"));
        }

        Ok(())
    }
}

/// Performance metrics for fusion operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Inference time in microseconds
    pub inference_time_us: u64,
    /// Compilation time in microseconds
    pub compilation_time_us: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// GPU memory usage in bytes (if applicable)
    pub gpu_memory_usage_bytes: Option<u64>,
    /// Number of operations performed
    pub operations_count: u64,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Device utilization percentage
    pub device_utilization_percent: Option<f32>,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new(inference_time_us: u64, compilation_time_us: u64) -> Self {
        Self {
            inference_time_us,
            compilation_time_us,
            memory_usage_bytes: 0,
            gpu_memory_usage_bytes: None,
            operations_count: 0,
            throughput_ops_per_sec: 0.0,
            device_utilization_percent: None,
        }
    }

    /// Calculate throughput
    pub fn calculate_throughput(&mut self, operations: u64) {
        self.operations_count = operations;
        if self.inference_time_us > 0 {
            self.throughput_ops_per_sec = (operations as f64) / (self.inference_time_us as f64 / 1_000_000.0);
        }
    }

    /// Check if sub-microsecond performance is achieved
    pub fn is_sub_microsecond(&self) -> bool {
        self.inference_time_us < 1
    }

    /// Get latency category
    pub fn latency_category(&self) -> LatencyCategory {
        match self.inference_time_us {
            0..=999 => LatencyCategory::SubMicrosecond,
            1000..=9999 => LatencyCategory::Microsecond,
            10000..=99999 => LatencyCategory::TenMicrosecond,
            100000..=999999 => LatencyCategory::HundredMicrosecond,
            _ => LatencyCategory::Millisecond,
        }
    }
}

/// Latency categories for performance classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyCategory {
    /// Sub-microsecond: < 1μs
    SubMicrosecond,
    /// Microsecond: 1-9μs
    Microsecond,
    /// Ten microseconds: 10-99μs
    TenMicrosecond,
    /// Hundred microseconds: 100-999μs
    HundredMicrosecond,
    /// Millisecond: ≥ 1ms
    Millisecond,
}

impl fmt::Display for LatencyCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatencyCategory::SubMicrosecond => write!(f, "Sub-microsecond"),
            LatencyCategory::Microsecond => write!(f, "Microsecond"),
            LatencyCategory::TenMicrosecond => write!(f, "Ten microseconds"),
            LatencyCategory::HundredMicrosecond => write!(f, "Hundred microseconds"),
            LatencyCategory::Millisecond => write!(f, "Millisecond"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fusion_type_parsing() {
        assert_eq!(FusionType::from_str("score").unwrap(), FusionType::Score);
        assert_eq!(FusionType::from_str("RANK").unwrap(), FusionType::Rank);
        assert_eq!(FusionType::from_str("Hybrid").unwrap(), FusionType::Hybrid);
        assert!(FusionType::from_str("invalid").is_err());
    }

    #[test]
    fn test_fusion_type_properties() {
        assert!(FusionType::Weighted.requires_diversity());
        assert!(!FusionType::Score.requires_diversity());
        
        assert!(FusionType::Rank.uses_ranking());
        assert!(!FusionType::Score.uses_ranking());
        
        assert_eq!(FusionType::Score.complexity(), ComputationalComplexity::Low);
        assert_eq!(FusionType::Adaptive.complexity(), ComputationalComplexity::High);
    }

    #[test]
    fn test_fusion_result_validation() {
        let fused_signal = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let confidence = Array1::from_vec(vec![0.8, 0.8, 0.8]);
        let weights = vec![
            Array1::from_vec(vec![0.6, 0.6, 0.6]),
            Array1::from_vec(vec![0.4, 0.4, 0.4]),
        ];
        
        let result = FusionResult::new(fused_signal, confidence, weights, FusionType::Score);
        assert!(result.validate().is_ok());
        
        assert_eq!(result.num_signals(), 2);
        assert_eq!(result.sequence_length(), 3);
        assert_abs_diff_eq!(result.average_confidence(), 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_fusion_result_validation_errors() {
        // Test dimension mismatch
        let fused_signal = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let confidence = Array1::from_vec(vec![0.8, 0.8]); // Wrong length
        let weights = vec![Array1::from_vec(vec![1.0, 1.0, 1.0])];
        
        let result = FusionResult::new(fused_signal, confidence, weights, FusionType::Score);
        assert!(result.validate().is_err());
        
        // Test weight sum not equal to 1.0
        let fused_signal = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let confidence = Array1::from_vec(vec![0.8, 0.8, 0.8]);
        let weights = vec![
            Array1::from_vec(vec![0.3, 0.3, 0.3]),
            Array1::from_vec(vec![0.3, 0.3, 0.3]), // Sum = 0.6, not 1.0
        ];
        
        let result = FusionResult::new(fused_signal, confidence, weights, FusionType::Score);
        assert!(result.validate().is_err());
    }

    #[test]
    fn test_fusion_params_validation() {
        let mut params = FusionParams::default();
        assert!(params.validate().is_ok());
        
        params.min_weight = -0.1;
        assert!(params.validate().is_err());
        
        params.min_weight = 0.01;
        params.score_alpha = 1.5;
        assert!(params.validate().is_err());
        
        params.score_alpha = 0.5;
        params.confidence_factor = 0.8;
        params.diversity_factor = 0.3; // Sum = 1.1, not 1.0
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_fusion_params_builder() {
        let params = FusionParams::new()
            .with_min_weight(0.05)
            .with_score_alpha(0.7)
            .with_confidence_factor(0.6)
            .with_diversity_factor(0.4)
            .with_nonlinear_weighting(false)
            .with_chunk_size(500);
        
        assert_eq!(params.min_weight, 0.05);
        assert_eq!(params.score_alpha, 0.7);
        assert_eq!(params.confidence_factor, 0.6);
        assert_eq!(params.diversity_factor, 0.4);
        assert!(!params.use_nonlinear_weighting);
        assert_eq!(params.chunk_size, 500);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new(500, 1000);
        assert_eq!(metrics.inference_time_us, 500);
        assert_eq!(metrics.compilation_time_us, 1000);
        assert!(metrics.is_sub_microsecond());
        assert_eq!(metrics.latency_category(), LatencyCategory::SubMicrosecond);
        
        metrics.calculate_throughput(10000);
        assert_eq!(metrics.operations_count, 10000);
        assert!(metrics.throughput_ops_per_sec > 0.0);
    }

    #[test]
    fn test_latency_categories() {
        let metrics_sub = PerformanceMetrics::new(500, 0);
        assert_eq!(metrics_sub.latency_category(), LatencyCategory::SubMicrosecond);
        
        let metrics_micro = PerformanceMetrics::new(5000, 0);
        assert_eq!(metrics_micro.latency_category(), LatencyCategory::Microsecond);
        
        let metrics_milli = PerformanceMetrics::new(1500000, 0);
        assert_eq!(metrics_milli.latency_category(), LatencyCategory::Millisecond);
    }

    #[test]
    fn test_fusion_metadata() {
        let mut metadata = FusionMetadata::default();
        metadata.set_timestamp();
        assert!(metadata.timestamp.is_some());
        
        metadata.add_custom("test_key".to_string(), serde_json::Value::String("test_value".to_string()));
        assert_eq!(metadata.get_custom("test_key").unwrap(), &serde_json::Value::String("test_value".to_string()));
    }
}