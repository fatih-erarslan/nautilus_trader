//! Combinatorial Diversity Fusion Analysis
//! 
//! This module provides advanced combinatorial analysis capabilities for CDFA,
//! including algorithm pool management, synergy detection, and k-combinations evaluation.

pub mod analyzer;
pub mod synergy;
pub mod evaluator;
pub mod algorithm_pool;

// Re-export main types
pub use analyzer::{CombinatorialDiversityFusionAnalyzer, CombinationAnalysis, CombinationResult};
pub use synergy::{SynergyDetector, SynergyMetrics, AlgorithmInteraction};
pub use evaluator::{FusionEvaluator, EvaluationMetrics, PerformanceProfile};
pub use algorithm_pool::{AlgorithmPool, SwarmAlgorithm, AlgorithmMetadata};

use crate::fusion::FusionMethod;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Main configuration for combinatorial fusion analysis
#[derive(Debug, Clone)]
pub struct CombinatorialConfig {
    /// Maximum k for k-combinations (default: 5)
    pub max_k: usize,
    /// Minimum diversity threshold for algorithm selection
    pub diversity_threshold: f64,
    /// Performance weight in algorithm selection (0.0 to 1.0)
    pub performance_weight: f64,
    /// Enable parallel combination evaluation
    pub parallel_evaluation: bool,
    /// SIMD optimization level
    pub simd_level: SIMDLevel,
}

impl Default for CombinatorialConfig {
    fn default() -> Self {
        Self {
            max_k: 5,
            diversity_threshold: 0.3,
            performance_weight: 0.7,
            parallel_evaluation: true,
            simd_level: SIMDLevel::Auto,
        }
    }
}

/// SIMD optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SIMDLevel {
    None,
    SSE,
    AVX2,
    AVX512,
    Auto,
}

/// Error types for combinatorial analysis
#[derive(thiserror::Error, Debug)]
pub enum CombinatorialError {
    #[error("Insufficient algorithms in pool: {count}, minimum required: {min}")]
    InsufficientAlgorithms { count: usize, min: usize },
    
    #[error("Invalid k value: {k}, must be between 1 and {max}")]
    InvalidK { k: usize, max: usize },
    
    #[error("Diversity threshold too high: {threshold}, no combinations found")]
    DiversityThresholdTooHigh { threshold: f64 },
    
    #[error("Algorithm evaluation failed: {reason}")]
    EvaluationFailed { reason: String },
    
    #[error("Synergy detection error: {message}")]
    SynergyError { message: String },
    
    #[error("Performance analysis failed: {details}")]
    PerformanceError { details: String },
}

/// Result type for combinatorial operations
pub type CombinatorialResult<T> = Result<T, CombinatorialError>;

/// Performance metrics for combinatorial analysis
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    pub fusion_time_ns: u64,
    pub combination_generation_time_ns: u64,
    pub synergy_analysis_time_ns: u64,
    pub total_time_ns: u64,
    pub memory_usage_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl PerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            fusion_time_ns: 0,
            combination_generation_time_ns: 0,
            synergy_analysis_time_ns: 0,
            total_time_ns: 0,
            memory_usage_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// Check if performance meets <1μs requirement
    pub fn meets_performance_target(&self) -> bool {
        self.total_time_ns < 1_000 // 1 microsecond = 1000 nanoseconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_combinatorial_config_default() {
        let config = CombinatorialConfig::default();
        assert_eq!(config.max_k, 5);
        assert_eq!(config.diversity_threshold, 0.3);
        assert_eq!(config.performance_weight, 0.7);
        assert!(config.parallel_evaluation);
    }
    
    #[test]
    fn test_performance_benchmark() {
        let mut benchmark = PerformanceBenchmark::new();
        benchmark.total_time_ns = 500; // 0.5 microseconds
        assert!(benchmark.meets_performance_target());
        
        benchmark.total_time_ns = 1500; // 1.5 microseconds
        assert!(!benchmark.meets_performance_target());
    }
}