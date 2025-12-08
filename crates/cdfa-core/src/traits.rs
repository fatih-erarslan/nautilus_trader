//! Core trait definitions for the CDFA system
//!
//! This module defines the fundamental traits that form the backbone of the
//! Cognitive Diversity Fusion Analysis system. All implementations should
//! adhere to these traits to ensure compatibility and performance.

use crate::error::Result;
use crate::types::{AnalysisResult, DiversityMatrix, FusionOutput, Signal};

#[cfg(feature = "async")]
use core::future::Future;

/// Core trait for cognitive diversity analyzers
///
/// Implementors of this trait provide different perspectives on market signals,
/// contributing to the overall cognitive diversity of the analysis system.
pub trait CognitiveDiversityAnalyzer: Send + Sync {
    /// Configuration type for this analyzer
    type Config: Default + Clone;

    /// Analyzes a batch of signals and produces diversity metrics
    ///
    /// # Arguments
    /// * `signals` - Slice of signals to analyze
    ///
    /// # Returns
    /// Analysis result containing diversity metrics and predictions
    ///
    /// # Performance
    /// Implementations should target sub-microsecond latency for typical batch sizes
    fn analyze(&self, signals: &[Signal]) -> Result<AnalysisResult>;

    /// Asynchronous version of analyze for I/O-bound operations
    #[cfg(feature = "async")]
    fn analyze_async<'a>(
        &'a self,
        signals: &'a [Signal],
    ) -> impl Future<Output = Result<AnalysisResult>> + Send + 'a {
        async move { self.analyze(signals) }
    }

    /// Returns the diversity metric IDs this analyzer computes
    fn diversity_metric_ids(&self) -> &[&'static str];

    /// Returns the current configuration
    fn config(&self) -> &Self::Config;

    /// Updates the analyzer configuration
    fn update_config(&mut self, config: Self::Config) -> Result<()>;

    /// Returns a unique identifier for this analyzer type
    fn analyzer_id(&self) -> &'static str;

    /// Estimates the computational complexity for a given input size
    ///
    /// Returns the estimated number of operations
    fn complexity_estimate(&self, input_size: usize) -> usize {
        input_size // Default O(n) complexity
    }

    /// Indicates whether this analyzer benefits from SIMD operations
    fn supports_simd(&self) -> bool {
        false
    }

    /// Optimal batch size for this analyzer
    fn optimal_batch_size(&self) -> usize {
        64 // Default to cache-line multiple
    }
}

/// Trait for fusion strategies that combine multiple analysis outputs
///
/// Fusion strategies determine how diverse analytical perspectives are
/// combined into a unified trading signal.
pub trait FusionStrategy: Send + Sync {
    /// Configuration type for this fusion strategy
    type Config: Default + Clone;

    /// Fuses multiple analysis results into a single output
    ///
    /// # Arguments
    /// * `results` - Analysis results from different analyzers
    /// * `weights` - Optional weights for each result
    ///
    /// # Returns
    /// Fused signal output with confidence metrics
    fn fuse_results(
        &self,
        results: &[AnalysisResult],
        weights: Option<&[f64]>,
    ) -> Result<FusionOutput>;

    /// Calculates the confidence score for the fusion
    ///
    /// Returns a value between 0.0 and 1.0
    fn confidence_score(&self, outputs: &[AnalysisResult]) -> f64;

    /// Returns the fusion strategy identifier
    fn strategy_id(&self) -> &'static str;

    /// Updates fusion parameters based on performance feedback
    ///
    /// # Arguments
    /// * `performance_metrics` - Recent performance data
    fn adapt(&mut self, performance_metrics: &PerformanceMetrics) -> Result<()>;

    /// Returns true if this strategy can process signals in parallel
    fn supports_parallel_fusion(&self) -> bool {
        true
    }

    /// Returns the minimum number of signals required for fusion
    fn min_signals_required(&self) -> usize {
        2
    }
}

/// Trait for diversity metrics computation
///
/// Diversity metrics quantify how different various analytical perspectives are,
/// enabling the system to maintain cognitive diversity.
pub trait DiversityMetric: Send + Sync {
    /// Computes the diversity score between two analysis results
    ///
    /// Returns a value typically between 0.0 (identical) and 1.0 (maximally diverse)
    fn compute(&self, result_a: &AnalysisResult, result_b: &AnalysisResult) -> f64;

    /// Computes a diversity matrix for multiple results
    ///
    /// # Arguments
    /// * `results` - Vector of analysis results
    ///
    /// # Returns
    /// A diversity matrix where element (i,j) represents diversity between results i and j
    fn compute_matrix(&self, results: &[AnalysisResult]) -> DiversityMatrix {
        let n = results.len();
        let mut matrix = DiversityMatrix::zeros(n);
        
        for i in 0..n {
            for j in i + 1..n {
                let diversity = self.compute(&results[i], &results[j]);
                matrix.set(i, j, diversity);
                matrix.set(j, i, diversity); // Symmetric matrix
            }
        }
        
        matrix
    }

    /// Returns the metric identifier
    fn metric_id(&self) -> &'static str;

    /// Indicates if this metric is symmetric
    fn is_symmetric(&self) -> bool {
        true
    }

    /// Returns the range of possible values [min, max]
    fn value_range(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
}

/// Trait for low-level signal processing operations
///
/// This trait provides the foundation for efficient signal manipulation
/// and feature extraction.
pub trait SignalProcessor: Send + Sync {
    /// Processes a raw signal and extracts features
    ///
    /// # Arguments
    /// * `signal` - Input signal to process
    ///
    /// # Returns
    /// Processed signal with extracted features
    fn process(&self, signal: &Signal) -> Result<Signal>;

    /// Batch processing for improved efficiency
    ///
    /// Default implementation processes signals sequentially
    fn process_batch(&self, signals: &[Signal]) -> Result<Vec<Signal>> {
        signals.iter().map(|s| self.process(s)).collect()
    }

    /// Returns true if this processor can utilize SIMD instructions
    fn supports_simd(&self) -> bool {
        false
    }

    /// Returns the processor identifier
    fn processor_id(&self) -> &'static str;

    /// Estimates processing latency in nanoseconds for a given signal size
    fn latency_estimate_ns(&self, signal_size: usize) -> u64 {
        (signal_size as u64) * 10 // Default 10ns per element
    }
}

/// Performance metrics for adaptive strategies
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Average processing latency in nanoseconds
    pub avg_latency_ns: u64,
    
    /// Prediction accuracy (0.0 to 1.0)
    pub accuracy: f64,
    
    /// Sharpe ratio of signals
    pub sharpe_ratio: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Number of signals processed
    pub signals_processed: u64,
    
    /// Timestamp of last update
    pub last_updated: u64,
}

/// Trait for components that can be initialized with hardware detection
pub trait HardwareAware {
    /// Initializes the component with detected hardware capabilities
    ///
    /// This allows components to optimize for specific CPU features
    /// like AVX-512, NEON, or other SIMD instruction sets.
    fn initialize_with_hardware(&mut self, capabilities: &HardwareCapabilities) -> Result<()>;
}

/// Hardware capabilities detection result
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Supports AVX2 instructions
    pub has_avx2: bool,
    
    /// Supports AVX-512 instructions
    pub has_avx512: bool,
    
    /// Supports ARM NEON instructions
    pub has_neon: bool,
    
    /// Number of physical CPU cores
    pub cpu_cores: usize,
    
    /// CPU cache line size in bytes
    pub cache_line_size: usize,
    
    /// Available RAM in bytes
    pub available_memory: usize,
    
    /// GPU acceleration available
    pub has_gpu: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.avg_latency_ns, 0);
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.signals_processed, 0);
    }

    #[test]
    fn test_hardware_capabilities() {
        let caps = HardwareCapabilities {
            has_avx2: true,
            has_avx512: false,
            has_neon: false,
            cpu_cores: 8,
            cache_line_size: 64,
            available_memory: 16 * 1024 * 1024 * 1024, // 16GB
            has_gpu: false,
        };
        
        assert!(caps.has_avx2);
        assert_eq!(caps.cpu_cores, 8);
        assert_eq!(caps.cache_line_size, 64);
    }
}