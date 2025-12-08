//! Configuration management for the CDFA system
//!
//! This module provides configuration structures and builders for setting up
//! CDFA components with optimal performance parameters.

use crate::error::{Error, Result};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Main configuration for the CDFA system
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CDFAConfig {
    /// Performance configuration
    pub performance: PerformanceConfig,
    
    /// Analysis configuration
    pub analysis: AnalysisConfig,
    
    /// Fusion configuration
    pub fusion: FusionConfig,
    
    /// Hardware optimization settings
    pub hardware: HardwareConfig,
    
    /// Memory management settings
    pub memory: MemoryConfig,
}

impl Default for CDFAConfig {
    fn default() -> Self {
        Self {
            performance: PerformanceConfig::default(),
            analysis: AnalysisConfig::default(),
            fusion: FusionConfig::default(),
            hardware: HardwareConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

/// Performance-related configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceConfig {
    /// Maximum latency target in nanoseconds
    pub max_latency_ns: u64,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Cache warming iterations
    pub cache_warm_iterations: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_latency_ns: 1_000_000, // 1ms default
            enable_parallel: true,
            worker_threads: 0, // Auto-detect
            batch_size: 64,
            enable_simd: true,
            cache_warm_iterations: 3,
        }
    }
}

/// Analysis-related configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnalysisConfig {
    /// Minimum number of signals required
    pub min_signals: usize,
    
    /// Maximum number of signals to process
    pub max_signals: usize,
    
    /// Window size for rolling analysis
    pub window_size: usize,
    
    /// Overlap between windows (0.0 to 1.0)
    pub window_overlap: f64,
    
    /// Enable adaptive analysis
    pub enable_adaptive: bool,
    
    /// Confidence threshold (0.0 to 1.0)
    pub confidence_threshold: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_signals: 2,
            max_signals: 1000,
            window_size: 100,
            window_overlap: 0.5,
            enable_adaptive: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Fusion strategy configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FusionConfig {
    /// Default fusion strategy
    pub default_strategy: FusionStrategyType,
    
    /// Enable dynamic weight adjustment
    pub enable_dynamic_weights: bool,
    
    /// Weight learning rate
    pub weight_learning_rate: f64,
    
    /// Minimum weight value
    pub min_weight: f64,
    
    /// Maximum weight value
    pub max_weight: f64,
    
    /// Enable outlier filtering
    pub enable_outlier_filter: bool,
    
    /// Outlier threshold (standard deviations)
    pub outlier_threshold: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            default_strategy: FusionStrategyType::WeightedAverage,
            enable_dynamic_weights: true,
            weight_learning_rate: 0.01,
            min_weight: 0.0,
            max_weight: 1.0,
            enable_outlier_filter: true,
            outlier_threshold: 3.0,
        }
    }
}

/// Fusion strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FusionStrategyType {
    /// Simple average of all signals
    SimpleAverage,
    
    /// Weighted average based on confidence
    WeightedAverage,
    
    /// Median-based fusion
    Median,
    
    /// Maximum confidence selection
    MaxConfidence,
    
    /// Ensemble voting
    EnsembleVoting,
    
    /// Bayesian fusion
    Bayesian,
    
    /// Custom strategy
    Custom,
}

/// Hardware optimization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HardwareConfig {
    /// Enable CPU feature detection
    pub enable_cpu_detection: bool,
    
    /// Force specific CPU features
    pub force_cpu_features: Vec<String>,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// GPU device index
    pub gpu_device_index: usize,
    
    /// NUMA node affinity (-1 = no affinity)
    pub numa_node: i32,
    
    /// CPU core affinity
    pub cpu_affinity: Vec<usize>,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            enable_cpu_detection: true,
            force_cpu_features: Vec::new(),
            enable_gpu: false,
            gpu_device_index: 0,
            numa_node: -1,
            cpu_affinity: Vec::new(),
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,
    
    /// Initial pool size in bytes
    pub initial_pool_size: usize,
    
    /// Maximum pool size in bytes
    pub max_pool_size: usize,
    
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    
    /// Prefetch distance (cache lines)
    pub prefetch_distance: usize,
    
    /// Enable huge pages
    pub enable_huge_pages: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            initial_pool_size: 1024 * 1024 * 10, // 10MB
            max_pool_size: 1024 * 1024 * 100, // 100MB
            enable_prefetch: true,
            prefetch_distance: 4,
            enable_huge_pages: false,
        }
    }
}

/// Configuration builder for fluent API
pub struct ConfigBuilder {
    config: CDFAConfig,
}

impl ConfigBuilder {
    /// Creates a new configuration builder
    pub fn new() -> Self {
        Self {
            config: CDFAConfig::default(),
        }
    }

    /// Creates a builder for ultra-low latency configuration
    pub fn ultra_low_latency() -> Self {
        let mut builder = Self::new();
        builder.config.performance.max_latency_ns = 100_000; // 100μs
        builder.config.performance.batch_size = 32;
        builder.config.performance.cache_warm_iterations = 10;
        builder.config.memory.enable_prefetch = true;
        builder.config.memory.prefetch_distance = 8;
        builder.config.memory.enable_huge_pages = true;
        builder
    }

    /// Creates a builder for high-throughput configuration
    pub fn high_throughput() -> Self {
        let mut builder = Self::new();
        builder.config.performance.batch_size = 256;
        builder.config.performance.enable_parallel = true;
        builder.config.analysis.max_signals = 10000;
        builder.config.memory.max_pool_size = 1024 * 1024 * 500; // 500MB
        builder
    }

    /// Sets the maximum latency target
    pub fn max_latency_ns(mut self, latency_ns: u64) -> Self {
        self.config.performance.max_latency_ns = latency_ns;
        self
    }

    /// Sets the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.performance.batch_size = size;
        self
    }

    /// Enables or disables parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.performance.enable_parallel = enable;
        self
    }

    /// Sets the number of worker threads
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.performance.worker_threads = threads;
        self
    }

    /// Enables or disables SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.performance.enable_simd = enable;
        self
    }

    /// Sets the fusion strategy
    pub fn fusion_strategy(mut self, strategy: FusionStrategyType) -> Self {
        self.config.fusion.default_strategy = strategy;
        self
    }

    /// Sets the confidence threshold
    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        if threshold >= 0.0 && threshold <= 1.0 {
            self.config.analysis.confidence_threshold = threshold;
        }
        self
    }

    /// Enables GPU acceleration
    pub fn enable_gpu(mut self, device_index: usize) -> Self {
        self.config.hardware.enable_gpu = true;
        self.config.hardware.gpu_device_index = device_index;
        self
    }

    /// Sets NUMA node affinity
    pub fn numa_node(mut self, node: i32) -> Self {
        self.config.hardware.numa_node = node;
        self
    }

    /// Sets CPU core affinity
    pub fn cpu_affinity(mut self, cores: Vec<usize>) -> Self {
        self.config.hardware.cpu_affinity = cores;
        self
    }

    /// Builds and validates the configuration
    pub fn build(self) -> Result<CDFAConfig> {
        self.validate()?;
        Ok(self.config)
    }

    /// Validates the configuration
    fn validate(&self) -> Result<()> {
        // Validate performance settings
        if self.config.performance.batch_size == 0 {
            return Err(Error::config("Batch size must be greater than 0"));
        }

        // Validate analysis settings
        if self.config.analysis.min_signals > self.config.analysis.max_signals {
            return Err(Error::config("min_signals cannot be greater than max_signals"));
        }

        if self.config.analysis.window_overlap < 0.0 || self.config.analysis.window_overlap > 1.0 {
            return Err(Error::config("window_overlap must be between 0.0 and 1.0"));
        }

        // Validate fusion settings
        if self.config.fusion.weight_learning_rate <= 0.0 {
            return Err(Error::config("weight_learning_rate must be positive"));
        }

        if self.config.fusion.min_weight > self.config.fusion.max_weight {
            return Err(Error::config("min_weight cannot be greater than max_weight"));
        }

        // Validate memory settings
        if self.config.memory.initial_pool_size > self.config.memory.max_pool_size {
            return Err(Error::config("initial_pool_size cannot be greater than max_pool_size"));
        }

        Ok(())
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CDFAConfig::default();
        assert_eq!(config.performance.max_latency_ns, 1_000_000);
        assert!(config.performance.enable_parallel);
        assert_eq!(config.performance.batch_size, 64);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .max_latency_ns(500_000)
            .batch_size(128)
            .parallel(false)
            .build()
            .unwrap();

        assert_eq!(config.performance.max_latency_ns, 500_000);
        assert_eq!(config.performance.batch_size, 128);
        assert!(!config.performance.enable_parallel);
    }

    #[test]
    fn test_ultra_low_latency_config() {
        let config = ConfigBuilder::ultra_low_latency().build().unwrap();
        assert_eq!(config.performance.max_latency_ns, 100_000);
        assert!(config.memory.enable_huge_pages);
    }

    #[test]
    fn test_invalid_config() {
        let result = ConfigBuilder::new()
            .batch_size(0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_confidence_threshold() {
        let config = ConfigBuilder::new()
            .confidence_threshold(0.85)
            .build()
            .unwrap();
        assert_eq!(config.analysis.confidence_threshold, 0.85);
    }
}