//! Configuration for ATS-Core mathematical operations
//!
//! This module provides comprehensive configuration structures for all ATS-CP components,
//! with a focus on performance tuning and real-time operation requirements.

use crate::error::{AtsCoreError, Result};
use serde::{Deserialize, Serialize};

/// Main configuration structure for ATS-CP engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtsCpConfig {
    /// Temperature scaling configuration
    pub temperature: TemperatureConfig,
    
    /// Conformal prediction configuration
    pub conformal: ConformalConfig,
    
    /// SIMD optimization configuration
    pub simd: SimdConfig,
    
    /// Memory management configuration
    pub memory: MemoryConfig,
    
    /// Parallel processing configuration
    pub parallel: ParallelConfig,
    
    /// Performance monitoring configuration
    pub performance: PerformanceConfig,
    
    /// ruv-FANN integration configuration
    pub integration: IntegrationConfig,
}

/// Temperature scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureConfig {
    /// Default temperature value
    pub default_temperature: f64,
    
    /// Minimum allowed temperature
    pub min_temperature: f64,
    
    /// Maximum allowed temperature
    pub max_temperature: f64,
    
    /// Binary search tolerance for temperature optimization
    pub search_tolerance: f64,
    
    /// Maximum iterations for binary search
    pub max_search_iterations: usize,
    
    /// Target latency for temperature scaling operations (microseconds)
    pub target_latency_us: u64,
    
    /// Enable adaptive temperature adjustment
    pub adaptive_enabled: bool,
    
    /// Learning rate for adaptive temperature
    pub adaptive_learning_rate: f64,
}

/// Conformal prediction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalConfig {
    /// Default confidence level (e.g., 0.95 for 95% confidence)
    pub default_confidence: f64,
    
    /// Minimum calibration data size
    pub min_calibration_size: usize,
    
    /// Maximum calibration data size (for performance)
    pub max_calibration_size: usize,
    
    /// Target latency for conformal prediction (microseconds)
    pub target_latency_us: u64,
    
    /// Enable exchangeability assumption validation
    pub validate_exchangeability: bool,
    
    /// Quantile estimation method
    pub quantile_method: QuantileMethod,
    
    /// Enable online calibration updates
    pub online_calibration: bool,
    
    /// Calibration update window size
    pub calibration_window_size: usize,
}

/// SIMD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    /// Enable SIMD operations
    pub enabled: bool,
    
    /// SIMD instruction set to use
    pub instruction_set: SimdInstructionSet,
    
    /// Vector width for SIMD operations
    pub vector_width: usize,
    
    /// Minimum array size to use SIMD
    pub min_simd_size: usize,
    
    /// Enable auto-vectorization hints
    pub auto_vectorization: bool,
    
    /// Alignment requirement for SIMD operations
    pub alignment_bytes: usize,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Default memory alignment (bytes)
    pub default_alignment: usize,
    
    /// Memory pool size for frequent allocations
    pub pool_size_mb: usize,
    
    /// Enable memory-mapped arrays for large data
    pub enable_mmap: bool,
    
    /// Maximum size for stack allocation
    pub max_stack_size: usize,
    
    /// Enable zero-copy optimizations
    pub zero_copy_enabled: bool,
    
    /// Memory prefetch distance
    pub prefetch_distance: usize,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads (0 = auto-detect)
    pub num_threads: usize,
    
    /// Minimum work size per thread
    pub min_work_per_thread: usize,
    
    /// Enable work stealing
    pub work_stealing: bool,
    
    /// Thread pool creation strategy
    pub thread_pool_strategy: ThreadPoolStrategy,
    
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinityConfig,
    
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    
    /// Monitoring granularity level
    pub granularity: MonitoringGranularity,
    
    /// Maximum latency before warning (microseconds)
    pub max_latency_us: u64,
    
    /// Enable latency histograms
    pub latency_histograms: bool,
    
    /// Sample rate for performance metrics (0.0 to 1.0)
    pub sample_rate: f64,
    
    /// Enable real-time performance alerts
    pub real_time_alerts: bool,
    
    /// Performance log file path
    pub log_file_path: Option<String>,
}

/// ruv-FANN integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable ruv-FANN integration
    pub enabled: bool,
    
    /// Network model path
    pub model_path: String,
    
    /// Batch size for neural network inference
    pub batch_size: usize,
    
    /// Number of inference threads
    pub inference_threads: usize,
    
    /// Enable GPU acceleration
    pub gpu_acceleration: bool,
    
    /// GPU device ID
    pub gpu_device_id: i32,
    
    /// Enable neural network caching
    pub enable_caching: bool,
    
    /// Cache size for neural network results
    pub cache_size: usize,
}

/// Quantile estimation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantileMethod {
    /// Linear interpolation
    Linear,
    /// Nearest neighbor
    Nearest,
    /// Higher value
    Higher,
    /// Lower value
    Lower,
    /// Midpoint
    Midpoint,
}

/// SIMD instruction sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    /// Auto-detect best available
    Auto,
    /// AVX2 instruction set
    Avx2,
    /// AVX-512 instruction set
    Avx512,
    /// SSE4.2 instruction set
    Sse42,
    /// ARM NEON instruction set
    Neon,
}

/// Thread pool strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadPoolStrategy {
    /// Global thread pool
    Global,
    /// Per-operation thread pool
    PerOperation,
    /// Adaptive thread pool
    Adaptive,
}

/// Thread affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// Enable thread affinity
    pub enabled: bool,
    
    /// CPU cores to bind threads to
    pub core_ids: Vec<usize>,
    
    /// Enable hyper-threading awareness
    pub hyper_threading_aware: bool,
}

/// Performance monitoring granularity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitoringGranularity {
    /// Coarse-grained monitoring
    Coarse,
    /// Fine-grained monitoring
    Fine,
    /// Ultra-fine monitoring (maximum detail)
    UltraFine,
}

impl Default for AtsCpConfig {
    fn default() -> Self {
        Self {
            temperature: TemperatureConfig::default(),
            conformal: ConformalConfig::default(),
            simd: SimdConfig::default(),
            memory: MemoryConfig::default(),
            parallel: ParallelConfig::default(),
            performance: PerformanceConfig::default(),
            integration: IntegrationConfig::default(),
        }
    }
}

impl Default for TemperatureConfig {
    fn default() -> Self {
        Self {
            default_temperature: 1.0,
            min_temperature: 0.1,
            max_temperature: 10.0,
            search_tolerance: 1e-6,
            max_search_iterations: 32,
            target_latency_us: 5, // Sub-5μs target
            adaptive_enabled: true,
            adaptive_learning_rate: 0.01,
        }
    }
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            default_confidence: 0.95,
            min_calibration_size: 100,
            max_calibration_size: 10000,
            target_latency_us: 20, // Sub-20μs target
            validate_exchangeability: true,
            quantile_method: QuantileMethod::Linear,
            online_calibration: true,
            calibration_window_size: 1000,
        }
    }
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instruction_set: SimdInstructionSet::Auto,
            vector_width: 8, // 8 x f64 for AVX-512
            min_simd_size: 32,
            auto_vectorization: true,
            alignment_bytes: 64, // Cache line alignment
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            default_alignment: 64,
            pool_size_mb: 128,
            enable_mmap: true,
            max_stack_size: 8192,
            zero_copy_enabled: true,
            prefetch_distance: 64,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            min_work_per_thread: 1000,
            work_stealing: true,
            thread_pool_strategy: ThreadPoolStrategy::Global,
            thread_affinity: ThreadAffinityConfig::default(),
            numa_aware: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            granularity: MonitoringGranularity::Fine,
            max_latency_us: 100, // Sub-100μs system target
            latency_histograms: true,
            sample_rate: 1.0,
            real_time_alerts: true,
            log_file_path: None,
        }
    }
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: "models/nhits_v1.bin".to_string(),
            batch_size: 32,
            inference_threads: 4,
            gpu_acceleration: true,
            gpu_device_id: 0,
            enable_caching: true,
            cache_size: 1000,
        }
    }
}

impl Default for ThreadAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            core_ids: Vec::new(),
            hyper_threading_aware: true,
        }
    }
}

impl AtsCpConfig {
    /// Validates the configuration and returns errors if invalid
    pub fn validate(&self) -> Result<()> {
        // Validate temperature configuration
        if self.temperature.default_temperature < self.temperature.min_temperature
            || self.temperature.default_temperature > self.temperature.max_temperature
        {
            return Err(AtsCoreError::validation(
                "temperature.default_temperature",
                "must be within min/max range",
            ));
        }

        if self.temperature.search_tolerance <= 0.0 {
            return Err(AtsCoreError::validation(
                "temperature.search_tolerance",
                "must be positive",
            ));
        }

        if self.temperature.max_search_iterations == 0 {
            return Err(AtsCoreError::validation(
                "temperature.max_search_iterations",
                "must be greater than 0",
            ));
        }

        // Validate conformal configuration
        if self.conformal.default_confidence <= 0.0 || self.conformal.default_confidence >= 1.0 {
            return Err(AtsCoreError::validation(
                "conformal.default_confidence",
                "must be between 0 and 1",
            ));
        }

        if self.conformal.min_calibration_size > self.conformal.max_calibration_size {
            return Err(AtsCoreError::validation(
                "conformal.min_calibration_size",
                "must be less than or equal to max_calibration_size",
            ));
        }

        // Validate SIMD configuration
        if self.simd.vector_width == 0 {
            return Err(AtsCoreError::validation(
                "simd.vector_width",
                "must be greater than 0",
            ));
        }

        if self.simd.alignment_bytes == 0 || !self.simd.alignment_bytes.is_power_of_two() {
            return Err(AtsCoreError::validation(
                "simd.alignment_bytes",
                "must be a power of 2",
            ));
        }

        // Validate memory configuration
        if self.memory.default_alignment == 0 || !self.memory.default_alignment.is_power_of_two() {
            return Err(AtsCoreError::validation(
                "memory.default_alignment",
                "must be a power of 2",
            ));
        }

        // Validate performance configuration
        if self.performance.sample_rate < 0.0 || self.performance.sample_rate > 1.0 {
            return Err(AtsCoreError::validation(
                "performance.sample_rate",
                "must be between 0.0 and 1.0",
            ));
        }

        Ok(())
    }

    /// Creates a high-performance configuration optimized for sub-100μs latency
    pub fn high_performance() -> Self {
        Self {
            temperature: TemperatureConfig {
                target_latency_us: 3, // Ultra-low latency
                search_tolerance: 1e-4, // Relaxed tolerance for speed
                max_search_iterations: 16, // Reduced iterations
                ..Default::default()
            },
            conformal: ConformalConfig {
                target_latency_us: 15, // Ultra-low latency
                max_calibration_size: 5000, // Reduced for speed
                ..Default::default()
            },
            simd: SimdConfig {
                enabled: true,
                instruction_set: SimdInstructionSet::Avx512,
                vector_width: 8,
                min_simd_size: 16, // Lower threshold
                ..Default::default()
            },
            memory: MemoryConfig {
                pool_size_mb: 256, // Larger pool
                zero_copy_enabled: true,
                prefetch_distance: 128, // Aggressive prefetch
                ..Default::default()
            },
            parallel: ParallelConfig {
                work_stealing: true,
                thread_affinity: ThreadAffinityConfig {
                    enabled: true,
                    core_ids: vec![0, 1, 2, 3], // Dedicated cores
                    hyper_threading_aware: true,
                },
                numa_aware: true,
                ..Default::default()
            },
            performance: PerformanceConfig {
                granularity: MonitoringGranularity::UltraFine,
                max_latency_us: 50, // Strict latency requirements
                sample_rate: 0.1, // Reduced monitoring overhead
                ..Default::default()
            },
            integration: IntegrationConfig {
                gpu_acceleration: true,
                enable_caching: true,
                cache_size: 2000, // Larger cache
                ..Default::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AtsCpConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_performance_config() {
        let config = AtsCpConfig::high_performance();
        assert!(config.validate().is_ok());
        assert_eq!(config.temperature.target_latency_us, 3);
        assert_eq!(config.conformal.target_latency_us, 15);
        assert!(config.simd.enabled);
        assert!(config.memory.zero_copy_enabled);
        assert!(config.parallel.thread_affinity.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AtsCpConfig::default();
        
        // Test invalid temperature
        config.temperature.default_temperature = -1.0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid confidence
        config = AtsCpConfig::default();
        config.conformal.default_confidence = 1.5;
        assert!(config.validate().is_err());
        
        // Reset and test invalid alignment
        config = AtsCpConfig::default();
        config.memory.default_alignment = 15; // Not power of 2
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serialization() {
        let config = AtsCpConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AtsCpConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.validate().is_ok());
    }
}