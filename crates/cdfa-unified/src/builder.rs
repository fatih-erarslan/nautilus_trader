//! Builder pattern for configuring UnifiedCdfa instances
//!
//! This module provides a fluent API for constructing UnifiedCdfa instances
//! with custom configurations, hardware backends, and component registrations.

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::unified::{UnifiedCdfa, ComponentRegistry};
use crate::prelude::{DiversityMethod, FusionMethod, PatternDetector, SystemAnalyzer};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Builder for constructing UnifiedCdfa instances with custom configuration
/// 
/// The builder pattern allows for fluent, readable configuration of all CDFA
/// components and backends. It validates configuration consistency and
/// initializes only the requested features.
/// 
/// # Example
/// 
/// ```rust
/// use cdfa_unified::UnifiedCdfaBuilder;
/// 
/// let cdfa = UnifiedCdfaBuilder::new()
///     .with_simd(true)
///     .with_parallel_threads(8)
///     .with_gpu(false)
///     .with_cache_size_mb(256)
///     .with_diversity_method("kendall")
///     .with_fusion_method("weighted_average")
///     .enable_detector("fibonacci")
///     .enable_analyzer("antifragility")
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct UnifiedCdfaBuilder {
    config: CdfaConfig,
    registry: ComponentRegistry,
    
    // Hardware backend configuration
    enable_simd: bool,
    enable_parallel: bool,
    enable_gpu: bool,
    enable_ml: bool,
    
    // Component selection
    enabled_detectors: Vec<String>,
    enabled_analyzers: Vec<String>,
    enabled_algorithms: Vec<String>,
    
    // Custom components
    custom_diversity_methods: HashMap<String, Box<dyn DiversityMethod + Send + Sync>>,
    custom_fusion_methods: HashMap<String, Box<dyn FusionMethod + Send + Sync>>,
    custom_detectors: HashMap<String, Box<dyn PatternDetector + Send + Sync>>,
    custom_analyzers: HashMap<String, Box<dyn SystemAnalyzer + Send + Sync>>,
}

impl UnifiedCdfaBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: CdfaConfig::default(),
            registry: ComponentRegistry::new(),
            enable_simd: cfg!(feature = "simd"),
            enable_parallel: cfg!(feature = "parallel"),
            enable_gpu: cfg!(feature = "gpu"),
            enable_ml: cfg!(feature = "ml"),
            enabled_detectors: Vec::new(),
            enabled_analyzers: Vec::new(),
            enabled_algorithms: Vec::new(),
            custom_diversity_methods: HashMap::new(),
            custom_fusion_methods: HashMap::new(),
            custom_detectors: HashMap::new(),
            custom_analyzers: HashMap::new(),
        }
    }
    
    /// Configure number of parallel threads (0 = auto-detect)
    pub fn with_parallel_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = threads;
        self.enable_parallel = threads > 0;
        self
    }
    
    /// Enable or disable SIMD optimizations
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable && cfg!(feature = "simd");
        self.config.enable_simd = self.enable_simd;
        self
    }
    
    /// Enable or disable GPU acceleration
    pub fn with_gpu(mut self, enable: bool) -> Self {
        self.enable_gpu = enable && cfg!(feature = "gpu");
        self.config.enable_gpu = self.enable_gpu;
        self
    }
    
    /// Enable or disable machine learning features
    pub fn with_ml(mut self, enable: bool) -> Self {
        self.enable_ml = enable && cfg!(feature = "ml");
        self
    }
    
    /// Set numerical tolerance for comparisons
    pub fn with_tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }
    
    /// Set maximum iterations for iterative algorithms
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_iterations = max_iter;
        self
    }
    
    /// Set convergence threshold for iterative algorithms
    pub fn with_convergence_threshold(mut self, threshold: Float) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }
    
    /// Set cache size in megabytes (0 = disable caching)
    pub fn with_cache_size_mb(mut self, size_mb: usize) -> Self {
        self.config.cache_size_mb = size_mb;
        self
    }
    
    /// Enable distributed processing
    pub fn with_distributed(mut self, enable: bool) -> Self {
        self.config.enable_distributed = enable && cfg!(feature = "distributed");
        self
    }
    
    /// Set default diversity method
    pub fn with_diversity_method<S: Into<String>>(mut self, method: S) -> Self {
        self.config.diversity_method = Some(method.into());
        self
    }
    
    /// Set default fusion method
    pub fn with_fusion_method<S: Into<String>>(mut self, method: S) -> Self {
        self.config.fusion_method = Some(method.into());
        self
    }
    
    /// Enable a specific pattern detector
    pub fn enable_detector<S: Into<String>>(mut self, detector: S) -> Self {
        self.enabled_detectors.push(detector.into());
        self
    }
    
    /// Enable multiple pattern detectors
    pub fn enable_detectors<I, S>(mut self, detectors: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enabled_detectors.extend(detectors.into_iter().map(|s| s.into()));
        self
    }
    
    /// Enable a specific analyzer
    pub fn enable_analyzer<S: Into<String>>(mut self, analyzer: S) -> Self {
        self.enabled_analyzers.push(analyzer.into());
        self
    }
    
    /// Enable multiple analyzers
    pub fn enable_analyzers<I, S>(mut self, analyzers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enabled_analyzers.extend(analyzers.into_iter().map(|s| s.into()));
        self
    }
    
    /// Enable a specific algorithm
    pub fn enable_algorithm<S: Into<String>>(mut self, algorithm: S) -> Self {
        self.enabled_algorithms.push(algorithm.into());
        self
    }
    
    /// Register a custom diversity method
    pub fn with_custom_diversity_method<S, T>(mut self, name: S, method: T) -> Self
    where
        S: Into<String>,
        T: DiversityMethod + Send + Sync + 'static,
    {
        self.custom_diversity_methods.insert(name.into(), Box::new(method));
        self
    }
    
    /// Register a custom fusion method
    pub fn with_custom_fusion_method<S, T>(mut self, name: S, method: T) -> Self
    where
        S: Into<String>,
        T: FusionMethod + Send + Sync + 'static,
    {
        self.custom_fusion_methods.insert(name.into(), Box::new(method));
        self
    }
    
    /// Register a custom pattern detector
    pub fn with_custom_detector<S, T>(mut self, name: S, detector: T) -> Self
    where
        S: Into<String>,
        T: PatternDetector + Send + Sync + 'static,
    {
        self.custom_detectors.insert(name.into(), Box::new(detector));
        self
    }
    
    /// Register a custom analyzer
    pub fn with_custom_analyzer<S, T>(mut self, name: S, analyzer: T) -> Self
    where
        S: Into<String>,
        T: SystemAnalyzer + Send + Sync + 'static,
    {
        self.custom_analyzers.insert(name.into(), Box::new(analyzer));
        self
    }
    
    /// Set a custom configuration
    pub fn with_config(mut self, config: CdfaConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Build the configured UnifiedCdfa instance
    pub fn build(mut self) -> Result<UnifiedCdfa> {
        // Validate configuration
        self.validate_configuration()?;
        
        // Apply enabled components filter
        self.apply_component_filters();
        
        // Register custom components
        self.register_custom_components();
        
        // Initialize hardware backends
        #[cfg(feature = "simd")]
        let simd_backend = if self.enable_simd {
            Some(crate::simd::SimdBackend::new()?)
        } else {
            None
        };
        
        #[cfg(feature = "parallel")]
        let parallel_backend = if self.enable_parallel {
            Some(crate::parallel::ParallelBackend::new(self.config.num_threads)?)
        } else {
            None
        };
        
        #[cfg(feature = "gpu")]
        let gpu_backend = if self.enable_gpu {
            Some(crate::gpu::GpuBackend::new()?)
        } else {
            None
        };
        
        #[cfg(feature = "ml")]
        let ml_backend = if self.enable_ml {
            Some(crate::ml::MLBackend::new()?)
        } else {
            None
        };
        
        // Create the UnifiedCdfa instance
        Ok(UnifiedCdfa {
            config: Arc::new(RwLock::new(self.config)),
            registry: Arc::new(self.registry),
            
            #[cfg(feature = "simd")]
            simd_backend,
            
            #[cfg(feature = "parallel")]
            parallel_backend,
            
            #[cfg(feature = "gpu")]
            gpu_backend,
            
            #[cfg(feature = "ml")]
            ml_backend,
            
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    // Private helper methods
    
    fn validate_configuration(&self) -> Result<()> {
        if self.config.tolerance <= 0.0 {
            return Err(CdfaError::config_error("Tolerance must be positive"));
        }
        
        if self.config.max_iterations == 0 {
            return Err(CdfaError::config_error("Max iterations must be positive"));
        }
        
        if self.config.convergence_threshold <= 0.0 {
            return Err(CdfaError::config_error("Convergence threshold must be positive"));
        }
        
        // Validate diversity method exists
        if let Some(ref method) = self.config.diversity_method {
            if !self.registry.diversity_methods.contains_key(method) && 
               !self.custom_diversity_methods.contains_key(method) {
                return Err(CdfaError::config_error(format!(
                    "Unknown diversity method: {}", method
                )));
            }
        }
        
        // Validate fusion method exists
        if let Some(ref method) = self.config.fusion_method {
            if !self.registry.fusion_methods.contains_key(method) && 
               !self.custom_fusion_methods.contains_key(method) {
                return Err(CdfaError::config_error(format!(
                    "Unknown fusion method: {}", method
                )));
            }
        }
        
        // Validate enabled detectors exist
        for detector in &self.enabled_detectors {
            if !self.registry.detectors.contains_key(detector) && 
               !self.custom_detectors.contains_key(detector) {
                return Err(CdfaError::config_error(format!(
                    "Unknown detector: {}", detector
                )));
            }
        }
        
        // Validate enabled analyzers exist
        for analyzer in &self.enabled_analyzers {
            if !self.registry.analyzers.contains_key(analyzer) && 
               !self.custom_analyzers.contains_key(analyzer) {
                return Err(CdfaError::config_error(format!(
                    "Unknown analyzer: {}", analyzer
                )));
            }
        }
        
        Ok(())
    }
    
    fn apply_component_filters(&mut self) {
        // Filter enabled detectors
        if !self.enabled_detectors.is_empty() {
            self.config.enabled_detectors = Some(self.enabled_detectors.clone());
        }
        
        // Filter enabled analyzers
        if !self.enabled_analyzers.is_empty() {
            self.config.enabled_analyzers = Some(self.enabled_analyzers.clone());
        }
        
        // Filter enabled algorithms
        if !self.enabled_algorithms.is_empty() {
            self.config.enabled_algorithms = Some(self.enabled_algorithms.clone());
        }
    }
    
    fn register_custom_components(&mut self) {
        // Register custom diversity methods
        for (name, method) in self.custom_diversity_methods.drain() {
            self.registry.diversity_methods.insert(name, method);
        }
        
        // Register custom fusion methods
        for (name, method) in self.custom_fusion_methods.drain() {
            self.registry.fusion_methods.insert(name, method);
        }
        
        // Register custom detectors
        for (name, detector) in self.custom_detectors.drain() {
            self.registry.detectors.insert(name, detector);
        }
        
        // Register custom analyzers
        for (name, analyzer) in self.custom_analyzers.drain() {
            self.registry.analyzers.insert(name, analyzer);
        }
    }
}

impl Default for UnifiedCdfaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience methods for common configurations
impl UnifiedCdfaBuilder {
    /// Create a high-performance configuration with all optimizations enabled
    pub fn high_performance() -> Self {
        Self::new()
            .with_simd(true)
            .with_parallel_threads(0) // Auto-detect
            .with_gpu(true)
            .with_cache_size_mb(512)
            .with_tolerance(1e-12)
            .with_max_iterations(10000)
    }
    
    /// Create a configuration optimized for real-time processing
    pub fn real_time() -> Self {
        Self::new()
            .with_simd(true)
            .with_parallel_threads(4)
            .with_cache_size_mb(64)
            .with_tolerance(1e-6)
            .with_max_iterations(100)
            .enable_detectors(["fibonacci", "black_swan"])
    }
    
    /// Create a configuration for research and analysis
    pub fn research() -> Self {
        Self::new()
            .with_simd(true)
            .with_parallel_threads(0) // Auto-detect
            .with_ml(true)
            .with_cache_size_mb(1024)
            .with_tolerance(1e-15)
            .with_max_iterations(50000)
            .enable_detectors(["fibonacci", "black_swan"])
            .enable_analyzers(["antifragility", "panarchy"])
    }
    
    /// Create a minimal configuration for basic operations
    pub fn minimal() -> Self {
        Self::new()
            .with_simd(false)
            .with_parallel_threads(1)
            .with_gpu(false)
            .with_cache_size_mb(0)
            .with_diversity_method("pearson")
            .with_fusion_method("weighted_average")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builder_creation() {
        let builder = UnifiedCdfaBuilder::new();
        assert_eq!(builder.config.num_threads, 0);
        assert!(builder.config.enable_simd);
    }
    
    #[test]
    fn test_builder_configuration() {
        let builder = UnifiedCdfaBuilder::new()
            .with_parallel_threads(8)
            .with_tolerance(1e-9)
            .with_diversity_method("kendall")
            .enable_detector("fibonacci");
        
        assert_eq!(builder.config.num_threads, 8);
        assert_eq!(builder.config.tolerance, 1e-9);
        assert_eq!(builder.config.diversity_method, Some("kendall".to_string()));
        assert!(builder.enabled_detectors.contains(&"fibonacci".to_string()));
    }
    
    #[test]
    fn test_preset_configurations() {
        let high_perf = UnifiedCdfaBuilder::high_performance();
        assert!(high_perf.enable_simd);
        assert!(high_perf.enable_gpu);
        assert_eq!(high_perf.config.cache_size_mb, 512);
        
        let minimal = UnifiedCdfaBuilder::minimal();
        assert!(!minimal.enable_simd);
        assert_eq!(minimal.config.num_threads, 1);
        assert_eq!(minimal.config.cache_size_mb, 0);
    }
    
    #[test]
    fn test_validation() {
        let result = UnifiedCdfaBuilder::new()
            .with_tolerance(0.0) // Invalid
            .build();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Tolerance must be positive"));
    }
    
    #[test]
    fn test_custom_components() {
        use crate::traits::*;
        
        struct CustomDiversity;
        impl DiversityMethod for CustomDiversity {
            fn calculate(&self, _data: &FloatArrayView2) -> Result<FloatArray1> {
                Ok(FloatArray1::zeros(4))
            }
        }
        
        let builder = UnifiedCdfaBuilder::new()
            .with_custom_diversity_method("custom", CustomDiversity)
            .with_diversity_method("custom");
        
        assert!(builder.custom_diversity_methods.contains_key("custom"));
        assert_eq!(builder.config.diversity_method, Some("custom".to_string()));
    }
}
