//! Unified CDFA API - Main interface consolidating all functionality
//!
//! This module provides the `UnifiedCdfa` struct which serves as the main entry point
//! for all CDFA functionality, consolidating algorithms, detectors, analyzers, and
//! processing capabilities into a single, coherent interface.

use crate::error::{CdfaError, Result};
use crate::types::*;
use crate::traits::{SystemAnalyzer, DiversityMethod, FusionMethod, PatternDetector, SignalAlgorithm};
// Diversity methods available through core module
use crate::builder::UnifiedCdfaBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

#[cfg(feature = "parallel")]
// Parallel features available through feature flag

#[cfg(feature = "simd")]
// SIMD features available through feature flag

#[cfg(feature = "gpu")]
use crate::gpu::GpuBackend;

#[cfg(feature = "ml")]
use crate::ml::MLBackend;

/// Main unified CDFA interface consolidating all functionality
/// 
/// This struct provides a comprehensive API that combines:
/// - Core diversity and fusion algorithms
/// - Signal processing algorithms (wavelets, entropy, etc.)
/// - Pattern detectors (Fibonacci, Black Swan, etc.)
/// - Specialized analyzers (Antifragility, Panarchy, etc.)
/// - Hardware acceleration (SIMD, GPU)
/// - Parallel processing
/// - Machine learning integration
/// 
/// # Example
/// 
/// ```rust
/// use cdfa_unified::UnifiedCdfa;
/// use ndarray::array;
/// 
/// let cdfa = UnifiedCdfa::builder()
///     .with_simd(true)
///     .with_parallel_threads(4)
///     .build()?;
/// 
/// let data = array![[1.0, 2.0, 3.0], [1.1, 2.1, 2.9]];
/// let result = cdfa.analyze(&data)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct UnifiedCdfa {
    /// Configuration for all CDFA operations
    config: Arc<RwLock<CdfaConfig>>,
    
    /// Registry of available analyzers and detectors
    registry: Arc<ComponentRegistry>,
    
    /// Hardware acceleration backends
    #[cfg(feature = "simd")]
    #[cfg(feature = "simd")]
    simd_backend: Option<crate::simd::SimdBackend>,
    
    #[cfg(feature = "parallel")]
    parallel_backend: Option<crate::parallel::ParallelBackend>,
    
    #[cfg(feature = "gpu")]
    gpu_backend: Option<crate::gpu::GpuBackend>,
    
    #[cfg(feature = "ml")]
    ml_backend: Option<MLBackend>,
    
    /// Performance monitoring
    metrics: Arc<RwLock<PerformanceMetrics>>,
    
    /// Cache for frequently used computations
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
}

/// Component registry for managing analyzers and detectors
pub struct ComponentRegistry {
    /// Core diversity methods
    pub diversity_methods: HashMap<String, Box<dyn DiversityMethod + Send + Sync>>,
    
    /// Fusion algorithms
    pub fusion_methods: HashMap<String, Box<dyn FusionMethod + Send + Sync>>,
    
    /// Pattern detectors
    pub detectors: HashMap<String, Box<dyn PatternDetector + Send + Sync>>,
    
    /// Specialized analyzers
    pub analyzers: HashMap<String, Box<dyn SystemAnalyzer + Send + Sync>>,
    
    /// Signal processing algorithms
    pub algorithms: HashMap<String, Box<dyn SignalAlgorithm + Send + Sync>>,
}

/// Cached computation result
#[derive(Debug, Clone)]
struct CachedResult {
    data: FloatArray1,
    timestamp: i64,
    ttl_seconds: u64,
}

impl UnifiedCdfa {
    /// Create a new builder for configuring UnifiedCdfa
    pub fn builder() -> UnifiedCdfaBuilder {
        UnifiedCdfaBuilder::new()
    }
    
    /// Create a new UnifiedCdfa with default configuration
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }
    
    /// Perform comprehensive CDFA analysis on the provided data
    /// 
    /// This is the main analysis method that applies the full CDFA pipeline:
    /// 1. Data validation and preprocessing
    /// 2. Diversity metric calculation
    /// 3. Fusion algorithm application
    /// 4. Pattern detection
    /// 5. Specialized analysis (if configured)
    /// 
    /// # Arguments
    /// 
    /// * `data` - Input data matrix (rows = observations, cols = features)
    /// 
    /// # Returns
    /// 
    /// Comprehensive analysis result containing all computed metrics and detected patterns
    pub fn analyze(&self, data: &FloatArrayView2) -> Result<AnalysisResult> {
        let start_time = std::time::Instant::now();
        
        // Validate input data
        self.validate_input_data(data)?;
        
        // Create analysis context
        let mut result = AnalysisResult::new(
            FloatArray1::zeros(data.ncols()),
            self.config.read().clone(),
        );
        
        // Step 1: Calculate diversity metrics
        let diversity_scores = self.calculate_diversity(data)?;
        result.add_secondary_data("diversity_scores".to_string(), diversity_scores.clone());
        
        // Step 2: Apply fusion algorithms
        let fused_scores = self.apply_fusion(&diversity_scores, data)?;
        result.data = fused_scores;
        
        // Step 3: Detect patterns
        let patterns = self.detect_patterns(data, &result.data)?;
        for pattern in patterns {
            result.add_pattern(pattern);
        }
        
        // Step 4: Run specialized analyzers
        let analysis_metrics = self.run_analyzers(data, &result.data)?;
        for (name, value) in analysis_metrics {
            result.add_metric(name, value);
        }
        
        // Update performance metrics
        result.performance.execution_time_us = start_time.elapsed().as_micros() as u64;
        self.update_performance_metrics(&result.performance)?;
        
        Ok(result)
    }
    
    /// Calculate diversity metrics for the input data
    pub fn calculate_diversity(&self, data: &FloatArrayView2) -> Result<FloatArray1> {
        let registry = &self.registry;
        let config = self.config.read();
        
        // Check cache first
        let cache_key = format!("diversity_{}", self.hash_data(data));
        if let Some(cached) = self.get_cached_result(&cache_key)? {
            return Ok(cached);
        }
        
        // Select diversity method based on configuration
        let method_name = config.diversity_method.clone().unwrap_or_else(|| "kendall".to_string());
        let method = registry.diversity_methods.get(&method_name)
            .ok_or_else(|| CdfaError::config_error(format!("Unknown diversity method: {}", method_name)))?;
        
        // Apply hardware acceleration if available
        #[cfg(feature = "simd")]
        if let Some(ref simd) = self.simd_backend {
            if simd.supports_diversity() {
                let result = simd.calculate_diversity(data, &**method)?;
                self.cache_result(cache_key, result.clone())?;
                return Ok(result);
            }
        }
        
        // Fallback to standard implementation
        let result = method.calculate(data)?;
        self.cache_result(cache_key, result.clone())?;
        Ok(result)
    }
    
    /// Apply fusion algorithms to combine multiple scores
    pub fn apply_fusion(&self, scores: &FloatArrayView1, data: &FloatArrayView2) -> Result<FloatArray1> {
        let registry = &self.registry;
        let config = self.config.read();
        
        let method_name = config.fusion_method.clone().unwrap_or_else(|| "weighted_average".to_string());
        let method = registry.fusion_methods.get(&method_name)
            .ok_or_else(|| CdfaError::config_error(format!("Unknown fusion method: {}", method_name)))?;
        
        // Apply parallel processing if available
        #[cfg(feature = "parallel")]
        if let Some(ref parallel) = self.parallel_backend {
            if config.num_threads > 1 {
                return parallel.apply_fusion(scores, data, &**method);
            }
        }
        
        method.fuse(scores, data)
    }
    
    /// Detect patterns in the data using registered detectors
    pub fn detect_patterns(&self, data: &FloatArrayView2, scores: &FloatArrayView1) -> Result<Vec<Pattern>> {
        let registry = &self.registry;
        let config = self.config.read();
        let mut patterns = Vec::new();
        
        // Run enabled detectors
        for (name, detector) in &registry.detectors {
            if config.enabled_detectors.as_ref().map_or(true, |enabled| enabled.contains(name)) {
                let detected = detector.detect(data, scores)?;
                patterns.extend(detected);
            }
        }
        
        Ok(patterns)
    }
    
    /// Run specialized analyzers on the data
    pub fn run_analyzers(&self, data: &FloatArrayView2, scores: &FloatArrayView1) -> Result<HashMap<String, Float>> {
        let registry = &self.registry;
        let config = self.config.read();
        let mut metrics = HashMap::new();
        
        // Run enabled analyzers
        for (name, analyzer) in &registry.analyzers {
            if config.enabled_analyzers.as_ref().map_or(true, |enabled| enabled.contains(name)) {
                let result = analyzer.analyze(data, scores)?;
                for (metric_name, value) in result {
                    metrics.insert(format!("{}_{}", name, metric_name), value);
                }
            }
        }
        
        Ok(metrics)
    }
    
    /// Get current configuration
    pub fn config(&self) -> CdfaConfig {
        self.config.read().clone()
    }
    
    /// Update configuration
    pub fn update_config<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut CdfaConfig),
    {
        let mut config = self.config.write();
        updater(&mut config);
        self.validate_config(&config)?;
        Ok(())
    }
    
    /// Get performance metrics
    pub fn performance_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().clone()
    }
    
    /// Clear all caches
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
    
    // Private helper methods
    
    fn validate_input_data(&self, data: &FloatArrayView2) -> Result<()> {
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Input data cannot be empty"));
        }
        
        if data.nrows() < 2 {
            return Err(CdfaError::invalid_input("Input data must have at least 2 rows"));
        }
        
        if data.ncols() < 2 {
            return Err(CdfaError::invalid_input("Input data must have at least 2 columns"));
        }
        
        // Check for NaN or infinite values
        for value in data.iter() {
            if !value.is_finite() {
                return Err(CdfaError::invalid_input("Input data contains NaN or infinite values"));
            }
        }
        
        Ok(())
    }
    
    fn validate_config(&self, config: &CdfaConfig) -> Result<()> {
        if config.tolerance <= 0.0 {
            return Err(CdfaError::config_error("Tolerance must be positive"));
        }
        
        if config.max_iterations == 0 {
            return Err(CdfaError::config_error("Max iterations must be positive"));
        }
        
        Ok(())
    }
    
    fn hash_data(&self, data: &FloatArrayView2) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.shape().hash(&mut hasher);
        
        // Sample a few values for hashing to avoid performance issues
        let sample_size = std::cmp::min(100, data.len());
        for i in (0..data.len()).step_by(data.len() / sample_size + 1) {
            data.as_slice().unwrap()[i].to_bits().hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    fn get_cached_result(&self, key: &str) -> Result<Option<FloatArray1>> {
        let cache = self.cache.read();
        if let Some(cached) = cache.get(key) {
            let now = chrono::Utc::now().timestamp();
            if (now - cached.timestamp) < cached.ttl_seconds as i64 {
                return Ok(Some(cached.data.clone()));
            }
        }
        Ok(None)
    }
    
    fn cache_result(&self, key: String, data: FloatArray1) -> Result<()> {
        let config = self.config.read();
        if config.cache_size_mb > 0 {
            let cached = CachedResult {
                data,
                timestamp: chrono::Utc::now().timestamp(),
                ttl_seconds: 300, // 5 minutes default TTL
            };
            self.cache.write().insert(key, cached);
        }
        Ok(())
    }
    
    fn update_performance_metrics(&self, new_metrics: &PerformanceMetrics) -> Result<()> {
        let mut metrics = self.metrics.write();
        
        // Update cumulative metrics
        metrics.execution_time_us = 
            (metrics.execution_time_us + new_metrics.execution_time_us) / 2; // Running average
        metrics.memory_used_bytes = 
            std::cmp::max(metrics.memory_used_bytes, new_metrics.memory_used_bytes);
        metrics.simd_operations += new_metrics.simd_operations;
        metrics.parallel_tasks += new_metrics.parallel_tasks;
        
        Ok(())
    }
}

impl Default for UnifiedCdfa {
    fn default() -> Self {
        Self::new().expect("Failed to create default UnifiedCdfa")
    }
}

impl ComponentRegistry {
    /// Create a new component registry with default components
    pub fn new() -> Self {
        let mut registry = Self {
            diversity_methods: HashMap::new(),
            fusion_methods: HashMap::new(),
            detectors: HashMap::new(),
            analyzers: HashMap::new(),
            algorithms: HashMap::new(),
        };
        
        registry.register_default_components();
        registry
    }
    
    /// Register default components
    fn register_default_components(&mut self) {
        // Register core diversity methods
        #[cfg(feature = "core")]
        {
            use crate::core::diversity::*;
            self.diversity_methods.insert("kendall".to_string(), Box::new(KendallTauDiversity));
            self.diversity_methods.insert("spearman".to_string(), Box::new(SpearmanDiversity));
            self.diversity_methods.insert("pearson".to_string(), Box::new(PearsonDiversity));
        }
        
        // Register fusion methods
        #[cfg(feature = "core")]
        {
            use crate::core::fusion::*;
            self.fusion_methods.insert("weighted_average".to_string(), Box::new(WeightedAverageFusion));
            self.fusion_methods.insert("rank_fusion".to_string(), Box::new(RankFusion));
        }
        
        // Register pattern detectors
        #[cfg(feature = "detectors")]
        {
            use crate::detectors::*;
            self.detectors.insert("fibonacci".to_string(), Box::new(FibonacciPatternDetector::new()));
            self.detectors.insert("black_swan".to_string(), Box::new(BlackSwanDetector::new()));
        }
        
        // Register analyzers
        #[cfg(feature = "detectors")]
        {
            use crate::detectors::*;
            self.analyzers.insert("antifragility".to_string(), Box::new(AntifragilityAnalyzer::new()));
            self.analyzers.insert("panarchy".to_string(), Box::new(PanarchyAnalyzer::new()));
        }
    }
    
    /// Register a custom diversity method
    pub fn register_diversity_method<T>(&mut self, name: String, method: T)
    where
        T: DiversityMethod + Send + Sync + 'static,
    {
        self.diversity_methods.insert(name, Box::new(method));
    }
    
    /// Register a custom fusion method
    pub fn register_fusion_method<T>(&mut self, name: String, method: T)
    where
        T: FusionMethod + Send + Sync + 'static,
    {
        self.fusion_methods.insert(name, Box::new(method));
    }
    
    /// Register a custom pattern detector
    pub fn register_detector<T>(&mut self, name: String, detector: T)
    where
        T: PatternDetector + Send + Sync + 'static,
    {
        self.detectors.insert(name, Box::new(detector));
    }
    
    /// Register a custom analyzer
    pub fn register_analyzer<T>(&mut self, name: String, analyzer: T)
    where
        T: SystemAnalyzer + Send + Sync + 'static,
    {
        self.analyzers.insert(name, Box::new(analyzer));
    }
    
    /// List all registered components
    pub fn list_components(&self) -> HashMap<String, Vec<String>> {
        let mut components = HashMap::new();
        
        components.insert(
            "diversity_methods".to_string(),
            self.diversity_methods.keys().cloned().collect(),
        );
        components.insert(
            "fusion_methods".to_string(),
            self.fusion_methods.keys().cloned().collect(),
        );
        components.insert(
            "detectors".to_string(),
            self.detectors.keys().cloned().collect(),
        );
        components.insert(
            "analyzers".to_string(),
            self.analyzers.keys().cloned().collect(),
        );
        components.insert(
            "algorithms".to_string(),
            self.algorithms.keys().cloned().collect(),
        );
        
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_unified_cdfa_creation() {
        let cdfa = UnifiedCdfa::new();
        assert!(cdfa.is_ok());
    }
    
    #[test]
    fn test_component_registry() {
        let registry = ComponentRegistry::new();
        let components = registry.list_components();
        
        assert!(!components.is_empty());
        assert!(components.contains_key("diversity_methods"));
        assert!(components.contains_key("fusion_methods"));
    }
    
    #[cfg(feature = "core")]
    #[test]
    fn test_basic_analysis() {
        let cdfa = UnifiedCdfa::new().unwrap();
        let data = array![
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 2.9, 4.1],
            [0.9, 1.9, 3.1, 3.9]
        ];
        
        let result = cdfa.analyze(&data.view());
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.data.len(), 4);
    }
    
    #[test]
    fn test_config_update() {
        let cdfa = UnifiedCdfa::new().unwrap();
        
        let initial_config = cdfa.config();
        assert_eq!(initial_config.num_threads, 0);
        
        cdfa.update_config(|config| {
            config.num_threads = 4;
        }).unwrap();
        
        let updated_config = cdfa.config();
        assert_eq!(updated_config.num_threads, 4);
    }
}
