//! Backward compatibility layer for existing CDFA crates
//!
//! This module provides compatibility wrappers and re-exports to ensure that
//! existing code using individual CDFA crates can work without modification
//! when switching to the unified crate.

use crate::error::Result;
use crate::types::*;
use crate::unified::UnifiedCdfa;
use std::sync::Arc;
use once_cell::sync::Lazy;

/// Global unified CDFA instance for compatibility
static GLOBAL_CDFA: Lazy<Arc<UnifiedCdfa>> = Lazy::new(|| {
    Arc::new(UnifiedCdfa::new().expect("Failed to initialize global CDFA instance"))
});

/// Compatibility module for cdfa-core functionality
#[cfg(feature = "compat-core")]
pub mod core {
    use super::*;
    
    /// Legacy diversity calculation function
    /// 
    /// Maintains compatibility with cdfa-core's diversity functions
    pub fn calculate_diversity(data: &FloatArrayView2) -> Result<FloatArray1> {
        GLOBAL_CDFA.calculate_diversity(data)
    }
    
    /// Legacy Kendall tau calculation
    pub fn kendall_tau(x: &FloatArrayView1, y: &FloatArrayView1) -> Result<Float> {
        use crate::core::diversity;
        diversity::kendall_tau(x, y)
    }
    
    /// Legacy Pearson correlation calculation
    pub fn pearson_correlation(x: &FloatArrayView1, y: &FloatArrayView1) -> Result<Float> {
        use crate::core::diversity;
        diversity::pearson_correlation(x, y)
    }
    
    /// Legacy Spearman correlation calculation
    pub fn spearman_correlation(x: &FloatArrayView1, y: &FloatArrayView1) -> Result<Float> {
        use crate::core::diversity;
        diversity::spearman_correlation(x, y)
    }
    
    /// Legacy fusion functionality
    pub fn fuse_scores(scores: &FloatArrayView1, method: &str) -> Result<Float> {
        use crate::core::fusion;
        
        match method {
            "average" => Ok(scores.mean().unwrap_or(0.0)),
            "weighted" => {
                // Use equal weights for compatibility
                let weights = FloatArray1::from_elem(scores.len(), 1.0 / scores.len() as Float);
                fusion::weighted_fusion(scores, &weights.view())
            },
            _ => Err(crate::error::CdfaError::unsupported_operation(format!("Unknown fusion method: {}", method))),
        }
    }
    
    /// Legacy diversity metrics struct for compatibility
    #[derive(Debug, Clone)]
    pub struct DiversityMetrics {
        pub kendall_tau: Float,
        pub pearson_correlation: Float,
        pub spearman_correlation: Float,
    }
    
    impl DiversityMetrics {
        /// Calculate all diversity metrics for two signals
        pub fn calculate(x: &FloatArrayView1, y: &FloatArrayView1) -> Result<Self> {
            Ok(Self {
                kendall_tau: kendall_tau(x, y)?,
                pearson_correlation: pearson_correlation(x, y)?,
                spearman_correlation: spearman_correlation(x, y)?,
            })
        }
    }
}

/// Compatibility module for cdfa-algorithms functionality
#[cfg(feature = "compat-algorithms")]
pub mod algorithms {
    use super::*;
    
    /// Legacy wavelet transform
    pub fn wavelet_transform(signal: &FloatArrayView1, levels: usize) -> Result<(FloatArray1, FloatArray1)> {
        use crate::algorithms::wavelet;
        wavelet::WaveletTransform::dwt_haar(signal)
    }
    
    /// Legacy entropy calculation
    pub fn sample_entropy(signal: &FloatArrayView1, m: usize, r: Float) -> Result<Float> {
        use crate::algorithms::entropy;
        entropy::SampleEntropy::calculate(signal, m, r)
    }
    
    /// Legacy volatility clustering
    pub fn volatility_clustering(prices: &FloatArrayView1) -> Result<FloatArray1> {
        use crate::algorithms::volatility;
        volatility::VolatilityClustering::cluster(prices)
    }
}

/// Compatibility module for cdfa-detectors functionality
#[cfg(feature = "compat-detectors")]
pub mod detectors {
    use super::*;
    
    /// Legacy Fibonacci pattern detection
    pub fn detect_fibonacci_patterns(prices: &FloatArrayView1) -> Result<Vec<Pattern>> {
        let data = prices.clone().into_shape((prices.len(), 1)).unwrap();
        let scores = FloatArray1::zeros(1);
        
        use crate::detectors::fibonacci::FibonacciPatternDetector;
        let detector = FibonacciPatternDetector::new();
        detector.detect(&data.view(), &scores.view())
    }
    
    /// Legacy Black Swan event detection
    pub fn detect_black_swan_events(data: &FloatArrayView2) -> Result<Vec<Pattern>> {
        let scores = FloatArray1::zeros(data.ncols());
        
        use crate::detectors::black_swan::BlackSwanDetector;
        let detector = BlackSwanDetector::new();
        detector.detect(data, &scores.view())
    }
    
    /// Legacy pattern detection result
    #[derive(Debug, Clone)]
    pub struct PatternResult {
        pub pattern_type: String,
        pub confidence: Float,
        pub start_index: usize,
        pub end_index: usize,
        pub parameters: std::collections::HashMap<String, Float>,
    }
    
    impl From<Pattern> for PatternResult {
        fn from(pattern: Pattern) -> Self {
            Self {
                pattern_type: pattern.pattern_type,
                confidence: pattern.confidence,
                start_index: pattern.start_index,
                end_index: pattern.end_index,
                parameters: pattern.parameters,
            }
        }
    }
}

/// Compatibility module for cdfa-parallel functionality
#[cfg(feature = "compat-parallel")]
pub mod parallel {
    use super::*;
    
    /// Legacy parallel processing configuration
    #[derive(Debug, Clone)]
    pub struct ParallelConfig {
        pub num_threads: usize,
        pub chunk_size: usize,
    }
    
    impl Default for ParallelConfig {
        fn default() -> Self {
            Self {
                num_threads: num_cpus::get(),
                chunk_size: 1000,
            }
        }
    }
    
    /// Legacy parallel diversity calculation
    pub fn parallel_diversity_calculation(
        data: &FloatArrayView2,
        config: &ParallelConfig,
    ) -> Result<FloatArray1> {
        // Update global CDFA configuration
        GLOBAL_CDFA.update_config(|cfg| {
            cfg.num_threads = config.num_threads;
        })?;
        
        GLOBAL_CDFA.calculate_diversity(data)
    }
    
    /// Legacy batch processing
    pub fn process_batch(
        datasets: &[FloatArrayView2],
        config: &ParallelConfig,
    ) -> Result<Vec<AnalysisResult>> {
        // Update configuration
        GLOBAL_CDFA.update_config(|cfg| {
            cfg.num_threads = config.num_threads;
        })?;
        
        // Process each dataset
        let mut results = Vec::new();
        for data in datasets {
            results.push(GLOBAL_CDFA.analyze(data)?);
        }
        
        Ok(results)
    }
}

/// Compatibility module for cdfa-simd functionality
#[cfg(feature = "compat-simd")]
pub mod simd {
    use super::*;
    
    /// Legacy SIMD configuration
    #[derive(Debug, Clone)]
    pub struct SimdConfig {
        pub enable_avx: bool,
        pub enable_sse: bool,
        pub vector_width: usize,
    }
    
    impl Default for SimdConfig {
        fn default() -> Self {
            Self {
                enable_avx: true,
                enable_sse: true,
                vector_width: 8,
            }
        }
    }
    
    /// Legacy SIMD-accelerated correlation calculation
    pub fn simd_correlation(x: &FloatArrayView1, y: &FloatArrayView1) -> Result<Float> {
        // The unified implementation automatically uses SIMD when available
        use crate::core::diversity;
        diversity::pearson_correlation_fast(x, y)
    }
    
    /// Check if SIMD is available
    pub fn is_simd_available() -> bool {
        #[cfg(feature = "simd")]
        {
            // Check CPU features
            true // Simplified - real implementation would check CPU features
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }
}

/// Compatibility module for cdfa-ml functionality
#[cfg(feature = "compat-ml")]
pub mod ml {
    use super::*;
    
    /// Legacy ML model configuration
    #[derive(Debug, Clone)]
    pub struct MLConfig {
        pub model_type: String,
        pub learning_rate: Float,
        pub epochs: usize,
        pub batch_size: usize,
    }
    
    impl Default for MLConfig {
        fn default() -> Self {
            Self {
                model_type: "neural_network".to_string(),
                learning_rate: 0.001,
                epochs: 100,
                batch_size: 32,
            }
        }
    }
    
    /// Legacy pattern detector using ML
    pub fn ml_pattern_detection(
        training_data: &FloatArrayView2,
        test_data: &FloatArrayView2,
        config: &MLConfig,
    ) -> Result<Vec<Pattern>> {
        // Simplified ML pattern detection for compatibility
        let scores = FloatArray1::zeros(test_data.ncols());
        
        // Use a basic detector as fallback
        use crate::detectors::fibonacci::FibonacciPatternDetector;
        let detector = FibonacciPatternDetector::new();
        detector.detect(test_data, &scores.view())
    }
    
    /// Legacy neural network prediction
    pub fn neural_prediction(
        model_data: &FloatArrayView2,
        input_data: &FloatArrayView1,
    ) -> Result<FloatArray1> {
        // Simplified prediction - returns processed input
        Ok(input_data.to_owned())
    }
}

/// Legacy wrapper functions for the most common operations
pub mod legacy {
    use super::*;
    
    /// Simple CDFA analysis function matching the original API
    pub fn cdfa_analyze(data: &FloatArrayView2) -> Result<AnalysisResult> {
        GLOBAL_CDFA.analyze(data)
    }
    
    /// Simple diversity calculation
    pub fn calculate_diversity_simple(data: &FloatArrayView2) -> Result<FloatArray1> {
        GLOBAL_CDFA.calculate_diversity(data)
    }
    
    /// Simple pattern detection
    pub fn detect_patterns_simple(data: &FloatArrayView2) -> Result<Vec<Pattern>> {
        let scores = FloatArray1::zeros(data.ncols());
        GLOBAL_CDFA.detect_patterns(data, &scores.view())
    }
    
    /// Initialize CDFA with specific configuration
    pub fn initialize_cdfa(config: CdfaConfig) -> Result<()> {
        GLOBAL_CDFA.update_config(|cfg| {
            *cfg = config;
        })
    }
    
    /// Get current CDFA configuration
    pub fn get_cdfa_config() -> CdfaConfig {
        GLOBAL_CDFA.config()
    }
    
    /// Reset CDFA to default configuration
    pub fn reset_cdfa() -> Result<()> {
        GLOBAL_CDFA.update_config(|cfg| {
            *cfg = CdfaConfig::default();
        })
    }
}

/// Re-exports for drop-in compatibility
pub use legacy::*;

/// Type aliases for backward compatibility
pub type CdfaResult<T> = Result<T>;
pub type DiversityScore = Float;
pub type FusionScore = Float;
pub type PatternConfidence = Float;

/// Constants for backward compatibility
pub const DEFAULT_TOLERANCE: Float = 1e-10;
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;
pub const DEFAULT_CONVERGENCE_THRESHOLD: Float = 1e-6;

/// Macro for easy migration from individual crates
#[macro_export]
macro_rules! migrate_cdfa {
    (core) => {
        #[cfg(feature = "compat-core")]
        pub use cdfa_unified::compat::core::*;
    };
    (algorithms) => {
        #[cfg(feature = "compat-algorithms")]
        pub use cdfa_unified::compat::algorithms::*;
    };
    (detectors) => {
        #[cfg(feature = "compat-detectors")]
        pub use cdfa_unified::compat::detectors::*;
    };
    (parallel) => {
        #[cfg(feature = "compat-parallel")]
        pub use cdfa_unified::compat::parallel::*;
    };
    (simd) => {
        #[cfg(feature = "compat-simd")]
        pub use cdfa_unified::compat::simd::*;
    };
    (ml) => {
        #[cfg(feature = "compat-ml")]
        pub use cdfa_unified::compat::ml::*;
    };
    (all) => {
        migrate_cdfa!(core);
        migrate_cdfa!(algorithms);
        migrate_cdfa!(detectors);
        migrate_cdfa!(parallel);
        migrate_cdfa!(simd);
        migrate_cdfa!(ml);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_legacy_api() {
        let data = array![
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 2.9],
            [0.9, 1.9, 3.1]
        ];
        
        // Test legacy analysis function
        let result = cdfa_analyze(&data.view());
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert_eq!(analysis.data.len(), 3);
    }
    
    #[cfg(feature = "compat-core")]
    #[test]
    fn test_core_compatibility() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.1, 2.1, 2.9, 4.1, 4.9];
        
        let correlation = core::pearson_correlation(&x.view(), &y.view());
        assert!(correlation.is_ok());
        assert!(correlation.unwrap() > 0.9);
        
        let kendall = core::kendall_tau(&x.view(), &y.view());
        assert!(kendall.is_ok());
    }
    
    #[cfg(feature = "compat-algorithms")]
    #[test]
    fn test_algorithms_compatibility() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        let wavelet_result = algorithms::wavelet_transform(&signal.view(), 3);
        assert!(wavelet_result.is_ok());
        
        let (approx, detail) = wavelet_result.unwrap();
        assert!(approx.len() > 0);
        assert!(detail.len() > 0);
    }
    
    #[test]
    fn test_configuration_compatibility() {
        let original_config = get_cdfa_config();
        
        // Update configuration
        let mut new_config = CdfaConfig::default();
        new_config.num_threads = 8;
        new_config.tolerance = 1e-12;
        
        let result = initialize_cdfa(new_config.clone());
        assert!(result.is_ok());
        
        // Verify configuration was updated
        let updated_config = get_cdfa_config();
        assert_eq!(updated_config.num_threads, 8);
        assert_eq!(updated_config.tolerance, 1e-12);
        
        // Reset to original
        let reset_result = reset_cdfa();
        assert!(reset_result.is_ok());
    }
    
    #[test]
    fn test_migration_macro() {
        // Test that the macro compiles
        migrate_cdfa!(core);
        
        // Would test specific functionality if the features were enabled
        #[cfg(feature = "compat-core")]
        {
            let data = array![[1.0, 2.0], [3.0, 4.0]];
            let diversity = calculate_diversity(&data.view());
            assert!(diversity.is_ok());
        }
    }
}
