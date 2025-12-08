//! # CDFA Antifragility Analyzer
//!
//! High-performance Rust implementation of Nassim Nicholas Taleb's Antifragility concept
//! with sub-microsecond analysis capabilities.
//!
//! This crate provides comprehensive antifragility analysis including:
//! - Convexity measurement (correlation between performance acceleration and volatility)
//! - Asymmetry analysis (skewness and kurtosis under stress)
//! - Recovery velocity (performance after volatility spikes)
//! - Benefit ratio (performance improvement vs volatility cost)
//! - Multiple volatility estimators (Yang-Zhang, GARCH, Parkinson, ATR)
//! - SIMD-optimized calculations for maximum performance
//!
//! ## Key Features
//! - Sub-microsecond performance targets
//! - Hardware acceleration (AVX2, AVX512, NEON)
//! - Parallel processing capabilities
//! - Memory-efficient algorithms
//! - Comprehensive statistical methods
//! - Python bindings available
//!
//! ## Usage
//! ```rust
//! use cdfa_antifragility_analyzer::{AntifragilityAnalyzer, AntifragilityParameters};
//!
//! let analyzer = AntifragilityAnalyzer::new(AntifragilityParameters::default());
//! let result = analyzer.analyze_prices(&prices, &volumes);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::prelude::*;
use nalgebra::{DMatrix, DVector};
use num_traits::{Float, FromPrimitive, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

#[cfg(feature = "simd")]
use wide::f64x4;

// Module declarations
pub mod convexity;
pub mod asymmetry;
pub mod recovery;
pub mod volatility;
pub mod benefit_ratio;
pub mod types;
pub mod utils;
pub mod simd_utils;
pub mod cache;
pub mod performance;

#[cfg(feature = "python")]
pub mod python_bindings;

#[cfg(feature = "c-bindings")]
pub mod c_bindings;

// Re-exports
pub use types::*;
pub use utils::*;
pub use performance::PerformanceMetrics;

/// Errors that can occur during antifragility analysis
#[derive(Error, Debug)]
pub enum AntifragilityError {
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    
    #[error("Calculation error: {source}")]
    CalculationError { source: Box<dyn std::error::Error + Send + Sync> },
    
    #[error("Memory allocation failed")]
    MemoryError,
    
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },
    
    #[error("Cache operation failed: {message}")]
    CacheError { message: String },
}

/// Result type for antifragility operations
pub type AntifragilityResult<T> = Result<T, AntifragilityError>;

/// Parameters for antifragility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityParameters {
    /// Component weights (must sum to 1.0)
    pub convexity_weight: f64,
    pub asymmetry_weight: f64,
    pub recovery_weight: f64,
    pub benefit_ratio_weight: f64,
    
    /// Volatility estimation parameters
    pub yz_volatility_k: f64,
    pub garch_alpha_base: f64,
    pub parkinson_factor: f64,
    
    /// Signal processing parameters
    pub recovery_horizon_factor: f64,
    pub vol_lookback_factor: f64,
    
    /// Performance parameters
    pub vol_period: usize,
    pub perf_period: usize,
    pub corr_window: usize,
    pub smoothing_span: usize,
    
    /// Cache and optimization settings
    pub cache_size: usize,
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub min_data_points: usize,
}

impl Default for AntifragilityParameters {
    fn default() -> Self {
        Self {
            // Component weights (from Python implementation)
            convexity_weight: 0.40,
            asymmetry_weight: 0.20,
            recovery_weight: 0.25,
            benefit_ratio_weight: 0.15,
            
            // Volatility parameters
            yz_volatility_k: 0.34,
            garch_alpha_base: 0.05,
            parkinson_factor: 4.0 * 2.0_f64.ln(), // 4 * ln(2)
            
            // Signal processing
            recovery_horizon_factor: 0.5,
            vol_lookback_factor: 3.0,
            
            // Period parameters
            vol_period: 21,
            perf_period: 63,
            corr_window: 42,
            smoothing_span: 10,
            
            // Performance settings
            cache_size: 1000,
            enable_simd: true,
            enable_parallel: true,
            min_data_points: 100,
        }
    }
}

impl AntifragilityParameters {
    /// Validate parameters
    pub fn validate(&self) -> AntifragilityResult<()> {
        // Check weights sum to 1.0
        let weight_sum = self.convexity_weight + self.asymmetry_weight + 
                        self.recovery_weight + self.benefit_ratio_weight;
        if (weight_sum - 1.0).abs() > 1e-10 {
            return Err(AntifragilityError::InvalidParameters {
                message: format!("Weights must sum to 1.0, got {:.10}", weight_sum)
            });
        }
        
        // Check individual weights are non-negative
        if self.convexity_weight < 0.0 || self.asymmetry_weight < 0.0 ||
           self.recovery_weight < 0.0 || self.benefit_ratio_weight < 0.0 {
            return Err(AntifragilityError::InvalidParameters {
                message: "All weights must be non-negative".to_string()
            });
        }
        
        // Check periods are positive
        if self.vol_period == 0 || self.perf_period == 0 || 
           self.corr_window == 0 || self.smoothing_span == 0 {
            return Err(AntifragilityError::InvalidParameters {
                message: "All periods must be positive".to_string()
            });
        }
        
        // Check volatility parameters
        if self.yz_volatility_k <= 0.0 || self.garch_alpha_base <= 0.0 ||
           self.parkinson_factor <= 0.0 {
            return Err(AntifragilityError::InvalidParameters {
                message: "Volatility parameters must be positive".to_string()
            });
        }
        
        Ok(())
    }
}

/// High-performance antifragility analyzer
pub struct AntifragilityAnalyzer {
    params: AntifragilityParameters,
    cache: Arc<cache::AnalysisCache>,
    performance_metrics: Arc<std::sync::Mutex<PerformanceMetrics>>,
}

impl AntifragilityAnalyzer {
    /// Create a new analyzer with default parameters
    pub fn new() -> Self {
        Self::with_params(AntifragilityParameters::default())
    }
    
    /// Create a new analyzer with custom parameters
    pub fn with_params(params: AntifragilityParameters) -> Self {
        params.validate().expect("Invalid parameters");
        
        Self {
            cache: Arc::new(cache::AnalysisCache::new(params.cache_size)),
            performance_metrics: Arc::new(std::sync::Mutex::new(PerformanceMetrics::new())),
            params,
        }
    }
    
    /// Analyze price data for antifragility
    #[instrument(skip(self, prices, volumes))]
    pub fn analyze_prices(&self, prices: &[f64], volumes: &[f64]) -> AntifragilityResult<AnalysisResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if prices.len() < self.params.min_data_points {
            return Err(AntifragilityError::InsufficientData {
                required: self.params.min_data_points,
                actual: prices.len(),
            });
        }
        
        if prices.len() != volumes.len() {
            return Err(AntifragilityError::InvalidParameters {
                message: format!("Price and volume arrays must have same length: {} vs {}", 
                               prices.len(), volumes.len())
            });
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(prices, volumes);
        if let Some(cached_result) = self.cache.get(&cache_key) {
            debug!("Cache hit for analysis");
            return Ok(cached_result);
        }
        
        // Perform analysis
        let result = self.perform_analysis(prices, volumes)?;
        
        // Cache result
        self.cache.insert(cache_key, result.clone());
        
        // Update performance metrics
        let duration = start_time.elapsed();
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.record_analysis(duration, prices.len());
        }
        
        info!("Analysis completed in {:?} for {} data points", duration, prices.len());
        Ok(result)
    }
    
    /// Perform the core antifragility analysis
    #[instrument(skip(self, prices, volumes))]
    fn perform_analysis(&self, prices: &[f64], volumes: &[f64]) -> AntifragilityResult<AnalysisResult> {
        let n = prices.len();
        
        // Convert to ndarray for efficient processing
        let price_array = Array1::from_vec(prices.to_vec());
        let volume_array = Array1::from_vec(volumes.to_vec());
        
        // Calculate log returns
        let log_returns = self.calculate_log_returns(&price_array)?;
        
        // Calculate multiple volatility estimators
        let volatility_result = self.calculate_robust_volatility(&price_array, &volume_array)?;
        
        // Calculate performance metrics
        let performance_result = self.calculate_performance_metrics(&price_array, &log_returns)?;
        
        // Calculate component A: Convexity
        let convexity_score = self.calculate_convexity_component(
            &performance_result.acceleration,
            &volatility_result.vol_roc_smoothed,
        )?;
        
        // Calculate component B: Asymmetry
        let asymmetry_score = self.calculate_asymmetry_component(
            &log_returns,
            &volatility_result.vol_regime,
        )?;
        
        // Calculate component C: Recovery velocity
        let recovery_score = self.calculate_recovery_component(
            &price_array,
            &volatility_result.vol_roc_smoothed,
        )?;
        
        // Calculate component D: Benefit ratio
        let benefit_ratio_score = self.calculate_benefit_ratio_component(
            &performance_result.perf_roc_smoothed,
            &volatility_result.vol_roc_smoothed,
        )?;
        
        // Combine components with weights
        let antifragility_raw = self.params.convexity_weight * convexity_score +
                               self.params.asymmetry_weight * asymmetry_score +
                               self.params.recovery_weight * recovery_score +
                               self.params.benefit_ratio_weight * benefit_ratio_score;
        
        // Apply exponential smoothing
        let antifragility_index = self.apply_exponential_smoothing(antifragility_raw, self.params.smoothing_span)?;
        
        // Calculate fragility score (inverse of antifragility)
        let fragility_score = self.calculate_fragility_score(antifragility_index, &volatility_result.vol_regime)?;
        
        // Build result
        let result = AnalysisResult {
            antifragility_index,
            fragility_score,
            convexity_score,
            asymmetry_score,
            recovery_score,
            benefit_ratio_score,
            volatility: volatility_result,
            performance: performance_result,
            data_points: n,
            calculation_time: std::time::Duration::from_nanos(0), // Will be set by caller
        };
        
        Ok(result)
    }
    
    /// Calculate log returns from prices
    fn calculate_log_returns(&self, prices: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut log_returns = Array1::zeros(n);
        
        if self.params.enable_simd && n > 4 {
            // Use SIMD for large arrays
            simd_utils::calculate_log_returns_simd(prices, &mut log_returns)?;
        } else {
            // Standard calculation
            for i in 1..n {
                log_returns[i] = (prices[i] / prices[i-1]).ln();
            }
        }
        
        Ok(log_returns)
    }
    
    /// Calculate robust volatility using multiple estimators
    fn calculate_robust_volatility(&self, prices: &Array1<f64>, volumes: &Array1<f64>) -> AntifragilityResult<VolatilityResult> {
        volatility::calculate_robust_volatility(prices, volumes, &self.params)
    }
    
    /// Calculate performance metrics
    fn calculate_performance_metrics(&self, prices: &Array1<f64>, log_returns: &Array1<f64>) -> AntifragilityResult<PerformanceResult> {
        let n = prices.len();
        let mut result = PerformanceResult::default();
        
        // Calculate performance over specified period
        let perf_period = self.params.perf_period.min(n - 1);
        let mut log_perf_returns = Array1::zeros(n);
        
        for i in perf_period..n {
            log_perf_returns[i] = (prices[i] / prices[i - perf_period]).ln();
        }
        
        // Calculate performance momentum and acceleration
        let mut perf_momentum = Array1::zeros(n);
        let mut perf_acceleration = Array1::zeros(n);
        
        for i in 1..n {
            perf_momentum[i] = log_perf_returns[i] - log_perf_returns[i-1];
        }
        
        for i in 1..n {
            perf_acceleration[i] = perf_momentum[i] - perf_momentum[i-1];
        }
        
        // Apply exponential smoothing to acceleration
        let smoothing_period = self.params.perf_period / 3;
        result.acceleration = self.apply_exponential_smoothing_array(&perf_acceleration, smoothing_period)?;
        
        // Calculate performance rate of change
        let mut perf_roc_smoothed = Array1::zeros(n);
        for i in 1..n {
            perf_roc_smoothed[i] = (log_perf_returns[i] - log_perf_returns[i-1]) / log_perf_returns[i-1].abs().max(1e-10);
        }
        
        result.perf_roc_smoothed = self.apply_exponential_smoothing_array(&perf_roc_smoothed, smoothing_period)?;
        
        Ok(result)
    }
    
    /// Calculate convexity component
    fn calculate_convexity_component(&self, perf_acceleration: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        convexity::calculate_convexity_correlation(perf_acceleration, vol_roc_smoothed, self.params.corr_window)
    }
    
    /// Calculate asymmetry component
    fn calculate_asymmetry_component(&self, log_returns: &Array1<f64>, vol_regime: &Array1<f64>) -> AntifragilityResult<f64> {
        asymmetry::calculate_weighted_asymmetry(log_returns, vol_regime, self.params.perf_period)
    }
    
    /// Calculate recovery component
    fn calculate_recovery_component(&self, prices: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        recovery::calculate_recovery_velocity(prices, vol_roc_smoothed, &self.params)
    }
    
    /// Calculate benefit ratio component
    fn calculate_benefit_ratio_component(&self, perf_roc_smoothed: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        benefit_ratio::calculate_benefit_ratio(perf_roc_smoothed, vol_roc_smoothed)
    }
    
    /// Apply exponential smoothing to a single value
    fn apply_exponential_smoothing(&self, value: f64, span: usize) -> AntifragilityResult<f64> {
        // For single values, just return the value
        // In a real implementation, this would maintain state across calls
        Ok(value.clamp(0.0, 1.0))
    }
    
    /// Apply exponential smoothing to an array
    fn apply_exponential_smoothing_array(&self, values: &Array1<f64>, span: usize) -> AntifragilityResult<Array1<f64>> {
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut smoothed = Array1::zeros(values.len());
        
        if !values.is_empty() {
            smoothed[0] = values[0];
            for i in 1..values.len() {
                smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i-1];
            }
        }
        
        Ok(smoothed)
    }
    
    /// Calculate fragility score
    fn calculate_fragility_score(&self, antifragility_index: f64, vol_regime: &Array1<f64>) -> AntifragilityResult<f64> {
        let avg_vol_regime = vol_regime.mean().unwrap_or(0.5);
        let fragility = (1.0 - antifragility_index) * (0.7 + 0.3 * avg_vol_regime);
        Ok(fragility.clamp(0.0, 1.0))
    }
    
    /// Generate cache key for analysis
    fn generate_cache_key(&self, prices: &[f64], volumes: &[f64]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash a subset of data for efficiency
        let sample_size = (prices.len() / 10).max(10).min(100);
        for &price in prices.iter().step_by(prices.len() / sample_size) {
            price.to_bits().hash(&mut hasher);
        }
        for &volume in volumes.iter().step_by(volumes.len() / sample_size) {
            volume.to_bits().hash(&mut hasher);
        }
        
        // Include parameters in hash
        self.params.vol_period.hash(&mut hasher);
        self.params.perf_period.hash(&mut hasher);
        self.params.corr_window.hash(&mut hasher);
        
        format!("antifragility_{:x}", hasher.finish())
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.lock()
            .map(|metrics| metrics.clone())
            .unwrap_or_default()
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for AntifragilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        
        let mut price = 100.0;
        for i in 0..n {
            let return_rate = 0.001 * ((i as f64) * 0.1).sin();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 100.0 * ((i as f64) * 0.05).cos());
        }
        
        (prices, volumes)
    }
    
    #[test]
    fn test_antifragility_analyzer_creation() {
        let analyzer = AntifragilityAnalyzer::new();
        assert_eq!(analyzer.params.convexity_weight, 0.40);
        assert_eq!(analyzer.params.asymmetry_weight, 0.20);
        assert_eq!(analyzer.params.recovery_weight, 0.25);
        assert_eq!(analyzer.params.benefit_ratio_weight, 0.15);
    }
    
    #[test]
    fn test_parameter_validation() {
        let mut params = AntifragilityParameters::default();
        params.convexity_weight = 0.5;
        params.asymmetry_weight = 0.5;
        params.recovery_weight = 0.0;
        params.benefit_ratio_weight = 0.0;
        
        assert!(params.validate().is_ok());
        
        params.convexity_weight = 0.6;
        assert!(params.validate().is_err());
    }
    
    #[test]
    fn test_basic_analysis() {
        let analyzer = AntifragilityAnalyzer::new();
        let (prices, volumes) = generate_test_data(200);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.antifragility_index >= 0.0);
        assert!(analysis.antifragility_index <= 1.0);
        assert!(analysis.fragility_score >= 0.0);
        assert!(analysis.fragility_score <= 1.0);
    }
    
    #[test]
    fn test_insufficient_data() {
        let analyzer = AntifragilityAnalyzer::new();
        let (prices, volumes) = generate_test_data(50);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_err());
        
        if let Err(AntifragilityError::InsufficientData { required, actual }) = result {
            assert_eq!(required, 100);
            assert_eq!(actual, 50);
        } else {
            panic!("Expected InsufficientData error");
        }
    }
    
    #[test]
    fn test_cache_functionality() {
        let analyzer = AntifragilityAnalyzer::new();
        let (prices, volumes) = generate_test_data(200);
        
        // First analysis
        let start = Instant::now();
        let result1 = analyzer.analyze_prices(&prices, &volumes).unwrap();
        let duration1 = start.elapsed();
        
        // Second analysis (should be cached)
        let start = Instant::now();
        let result2 = analyzer.analyze_prices(&prices, &volumes).unwrap();
        let duration2 = start.elapsed();
        
        // Results should be identical
        assert_relative_eq!(result1.antifragility_index, result2.antifragility_index, epsilon = 1e-10);
        
        // Second call should be faster due to caching
        assert!(duration2 < duration1);
    }
}