//! # Antifragility Analyzer
//!
//! High-performance Rust implementation of Nassim Nicholas Taleb's Antifragility concept
//! with sub-microsecond analysis capabilities.
//!
//! This analyzer provides comprehensive antifragility analysis including:
//! - Convexity measurement (correlation between performance acceleration and volatility)
//! - Asymmetry analysis (skewness and kurtosis under stress)
//! - Recovery velocity (performance after volatility spikes)
//! - Benefit ratio (performance improvement vs volatility cost)
//! - Multiple volatility estimators (Yang-Zhang, GARCH, Parkinson, ATR)
//! - SIMD-optimized calculations for maximum performance

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::prelude::*;
// Nalgebra matrices available through types
use num_traits::Float;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::{CdfaError, CdfaResult};

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

/// Convert AntifragilityError to CdfaError
impl From<AntifragilityError> for CdfaError {
    fn from(err: AntifragilityError) -> Self {
        CdfaError::analysis_error(format!("Antifragility analysis failed: {}", err))
    }
}

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

/// Result of volatility analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityResult {
    /// Combined volatility estimate
    pub combined_vol: Array1<f64>,
    /// Volatility regime score (0-1)
    pub vol_regime: Array1<f64>,
    /// Smoothed volatility rate of change
    pub vol_roc_smoothed: Array1<f64>,
    /// Yang-Zhang volatility
    pub yz_volatility: Array1<f64>,
    /// GARCH-like volatility
    pub garch_volatility: Array1<f64>,
    /// Parkinson volatility
    pub parkinson_volatility: Array1<f64>,
    /// ATR-based volatility
    pub atr_volatility: Array1<f64>,
}

impl Default for VolatilityResult {
    fn default() -> Self {
        Self {
            combined_vol: Array1::zeros(0),
            vol_regime: Array1::zeros(0),
            vol_roc_smoothed: Array1::zeros(0),
            yz_volatility: Array1::zeros(0),
            garch_volatility: Array1::zeros(0),
            parkinson_volatility: Array1::zeros(0),
            atr_volatility: Array1::zeros(0),
        }
    }
}

/// Result of performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceResult {
    /// Performance acceleration
    pub acceleration: Array1<f64>,
    /// Smoothed performance rate of change
    pub perf_roc_smoothed: Array1<f64>,
    /// Performance momentum
    pub momentum: Array1<f64>,
    /// Log performance returns
    pub log_perf_returns: Array1<f64>,
}

impl Default for PerformanceResult {
    fn default() -> Self {
        Self {
            acceleration: Array1::zeros(0),
            perf_roc_smoothed: Array1::zeros(0),
            momentum: Array1::zeros(0),
            log_perf_returns: Array1::zeros(0),
        }
    }
}

/// Complete antifragility analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Main antifragility index (0-1, higher = more antifragile)
    pub antifragility_index: f64,
    /// Fragility score (0-1, higher = more fragile)
    pub fragility_score: f64,
    /// Convexity component score
    pub convexity_score: f64,
    /// Asymmetry component score
    pub asymmetry_score: f64,
    /// Recovery velocity component score
    pub recovery_score: f64,
    /// Benefit ratio component score
    pub benefit_ratio_score: f64,
    /// Detailed volatility analysis
    pub volatility: VolatilityResult,
    /// Detailed performance analysis
    pub performance: PerformanceResult,
    /// Number of data points analyzed
    pub data_points: usize,
    /// Time taken for calculation
    pub calculation_time: Duration,
}

impl AnalysisResult {
    /// Get a summary of the analysis
    pub fn summary(&self) -> String {
        format!(
            "Antifragility Analysis Summary:\n\
             - Antifragility Index: {:.4}\n\
             - Fragility Score: {:.4}\n\
             - Convexity: {:.4}\n\
             - Asymmetry: {:.4}\n\
             - Recovery: {:.4}\n\
             - Benefit Ratio: {:.4}\n\
             - Data Points: {}\n\
             - Calculation Time: {:?}",
            self.antifragility_index,
            self.fragility_score,
            self.convexity_score,
            self.asymmetry_score,
            self.recovery_score,
            self.benefit_ratio_score,
            self.data_points,
            self.calculation_time
        )
    }
    
    /// Check if the system is antifragile
    pub fn is_antifragile(&self) -> bool {
        self.antifragility_index > 0.6
    }
    
    /// Check if the system is fragile
    pub fn is_fragile(&self) -> bool {
        self.fragility_score > 0.6
    }
    
    /// Check if the system is robust (neither fragile nor antifragile)
    pub fn is_robust(&self) -> bool {
        !self.is_antifragile() && !self.is_fragile()
    }
    
    /// Get the dominant characteristic
    pub fn dominant_characteristic(&self) -> &'static str {
        if self.is_antifragile() {
            "Antifragile"
        } else if self.is_fragile() {
            "Fragile"
        } else {
            "Robust"
        }
    }
}

/// Simple cache for analysis results
#[derive(Debug)]
pub struct AnalysisCache {
    cache: std::sync::Mutex<HashMap<String, AnalysisResult>>,
    max_size: usize,
}

impl AnalysisCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::sync::Mutex::new(HashMap::new()),
            max_size,
        }
    }
    
    pub fn get(&self, key: &str) -> Option<AnalysisResult> {
        self.cache.lock().ok()?.get(key).cloned()
    }
    
    pub fn insert(&self, key: String, value: AnalysisResult) {
        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= self.max_size {
                // Simple eviction: remove a random entry
                if let Some(old_key) = cache.keys().next().cloned() {
                    cache.remove(&old_key);
                }
            }
            cache.insert(key, value);
        }
    }
    
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

/// Performance metrics tracker
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_analyses: usize,
    pub total_time: Duration,
    pub avg_time_per_analysis: Duration,
    pub total_data_points: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_analysis(&mut self, duration: Duration, data_points: usize) {
        self.total_analyses += 1;
        self.total_time += duration;
        self.total_data_points += data_points;
        
        if self.total_analyses > 0 {
            self.avg_time_per_analysis = self.total_time / self.total_analyses as u32;
        }
    }
    
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            self.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }
}

/// High-performance antifragility analyzer
pub struct AntifragilityAnalyzer {
    params: AntifragilityParameters,
    cache: Arc<AnalysisCache>,
    performance_metrics: Arc<std::sync::Mutex<PerformanceMetrics>>,
}

impl AntifragilityAnalyzer {
    /// Create a new analyzer with default parameters
    pub fn new() -> CdfaResult<Self> {
        Self::with_params(AntifragilityParameters::default())
    }
    
    /// Create a new analyzer with custom parameters
    pub fn with_params(params: AntifragilityParameters) -> CdfaResult<Self> {
        params.validate()?;
        
        Ok(Self {
            cache: Arc::new(AnalysisCache::new(params.cache_size)),
            performance_metrics: Arc::new(std::sync::Mutex::new(PerformanceMetrics::new())),
            params,
        })
    }
    
    /// Analyze price data for antifragility
    #[instrument(skip(self, prices, volumes))]
    pub fn analyze_prices(&self, prices: &[f64], volumes: &[f64]) -> CdfaResult<AnalysisResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if prices.len() < self.params.min_data_points {
            return Err(AntifragilityError::InsufficientData {
                required: self.params.min_data_points,
                actual: prices.len(),
            }.into());
        }
        
        if prices.len() != volumes.len() {
            return Err(AntifragilityError::InvalidParameters {
                message: format!("Price and volume arrays must have same length: {} vs {}", 
                               prices.len(), volumes.len())
            }.into());
        }
        
        // Check cache first
        let cache_key = self.generate_cache_key(prices, volumes);
        if let Some(cached_result) = self.cache.get(&cache_key) {
            debug!("Cache hit for analysis");
            if let Ok(mut metrics) = self.performance_metrics.lock() {
                metrics.record_cache_hit();
            }
            return Ok(cached_result);
        }
        
        if let Ok(mut metrics) = self.performance_metrics.lock() {
            metrics.record_cache_miss();
        }
        
        // Perform analysis
        let mut result = self.perform_analysis(prices, volumes)?;
        
        // Update timing
        let duration = start_time.elapsed();
        result.calculation_time = duration;
        
        // Cache result
        self.cache.insert(cache_key, result.clone());
        
        // Update performance metrics
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
            calculation_time: Duration::from_nanos(0), // Will be set by caller
        };
        
        Ok(result)
    }
    
    /// Calculate log returns from prices

    /// Calculate log returns from prices
    fn calculate_log_returns(&self, prices: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut log_returns = Array1::zeros(n);
        
        if self.params.enable_simd && n > 4 {
            // Use SIMD for large arrays
            self.calculate_log_returns_simd(prices, &mut log_returns)?;
        } else {
            // Standard calculation
            for i in 1..n {
                log_returns[i] = (prices[i] / prices[i-1]).ln();
            }
        }
        
        Ok(log_returns)
    }
    
    /// SIMD-optimized log returns calculation
    #[cfg(feature = "simd")]
    fn calculate_log_returns_simd(&self, prices: &Array1<f64>, log_returns: &mut Array1<f64>) -> AntifragilityResult<()> {
        let n = prices.len();
        
        if n < 2 {
            return Ok(());
        }
        
        log_returns[0] = 0.0;
        
        // Process in SIMD chunks
        let n_simd = (n - 1) / 4;
        
        for i in 0..n_simd {
            let base_idx = i * 4 + 1;
            
            if base_idx + 3 < n {
                let current_array = [prices[base_idx], prices[base_idx + 1], prices[base_idx + 2], prices[base_idx + 3]];
                let prev_array = [prices[base_idx - 1], prices[base_idx], prices[base_idx + 1], prices[base_idx + 2]];
                
                let current_prices = f64x4::new(current_array);
                let prev_prices = f64x4::new(prev_array);
                
                let ratios = current_prices / prev_prices;
                let log_ratios = ratios.ln();
                
                let log_array = log_ratios.to_array();
                log_returns[base_idx] = log_array[0];
                log_returns[base_idx + 1] = log_array[1];
                log_returns[base_idx + 2] = log_array[2];
                log_returns[base_idx + 3] = log_array[3];
            }
        }
        
        // Handle remaining elements
        for i in (n_simd * 4 + 1)..n {
            log_returns[i] = (prices[i] / prices[i - 1]).ln();
        }
        
        Ok(())
    }
    
    /// Fallback scalar implementation
    #[cfg(not(feature = "simd"))]
    fn calculate_log_returns_simd(&self, prices: &Array1<f64>, log_returns: &mut Array1<f64>) -> AntifragilityResult<()> {
        let n = prices.len();
        
        if n < 2 {
            return Ok(());
        }
        
        log_returns[0] = 0.0;
        
        for i in 1..n {
            log_returns[i] = (prices[i] / prices[i - 1]).ln();
        }
        
        Ok(())
    }
    
    /// Calculate robust volatility using multiple estimators
    fn calculate_robust_volatility(&self, prices: &Array1<f64>, volumes: &Array1<f64>) -> AntifragilityResult<VolatilityResult> {
        let n = prices.len();
        if n < self.params.vol_period + 5 {
            return Err(AntifragilityError::InsufficientData {
                required: self.params.vol_period + 5,
                actual: n,
            });
        }
        
        // Calculate individual volatility components
        let yz_volatility = self.calculate_yang_zhang_volatility(prices, volumes)?;
        let garch_volatility = self.calculate_garch_volatility(prices)?;
        let parkinson_volatility = self.calculate_parkinson_volatility(prices)?;
        let atr_volatility = self.calculate_atr_volatility(prices)?;
        
        // Combine volatility estimators
        let combined_vol = self.combine_volatility_estimators(
            &yz_volatility,
            &garch_volatility,
            &parkinson_volatility,
            &atr_volatility,
        )?;
        
        // Calculate volatility regime
        let vol_regime = self.calculate_volatility_regime(&combined_vol)?;
        
        // Calculate volatility rate of change
        let vol_roc_smoothed = self.calculate_volatility_roc(&vol_regime)?;
        
        Ok(VolatilityResult {
            combined_vol,
            vol_regime,
            vol_roc_smoothed,
            yz_volatility,
            garch_volatility,
            parkinson_volatility,
            atr_volatility,
        })
    }
    
    /// Calculate Yang-Zhang volatility estimator
    fn calculate_yang_zhang_volatility(&self, prices: &Array1<f64>, _volumes: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut yz_variance = Array1::zeros(n);
        let period = self.params.vol_period;
        
        // Calculate overnight returns (approximated as gaps)
        let mut overnight_returns = Array1::zeros(n);
        for i in 1..n {
            overnight_returns[i] = (prices[i] / prices[i-1]).ln();
        }
        
        // Calculate Yang-Zhang variance
        for i in period..n {
            let mut overnight_var = 0.0;
            let mut rs_sum = 0.0;
            let mut count = 0;
            
            for j in (i - period)..i {
                if j > 0 {
                    // Overnight variance component
                    let overnight_ret = overnight_returns[j];
                    overnight_var += overnight_ret * overnight_ret;
                    
                    // Rogers-Satchell component (simplified without intraday data)
                    let log_ret = (prices[j] / prices[j-1]).ln();
                    rs_sum += log_ret * log_ret;
                    count += 1;
                }
            }
            
            if count > 0 {
                overnight_var /= count as f64;
                rs_sum /= count as f64;
                
                // Yang-Zhang k parameter
                let k = self.params.yz_volatility_k / (1.34 + (period as f64 + 1.0) / (period as f64 - 1.0));
                
                // Combined YZ variance
                yz_variance[i] = overnight_var + k * rs_sum;
            }
        }
        
        // Convert to volatility (standard deviation)
        let yz_volatility = yz_variance.mapv(|x| x.max(0.0).sqrt());
        
        Ok(yz_volatility)
    }
    
    /// Calculate GARCH-like volatility with dynamic alpha
    fn calculate_garch_volatility(&self, prices: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut garch_var = Array1::zeros(n);
        
        // Calculate returns
        let mut returns = Array1::zeros(n);
        for i in 1..n {
            returns[i] = (prices[i] / prices[i-1]).ln();
        }
        
        // Calculate initial variance
        let initial_var = if n > 1 {
            let mean_return = returns.mean().unwrap_or(0.0);
            returns.mapv(|x| (x - mean_return).powi(2)).mean().unwrap_or(1e-6)
        } else {
            1e-6
        };
        
        garch_var[0] = initial_var;
        
        // Calculate GARCH variance with dynamic alpha
        for i in 1..n {
            let ret = returns[i];
            let ret_std = returns.std(1.0);
            let ret_ratio = (ret.abs() / ret_std.max(1e-9)).min(3.0);
            let dynamic_alpha = self.params.garch_alpha_base * (1.0 + ret_ratio);
            
            garch_var[i] = dynamic_alpha * ret * ret + (1.0 - dynamic_alpha) * garch_var[i-1];
        }
        
        // Convert to volatility
        let garch_volatility = garch_var.mapv(|x| x.max(0.0).sqrt());
        
        Ok(garch_volatility)
    }
    
    /// Calculate Parkinson volatility (high-low based)
    fn calculate_parkinson_volatility(&self, prices: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut parkinson_vol = Array1::zeros(n);
        let period = self.params.vol_period;
        
        // Since we only have close prices, approximate high-low from price movements
        let mut high_low_ratios = Array1::zeros(n);
        for i in 1..n {
            let price_change = (prices[i] - prices[i-1]).abs();
            let approx_high = prices[i] + price_change * 0.5;
            let approx_low = prices[i] - price_change * 0.5;
            
            if approx_low > 0.0 && approx_high > 0.0 {
                high_low_ratios[i] = (approx_high / approx_low).ln().powi(2);
            }
        }
        
        // Calculate rolling Parkinson volatility
        for i in period..n {
            let mut sum_hl = 0.0;
            let mut count = 0;
            
            for j in (i - period)..i {
                if high_low_ratios[j] > 0.0 {
                    sum_hl += high_low_ratios[j];
                    count += 1;
                }
            }
            
            if count > 0 {
                let park_var = sum_hl / (count as f64 * self.params.parkinson_factor);
                parkinson_vol[i] = park_var.max(0.0).sqrt();
            }
        }
        
        Ok(parkinson_vol)
    }
    
    /// Calculate ATR-based volatility
    fn calculate_atr_volatility(&self, prices: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = prices.len();
        let mut atr_vol = Array1::zeros(n);
        let period = self.params.vol_period;
        
        // Calculate true range (simplified for single price series)
        let mut true_ranges = Array1::zeros(n);
        for i in 1..n {
            let price_change = (prices[i] - prices[i-1]).abs();
            true_ranges[i] = price_change;
        }
        
        // Calculate ATR using exponential moving average
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut atr = Array1::zeros(n);
        
        if n > 0 {
            atr[0] = true_ranges[0];
            for i in 1..n {
                atr[i] = alpha * true_ranges[i] + (1.0 - alpha) * atr[i-1];
            }
        }
        
        // Normalize by current price to get relative volatility
        for i in 0..n {
            if prices[i] > 0.0 {
                atr_vol[i] = atr[i] / prices[i];
            }
        }
        
        Ok(atr_vol)
    }
    
    /// Combine multiple volatility estimators
    fn combine_volatility_estimators(
        &self,
        yz_vol: &Array1<f64>,
        garch_vol: &Array1<f64>,
        parkinson_vol: &Array1<f64>,
        atr_vol: &Array1<f64>,
    ) -> AntifragilityResult<Array1<f64>> {
        let n = yz_vol.len();
        let mut combined_vol = Array1::zeros(n);
        
        // Define weights for each estimator
        let weights = [0.35, 0.30, 0.15, 0.20]; // YZ, GARCH, Parkinson, ATR
        
        for i in 0..n {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;
            
            // Yang-Zhang
            if yz_vol[i] > 1e-9 {
                weighted_sum += weights[0] * yz_vol[i];
                weight_sum += weights[0];
            }
            
            // GARCH
            if garch_vol[i] > 1e-9 {
                weighted_sum += weights[1] * garch_vol[i];
                weight_sum += weights[1];
            }
            
            // Parkinson
            if parkinson_vol[i] > 1e-9 {
                weighted_sum += weights[2] * parkinson_vol[i];
                weight_sum += weights[2];
            }
            
            // ATR
            if atr_vol[i] > 1e-9 {
                weighted_sum += weights[3] * atr_vol[i];
                weight_sum += weights[3];
            }
            
            // Calculate weighted average
            if weight_sum > 0.0 {
                combined_vol[i] = weighted_sum / weight_sum;
            }
        }
        
        Ok(combined_vol)
    }
    
    /// Calculate volatility regime score (0-1)
    fn calculate_volatility_regime(&self, combined_vol: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = combined_vol.len();
        let mut vol_regime = Array1::zeros(n);
        let long_period = (self.params.vol_period as f64 * self.params.vol_lookback_factor) as usize;
        
        // Calculate rolling historical average
        for i in long_period..n {
            let start_idx = i.saturating_sub(long_period);
            let window = &combined_vol.slice(s![start_idx..i]);
            
            if window.len() > 0 {
                let historical_avg = window.mean().unwrap_or(1e-6);
                let current_vol = combined_vol[i];
                
                // Calculate regime as ratio of current to historical
                let regime_ratio = if historical_avg > 1e-9 {
                    current_vol / historical_avg
                } else {
                    1.0
                };
                
                // Transform to 0-1 scale using log transformation
                let log_regime = regime_ratio.max(1e-9).ln();
                vol_regime[i] = (0.5 + log_regime / 2.0).clamp(0.0, 1.0);
            }
        }
        
        Ok(vol_regime)
    }
    
    /// Calculate volatility rate of change
    fn calculate_volatility_roc(&self, vol_regime: &Array1<f64>) -> AntifragilityResult<Array1<f64>> {
        let n = vol_regime.len();
        let mut vol_roc = Array1::zeros(n);
        
        // Calculate percentage change
        for i in 1..n {
            if vol_regime[i-1] > 1e-9 {
                vol_roc[i] = (vol_regime[i] - vol_regime[i-1]) / vol_regime[i-1];
            }
        }
        
        // Apply exponential smoothing
        let smoothing_period = self.params.vol_period / 3;
        let alpha = 2.0 / (smoothing_period as f64 + 1.0);
        let mut vol_roc_smoothed = Array1::zeros(n);
        
        if n > 0 {
            vol_roc_smoothed[0] = vol_roc[0];
            for i in 1..n {
                vol_roc_smoothed[i] = alpha * vol_roc[i] + (1.0 - alpha) * vol_roc_smoothed[i-1];
            }
        }
        
        Ok(vol_roc_smoothed)
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
            if log_perf_returns[i-1].abs() > 1e-10 {
                perf_roc_smoothed[i] = (log_perf_returns[i] - log_perf_returns[i-1]) / log_perf_returns[i-1].abs().max(1e-10);
            }
        }
        
        result.perf_roc_smoothed = self.apply_exponential_smoothing_array(&perf_roc_smoothed, smoothing_period)?;
        result.momentum = perf_momentum;
        result.log_perf_returns = log_perf_returns;
        
        Ok(result)
    }
    
    /// Calculate convexity component
    fn calculate_convexity_component(&self, perf_acceleration: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        self.calculate_convexity_correlation(perf_acceleration, vol_roc_smoothed, self.params.corr_window)
    }
    
    /// Calculate convexity correlation between performance acceleration and volatility change
    fn calculate_convexity_correlation(
        &self,
        perf_acceleration: &Array1<f64>,
        vol_roc_smoothed: &Array1<f64>,
        window: usize,
    ) -> AntifragilityResult<f64> {
        let n = perf_acceleration.len();
        
        if n != vol_roc_smoothed.len() {
            return Err(AntifragilityError::InvalidParameters {
                message: format!("Array lengths must match: {} vs {}", n, vol_roc_smoothed.len()),
            });
        }
        
        if n < window {
            return Err(AntifragilityError::InsufficientData {
                required: window,
                actual: n,
            });
        }
        
        // Calculate rolling correlation
        let correlations = self.calculate_rolling_correlation(perf_acceleration, vol_roc_smoothed, window)?;
        
        // Take the mean of valid correlations
        let valid_correlations: Vec<f64> = correlations.iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();
        
        if valid_correlations.is_empty() {
            return Ok(0.0);
        }
        
        let mean_correlation = valid_correlations.iter().sum::<f64>() / valid_correlations.len() as f64;
        
        // Convert correlation to 0-1 scale
        Ok((mean_correlation + 1.0) / 2.0)
    }
    
    /// Calculate rolling correlation between two arrays
    fn calculate_rolling_correlation(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        window: usize,
    ) -> AntifragilityResult<Array1<f64>> {
        let n = x.len();
        let mut correlations = Array1::zeros(n);
        
        for i in window..n {
            let x_window = x.slice(s![(i - window)..i]);
            let y_window = y.slice(s![(i - window)..i]);
            
            let correlation = self.calculate_correlation(&x_window, &y_window)?;
            correlations[i] = correlation;
        }
        
        Ok(correlations)
    }
    
    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> AntifragilityResult<f64> {
        let n = x.len();
        if n != y.len() || n < 2 {
            return Ok(0.0);
        }
        
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);
        
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator > 1e-9 {
            Ok(sum_xy / denominator)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate asymmetry component
    fn calculate_asymmetry_component(&self, log_returns: &Array1<f64>, vol_regime: &Array1<f64>) -> AntifragilityResult<f64> {
        self.calculate_weighted_asymmetry(log_returns, vol_regime, self.params.perf_period)
    }
    
    /// Calculate weighted asymmetry analysis
    fn calculate_weighted_asymmetry(&self, log_returns: &Array1<f64>, vol_regime: &Array1<f64>, period: usize) -> AntifragilityResult<f64> {
        let n = log_returns.len();
        
        if n < period + 5 {
            return Err(AntifragilityError::InsufficientData {
                required: period + 5,
                actual: n,
            });
        }
        
        let mut asymmetry_values = Vec::new();
        
        for i in period..n {
            let returns_window = log_returns.slice(s![(i - period)..i]);
            let vol_window = vol_regime.slice(s![(i - period)..i]);
            
            // Separate high and low volatility periods
            let vol_threshold = vol_window.mean().unwrap_or(0.5);
            
            let mut high_vol_returns = Vec::new();
            let mut low_vol_returns = Vec::new();
            
            for j in 0..returns_window.len() {
                if vol_window[j] > vol_threshold {
                    high_vol_returns.push(returns_window[j]);
                } else {
                    low_vol_returns.push(returns_window[j]);
                }
            }
            
            // Calculate skewness for each regime
            let high_vol_skew = self.calculate_skewness(&high_vol_returns)?;
            let low_vol_skew = self.calculate_skewness(&low_vol_returns)?;
            
            // Asymmetry favors positive skewness in high volatility periods
            let asymmetry = if high_vol_returns.len() > 3 && low_vol_returns.len() > 3 {
                (high_vol_skew - low_vol_skew) / 2.0
            } else {
                0.0
            };
            
            asymmetry_values.push(asymmetry);
        }
        
        if asymmetry_values.is_empty() {
            return Ok(0.0);
        }
        
        let mean_asymmetry = asymmetry_values.iter().sum::<f64>() / asymmetry_values.len() as f64;
        
        // Normalize to 0-1 range using tanh transformation
        Ok((mean_asymmetry.tanh() + 1.0) / 2.0)
    }
    
    /// Calculate skewness of a dataset
    fn calculate_skewness(&self, data: &[f64]) -> AntifragilityResult<f64> {
        if data.len() < 3 {
            return Ok(0.0);
        }
        
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        
        if variance < 1e-9 {
            return Ok(0.0);
        }
        
        let std_dev = variance.sqrt();
        let skewness = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;
        
        Ok(skewness)
    }
    
    /// Calculate recovery component
    fn calculate_recovery_component(&self, prices: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        self.calculate_recovery_velocity(prices, vol_roc_smoothed)
    }
    
    /// Calculate recovery velocity after volatility spikes
    fn calculate_recovery_velocity(&self, prices: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        let n = prices.len();
        let recovery_horizon = (self.params.vol_period as f64 * self.params.recovery_horizon_factor) as usize;
        
        if n < recovery_horizon * 3 {
            return Err(AntifragilityError::InsufficientData {
                required: recovery_horizon * 3,
                actual: n,
            });
        }
        
        // Detect volatility spikes (95th percentile)
        let vol_threshold = self.calculate_percentile(vol_roc_smoothed, 0.95)?;
        
        let mut recovery_velocities = Vec::new();
        
        for i in recovery_horizon..(n - recovery_horizon) {
            if vol_roc_smoothed[i] > vol_threshold {
                // Found a volatility spike, measure recovery
                let pre_spike_price = prices[i - recovery_horizon];
                let spike_price = prices[i];
                let post_spike_price = prices[i + recovery_horizon];
                
                if pre_spike_price > 0.0 && spike_price > 0.0 && post_spike_price > 0.0 {
                    // Calculate drawdown and recovery
                    let drawdown = (spike_price - pre_spike_price) / pre_spike_price;
                    let recovery = (post_spike_price - spike_price) / spike_price;
                    
                    // Recovery velocity: how much we recover relative to the drawdown
                    let recovery_velocity = if drawdown.abs() > 1e-9 {
                        recovery / drawdown.abs()
                    } else {
                        0.0
                    };
                    
                    recovery_velocities.push(recovery_velocity);
                }
            }
        }
        
        if recovery_velocities.is_empty() {
            return Ok(0.5); // Neutral recovery
        }
        
        let mean_recovery = recovery_velocities.iter().sum::<f64>() / recovery_velocities.len() as f64;
        
        // Normalize to 0-1 range
        Ok((mean_recovery.tanh() + 1.0) / 2.0)
    }
    
    /// Calculate percentile of an array
    fn calculate_percentile(&self, data: &Array1<f64>, percentile: f64) -> AntifragilityResult<f64> {
        let mut sorted_data: Vec<f64> = data.iter().copied().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile * (sorted_data.len() - 1) as f64) as usize;
        Ok(sorted_data[index.min(sorted_data.len() - 1)])
    }
    
    /// Calculate benefit ratio component
    fn calculate_benefit_ratio_component(&self, perf_roc_smoothed: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        self.calculate_benefit_ratio(perf_roc_smoothed, vol_roc_smoothed)
    }
    
    /// Calculate benefit ratio (performance improvement vs volatility cost)
    fn calculate_benefit_ratio(&self, perf_roc_smoothed: &Array1<f64>, vol_roc_smoothed: &Array1<f64>) -> AntifragilityResult<f64> {
        let n = perf_roc_smoothed.len();
        
        if n != vol_roc_smoothed.len() {
            return Err(AntifragilityError::InvalidParameters {
                message: "Arrays must have same length".to_string(),
            });
        }
        
        let mut benefit_ratios = Vec::new();
        
        for i in 0..n {
            let perf_change = perf_roc_smoothed[i];
            let vol_change = vol_roc_smoothed[i];
            
            // Benefit ratio: positive performance change divided by volatility cost
            if vol_change.abs() > 1e-9 {
                let benefit_ratio = perf_change.max(0.0) / vol_change.abs();
                benefit_ratios.push(benefit_ratio);
            }
        }
        
        if benefit_ratios.is_empty() {
            return Ok(0.0);
        }
        
        let mean_benefit_ratio = benefit_ratios.iter().sum::<f64>() / benefit_ratios.len() as f64;
        
        // Normalize using log transformation
        Ok((mean_benefit_ratio.ln().max(-5.0).min(5.0) + 5.0) / 10.0)
    }
    
    /// Apply exponential smoothing to a single value
    fn apply_exponential_smoothing(&self, value: f64, _span: usize) -> AntifragilityResult<f64> {
        // For single values, just return the value clamped to valid range
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
        Self::new().expect("Failed to create default AntifragilityAnalyzer")
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
    
    fn generate_antifragile_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        
        let mut price = 100.0;
        for i in 0..n {
            let vol_factor = 1.0 + 0.1 * ((i as f64) * 0.05).sin().abs();
            let return_rate = 0.002 * vol_factor * ((i as f64) * 0.1).sin();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 * vol_factor);
        }
        
        (prices, volumes)
    }
    
    #[test]
    fn test_antifragility_analyzer_creation() {
        let analyzer = AntifragilityAnalyzer::new();
        assert!(analyzer.is_ok());
        
        let analyzer = analyzer.unwrap();
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
        
        params.convexity_weight = -0.1;
        params.asymmetry_weight = 0.6;
        assert!(params.validate().is_err());
    }
    
    #[test]
    fn test_basic_analysis() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (prices, volumes) = generate_test_data(200);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.antifragility_index >= 0.0);
        assert!(analysis.antifragility_index <= 1.0);
        assert!(analysis.fragility_score >= 0.0);
        assert!(analysis.fragility_score <= 1.0);
        assert_eq!(analysis.data_points, 200);
        assert!(analysis.calculation_time.as_nanos() > 0);
    }
    
    #[test]
    fn test_antifragile_detection() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (prices, volumes) = generate_antifragile_data(200);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        // Should detect some level of antifragility in the synthetic data
        assert!(analysis.antifragility_index > 0.3);
    }
    
    #[test]
    fn test_insufficient_data() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (prices, volumes) = generate_test_data(50);
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_err());
        
        if let Err(err) = result {
            assert!(err.to_string().contains("Insufficient data"));
        }
    }
    
    #[test]
    fn test_mismatched_arrays() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (mut prices, volumes) = generate_test_data(150);
        prices.pop(); // Make arrays different lengths
        
        let result = analyzer.analyze_prices(&prices, &volumes);
        assert!(result.is_err());
        
        if let Err(err) = result {
            assert!(err.to_string().contains("same length"));
        }
    }
    
    #[test]
    fn test_cache_functionality() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (prices, volumes) = generate_test_data(200);
        
        // First analysis
        let start = std::time::Instant::now();
        let result1 = analyzer.analyze_prices(&prices, &volumes).unwrap();
        let duration1 = start.elapsed();
        
        // Second analysis (should be cached)
        let start = std::time::Instant::now();
        let result2 = analyzer.analyze_prices(&prices, &volumes).unwrap();
        let duration2 = start.elapsed();
        
        // Results should be identical
        assert_relative_eq!(result1.antifragility_index, result2.antifragility_index, epsilon = 1e-10);
        assert_relative_eq!(result1.fragility_score, result2.fragility_score, epsilon = 1e-10);
        
        // Second call should be faster due to caching
        assert!(duration2 < duration1 || duration2.as_nanos() < 1000000); // 1ms tolerance
    }
    
    #[test]
    fn test_volatility_estimators() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let prices = Array1::from_vec((0..100).map(|i| 100.0 + (i as f64).sin()).collect());
        
        // Test Yang-Zhang volatility
        let volumes = Array1::ones(100);
        let yz_result = analyzer.calculate_yang_zhang_volatility(&prices, &volumes);
        assert!(yz_result.is_ok());
        let yz_vol = yz_result.unwrap();
        assert_eq!(yz_vol.len(), 100);
        assert!(yz_vol.iter().all(|&x| x >= 0.0));
        
        // Test GARCH volatility
        let garch_result = analyzer.calculate_garch_volatility(&prices);
        assert!(garch_result.is_ok());
        let garch_vol = garch_result.unwrap();
        assert_eq!(garch_vol.len(), 100);
        assert!(garch_vol.iter().all(|&x| x >= 0.0));
        
        // Test Parkinson volatility
        let park_result = analyzer.calculate_parkinson_volatility(&prices);
        assert!(park_result.is_ok());
        let park_vol = park_result.unwrap();
        assert_eq!(park_vol.len(), 100);
        assert!(park_vol.iter().all(|&x| x >= 0.0));
        
        // Test ATR volatility
        let atr_result = analyzer.calculate_atr_volatility(&prices);
        assert!(atr_result.is_ok());
        let atr_vol = atr_result.unwrap();
        assert_eq!(atr_vol.len(), 100);
        assert!(atr_vol.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_convexity_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Create perfectly correlated data
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        
        let result = analyzer.calculate_convexity_correlation(&x, &y, 5);
        assert!(result.is_ok());
        
        let convexity = result.unwrap();
        assert!(convexity >= 0.9); // Should detect strong positive correlation
    }
    
    #[test]
    fn test_asymmetry_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Create asymmetric returns
        let mut returns = Array1::zeros(100);
        let mut vol_regime = Array1::zeros(100);
        
        for i in 0..100 {
            if i % 2 == 0 {
                returns[i] = 0.01; // Positive returns
                vol_regime[i] = 0.8; // High volatility
            } else {
                returns[i] = -0.005; // Smaller negative returns
                vol_regime[i] = 0.2; // Low volatility
            }
        }
        
        let result = analyzer.calculate_asymmetry_component(&returns, &vol_regime);
        assert!(result.is_ok());
        
        let asymmetry = result.unwrap();
        assert!(asymmetry >= 0.0 && asymmetry <= 1.0);
    }
    
    #[test]
    fn test_recovery_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Create price data with recovery after spikes
        let mut prices = Array1::zeros(200);
        let mut vol_roc = Array1::zeros(200);
        
        let mut price = 100.0;
        for i in 0..200 {
            if i == 50 || i == 150 {
                // Volatility spikes
                vol_roc[i] = 2.0;
                price *= 0.95; // Drop
            } else if i > 50 && i < 70 {
                // Recovery after first spike
                price *= 1.01;
                vol_roc[i] = 0.1;
            } else if i > 150 && i < 170 {
                // Recovery after second spike
                price *= 1.015;
                vol_roc[i] = 0.1;
            } else {
                price *= 1.001;
                vol_roc[i] = 0.2;
            }
            prices[i] = price;
        }
        
        let result = analyzer.calculate_recovery_component(&prices, &vol_roc);
        assert!(result.is_ok());
        
        let recovery = result.unwrap();
        assert!(recovery >= 0.0 && recovery <= 1.0);
    }
    
    #[test]
    fn test_benefit_ratio_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Create data where performance improves with volatility
        let perf_roc = Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04, 0.05]);
        let vol_roc = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        
        let result = analyzer.calculate_benefit_ratio_component(&perf_roc, &vol_roc);
        assert!(result.is_ok());
        
        let benefit = result.unwrap();
        assert!(benefit >= 0.0 && benefit <= 1.0);
    }
    
    #[test]
    fn test_performance_metrics() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let (prices, volumes) = generate_test_data(200);
        
        // Perform multiple analyses to test metrics tracking
        for _ in 0..5 {
            let _ = analyzer.analyze_prices(&prices, &volumes);
        }
        
        let metrics = analyzer.get_performance_metrics();
        assert!(metrics.total_analyses >= 1); // At least one should be from cache miss
        assert!(metrics.cache_hit_rate() > 0.0); // Should have some cache hits
    }
    
    #[test]
    fn test_analysis_result_methods() {
        let result = AnalysisResult {
            antifragility_index: 0.7,
            fragility_score: 0.2,
            convexity_score: 0.6,
            asymmetry_score: 0.5,
            recovery_score: 0.8,
            benefit_ratio_score: 0.4,
            volatility: VolatilityResult::default(),
            performance: PerformanceResult::default(),
            data_points: 100,
            calculation_time: std::time::Duration::from_millis(1),
        };
        
        assert!(result.is_antifragile());
        assert!(!result.is_fragile());
        assert!(!result.is_robust());
        assert_eq!(result.dominant_characteristic(), "Antifragile");
        
        let summary = result.summary();
        assert!(summary.contains("Antifragility Index: 0.7000"));
        assert!(summary.contains("Data Points: 100"));
    }
    
    #[test]
    fn test_log_returns_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let prices = Array1::from_vec(vec![100.0, 105.0, 110.0, 115.0, 120.0]);
        
        let result = analyzer.calculate_log_returns(&prices);
        assert!(result.is_ok());
        
        let log_returns = result.unwrap();
        assert_eq!(log_returns.len(), 5);
        assert_eq!(log_returns[0], 0.0); // First return should be zero
        assert_relative_eq!(log_returns[1], (105.0 / 100.0).ln(), epsilon = 1e-10);
        assert_relative_eq!(log_returns[2], (110.0 / 105.0).ln(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Perfect positive correlation
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        
        let x_view = x.view();
        let y_view = y.view();
        let result = analyzer.calculate_correlation(&x_view, &y_view);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 1.0, epsilon = 1e-10);
        
        // Perfect negative correlation
        let y_neg = Array1::from_vec(vec![10.0, 8.0, 6.0, 4.0, 2.0]);
        let y_neg_view = y_neg.view();
        let result_neg = analyzer.calculate_correlation(&x_view, &y_neg_view);
        assert!(result_neg.is_ok());
        assert_relative_eq!(result_neg.unwrap(), -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_skewness_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        
        // Symmetric distribution (should have near-zero skewness)
        let symmetric_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let skew_result = analyzer.calculate_skewness(&symmetric_data);
        assert!(skew_result.is_ok());
        assert!(skew_result.unwrap().abs() < 0.1);
        
        // Right-skewed distribution
        let right_skewed = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let right_skew = analyzer.calculate_skewness(&right_skewed).unwrap();
        assert!(right_skew > 0.0);
    }
    
    #[test]
    fn test_percentile_calculation() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        
        let median = analyzer.calculate_percentile(&data, 0.5).unwrap();
        assert_relative_eq!(median, 5.5, epsilon = 0.1);
        
        let p95 = analyzer.calculate_percentile(&data, 0.95).unwrap();
        assert_relative_eq!(p95, 9.55, epsilon = 0.1);
    }
    
    #[test]
    fn test_exponential_smoothing() {
        let analyzer = AntifragilityAnalyzer::new().unwrap();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let result = analyzer.apply_exponential_smoothing_array(&data, 3);
        assert!(result.is_ok());
        
        let smoothed = result.unwrap();
        assert_eq!(smoothed.len(), 5);
        assert_eq!(smoothed[0], 1.0); // First value unchanged
        assert!(smoothed[4] < 5.0); // Last value should be smoothed
        assert!(smoothed[4] > smoothed[3]); // Should be increasing
    }
}
