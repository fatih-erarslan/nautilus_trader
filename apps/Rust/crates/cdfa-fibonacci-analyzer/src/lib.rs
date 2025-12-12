//! # CDFA Fibonacci Analyzer
//!
//! High-performance Fibonacci analysis with sub-microsecond precision for CDFA.
//! This crate provides optimized implementations of Fibonacci retracement and extension
//! calculations, swing point detection, and alignment scoring with SIMD acceleration.
//!
//! ## Features
//! - Sub-microsecond performance using SIMD optimization
//! - Fibonacci retracement levels: 0.0%, 23.6%, 38.2%, 50.0%, 61.8%, 78.6%, 100.0%
//! - Extension levels: 100.0%, 127.2%, 161.8%, 261.8%, 361.8%
//! - ATR-based volatility bands
//! - Swing point detection with configurable periods
//! - Alignment scoring with tolerance-based proximity
//! - Multi-timeframe confluence analysis
//! - Hardware acceleration compatibility
//!
//! ## Example
//! ```
//! use cdfa_fibonacci_analyzer::{FibonacciAnalyzer, FibonacciConfig};
//!
//! let config = FibonacciConfig::default();
//! let analyzer = FibonacciAnalyzer::new(config);
//! 
//! let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0];
//! let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0];
//! 
//! let result = analyzer.analyze(&prices, &volumes);
//! println!("Fibonacci signal: {}", result.signal);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{debug, error, info, warn};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "simd")]
use wide::f64x4;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Re-export core types
pub use cdfa_core::{AnalyzerResult, Signal, Confidence};

mod config;
mod core;
mod simd;
mod swing_detection;
mod alignment;
mod extensions;
mod volatility;
mod utils;

pub use config::*;
pub use core::*;
pub use simd::*;
pub use swing_detection::*;
pub use alignment::*;
pub use extensions::*;
pub use volatility::*;
pub use utils::*;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "c-ffi")]
pub mod ffi;

/// Main error type for the Fibonacci analyzer
#[derive(Error, Debug)]
pub enum FibonacciError {
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    
    #[error("Calculation error: {0}")]
    CalculationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("SIMD operation error: {0}")]
    SimdError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for Fibonacci operations
pub type FibonacciResult<T> = Result<T, FibonacciError>;

/// Main Fibonacci analyzer struct
#[derive(Debug)]
pub struct FibonacciAnalyzer {
    config: FibonacciConfig,
    swing_detector: SwingPointDetector,
    alignment_scorer: AlignmentScorer,
    extension_calculator: ExtensionCalculator,
    volatility_analyzer: VolatilityAnalyzer,
    cache: Arc<std::sync::RwLock<AnalysisCache>>,
}

impl FibonacciAnalyzer {
    /// Create a new Fibonacci analyzer with the given configuration
    pub fn new(config: FibonacciConfig) -> Self {
        let swing_detector = SwingPointDetector::new(config.swing_period);
        let alignment_scorer = AlignmentScorer::new(config.alignment_tolerance);
        let extension_calculator = ExtensionCalculator::new();
        let volatility_analyzer = VolatilityAnalyzer::new(config.atr_period);
        let cache = Arc::new(std::sync::RwLock::new(AnalysisCache::new()));
        
        Self {
            config,
            swing_detector,
            alignment_scorer,
            extension_calculator,
            volatility_analyzer,
            cache,
        }
    }
    
    /// Create a new analyzer with default configuration
    pub fn default() -> Self {
        Self::new(FibonacciConfig::default())
    }
    
    /// Analyze price and volume data to generate Fibonacci-based signals
    pub fn analyze(&self, prices: &[f64], volumes: &[f64]) -> FibonacciResult<AnalyzerResult> {
        let start_time = Instant::now();
        
        // Input validation
        if prices.is_empty() || volumes.is_empty() {
            return Err(FibonacciError::InvalidInput("Empty price or volume data".to_string()));
        }
        
        if prices.len() != volumes.len() {
            return Err(FibonacciError::InvalidInput("Price and volume arrays must have same length".to_string()));
        }
        
        // Check for invalid values
        for (i, &price) in prices.iter().enumerate() {
            if !price.is_finite() || price <= 0.0 {
                return Err(FibonacciError::InvalidInput(format!("Invalid price at index {}: {}", i, price)));
            }
        }
        
        debug!("Starting Fibonacci analysis for {} data points", prices.len());
        
        // Detect swing points
        let swing_points = self.swing_detector.detect_swings(prices)?;
        debug!("Detected {} swing highs and {} swing lows", 
               swing_points.swing_highs.len(), swing_points.swing_lows.len());
        
        // Calculate retracement levels
        let retracements = self.calculate_retracements(prices, &swing_points)?;
        debug!("Calculated {} retracement levels", retracements.levels.len());
        
        // Calculate extension levels
        let extensions = self.extension_calculator.calculate_extensions(prices, &swing_points)?;
        debug!("Calculated {} extension levels", extensions.levels.len());
        
        // Calculate alignment score
        let alignment_score = self.alignment_scorer.calculate_alignment(
            prices.last().unwrap_or(&0.0),
            &retracements,
            &extensions,
        )?;
        debug!("Calculated alignment score: {:.4}", alignment_score);
        
        // Calculate volatility-based bands
        let volatility_bands = self.volatility_analyzer.calculate_bands(prices, volumes)?;
        debug!("Calculated volatility bands with {} levels", volatility_bands.bands.len());
        
        // Generate final signal
        let signal = self.generate_signal(alignment_score, &retracements, &extensions, &volatility_bands)?;
        let confidence = self.calculate_confidence(&retracements, &extensions, alignment_score)?;
        
        let analysis_time = start_time.elapsed();
        debug!("Fibonacci analysis completed in {:?}", analysis_time);
        
        // Store result in cache
        let cache_key = self.generate_cache_key(prices, volumes);
        if let Ok(mut cache) = self.cache.write() {
            cache.store(cache_key, signal, confidence, alignment_score);
        }
        
        Ok(AnalyzerResult {
            signal,
            confidence,
            metadata: HashMap::from([
                ("analysis_type".to_string(), "fibonacci".to_string()),
                ("data_points".to_string(), prices.len().to_string()),
                ("current_price".to_string(), prices.last().unwrap_or(&0.0).to_string()),
                ("fibonacci_alignment".to_string(), alignment_score.to_string()),
                ("swing_highs".to_string(), swing_points.swing_highs.len().to_string()),
                ("swing_lows".to_string(), swing_points.swing_lows.len().to_string()),
                ("analysis_time_us".to_string(), analysis_time.as_micros().to_string()),
            ]),
        })
    }
    
    /// Calculate Fibonacci retracement levels based on swing points
    fn calculate_retracements(&self, prices: &[f64], swing_points: &SwingPoints) -> FibonacciResult<RetracementLevels> {
        if swing_points.swing_highs.is_empty() || swing_points.swing_lows.is_empty() {
            return Ok(RetracementLevels::empty());
        }
        
        // Find the most recent swing high and low
        let recent_high = swing_points.swing_highs.last().unwrap();
        let recent_low = swing_points.swing_lows.last().unwrap();
        
        let high_price = prices[recent_high.index];
        let low_price = prices[recent_low.index];
        
        // Determine trend direction
        let trend = if recent_high.index > recent_low.index {
            TrendDirection::Up
        } else {
            TrendDirection::Down
        };
        
        let price_range = high_price - low_price;
        if price_range.abs() < f64::EPSILON {
            return Ok(RetracementLevels::empty());
        }
        
        let mut levels = HashMap::new();
        
        // Calculate retracement levels
        for (&level_name, &level_ratio) in &self.config.retracement_levels {
            let level_price = match trend {
                TrendDirection::Up => high_price - (level_ratio * price_range),
                TrendDirection::Down => low_price + (level_ratio * price_range),
            };
            
            levels.insert(level_name.clone(), level_price);
        }
        
        Ok(RetracementLevels {
            levels,
            trend,
            high_price,
            low_price,
            price_range,
        })
    }
    
    /// Generate trading signal based on Fibonacci analysis
    fn generate_signal(
        &self,
        alignment_score: f64,
        retracements: &RetracementLevels,
        extensions: &ExtensionLevels,
        volatility_bands: &VolatilityBands,
    ) -> FibonacciResult<Signal> {
        // Base signal from alignment score
        let base_signal = alignment_score;
        
        // Adjust signal based on trend direction
        let trend_adjustment = match retracements.trend {
            TrendDirection::Up => 0.1,
            TrendDirection::Down => -0.1,
        };
        
        // Volatility adjustment
        let volatility_adjustment = volatility_bands.normalized_volatility * 0.05;
        
        // Combine signals
        let combined_signal = (base_signal + trend_adjustment + volatility_adjustment).clamp(-1.0, 1.0);
        
        Ok(Signal::new(combined_signal))
    }
    
    /// Calculate confidence based on multiple factors
    fn calculate_confidence(
        &self,
        retracements: &RetracementLevels,
        extensions: &ExtensionLevels,
        alignment_score: f64,
    ) -> FibonacciResult<Confidence> {
        let mut confidence_factors = Vec::new();
        
        // Base confidence from alignment score
        confidence_factors.push(alignment_score);
        
        // Confidence from number of levels
        let level_count = retracements.levels.len() + extensions.levels.len();
        let level_confidence = (level_count as f64 / 12.0).min(1.0); // Max 12 levels total
        confidence_factors.push(level_confidence);
        
        // Confidence from price range (larger ranges = higher confidence)
        let range_confidence = (retracements.price_range / retracements.high_price).min(1.0);
        confidence_factors.push(range_confidence);
        
        // Calculate weighted average
        let total_confidence = confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64;
        
        Ok(Confidence::new(total_confidence))
    }
    
    /// Generate cache key for results
    fn generate_cache_key(&self, prices: &[f64], volumes: &[f64]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash last few prices and volumes for cache key
        let n = prices.len().min(10);
        for i in (prices.len() - n)..prices.len() {
            prices[i].to_bits().hash(&mut hasher);
            volumes[i].to_bits().hash(&mut hasher);
        }
        
        format!("fib_{:x}", hasher.finish())
    }
    
    /// Get configuration
    pub fn config(&self) -> &FibonacciConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: FibonacciConfig) {
        self.config = config;
        self.swing_detector.update_period(config.swing_period);
        self.alignment_scorer.update_tolerance(config.alignment_tolerance);
        self.volatility_analyzer.update_period(config.atr_period);
    }
    
    /// Clear analysis cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.read().ok().map(|cache| cache.stats())
    }
}

/// Internal cache for analysis results
#[derive(Debug)]
struct AnalysisCache {
    results: HashMap<String, CachedResult>,
    max_size: usize,
    hits: usize,
    misses: usize,
}

#[derive(Debug, Clone)]
struct CachedResult {
    signal: Signal,
    confidence: Confidence,
    alignment_score: f64,
    timestamp: Instant,
}

impl AnalysisCache {
    fn new() -> Self {
        Self {
            results: HashMap::new(),
            max_size: 1000,
            hits: 0,
            misses: 0,
        }
    }
    
    fn store(&mut self, key: String, signal: Signal, confidence: Confidence, alignment_score: f64) {
        if self.results.len() >= self.max_size {
            // Remove oldest entry
            if let Some(oldest_key) = self.results.keys().next().cloned() {
                self.results.remove(&oldest_key);
            }
        }
        
        self.results.insert(key, CachedResult {
            signal,
            confidence,
            alignment_score,
            timestamp: Instant::now(),
        });
    }
    
    fn get(&mut self, key: &str) -> Option<&CachedResult> {
        if let Some(result) = self.results.get(key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }
    
    fn clear(&mut self) {
        self.results.clear();
    }
    
    fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.results.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_fibonacci_analyzer_creation() {
        let config = FibonacciConfig::default();
        let analyzer = FibonacciAnalyzer::new(config);
        
        assert_eq!(analyzer.config().swing_period, 14);
        assert_relative_eq!(analyzer.config().alignment_tolerance, 0.006);
    }
    
    #[test]
    fn test_fibonacci_analysis() {
        let config = FibonacciConfig::default();
        let analyzer = FibonacciAnalyzer::new(config);
        
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 120.0, 85.0, 125.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0, 1800.0, 700.0, 2000.0];
        
        let result = analyzer.analyze(&prices, &volumes);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.signal.value() >= -1.0 && result.signal.value() <= 1.0);
        assert!(result.confidence.value() >= 0.0 && result.confidence.value() <= 1.0);
    }
    
    #[test]
    fn test_invalid_input() {
        let analyzer = FibonacciAnalyzer::default();
        
        // Empty arrays
        let result = analyzer.analyze(&[], &[]);
        assert!(result.is_err());
        
        // Mismatched lengths
        let result = analyzer.analyze(&[100.0], &[]);
        assert!(result.is_err());
        
        // Invalid prices
        let result = analyzer.analyze(&[0.0], &[100.0]);
        assert!(result.is_err());
        
        let result = analyzer.analyze(&[f64::NAN], &[100.0]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_cache_functionality() {
        let analyzer = FibonacciAnalyzer::default();
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0];
        
        // First analysis
        let _result1 = analyzer.analyze(&prices, &volumes).unwrap();
        
        // Check cache stats
        let stats = analyzer.cache_stats().unwrap();
        assert_eq!(stats.size, 1);
        
        // Clear cache
        analyzer.clear_cache();
        let stats = analyzer.cache_stats().unwrap();
        assert_eq!(stats.size, 0);
    }
}