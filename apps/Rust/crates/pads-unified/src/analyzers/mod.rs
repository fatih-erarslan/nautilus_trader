//! # PADS Unified Analyzer Module
//!
//! This module contains sophisticated pattern analyzers harvested from CDFA crates,
//! providing unified access to advanced market analysis capabilities.
//!
//! ## Harvested Analyzers
//!
//! ### CDFA Antifragility Analyzer
//! - Convexity measurement (correlation between performance acceleration and volatility)
//! - Asymmetry analysis (skewness and kurtosis under stress)
//! - Recovery velocity (performance after volatility spikes)
//! - Benefit ratio (performance improvement vs volatility cost)
//! - Multiple volatility estimators (Yang-Zhang, GARCH, Parkinson, ATR)
//!
//! ### CDFA SOC (Self-Organized Criticality) Analyzer
//! - Sample entropy and entropy rate calculation
//! - Avalanche event detection and classification
//! - Regime classification (sub-critical, critical, super-critical)
//! - Equilibrium and fragility score computation
//! - Power-law analysis
//!
//! ### CDFA Panarchy Analyzer
//! - Four-phase adaptive cycle model (r, K, Ω, α)
//! - PCR (Potential, Connectedness, Resilience) component analysis
//! - Phase identification with hysteresis
//! - Market regime detection
//!
//! ### CDFA Fibonacci Analyzer
//! - Fibonacci retracement and extension levels
//! - Swing point detection with configurable periods
//! - Alignment scoring with tolerance-based proximity
//! - Multi-timeframe confluence analysis
//! - ATR-based volatility bands
//!
//! ## Performance Features
//! - Sub-microsecond analysis capabilities
//! - SIMD optimization using AVX2/AVX512/NEON
//! - Parallel processing support
//! - Memory-efficient algorithms
//! - Comprehensive caching system
//!
//! ## Usage
//! ```rust
//! use pads_unified::analyzers::{AnalyzerManager, AnalyzerType};
//!
//! let mut manager = AnalyzerManager::new().await?;
//! let market_data = MarketData::new(prices, volumes);
//! 
//! // Run specific analyzer
//! let antifragility = manager.analyze(AnalyzerType::Antifragility, &market_data).await?;
//! 
//! // Run all analyzers
//! let results = manager.analyze_all(&market_data).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use ndarray::Array1;

use crate::types::{MarketState, FactorValues};
use crate::error::{PadsError, PadsResult};

// Sub-modules for individual analyzers
pub mod antifragility;
pub mod soc;
pub mod panarchy;
pub mod fibonacci;
pub mod black_swan;
pub mod narrative_forecaster;

// Advanced analyzers from standalone files
pub mod adaptive_cycles;
pub mod antifragility_integration;
pub mod cross_scale;
pub mod narrative_forecasting;
pub mod phase_management;
pub mod regime_detection;
pub mod sentiment_analysis;

// Utility modules
pub mod manager;
pub mod factory;
pub mod simd_utils;
pub mod cache;
pub mod aggregation;

// Re-exports
pub use manager::AnalyzerManager;
pub use factory::AnalyzerFactory;
pub use antifragility::AntifragilityAnalyzer;
pub use soc::SOCAnalyzer;
pub use panarchy::PanarchyAnalyzer;
pub use fibonacci::FibonacciAnalyzer;
pub use black_swan::BlackSwanDetector;
pub use narrative_forecaster::NarrativeForecaster;

/// Unified analyzer result containing all pattern analysis outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Antifragility analysis
    pub antifragility: Option<antifragility::AntifragilityResult>,
    
    /// Self-Organized Criticality analysis
    pub soc: Option<soc::SOCResult>,
    
    /// Panarchy cycle analysis
    pub panarchy: Option<panarchy::PanarchyAnalysisResult>,
    
    /// Fibonacci analysis
    pub fibonacci: Option<fibonacci::FibonacciResult>,
    
    /// Black swan detection
    pub black_swan: Option<black_swan::DetectionResult>,
    
    /// Narrative forecasting
    pub narrative_forecast: Option<narrative_forecaster::NarrativeForecast>,
    
    /// Aggregated signals and scores
    pub aggregated: AggregatedSignals,
    
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Aggregated signals from all analyzers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSignals {
    /// Combined signal strength (-1.0 to 1.0)
    pub signal: f64,
    
    /// Combined confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Risk assessment score (0.0 to 1.0)
    pub risk_score: f64,
    
    /// Market regime classification
    pub regime: MarketRegime,
    
    /// Pattern strength indicators
    pub pattern_strength: HashMap<String, f64>,
}

/// Market regime classification from pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MarketRegime {
    /// Stable growth phase
    Growth,
    /// Conservation/consolidation phase
    Conservation,
    /// Release/breakdown phase
    Release,
    /// Reorganization/recovery phase
    Reorganization,
    /// Critical transition state
    Critical,
    /// Unknown/insufficient data
    Unknown,
}

/// Analysis metadata and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Total analysis time
    pub total_time: Duration,
    
    /// Individual analyzer timings
    pub analyzer_timings: HashMap<String, Duration>,
    
    /// Data points analyzed
    pub data_points: usize,
    
    /// Enabled analyzers
    pub enabled_analyzers: Vec<String>,
    
    /// SIMD optimization used
    pub simd_enabled: bool,
    
    /// Parallel processing used
    pub parallel_enabled: bool,
    
    /// Cache hits/misses
    pub cache_stats: CacheStatistics,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
    pub cache_size: usize,
}

/// Configuration for analyzer system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable/disable individual analyzers
    pub enabled_analyzers: HashMap<String, bool>,
    
    /// SIMD optimization settings
    pub simd_config: SimdConfig,
    
    /// Parallel processing settings
    pub parallel_config: ParallelConfig,
    
    /// Cache configuration
    pub cache_config: CacheConfig,
    
    /// Performance targets
    pub performance_targets: PerformanceTargets,
}

/// SIMD optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    pub enable_simd: bool,
    pub prefer_avx512: bool,
    pub prefer_avx2: bool,
    pub prefer_neon: bool,
    pub min_data_points_for_simd: usize,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    pub enable_parallel: bool,
    pub max_threads: Option<usize>,
    pub min_data_points_for_parallel: usize,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enable_cache: bool,
    pub max_cache_size: usize,
    pub cache_ttl_seconds: u64,
}

/// Performance targets for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target for individual analyzer execution (nanoseconds)
    pub analyzer_target_ns: u64,
    
    /// Target for full analysis pipeline (nanoseconds)
    pub full_analysis_target_ns: u64,
    
    /// Target for SIMD operations (nanoseconds)
    pub simd_target_ns: u64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        let mut enabled_analyzers = HashMap::new();
        enabled_analyzers.insert("antifragility".to_string(), true);
        enabled_analyzers.insert("soc".to_string(), true);
        enabled_analyzers.insert("panarchy".to_string(), true);
        enabled_analyzers.insert("fibonacci".to_string(), true);
        enabled_analyzers.insert("black_swan".to_string(), true);
        enabled_analyzers.insert("narrative_forecaster".to_string(), true);
        
        Self {
            enabled_analyzers,
            simd_config: SimdConfig {
                enable_simd: true,
                prefer_avx512: true,
                prefer_avx2: true,
                prefer_neon: true,
                min_data_points_for_simd: 64,
            },
            parallel_config: ParallelConfig {
                enable_parallel: true,
                max_threads: None, // Use system default
                min_data_points_for_parallel: 1000,
            },
            cache_config: CacheConfig {
                enable_cache: true,
                max_cache_size: 10000,
                cache_ttl_seconds: 300, // 5 minutes
            },
            performance_targets: PerformanceTargets {
                analyzer_target_ns: 1_000, // 1 microsecond per analyzer
                full_analysis_target_ns: 10_000, // 10 microseconds total
                simd_target_ns: 100, // 100 nanoseconds for SIMD ops
            },
        }
    }
}

/// Market data structure for analyzer input
#[derive(Debug, Clone)]
pub struct MarketData {
    pub prices: Array1<f64>,
    pub volumes: Array1<f64>,
    pub timestamps: Option<Array1<u64>>,
    pub high: Option<Array1<f64>>,
    pub low: Option<Array1<f64>>,
    pub open: Option<Array1<f64>>,
    pub close: Option<Array1<f64>>,
}

impl MarketData {
    /// Create new market data from prices and volumes
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>) -> Self {
        Self {
            prices: Array1::from_vec(prices),
            volumes: Array1::from_vec(volumes),
            timestamps: None,
            high: None,
            low: None,
            open: None,
            close: None,
        }
    }
    
    /// Create market data with OHLCV data
    pub fn from_ohlcv(
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volumes: Vec<f64>,
        timestamps: Option<Vec<u64>>,
    ) -> Self {
        Self {
            prices: Array1::from_vec(close.clone()),
            volumes: Array1::from_vec(volumes),
            timestamps: timestamps.map(Array1::from_vec),
            high: Some(Array1::from_vec(high)),
            low: Some(Array1::from_vec(low)),
            open: Some(Array1::from_vec(open)),
            close: Some(Array1::from_vec(close)),
        }
    }
    
    /// Get the number of data points
    pub fn len(&self) -> usize {
        self.prices.len()
    }
    
    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
    
    /// Validate data consistency
    pub fn validate(&self) -> PadsResult<()> {
        if self.prices.len() != self.volumes.len() {
            return Err(PadsError::ValidationError(
                "Prices and volumes must have same length".to_string()
            ));
        }
        
        // Check for invalid values
        for (i, &price) in self.prices.iter().enumerate() {
            if !price.is_finite() || price <= 0.0 {
                return Err(PadsError::ValidationError(
                    format!("Invalid price at index {}: {}", i, price)
                ));
            }
        }
        
        for (i, &volume) in self.volumes.iter().enumerate() {
            if !volume.is_finite() || volume < 0.0 {
                return Err(PadsError::ValidationError(
                    format!("Invalid volume at index {}: {}", i, volume)
                ));
            }
        }
        
        Ok(())
    }
}

/// Trait for individual pattern analyzers
pub trait PatternAnalyzer: Send + Sync {
    type Result: Clone + Send + Sync;
    
    /// Analyze market data and return results
    fn analyze(&self, data: &MarketData) -> PadsResult<Self::Result>;
    
    /// Get analyzer name
    fn name(&self) -> &'static str;
    
    /// Check if analyzer supports SIMD
    fn supports_simd(&self) -> bool { false }
    
    /// Check if analyzer supports parallel processing
    fn supports_parallel(&self) -> bool { false }
    
    /// Get minimum data points required
    fn min_data_points(&self) -> usize { 50 }
    
    /// Get analyzer configuration
    fn config(&self) -> serde_json::Value { serde_json::Value::Null }
    
    /// Update analyzer configuration
    fn update_config(&mut self, _config: serde_json::Value) -> PadsResult<()> { Ok(()) }
}

/// Helper function to detect available SIMD features
pub fn detect_simd_features() -> SimdFeatures {
    SimdFeatures {
        avx512: cfg!(target_feature = "avx512f"),
        avx2: cfg!(target_feature = "avx2"),
        avx: cfg!(target_feature = "avx"),
        sse4_2: cfg!(target_feature = "sse4.2"),
        neon: cfg!(target_feature = "neon"),
    }
}

/// Available SIMD features on current platform
#[derive(Debug, Clone)]
pub struct SimdFeatures {
    pub avx512: bool,
    pub avx2: bool,
    pub avx: bool,
    pub sse4_2: bool,
    pub neon: bool,
}

impl SimdFeatures {
    /// Check if any SIMD features are available
    pub fn has_simd(&self) -> bool {
        self.avx512 || self.avx2 || self.avx || self.sse4_2 || self.neon
    }
    
    /// Get the best available SIMD feature
    pub fn best_feature(&self) -> Option<&'static str> {
        if self.avx512 { Some("avx512") }
        else if self.avx2 { Some("avx2") }
        else if self.avx { Some("avx") }
        else if self.sse4_2 { Some("sse4.2") }
        else if self.neon { Some("neon") }
        else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_creation() {
        let prices = vec![100.0, 101.0, 99.0, 102.0];
        let volumes = vec![1000.0, 1100.0, 900.0, 1200.0];
        
        let market_data = MarketData::new(prices, volumes);
        assert_eq!(market_data.len(), 4);
        assert!(!market_data.is_empty());
        assert!(market_data.validate().is_ok());
    }
    
    #[test]
    fn test_market_data_validation() {
        // Mismatched lengths
        let prices = vec![100.0, 101.0];
        let volumes = vec![1000.0];
        let market_data = MarketData::new(prices, volumes);
        assert!(market_data.validate().is_err());
        
        // Invalid price
        let prices = vec![0.0];
        let volumes = vec![1000.0];
        let market_data = MarketData::new(prices, volumes);
        assert!(market_data.validate().is_err());
        
        // Invalid volume
        let prices = vec![100.0];
        let volumes = vec![-1.0];
        let market_data = MarketData::new(prices, volumes);
        assert!(market_data.validate().is_err());
    }
    
    #[test]
    fn test_simd_feature_detection() {
        let features = detect_simd_features();
        // Just ensure the function runs without error
        let _ = features.has_simd();
        let _ = features.best_feature();
    }
    
    #[test]
    fn test_default_config() {
        let config = AnalyzerConfig::default();
        assert!(config.enabled_analyzers.get("antifragility").unwrap_or(&false));
        assert!(config.simd_config.enable_simd);
        assert!(config.parallel_config.enable_parallel);
        assert!(config.cache_config.enable_cache);
    }
}