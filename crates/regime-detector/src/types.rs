//! Core types for regime detection

use serde::{Deserialize, Serialize};
use std::fmt;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong upward trend (Bull market)
    Bullish,
    /// Strong downward trend (Bear market)
    Bearish,
    /// Sideways movement with low volatility
    Ranging,
    /// High volatility regime
    Volatile,
    /// Crisis/extreme regime
    Crisis,
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketRegime::Bullish => write!(f, "Bullish"),
            MarketRegime::Bearish => write!(f, "Bearish"),
            MarketRegime::Ranging => write!(f, "Ranging"),
            MarketRegime::Volatile => write!(f, "Volatile"),
            MarketRegime::Crisis => write!(f, "Crisis"),
        }
    }
}

/// Features extracted for regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeFeatures {
    /// Linear regression slope (normalized)
    pub trend_strength: f32,
    /// Standard deviation of returns
    pub volatility: f32,
    /// Autocorrelation coefficient
    pub autocorrelation: f32,
    /// Volume-weighted average price ratio
    pub vwap_ratio: f32,
    /// Hurst exponent for trend persistence
    pub hurst_exponent: f32,
    /// Relative strength index
    pub rsi: f32,
    /// Market microstructure noise
    pub microstructure_noise: f32,
    /// Order flow imbalance
    pub order_flow_imbalance: f32,
}

impl Default for RegimeFeatures {
    fn default() -> Self {
        Self {
            trend_strength: 0.0,
            volatility: 0.0,
            autocorrelation: 0.0,
            vwap_ratio: 1.0,
            hurst_exponent: 0.5,
            rsi: 50.0,
            microstructure_noise: 0.0,
            order_flow_imbalance: 0.0,
        }
    }
}

/// Result of regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionResult {
    /// Detected regime
    pub regime: MarketRegime,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Features used for detection
    pub features: RegimeFeatures,
    /// Transition probability to other regimes
    pub transition_probs: Vec<(MarketRegime, f32)>,
    /// Detection latency in nanoseconds
    pub latency_ns: u64,
}

/// Configuration for regime detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Window size for feature calculation
    pub window_size: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Enable cache for faster detection
    pub enable_cache: bool,
    /// Cache size (number of entries)
    pub cache_size: usize,
    /// Use AVX-512 if available
    pub use_avx512: bool,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            min_confidence: 0.7,
            enable_cache: true,
            cache_size: 1024,
            use_avx512: true,
        }
    }
}