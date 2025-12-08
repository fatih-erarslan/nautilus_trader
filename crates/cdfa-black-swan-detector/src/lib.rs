//! # CDFA Black Swan Detector
//!
//! Ultra-low latency Black Swan event detector implementing Extreme Value Theory (EVT)
//! with sub-microsecond performance targets.
//!
//! ## Features
//!
//! - **Extreme Value Theory**: Hill estimator for tail risk assessment
//! - **Real-time Processing**: Sub-microsecond latency with SIMD optimizations
//! - **GPU Acceleration**: Candle framework integration for parallel processing
//! - **Memory Efficient**: Rolling window calculations with zero-copy operations
//! - **Statistical Significance**: Rigorous hypothesis testing for event detection
//! - **CDFA Integration**: Seamless integration with existing CDFA ecosystem
//!
//! ## Quick Start
//!
//! ```rust
//! use cdfa_black_swan_detector::BlackSwanDetector;
//!
//! let detector = BlackSwanDetector::new(Default::default())?;
//! let probability = detector.detect_real_time(&prices, &volumes)?;
//! ```

pub mod config;
pub mod detector;
pub mod error;
pub mod evt;
pub mod metrics;
pub mod simd;
pub mod types;
pub mod utils;
pub mod pbit_detector;

// Re-export main components
pub use config::*;
pub use detector::*;
pub use error::*;
pub use evt::*;
pub use metrics::*;
pub use types::*;
pub use utils::validation;

// Feature-gated exports
#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "python")]
pub mod python;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration for production use
pub fn default_config() -> BlackSwanConfig {
    BlackSwanConfig {
        window_size: 1000,
        tail_threshold: 0.95,
        min_tail_points: 50,
        significance_level: 0.01,
        hill_estimator_k: 100,
        extreme_z_threshold: 3.0,
        volatility_clustering_alpha: 0.1,
        liquidity_crisis_threshold: 0.3,
        correlation_breakdown_threshold: 0.5,
        memory_pool_size: 1024 * 1024, // 1MB
        use_gpu: true,
        use_simd: true,
        parallel_processing: true,
        cache_size: 10000,
    }
}

/// Initialize the Black Swan detector with optimal settings
pub fn init_detector() -> Result<BlackSwanDetector, BlackSwanError> {
    let config = default_config();
    BlackSwanDetector::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = default_config();
        assert_eq!(config.window_size, 1000);
        assert_relative_eq!(config.tail_threshold, 0.95);
        assert!(config.use_gpu);
        assert!(config.use_simd);
    }

    #[test]
    fn test_init_detector() {
        let detector = init_detector();
        assert!(detector.is_ok());
    }
}