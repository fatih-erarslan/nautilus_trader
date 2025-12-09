//! CDFA Advanced Market Detectors
//!
//! Enterprise-grade implementations of sophisticated market pattern detection algorithms
//! with sub-microsecond performance targets. This crate provides:
//!
//! - **AccumulationDetector**: Identifies accumulation zones using volume profile analysis,
//!   higher lows detection, and buy/sell pressure metrics
//! - **DistributionDetector**: Detects distribution patterns through lower highs analysis,
//!   volume divergence, and supply absorption detection
//! - **ConfluenceAreaDetector**: Multi-indicator confluence analysis combining support/resistance,
//!   Fibonacci levels, moving averages, and volume profile nodes
//! - **BubbleDetector**: Exponential growth pattern detection with regime analysis and
//!   social sentiment integration
//!
//! All detectors feature:
//! - SIMD optimization using f32x8 vectorization
//! - Parallel processing with Rayon
//! - Cache-aligned data structures
//! - Comprehensive mathematical analysis
//! - Real-time performance monitoring
//!
//! ## Performance Targets
//! - Full detection cycles: < 1 microsecond
//! - Individual pattern analysis: < 200 nanoseconds
//! - SIMD operations: < 50 nanoseconds
//!
//! ## Example Usage
//! ```rust
//! use cdfa_advanced_detectors::{AccumulationDetector, MarketData};
//!
//! let detector = AccumulationDetector::new();
//! let market_data = MarketData::new(prices, volumes, timestamps);
//! let result = detector.detect(&market_data)?;
//! 
//! if result.accumulation_detected {
//!     println!(\"Accumulation zone detected with strength: {}\", result.strength);
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use log::{debug, info, warn, error};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ndarray::{Array1, Array2, ArrayView1};

#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Error types
#[derive(Error, Debug)]
pub enum DetectorError {
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    #[error("Calculation error: {message}")]
    CalculationError { message: String },
    
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
}

pub type Result<T> = std::result::Result<T, DetectorError>;

// Core data structures
/// Market data container for all detectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub prices: Vec<f32>,
    pub volumes: Vec<f32>,
    pub timestamps: Vec<i64>,
    pub highs: Vec<f32>,
    pub lows: Vec<f32>,
    pub opens: Vec<f32>,
}

impl MarketData {
    pub fn new(prices: Vec<f32>, volumes: Vec<f32>, timestamps: Vec<i64>) -> Self {
        let highs = prices.clone();
        let lows = prices.clone();
        let opens = prices.clone();
        
        Self {
            prices,
            volumes,
            timestamps,
            highs,
            lows,
            opens,
        }
    }
    
    pub fn with_ohlc(
        opens: Vec<f32>,
        highs: Vec<f32>,
        lows: Vec<f32>,
        closes: Vec<f32>,
        volumes: Vec<f32>,
        timestamps: Vec<i64>,
    ) -> Self {
        Self {
            prices: closes,
            volumes,
            timestamps,
            highs,
            lows,
            opens,
        }
    }
    
    pub fn len(&self) -> usize {
        self.prices.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.prices.len() != self.volumes.len() {
            return Err(DetectorError::InvalidInput {
                message: "Prices and volumes must have the same length".to_string(),
            });
        }
        
        if self.prices.len() != self.timestamps.len() {
            return Err(DetectorError::InvalidInput {
                message: "Prices and timestamps must have the same length".to_string(),
            });
        }
        
        // Check for invalid values
        for (i, &price) in self.prices.iter().enumerate() {
            if !price.is_finite() || price <= 0.0 {
                return Err(DetectorError::InvalidInput {
                    message: format!("Invalid price at index {}: {}", i, price),
                });
            }
        }
        
        for (i, &volume) in self.volumes.iter().enumerate() {
            if !volume.is_finite() || volume < 0.0 {
                return Err(DetectorError::InvalidInput {
                    message: format!("Invalid volume at index {}: {}", i, volume),
                });
            }
        }
        
        Ok(())
    }
}

/// Detection result base structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub detected: bool,
    pub strength: f32,
    pub confidence: f32,
    pub start_index: Option<usize>,
    pub end_index: Option<usize>,
    pub calculation_time_ns: u64,
    pub metadata: HashMap<String, f32>,
}

impl DetectionResult {
    pub fn new(detected: bool, strength: f32, confidence: f32) -> Self {
        Self {
            detected,
            strength,
            confidence,
            start_index: None,
            end_index: None,
            calculation_time_ns: 0,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_range(mut self, start: usize, end: usize) -> Self {
        self.start_index = Some(start);
        self.end_index = Some(end);
        self
    }
    
    pub fn with_timing(mut self, time_ns: u64) -> Self {
        self.calculation_time_ns = time_ns;
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: f32) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// Module declarations
pub mod accumulation;
pub mod distribution;
pub mod confluence;
pub mod bubble;
pub mod utils;
pub mod simd_ops;
pub mod pbit_detectors;

// Re-exports
pub use accumulation::*;
pub use distribution::*;
pub use confluence::*;
pub use bubble::*;
pub use utils::*;
pub use simd_ops::*;

/// Performance monitoring structure
#[derive(Debug, Default)]
pub struct PerformanceMonitor {
    pub total_detections: AtomicU64,
    pub total_time_ns: AtomicU64,
    pub accumulation_calls: AtomicU64,
    pub distribution_calls: AtomicU64,
    pub confluence_calls: AtomicU64,
    pub bubble_calls: AtomicU64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn record_detection(&self, time_ns: u64, detector_type: &str) {
        self.total_detections.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(time_ns, Ordering::Relaxed);
        
        match detector_type {
            "accumulation" => self.accumulation_calls.fetch_add(1, Ordering::Relaxed),
            "distribution" => self.distribution_calls.fetch_add(1, Ordering::Relaxed),
            "confluence" => self.confluence_calls.fetch_add(1, Ordering::Relaxed),
            "bubble" => self.bubble_calls.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
    }
    
    pub fn get_stats(&self) -> PerformanceStats {
        let total_detections = self.total_detections.load(Ordering::Relaxed);
        let total_time_ns = self.total_time_ns.load(Ordering::Relaxed);
        
        PerformanceStats {
            total_detections,
            total_time_ns,
            average_time_ns: if total_detections > 0 {
                total_time_ns as f64 / total_detections as f64
            } else {
                0.0
            },
            accumulation_calls: self.accumulation_calls.load(Ordering::Relaxed),
            distribution_calls: self.distribution_calls.load(Ordering::Relaxed),
            confluence_calls: self.confluence_calls.load(Ordering::Relaxed),
            bubble_calls: self.bubble_calls.load(Ordering::Relaxed),
        }
    }
    
    pub fn reset(&self) {
        self.total_detections.store(0, Ordering::Relaxed);
        self.total_time_ns.store(0, Ordering::Relaxed);
        self.accumulation_calls.store(0, Ordering::Relaxed);
        self.distribution_calls.store(0, Ordering::Relaxed);
        self.confluence_calls.store(0, Ordering::Relaxed);
        self.bubble_calls.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_detections: u64,
    pub total_time_ns: u64,
    pub average_time_ns: f64,
    pub accumulation_calls: u64,
    pub distribution_calls: u64,
    pub confluence_calls: u64,
    pub bubble_calls: u64,
}

/// Global performance monitor instance
static PERFORMANCE_MONITOR: once_cell::sync::Lazy<PerformanceMonitor> = 
    once_cell::sync::Lazy::new(|| PerformanceMonitor::new());

/// Get global performance statistics
pub fn get_performance_stats() -> PerformanceStats {
    PERFORMANCE_MONITOR.get_stats()
}

/// Reset global performance counters
pub fn reset_performance_stats() {
    PERFORMANCE_MONITOR.reset();
}

// Performance targets (in nanoseconds)
pub mod perf {
    /// Target: Complete accumulation detection in under 800ns
    pub const ACCUMULATION_DETECTION_TARGET_NS: u64 = 800;
    
    /// Target: Complete distribution detection in under 800ns
    pub const DISTRIBUTION_DETECTION_TARGET_NS: u64 = 800;
    
    /// Target: Complete confluence detection in under 1200ns
    pub const CONFLUENCE_DETECTION_TARGET_NS: u64 = 1200;
    
    /// Target: Complete bubble detection in under 1000ns
    pub const BUBBLE_DETECTION_TARGET_NS: u64 = 1000;
    
    /// Target: SIMD operations in under 50ns
    pub const SIMD_OPERATION_TARGET_NS: u64 = 50;
    
    /// Target: Individual pattern analysis in under 200ns
    pub const PATTERN_ANALYSIS_TARGET_NS: u64 = 200;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_data_creation() {
        let prices = vec![100.0, 105.0, 102.0, 108.0, 104.0];
        let volumes = vec![1000.0, 1200.0, 800.0, 1500.0, 900.0];
        let timestamps = vec![1000, 1001, 1002, 1003, 1004];
        
        let market_data = MarketData::new(prices, volumes, timestamps);
        assert_eq!(market_data.len(), 5);
        assert!(!market_data.is_empty());
        assert!(market_data.validate().is_ok());
    }
    
    #[test]
    fn test_market_data_validation() {
        let prices = vec![100.0, 105.0, 102.0];
        let volumes = vec![1000.0, 1200.0]; // Different length
        let timestamps = vec![1000, 1001, 1002];
        
        let market_data = MarketData::new(prices, volumes, timestamps);
        assert!(market_data.validate().is_err());
    }
    
    #[test]
    fn test_detection_result() {
        let mut result = DetectionResult::new(true, 0.8, 0.9);
        result = result.with_range(10, 20);
        result = result.with_timing(500);
        result = result.with_metadata("test_key".to_string(), 42.0);
        
        assert!(result.detected);
        assert_eq!(result.strength, 0.8);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.start_index, Some(10));
        assert_eq!(result.end_index, Some(20));
        assert_eq!(result.calculation_time_ns, 500);
        assert_eq!(result.metadata.get("test_key"), Some(&42.0));
    }
    
    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        monitor.record_detection(1000, "accumulation");
        monitor.record_detection(1500, "distribution");
        
        let stats = monitor.get_stats();
        assert_eq!(stats.total_detections, 2);
        assert_eq!(stats.total_time_ns, 2500);
        assert_eq!(stats.average_time_ns, 1250.0);
        assert_eq!(stats.accumulation_calls, 1);
        assert_eq!(stats.distribution_calls, 1);
        
        monitor.reset();
        let reset_stats = monitor.get_stats();
        assert_eq!(reset_stats.total_detections, 0);
    }
}