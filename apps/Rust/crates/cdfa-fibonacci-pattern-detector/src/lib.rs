//! # CDFA Fibonacci Pattern Detector
//!
//! Advanced harmonic pattern detection for CDFA with sub-microsecond performance.
//! 
//! This crate provides high-performance detection of harmonic patterns including:
//! - Gartley patterns (XABCD formation)
//! - Butterfly patterns  
//! - Bat patterns
//! - Crab patterns
//! - Shark patterns
//! - GPU acceleration with fallback to SIMD optimization

#![warn(clippy::all)]
#![allow(dead_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub mod patterns;
pub mod detector;
pub mod types;
pub mod simd;
pub mod validation;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "parallel")]
pub mod parallel;

pub use detector::FibonacciPatternDetector;
pub use types::{
    PatternType, PatternPoint, DetectedPattern, PatternConfig,
    HarmonicRatios, PatternResult, PatternParameters
};
pub use patterns::{gartley_config, butterfly_config, bat_config, crab_config, shark_config};

use thiserror::Error;

/// Pattern detection errors
#[derive(Error, Debug)]
pub enum PatternError {
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Pattern validation failed: {message}")]
    ValidationError { message: String },
    
    #[cfg(feature = "gpu")]
    #[error("GPU error: {message}")]
    GpuError { message: String },
    
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },
}

pub type Result<T> = std::result::Result<T, PatternError>;

/// Performance constants for sub-microsecond targets
pub mod perf {
    /// Target latency for pattern scanning (nanoseconds)
    pub const PATTERN_SCAN_TARGET_NS: u64 = 600;
    
    /// Target latency for pattern validation (nanoseconds) 
    pub const PATTERN_VALIDATION_TARGET_NS: u64 = 200;
    
    /// Target latency for ratio calculation (nanoseconds)
    pub const RATIO_CALCULATION_TARGET_NS: u64 = 150;
    
    /// Target latency for full pattern detection (nanoseconds)
    pub const FULL_DETECTION_TARGET_NS: u64 = 800;
}

/// Initialize pattern detector with optimized settings
pub fn init() -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        gpu::init_gpu()?;
    }
    
    #[cfg(feature = "parallel")]
    {
        parallel::init_thread_pool()?;
    }
    
    Ok(())
}

/// Check if hardware acceleration is available
pub fn acceleration_available() -> bool {
    #[cfg(feature = "gpu")]
    return gpu::gpu_available();
    
    #[cfg(not(feature = "gpu"))]
    false
}

/// Get pattern detector version info
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
    
    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
    
    #[test]
    fn test_basic_pattern_detection() {
        let high = Array1::from_vec(vec![1.0, 1.2, 1.1, 1.3, 1.05, 1.25]);
        let low = Array1::from_vec(vec![0.95, 1.15, 1.05, 1.25, 1.0, 1.2]);
        let close = Array1::from_vec(vec![1.0, 1.18, 1.08, 1.28, 1.02, 1.22]);
        
        let detector = FibonacciPatternDetector::new();
        let result = detector.detect_patterns(&high, &low, &close);
        assert!(result.is_ok());
    }
}