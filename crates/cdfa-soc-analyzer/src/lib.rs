//! # CDFA SOC Analyzer
//!
//! Self-Organized Criticality (SOC) analyzer for CDFA with sub-microsecond performance.
//! 
//! This crate provides high-performance SOC analysis including:
//! - Sample entropy calculation
//! - Entropy rate estimation 
//! - Avalanche detection
//! - SOC regime classification (critical, stable, unstable)
//! - Hardware acceleration with SIMD and GPU support

#![warn(clippy::all)]
#![allow(dead_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub mod analyzer;
pub mod entropy;
pub mod regimes;
pub mod types;
pub mod simd;
pub mod pbit_soc;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "parallel")]
pub mod parallel;

pub use analyzer::SOCAnalyzer;
pub use types::{SOCParameters, SOCResult, SOCRegime, AvalancheEvent};
pub use entropy::{sample_entropy, entropy_rate};
pub use regimes::classify_regime;

use thiserror::Error;

/// SOC analyzer errors
#[derive(Error, Debug)]
pub enum SOCError {
    #[error("Invalid parameters: {message}")]
    InvalidParameters { message: String },
    
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Computation error: {message}")]
    ComputationError { message: String },
    
    #[cfg(feature = "gpu")]
    #[error("GPU error: {message}")]
    GpuError { message: String },
    
    #[error("SIMD operation failed: {message}")]
    SimdError { message: String },
}

pub type Result<T> = std::result::Result<T, SOCError>;

/// Performance constants for sub-microsecond targets
pub mod perf {
    /// Target latency for sample entropy calculation (nanoseconds)
    pub const SAMPLE_ENTROPY_TARGET_NS: u64 = 800;
    
    /// Target latency for entropy rate calculation (nanoseconds) 
    pub const ENTROPY_RATE_TARGET_NS: u64 = 600;
    
    /// Target latency for regime classification (nanoseconds)
    pub const REGIME_CLASSIFICATION_TARGET_NS: u64 = 400;
    
    /// Target latency for full SOC analysis (nanoseconds)
    pub const FULL_ANALYSIS_TARGET_NS: u64 = 950;
}

/// Initialize SOC analyzer with optimized settings
pub fn init() -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        // Placeholder for GPU initialization
        // gpu::init_gpu()?;
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
    {
        // Placeholder for GPU availability check
        false
    }
    
    #[cfg(not(feature = "gpu"))]
    false
}

/// Get SOC analyzer version info
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
    fn test_sample_data() {
        let data = Array1::from_vec((0..100).map(|x| x as f64 * 0.1).collect());
        let params = SOCParameters::default();
        let analyzer = SOCAnalyzer::new(params);
        
        let result = analyzer.analyze(&data);
        assert!(result.is_ok());
    }
}