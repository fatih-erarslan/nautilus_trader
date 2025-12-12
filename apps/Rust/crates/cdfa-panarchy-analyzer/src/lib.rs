//! # CDFA Panarchy Analyzer
//!
//! High-performance Panarchy cycle analyzer for CDFA with sub-microsecond performance.
//! 
//! This crate implements the four-phase Panarchy adaptive cycle model:
//! - Growth (r): Exploitation of opportunities
//! - Conservation (K): Stability and efficiency
//! - Release (Ω): Creative destruction
//! - Reorganization (α): Innovation and renewal
//!
//! ## Features
//! - Sub-microsecond performance for all operations
//! - SIMD optimization using the `wide` crate
//! - PCR (Potential, Connectedness, Resilience) component analysis
//! - Phase identification with hysteresis
//! - ADX calculation for trend strength
//! - Compatible with CDFA server integration
//!
//! ## Example
//! ```rust,no_run
//! use cdfa_panarchy_analyzer::{PanarchyAnalyzer, PanarchyParameters};
//!
//! let mut analyzer = PanarchyAnalyzer::new();
//! let prices = vec![100.0, 101.0, 99.5, 102.0, 103.0];
//! let volumes = vec![1000.0, 1100.0, 950.0, 1200.0, 1150.0];
//! 
//! let result = analyzer.analyze(&prices, &volumes).unwrap();
//! println!("Current phase: {}", result.phase);
//! println!("Signal: {:.2}", result.signal);
//! println!("Confidence: {:.2}", result.confidence);
//! ```

#![warn(clippy::all)]
#![allow(dead_code)]

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

pub mod analyzer;
pub mod pcr;
pub mod phase;
pub mod simd;
pub mod types;

pub use analyzer::{PanarchyAnalyzer, BatchPanarchyAnalyzer};
pub use types::{
    MarketPhase, PanarchyParameters, PCRComponents, PhaseScores,
    PanarchyResult, MarketData, RegimeScoreConfig, PhaseWeights,
};
pub use phase::{PhaseIdentifier, PhaseTransitionTracker};
pub use pcr::{calculate_pcr_components, calculate_pcr_batch, FastPCRCalculator};

use thiserror::Error;

/// Panarchy analyzer errors
#[derive(Error, Debug)]
pub enum PanarchyError {
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

pub type Result<T> = std::result::Result<T, PanarchyError>;

/// Performance constants for sub-microsecond targets
pub mod performance {
    /// Target latency for PCR calculation (nanoseconds)
    pub const PCR_CALCULATION_TARGET_NS: u64 = 300;
    
    /// Target latency for phase classification (nanoseconds)
    pub const PHASE_CLASSIFICATION_TARGET_NS: u64 = 200;
    
    /// Target latency for regime score calculation (nanoseconds)
    pub const REGIME_SCORE_TARGET_NS: u64 = 150;
    
    /// Target latency for full analysis (nanoseconds)
    pub const FULL_ANALYSIS_TARGET_NS: u64 = 800;
}

/// Initialize the Panarchy analyzer with optimized settings
pub fn init() -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        // Initialize GPU if available
        // gpu::init_gpu()?;
    }
    
    #[cfg(feature = "parallel")]
    {
        // Initialize thread pool for parallel processing
        // parallel::init_thread_pool()?;
    }
    
    Ok(())
}

/// Check if hardware acceleration is available
pub fn acceleration_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Check GPU availability
        // return gpu::gpu_available();
    }
    
    // Check SIMD availability
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Quick analysis function for simple use cases
pub fn quick_analyze(prices: &[f64], volumes: &[f64]) -> Result<PanarchyResult> {
    let mut analyzer = PanarchyAnalyzer::new();
    analyzer.analyze(prices, volumes)
}

/// Calculate just the PCR components without full analysis
pub fn calculate_pcr(
    prices: &[f64],
    period: usize,
) -> Result<Vec<PCRComponents>> {
    let analyzer = PanarchyAnalyzer::new();
    analyzer.calculate_pcr(prices, period)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
    
    #[test]
    fn test_acceleration_available() {
        // This will vary by platform
        let _ = acceleration_available();
    }
    
    #[test]
    fn test_quick_analyze() {
        let prices = vec![100.0; 50];
        let volumes = vec![1000.0; 50];
        
        let result = quick_analyze(&prices, &volumes);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_calculate_pcr_simple() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0; 20];
        let pcr = calculate_pcr(&prices, 10);
        assert!(pcr.is_ok());
        
        let pcr_values = pcr.unwrap();
        assert_eq!(pcr_values.len(), prices.len());
    }
}