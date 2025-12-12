//! # CDFA STDP Optimizer
//!
//! Ultra-fast Spike-Timing Dependent Plasticity (STDP) optimizer with sub-microsecond performance
//! 
//! This crate provides biologically-inspired weight optimization using principles from
//! neuroplasticity, optimized for real-time trading applications.
//!
//! ## Features
//!
//! - **Sub-microsecond performance**: Lock-free data structures and SIMD acceleration
//! - **Temporal pattern recognition**: Advanced spike-timing dependent plasticity
//! - **Memory-efficient**: Cache-aligned data structures with optimal memory layout
//! - **GPU acceleration**: Optional WGPU backend for massive parallel processing
//! - **Real-time learning**: Adaptive weight updates for dynamic market conditions
//! - **Homeostatic plasticity**: System stability through biological feedback mechanisms
//!
//! ## Quick Start
//!
//! ```rust
//! use cdfa_stdp_optimizer::*;
//!
//! // Create STDP optimizer
//! let mut optimizer = STDPOptimizer::new(STDPConfig::default());
//!
//! // Initialize weights
//! let weights = optimizer.initialize_weights(100, 10);
//!
//! // Apply STDP learning
//! let result = optimizer.apply_stdp(&pre_spikes, &post_spikes, &weights, None);
//! ```

pub mod config;
pub mod core;
pub mod error;
pub mod memory;
pub mod patterns;
pub mod plasticity;
pub mod simd;
pub mod temporal;
pub mod types;
pub mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "python")]
pub mod python;

// Re-export public API
pub use config::*;
pub use core::*;
pub use error::*;
pub use memory::*;
pub use patterns::*;
pub use plasticity::*;
pub use temporal::*;
pub use types::*;
pub use utils::*;

#[cfg(feature = "gpu")]
pub use gpu::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Performance constants
pub const CACHE_LINE_SIZE: usize = 64;
pub const SIMD_WIDTH: usize = 8; // AVX2 float width
pub const MAX_NEURONS: usize = 1024 * 1024; // 1M neurons max
pub const MAX_SYNAPSES: usize = 1024 * 1024 * 1024; // 1B synapses max
pub const MICROSECOND_PRECISION: bool = true;
pub const NANOSECOND_PRECISION: bool = true;

/// Initialize the STDP optimizer library
pub fn init() -> Result<(), STDPError> {
    env_logger::init();
    log::info!("Initializing CDFA STDP Optimizer v{}", VERSION);
    
    // Initialize SIMD detection
    simd::init_simd()?;
    
    // Initialize memory allocators
    memory::init_allocators()?;
    
    log::info!("STDP Optimizer initialized successfully");
    Ok(())
}

/// Get library information
pub fn info() -> std::collections::HashMap<&'static str, String> {
    let mut info = std::collections::HashMap::new();
    info.insert("name", NAME.to_string());
    info.insert("version", VERSION.to_string());
    info.insert("description", DESCRIPTION.to_string());
    info.insert("simd_support", simd::get_simd_info());
    info.insert("cache_line_size", CACHE_LINE_SIZE.to_string());
    info.insert("max_neurons", MAX_NEURONS.to_string());
    info.insert("max_synapses", MAX_SYNAPSES.to_string());
    info.insert("microsecond_precision", MICROSECOND_PRECISION.to_string());
    info.insert("nanosecond_precision", NANOSECOND_PRECISION.to_string());
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_info() {
        let info = info();
        assert!(info.contains_key("name"));
        assert!(info.contains_key("version"));
        assert_eq!(info["name"], NAME);
        assert_eq!(info["version"], VERSION);
    }

    #[test]
    fn test_constants() {
        assert_eq!(CACHE_LINE_SIZE, 64);
        assert_eq!(SIMD_WIDTH, 8);
        assert!(MAX_NEURONS > 0);
        assert!(MAX_SYNAPSES > 0);
        assert!(MICROSECOND_PRECISION);
        assert!(NANOSECOND_PRECISION);
    }
}