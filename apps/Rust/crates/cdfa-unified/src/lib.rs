//! # CDFA Unified - Cross-Domain Feature Alignment Library
//!
//! This library provides a unified interface to all Cross-Domain Feature Alignment (CDFA) functionality,
//! consolidating 15+ specialized crates into a single, high-performance library.
//!
//! ## Features
//!
//! - **Core Algorithms**: Diversity metrics, signal fusion, and combinatorial analysis
//! - **Advanced Detectors**: Pattern detection (Fibonacci, Black Swan, etc.)
//! - **Performance Optimization**: SIMD, parallel processing, and GPU acceleration
//! - **Machine Learning**: Neural networks, classical ML, and optimization
//! - **Distributed Computing**: Redis integration and distributed coordination
//! - **Production Ready**: Health monitoring, configuration management, and deployment tools
//!
//! ## Quick Start
//!
//! ```rust
//! use cdfa_unified::UnifiedCdfa;
//! use ndarray::array;
//!
//! // Create a unified CDFA instance
//! let cdfa = UnifiedCdfa::new()?;
//!
//! // Analyze data
//! let data = array![[1.0, 2.0, 3.0], [1.1, 2.1, 2.9]];
//! let result = cdfa.analyze(&data.view())?;
//!
//! println!("Analysis result: {:?}", result);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core exports
pub use error::{CdfaError, CdfaResult};
pub use types::*;

// Unified API
#[cfg(feature = "core")]
pub use unified::UnifiedCdfa;

#[cfg(feature = "core")]
pub use builder::UnifiedCdfaBuilder;

// Configuration system
#[cfg(feature = "serde")]
pub use config::CdfaConfig;

// Prelude for convenient imports
pub mod prelude;

// Core modules
#[cfg(feature = "core")]
pub mod core;

#[cfg(feature = "algorithms")]
pub mod algorithms;

#[cfg(feature = "detectors")]
pub mod detectors;

// Analyzers module for specialized analysis tools
pub mod analyzers;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "ml")]
pub mod ml;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "redis-integration")]
pub mod integration;

#[cfg(feature = "serde")]
pub mod config;

// Infrastructure modules
pub mod backends;
pub mod error;
pub mod types;
pub mod utils;
// pub mod audit; // Disabled for compilation
pub mod validation;
pub mod traits;

// High-precision numerical computing
pub mod precision;

// FFI modules
#[cfg(any(feature = "ffi", feature = "c-bindings", feature = "python"))]
pub mod ffi;

#[cfg(feature = "core")]
pub mod unified;

#[cfg(feature = "core")]
pub mod builder;

#[cfg(feature = "core")]
pub mod registry;

// Optimizers module for neural network optimization
#[cfg(feature = "stdp")]
pub mod optimizers;

// Backward compatibility
#[cfg(any(feature = "compat-core", feature = "compat-algorithms", feature = "compat-parallel", feature = "compat-simd", feature = "compat-ml", feature = "compat-detectors"))]
pub mod compat;

// Version information
/// Version information for the CDFA unified library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Get build information
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: VERSION,
        features: get_enabled_features(),
        build_date: env!("BUILD_DATE"),
        git_hash: env!("GIT_HASH"),
        rust_version: env!("RUST_VERSION"),
    }
}

/// Build information structure
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Library version
    pub version: &'static str,
    /// Enabled features
    pub features: Vec<&'static str>,
    /// Build date
    pub build_date: &'static str,
    /// Git commit hash
    pub git_hash: &'static str,
    /// Rust version used for build
    pub rust_version: &'static str,
}

fn get_enabled_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    
    #[cfg(feature = "core")]
    features.push("core");
    
    #[cfg(feature = "algorithms")]
    features.push("algorithms");
    
    #[cfg(feature = "simd")]
    features.push("simd");
    
    #[cfg(feature = "parallel")]
    features.push("parallel");
    
    #[cfg(feature = "ml")]
    features.push("ml");
    
    #[cfg(feature = "detectors")]
    features.push("detectors");
    
    #[cfg(feature = "redis-integration")]
    features.push("redis-integration");
    
    #[cfg(feature = "gpu")]
    features.push("gpu");
    
    #[cfg(feature = "distributed")]
    features.push("distributed");
    
    #[cfg(feature = "cuda")]
    features.push("cuda");
    
    #[cfg(feature = "metal")]
    features.push("metal");
    
    #[cfg(feature = "webgpu")]
    features.push("webgpu");
    
    features
}

// Re-export key types for convenience
pub use ndarray;
pub use nalgebra;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
    
    #[test]
    fn test_build_info() {
        let info = build_info();
        assert!(!info.version.is_empty());
        assert!(!info.features.is_empty());
    }
}