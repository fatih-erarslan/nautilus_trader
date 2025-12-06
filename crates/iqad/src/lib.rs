//! # Immune-Inspired Quantum Anomaly Detection (IQAD)
//! 
//! Enterprise-grade implementation of quantum-enhanced anomaly detection
//! inspired by biological immune systems. This crate provides production-ready
//! pBit-based probabilistic computing capabilities with hardware acceleration.
//!
//! ## Features
//! - pBit-based quantum-inspired circuits (replaces roqoqo)
//! - AVX-512 SIMD acceleration with fallbacks
//! - GPU acceleration (CUDA/OpenCL/WebGPU)
//! - Production-ready error handling
//! - Immune system algorithms: negative selection, clonal selection

// Feature portable_simd removed for stable compatibility
#![warn(missing_docs)]

pub mod detector;
pub mod pbit_circuits;
pub mod immune_system;
pub mod hardware;
pub mod cache;
pub mod error;
pub mod types;

// Legacy support
#[cfg(feature = "roqoqo")]
pub mod quantum_circuits;

// Re-exports
pub use detector::ImmuneQuantumAnomalyDetector;
pub use error::{IqadError, IqadResult};
pub use types::*;

// Version information
/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "1.0.0");
    }
}