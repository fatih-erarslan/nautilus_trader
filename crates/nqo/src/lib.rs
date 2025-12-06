//! # Neuromorphic Quantum Optimizer (NQO)
//! 
//! Enterprise-grade implementation of hybrid neural-quantum optimization
//! with pBit-based quantum circuits and neural network integration.
//!
//! ## Features
//! - pBit-based quantum optimization (replaces roqoqo)
//! - Neural network integration with Candle
//! - QAOA and VQE via simulated annealing
//! - GPU acceleration (CUDA/OpenCL/WebGPU)
//! - Production-ready without mocks or stubs

// Feature portable_simd removed for stable compatibility
#![warn(missing_docs)]

pub mod optimizer;
pub mod pbit_optimizer;
pub mod neural_network;
pub mod hardware;
pub mod cache;
pub mod error;
pub mod types;

// Legacy support
#[cfg(feature = "roqoqo")]
pub mod quantum_circuits;

// Re-exports
pub use optimizer::NeuromorphicQuantumOptimizer;
pub use error::{NqoError, NqoResult};
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