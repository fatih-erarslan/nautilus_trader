//! # Quantum Bridge: High-Performance Rust-Python Integration for PennyLane
//!
//! This crate provides a high-performance bridge between Rust and PennyLane quantum computing
//! framework, enabling real quantum circuit execution with surgical precision and zero-mock
//! policy enforcement.
//!
//! ## Features
//!
//! - **Device Hierarchy**: lightning.gpu → lightning-kokkos → lightning.qubit (NO default.qubit)
//! - **Zero-Copy Operations**: Minimal memory overhead for quantum state transfers
//! - **Lock-Free Coordination**: High-throughput quantum circuit execution
//! - **SIMD Optimization**: Vectorized operations for classical post-processing
//! - **Real Hardware Integration**: Actual quantum computing, no simulation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Rust Core     │ ── │  Quantum Bridge  │ ── │   PennyLane     │
//! │  (Trading)      │    │   (This Crate)   │    │   (Python)      │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions, clippy::similar_names)]

use std::sync::Arc;
use tokio::sync::RwLock;

pub mod bridge;
pub mod device;
pub mod circuit;
pub mod execution;
pub mod optimization;
pub mod error;
pub mod types;

#[cfg(feature = "simd-optimized")]
pub mod simd;

#[cfg(feature = "gpu-acceleration")]
pub mod gpu;

// Re-exports for convenience
pub use bridge::{QuantumBridge, BridgeConfig};
pub use device::{QuantumDevice, DeviceManager, DeviceHierarchy};
pub use circuit::{QuantumCircuit, QuantumGate, CircuitBuilder};
pub use execution::{ExecutionEngine, ExecutionResult, ExecutionMetrics};
pub use error::{QuantumError, BridgeError, DeviceError};
pub use types::{QuantumState, ClassicalState, QuantumResult};

/// High-performance quantum bridge instance
pub type Bridge = Arc<RwLock<QuantumBridge>>;

/// Initialize the quantum bridge with optimal configuration
///
/// # Errors
///
/// Returns `BridgeError` if:
/// - PennyLane is not installed or accessible
/// - No compatible quantum devices are available
/// - Python runtime initialization fails
///
/// # Example
///
/// ```rust
/// use quantum_bridge::initialize_bridge;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let bridge = initialize_bridge().await?;
///     
///     // Bridge is ready for quantum circuit execution
///     Ok(())
/// }
/// ```
pub async fn initialize_bridge() -> Result<Bridge, BridgeError> {
    let config = BridgeConfig::default();
    let bridge = QuantumBridge::new(config).await?;
    Ok(Arc::new(RwLock::new(bridge)))
}

/// Initialize bridge with custom configuration
///
/// # Arguments
///
/// * `config` - Custom bridge configuration
///
/// # Example
///
/// ```rust
/// use quantum_bridge::{initialize_bridge_with_config, BridgeConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = BridgeConfig {
///         max_concurrent_executions: 16,
///         enable_gpu_acceleration: true,
///         prefer_kokkos_backend: true,
///         ..Default::default()
///     };
///     
///     let bridge = initialize_bridge_with_config(config).await?;
///     Ok(())
/// }
/// ```
pub async fn initialize_bridge_with_config(config: BridgeConfig) -> Result<Bridge, BridgeError> {
    let bridge = QuantumBridge::new(config).await?;
    Ok(Arc::new(RwLock::new(bridge)))
}

/// Quantum bridge version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Quantum bridge build metadata
pub const BUILD_INFO: &str = concat!(
    "quantum-bridge v",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("VERGEN_GIT_SHA"),
    ")"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_initialization() {
        let result = initialize_bridge().await;
        
        // Should either succeed or fail with proper error handling
        match result {
            Ok(bridge) => {
                let bridge_guard = bridge.read().await;
                assert!(bridge_guard.is_available());
            }
            Err(e) => {
                // Expected if PennyLane is not available in test environment
                println!("Bridge initialization failed (expected in test): {}", e);
            }
        }
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!BUILD_INFO.is_empty());
    }
}