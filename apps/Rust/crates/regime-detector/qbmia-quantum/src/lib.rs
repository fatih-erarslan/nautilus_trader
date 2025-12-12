//! QBMIA Quantum Backend with PennyLane Integration
//!
//! This crate provides quantum computing capabilities for QBMIA Nash equilibrium solving
//! using PennyLane as the quantum backend. It supports GPU acceleration through the
//! lightning.gpu device and provides fallback options for optimal performance.

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![feature(portable_simd)]

pub mod device;
pub mod nash;
pub mod circuits;
pub mod vqe;
pub mod qaoa;
pub mod backend;
pub mod cache;
pub mod error;
pub mod types;
pub mod utils;

pub use device::{QuantumDevice, DeviceManager};
pub use nash::{QuantumNashSolver, NashEquilibrium};
pub use circuits::{QuantumCircuit, CircuitBuilder};
pub use vqe::VQESolver;
pub use qaoa::QAOASolver;
pub use error::{QuantumError, Result};
pub use types::*;

use once_cell::sync::Lazy;
use std::sync::Arc;
use parking_lot::RwLock;

/// Global device manager for quantum computations
pub static DEVICE_MANAGER: Lazy<Arc<RwLock<DeviceManager>>> = Lazy::new(|| {
    Arc::new(RwLock::new(DeviceManager::new()))
});

/// Initialize the quantum backend with optimal device selection
pub fn initialize() -> Result<()> {
    tracing::info!("Initializing QBMIA quantum backend");
    let mut manager = DEVICE_MANAGER.write();
    manager.initialize_devices()?;
    tracing::info!("Quantum backend initialized successfully");
    Ok(())
}

/// Shutdown the quantum backend and release resources
pub fn shutdown() -> Result<()> {
    tracing::info!("Shutting down QBMIA quantum backend");
    let mut manager = DEVICE_MANAGER.write();
    manager.shutdown()?;
    tracing::info!("Quantum backend shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        assert!(initialize().is_ok());
        assert!(shutdown().is_ok());
    }
}