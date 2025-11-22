//! # Quantum LSTM
//! 
//! Quantum-enhanced LSTM implementation with biological quantum effects for cryptocurrency trading.
//! This crate provides a high-performance, quantum-inspired LSTM architecture that incorporates:
//! 
//! - Quantum state encoding for efficient data representation
//! - Quantum gates for LSTM cell operations
//! - Biological quantum effects (tunneling, coherence, criticality)
//! - Multi-head quantum attention mechanisms
//! - Quantum associative memory with error correction
//! - GPU acceleration support via PennyLane bridge
//! 
//! ## Features
//! 
//! - **Quantum State Encoding**: Multiple encoding schemes (amplitude, angle, basis)
//! - **Quantum LSTM Gates**: Forget, input, output, and cell update gates using quantum circuits
//! - **Biological Effects**: Quantum tunneling, coherence maintenance, and criticality detection
//! - **Quantum Attention**: Multi-head attention using quantum inner products
//! - **Quantum Memory**: Associative memory with quantum error correction
//! - **Hardware Acceleration**: GPU, CUDA, ROCm, Metal support
//! - **Real-time Processing**: Optimized for sub-microsecond latency
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use quantum_lstm::{QuantumLSTM, QuantumLSTMConfig};
//! 
//! let config = QuantumLSTMConfig::default()
//!     .with_num_qubits(8)
//!     .with_num_layers(2)
//!     .with_biological_effects(true);
//! 
//! let mut qlstm = QuantumLSTM::new(config)?;
//! 
//! // Process time series data
//! let input = ndarray::Array3::zeros((batch_size, seq_len, input_size));
//! let output = qlstm.forward(&input).await?;
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod config;
pub mod core;
pub mod encoding;
pub mod gates;
pub mod attention;
pub mod memory;
pub mod biological;
pub mod cell;
pub mod model;
pub mod cache;
pub mod utils;
pub mod error;
pub mod types;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "candle")]
pub mod neural;

#[cfg(test)]
mod tests;

// Re-exports
pub use config::QuantumLSTMConfig;
pub use core::{QuantumDevice, QuantumCircuit};
pub use encoding::QuantumStateEncoder;
pub use gates::QuantumLSTMGate;
pub use attention::QuantumAttention;
pub use memory::QuantumMemory;
pub use biological::BiologicalQuantumEffects;
pub use cell::QuantumLSTMCell;
pub use model::QuantumLSTM;
pub use error::{QuantumLSTMError, Result};
pub use types::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize the quantum LSTM library
pub fn init() -> Result<()> {
    // Initialize logging
    tracing::info!("Initializing Quantum LSTM v{}", VERSION);
    
    // Initialize hardware acceleration if available
    #[cfg(feature = "gpu")]
    gpu::init_gpu()?;
    
    #[cfg(feature = "mkl")]
    {
        // MKL is initialized automatically when linked
        tracing::info!("Intel MKL acceleration enabled");
    }
    
    #[cfg(feature = "metal")]
    {
        // Metal is initialized automatically on macOS
        tracing::info!("Apple Metal acceleration enabled");
    }
    
    Ok(())
}

/// Get library information
pub fn info() -> LibraryInfo {
    LibraryInfo {
        version: VERSION.to_string(),
        features: get_enabled_features(),
        quantum_backend: get_quantum_backend(),
        hardware_acceleration: get_hardware_acceleration(),
    }
}

/// Library information structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LibraryInfo {
    /// Library version
    pub version: String,
    /// Enabled features
    pub features: Vec<String>,
    /// Quantum backend in use
    pub quantum_backend: String,
    /// Hardware acceleration status
    pub hardware_acceleration: Vec<String>,
}

fn get_enabled_features() -> Vec<String> {
    let mut features = Vec::new();
    
    #[cfg(feature = "simd")]
    features.push("simd".to_string());
    
    #[cfg(feature = "parallel")]
    features.push("parallel".to_string());
    
    #[cfg(feature = "real-time")]
    features.push("real-time".to_string());
    
    #[cfg(feature = "memory-efficient")]
    features.push("memory-efficient".to_string());
    
    #[cfg(feature = "biological-effects")]
    features.push("biological-effects".to_string());
    
    #[cfg(feature = "gpu")]
    features.push("gpu".to_string());
    
    #[cfg(feature = "candle")]
    features.push("candle".to_string());
    
    #[cfg(feature = "monitoring")]
    features.push("monitoring".to_string());
    
    features
}

fn get_quantum_backend() -> String {
    // In the future, this will detect PennyLane, Qiskit, etc.
    "Simulated".to_string()
}

fn get_hardware_acceleration() -> Vec<String> {
    let mut accel = Vec::new();
    
    #[cfg(feature = "gpu")]
    {
        #[cfg(feature = "cuda")]
        accel.push("CUDA".to_string());
        
        #[cfg(feature = "rocm")]
        accel.push("ROCm".to_string());
    }
    
    #[cfg(feature = "metal")]
    accel.push("Metal".to_string());
    
    #[cfg(feature = "mkl")]
    accel.push("Intel MKL".to_string());
    
    if accel.is_empty() {
        accel.push("CPU".to_string());
    }
    
    accel
}