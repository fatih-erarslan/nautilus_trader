//! # HyperPhysics ML
//!
//! Multi-backend machine learning framework for high-frequency trading neural forecasting.
//!
//! ## Supported Backends
//!
//! | Backend | Platform | GPU | Feature Flag |
//! |---------|----------|-----|--------------|
//! | ndarray | All | CPU | `cpu` (default) |
//! | CUDA | Linux | NVIDIA | `cuda` |
//! | ROCm | Linux | AMD | `rocm` |
//! | Metal | macOS | Apple Silicon | `metal` |
//! | Vulkan | All | Any | `vulkan` |
//! | WebGPU | All | Any | `wgpu` |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_ml::prelude::*;
//!
//! // Automatic backend selection based on platform
//! let backend = Backend::auto();
//!
//! // Create LSTM forecaster
//! let config = LstmConfig::default()
//!     .with_hidden_size(256)
//!     .with_num_layers(2);
//!
//! let model = LstmForecaster::new(config, &backend);
//!
//! // Run inference
//! let input = Tensor::from_slice(&market_data, &backend);
//! let forecast = model.forward(input);
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod backends;
pub mod error;
pub mod layers;
pub mod models;
pub mod quantum;
pub mod tensor;
pub mod utils;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::backends::{Backend, BackendType, Device};
    pub use crate::error::{MlError, MlResult};
    pub use crate::layers::*;
    pub use crate::models::*;
    pub use crate::tensor::{Tensor, TensorOps};

    // Quantum-inspired types
    pub use crate::quantum::{
        // Core types
        Complex, StateVector, EncodingType, BiologicalEffect,
        QuantumHiddenState, QuantumLSTMOutput, CoherenceMetrics, GateType,
        // Configuration
        QuantumLSTMConfig, BioCognitiveConfig,
        // Encoding
        StateEncoder, TimeSeriesEncoder,
        // Models
        BioCognitiveLSTM, BioCognitiveState,
        ComplexLSTM, QuantumInspiredLSTM,
    };
}

pub use error::{MlError, MlResult};
