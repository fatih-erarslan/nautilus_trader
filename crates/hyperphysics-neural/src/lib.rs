//! # HyperPhysics Neural Intelligence Layer
//!
//! Central neural network crate for HyperPhysics HFT ecosystem.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    hyperphysics-neural                          │
//! │                                                                 │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
//! │  │   Core NN   │  │ Forecasting │  │  Surrogate  │             │
//! │  │  (ruv-FANN) │  │ (N-BEATS,   │  │  Physics    │             │
//! │  │             │  │  LSTM, etc) │  │             │             │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
//! │         │                │                │                     │
//! │         └────────────────┼────────────────┘                     │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │    NeuralRouter       │                          │
//! │              │  (Meta-Learning)      │                          │
//! │              └───────────┬───────────┘                          │
//! │                          │                                      │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │  ReasoningBackend     │                          │
//! │              │  Integration          │                          │
//! │              └───────────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **core**: Base neural network types (Network, Layer, Activation)
//! - **forecasting**: Time-series models (N-BEATS, LSTM, Transformer)
//! - **surrogate**: Fast physics approximation via trained networks
//! - **meta-router**: Neural-guided backend selection
//! - **gpu**: WebGPU acceleration for inference
//!
//! ## HFT Application
//!
//! This crate serves as the central neural intelligence hub for:
//! - Market regime detection
//! - Latency-critical inference (<1μs)
//! - Physics-informed trading signals
//! - Adaptive routing optimization

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod activation;
pub mod core;
pub mod error;
pub mod layer;
pub mod network;

#[cfg(feature = "forecasting")]
pub mod forecasting;

#[cfg(feature = "surrogate")]
pub mod surrogate;

#[cfg(feature = "meta-router")]
pub mod meta_router;

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod backend;

/// ruv-FANN integration for high-performance neural networks
pub mod fann;

/// ATS-Core integration for conformal prediction and temperature scaling
#[cfg(feature = "ats")]
pub mod ats;

pub mod prelude {
    //! Convenience re-exports
    pub use crate::activation::Activation;
    pub use crate::core::{Tensor, TensorShape};
    pub use crate::error::{NeuralError, NeuralResult};
    pub use crate::layer::{Layer, LayerConfig, LayerType};
    pub use crate::network::{Network, NetworkConfig, NetworkBuilder};
    pub use crate::backend::NeuralBackend;
    pub use crate::fann::{FannNetwork, FannConfig, FannBackend};

    #[cfg(feature = "forecasting")]
    pub use crate::forecasting::{Forecaster, ForecastConfig, TimeSeriesModel};

    #[cfg(feature = "surrogate")]
    pub use crate::surrogate::{SurrogatePhysics, SurrogateConfig};

    #[cfg(feature = "meta-router")]
    pub use crate::meta_router::{NeuralRouter, RouterPolicy};

    #[cfg(feature = "ats")]
    pub use crate::ats::{
        CalibratedFannBackend, FastConformalPredictor, NeuralCalibrator,
        UncertaintyBounds, UncertaintyAwareResult, ConformalVariant,
    };

    #[cfg(feature = "architectures")]
    pub use crate::ats::architectures::{ArchitectureCatalog, ArchitectureCategory, ArchitectureSpec};
}

/// Re-export common types
pub use activation::Activation;
pub use core::{Tensor, TensorShape};
pub use error::{NeuralError, NeuralResult};
pub use layer::{Layer, LayerConfig, LayerType};
pub use network::{Network, NetworkConfig, NetworkBuilder};
pub use backend::NeuralBackend;
pub use fann::{FannNetwork, FannConfig, FannBackend};

#[cfg(feature = "ats")]
pub use ats::{
    CalibratedFannBackend, FastConformalPredictor, NeuralCalibrator,
    UncertaintyBounds, UncertaintyAwareResult, ConformalVariant,
    CalibratedFannConfig, ConformalConfig, CalibrationConfig,
};

#[cfg(feature = "architectures")]
pub use ats::architectures::{ArchitectureCatalog, ArchitectureCategory, ArchitectureSpec};
