//! HyperPhysics-Neural Trader Integration Bridge
//!
//! This crate provides seamless integration between the HyperPhysics physics-based
//! trading ecosystem and Neural Trader's ML forecasting capabilities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Neural Trader                                 │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ NHITS       │  │ Transformer │  │ Conformal Prediction    │  │
//! │  │ LSTM-Attn   │  │ GRU/TCN     │  │ (Uncertainty Bounds)    │  │
//! │  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
//! └─────────┼────────────────┼──────────────────────┼───────────────┘
//!           │                │                      │
//!           ▼                ▼                      ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              HyperPhysics-Neural Trader Bridge                   │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
//! │  │ NeuralAdapter   │  │ ForecastEngine  │  │ ConfidenceMgr   │  │
//! │  │ (Feed → Input)  │  │ (Model Ensemble)│  │ (Uncertainty)   │  │
//! │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
//! └───────────┼────────────────────┼────────────────────┼───────────┘
//!             │                    │                    │
//!             ▼                    ▼                    ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   HyperPhysics Pipeline                          │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ Market Data │→ │ Physics Sim │→ │ Optimization + Consensus│  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Neural Forecasting**: Time series prediction with NHITS, LSTM, Transformer
//! - **Confidence Estimation**: Conformal prediction for uncertainty bounds
//! - **Ensemble Methods**: Combine multiple neural models for robust predictions
//! - **GPU Acceleration**: CUDA/Metal support for fast inference
//!
//! # Example
//!
//! ```rust,ignore
//! use hyperphysics_neural_trader::prelude::*;
//!
//! // Create the neural bridge
//! let config = NeuralBridgeConfig::default();
//! let bridge = NeuralTradingBridge::new(config)?;
//!
//! // Generate neural forecast from market data
//! let forecast = bridge.forecast(&market_feed).await?;
//!
//! // Get prediction with confidence bounds
//! println!("Forecast: {:.4} (±{:.4})", forecast.prediction, forecast.confidence_interval);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod adapter;
pub mod config;
pub mod ensemble;
pub mod error;
pub mod forecast;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adapter::{MarketFeed, NeuralDataAdapter};
    pub use crate::config::NeuralBridgeConfig;
    pub use crate::ensemble::EnsemblePredictor;
    pub use crate::error::{NeuralBridgeError, Result};
    pub use crate::forecast::{ForecastResult, NeuralForecastEngine};
}

// Re-exports for convenience
pub use adapter::{MarketFeed, NeuralDataAdapter};
pub use config::NeuralBridgeConfig;
pub use ensemble::EnsemblePredictor;
pub use error::{NeuralBridgeError, Result};
pub use forecast::{ForecastResult, NeuralForecastEngine};
