//! HyperPhysics-Nautilus Integration Bridge
//!
//! This crate provides seamless integration between the HyperPhysics physics-based
//! trading ecosystem and NautilusTrader's production-grade execution infrastructure.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Nautilus Trader                               │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ DataEngine  │  │ ExecEngine  │  │ Venue Adapters          │  │
//! │  │ (QuoteTick) │  │ (Orders)    │  │ (Binance, IB, Bybit...) │  │
//! │  └──────┬──────┘  └──────▲──────┘  └─────────────────────────┘  │
//! └─────────┼────────────────┼──────────────────────────────────────┘
//!           │                │
//!           ▼                │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │              HyperPhysics-Nautilus Bridge                        │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
//! │  │ NautilusAdapter │  │ TypeConversions │  │ ExecBridge      │  │
//! │  │ (Data → Feed)   │  │ (NT ↔ HP)       │  │ (Signal → Ord)  │  │
//! │  └────────┬────────┘  └─────────────────┘  └────────▲────────┘  │
//! └───────────┼────────────────────────────────────────┼────────────┘
//!             │                                         │
//!             ▼                                         │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   HyperPhysics Core                              │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
//! │  │ Physics     │  │ Biomimetic  │  │ Byzantine Consensus     │  │
//! │  │ Simulation  │  │ Optimization│  │ (PBFT/Raft)             │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Type-safe conversions**: Zero-copy where possible between NT and HP types
//! - **Actor integration**: HyperPhysics runs as a Nautilus Strategy actor
//! - **Backtest support**: Use HP physics with NT's MatchingEngine
//! - **Live trading ready**: Same code path for backtest and live
//!
//! # Example
//!
//! ```rust,ignore
//! use hyperphysics_nautilus::prelude::*;
//!
//! // Create the integration bridge
//! let config = IntegrationConfig::default();
//! let bridge = NautilusBridge::new(config)?;
//!
//! // Convert Nautilus quote to HyperPhysics feed
//! let feed = bridge.quote_to_feed(&quote_tick)?;
//!
//! // Execute HyperPhysics pipeline
//! let decision = bridge.execute_pipeline(&feed).await?;
//!
//! // Convert back to Nautilus order
//! let order = bridge.decision_to_order(&decision)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod adapter;
pub mod backtest;
pub mod config;
pub mod error;
pub mod strategy;
pub mod types;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adapter::{NautilusDataAdapter, NautilusExecBridge};
    pub use crate::config::IntegrationConfig;
    pub use crate::error::{IntegrationError, Result};
    pub use crate::strategy::HyperPhysicsStrategy;
    pub use crate::types::*;
}

// Re-exports for convenience
pub use adapter::{NautilusDataAdapter, NautilusExecBridge};
pub use config::IntegrationConfig;
pub use error::{IntegrationError, Result};
pub use strategy::HyperPhysicsStrategy;
