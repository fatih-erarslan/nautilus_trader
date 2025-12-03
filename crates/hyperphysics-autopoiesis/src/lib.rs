//! # HyperPhysics-Autopoiesis Integration Bridge
//!
//! This crate provides seamless integration between the autopoiesis framework
//! (Maturana-Varela autopoiesis, Prigogine dissipative structures, Bateson ecology
//! of mind, Grinberg syntergy) and the HyperPhysics physics-based trading ecosystem.
//!
//! ## Theoretical Foundation
//!
//! Based on peer-reviewed research:
//! - Maturana & Varela (1980) "Autopoiesis and Cognition" D. Reidel
//! - Prigogine & Stengers (1984) "Order Out of Chaos" Bantam
//! - Bateson (1972) "Steps to an Ecology of Mind" Ballantine
//! - Grinberg-Zylberbaum (1995) "Syntergic Theory" INPEC
//! - Strogatz (2003) "Sync: The Emerging Science of Spontaneous Order" Hyperion
//! - Capra (1996) "The Web of Life" Anchor
//! - Bak (1996) "How Nature Works: Self-Organized Criticality" Copernicus
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                      Autopoiesis Workspace                               │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
//! │  │ AutopoieticSystem│  │DissipativeStruct│  │ Syntergic + EcologyMind│  │
//! │  │ (Maturana-Varela)│  │ (Prigogine)     │  │ (Grinberg + Bateson)   │  │
//! │  └────────┬─────────┘  └────────┬────────┘  └───────────┬────────────┘  │
//! └───────────┼──────────────────────┼───────────────────────┼──────────────┘
//!             │                      │                       │
//!             ▼                      ▼                       ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │              hyperphysics-autopoiesis Bridge Layer                       │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
//! │  │ ThermoAdapter   │  │ ConsciousnessAdp│  │ NeuralTraderBridge     │  │
//! │  │ (Entropy↔Dissip)│  │ (Φ↔Syntergy)    │  │ (NHITS↔Autopoietic)    │  │
//! │  └────────┬────────┘  └────────┬────────┘  └───────────┬────────────┘  │
//! └───────────┼──────────────────────┼───────────────────────┼──────────────┘
//!             │                      │                       │
//!             ▼                      ▼                       ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     HyperPhysics Ecosystem                               │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
//! │  │ hyperphysics-   │  │ hyperphysics-   │  │ hyperphysics-neural-   │  │
//! │  │ thermo          │  │ consciousness   │  │ trader                 │  │
//! │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Core Integration Points
//!
//! | Autopoiesis Component | HyperPhysics Component | Bridge Mechanism |
//! |----------------------|------------------------|------------------|
//! | `DissipativeStructure::entropy_production()` | `hyperphysics-thermo::EntropyCalculator` | `ThermoAdapter` |
//! | `Syntergic::neuronal_field_coherence()` | `hyperphysics-consciousness::PhiCalculator` | `ConsciousnessAdapter` |
//! | `AutopoieticSystem::autopoietic_cycle()` | `hyperphysics-neural-trader::NeuralForecastEngine` | `AutopoieticTrader` |
//! | `SynchronizationDynamics::kuramoto_order_parameter()` | `hyperphysics-syntergic::SyntergicField` | `SyncAdapter` |
//! | `WebOfLife::network_pattern()` | `hyperphysics-risk::RiskNetwork` | `NetworkAdapter` |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_autopoiesis::prelude::*;
//!
//! // Create autopoietic trading system
//! let config = AutopoieticTradingConfig::default();
//! let system = AutopoieticTradingSystem::new(config)?;
//!
//! // Run autopoietic cycle with market data
//! let result = system.autopoietic_cycle(&market_data).await?;
//!
//! // Access emergent trading signals
//! println!("Signal: {:?}, Coherence: {:.4}", result.signal, result.coherence);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

pub mod adapters;
pub mod bridges;
pub mod dynamics;
pub mod emergence;
pub mod trading;
pub mod error;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adapters::{
        ThermoAdapter, ConsciousnessAdapter, SyncAdapter, NetworkAdapter,
    };
    pub use crate::bridges::{
        AutopoieticBridge, DissipativeBridge, SyntergicBridge,
    };
    pub use crate::dynamics::{
        AutopoieticDynamics, BifurcationDetector, EmergenceMonitor,
    };
    pub use crate::emergence::{
        EmergentPattern, EmergenceEvent, CollectiveState,
    };
    pub use crate::trading::{
        AutopoieticTradingConfig, AutopoieticTradingSystem,
        AutopoieticSignal, TradingResult,
    };
    pub use crate::error::{AutopoiesisError, Result};
}

// Re-exports for convenience
pub use adapters::{ThermoAdapter, ConsciousnessAdapter, SyncAdapter, NetworkAdapter};
pub use bridges::{AutopoieticBridge, DissipativeBridge, SyntergicBridge};
pub use dynamics::{AutopoieticDynamics, BifurcationDetector, EmergenceMonitor};
pub use emergence::{EmergentPattern, EmergenceEvent, CollectiveState};
pub use trading::{AutopoieticTradingConfig, AutopoieticTradingSystem, AutopoieticSignal, TradingResult};
pub use error::{AutopoiesisError, Result};

/// Boltzmann constant for thermodynamic calculations (J/K)
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Critical coherence threshold for syntergic unity detection
pub const SYNTERGIC_UNITY_THRESHOLD: f64 = 0.9;

/// Minimum operational closure ratio for autopoietic health
pub const OPERATIONAL_CLOSURE_THRESHOLD: f64 = 0.8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!((BOLTZMANN_CONSTANT - 1.380649e-23).abs() < 1e-30);
        assert!((SYNTERGIC_UNITY_THRESHOLD - 0.9).abs() < f64::EPSILON);
        assert!((OPERATIONAL_CLOSURE_THRESHOLD - 0.8).abs() < f64::EPSILON);
    }
}
