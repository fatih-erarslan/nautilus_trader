//! # HyperPhysics-Tengri Bridge
//!
//! Integration bridge connecting the Tengri trading strategy with HyperPhysics
//! scientific computing crates for enhanced market analysis and trading signals.
//!
//! ## Scientific Foundations
//!
//! This bridge integrates multiple scientific frameworks:
//!
//! - **Autopoiesis** (Maturana & Varela): Market regime detection via self-organizing systems
//! - **Integrated Information Theory** (Tononi): Market coherence metrics via Φ
//! - **Thermodynamics** (Prigogine): Entropy production for volatility analysis
//! - **Codependent Risk Models**: Enhanced portfolio risk with network effects
//! - **P-Bits**: Probabilistic computing for signal uncertainty quantification
//! - **Quantum Circuits**: Real quantum gate operations for pattern detection
//! - **Syntergy** (Grinberg): Coherence field integration for market synchronization
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    HyperPhysicsTradingSystem                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
//! │  │ Autopoiesis   │  │ Consciousness │  │    Thermo     │       │
//! │  │  Integration  │  │  Integration  │  │  Integration  │       │
//! │  │ (Regime Det.) │  │   (IIT Φ)     │  │  (Entropy)    │       │
//! │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
//! │          │                  │                  │                │
//! │  ┌───────┴───────┐  ┌───────┴───────┐  ┌───────┴───────┐       │
//! │  │     Risk      │  │     Pbit      │  │    Quantum    │       │
//! │  │  Integration  │  │  Integration  │  │  Integration  │       │
//! │  │ (Codependent) │  │ (Uncertainty) │  │   (Gates)     │       │
//! │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │
//! │          │                  │                  │                │
//! │  ┌───────┴──────────────────┴──────────────────┴───────┐       │
//! │  │              Syntergic Integration                   │       │
//! │  │           (Coherence Field Analysis)                 │       │
//! │  └─────────────────────────┬───────────────────────────┘       │
//! │                            │                                    │
//! │                   ┌────────┴────────┐                          │
//! │                   │ Trading Signals │                          │
//! │                   └─────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_tengri_bridge::{
//!     HyperPhysicsTradingSystem, TradingConfig, MarketData
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = TradingConfig::default();
//!     let mut system = HyperPhysicsTradingSystem::new(config)?;
//!
//!     // Process market data
//!     let market_data = MarketData::new(prices, volumes);
//!     let signal = system.process_market_data(&market_data).await?;
//!
//!     println!("Signal: {:?}, Confidence: {:.4}", signal.direction, signal.confidence);
//!     Ok(())
//! }
//! ```

pub mod adapters;
pub mod error;
pub mod integrations;
pub mod trading;

// Re-export main types
pub use adapters::{
    AutopoiesisAdapter, ConsciousnessAdapter, PbitAdapter, QuantumAdapter, RiskAdapter,
    SyntergicAdapter, ThermoAdapter,
};
pub use error::{BridgeError, Result};
pub use integrations::{
    AutopoiesisIntegration, ConsciousnessIntegration, PbitIntegration, QuantumIntegration,
    RiskIntegration, SyntergicIntegration, ThermoIntegration,
};
pub use trading::{
    HyperPhysicsSignal, HyperPhysicsTradingSystem, MarketData, MarketRegime, SignalDirection,
    TradingConfig, TradingResult,
};

/// Boltzmann constant (J/K)
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Planck constant (J·s)
pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;

/// Speed of light (m/s)
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(BOLTZMANN_CONSTANT > 0.0);
        assert!(PLANCK_CONSTANT > 0.0);
        assert!(SPEED_OF_LIGHT > 0.0);
    }
}
