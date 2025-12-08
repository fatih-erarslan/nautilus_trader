//! # Risk Management System
//!
//! pBit-enhanced risk management for quantitative trading.
//!
//! ## Features
//!
//! - **pBit VaR/CVaR**: Boltzmann sampling for tail risk (Wolfram validated)
//! - **Ising Correlations**: Asset correlations via coupling J = arctanh(ρ)
//! - **Stress Testing**: pBit annealing for extreme scenarios
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - VaR: z_0.95 = 1.6449, z_0.99 = 2.3263
//! - CVaR: φ(z_α)/α → CVaR(95%)/σ = 2.0627
//! - pBit: P(up) = 1/(1 + exp(E/T))
//! - Correlation: ρ ≈ tanh(J)

#![warn(missing_docs)]

// Core types
pub mod error;
pub mod types;

// pBit-enhanced risk (real implementation)
pub mod pbit_risk;

// Re-exports
pub use error::{RiskError, RiskResult};
pub use types::{Portfolio, Position, Asset, AssetClass, MarketData};
pub use pbit_risk::{PBitRiskEngine, PBitVarResult, PBitStressResult, PBitRiskConfig, PBitState};

// Prelude
pub mod prelude {
    //! Convenient imports
    pub use crate::error::{RiskError, RiskResult};
    pub use crate::types::*;
    pub use crate::pbit_risk::*;
}
