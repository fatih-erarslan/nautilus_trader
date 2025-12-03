//! # pBit-Aware Mixture of Experts Strategy Router
//!
//! This crate implements a Mixture of Experts (MoE) routing system that leverages
//! probabilistic computing (pBits) for strategy selection in high-frequency trading.
//!
//! ## Key Features
//!
//! - **pBit-Based Routing**: Uses probabilistic bits for stochastic, energy-efficient routing
//! - **Hyperbolic Expert Space**: Experts organized in hyperbolic geometry for hierarchical representation
//! - **Adaptive Load Balancing**: Dynamic capacity management with auxiliary losses
//! - **Market Regime Awareness**: Strategy selection adapts to market conditions
//!
//! ## Architecture
//!
//! ```text
//! Input Features → Gating Network → pBit Stochastic Selection → Top-K Experts → Output
//!                         ↓
//!                  Hyperbolic Space (Poincaré/Lorentz)
//! ```
//!
//! ## Mathematical Foundation
//!
//! The gating function incorporates pBit dynamics:
//!
//! ```text
//! P(expert_i | x) = softmax(W_g · x + pBit_noise(T))
//! ```
//!
//! where pBit_noise follows Boltzmann statistics at temperature T.
//!
//! ## References
//!
//! - Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
//! - Camsari et al. (2017) "Stochastic p-bits for invertible logic"
//! - Ganea et al. (2018) "Hyperbolic Neural Networks"

#![deny(missing_docs)]

mod error;
mod expert;
mod router;
mod gating;
mod pbit_noise;
mod market_regime;
mod load_balancer;

pub use error::{RouterError, Result};
pub use expert::{Expert, ExpertType, ExpertConfig, HyperbolicExpert, LinearExpert, StandardExpert};
pub use router::{StrategyRouter, RouterConfig, RoutingResult};
pub use gating::{GatingNetwork, GatingConfig};
pub use pbit_noise::{PBitNoiseGenerator, NoiseConfig};
pub use market_regime::{MarketRegime, RegimeDetector};
pub use load_balancer::{LoadBalancer, LoadBalancerConfig};
