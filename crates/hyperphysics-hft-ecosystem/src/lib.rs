//! HyperPhysics HFT Ecosystem
//!
//! Enterprise-grade high-frequency trading ecosystem integrating:
//! - HyperPhysics: pBit dynamics, hyperbolic geometry, consciousness metrics
//! - 7 Physics Engines: JoltPhysics, Rapier, Avian, Warp, Taichi, MuJoCo, Genesis
//! - 14+ Biomimetic Algorithms: Dynamic swarm intelligence
//! - Formal Verification: Z3 SMT + Lean 4 + Property-based testing
//!
//! # Features
//!
//! - **Sub-millisecond latency**: <1ms execution for Tier 1 algorithms
//! - **Deterministic replay**: Regulatory compliance via JoltPhysics
//! - **GPU acceleration**: 100-1000× speedup via Warp/Taichi
//! - **Formal verification**: Institution-grade correctness proofs
//! - **Multi-language**: Rust/WASM/TypeScript/Python/C++ bindings
//!
//! # Architecture
//!
//! ```text
//! Tier 0: Formal Verification (Z3 + Lean 4)
//!    ↓
//! Tier 1: Core Physics (HyperPhysics + JoltPhysics + Rapier + Avian)
//!    ↓
//! Tier 2: GPU Acceleration (Warp + Taichi + MuJoCo + Genesis)
//!    ↓  
//! Tier 3: Biomimetic Swarms (14+ Algorithms)
//!    ↓
//! Tier 4: Trading Systems (CWTS-Ultra + ATS-Core)
//!    ↓
//! Tier 5: Language Bindings (WASM + TypeScript + Python + C++)
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use hyperphysics_hft_ecosystem::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the ecosystem
//!     let ecosystem = HFTEcosystem::builder()
//!         .with_physics_engine(PhysicsEngine::Rapier)
//!         .with_biomimetic_tier(BiomimeticTier::Tier1)
//!         .with_formal_verification(true)
//!         .build()
//!         .await?;
//!
//!     // Execute HFT cycle
//!     let market_tick = MarketTick::default();
//!     let decision = ecosystem.execute_cycle(&market_tick).await?;
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod core;
pub mod execution;
pub mod swarms;
pub mod trading;

#[cfg(feature = "verification")]
pub mod verification;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "wasm-bindings")]
pub mod bindings;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::*;
    // Swarms module provides biomimetic algorithms
    #[cfg(feature = "biomimetic-tier1")]
    pub use crate::swarms::*;
    // Execution infrastructure (requires explicit import)
    // pub use crate::execution::*;
}

use thiserror::Error;

/// Main error type for the HFT ecosystem
#[derive(Error, Debug)]
pub enum EcosystemError {
    /// Physics engine integration error
    #[error("Physics engine error: {0}")]
    PhysicsEngine(String),

    /// Biomimetic algorithm error
    #[error("Biomimetic algorithm error: {0}")]
    BiomimeticAlgorithm(String),

    /// Trading system error
    #[error("Trading system error: {0}")]
    TradingSystem(String),

    /// Formal verification error
    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    /// HyperPhysics integration error
    #[error("HyperPhysics error: {0}")]
    HyperPhysics(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// I/O error
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Any other error
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for the HFT ecosystem
pub type Result<T> = std::result::Result<T, EcosystemError>;

// Implement From for RapierHyperPhysicsError when physics-rapier feature is enabled
#[cfg(feature = "physics-rapier")]
impl From<rapier_hyperphysics::RapierHyperPhysicsError> for EcosystemError {
    fn from(err: rapier_hyperphysics::RapierHyperPhysicsError) -> Self {
        EcosystemError::PhysicsEngine(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(env!("CARGO_PKG_VERSION"), "0.1.0");
    }
}
