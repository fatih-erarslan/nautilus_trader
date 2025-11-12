//! # HyperPhysics Core Engine
//!
//! Integrated physics engine combining:
//! - Hyperbolic geometry (H³ manifold, K=-1)
//! - pBit stochastic dynamics
//! - Thermodynamics (Landauer principle)
//! - Consciousness metrics (Φ and CI)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hyperphysics_core::HyperPhysicsEngine;
//!
//! // Create 48-node ROI system
//! let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0).unwrap();
//!
//! // Run simulation
//! engine.step().unwrap();
//!
//! // Get metrics
//! let phi = engine.integrated_information().unwrap();
//! let ci = engine.resonance_complexity().unwrap();
//! ```

// Enable nightly features for SIMD
#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod engine;
pub mod config;
pub mod metrics;
pub mod crypto;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "simd")]
pub mod simd;

pub use engine::HyperPhysicsEngine;
pub use config::{EngineConfig, Scale};
pub use metrics::{EngineMetrics, SimulationState};

// Re-export key types
pub use hyperphysics_geometry::{PoincarePoint, HyperbolicTessellation};
pub use hyperphysics_pbit::{PBit, PBitLattice, PBitDynamics, Algorithm};
pub use hyperphysics_thermo::{
    HamiltonianCalculator, EntropyCalculator, LandauerEnforcer
};
pub use hyperphysics_consciousness::{
    IntegratedInformation, ResonanceComplexity,
    PhiCalculator, CICalculator
};

#[cfg(feature = "simd")]
pub use simd::{Backend, optimal_backend};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Geometry error: {0}")]
    Geometry(#[from] hyperphysics_geometry::GeometryError),

    #[error("pBit error: {0}")]
    PBit(#[from] hyperphysics_pbit::PBitError),

    #[error("Thermodynamics error: {0}")]
    Thermodynamics(#[from] hyperphysics_thermo::ThermoError),

    #[error("Consciousness error: {0}")]
    Consciousness(#[from] hyperphysics_consciousness::ConsciousnessError),

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Simulation error: {message}")]
    Simulation { message: String },
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = HyperPhysicsEngine::roi_48(1.0, 300.0);
        assert!(engine.is_ok());
    }
}
