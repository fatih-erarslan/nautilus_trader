//! # HyperPhysics Plugin
//!
//! Drop-in plugin for accessing HyperPhysics swarm intelligence from any Rust application.
//!
//! ## Quick Start
//!
//! ```rust
//! use hyperphysics_plugin::prelude::*;
//!
//! // Simple optimization
//! let result = HyperPhysics::optimize()
//!     .dimensions(10)
//!     .bounds(-100.0, 100.0)
//!     .strategy(Strategy::GreyWolf)
//!     .minimize(|x| x.iter().map(|xi| xi * xi).sum())
//!     .unwrap();
//!
//! println!("Best: {:?} = {}", result.solution, result.fitness);
//! ```
//!
//! ## Features
//!
//! - **14+ biomimetic algorithms**: PSO, Grey Wolf, Whale, Firefly, Cuckoo, etc.
//! - **8 network topologies**: Star, Ring, Mesh, Hyperbolic, SmallWorld, etc.
//! - **pBit lattice**: Probabilistic computing fabric
//! - **Evolution engine**: Evolve optimal strategies
//! - **Knowledge learning**: Record insights, get recommendations

#![allow(clippy::module_name_repetitions)]

pub mod optimizer;
pub mod swarm;
pub mod lattice;
pub mod builder;
pub mod types;

#[cfg(feature = "serde")]
pub mod persistence;

// Re-exports for easy access
pub use optimizer::{Optimizer, OptimizationResult};
pub use swarm::{Swarm, SwarmBuilder, SwarmResult as SwarmOptResult};
pub use lattice::{Lattice, LatticeBuilder};
pub use builder::{HyperPhysics, OptimizeBuilder};
pub use types::*;

/// Prelude module - import everything you need
pub mod prelude {
    pub use crate::optimizer::{Optimizer, OptimizationResult};
    pub use crate::swarm::{Swarm, SwarmBuilder};
    pub use crate::lattice::{Lattice, LatticeBuilder};
    pub use crate::builder::{HyperPhysics, OptimizeBuilder};
    pub use crate::types::*;
    
    #[cfg(feature = "serde")]
    pub use crate::persistence::*;
}

/// Error types
#[derive(thiserror::Error, Debug)]
pub enum HyperPhysicsError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Optimization failed
    #[error("Optimization failed: {0}")]
    Optimization(String),
    
    /// Invalid bounds
    #[error("Invalid bounds: {0}")]
    Bounds(String),
    
    /// Convergence failure
    #[error("Convergence failure: {0}")]
    Convergence(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for HyperPhysics operations
pub type Result<T> = std::result::Result<T, HyperPhysicsError>;
