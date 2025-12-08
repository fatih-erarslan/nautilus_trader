//! # HyperPhysics Swarm Intelligence
//!
//! Meta-swarm system combining:
//! - **pBit SpatioTemporal Lattice**: Quantum-inspired probabilistic computing
//! - **Biomimetic Strategies**: 14+ animal-inspired algorithms
//! - **Multiple Topologies**: Star, ring, mesh, hierarchical, hyperbolic
//! - **Emergent Intelligence**: Strategy combination creates new behaviors
//! - **Evolution Engine**: Record and evolve successful strategies
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    META-SWARM INTELLIGENCE                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
//! │  │   LATTICE   │◄─►│  TOPOLOGY   │◄─►│  STRATEGY   │               │
//! │  │  pBit Grid  │   │   Manager   │   │   Engine    │               │
//! │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
//! │         │                 │                 │                       │
//! │         └─────────────────┼─────────────────┘                       │
//! │                           ▼                                         │
//! │                   ┌──────────────┐                                  │
//! │                   │  EVOLUTION   │                                  │
//! │                   │    ENGINE    │                                  │
//! │                   │  (Intellect) │                                  │
//! │                   └──────────────┘                                  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod lattice;
pub mod topology;
pub mod strategy;
pub mod evolution;
pub mod swarm;
pub mod intellect;

// Re-exports
pub use lattice::{PBitLattice, LatticeNode, LatticeConfig, SpatioTemporalState};
pub use topology::{SwarmTopology, TopologyType, Connection, TopologyMetrics};
pub use strategy::{BiomimeticStrategy, StrategyType, StrategyConfig, StrategyResult};
pub use evolution::{EvolutionEngine, Genome, Fitness, Generation, EvolutionConfig};
pub use swarm::{MetaSwarm, SwarmAgent, SwarmState, SwarmConfig};
pub use intellect::{EmergentIntellect, IntellectRecord, KnowledgeGraph, Insight};

/// Prelude for common imports
pub mod prelude {
    pub use crate::lattice::*;
    pub use crate::topology::*;
    pub use crate::strategy::*;
    pub use crate::evolution::*;
    pub use crate::swarm::*;
    pub use crate::intellect::*;
}

/// Error types for swarm intelligence
#[derive(thiserror::Error, Debug)]
pub enum SwarmIntelligenceError {
    /// Lattice configuration error
    #[error("Lattice error: {0}")]
    LatticeError(String),
    
    /// Topology error
    #[error("Topology error: {0}")]
    TopologyError(String),
    
    /// Strategy execution error
    #[error("Strategy error: {0}")]
    StrategyError(String),
    
    /// Evolution error
    #[error("Evolution error: {0}")]
    EvolutionError(String),
    
    /// Convergence failure
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
    
    /// Invalid configuration
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

/// Result type for swarm intelligence operations
pub type SwarmResult<T> = Result<T, SwarmIntelligenceError>;
