//! # HyperPhysics Plugin
//!
//! Enterprise plugin for HyperPhysics ecosystem providing:
//! - **Swarm Intelligence**: 14+ biomimetic optimization algorithms
//! - **Hyperbolic Geometry**: H^11 Lorentz model and Poincaré ball operations
//! - **Consciousness Modeling**: IIT Phi and Self-Organized Criticality
//! - **Neural Learning**: Fibonacci STDP with multi-scale time constants
//! - **pBit Computing**: Probabilistic bits with Boltzmann statistics
//! - **Dilithium MCP**: Post-quantum secure AI agent coordination
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use hyperphysics_plugin::prelude::*;
//!
//! // Swarm optimization
//! let result = HyperPhysics::optimize()
//!     .dimensions(10)
//!     .bounds(-100.0, 100.0)
//!     .strategy(Strategy::GreyWolf)
//!     .minimize(|x| x.iter().map(|xi| xi * xi).sum())
//!     .unwrap();
//!
//! // Hyperbolic geometry
//! let p1 = LorentzPoint11::from_tangent_at_origin(&[0.5, 0.0, 0.0]);
//! let p2 = LorentzPoint11::from_tangent_at_origin(&[0.0, 0.5, 0.0]);
//! let distance = p1.distance(&p2);
//!
//! // Pentagon pBit system
//! let mut pentagon = PentagonPBit::at_criticality();
//! pentagon.sweep(&mut rng);
//! let coherence = pentagon.phase_coherence();
//!
//! // Fibonacci STDP learning
//! let stdp = FibonacciSTDP::new();
//! let dw = stdp.compute_weight_change(10.0); // 10ms timing
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                       HYPERPHYSICS PLUGIN v0.2                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
//! │  │   Swarm     │  │  Hyperbolic │  │Consciousness│  │   Neural    │    │
//! │  │Intelligence │  │  Geometry   │  │  Emergence  │  │  Learning   │    │
//! │  │  (14 algo)  │  │   (H^11)    │  │  (IIT Phi)  │  │   (STDP)    │    │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
//! │         │                │                │                │           │
//! │         └────────────────┼────────────────┼────────────────┘           │
//! │                          │                │                            │
//! │  ┌─────────────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────┐    │
//! │  │    pBit     │  │  Pentagon   │  │  Criticality │  │  Dilithium  │    │
//! │  │   Lattice   │  │  Topology   │  │  Analysis   │  │     MCP     │    │
//! │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - `full` (default): All features enabled
//! - `parallel`: Rayon parallel processing
//! - `serde`: Serialization support
//! - `async`: Async/await support
//! - `mcp`: Dilithium MCP client
//! - `hyperbolic`: Hyperbolic geometry
//! - `consciousness`: IIT and SOC metrics
//! - `neural`: STDP learning
//! - `pbit`: Probabilistic computing
//!
//! ## Modules
//!
//! - [`optimizer`]: Single-strategy optimization
//! - [`swarm`]: Multi-strategy swarm optimization
//! - [`lattice`]: pBit lattice substrate
//! - [`builder`]: Fluent builder API
//! - [`hyperbolic`]: H^11 geometry operations
//! - [`consciousness`]: IIT Phi and emergence
//! - [`neural`]: Fibonacci STDP learning
//! - [`pbit`]: Probabilistic computing
//! - [`dilithium_mcp`]: MCP client for AI coordination

#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Core Modules
// ============================================================================

pub mod optimizer;
pub mod swarm;
pub mod lattice;
pub mod builder;
pub mod types;

#[cfg(feature = "serde")]
pub mod persistence;

// ============================================================================
// Enhanced Modules
// ============================================================================

/// Hyperbolic geometry (H^11 Lorentz and Poincaré ball models)
#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;

/// Consciousness emergence (IIT Phi and Self-Organized Criticality)
#[cfg(feature = "consciousness")]
pub mod consciousness;

/// Neural learning (Fibonacci STDP and eligibility traces)
#[cfg(feature = "neural")]
pub mod neural;

/// Probabilistic computing (pBit dynamics and Ising models)
#[cfg(feature = "pbit")]
pub mod pbit;

/// Dilithium MCP client (post-quantum secure AI coordination)
#[cfg(feature = "mcp")]
pub mod dilithium_mcp;

/// Cybernetic Agency (Free Energy Principle, IIT Phi, Active Inference)
#[cfg(feature = "agency")]
pub mod agency;

/// Bio-Digital Isomorphic Cognition (Hyperbolic Attention, Self-Referential Loops, Dream States, Bateson Learning)
#[cfg(feature = "cognition")]
pub mod cognition;

// ============================================================================
// Re-exports: Core
// ============================================================================

pub use optimizer::{Optimizer, OptimizationResult};
pub use swarm::{Swarm, SwarmBuilder, SwarmResult as SwarmOptResult};
pub use lattice::{Lattice, LatticeBuilder};
pub use builder::{HyperPhysics, OptimizeBuilder};
pub use types::*;

// ============================================================================
// Re-exports: Hyperbolic
// ============================================================================

#[cfg(feature = "hyperbolic")]
pub use hyperbolic::{
    // Constants
    HYPERBOLIC_DIM, LORENTZ_DIM, DEFAULT_CURVATURE,
    PHI as HYPERBOLIC_PHI, PHI_INV as HYPERBOLIC_PHI_INV,
    // Types
    LorentzPoint11, PoincareBallPoint, PentagonVertex, PentagonTopology,
    // Functions
    lorentz_inner, hyperbolic_distance, stable_acosh,
    mobius_add, poincare_distance,
    exp_map_origin, log_map_origin,
    hyperbolic_centroid, hyperbolic_attention, batch_distances,
    // Pentagon
    PENTAGON_PHASES,
};

// ============================================================================
// Re-exports: Consciousness
// ============================================================================

#[cfg(feature = "consciousness")]
pub use consciousness::{
    // Constants
    MIN_ENTROPY, CRITICAL_BRANCHING_RATIO, CRITICAL_HURST,
    AVALANCHE_EXPONENT_CRITICAL, CRITICALITY_TOLERANCE,
    // Information theory
    shannon_entropy, joint_entropy, conditional_entropy,
    mutual_information, transfer_entropy,
    // IIT
    Partition, PhiCalculator,
    // SOC
    Avalanche, CriticalityAnalysis,
    // Pentagon emergence
    PentagonEmergence, EmergenceLevel,
};

// ============================================================================
// Re-exports: Neural
// ============================================================================

#[cfg(feature = "neural")]
pub use neural::{
    // Constants
    FIBONACCI_TAU, STDP_A_PLUS, STDP_A_MINUS, STDP_TAU,
    DEFAULT_LEARNING_RATE, DEFAULT_LAMBDA, DEFAULT_WEIGHT_BOUNDS,
    // STDP functions
    stdp_weight_change, fibonacci_stdp_weight_change,
    effective_time_constant, compute_stdp_balance,
    // Types
    FibonacciSTDP, EligibilityTrace, EligibilityTraceManager,
    SpikingNeuron, SpikingLayer,
};

// ============================================================================
// Re-exports: pBit
// ============================================================================

#[cfg(feature = "pbit")]
pub use pbit::{
    // Constants
    ISING_CRITICAL_TEMP, PENTAGON_COUPLING, PENTAGON_ENGINES,
    PHI as PBIT_PHI, PHI_INV as PBIT_PHI_INV,
    // Functions
    pbit_probability, sigmoid, boltzmann_weight,
    ising_energy, metropolis_accept,
    // Types
    PBit, PentagonPBit, PBitLattice,
    LatticeConfig as PBitLatticeConfig,
};

// ============================================================================
// Re-exports: Dilithium MCP
// ============================================================================

#[cfg(feature = "mcp")]
pub use dilithium_mcp::{
    // Config
    McpConfig, DEFAULT_MCP_PATH, DEFAULT_RUNTIME,
    // Types
    DilithiumKeyPair, LorentzPoint as McpLorentzPoint,
    PBitState, StdpResult, SwarmState, McpToolResult,
    // Client
    DilithiumClient,
    // Local fallbacks
    local as mcp_local,
};

// ============================================================================
// Prelude Module
// ============================================================================

/// Prelude module - import everything you need
pub mod prelude {
    // Core
    pub use crate::optimizer::{Optimizer, OptimizationResult};
    pub use crate::swarm::{Swarm, SwarmBuilder};
    pub use crate::lattice::{Lattice, LatticeBuilder};
    pub use crate::builder::{HyperPhysics, OptimizeBuilder};
    pub use crate::types::*;

    #[cfg(feature = "serde")]
    pub use crate::persistence::*;

    // Hyperbolic
    #[cfg(feature = "hyperbolic")]
    pub use crate::hyperbolic::{
        LorentzPoint11, PoincareBallPoint, PentagonTopology,
        lorentz_inner, hyperbolic_distance, mobius_add,
        HYPERBOLIC_DIM, LORENTZ_DIM,
    };

    // Consciousness
    #[cfg(feature = "consciousness")]
    pub use crate::consciousness::{
        PhiCalculator, CriticalityAnalysis, PentagonEmergence, EmergenceLevel,
        shannon_entropy, mutual_information,
    };

    // Neural
    #[cfg(feature = "neural")]
    pub use crate::neural::{
        FibonacciSTDP, EligibilityTrace, SpikingNeuron, SpikingLayer,
        fibonacci_stdp_weight_change, FIBONACCI_TAU,
    };

    // pBit
    #[cfg(feature = "pbit")]
    pub use crate::pbit::{
        PBit, PentagonPBit, PBitLattice,
        LatticeConfig as PBitLatticeConfig,
        pbit_probability, boltzmann_weight, ISING_CRITICAL_TEMP,
    };

    // MCP
    #[cfg(feature = "mcp")]
    pub use crate::dilithium_mcp::{
        DilithiumClient, McpConfig, DilithiumKeyPair,
    };

    // Agency
    #[cfg(feature = "agency")]
    pub use crate::agency::{
        CyberneticAgent, AgentConfig, AgentState, AgencyObservation, AgencyAction,
        FreeEnergyEngine, SurvivalDrive, HomeostaticController,
        ActiveInferenceEngine, PolicySelector, PhiCalculatorTrait,
        // Negentropy framework (Pedagogic Scaffolding)
        NegentropyEngine, NegentropyConfig, BatesonLevel, ScaffoldMode,
        CognitiveRegulator, PedagogicScaffold,
        // Brain-inspired modules
        PrefrontalCortex, AnteriorCingulate, Insula, BasalGanglia, Hippocampus, Episode,
    };

    // Cognition
    #[cfg(feature = "cognition")]
    pub use crate::cognition::{
        // Core system
        CognitionSystem, CognitionConfig,
        // Attention
        AttentionState, HyperbolicAttention, CurvatureModulator, LocusCoeruleusGain,
        // Loop coordinator
        SelfReferentialLoop, LoopState, LoopConfig, LoopMessage,
        PerceptionInput, CognitionOutput, NeocortexState, AgencyIntent, ConsciousnessIntegration,
        // Dream state
        DreamState, DreamConfig, DreamConsolidator,
        ReplayBuffer, EpisodicMemory, ConsolidationMetrics,
        // Learning
        LearningLevel, BatesonLearner, LearningContext,
        ProtoLearning, Learning, DeuteroLearning, LearningIII,
        // Integration
        CorticalBusIntegration, MessageRouter, RouteConfig,
        // Types
        CognitionPhase, ArousalLevel, CognitiveLoad, AttentionBandwidth,
    };
}

// ============================================================================
// Error Types
// ============================================================================

/// Error types for HyperPhysics operations
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

    /// Hyperbolic geometry error
    #[error("Hyperbolic error: {0}")]
    Hyperbolic(String),

    /// Neural network error
    #[error("Neural error: {0}")]
    Neural(String),

    /// MCP communication error
    #[error("MCP error: {0}")]
    Mcp(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for HyperPhysics operations
pub type Result<T> = std::result::Result<T, HyperPhysicsError>;

// ============================================================================
// Version Info
// ============================================================================

/// Plugin version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get version information
pub fn version() -> &'static str {
    VERSION
}

/// Get feature summary
pub fn features() -> Vec<&'static str> {
    let mut features = vec!["core", "optimization", "swarm", "lattice"];

    #[cfg(feature = "parallel")]
    features.push("parallel");

    #[cfg(feature = "serde")]
    features.push("serde");

    #[cfg(feature = "async")]
    features.push("async");

    #[cfg(feature = "hyperbolic")]
    features.push("hyperbolic");

    #[cfg(feature = "consciousness")]
    features.push("consciousness");

    #[cfg(feature = "neural")]
    features.push("neural");

    #[cfg(feature = "pbit")]
    features.push("pbit");

    #[cfg(feature = "mcp")]
    features.push("mcp");

    features
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_features() {
        let f = features();
        assert!(f.contains(&"core"));
    }

    #[test]
    #[cfg(feature = "hyperbolic")]
    fn test_hyperbolic_import() {
        let _point = LorentzPoint11::origin();
    }

    #[test]
    #[cfg(feature = "consciousness")]
    fn test_consciousness_import() {
        let _entropy = shannon_entropy(&[0.5, 0.5]);
    }

    #[test]
    #[cfg(feature = "neural")]
    fn test_neural_import() {
        let _stdp = FibonacciSTDP::new();
    }

    #[test]
    #[cfg(feature = "pbit")]
    fn test_pbit_import() {
        let _pentagon = PentagonPBit::at_criticality();
    }
}
