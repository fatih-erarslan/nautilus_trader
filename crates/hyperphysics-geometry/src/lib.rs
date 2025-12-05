//! # Hyperbolic Geometry Engine
//!
//! Implementation of hyperbolic 3-space (H³) with constant negative curvature K=-1.
//! Uses the Poincaré disk model for computational efficiency.
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Cannon et al. (1997) "Hyperbolic Geometry" Springer GTM 31
//! - Lee (2018) "Introduction to Riemannian Manifolds"
//! - Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50
//!
//! ## Poincaré Disk Model
//!
//! Points: D³ = {x ∈ ℝ³ : ||x|| < 1}
//! Metric: ds² = 4(dx₁² + dx₂² + dx₃²) / (1 - ||x||²)²
//! Distance: d_H(p,q) = acosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))

pub mod poincare;
pub mod geodesic;
pub mod distance;
pub mod tessellation;
pub mod tessellation_73;
pub mod crypto_substrate;
pub mod curvature;
pub mod adversarial_lattice;
pub mod sentry_integration;

// Hyperbolic Spiking Neural Network modules
pub mod hyperbolic_snn;
pub mod markov_kernels;
pub mod chunk_processor;
pub mod enactive_layer;
pub mod snn_gnn_simd;
pub mod snn_gpu;

// Phase 2 & 3: Language Acquisition and Evolution
pub mod stdp_learning;
pub mod topology_evolution;

// Graph Attention with Hyperbolic Modulation
pub mod graph_attention;

// Integrated Cognitive System - connects all modules
pub mod integrated_system;

/// Lorentz/Hyperboloid model bridge (requires `lorentz` feature)
#[cfg(feature = "lorentz")]
pub mod lorentz_bridge;

pub use poincare::PoincarePoint;
pub use geodesic::Geodesic;
pub use distance::HyperbolicDistance;
pub use tessellation::HyperbolicTessellation;
pub use tessellation_73::{HeptagonalTessellation, HeptagonalTile, TessellationVertex, FuchsianGroup, TileId, VertexId};
pub use crypto_substrate::{CryptoSubstrate, TileCryptoState, SubstrateStats};
pub use curvature::CurvatureTensor;
pub use adversarial_lattice::{
    SchlafliSymbol, DefenseTopology, HyperboloidPoint, SentryNode,
    AdversarialLattice, DetectionResult, LatticeStats, DetectionEvent,
};
pub use sentry_integration::{
    LorentzVec, LorentzBatch, BeliefUpdateMode, DualModeBeliefUpdater,
    ThermodynamicLearner, ThermodynamicStats, HyperbolicLattice,
};

// Re-export Hyperbolic SNN types
pub use hyperbolic_snn::{
    LorentzVec as LorentzVec4D, SpikingNeuron, Synapse, SpikeEvent as SNNSpikeEvent,
    HyperbolicSTDP, SOCMonitor, SOCStats, SNNConfig, HyperbolicSNN, SimulationResult, NetworkStats,
};

// Re-export Markov kernel types
pub use markov_kernels::{
    HyperbolicHeatKernel, TransitionOperator, ChapmanKolmogorov,
    HyperbolicRandomWalk, HittingTime, SpectralAnalysis,
};

// Re-export Chunk processor types
pub use chunk_processor::{
    SpikeEvent, SpikePacket, TemporalChunk, ChunkProcessor, ChunkProcessorConfig,
    NowOrNeverBottleneck, CompressedUnit, ProcessorStats,
};

// Re-export Enactive layer types
pub use enactive_layer::{
    Observation, Action, ActionType, BeliefState, Policy, PolicyType,
    EnactiveLayer, EnactiveConfig, EnactiveStats, EnactiveSensorimotorAgent,
    Modality, SensorimotorCoupling,
};

// Re-export SNN-GNN SIMD types
pub use snn_gnn_simd::{
    SpikingGraphNode, SpikeEvent as GraphSpikeEvent, SimdMembraneBatch,
    SimdDistanceCalculator, EventDrivenGraphProcessor, ProcessorConfig as GraphProcessorConfig,
    ProcessorStats as GraphProcessorStats, SpikeCode, EnergyMetrics, AvalancheInfo, NetworkState,
    SOCAnalyzer, SOCStats as GnnSOCStats,
};

// Re-export STDP Learning types (Phase 2)
pub use stdp_learning::{
    STDPConfig, STDPLearner, STDPStats, EligibilityTrace,
    LearningRateSchedule, LearningRateScheduler,
    HomeostaticPlasticity, NormalizationType, WeightNormalizer,
    ChunkAwareSTDP,
};

// Re-export Topology Evolution types (Phase 3)
pub use topology_evolution::{
    TopologyConfig, TopologyEvolver, TopologyStats, TopologyUpdate,
    ConnectionState, NeuronTopologyState,
    PruningStrategy, CreationStrategy,
    TessellationRefinement,
};

// Re-export Graph Attention types
pub use graph_attention::{
    AttentionConfig, HyperbolicAttentionHead, MultiHeadGraphAttention,
    AttentionStats, SpikeAwareGraphAttention,
    HyperbolicMessagePassing, MessageAggregation,
};

// Re-export GPU types
pub use snn_gpu::{
    GpuContext, GpuConfig, GpuBackend, GpuNeuronState, GpuPosition, GpuSynapse,
    GpuSimParams, GpuError, HybridProcessor, ProcessingMode, HybridMetrics,
};

// Re-export Integrated System types
pub use integrated_system::{
    SOCCoordinator, SOCModulation, HyperbolicEnvironment,
    ActiveInferenceLayer, IntegratedCognitiveSystem,
    IntegratedSystemConfig, IntegratedStats,
};

use thiserror::Error;

/// Errors specific to hyperbolic geometry operations
#[derive(Error, Debug)]
pub enum GeometryError {
    #[error("Point outside Poincaré disk: norm = {norm}")]
    OutsideDisk { norm: f64 },

    #[error("Numerical instability in distance calculation")]
    NumericalInstability,

    #[error("Invalid tessellation parameters: {message}")]
    InvalidTessellation { message: String },

    #[error("Geodesic integration failed: {reason}")]
    GeodesicFailure { reason: String },
}

pub type Result<T> = std::result::Result<T, GeometryError>;

/// Constant negative curvature
pub const CURVATURE: f64 = -1.0;

/// Numerical tolerance for boundary checks
pub const EPSILON: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curvature_constant() {
        assert_eq!(CURVATURE, -1.0, "H³ must have K = -1");
    }
}

pub mod moebius;
pub mod fuchsian;
pub use moebius::{MoebiusTransform, TransformType};
pub use fuchsian::FuchsianGroup as FuchsianGroupAlgebraic;

// Re-export Lorentz bridge types when feature is enabled
#[cfg(feature = "lorentz")]
pub use lorentz_bridge::{ToLorentz, ToPoincare, simd_hyperbolic_distance, batch_to_lorentz, batch_to_poincare, pairwise_distances_simd, lorentz_model};

// Re-export core Lorentz types when feature is enabled
#[cfg(feature = "lorentz")]
pub use hyperphysics_lorentz::{LorentzPoint, LorentzModel, SimdMinkowski};
