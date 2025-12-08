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

// Phase 2: SIMD batch operations and lock-free state
pub mod simd_batch;

// Phase 4: Free Energy Principle and Active Inference
pub mod free_energy;

// Phase 4: Global Workspace Theory (Baars)
pub mod global_workspace;

// Phase 4: Bateson's Ecological Epistemology
pub mod bateson_ecology;

// Phase 4: Integrated Information Theory 4.0
pub mod iit_phi;

// Phase 4: Hyperbolic Replicator Dynamics
pub mod replicator_dynamics;

// Phase 4: Predictive Coding + Enactive Unification
pub mod predictive_coding;

// Phase 5: Language Creation Framework (Christiansen & Chater)
pub mod language_creation;

// Phase 5: Interoceptive Inference (Seth & Friston)
pub mod interoception;

/// Lorentz/Hyperboloid model bridge (requires `lorentz` feature)
#[cfg(feature = "lorentz")]
pub mod lorentz_bridge;

pub use poincare::PoincarePoint;
pub use geodesic::Geodesic;
pub use distance::HyperbolicDistance;
pub use tessellation::{
    HyperbolicTessellation,
    // Hyperbolic Voronoi and Delaunay (Nielsen & Nock 2010)
    HyperbolicDelaunay, DelaunayTriangle,
    HyperbolicVoronoi, VoronoiCell,
};
pub use tessellation_73::{HeptagonalTessellation, HeptagonalTile, TessellationVertex, FuchsianGroup, TileId, VertexId};
pub use crypto_substrate::{CryptoSubstrate, TileCryptoState, SubstrateStats};
pub use curvature::{CurvatureTensor, RiemannSymmetryCheck, CURVATURE_K, DIMENSION};
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
    // Spectral analysis with convergence bounds (Chung 1997, Spielman 2012)
    ConvergenceBound, SpectralConfig, WeylLawResult,
};

// Re-export Chunk processor types
pub use chunk_processor::{
    SpikeEvent, SpikePacket, TemporalChunk, ChunkProcessor, ChunkProcessorConfig,
    NowOrNeverBottleneck, CompressedUnit, ProcessorStats,
    ChunkRepresentation, ChunkPredictionError, ChunkPrediction,
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

// Re-export STDP Learning types (Phase 2 + Phase 4 Shapley)
pub use stdp_learning::{
    STDPConfig, STDPLearner, STDPStats, EligibilityTrace,
    LearningRateSchedule, LearningRateScheduler,
    HomeostaticPlasticity, NormalizationType, WeightNormalizer,
    ChunkAwareSTDP,
    // Phase 4: Shapley value credit assignment
    ShapleyCreditor, ShapleySTDP, ShapleyStats,
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
    // Phase 4 Integration bridges
    WorkspacePredictiveBridge, PhiWorkspaceBridge, EcologyLearningBridge,
    // Unified Conscious Integration Hub
    ConsciousIntegrationHub, ConsciousHubConfig, ConsciousHubStats,
};

// Re-export SIMD batch and lock-free types (Phase 2)
pub use simd_batch::{
    SimdPoincareBatch, SimdF32Batch,
    AtomicMembranePotential, AtomicWeight, AtomicSpikeCounter,
    LockFreeNeuronState, fast_acosh, fast_exp,
    CACHE_LINE, SIMD_F32_WIDTH, SIMD_F64_WIDTH,
};

// Re-export Free Energy types (Phase 4)
pub use free_energy::{
    FreeEnergyCalculator, FreeEnergyResult, ExpectedFreeEnergy,
    PrecisionWeightedError, HierarchicalErrorAggregator, Precision,
    // ELBO Variational Inference (Blei et al. 2017)
    VariationalELBO, ELBOResult, ELBOFitResult, ELBOConfig,
    ELBOConvergenceMonitor, VariationalMethod, ConvergenceReason,
};

// Re-export Global Workspace types (Phase 4)
pub use global_workspace::{
    GlobalWorkspaceConfig, GlobalWorkspace, SpecialistModule, SpecialistType,
    WorkspaceContent, Coalition, BroadcastEvent, WorkspaceStats,
};

// Re-export Bateson Ecology types (Phase 4)
pub use bateson_ecology::{
    EcologyConfig, EcologicalMind, LearningLevel, Difference, Context,
    DoubleBind, DeuteroStats, EcologyStats, LearningResult,
};

// Re-export IIT Phi types (Phase 4)
pub use iit_phi::{
    PhiConfig, PhiCalculator, PhiResult, MechanismState, CauseEffectRepertoire,
    ProbabilityDistribution, Partition, PartitionType, IntrinsicCausalPower,
    Complex, PhiStats,
};

// Re-export Replicator Dynamics types (Phase 4)
pub use replicator_dynamics::{
    ReplicatorConfig, HyperbolicReplicator, Strategy, PayoffMatrix,
    ReplicatorStats, StepResult as ReplicatorStepResult,
};

// Re-export Predictive Coding types (Phase 4)
pub use predictive_coding::{
    PredictiveCodingConfig, PredictiveEnactiveSystem, HierarchicalBelief,
    LevelPrediction, LevelError, PredictiveStats, ProcessResult,
};

// Re-export Language Creation types (Phase 5)
pub use language_creation::{
    LanguageCreationConfig, LanguageCreationSystem, Construction,
    ConstraintConfig, LanguageCreationStats,
    EvolutionAcquisitionBridge, AcquisitionProcessingBridge, EvolutionProcessingBridge,
};

// Re-export Interoception types (Phase 5)
pub use interoception::{
    InteroceptionConfig, InteroceptiveInference, InteroceptiveState,
    CardiacState, RespiratoryState, MetabolicState,
    AllostaticRegulator, InteroceptiveStats,
    // Russell Circumplex Model of Affect
    RussellCircumplex, CircumplexPoint, CircumplexEmotion,
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
pub use fuchsian::{
    FuchsianGroup as FuchsianGroupAlgebraic,
    // Exact orbit computation (Beardon 1983, Katok 1992)
    ExactOrbitPoint, OrbitStatistics, OrbitConfig, ExactOrbitEnumerator,
    // Dirichlet domain (Ratcliffe 2006)
    DirichletDomain, DirichletEdge, SidePairing,
    // Ford polygon (Ford 1929)
    FordPolygon,
    // Hyperbolic geometry utilities
    hyperbolic_distance as fuchsian_hyperbolic_distance,
    hyperbolic_midpoint, perpendicular_bisector,
};

// Re-export Lorentz bridge types when feature is enabled
#[cfg(feature = "lorentz")]
pub use lorentz_bridge::{ToLorentz, ToPoincare, simd_hyperbolic_distance, batch_to_lorentz, batch_to_poincare, pairwise_distances_simd, lorentz_model};

// Re-export core Lorentz types when feature is enabled
#[cfg(feature = "lorentz")]
pub use hyperphysics_lorentz::{LorentzPoint, LorentzModel, SimdMinkowski};
