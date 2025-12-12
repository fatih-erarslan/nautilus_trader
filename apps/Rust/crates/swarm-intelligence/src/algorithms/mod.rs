//! Swarm intelligence algorithm implementations

pub mod pso;
pub mod aco;
pub mod de;
pub mod abc;
pub mod cs; // Cuckoo Search
pub mod bfo; // Bacterial Foraging Optimization  
pub mod fa; // Firefly Algorithm
pub mod gwo; // Grey Wolf Optimizer
pub mod salp_swarm; // Salp Swarm Algorithm - COMPLETED
pub mod ba; // Bat Algorithm - COMPLETED
pub mod woa; // Whale Optimization Algorithm - COMPLETED
pub mod mfo; // Moth-Flame Optimization - COMPLETED
pub mod sca; // Sine Cosine Algorithm - COMPLETED

// Re-exports for convenience
pub use pso::{ParticleSwarmOptimization, PsoParameters, PsoVariant};
pub use aco::{AntColonyOptimization, AcoParameters};
pub use de::{DifferentialEvolution, DeParameters, DeStrategy};
pub use abc::{ArtificialBeeColony, AbcParameters};
pub use cs::{CuckooSearch, CsParameters};
pub use bfo::{BacterialForagingOptimization, BfoParameters};
pub use fa::{FireflyAlgorithm, FaParameters};
pub use gwo::{GreyWolfOptimizer, GwoParameters, GwoVariant, WolfRole, HuntingStrategy, PackCommunication, GreyWolf};
pub use salp_swarm::{
    SalpSwarmAlgorithm, SsaParameters, SsaVariant, ChainTopology, 
    OceanCurrentPattern, MarineEnvironment, Salp, SalpChain
};

// NEW ALGORITHMS - COMPREHENSIVE IMPLEMENTATIONS
pub use ba::{
    BatAlgorithm, BaParameters, BaVariant, Bat, EcholocationStrategy, 
    HuntingStrategy as BatHuntingStrategy, RoostFormation
};
pub use woa::{
    WhaleOptimizationAlgorithm, WoaParameters, WoaVariant, Whale, 
    HuntingStrategy as WhaleHuntingStrategy, PodFormation, BubbleNetPattern, HuntingRole
};
pub use mfo::{
    MothFlameOptimization, MfoParameters, MfoVariant, Moth, Flame, 
    NavigationStrategy, FlameUpdateStrategy, SpiralPattern
};
pub use sca::{
    SineCosineAlgorithm, ScaParameters, ScaVariant, ScaAgent, 
    OscillationStrategy, WavePattern, ParameterUpdateStrategy
};