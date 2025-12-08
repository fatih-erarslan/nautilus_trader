//! Core types for HyperPhysics plugin

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Optimization strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Strategy {
    /// Particle Swarm Optimization (birds flocking)
    #[default]
    ParticleSwarm,
    /// Grey Wolf Optimization (wolf pack hunting)
    GreyWolf,
    /// Whale Optimization Algorithm (bubble-net feeding)
    Whale,
    /// Firefly Algorithm (bioluminescence)
    Firefly,
    /// Bat Algorithm (echolocation)
    Bat,
    /// Cuckoo Search (Lévy flights)
    Cuckoo,
    /// Differential Evolution
    DifferentialEvolution,
    /// Genetic Algorithm
    Genetic,
    /// Ant Colony Optimization
    AntColony,
    /// Artificial Bee Colony
    BeeColony,
    /// Bacterial Foraging
    Bacterial,
    /// Social Spider Optimization
    SocialSpider,
    /// Salp Swarm Algorithm
    SalpSwarm,
    /// Moth-Flame Optimization
    MothFlame,
    /// Quantum-enhanced PSO
    QuantumPSO,
    /// Adaptive hybrid (auto-selects best)
    Adaptive,
}

impl Strategy {
    /// Get biological inspiration description
    pub fn description(&self) -> &'static str {
        match self {
            Strategy::ParticleSwarm => "Birds flock by following local and global best positions",
            Strategy::GreyWolf => "Wolf packs hunt with alpha, beta, delta hierarchy",
            Strategy::Whale => "Whales encircle prey with bubble-net spiral",
            Strategy::Firefly => "Fireflies attract mates with brighter flashes",
            Strategy::Bat => "Bats use echolocation frequency and loudness",
            Strategy::Cuckoo => "Cuckoos use Lévy flights and brood parasitism",
            Strategy::DifferentialEvolution => "Mutation and crossover evolve population",
            Strategy::Genetic => "Natural selection favors fittest individuals",
            Strategy::AntColony => "Ants deposit pheromones to mark successful paths",
            Strategy::BeeColony => "Bees share food source info via waggle dance",
            Strategy::Bacterial => "Bacteria swim via tumbling and running",
            Strategy::SocialSpider => "Spiders communicate via web vibrations",
            Strategy::SalpSwarm => "Salps form chains led by leader",
            Strategy::MothFlame => "Moths navigate toward distant light sources",
            Strategy::QuantumPSO => "Quantum superposition enables parallel exploration",
            Strategy::Adaptive => "Dynamically selects best strategy for the problem",
        }
    }
    
    /// Get exploration/exploitation balance (0 = exploitation, 1 = exploration)
    pub fn exploration_ratio(&self) -> f64 {
        match self {
            Strategy::ParticleSwarm => 0.5,
            Strategy::GreyWolf => 0.3,
            Strategy::Whale => 0.7,
            Strategy::Firefly => 0.5,
            Strategy::Bat => 0.6,
            Strategy::Cuckoo => 0.8,
            Strategy::DifferentialEvolution => 0.5,
            Strategy::Genetic => 0.6,
            Strategy::AntColony => 0.6,
            Strategy::BeeColony => 0.5,
            Strategy::Bacterial => 0.7,
            Strategy::SocialSpider => 0.5,
            Strategy::SalpSwarm => 0.4,
            Strategy::MothFlame => 0.4,
            Strategy::QuantumPSO => 0.8,
            Strategy::Adaptive => 0.5,
        }
    }
}

/// Network topology for swarm organization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Topology {
    /// Star - central hub connected to all agents
    Star,
    /// Ring - circular connection
    Ring,
    /// Mesh - fully connected
    #[default]
    Mesh,
    /// Hierarchical - tree structure
    Hierarchical,
    /// Hyperbolic - Poincaré disk embedding
    Hyperbolic,
    /// Small World - Watts-Strogatz model
    SmallWorld,
    /// Scale Free - Barabási-Albert model
    ScaleFree,
    /// Lattice - grid-based
    Lattice,
    /// Dynamic - evolves over time
    Dynamic,
}

impl Topology {
    /// Get topology description
    pub fn description(&self) -> &'static str {
        match self {
            Topology::Star => "Hub-and-spoke: fast broadcast, single point of failure",
            Topology::Ring => "Circular: ordered updates, medium fault tolerance",
            Topology::Mesh => "Fully connected: maximum communication, O(n²) edges",
            Topology::Hierarchical => "Tree: scalable, top-down information flow",
            Topology::Hyperbolic => "Poincaré disk: hierarchical with local clustering",
            Topology::SmallWorld => "Watts-Strogatz: short paths, high clustering",
            Topology::ScaleFree => "Barabási-Albert: power law degree distribution",
            Topology::Lattice => "Grid: regular structure, local communication",
            Topology::Dynamic => "Evolving: adapts to swarm performance",
        }
    }
}

/// Problem type hint for strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProblemType {
    /// Single global optimum
    Unimodal,
    /// Few local optima
    FewModes,
    /// Many local optima
    #[default]
    Multimodal,
    /// Very many local optima
    HighlyMultimodal,
    /// Noisy objective function
    Noisy,
    /// Expensive to evaluate
    Expensive,
    /// Real-time constraint
    RealTime,
}

impl ProblemType {
    /// Get recommended strategy for this problem type
    pub fn recommended_strategy(&self) -> Strategy {
        match self {
            ProblemType::Unimodal => Strategy::GreyWolf,
            ProblemType::FewModes => Strategy::ParticleSwarm,
            ProblemType::Multimodal => Strategy::Whale,
            ProblemType::HighlyMultimodal => Strategy::Cuckoo,
            ProblemType::Noisy => Strategy::DifferentialEvolution,
            ProblemType::Expensive => Strategy::BeeColony,
            ProblemType::RealTime => Strategy::GreyWolf,
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationConfig {
    /// Number of dimensions
    pub dimensions: usize,
    /// Search bounds (min, max) per dimension
    pub bounds: Vec<(f64, f64)>,
    /// Strategy to use
    pub strategy: Strategy,
    /// Topology for multi-agent
    pub topology: Topology,
    /// Population size
    pub population_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to maximize (true) or minimize (false)
    pub maximize: bool,
    /// Problem type hint
    pub problem_type: ProblemType,
    /// Custom parameters
    pub params: HashMap<String, f64>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            dimensions: 10,
            bounds: vec![(-100.0, 100.0); 10],
            strategy: Strategy::default(),
            topology: Topology::default(),
            population_size: 50,
            max_iterations: 1000,
            tolerance: 1e-6,
            maximize: false,
            problem_type: ProblemType::default(),
            params: HashMap::new(),
        }
    }
}

/// Convergence metrics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceMetrics {
    /// Final fitness value
    pub final_fitness: f64,
    /// Iterations to converge
    pub iterations: usize,
    /// Function evaluations
    pub evaluations: usize,
    /// Convergence rate (improvement per iteration)
    pub convergence_rate: f64,
    /// Final population diversity
    pub diversity: f64,
    /// Time in milliseconds
    pub time_ms: u64,
}

/// Lattice configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatticeConfig {
    /// Dimensions (x, y, z)
    pub dimensions: (usize, usize, usize),
    /// Temperature (Boltzmann)
    pub temperature: f64,
    /// Coupling strength
    pub coupling: f64,
    /// External field
    pub field: f64,
    /// STDP learning rate
    pub learning_rate: f64,
    /// Periodic boundaries
    pub periodic: bool,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            dimensions: (16, 16, 4),
            temperature: 1.0,
            coupling: 1.0,
            field: 0.0,
            learning_rate: 0.01,
            periodic: true,
        }
    }
}

/// Swarm configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmConfig {
    /// Number of agents
    pub agent_count: usize,
    /// Strategies to use
    pub strategies: Vec<Strategy>,
    /// Network topology
    pub topology: Topology,
    /// Problem dimensions
    pub dimensions: usize,
    /// Search bounds
    pub bounds: Vec<(f64, f64)>,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Use pBit lattice
    pub use_lattice: bool,
    /// Enable evolution
    pub enable_evolution: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            agent_count: 50,
            strategies: vec![Strategy::ParticleSwarm, Strategy::GreyWolf, Strategy::Whale],
            topology: Topology::Hyperbolic,
            dimensions: 10,
            bounds: vec![(-100.0, 100.0); 10],
            max_iterations: 1000,
            use_lattice: true,
            enable_evolution: false,
        }
    }
}
