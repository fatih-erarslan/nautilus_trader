// Dynamic Swarm Selection Engine - 14+ Bio-Inspired Optimization Algorithms
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, warn, error, debug};

pub mod selector;
pub mod algorithms;
pub mod performance_tracking;
pub mod adaptive_selection;

pub use selector::*;
pub use algorithms::*;
pub use performance_tracking::*;
pub use adaptive_selection::*;

use market_regime_detector::MarketRegime;

/// Swarm algorithm types with biological inspiration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwarmAlgorithm {
    // Classic swarm algorithms
    ParticleSwarm,          // PSO - Birds flocking behavior
    AntColony,              // ACO - Ant foraging behavior
    ArtificialBeeColony,    // ABC - Bee foraging behavior
    
    // Evolution-based algorithms
    GeneticAlgorithm,       // GA - Natural selection
    DifferentialEvolution,  // DE - Evolutionary mutation
    
    // Animal-inspired algorithms
    GreyWolf,               // GWO - Wolf pack hunting
    WhaleOptimization,      // WOA - Humpback whale hunting
    BatAlgorithm,           // BA - Bat echolocation
    FireflyAlgorithm,       // FA - Firefly bioluminescence
    CuckooSearch,           // CS - Cuckoo brood parasitism
    
    // Organism-inspired algorithms
    BacterialForaging,      // BFO - E. coli foraging
    SocialSpider,           // SSO - Spider web behavior
    MothFlame,              // MFO - Moth navigation
    SalpSwarm,              // SSA - Salp chain behavior
    
    // Hybrid and adaptive
    AdaptiveHybrid,         // Dynamic combination
    MultiObjective,         // Multi-objective optimization
    
    // Custom high-performance variants
    QuantumParticleSwarm,   // Quantum-enhanced PSO
    NeuralEvolution,        // Neural network evolution
    ChaosEnhanced,          // Chaos theory enhancement
}

impl SwarmAlgorithm {
    /// Get the biological inspiration behind the algorithm
    pub fn biological_inspiration(&self) -> &'static str {
        match self {
            SwarmAlgorithm::ParticleSwarm => "Bird flocking and fish schooling behavior",
            SwarmAlgorithm::AntColony => "Ant pheromone trail following and foraging",
            SwarmAlgorithm::ArtificialBeeColony => "Honeybee foraging and waggle dance communication",
            SwarmAlgorithm::GeneticAlgorithm => "Natural selection and genetic inheritance",
            SwarmAlgorithm::DifferentialEvolution => "Evolutionary mutation and crossover",
            SwarmAlgorithm::GreyWolf => "Grey wolf pack hunting hierarchy and coordination",
            SwarmAlgorithm::WhaleOptimization => "Humpback whale bubble-net feeding",
            SwarmAlgorithm::BatAlgorithm => "Bat echolocation and frequency tuning",
            SwarmAlgorithm::FireflyAlgorithm => "Firefly bioluminescent attraction",
            SwarmAlgorithm::CuckooSearch => "Cuckoo brood parasitism and Lévy flights",
            SwarmAlgorithm::BacterialForaging => "E. coli bacteria chemotaxis and reproduction",
            SwarmAlgorithm::SocialSpider => "Spider web vibration communication",
            SwarmAlgorithm::MothFlame => "Moth navigation using distant light sources",
            SwarmAlgorithm::SalpSwarm => "Salp chain formation and jet propulsion",
            SwarmAlgorithm::AdaptiveHybrid => "Dynamic ecosystem adaptation",
            SwarmAlgorithm::MultiObjective => "Multi-species ecosystem balance",
            SwarmAlgorithm::QuantumParticleSwarm => "Quantum superposition and entanglement",
            SwarmAlgorithm::NeuralEvolution => "Neural plasticity and synaptic evolution",
            SwarmAlgorithm::ChaosEnhanced => "Chaotic dynamics in natural systems",
        }
    }
    
    /// Get convergence characteristics
    pub fn convergence_profile(&self) -> ConvergenceProfile {
        match self {
            SwarmAlgorithm::ParticleSwarm => ConvergenceProfile::Fast,
            SwarmAlgorithm::AntColony => ConvergenceProfile::Steady,
            SwarmAlgorithm::GeneticAlgorithm => ConvergenceProfile::Balanced,
            SwarmAlgorithm::DifferentialEvolution => ConvergenceProfile::Robust,
            SwarmAlgorithm::GreyWolf => ConvergenceProfile::Aggressive,
            SwarmAlgorithm::WhaleOptimization => ConvergenceProfile::Explorative,
            SwarmAlgorithm::AdaptiveHybrid => ConvergenceProfile::Dynamic,
            _ => ConvergenceProfile::Moderate,
        }
    }
    
    /// Check regime compatibility
    pub fn is_regime_compatible(&self, regime: &MarketRegime) -> bool {
        match (self, regime) {
            // Neutral regime - all base algorithms are compatible
            (SwarmAlgorithm::ParticleSwarm, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::AntColony, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::GeneticAlgorithm, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::DifferentialEvolution, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::GreyWolf, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::WhaleOptimization, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::BatAlgorithm, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::FireflyAlgorithm, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::CuckooSearch, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::ArtificialBeeColony, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::BacterialForaging, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::SocialSpider, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::MothFlame, MarketRegime::Neutral) => true,
            (SwarmAlgorithm::SalpSwarm, MarketRegime::Neutral) => true,

            // High volatility regimes - use exploration-heavy algorithms
            (SwarmAlgorithm::WhaleOptimization, MarketRegime::HighVolatility) => true,
            (SwarmAlgorithm::CuckooSearch, MarketRegime::VolatilitySpike) => true,
            (SwarmAlgorithm::BatAlgorithm, MarketRegime::FlashCrash) => true,

            // Stable regimes - use exploitation-heavy algorithms
            (SwarmAlgorithm::ParticleSwarm, MarketRegime::LowVolatility) => true,
            (SwarmAlgorithm::AntColony, MarketRegime::Consolidation) => true,

            // Trending markets - use directional algorithms
            (SwarmAlgorithm::GreyWolf, MarketRegime::StrongUptrend) => true,
            (SwarmAlgorithm::GreyWolf, MarketRegime::StrongDowntrend) => true,

            // Quantum regimes - use quantum-enhanced algorithms
            (SwarmAlgorithm::QuantumParticleSwarm, MarketRegime::QuantumCoherent) => true,
            (SwarmAlgorithm::QuantumParticleSwarm, MarketRegime::QuantumEntangled) => true,

            // Adaptive hybrid is always compatible
            (SwarmAlgorithm::AdaptiveHybrid, _) => true,

            _ => false,
        }
    }
    
    /// Get computational complexity
    pub fn computational_complexity(&self) -> ComputationalComplexity {
        match self {
            SwarmAlgorithm::ParticleSwarm => ComputationalComplexity::Low,
            SwarmAlgorithm::AntColony => ComputationalComplexity::Medium,
            SwarmAlgorithm::GeneticAlgorithm => ComputationalComplexity::Medium,
            SwarmAlgorithm::QuantumParticleSwarm => ComputationalComplexity::High,
            SwarmAlgorithm::AdaptiveHybrid => ComputationalComplexity::VeryHigh,
            _ => ComputationalComplexity::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceProfile {
    Fast,           // Quick convergence, risk of local optima
    Steady,         // Consistent progress towards optimum
    Balanced,       // Good exploration-exploitation balance
    Robust,         // Resistant to noise and disturbances
    Aggressive,     // Fast convergence with high exploitation
    Explorative,    // Strong exploration, slower convergence
    Dynamic,        // Adaptive behavior based on problem
    Moderate,       // Standard convergence characteristics
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Low,            // O(n) or O(n log n)
    Medium,         // O(n²)
    High,           // O(n³) or quantum operations
    VeryHigh,       // Complex hybrid operations
}

/// Performance metrics for swarm algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceMetrics {
    pub algorithm: SwarmAlgorithm,
    pub regime: MarketRegime,
    pub optimization_score: f64,
    pub convergence_time: chrono::Duration,
    pub function_evaluations: u32,
    pub success_rate: f64,
    pub stability_score: f64,
    pub exploration_ratio: f64,
    pub exploitation_ratio: f64,
    pub diversity_index: f64,
}

/// Market conditions influencing swarm selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub volume: f64,
    pub trend_strength: f64,
    pub liquidity: f64,
    pub correlation: f64,
    pub noise_level: f64,
    pub regime_stability: f64,
}

/// Swarm selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionCriteria {
    pub optimization_target: OptimizationTarget,
    pub time_constraint: chrono::Duration,
    pub accuracy_requirement: f64,
    pub computational_budget: ComputationalBudget,
    pub risk_tolerance: f64,
    pub regime_specific: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationTarget {
    Profit,
    RiskAdjustedReturn,
    Sharpe,
    MaxDrawdown,
    WinRate,
    ProfitFactor,
    Latency,
    Accuracy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalBudget {
    pub max_function_evaluations: u32,
    pub max_computation_time: chrono::Duration,
    pub max_memory_usage: usize,
    pub parallelization_factor: u32,
}

/// Core trait for swarm optimization algorithms
#[async_trait]
pub trait SwarmOptimizer: Send + Sync {
    async fn optimize(&mut self, 
                     objective_function: &dyn ObjectiveFunction,
                     parameters: &OptimizationParameters) -> anyhow::Result<OptimizationResult>;
    
    async fn update_population(&mut self, performance_feedback: &PerformanceFeedback) -> anyhow::Result<()>;
    
    fn get_algorithm_type(&self) -> SwarmAlgorithm;
    fn get_current_best(&self) -> Option<Solution>;
    fn get_population_diversity(&self) -> f64;
    fn is_converged(&self) -> bool;
}

/// Objective function trait for optimization problems
#[async_trait]
pub trait ObjectiveFunction: Send + Sync {
    async fn evaluate(&self, solution: &Solution) -> anyhow::Result<f64>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
    fn is_maximization(&self) -> bool;
}

/// Solution representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub parameters: Vec<f64>,
    pub fitness: f64,
    pub evaluation_time: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, f64>,
}

/// Optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub bounds: Vec<(f64, f64)>,
    pub constraints: Vec<Constraint>,
    pub initialization_strategy: InitializationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    Linear { coefficients: Vec<f64>, bound: f64 },
    Nonlinear { expression: String },
    Box { min: f64, max: f64, variables: Vec<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitializationStrategy {
    Random,
    LatinHypercube,
    Sobol,
    Halton,
    SolutionSeeded { seed_count: usize },
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub best_solution: Solution,
    pub convergence_history: Vec<f64>,
    pub algorithm_used: SwarmAlgorithm,
    pub iterations_performed: u32,
    pub function_evaluations: u32,
    pub optimization_time: chrono::Duration,
    pub success: bool,
    pub termination_reason: TerminationReason,
    pub population_diversity_history: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    MaxIterationsReached,
    ToleranceAchieved,
    TimeoutReached,
    NoImprovement,
    ConvergenceDetected,
    UserTerminated,
    Error,
}

/// Performance feedback for algorithm adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFeedback {
    pub algorithm: SwarmAlgorithm,
    pub regime: MarketRegime,
    pub optimization_quality: f64,
    pub execution_time: chrono::Duration,
    pub resource_usage: ResourceUsage,
    pub success_indicator: bool,
    pub improvement_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_time: chrono::Duration,
    pub memory_peak: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub parallelization_efficiency: f64,
}

/// Error types for swarm selection
#[derive(thiserror::Error, Debug)]
pub enum SwarmSelectionError {
    #[error("No suitable algorithm found for regime: {0:?}")]
    NoSuitableAlgorithm(MarketRegime),
    
    #[error("Algorithm {0:?} not available")]
    AlgorithmNotAvailable(SwarmAlgorithm),
    
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    
    #[error("Insufficient computational budget: {0}")]
    InsufficientBudget(String),
    
    #[error("Performance tracking error: {0}")]
    PerformanceTrackingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

pub type SwarmSelectionResult<T> = Result<T, SwarmSelectionError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_algorithm_properties() {
        assert_eq!(SwarmAlgorithm::ParticleSwarm.convergence_profile(), ConvergenceProfile::Fast);
        assert_eq!(SwarmAlgorithm::WhaleOptimization.convergence_profile(), ConvergenceProfile::Explorative);
        assert!(SwarmAlgorithm::AdaptiveHybrid.is_regime_compatible(&MarketRegime::HighVolatility));
    }

    #[test]
    fn test_biological_inspiration() {
        let pso_inspiration = SwarmAlgorithm::ParticleSwarm.biological_inspiration();
        assert!(pso_inspiration.contains("Bird flocking"));
        
        let whale_inspiration = SwarmAlgorithm::WhaleOptimization.biological_inspiration();
        assert!(whale_inspiration.contains("whale"));
    }

    #[test]
    fn test_regime_compatibility() {
        assert!(SwarmAlgorithm::QuantumParticleSwarm.is_regime_compatible(&MarketRegime::QuantumCoherent));
        assert!(SwarmAlgorithm::ParticleSwarm.is_regime_compatible(&MarketRegime::LowVolatility));
        assert!(!SwarmAlgorithm::ParticleSwarm.is_regime_compatible(&MarketRegime::FlashCrash));
    }

    #[test]
    fn test_computational_complexity() {
        assert_eq!(SwarmAlgorithm::ParticleSwarm.computational_complexity(), ComputationalComplexity::Low);
        assert_eq!(SwarmAlgorithm::AdaptiveHybrid.computational_complexity(), ComputationalComplexity::VeryHigh);
    }

    #[test]
    fn test_solution_serialization() {
        let solution = Solution {
            parameters: vec![1.0, 2.0, 3.0],
            fitness: 0.95,
            evaluation_time: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        let serialized = serde_json::to_string(&solution).expect("Serialization failed");
        let deserialized: Solution = serde_json::from_str(&serialized).expect("Deserialization failed");
        
        assert_eq!(solution.parameters, deserialized.parameters);
        assert!((solution.fitness - deserialized.fitness).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_optimization_parameters() {
        let params = OptimizationParameters {
            population_size: 50,
            max_iterations: 1000,
            tolerance: 1e-6,
            bounds: vec![(0.0, 1.0), (-10.0, 10.0)],
            constraints: vec![],
            initialization_strategy: InitializationStrategy::LatinHypercube,
        };

        assert_eq!(params.population_size, 50);
        assert_eq!(params.bounds.len(), 2);
    }
}