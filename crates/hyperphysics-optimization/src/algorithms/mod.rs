//! Bio-inspired optimization algorithms.
//!
//! # Algorithm Tiers
//!
//! - **Tier 1**: PSO, GA, DE (most widely validated)
//! - **Tier 2**: GWO, WOA, ACO, BA, FA (proven effective)
//! - **Tier 3**: CS, ABC, BFO, SSO, MFO, SSA (specialized use cases)

// Tier 1 - Classic algorithms
mod pso;
mod genetic;
mod differential_evolution;

// Tier 2 - Animal-inspired
mod grey_wolf;
mod whale;
mod ant_colony;
mod bat;
mod firefly;

// Tier 3 - Specialized
mod cuckoo;
mod bee_colony;
mod bacterial;
mod social_spider;
mod moth_flame;
mod salp_swarm;
mod wasp_foraging;

// Tier 1 exports
pub use pso::*;
pub use genetic::*;
pub use differential_evolution::*;

// Tier 2 exports
pub use grey_wolf::*;
pub use whale::*;
pub use ant_colony::*;
pub use bat::*;
pub use firefly::*;

// Tier 3 exports
pub use cuckoo::*;
pub use bee_colony::*;
pub use bacterial::*;
pub use social_spider::*;
pub use moth_flame::*;
pub use salp_swarm::*;
pub use wasp_foraging::*;

use serde::{Deserialize, Serialize};

/// Algorithm types available in the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmType {
    // Tier 1 - Classic algorithms
    /// Particle Swarm Optimization
    ParticleSwarm,
    /// Genetic Algorithm
    GeneticAlgorithm,
    /// Differential Evolution
    DifferentialEvolution,

    // Tier 2 - Animal-inspired
    /// Grey Wolf Optimizer
    GreyWolf,
    /// Whale Optimization Algorithm
    WhaleOptimization,
    /// Ant Colony Optimization
    AntColony,
    /// Bat Algorithm
    BatAlgorithm,
    /// Firefly Algorithm
    FireflyAlgorithm,

    // Tier 3 - Specialized
    /// Cuckoo Search
    CuckooSearch,
    /// Artificial Bee Colony
    ArtificialBeeColony,
    /// Bacterial Foraging Optimization
    BacterialForaging,
    /// Social Spider Optimization
    SocialSpider,
    /// Moth-Flame Optimization
    MothFlame,
    /// Salp Swarm Algorithm
    SalpSwarm,
    /// Wasp Swarm Optimization
    WaspSwarm,
}

impl AlgorithmType {
    /// Get the biological inspiration behind this algorithm.
    #[must_use]
    pub fn biological_inspiration(&self) -> &'static str {
        match self {
            Self::ParticleSwarm => "Bird flocking and fish schooling behavior",
            Self::GeneticAlgorithm => "Natural selection and genetic inheritance",
            Self::DifferentialEvolution => "Evolutionary mutation and recombination",
            Self::GreyWolf => "Grey wolf pack hunting hierarchy",
            Self::WhaleOptimization => "Humpback whale bubble-net feeding",
            Self::AntColony => "Ant pheromone trail communication",
            Self::BatAlgorithm => "Bat echolocation and frequency tuning",
            Self::FireflyAlgorithm => "Firefly bioluminescent attraction",
            Self::CuckooSearch => "Cuckoo brood parasitism with Lévy flights",
            Self::ArtificialBeeColony => "Honeybee foraging and waggle dance",
            Self::BacterialForaging => "E. coli chemotaxis and reproduction",
            Self::SocialSpider => "Spider web vibration communication",
            Self::MothFlame => "Moth navigation towards light sources",
            Self::SalpSwarm => "Salp chain formation and jet propulsion",
            Self::WaspSwarm => "Wasp colony dominance hierarchy and foraging",
        }
    }

    /// Get convergence characteristics.
    #[must_use]
    pub fn convergence_profile(&self) -> ConvergenceProfile {
        match self {
            Self::ParticleSwarm => ConvergenceProfile::Fast,
            Self::GeneticAlgorithm => ConvergenceProfile::Balanced,
            Self::DifferentialEvolution => ConvergenceProfile::Robust,
            Self::GreyWolf => ConvergenceProfile::Aggressive,
            Self::WhaleOptimization => ConvergenceProfile::Explorative,
            Self::AntColony => ConvergenceProfile::Steady,
            Self::BatAlgorithm => ConvergenceProfile::Adaptive,
            Self::FireflyAlgorithm => ConvergenceProfile::Multimodal,
            Self::CuckooSearch => ConvergenceProfile::Explorative,
            Self::ArtificialBeeColony => ConvergenceProfile::Balanced,
            Self::BacterialForaging => ConvergenceProfile::Slow,
            Self::SocialSpider => ConvergenceProfile::Steady,
            Self::MothFlame => ConvergenceProfile::Fast,
            Self::SalpSwarm => ConvergenceProfile::Balanced,
            Self::WaspSwarm => ConvergenceProfile::Adaptive,
        }
    }

    /// Get computational complexity class.
    #[must_use]
    pub fn computational_complexity(&self) -> ComputationalComplexity {
        match self {
            Self::ParticleSwarm => ComputationalComplexity::Linear,
            Self::GeneticAlgorithm => ComputationalComplexity::LinearLog,
            Self::DifferentialEvolution => ComputationalComplexity::Linear,
            Self::GreyWolf => ComputationalComplexity::Linear,
            Self::WhaleOptimization => ComputationalComplexity::Linear,
            Self::AntColony => ComputationalComplexity::Quadratic,
            Self::BatAlgorithm => ComputationalComplexity::Linear,
            Self::FireflyAlgorithm => ComputationalComplexity::Quadratic,
            Self::CuckooSearch => ComputationalComplexity::Linear,
            Self::ArtificialBeeColony => ComputationalComplexity::Linear,
            Self::BacterialForaging => ComputationalComplexity::Linear,
            Self::SocialSpider => ComputationalComplexity::Quadratic,
            Self::MothFlame => ComputationalComplexity::LinearLog,
            Self::SalpSwarm => ComputationalComplexity::Linear,
            Self::WaspSwarm => ComputationalComplexity::Linear,
        }
    }

    /// Check if algorithm is suitable for multimodal problems.
    #[must_use]
    pub fn is_multimodal_suitable(&self) -> bool {
        matches!(
            self,
            Self::GeneticAlgorithm
                | Self::DifferentialEvolution
                | Self::FireflyAlgorithm
                | Self::CuckooSearch
                | Self::ArtificialBeeColony
        )
    }

    /// Check if algorithm supports constraints natively.
    #[must_use]
    pub fn supports_constraints(&self) -> bool {
        matches!(
            self,
            Self::GeneticAlgorithm | Self::DifferentialEvolution | Self::AntColony
        )
    }
}

/// Convergence behavior profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceProfile {
    /// Fast convergence, risk of premature convergence
    Fast,
    /// Good exploration-exploitation balance
    Balanced,
    /// Resistant to local optima
    Robust,
    /// High exploitation, fast local search
    Aggressive,
    /// High exploration, slower convergence
    Explorative,
    /// Steady progress towards optimum
    Steady,
    /// Adapts behavior during search
    Adaptive,
    /// Specialized for multimodal landscapes
    Multimodal,
    /// Slow but thorough search
    Slow,
}

/// Computational complexity per iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    /// O(n) - linear in population size
    Linear,
    /// O(n log n) - linearithmic
    LinearLog,
    /// O(n²) - quadratic (pairwise comparisons)
    Quadratic,
    /// O(n³) - cubic (rare, matrix operations)
    Cubic,
}

/// Base trait for all optimization algorithms.
pub trait Algorithm: Send + Sync {
    /// Get algorithm type.
    fn algorithm_type(&self) -> AlgorithmType;

    /// Get algorithm name.
    fn name(&self) -> &str;

    /// Check if algorithm has converged.
    fn is_converged(&self) -> bool;

    /// Get current best fitness.
    fn best_fitness(&self) -> Option<f64>;

    /// Get current iteration.
    fn iteration(&self) -> u32;
}

/// Algorithm configuration trait.
pub trait AlgorithmConfig: Clone + Default + Send + Sync {
    /// Validate configuration parameters.
    fn validate(&self) -> Result<(), String>;

    /// Create HFT-optimized configuration.
    fn hft_optimized() -> Self;

    /// Create high-accuracy configuration.
    fn high_accuracy() -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_type_properties() {
        assert_eq!(
            AlgorithmType::ParticleSwarm.convergence_profile(),
            ConvergenceProfile::Fast
        );
        assert_eq!(
            AlgorithmType::AntColony.computational_complexity(),
            ComputationalComplexity::Quadratic
        );
    }

    #[test]
    fn test_multimodal_suitability() {
        assert!(AlgorithmType::FireflyAlgorithm.is_multimodal_suitable());
        assert!(!AlgorithmType::GreyWolf.is_multimodal_suitable());
    }

    #[test]
    fn test_biological_inspiration() {
        let inspiration = AlgorithmType::WhaleOptimization.biological_inspiration();
        assert!(inspiration.contains("whale"));
    }
}
