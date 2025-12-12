//! # Swarm Intelligence Framework
//!
//! A comprehensive swarm intelligence library implementing 14+ optimization algorithms
//! with advanced Combinatorial Diversity Fusion Analysis (CDFA) integration.
//!
//! ## Features
//!
//! ### Core Algorithms
//! - **PSO**: Particle Swarm Optimization with multiple variants
//! - **ACO**: Ant Colony Optimization for discrete problems
//! - **DE**: Differential Evolution with adaptive parameters
//! - **ABC**: Artificial Bee Colony for global optimization
//! - **CS**: Cuckoo Search with Lévy flights
//! - **BFO**: Bacterial Foraging Optimization
//! - **FA**: Firefly Algorithm with dynamic light intensity
//! - **GWO**: Grey Wolf Optimizer with pack hierarchy
//! - **SSA**: Salp Swarm Algorithm for optimization
//! - **BA**: Bat Algorithm with echolocation
//! - **DFA**: Dragonfly Algorithm with flocking behavior
//! - **WOA**: Whale Optimization Algorithm
//! - **MFO**: Moth-Flame Optimization
//! - **SCA**: Sine Cosine Algorithm
//!
//! ### Advanced Features
//! - **CombinatorialDiversityFusionAnalyzer**: Dynamic algorithm fusion
//! - **Lock-free parallel execution**: Ultra-high performance
//! - **SIMD optimization**: Vectorized operations
//! - **Adaptive parameter tuning**: Self-optimizing algorithms
//! - **Real-time performance monitoring**: Comprehensive metrics
//!
//! ## Architecture
//!
//! The framework is built on the existing CDFA infrastructure, leveraging:
//! - Ultra-optimized parallel processing from `cdfa-parallel`
//! - SIMD acceleration from `cdfa-simd` 
//! - Lock-free data structures for minimal contention
//! - NUMA-aware thread management
//!
//! ## Example Usage
//!
//! ```rust
//! use swarm_intelligence::{
//!     SwarmAlgorithm, ParticleSwarmOptimization, 
//!     CombinatorialDiversityFusionAnalyzer, OptimizationProblem
//! };
//!
//! // Define optimization problem
//! let problem = OptimizationProblem::new()
//!     .dimensions(10)
//!     .bounds(-10.0, 10.0)
//!     .objective(|x| x.iter().map(|xi| xi.powi(2)).sum::<f64>());
//!
//! // Create PSO algorithm
//! let mut pso = ParticleSwarmOptimization::new()
//!     .population_size(50)
//!     .inertia_weight(0.9)
//!     .cognitive_coefficient(2.0)
//!     .social_coefficient(2.0);
//!
//! // Initialize and run optimization
//! pso.initialize(problem)?;
//! let result = pso.optimize(1000)?;
//!
//! println!("Best solution: {:?}", result.best_position);
//! println!("Best fitness: {}", result.best_fitness);
//! ```

pub mod core;
pub mod algorithms;
pub mod cdfa;
pub mod benchmarks;

// PADS (Panarchy Adaptive Decision System) module
pub mod pads;

// Core traits and types
pub use core::{
    SwarmAlgorithm, SwarmResult, OptimizationProblem, 
    Population, Individual, Fitness, Position
};

// Algorithm implementations
pub use algorithms::{
    ParticleSwarmOptimization, AntColonyOptimization,
    DifferentialEvolution, ArtificialBeeColony,
    CuckooSearch, BacterialForagingOptimization,
    FireflyAlgorithm, GreyWolfOptimizer,
    SalpSwarmAlgorithm, BatAlgorithm,
    DragonflyAlgorithm, WhaleOptimizationAlgorithm,
    MothFlameOptimization, SineCosineAlgorithm
};

// CDFA integration
pub use cdfa::{
    CombinatorialDiversityFusionAnalyzer,
    AlgorithmPool, FusionStrategy, DiversityMetrics,
    PerformanceTracker, AdaptiveParameterTuning
};

// PADS exports
pub use pads::{
    PadsSystem, PadsConfig, PanarchyFramework, AdaptiveDecisionEngine,
    SystemCoordinator, DecisionLayer, AdaptiveCyclePhase,
    init_pads, init_pads_with_config
};

// Error types
pub use core::SwarmError;

/// Initialize the swarm intelligence framework
///
/// This sets up the parallel processing infrastructure, configures
/// NUMA-aware thread pools, and prepares SIMD optimizations.
///
/// # Arguments
/// * `threads` - Optional number of threads to use (defaults to CPU count)
/// * `enable_simd` - Whether to enable SIMD optimizations
/// * `enable_metrics` - Whether to enable performance metrics collection
///
/// # Returns
/// Result indicating successful initialization or error
pub async fn initialize(
    threads: Option<usize>,
    enable_simd: bool,
    enable_metrics: bool,
) -> Result<(), SwarmError> {
    // Initialize CDFA parallel infrastructure
    cdfa_parallel::initialize(threads)
        .map_err(|e| SwarmError::InitializationError(e.to_string()))?;
    
    // Configure SIMD if requested and available
    #[cfg(feature = "simd")]
    if enable_simd {
        core::simd::initialize_simd_support()?;
    }
    
    // Initialize metrics collection
    #[cfg(feature = "metrics")]
    if enable_metrics {
        core::metrics::initialize_metrics_collection()?;
    }
    
    tracing::info!(
        "Swarm Intelligence framework initialized with {} threads, SIMD: {}, Metrics: {}",
        threads.unwrap_or_else(num_cpus::get),
        enable_simd,
        enable_metrics
    );
    
    Ok(())
}

/// Get information about available algorithms
pub fn available_algorithms() -> Vec<&'static str> {
    vec![
        "ParticleSwarmOptimization",
        "AntColonyOptimization", 
        "DifferentialEvolution",
        "ArtificialBeeColony",
        "CuckooSearch",
        "BacterialForagingOptimization",
        "FireflyAlgorithm",
        "GreyWolfOptimizer",
        "SalpSwarmAlgorithm",
        "BatAlgorithm",
        "DragonflyAlgorithm",
        "WhaleOptimizationAlgorithm",
        "MothFlameOptimization",
        "SineCosineAlgorithm",
    ]
}

/// Get system capabilities and performance characteristics
pub fn system_info() -> core::SystemInfo {
    core::SystemInfo::collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_framework_initialization() {
        let result = initialize(Some(2), false, false).await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_available_algorithms() {
        let algorithms = available_algorithms();
        assert!(algorithms.len() >= 14);
        assert!(algorithms.contains(&"ParticleSwarmOptimization"));
        assert!(algorithms.contains(&"AntColonyOptimization"));
    }
    
    #[test]
    fn test_system_info() {
        let info = system_info();
        assert!(info.cpu_cores > 0);
        assert!(info.memory_gb > 0.0);
    }
}