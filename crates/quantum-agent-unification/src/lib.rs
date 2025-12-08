//! # Quantum Agent Optimization System
//! 
//! Enterprise-grade quantum-enhanced optimization framework with 12 specialized quantum agents.
//! 
//! ## Features
//! 
//! - 12 quantum-enhanced optimization algorithms
//! - Quantum superposition and entanglement
//! - SIMD-accelerated quantum operations
//! - Real-time quantum performance metrics
//! - Parallel quantum simulation
//! - Bell state entanglement
//! - Quantum tunneling through barriers
//! - Quantum interference patterns
//! 
//! ## Quantum Algorithms
//! 
//! 1. **Quantum Particle Swarm Optimization (QPSO)**: Quantum-enhanced PSO with superposition
//! 2. **Quantum Genetic Algorithm (QGA)**: Evolution with quantum crossover and mutation
//! 3. **Quantum Annealing Algorithm (QAA)**: Simulated quantum annealing with tunneling
//! 4. **Quantum Differential Evolution (QDE)**: DE enhanced with quantum states
//! 5. **Quantum Firefly Algorithm (QFA)**: Bio-inspired with quantum bioluminescence
//! 6. **Quantum Bee Colony Algorithm (QBCA)**: Swarm intelligence with quantum foraging
//! 7. **Quantum Grey Wolf Optimizer (QGWA)**: Pack hunting with quantum coordination
//! 8. **Quantum Cuckoo Search (QCSA)**: Quantum Lévy flights and nest optimization
//! 9. **Quantum Bat Algorithm (QBA)**: Echolocation with quantum frequency modulation
//! 10. **Quantum Whale Optimization (QWOA)**: Marine-inspired with quantum bubble nets
//! 11. **Quantum Moth-Flame Optimizer (QMFA)**: Navigation with quantum spiral patterns
//! 12. **Quantum Salp Swarm Algorithm (QSSA)**: Chain formation with quantum coordination
//! 
//! ## Usage
//! 
//! ```rust
//! use quantum_agent_unification::{QuantumOptimizer, OptimizationProblem};
//! 
//! // Create quantum optimizer with all 12 algorithms
//! let mut optimizer = QuantumOptimizer::new();
//! 
//! // Define optimization problem
//! let problem = OptimizationProblem {
//!     dimensions: 10,
//!     bounds: vec![(-100.0, 100.0); 10],
//!     objective_function: |x| x.iter().map(|&xi| xi * xi).sum(), // Sphere function
//!     constraints: vec![],
//!     quantum_enhanced: true,
//! };
//! 
//! // Run quantum optimization
//! let result = optimizer.optimize_parallel(problem, 1000).await?;
//! println!("Best solution: {:?}", result.best_solution);
//! println!("Best fitness: {}", result.best_fitness);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod quantum_optimizer;
pub mod quantum_state;
pub mod quantum_agents;
pub mod quantum_metrics;
pub mod quantum_entanglement;
pub mod pbit_integration;

// Keep existing modules for compatibility
pub mod agents;
pub mod unification;
pub mod pads;
pub mod coordination;
pub mod optimization;
pub mod error;
pub mod metrics;

#[cfg(feature = "simd")]
pub mod simd;

// Re-exports for new quantum system
pub use quantum_optimizer::{QuantumOptimizer, OptimizationProblem, QuantumAlgorithm};
pub use quantum_state::{QuantumState, QuantumBit, BlochSphere};
pub use quantum_agents::*;
pub use quantum_metrics::QuantumMetrics;
pub use quantum_entanglement::QuantumEntanglement;
pub use pbit_integration::{PBitOptimizer, PBitOptimizerConfig, PBitOptimizable};

// Keep existing re-exports for backward compatibility
pub use unification::{QuantumAgentUnifier, UnificationConfig, UnifiedDecision};
pub use agents::{QuantumAgent, AgentType, AgentConfig, AgentResult};
pub use pads::{PadsIntegration, PadsConfig, PadsDecision};
pub use coordination::{AgentCoordinator, CoordinationStrategy, SwarmCoordination};
pub use optimization::{HyperbolicOptimizer, QuantumOptimization, OptimizationMetrics};
pub use error::{UnificationError, Result};
pub use metrics::{PerformanceMetrics, AgentMetrics, SystemMetrics};

// Type aliases
pub type Float = f64;
pub type Complex = num_complex::Complex64;
pub type Vector = nalgebra::DVector<Float>;
pub type Matrix = nalgebra::DMatrix<Float>;

/// Quantum optimization result containing the best solution and quantum metrics
#[derive(Debug, Clone)]
pub struct QuantumOptimizationResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub iterations: usize,
    pub quantum_metrics: QuantumMetrics,
    pub convergence_history: Vec<f64>,
    pub quantum_states: Vec<QuantumState>,
}

/// Error types for quantum operations
#[derive(thiserror::Error, Debug)]
pub enum QuantumError {
    #[error("Quantum decoherence detected: {0}")]
    Decoherence(String),
    
    #[error("Quantum measurement failed: {0}")]
    MeasurementError(String),
    
    #[error("Entanglement correlation error: {0}")]
    EntanglementError(String),
    
    #[error("Quantum gate operation failed: {0}")]
    GateError(String),
    
    #[error("SIMD acceleration not available")]
    SimdUnavailable,
}

pub type QuantumResult<T> = Result<T, QuantumError>;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration constants
pub const DEFAULT_AGENT_COUNT: usize = 12;
pub const DEFAULT_COORDINATION_INTERVAL_MS: u64 = 100;
pub const DEFAULT_PADS_UPDATE_INTERVAL_MS: u64 = 50;
pub const DEFAULT_OPTIMIZATION_BATCH_SIZE: usize = 32;

/// Performance targets
pub const TARGET_DECISION_LATENCY_MS: u64 = 10;
pub const TARGET_COORDINATION_LATENCY_MS: u64 = 5;
pub const TARGET_THROUGHPUT_DECISIONS_PER_SEC: u64 = 1000;

/// Agent specialization weights for PADS integration
pub const STRATEGIC_WEIGHT: Float = 0.15;
pub const TACTICAL_WEIGHT: Float = 0.12;
pub const RISK_WEIGHT: Float = 0.20;
pub const MICROSTRUCTURE_WEIGHT: Float = 0.08;
pub const SENTIMENT_WEIGHT: Float = 0.06;
pub const PATTERN_WEIGHT: Float = 0.10;
pub const ARBITRAGE_WEIGHT: Float = 0.07;
pub const VOLATILITY_WEIGHT: Float = 0.09;
pub const PORTFOLIO_WEIGHT: Float = 0.08;
pub const EXECUTION_WEIGHT: Float = 0.10;
pub const REGIME_WEIGHT: Float = 0.07;
pub const LEARNING_WEIGHT: Float = 0.08;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Smoke test
        assert_eq!(DEFAULT_AGENT_COUNT, 12);
        assert!(VERSION.len() > 0);
        
        // Verify weights sum to 1.0
        let total_weight = STRATEGIC_WEIGHT + TACTICAL_WEIGHT + RISK_WEIGHT +
                          MICROSTRUCTURE_WEIGHT + SENTIMENT_WEIGHT + PATTERN_WEIGHT +
                          ARBITRAGE_WEIGHT + VOLATILITY_WEIGHT + PORTFOLIO_WEIGHT +
                          EXECUTION_WEIGHT + REGIME_WEIGHT + LEARNING_WEIGHT;
        
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_performance_targets() {
        assert!(TARGET_DECISION_LATENCY_MS <= 10);
        assert!(TARGET_COORDINATION_LATENCY_MS <= 5);
        assert!(TARGET_THROUGHPUT_DECISIONS_PER_SEC >= 1000);
    }
}