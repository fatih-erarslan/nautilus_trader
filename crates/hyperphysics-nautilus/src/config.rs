//! Configuration for the HyperPhysics-Nautilus integration.

use serde::{Deserialize, Serialize};

/// Configuration for the integration bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Minimum confidence threshold for order submission
    pub min_confidence_threshold: f64,

    /// Maximum position size (normalized 0-1)
    pub max_position_size: f64,

    /// Enable physics simulation
    pub enable_physics: bool,

    /// Enable biomimetic optimization
    pub enable_optimization: bool,

    /// Enable Byzantine consensus validation
    pub enable_consensus: bool,

    /// Target latency in microseconds
    pub target_latency_us: u64,

    /// Number of return periods to track
    pub return_lookback: usize,

    /// Volatility estimation window
    pub volatility_window: usize,

    /// Order ID prefix for HyperPhysics orders
    pub order_id_prefix: String,

    /// Strategy ID for Nautilus registration
    pub strategy_id: String,

    /// Enable detailed latency logging
    pub log_latency: bool,

    /// Physics engine selection
    pub physics_engine: PhysicsEngineConfig,

    /// Optimization algorithm selection
    pub optimization_config: OptimizationConfig,
}

/// Physics engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsEngineConfig {
    /// Rapier physics (default, SIMD optimized)
    Rapier,
    /// Jolt physics (deterministic)
    Jolt,
    /// Warp GPU physics
    Warp,
    /// No physics simulation
    None,
}

/// Optimization algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Use Tier 1 algorithms (<1ms)
    pub tier1_enabled: bool,
    /// Use ensemble of algorithms
    pub use_ensemble: bool,
    /// Population size for optimization
    pub population_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.5,
            max_position_size: 1.0,
            enable_physics: true,
            enable_optimization: true,
            enable_consensus: true,
            target_latency_us: 1000, // 1ms target
            return_lookback: 20,
            volatility_window: 20,
            order_id_prefix: "HP".to_string(),
            strategy_id: "HyperPhysicsStrategy-001".to_string(),
            log_latency: true,
            physics_engine: PhysicsEngineConfig::Rapier,
            optimization_config: OptimizationConfig::default(),
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            tier1_enabled: true,
            use_ensemble: false, // Single algorithm for speed
            population_size: 20,
            max_iterations: 25,
        }
    }
}

impl IntegrationConfig {
    /// Create a configuration optimized for HFT
    pub fn hft_optimized() -> Self {
        Self {
            min_confidence_threshold: 0.6,
            max_position_size: 0.5,
            enable_physics: true,
            enable_optimization: true,
            enable_consensus: true,
            target_latency_us: 500, // 500μs target
            return_lookback: 10,
            volatility_window: 10,
            order_id_prefix: "HP-HFT".to_string(),
            strategy_id: "HyperPhysicsHFT-001".to_string(),
            log_latency: false, // Disable for performance
            physics_engine: PhysicsEngineConfig::Rapier,
            optimization_config: OptimizationConfig {
                tier1_enabled: true,
                use_ensemble: false,
                population_size: 15,
                max_iterations: 15,
            },
        }
    }

    /// Create a configuration for backtesting
    pub fn backtest() -> Self {
        Self {
            min_confidence_threshold: 0.4,
            max_position_size: 1.0,
            enable_physics: true,
            enable_optimization: true,
            enable_consensus: false, // Skip consensus in backtest
            target_latency_us: 10000, // 10ms acceptable in backtest
            return_lookback: 50,
            volatility_window: 50,
            order_id_prefix: "HP-BT".to_string(),
            strategy_id: "HyperPhysicsBacktest-001".to_string(),
            log_latency: true,
            physics_engine: PhysicsEngineConfig::Jolt, // Deterministic for reproducibility
            optimization_config: OptimizationConfig {
                tier1_enabled: true,
                use_ensemble: true, // Use ensemble for better signals
                population_size: 30,
                max_iterations: 50,
            },
        }
    }
}
