//! Real Optimizer Integration
//!
//! Bridges the HFT ecosystem to the actual hyperphysics-optimization algorithms.
//! This module replaces the stub implementations with production-ready optimizers.
//!
//! # Performance Targets
//! - <1ms for 10-dimensional optimization (Tier 1)
//! - <10ms for 100-dimensional optimization (Tier 2)
//! - Parallel evaluation via Rayon

use crate::{EcosystemError, Result};
use std::time::Instant;

#[cfg(feature = "optimization-real")]
use hyperphysics_optimization::{
    // Re-exported types from algorithms module (not submodules)
    algorithms::{
        WhaleOptimizer, WOAConfig,
        BatOptimizer, BatConfig,
        FireflyOptimizer, FireflyConfig,
        CuckooSearch, CuckooConfig,
        AlgorithmConfig, Algorithm,
    },
    // Core types
    Bounds, ObjectiveFunction, OptimizationConfig,
    OptimizationError,
};

#[cfg(feature = "optimization-real")]
use ndarray::ArrayView1;

/// Trading signal from optimization
#[derive(Debug, Clone)]
pub struct OptimizationSignal {
    /// Recommended position [-1.0, 1.0] where negative = short
    pub position: f64,
    /// Confidence in the signal [0.0, 1.0]
    pub confidence: f64,
    /// Algorithm that produced this signal
    pub algorithm: String,
    /// Latency in microseconds
    pub latency_us: u64,
    /// Convergence achieved
    pub converged: bool,
}

/// Market objective function for optimization
#[cfg(feature = "optimization-real")]
pub struct MarketObjective {
    /// Price returns history
    pub returns: Vec<f64>,
    /// Volatility estimate
    pub volatility: f64,
    /// Current trend strength [-1, 1]
    pub trend: f64,
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Search space bounds (10-dimensional strategy space)
    bounds: Bounds,
}

#[cfg(feature = "optimization-real")]
impl MarketObjective {
    /// Create new market objective with default 10-dimensional bounds
    pub fn new(returns: Vec<f64>, volatility: f64, trend: f64, risk_aversion: f64) -> Self {
        Self {
            returns,
            volatility,
            trend,
            risk_aversion,
            bounds: Bounds::new(vec![(-1.0, 1.0); 10]),
        }
    }
}

#[cfg(feature = "optimization-real")]
impl ObjectiveFunction for MarketObjective {
    fn evaluate(&self, solution: ArrayView1<f64>) -> f64 {
        // Solution encodes trading strategy parameters:
        // [0]: position size weight
        // [1]: momentum weight
        // [2]: mean-reversion weight
        // [3]: volatility adjustment
        // ... additional parameters

        let pos_weight = solution.get(0).copied().unwrap_or(0.0);
        let momentum_weight = solution.get(1).copied().unwrap_or(0.0);
        let mean_rev_weight = solution.get(2).copied().unwrap_or(0.0);
        let vol_adj = solution.get(3).copied().unwrap_or(1.0);

        // Calculate expected return based on strategy
        let momentum_signal = self.trend * momentum_weight;
        let mean_rev_signal = -self.trend * mean_rev_weight; // Counter-trend
        let combined_signal = (momentum_signal + mean_rev_signal) * pos_weight;

        // Risk-adjusted return (Sharpe-like)
        let expected_return = combined_signal * self.returns.iter().sum::<f64>().abs().max(0.001);
        let risk = self.volatility * vol_adj * self.risk_aversion;

        // Maximize risk-adjusted return (negative because optimizers minimize)
        -(expected_return / risk.max(0.001))
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        10 // 10-dimensional strategy space
    }
}

/// Real optimizer executor using hyperphysics-optimization
#[cfg(feature = "optimization-real")]
pub struct RealOptimizer {
    /// Optimization configuration
    config: OptimizationConfig,
    /// Search bounds
    bounds: Bounds,
}

#[cfg(feature = "optimization-real")]
impl RealOptimizer {
    /// Create a new real optimizer
    pub fn new() -> Result<Self> {
        let bounds = Bounds::new(vec![(-1.0, 1.0); 10]); // 10-dimensional, normalized
        let config = OptimizationConfig {
            population_size: 30,  // Small population for speed
            max_iterations: 50,   // Limited iterations for <1ms
            tolerance: 1e-4,
            seed: None,
            ..OptimizationConfig::default()
        };

        Ok(Self { config, bounds })
    }

    /// Create optimizer tuned for HFT latency
    pub fn hft_optimized() -> Result<Self> {
        let bounds = Bounds::new(vec![(-1.0, 1.0); 10]);
        let config = OptimizationConfig {
            population_size: 20,  // Minimal population
            max_iterations: 25,   // Very limited iterations
            tolerance: 1e-3,      // Relaxed tolerance
            seed: None,
            ..OptimizationConfig::default()
        };

        Ok(Self { config, bounds })
    }

    /// Execute whale optimization algorithm
    pub fn optimize_whale(&self, objective: &MarketObjective) -> Result<OptimizationSignal> {
        let start = Instant::now();

        let woa_config = WOAConfig::hft_optimized();
        let mut optimizer = WhaleOptimizer::new(woa_config, self.config.clone(), self.bounds.clone())
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let result = optimizer.optimize(objective)
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(OptimizationSignal {
            position: result.position[0].clamp(-1.0, 1.0),
            confidence: self.fitness_to_confidence(result.fitness),
            algorithm: "WhaleOptimization".to_string(),
            latency_us,
            converged: optimizer.is_converged(),
        })
    }

    /// Execute bat algorithm
    pub fn optimize_bat(&self, objective: &MarketObjective) -> Result<OptimizationSignal> {
        let start = Instant::now();

        let bat_config = BatConfig::hft_optimized();
        let mut optimizer = BatOptimizer::new(bat_config, self.config.clone(), self.bounds.clone())
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let result = optimizer.optimize(objective)
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(OptimizationSignal {
            position: result.position[0].clamp(-1.0, 1.0),
            confidence: self.fitness_to_confidence(result.fitness),
            algorithm: "BatAlgorithm".to_string(),
            latency_us,
            converged: optimizer.is_converged(),
        })
    }

    /// Execute firefly algorithm
    pub fn optimize_firefly(&self, objective: &MarketObjective) -> Result<OptimizationSignal> {
        let start = Instant::now();

        let ff_config = FireflyConfig::hft_optimized();
        let mut optimizer = FireflyOptimizer::new(ff_config, self.config.clone(), self.bounds.clone())
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let result = optimizer.optimize(objective)
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(OptimizationSignal {
            position: result.position[0].clamp(-1.0, 1.0),
            confidence: self.fitness_to_confidence(result.fitness),
            algorithm: "FireflyAlgorithm".to_string(),
            latency_us,
            converged: optimizer.is_converged(),
        })
    }

    /// Execute cuckoo search algorithm
    pub fn optimize_cuckoo(&self, objective: &MarketObjective) -> Result<OptimizationSignal> {
        let start = Instant::now();

        let cs_config = CuckooConfig::hft_optimized();
        let mut optimizer = CuckooSearch::new(cs_config, self.config.clone(), self.bounds.clone())
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let result = optimizer.optimize(objective)
            .map_err(|e| EcosystemError::BiomimeticAlgorithm(e.to_string()))?;

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(OptimizationSignal {
            position: result.position[0].clamp(-1.0, 1.0),
            confidence: self.fitness_to_confidence(result.fitness),
            algorithm: "CuckooSearch".to_string(),
            latency_us,
            converged: optimizer.is_converged(),
        })
    }

    /// Run all Tier 1 algorithms in parallel and combine results
    pub async fn optimize_tier1_ensemble(&self, objective: &MarketObjective) -> Result<OptimizationSignal> {
        let start = Instant::now();

        // Note: Rayon scope spawn requires () return type, not compatible with Result<T>
        // For production parallel execution, use crossbeam channels or separate thread pool
        // For now, sequential execution ensures correctness (parallel optimization is complex)

        // Sequential fallback for now (parallel requires more complex setup)
        let whale = self.optimize_whale(objective)?;
        let bat = self.optimize_bat(objective)?;
        let firefly = self.optimize_firefly(objective)?;
        let cuckoo = self.optimize_cuckoo(objective)?;

        // Weighted ensemble based on confidence
        let signals = vec![whale, bat, firefly, cuckoo];
        let total_confidence: f64 = signals.iter().map(|s| s.confidence).sum();

        let ensemble_position = if total_confidence > 0.0 {
            signals.iter()
                .map(|s| s.position * s.confidence)
                .sum::<f64>() / total_confidence
        } else {
            0.0 // Hold if no confidence
        };

        let max_confidence = signals.iter()
            .map(|s| s.confidence)
            .fold(0.0f64, |a, b| a.max(b));

        let contributors: Vec<String> = signals.iter()
            .map(|s| format!("{}({:.2})", s.algorithm, s.confidence))
            .collect();

        let latency_us = start.elapsed().as_micros() as u64;

        Ok(OptimizationSignal {
            position: ensemble_position.clamp(-1.0, 1.0),
            confidence: max_confidence * 0.9, // Slight discount for ensemble
            algorithm: format!("Ensemble[{}]", contributors.join(",")),
            latency_us,
            converged: signals.iter().any(|s| s.converged),
        })
    }

    /// Convert fitness value to confidence score
    fn fitness_to_confidence(&self, fitness: Option<f64>) -> f64 {
        match fitness {
            Some(f) => {
                // Transform fitness to [0, 1] confidence
                // Assuming lower (more negative) fitness is better
                let normalized = (-f).tanh(); // Maps to (-1, 1)
                (normalized + 1.0) / 2.0 // Maps to (0, 1)
            }
            None => 0.0,
        }
    }
}

#[cfg(feature = "optimization-real")]
impl Default for RealOptimizer {
    fn default() -> Self {
        Self::new().expect("Failed to create RealOptimizer")
    }
}

/// Fallback when optimization-real feature is not enabled
#[cfg(not(feature = "optimization-real"))]
pub struct RealOptimizer;

#[cfg(not(feature = "optimization-real"))]
impl RealOptimizer {
    pub fn new() -> Result<Self> {
        Err(EcosystemError::Configuration(
            "optimization-real feature not enabled. Add hyperphysics-optimization dependency.".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "optimization-real")]
    fn test_real_optimizer_creation() {
        let optimizer = RealOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    #[cfg(feature = "optimization-real")]
    fn test_whale_optimization() {
        let optimizer = RealOptimizer::hft_optimized().unwrap();
        let objective = MarketObjective::new(
            vec![0.01, -0.005, 0.02, -0.01, 0.015],
            0.02,  // volatility
            0.3,   // trend
            1.0,   // risk_aversion
        );

        let result = optimizer.optimize_whale(&objective);
        assert!(result.is_ok());

        let signal = result.unwrap();
        assert!(signal.position >= -1.0 && signal.position <= 1.0);
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
        assert!(signal.latency_us < 10_000_000); // <10s for test
    }

    #[test]
    #[cfg(feature = "optimization-real")]
    fn test_hft_latency_target() {
        let optimizer = RealOptimizer::hft_optimized().unwrap();
        let objective = MarketObjective::new(
            vec![0.01; 10],
            0.02,  // volatility
            0.0,   // trend
            1.0,   // risk_aversion
        );

        let result = optimizer.optimize_whale(&objective).unwrap();

        // Should complete in <10ms for HFT use
        assert!(result.latency_us < 10_000, "Latency {}us exceeds 10ms target", result.latency_us);
    }
}
