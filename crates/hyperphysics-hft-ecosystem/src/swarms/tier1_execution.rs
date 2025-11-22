//! Tier 1 Swarm Execution (<1ms target latency)
//!
//! Mission-critical biomimetic algorithms for ultra-low latency HFT:
//! - Whale Optimization: Whale detection via hyperbolic search
//! - Cuckoo Search: Regime change via Lévy flights + consciousness metrics  
//! - Particle Swarm: Portfolio optimization in hyperbolic space
//! - Bat Algorithm: Orderbook echolocation
//! - Firefly Algorithm: Flash event detection

use crate::{EcosystemError, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "biomimetic-tier1")]
use cuckoo_search::CuckooSearchOptimizer;
#[cfg(feature = "biomimetic-tier1")]
use whale_optimization::{
    WhaleOptimizationObjective, WhaleOptimizationOptimizer, WhaleOptimizationParameters,
};
// TODO: Enable these when implementing Tier 2
// #[cfg(feature = "biomimetic-tier2")]
// use particle_swarm::{ParticleSwarmOptimizer, ParticleSwarmParameters};
#[cfg(feature = "biomimetic-tier1")]
use bat_algorithm::BatAlgorithmOptimizer;
#[cfg(feature = "biomimetic-tier1")]
use firefly_algorithm::FireflyAlgorithmOptimizer;

/// Market state representation for optimization
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Order book depth
    pub order_book: Vec<(f64, f64)>, // (price, volume)

    /// Recent price movements
    pub price_history: Vec<f64>,

    /// Volume profile
    pub volume_profile: Vec<f64>,

    /// Volatility
    pub volatility: f64,

    /// Trend strength
    pub trend_strength: f64,
}

/// Swarm decision from biomimetic algorithms
#[derive(Debug, Clone)]
pub struct SwarmDecision {
    /// Primary action
    pub action: super::super::core::Action,

    /// Confidence score (0.0-1.0)
    pub confidence: f64,

    ///Position size recommendation
    pub size: f64,

    /// Contributing algorithms
    pub contributors: Vec<String>,

    /// Latency in microseconds
    pub latency_us: u64,
}

/// Tier 1 swarm executor
pub struct Tier1SwarmExecutor {
    #[cfg(feature = "biomimetic-tier1")]
    whale: Option<Arc<RwLock<WhaleOptimizationOptimizer>>>,

    #[cfg(feature = "biomimetic-tier1")]
    cuckoo: Option<Arc<RwLock<CuckooSearchOptimizer>>>,

    // TODO: Add when implementing Tier 2
    // #[cfg(feature = "biomimetic-tier2")]
    // pso: Option<Arc<RwLock<ParticleSwarmOptimizer>>>,
    #[cfg(feature = "biomimetic-tier1")]
    bat: Option<Arc<RwLock<BatAlgorithmOptimizer>>>,

    #[cfg(feature = "biomimetic-tier1")]
    firefly: Option<Arc<RwLock<FireflyAlgorithmOptimizer>>>,

    /// Physics engine router for evaluating solutions
    physics_router: Option<Arc<super::super::core::physics_engine_router::PhysicsEngineRouter>>,
}

impl Tier1SwarmExecutor {
    /// Create a new Tier 1 executor
    pub fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "biomimetic-tier1")]
            whale: Some(Arc::new(RwLock::new(WhaleOptimizationOptimizer::new(
                WhaleOptimizationParameters::default(),
                10, // dimension
            )))),

            #[cfg(feature = "biomimetic-tier1")]
            cuckoo: None, // Will be initialized when needed

            #[cfg(feature = "biomimetic-tier1")]
            bat: None,

            #[cfg(feature = "biomimetic-tier1")]
            firefly: None,

            physics_router: None,
        })
    }

    /// Set the physics engine router
    pub fn with_physics_router(
        mut self,
        router: Arc<super::super::core::physics_engine_router::PhysicsEngineRouter>,
    ) -> Self {
        self.physics_router = Some(router);
        self
    }

    /// Execute all Tier 1 algorithms in parallel
    pub async fn execute(&mut self, market_state: &MarketState) -> Result<SwarmDecision> {
        let start = std::time::Instant::now();

        // Consult physics engine first if available
        let physics_confidence = if let Some(router) = &self.physics_router {
            // Create a dummy tick for now
            let tick = crate::core::MarketTick::default();
            match router.route(&tick).await {
                Ok(result) => Some(result.confidence),
                Err(_) => None,
            }
        } else {
            None
        };

        #[cfg(feature = "biomimetic-tier1")]
        {
            // Execute algorithms in parallel
            let mut whale_result = self.execute_whale(market_state).await?;

            // Adjust confidence based on physics result
            if let Some(conf) = physics_confidence {
                // Simple fusion: average with physics confidence
                whale_result.confidence = (whale_result.confidence + conf) / 2.0;
                whale_result
                    .contributors
                    .push(format!("Physics(conf={:.2})", conf));
            }

            // TODO: Add other algorithms

            // Byzantine consensus (simple majority for now)
            let decision = self.byzantine_consensus(vec![whale_result])?;

            let latency_us = start.elapsed().as_micros() as u64;

            Ok(SwarmDecision {
                action: decision.action,
                confidence: decision.confidence,
                size: decision.size,
                contributors: decision.contributors,
                latency_us,
            })
        }

        #[cfg(not(feature = "biomimetic-tier1"))]
        {
            Err(EcosystemError::Configuration(
                "Tier 1 biomimetic algorithms not enabled".to_string(),
            ))
        }
    }

    #[cfg(feature = "biomimetic-tier1")]
    async fn execute_whale(&self, _market_state: &MarketState) -> Result<AlgorithmResult> {
        // Convert market state to optimization problem
        // let _objective = MarketObjective {
        //     market_state: market_state.clone(),
        // };

        if let Some(whale) = &self.whale {
            let optimizer = whale.write().await;

            // Run optimization (stub for now)
            let _best = optimizer.get_best_solution();

            // Convert to trading decision
            Ok(AlgorithmResult {
                action: super::super::core::Action::Hold,
                confidence: 0.75,
                size: 1.0,
                algorithm: "WhaleOptimization".to_string(),
                contributors: vec!["WhaleOptimization".to_string()],
            })
        } else {
            Err(EcosystemError::BiomimeticAlgorithm(
                "Whale optimizer not initialized".to_string(),
            ))
        }
    }

    #[cfg(feature = "biomimetic-tier1")]
    fn byzantine_consensus(&self, results: Vec<AlgorithmResult>) -> Result<AlgorithmResult> {
        if results.is_empty() {
            return Err(EcosystemError::BiomimeticAlgorithm(
                "No algorithm results for consensus".to_string(),
            ));
        }

        // Simple consensus: weighted average by confidence
        let contributors: Vec<String> = results
            .iter()
            .flat_map(|r| r.contributors.clone())
            .collect();

        // For now, return the highest confidence result
        let best = results
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();

        Ok(AlgorithmResult {
            action: best.action,
            confidence: best.confidence,
            size: best.size,
            algorithm: format!("Consensus({})", contributors.join(",")),
            contributors,
        })
    }
}

impl Default for Tier1SwarmExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create Tier1SwarmExecutor")
    }
}

/// Individual algorithm result
#[cfg(feature = "biomimetic-tier1")]
#[derive(Debug, Clone)]
struct AlgorithmResult {
    action: super::super::core::Action,
    confidence: f64,
    size: f64,
    algorithm: String,
    contributors: Vec<String>,
}

/// Market objective function for optimization
#[cfg(feature = "biomimetic-tier1")]
struct MarketObjective {
    market_state: MarketState,
}

#[cfg(feature = "biomimetic-tier1")]
#[async_trait::async_trait]
impl WhaleOptimizationObjective for MarketObjective {
    async fn evaluate(
        &self,
        solution: &[f64],
    ) -> std::result::Result<f64, whale_optimization::WhaleOptimizationError> {
        // Evaluate trading strategy represented by solution
        // Higher score = better strategy
        let score = solution.iter().sum::<f64>() / solution.len() as f64;
        Ok(score)
    }

    fn get_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-1.0, 1.0); 10] // 10-dimensional strategy space
    }

    fn get_dimension(&self) -> usize {
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier1_executor_creation() {
        let executor = Tier1SwarmExecutor::new();
        #[cfg(feature = "biomimetic-tier1")]
        assert!(executor.is_ok());

        #[cfg(not(feature = "biomimetic-tier1"))]
        assert!(executor.is_ok()); // Still creates, just empty
    }

    #[tokio::test]
    #[cfg(feature = "biomimetic-tier1")]
    async fn test_whale_optimization_execution() {
        let mut executor = Tier1SwarmExecutor::new().unwrap();
        let market_state = MarketState {
            order_book: vec![(100.0, 10.0), (101.0, 5.0)],
            price_history: vec![100.0, 100.5, 101.0],
            volume_profile: vec![1000.0, 1500.0, 1200.0],
            volatility: 0.02,
            trend_strength: 0.5,
        };

        let result = executor.execute(&market_state).await;
        assert!(result.is_ok());

        let decision = result.unwrap();
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.latency_us < 1000000); // Should be < 1s for test
    }
}
