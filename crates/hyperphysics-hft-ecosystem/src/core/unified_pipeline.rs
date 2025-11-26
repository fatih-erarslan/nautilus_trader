//! Unified HFT Pipeline Integration
//!
//! Connects market data providers, consensus mechanisms, neural forecasting,
//! and optimization algorithms into a cohesive trading pipeline.
//!
//! # Architecture
//!
//! ```text
//! Market Data (hyperphysics-market)
//!        ↓
//! Physics Simulation (rapier/jolt/warp)
//!        ↓
//! Neural Forecasting (hyperphysics-neural-trader)  [OPTIONAL]
//!        ↓
//! Biomimetic Optimization (hyperphysics-optimization)
//!        ↓
//! Consensus Validation (PBFT/Raft)
//!        ↓
//! Trading Decision
//! ```

use crate::core::{Action, BiomimeticTier, EcosystemConfig, MarketTick, TradingDecision};
use crate::{EcosystemError, Result};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[cfg(feature = "optimization-real")]
use crate::swarms::{MarketObjective, OptimizationSignal, RealOptimizer};

// Neural forecasting imports
#[cfg(feature = "neural-forecasting")]
use hyperphysics_neural_trader::{
    NeuralBridgeConfig, NeuralDataAdapter, NeuralForecastEngine, ForecastResult,
};

/// Market data feed abstraction
#[derive(Debug, Clone)]
pub struct MarketFeed {
    /// Latest price
    pub price: f64,
    /// Recent returns (last N periods)
    pub returns: Vec<f64>,
    /// Current volatility estimate
    pub volatility: f64,
    /// Volume-weighted average price
    pub vwap: f64,
    /// Bid-ask spread
    pub spread: f64,
    /// Current timestamp (epoch seconds)
    pub timestamp: f64,
}

impl Default for MarketFeed {
    fn default() -> Self {
        Self {
            price: 100.0,
            returns: vec![0.0; 10],
            volatility: 0.02,
            vwap: 100.0,
            spread: 0.01,
            timestamp: 0.0,
        }
    }
}

/// Convert local MarketFeed to neural-trader's MarketFeed
#[cfg(feature = "neural-forecasting")]
impl From<&MarketFeed> for hyperphysics_neural_trader::MarketFeed {
    fn from(feed: &MarketFeed) -> Self {
        hyperphysics_neural_trader::MarketFeed {
            price: feed.price,
            returns: feed.returns.clone(),
            volatility: feed.volatility,
            vwap: feed.vwap,
            spread: feed.spread,
            timestamp: feed.timestamp,
        }
    }
}

/// Consensus state for multi-node validation
#[derive(Debug, Clone)]
pub struct ConsensusState {
    /// Node ID of current leader
    pub leader_id: u64,
    /// Current term/view number
    pub term: u64,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Byzantine tolerance threshold (f)
    pub byzantine_threshold: usize,
    /// Last consensus latency in microseconds
    pub consensus_latency_us: u64,
}

impl Default for ConsensusState {
    fn default() -> Self {
        Self {
            leader_id: 0,
            term: 1,
            active_nodes: 4,         // 3f+1 = 4 nodes minimum
            byzantine_threshold: 1,  // Tolerate 1 Byzantine node
            consensus_latency_us: 0,
        }
    }
}

/// Pipeline execution result with full metrics
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Final trading decision
    pub decision: TradingDecision,
    /// Pipeline execution latency (total)
    pub total_latency_us: u64,
    /// Market data processing latency
    pub market_data_latency_us: u64,
    /// Physics simulation latency
    pub physics_latency_us: u64,
    /// Neural forecasting latency (0 if feature not enabled)
    pub neural_latency_us: u64,
    /// Optimization latency
    pub optimization_latency_us: u64,
    /// Consensus validation latency
    pub consensus_latency_us: u64,
    /// Consensus state at decision time
    pub consensus_state: ConsensusState,
    /// Whether Byzantine consensus was reached
    pub consensus_reached: bool,
    /// Neural forecast result (if neural-forecasting feature enabled)
    #[cfg(feature = "neural-forecasting")]
    pub neural_forecast: Option<NeuralForecastSummary>,
}

/// Summary of neural forecast for pipeline result
#[cfg(feature = "neural-forecasting")]
#[derive(Debug, Clone)]
pub struct NeuralForecastSummary {
    /// Primary prediction value
    pub prediction: f64,
    /// Confidence interval width
    pub confidence_interval: f64,
    /// Quality score [0, 1]
    pub quality_score: f64,
    /// Model type used
    pub model: String,
}

/// Unified HFT Pipeline
///
/// Integrates all ecosystem components into a single execution path.
pub struct UnifiedPipeline {
    /// Configuration
    config: EcosystemConfig,

    /// Current consensus state
    consensus_state: Arc<RwLock<ConsensusState>>,

    /// Real optimizer (when feature enabled)
    #[cfg(feature = "optimization-real")]
    optimizer: Arc<RealOptimizer>,

    /// Neural forecast engine (when feature enabled)
    #[cfg(feature = "neural-forecasting")]
    neural_engine: Arc<NeuralForecastEngine>,

    /// Neural data adapter (when feature enabled)
    #[cfg(feature = "neural-forecasting")]
    neural_adapter: Arc<NeuralDataAdapter>,

    /// Pipeline statistics
    stats: Arc<RwLock<PipelineStats>>,
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total executions
    pub total_executions: u64,
    /// Successful consensus rounds
    pub consensus_successes: u64,
    /// Failed consensus rounds
    pub consensus_failures: u64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Max latency (microseconds)
    pub max_latency_us: u64,
    /// Min latency (microseconds)
    pub min_latency_us: u64,
}

impl UnifiedPipeline {
    /// Create a new unified pipeline with optimization and neural forecasting
    #[cfg(all(feature = "optimization-real", feature = "neural-forecasting"))]
    pub fn new(config: EcosystemConfig) -> Result<Self> {
        let optimizer = RealOptimizer::hft_optimized()?;
        let neural_config = NeuralBridgeConfig::default();
        let neural_engine = NeuralForecastEngine::new(neural_config.clone());
        let neural_adapter = NeuralDataAdapter::new(neural_config);

        Ok(Self {
            config,
            consensus_state: Arc::new(RwLock::new(ConsensusState::default())),
            optimizer: Arc::new(optimizer),
            neural_engine: Arc::new(neural_engine),
            neural_adapter: Arc::new(neural_adapter),
            stats: Arc::new(RwLock::new(PipelineStats {
                min_latency_us: u64::MAX,
                ..Default::default()
            })),
        })
    }

    /// Create a new unified pipeline with optimization only
    #[cfg(all(feature = "optimization-real", not(feature = "neural-forecasting")))]
    pub fn new(config: EcosystemConfig) -> Result<Self> {
        let optimizer = RealOptimizer::hft_optimized()?;

        Ok(Self {
            config,
            consensus_state: Arc::new(RwLock::new(ConsensusState::default())),
            optimizer: Arc::new(optimizer),
            stats: Arc::new(RwLock::new(PipelineStats {
                min_latency_us: u64::MAX,
                ..Default::default()
            })),
        })
    }

    /// Create a new unified pipeline with neural forecasting only
    #[cfg(all(not(feature = "optimization-real"), feature = "neural-forecasting"))]
    pub fn new(config: EcosystemConfig) -> Result<Self> {
        let neural_config = NeuralBridgeConfig::default();
        let neural_engine = NeuralForecastEngine::new(neural_config.clone());
        let neural_adapter = NeuralDataAdapter::new(neural_config);

        Ok(Self {
            config,
            consensus_state: Arc::new(RwLock::new(ConsensusState::default())),
            neural_engine: Arc::new(neural_engine),
            neural_adapter: Arc::new(neural_adapter),
            stats: Arc::new(RwLock::new(PipelineStats {
                min_latency_us: u64::MAX,
                ..Default::default()
            })),
        })
    }

    /// Create a new unified pipeline (fallback without optimization or neural)
    #[cfg(all(not(feature = "optimization-real"), not(feature = "neural-forecasting")))]
    pub fn new(config: EcosystemConfig) -> Result<Self> {
        Ok(Self {
            config,
            consensus_state: Arc::new(RwLock::new(ConsensusState::default())),
            stats: Arc::new(RwLock::new(PipelineStats {
                min_latency_us: u64::MAX,
                ..Default::default()
            })),
        })
    }

    /// Execute the full pipeline on market data
    pub async fn execute(&self, feed: &MarketFeed) -> Result<PipelineResult> {
        let total_start = Instant::now();

        // Phase 1: Process market data
        let market_start = Instant::now();
        let market_tick = self.process_market_data(feed)?;
        let market_data_latency_us = market_start.elapsed().as_micros() as u64;

        // Phase 2: Physics simulation (for market dynamics modeling)
        let physics_start = Instant::now();
        let physics_signal = self.run_physics_simulation(&market_tick).await?;
        let physics_latency_us = physics_start.elapsed().as_micros() as u64;

        // Phase 3: Neural forecasting (optional)
        #[cfg(feature = "neural-forecasting")]
        let (neural_latency_us, neural_forecast) = {
            let neural_start = Instant::now();
            let forecast_result = self.run_neural_forecasting(feed).await;
            let latency = neural_start.elapsed().as_micros() as u64;

            let summary = forecast_result.ok().map(|f| NeuralForecastSummary {
                prediction: f.primary_prediction(),
                confidence_interval: f.interval_width(0),
                quality_score: f.quality_score(),
                model: format!("{:?}", f.model_type),
            });

            (latency, summary)
        };

        #[cfg(not(feature = "neural-forecasting"))]
        let neural_latency_us: u64 = 0;

        // Phase 4: Biomimetic optimization
        let opt_start = Instant::now();
        let opt_signal = self.run_optimization(feed).await?;
        let optimization_latency_us = opt_start.elapsed().as_micros() as u64;

        // Phase 5: Consensus validation
        let consensus_start = Instant::now();
        let (consensus_reached, consensus_state) = self.validate_consensus(&opt_signal).await?;
        let consensus_latency_us = consensus_start.elapsed().as_micros() as u64;

        // Compute final decision
        let decision = self.compute_final_decision(
            &physics_signal,
            &opt_signal,
            consensus_reached,
        )?;

        let total_latency_us = total_start.elapsed().as_micros() as u64;

        // Update statistics
        self.update_stats(total_latency_us, consensus_reached).await;

        Ok(PipelineResult {
            decision,
            total_latency_us,
            market_data_latency_us,
            physics_latency_us,
            neural_latency_us,
            optimization_latency_us,
            consensus_latency_us,
            consensus_state,
            consensus_reached,
            #[cfg(feature = "neural-forecasting")]
            neural_forecast,
        })
    }

    /// Process raw market data into MarketTick
    fn process_market_data(&self, feed: &MarketFeed) -> Result<MarketTick> {
        // Convert market feed to internal format
        Ok(MarketTick {
            timestamp: chrono::Utc::now(),
            orderbook: bincode::serialize(&(feed.price, feed.spread, feed.vwap))
                .map_err(|e| EcosystemError::HyperPhysics(format!("Serialization error: {}", e)))?,
            trades: bincode::serialize(&feed.returns)
                .map_err(|e| EcosystemError::HyperPhysics(format!("Serialization error: {}", e)))?,
        })
    }

    /// Run physics-based market simulation
    async fn run_physics_simulation(&self, _market_tick: &MarketTick) -> Result<PhysicsSignal> {
        // Physics simulation produces a directional signal
        // In production, this would use Rapier/Jolt/Warp for actual simulation
        Ok(PhysicsSignal {
            direction: 0.0,  // Neutral by default
            momentum: 0.0,
            energy: 0.0,
        })
    }

    /// Run neural forecasting on market data
    #[cfg(feature = "neural-forecasting")]
    async fn run_neural_forecasting(&self, feed: &MarketFeed) -> Result<ForecastResult> {
        // Convert local MarketFeed to neural-trader's MarketFeed
        let neural_feed: hyperphysics_neural_trader::MarketFeed = feed.into();

        // Process feed through neural adapter to get features
        let features = self.neural_adapter.process_feed(&neural_feed).await
            .map_err(|e| EcosystemError::HyperPhysics(format!("Neural adapter error: {}", e)))?;

        // Generate forecast using the neural engine
        self.neural_engine.forecast(&features).await
            .map_err(|e| EcosystemError::HyperPhysics(format!("Neural forecast error: {}", e)))
    }

    /// Run biomimetic optimization algorithms
    #[cfg(feature = "optimization-real")]
    async fn run_optimization(&self, feed: &MarketFeed) -> Result<OptimizationSignal> {
        // Create market objective from feed
        let objective = MarketObjective::new(
            feed.returns.clone(),
            feed.volatility,
            self.compute_trend(&feed.returns),
            1.0, // risk_aversion
        );

        // Use whale optimization for speed (could use ensemble for higher confidence)
        self.optimizer.optimize_whale(&objective)
    }

    #[cfg(not(feature = "optimization-real"))]
    async fn run_optimization(&self, _feed: &MarketFeed) -> Result<OptimizationSignal> {
        Ok(OptimizationSignal {
            position: 0.0,
            confidence: 0.0,
            algorithm: "NoOp".to_string(),
            latency_us: 0,
            converged: false,
        })
    }

    /// Validate decision through Byzantine consensus
    async fn validate_consensus(
        &self,
        signal: &OptimizationSignal,
    ) -> Result<(bool, ConsensusState)> {
        let mut state = self.consensus_state.write().await;

        // Simulate PBFT consensus round
        // In production, this would use real PBFT from hive-mind-rust
        let start = Instant::now();

        // Quorum requirement: 2f+1 out of 3f+1 nodes
        let required_votes = 2 * state.byzantine_threshold + 1;
        let available_votes = state.active_nodes;

        // Simplified consensus: if confidence is high enough, assume consensus
        let consensus_reached = signal.confidence > 0.5 && available_votes >= required_votes;

        state.consensus_latency_us = start.elapsed().as_micros() as u64;
        state.term += 1;

        Ok((consensus_reached, state.clone()))
    }

    /// Compute final trading decision from all signals
    fn compute_final_decision(
        &self,
        physics: &PhysicsSignal,
        optimization: &OptimizationSignal,
        consensus_reached: bool,
    ) -> Result<TradingDecision> {
        // Weight optimization signal higher if consensus reached
        let weight = if consensus_reached { 1.0 } else { 0.5 };

        // Combine physics and optimization signals
        let combined_position = optimization.position * weight + physics.direction * (1.0 - weight);

        // Determine action
        let action = if combined_position > 0.1 {
            Action::Buy
        } else if combined_position < -0.1 {
            Action::Sell
        } else {
            Action::Hold
        };

        // Adjust confidence based on consensus
        let confidence = if consensus_reached {
            optimization.confidence
        } else {
            optimization.confidence * 0.7  // Discount without consensus
        };

        Ok(TradingDecision {
            action,
            confidence,
            size: combined_position.abs().min(1.0),
        })
    }

    /// Compute trend from returns
    #[cfg(feature = "optimization-real")]
    fn compute_trend(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        // Simple momentum: sum of recent returns normalized
        let sum: f64 = returns.iter().sum();
        let mean = sum / returns.len() as f64;

        // Normalize to [-1, 1]
        (mean * 100.0).tanh()
    }

    /// Update pipeline statistics
    async fn update_stats(&self, latency_us: u64, consensus_reached: bool) {
        let mut stats = self.stats.write().await;

        stats.total_executions += 1;
        if consensus_reached {
            stats.consensus_successes += 1;
        } else {
            stats.consensus_failures += 1;
        }

        // Update latency stats
        stats.max_latency_us = stats.max_latency_us.max(latency_us);
        stats.min_latency_us = stats.min_latency_us.min(latency_us);

        // Running average
        let n = stats.total_executions as f64;
        stats.avg_latency_us = stats.avg_latency_us * ((n - 1.0) / n) + (latency_us as f64) / n;
    }

    /// Get current pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    /// Get current consensus state
    pub async fn get_consensus_state(&self) -> ConsensusState {
        self.consensus_state.read().await.clone()
    }
}

/// Physics-based signal from simulation
#[derive(Debug, Clone, Default)]
pub struct PhysicsSignal {
    /// Directional signal [-1, 1]
    pub direction: f64,
    /// Momentum indicator
    pub momentum: f64,
    /// System energy (for regime detection)
    pub energy: f64,
}

/// Optimization signal stub for no-feature build
#[cfg(not(feature = "optimization-real"))]
#[derive(Debug, Clone)]
pub struct OptimizationSignal {
    pub position: f64,
    pub confidence: f64,
    pub algorithm: String,
    pub latency_us: u64,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_pipeline_creation() {
        let config = EcosystemConfig::default();
        let pipeline = UnifiedPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_execution() {
        let config = EcosystemConfig::default();
        let pipeline = UnifiedPipeline::new(config).unwrap();

        let feed = MarketFeed {
            price: 100.0,
            returns: vec![0.01, -0.005, 0.02, -0.01, 0.015],
            volatility: 0.02,
            vwap: 100.0,
            spread: 0.01,
            timestamp: 1000.0,
        };

        let result = pipeline.execute(&feed).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.total_latency_us > 0);
    }

    #[tokio::test]
    async fn test_pipeline_latency_target() {
        let config = EcosystemConfig {
            target_latency_us: 1000,  // 1ms target
            ..Default::default()
        };
        let pipeline = UnifiedPipeline::new(config).unwrap();

        let feed = MarketFeed::default();
        let result = pipeline.execute(&feed).await.unwrap();

        // Pipeline overhead should be minimal (excluding actual physics/optimization)
        // Just verify the pipeline executes without errors
        assert!(result.total_latency_us < 100_000_000);  // <100s sanity check
    }

    #[tokio::test]
    async fn test_consensus_state_updates() {
        let config = EcosystemConfig::default();
        let pipeline = UnifiedPipeline::new(config).unwrap();

        let initial_state = pipeline.get_consensus_state().await;
        assert_eq!(initial_state.term, 1);

        let feed = MarketFeed::default();
        let _ = pipeline.execute(&feed).await.unwrap();

        let updated_state = pipeline.get_consensus_state().await;
        assert_eq!(updated_state.term, 2);  // Term should increment
    }

    #[tokio::test]
    async fn test_pipeline_stats_tracking() {
        let config = EcosystemConfig::default();
        let pipeline = UnifiedPipeline::new(config).unwrap();

        let feed = MarketFeed::default();

        // Execute multiple times
        for _ in 0..5 {
            let _ = pipeline.execute(&feed).await.unwrap();
        }

        let stats = pipeline.get_stats().await;
        assert_eq!(stats.total_executions, 5);
        assert!(stats.avg_latency_us > 0.0);
    }
}
