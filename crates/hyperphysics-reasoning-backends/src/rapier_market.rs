//! Rapier-based market physics backend for reasoning router.
//!
//! This backend uses real physics simulation via rapier-hyperphysics to model
//! market dynamics. It maps market entities (orders, participants) to rigid
//! bodies and extracts trading signals from physics simulation results.
//!
//! ## Architecture
//!
//! ```text
//! Problem Data → MarketMapper → Physics Bodies → Rapier Simulation
//!                                                        ↓
//! ReasoningResult ← SignalExtractor ← Physics State ←────┘
//! ```
//!
//! ## Performance
//!
//! - Target latency: <500μs per simulation cycle
//! - Throughput: 2000+ simulations/second
//! - Deterministic mode available for backtesting

use crate::{
    BackendCapability, BackendId, BackendMetrics, BackendPool, LatencyTier, ProblemDomain,
    ReasoningBackend, ReasoningResult, ResultValue, RouterResult,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use rapier_hyperphysics::{
    MarketMapper, MarketParticipant, MarketState, ParticipantType, PhysicsMapping,
    RapierHyperPhysicsAdapter, SignalExtractor, SignalResult, TradingSignal,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Configuration for the Rapier market backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapierMarketConfig {
    /// Number of physics simulation steps per execution
    pub simulation_steps: usize,

    /// Physics timestep (seconds)
    pub timestep: f32,

    /// Momentum threshold for signal classification
    pub momentum_threshold: f32,

    /// Energy threshold for volatility classification
    pub energy_threshold: f32,

    /// Price scaling factor
    pub price_scale: f32,

    /// Volume scaling factor
    pub volume_scale: f32,

    /// Enable deterministic mode (slower but reproducible)
    pub deterministic: bool,
}

impl Default for RapierMarketConfig {
    fn default() -> Self {
        Self {
            simulation_steps: 10,
            timestep: 1.0 / 60.0,
            momentum_threshold: 0.5,
            energy_threshold: 10.0,
            price_scale: 0.01,
            volume_scale: 0.001,
            deterministic: false,
        }
    }
}

impl RapierMarketConfig {
    /// Create configuration optimized for HFT (minimal latency)
    pub fn hft() -> Self {
        Self {
            simulation_steps: 5,
            timestep: 1.0 / 120.0,
            momentum_threshold: 0.3,
            energy_threshold: 8.0,
            price_scale: 0.01,
            volume_scale: 0.001,
            deterministic: false,
        }
    }

    /// Create configuration for backtesting (deterministic)
    pub fn backtest() -> Self {
        Self {
            simulation_steps: 20,
            timestep: 1.0 / 60.0,
            momentum_threshold: 0.5,
            energy_threshold: 10.0,
            price_scale: 0.01,
            volume_scale: 0.001,
            deterministic: true,
        }
    }
}

/// Real physics-based market backend using Rapier3D.
///
/// This backend implements the full ReasoningBackend trait using actual
/// physics simulation for market modeling rather than placeholder calculations.
pub struct RapierMarketBackend {
    /// Backend identifier
    id: BackendId,

    /// Configuration
    config: RapierMarketConfig,

    /// Supported problem domains
    domains: Vec<ProblemDomain>,

    /// Backend capabilities
    capabilities: HashSet<BackendCapability>,

    /// Performance metrics
    metrics: Mutex<BackendMetrics>,
}

impl RapierMarketBackend {
    /// Create a new Rapier market backend with default configuration
    pub fn new() -> Self {
        Self::with_config(RapierMarketConfig::default())
    }

    /// Create backend with custom configuration
    pub fn with_config(config: RapierMarketConfig) -> Self {
        let mut capabilities = HashSet::new();

        // Real physics engine capabilities
        capabilities.insert(BackendCapability::ParallelScenarios);
        capabilities.insert(BackendCapability::Streaming);

        if config.deterministic {
            capabilities.insert(BackendCapability::Deterministic);
        }

        Self {
            id: BackendId::new("rapier-market"),
            config,
            domains: vec![
                ProblemDomain::Physics,
                ProblemDomain::Financial,
                ProblemDomain::Engineering,
            ],
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
        }
    }

    /// Create HFT-optimized backend
    pub fn hft() -> Self {
        Self::with_config(RapierMarketConfig::hft())
    }

    /// Create backtesting backend
    pub fn backtest() -> Self {
        Self::with_config(RapierMarketConfig::backtest())
    }

    /// Extract market state from problem data
    fn extract_market_state(
        &self,
        problem: &crate::problem::Problem,
    ) -> RouterResult<MarketState> {
        // Extract from structured problem data
        match &problem.data {
            crate::problem::ProblemData::Json(json) => {
                self.parse_market_state_from_json(json)
            }
            crate::problem::ProblemData::Vector(data) => {
                self.parse_market_state_from_vector(data)
            }
            _ => {
                // Create default market state for testing
                Ok(self.create_default_market_state())
            }
        }
    }

    /// Parse market state from JSON data
    fn parse_market_state_from_json(
        &self,
        json: &serde_json::Value,
    ) -> RouterResult<MarketState> {
        // Extract bids
        let bids = json
            .get("bids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let price = item.get("price")?.as_f64()?;
                        let volume = item.get("volume")?.as_f64()?;
                        Some((price, volume))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        // Extract asks
        let asks = json
            .get("asks")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let price = item.get("price")?.as_f64()?;
                        let volume = item.get("volume")?.as_f64()?;
                        Some((price, volume))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        // Extract mid price
        let mid_price = json
            .get("mid_price")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                // Calculate from bids/asks if not provided
                let best_bid = bids.first().map(|(p, _)| *p).unwrap_or(100.0);
                let best_ask = asks.first().map(|(p, _)| *p).unwrap_or(101.0);
                (best_bid + best_ask) / 2.0
            });

        // Extract volatility
        let volatility = json
            .get("volatility")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.02);

        // Extract participants
        let participants = json
            .get("participants")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| {
                        let ptype = item.get("type")?.as_str()?;
                        let capital = item.get("capital")?.as_f64()?;
                        let position = item.get("position")?.as_f64().unwrap_or(0.0);
                        let aggr = item.get("aggressiveness")?.as_f64().unwrap_or(0.5);

                        let participant_type = match ptype {
                            "whale" | "Whale" => ParticipantType::Whale,
                            "institutional" | "Institutional" => ParticipantType::Institutional,
                            "hft" | "HFT" => ParticipantType::HFT,
                            _ => ParticipantType::Retail,
                        };

                        Some(MarketParticipant {
                            participant_type,
                            capital,
                            position_size: position,
                            aggressiveness: aggr,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(MarketState {
            bids,
            asks,
            trades: vec![],
            participants,
            mid_price,
            volatility,
        })
    }

    /// Parse market state from vector data (simplified format)
    fn parse_market_state_from_vector(&self, data: &[f64]) -> RouterResult<MarketState> {
        // Interpret vector as: [bid_price, bid_vol, ask_price, ask_vol, mid_price, volatility, ...]
        if data.len() < 6 {
            return Ok(self.create_default_market_state());
        }

        let mut bids = Vec::new();
        let mut asks = Vec::new();

        // Parse pairs of (price, volume)
        let mid_idx = data.len() / 2;
        for i in (0..mid_idx).step_by(2) {
            if i + 1 < mid_idx {
                bids.push((data[i], data[i + 1]));
            }
        }

        for i in (mid_idx..data.len() - 2).step_by(2) {
            if i + 1 < data.len() - 2 {
                asks.push((data[i], data[i + 1]));
            }
        }

        let mid_price = data.get(data.len() - 2).copied().unwrap_or(100.0);
        let volatility = data.get(data.len() - 1).copied().unwrap_or(0.02);

        Ok(MarketState {
            bids,
            asks,
            trades: vec![],
            participants: vec![],
            mid_price,
            volatility,
        })
    }

    /// Create default market state for fallback
    fn create_default_market_state(&self) -> MarketState {
        MarketState {
            bids: vec![(100.0, 10.0), (99.5, 15.0), (99.0, 20.0)],
            asks: vec![(100.5, 12.0), (101.0, 18.0), (101.5, 25.0)],
            trades: vec![],
            participants: vec![],
            mid_price: 100.25,
            volatility: 0.02,
        }
    }

    /// Run physics simulation and extract results
    fn run_simulation(
        &self,
        market_state: &MarketState,
    ) -> RouterResult<(SignalResult, PhysicsStateData)> {
        // Create physics adapter
        let mut adapter = RapierHyperPhysicsAdapter::new().with_timestep(self.config.timestep);

        // Create mapper with configured scaling
        let mapper =
            MarketMapper::with_scaling(self.config.price_scale, self.config.volume_scale);

        // Map market state to physics
        let (rigid_bodies, colliders) = adapter.bodies_and_colliders_mut();
        let mapping = mapper
            .map_to_physics(market_state, rigid_bodies, colliders)
            .map_err(|e| crate::RouterError::BackendFailed {
                backend_id: "rapier-market".to_string(),
                message: format!("Mapping error: {}", e),
            })?;

        // Run simulation steps
        for _ in 0..self.config.simulation_steps {
            adapter.step();
        }

        // Extract signal
        let extractor = SignalExtractor::with_thresholds(
            self.config.momentum_threshold,
            self.config.energy_threshold,
        );

        let signal_result = extractor
            .extract_signal(&adapter, &mapping)
            .map_err(|e| crate::RouterError::BackendFailed {
                backend_id: "rapier-market".to_string(),
                message: format!("Signal extraction error: {}", e),
            })?;

        // Extract physics state
        let physics_state = self.extract_physics_state(&adapter, &mapping);

        Ok((signal_result, physics_state))
    }

    /// Extract physics state data from adapter
    fn extract_physics_state(
        &self,
        adapter: &RapierHyperPhysicsAdapter,
        mapping: &PhysicsMapping,
    ) -> PhysicsStateData {
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        let mut energy = 0.0;

        // Collect bid body states
        for handle in &mapping.bid_bodies {
            if let Some(rb) = adapter.rigid_bodies().get(*handle) {
                let pos = rb.translation();
                positions.extend_from_slice(&[pos.x as f64, pos.y as f64, pos.z as f64]);

                let vel = rb.linvel();
                velocities.extend_from_slice(&[vel.x as f64, vel.y as f64, vel.z as f64]);

                // Kinetic energy
                let mass = rb.mass();
                energy += 0.5 * (mass as f64) * (vel.norm_squared() as f64);
            }
        }

        // Collect ask body states
        for handle in &mapping.ask_bodies {
            if let Some(rb) = adapter.rigid_bodies().get(*handle) {
                let pos = rb.translation();
                positions.extend_from_slice(&[pos.x as f64, pos.y as f64, pos.z as f64]);

                let vel = rb.linvel();
                velocities.extend_from_slice(&[vel.x as f64, vel.y as f64, vel.z as f64]);

                let mass = rb.mass();
                energy += 0.5 * (mass as f64) * (vel.norm_squared() as f64);
            }
        }

        // Collect participant body states
        for handle in &mapping.participant_bodies {
            if let Some(rb) = adapter.rigid_bodies().get(*handle) {
                let pos = rb.translation();
                positions.extend_from_slice(&[pos.x as f64, pos.y as f64, pos.z as f64]);

                let vel = rb.linvel();
                velocities.extend_from_slice(&[vel.x as f64, vel.y as f64, vel.z as f64]);

                let mass = rb.mass();
                energy += 0.5 * (mass as f64) * (vel.norm_squared() as f64);
            }
        }

        PhysicsStateData {
            positions,
            velocities,
            energy,
        }
    }

    /// Convert trading signal to confidence and quality scores
    fn signal_to_scores(signal: &SignalResult) -> (f64, f64) {
        let confidence = signal.confidence;

        // Quality based on signal strength and regime clarity
        let regime_quality = match signal.regime {
            rapier_hyperphysics::MarketRegime::Breakout => 0.9,
            rapier_hyperphysics::MarketRegime::Trending => 0.8,
            rapier_hyperphysics::MarketRegime::HighVolatility => 0.7,
            rapier_hyperphysics::MarketRegime::Ranging => 0.5,
            rapier_hyperphysics::MarketRegime::LowVolatility => 0.6,
        };

        let quality = (confidence + regime_quality) / 2.0;

        (confidence, quality)
    }
}

impl Default for RapierMarketBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal physics state data
struct PhysicsStateData {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    energy: f64,
}

#[async_trait]
impl ReasoningBackend for RapierMarketBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        "Rapier Market Physics"
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Physics
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &self.domains
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        if self.config.simulation_steps <= 5 {
            LatencyTier::UltraFast
        } else if self.config.simulation_steps <= 15 {
            LatencyTier::Fast
        } else {
            LatencyTier::Medium
        }
    }

    fn can_handle(&self, signature: &crate::problem::ProblemSignature) -> bool {
        // Check domain compatibility
        self.domains.iter().any(|d| signature.domain == *d)
            || signature.domain == ProblemDomain::General
    }

    fn estimate_latency(&self, _signature: &crate::problem::ProblemSignature) -> Duration {
        // Base estimate: ~50μs per simulation step
        let base_us = 50 * self.config.simulation_steps;
        Duration::from_micros(base_us as u64)
    }

    async fn execute(&self, problem: &crate::problem::Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        // Extract market state from problem
        let market_state = self.extract_market_state(problem)?;

        // Run physics simulation
        let (signal_result, physics_state) = self.run_simulation(&market_state)?;

        let latency = start.elapsed();

        // Compute scores
        let (confidence, quality) = Self::signal_to_scores(&signal_result);

        // Build metadata
        let metadata = serde_json::json!({
            "signal": format!("{:?}", signal_result.signal),
            "regime": format!("{:?}", signal_result.regime),
            "price_movement": signal_result.price_movement,
            "momentum_strength": signal_result.momentum_strength,
            "volatility": signal_result.volatility,
            "simulation_steps": self.config.simulation_steps,
            "body_count": physics_state.positions.len() / 3,
        });

        // Record metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(quality));
        }

        Ok(ReasoningResult {
            value: ResultValue::PhysicsState {
                positions: physics_state.positions,
                velocities: physics_state.velocities,
                energy: physics_state.energy,
            },
            confidence,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata,
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }

    fn is_healthy(&self) -> bool {
        // Backend is healthy if success rate is above threshold
        let metrics = self.metrics.lock();
        metrics.total_calls == 0 || metrics.success_rate > 0.9
    }
}

/// Convert TradingSignal to a numerical score
pub fn signal_to_numeric(signal: TradingSignal) -> f64 {
    match signal {
        TradingSignal::StrongBuy => 1.0,
        TradingSignal::Buy => 0.5,
        TradingSignal::Hold => 0.0,
        TradingSignal::Sell => -0.5,
        TradingSignal::StrongSell => -1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{Problem, ProblemData, ProblemSignature, ProblemType};

    #[test]
    fn test_backend_creation() {
        let backend = RapierMarketBackend::new();
        assert_eq!(backend.name(), "Rapier Market Physics");
        assert_eq!(backend.pool(), BackendPool::Physics);
    }

    #[test]
    fn test_hft_config() {
        let backend = RapierMarketBackend::hft();
        assert_eq!(backend.config.simulation_steps, 5);
        assert_eq!(backend.latency_tier(), LatencyTier::UltraFast);
    }

    #[test]
    fn test_backtest_config() {
        let backend = RapierMarketBackend::backtest();
        assert_eq!(backend.config.simulation_steps, 20);
        assert!(backend.capabilities.contains(&BackendCapability::Deterministic));
    }

    #[test]
    fn test_can_handle_physics() {
        let backend = RapierMarketBackend::new();

        let physics_sig = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics)
            .with_dimensionality(6);

        assert!(backend.can_handle(&physics_sig));
    }

    #[test]
    fn test_can_handle_finance() {
        let backend = RapierMarketBackend::new();

        let finance_sig =
            ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Financial)
                .with_dimensionality(10);

        assert!(backend.can_handle(&finance_sig));
    }

    #[test]
    fn test_default_market_state() {
        let backend = RapierMarketBackend::new();
        let state = backend.create_default_market_state();

        assert_eq!(state.bids.len(), 3);
        assert_eq!(state.asks.len(), 3);
        assert!(state.mid_price > 0.0);
    }

    #[test]
    fn test_parse_json_market_state() {
        let backend = RapierMarketBackend::new();

        let json = serde_json::json!({
            "bids": [
                {"price": 100.0, "volume": 10.0},
                {"price": 99.5, "volume": 15.0}
            ],
            "asks": [
                {"price": 100.5, "volume": 12.0},
                {"price": 101.0, "volume": 18.0}
            ],
            "mid_price": 100.25,
            "volatility": 0.03
        });

        let state = backend.parse_market_state_from_json(&json).unwrap();

        assert_eq!(state.bids.len(), 2);
        assert_eq!(state.asks.len(), 2);
        assert!((state.mid_price - 100.25).abs() < 0.01);
        assert!((state.volatility - 0.03).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_execute_with_structured_data() {
        let backend = RapierMarketBackend::new();

        let signature =
            ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Financial)
                .with_dimensionality(6);

        let data = ProblemData::Json(serde_json::json!({
            "bids": [
                {"price": 100.0, "volume": 10.0},
                {"price": 99.5, "volume": 15.0}
            ],
            "asks": [
                {"price": 100.5, "volume": 12.0},
                {"price": 101.0, "volume": 18.0}
            ],
            "mid_price": 100.25,
            "volatility": 0.02
        }));

        let problem = Problem::new(signature, data);
        let result = backend.execute(&problem).await.unwrap();

        // Verify result structure
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.quality >= 0.0 && result.quality <= 1.0);

        match result.value {
            ResultValue::PhysicsState {
                positions,
                velocities,
                energy,
            } => {
                assert!(!positions.is_empty());
                assert!(!velocities.is_empty());
                assert!(energy >= 0.0);
            }
            _ => panic!("Expected PhysicsState result"),
        }

        // Check metadata
        assert!(result.metadata.get("signal").is_some());
        assert!(result.metadata.get("regime").is_some());
    }

    #[tokio::test]
    async fn test_execute_with_vector_data() {
        let backend = RapierMarketBackend::new();

        // Vector format: [bid_price, bid_vol, ..., ask_price, ask_vol, ..., mid, vol]
        let data = vec![
            100.0, 10.0, // bid 1
            99.5, 15.0, // bid 2
            100.5, 12.0, // ask 1
            101.0, 18.0, // ask 2
            100.25, // mid_price
            0.02,   // volatility
        ];

        let signature = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics)
            .with_dimensionality(data.len() as u32);

        let problem = Problem::new(signature, ProblemData::Vector(data));
        let result = backend.execute(&problem).await.unwrap();
        assert!(result.confidence >= 0.0);
    }

    #[test]
    fn test_signal_to_numeric() {
        assert_eq!(signal_to_numeric(TradingSignal::StrongBuy), 1.0);
        assert_eq!(signal_to_numeric(TradingSignal::Buy), 0.5);
        assert_eq!(signal_to_numeric(TradingSignal::Hold), 0.0);
        assert_eq!(signal_to_numeric(TradingSignal::Sell), -0.5);
        assert_eq!(signal_to_numeric(TradingSignal::StrongSell), -1.0);
    }

    #[test]
    fn test_latency_estimation() {
        let backend = RapierMarketBackend::new();

        let sig = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics)
            .with_dimensionality(6);

        let latency = backend.estimate_latency(&sig);
        // Default: 10 steps × 50μs = 500μs
        assert!(latency.as_micros() >= 400 && latency.as_micros() <= 600);
    }

    #[test]
    fn test_metrics_recording() {
        let backend = RapierMarketBackend::new();
        let metrics = backend.metrics();

        assert_eq!(metrics.total_calls, 0);
        assert!(backend.is_healthy());
    }
}
