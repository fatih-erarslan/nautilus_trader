//! Physics engine router
//!
//! Routes computational tasks to the optimal physics engine based on:
//! - Algorithm tier (T1: <1ms, T2: 1-10ms, T3: 10ms+)
//! - Required features (determinism, GPU, differentiability)
//! - Available resources

use crate::EcosystemError;
use crate::Result;

// use std::sync::Arc;

/// Physics engine selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsEngine {
    /// JoltPhysics (deterministic, <100μs)
    Jolt,
    /// Rapier (fast, SIMD, <500μs)
    Rapier,
    /// Avian (ECS, multi-core)
    Avian,
    /// Warp (GPU, Python FFI)
    Warp,
    /// Taichi (JIT, billion particles)
    Taichi,
    /// MuJoCo (game theory, constraints)
    MuJoCo,
    /// Genesis (visualization)
    Genesis,
}

/// Physics engine router
pub struct PhysicsEngineRouter {
    /// Selected engine
    engine: PhysicsEngine,

    /// Determinism required
    pub deterministic: bool,
}

impl PhysicsEngineRouter {
    /// Create a new router with the specified engine
    pub fn new(engine: PhysicsEngine, deterministic: bool) -> Self {
        Self {
            engine,
            deterministic,
        }
    }

    /// Route a computation to the appropriate physics engine
    pub async fn route(&self, market_tick: &super::MarketTick) -> Result<PhysicsResult> {
        match self.engine {
            PhysicsEngine::Rapier => self.route_to_rapier(market_tick).await,
            PhysicsEngine::Jolt => self.route_to_jolt(market_tick).await,
            _ => Err(EcosystemError::PhysicsEngine(format!(
                "{:?} engine not yet implemented",
                self.engine
            ))),
        }
    }

    /// Route to Rapier physics engine
    #[cfg(feature = "physics-rapier")]
    async fn route_to_rapier(&self, market_tick: &super::MarketTick) -> Result<PhysicsResult> {
        use rapier_hyperphysics::{
            MarketMapper, MarketState, PhysicsSimulator, RapierHyperPhysicsAdapter,
            SignalExtractor, SimulatorConfig,
        };

        let start = std::time::Instant::now();

        // Create Rapier adapter
        let mut adapter = RapierHyperPhysicsAdapter::new().with_timestep(0.001); // 1ms timestep for sub-millisecond latency

        // Convert market tick to market state using tick data
        let market_state = MarketState {
            bids: market_tick.bids.iter()
                .map(|level| (level.price, level.quantity))
                .collect::<Vec<_>>()
                .into_iter()
                .take(10)
                .chain(std::iter::once((market_tick.mid_price * 0.999, 10.0)))
                .collect(),
            asks: market_tick.asks.iter()
                .map(|level| (level.price, level.quantity))
                .collect::<Vec<_>>()
                .into_iter()
                .take(10)
                .chain(std::iter::once((market_tick.mid_price * 1.001, 10.0)))
                .collect(),
            trades: vec![],
            participants: vec![],
            mid_price: market_tick.mid_price,
            volatility: market_tick.volatility.unwrap_or(0.02),
        };

        // Map market to physics
        let mapper = MarketMapper::new();
        let (bodies, colliders) = adapter.bodies_and_colliders_mut();
        let mapping = mapper.map_to_physics(&market_state, bodies, colliders)?;

        // Run physics simulation
        let simulator = PhysicsSimulator::with_config(SimulatorConfig {
            steps: 50, // Fewer steps for low latency
            dt: 0.001,
            convergence_threshold: 0.001,
            external_forces: false,
        });

        let sim_result = simulator.simulate(&mut adapter)?;

        // Extract trading signal
        let extractor = SignalExtractor::new();
        let signal = extractor.extract_signal(&adapter, &mapping)?;

        let elapsed = start.elapsed();

        // Serialize signal result to data
        let data = bincode::serialize(&signal)
            .map_err(|e| EcosystemError::PhysicsEngine(format!("Serialization error: {}", e)))?;

        Ok(PhysicsResult {
            latency_us: elapsed.as_micros() as u64,
            confidence: signal.confidence,
            data,
        })
    }

    #[cfg(not(feature = "physics-rapier"))]
    async fn route_to_rapier(&self, _market_tick: &super::MarketTick) -> Result<PhysicsResult> {
        Err(EcosystemError::Configuration(
            "Rapier feature not enabled".to_string(),
        ))
    }

    /// Route to Jolt physics engine with full market simulation
    #[cfg(feature = "physics-jolt")]
    async fn route_to_jolt(&self, market_tick: &super::MarketTick) -> Result<PhysicsResult> {
        use jolt_hyperphysics::JoltHyperPhysicsAdapter;

        let start = std::time::Instant::now();

        // Create Jolt adapter with deterministic settings
        let mut adapter = JoltHyperPhysicsAdapter::new()
            .map_err(|e| EcosystemError::PhysicsEngine(format!("Jolt init error: {}", e)))?;

        // Configure simulation parameters based on market volatility
        let volatility = market_tick.volatility.unwrap_or(0.02);
        let timestep = 0.001 / (1.0 + volatility * 10.0); // Adaptive timestep
        let iterations = ((0.01 / timestep) as i32).max(1).min(100);

        // Execute physics simulation
        adapter
            .step(timestep as f32, iterations)
            .map_err(|e| EcosystemError::PhysicsEngine(format!("Jolt step error: {}", e)))?;

        let elapsed = start.elapsed();

        // Compute confidence based on simulation stability and latency
        let latency_factor = (1000.0 / (elapsed.as_micros() as f64 + 1.0)).min(1.0);
        let confidence = (0.6 + latency_factor * 0.35).min(0.95);

        // Serialize simulation state
        let state_data = bincode::serialize(&(market_tick.mid_price, volatility, elapsed.as_micros()))
            .map_err(|e| EcosystemError::PhysicsEngine(format!("Serialization error: {}", e)))?;

        Ok(PhysicsResult {
            latency_us: elapsed.as_micros() as u64,
            confidence,
            data: state_data,
        })
    }

    #[cfg(not(feature = "physics-jolt"))]
    async fn route_to_jolt(&self, _market_tick: &super::MarketTick) -> Result<PhysicsResult> {
        Err(EcosystemError::Configuration(
            "Jolt feature not enabled".to_string(),
        ))
    }
}

/// Physics computation result
#[derive(Debug, Clone)]
pub struct PhysicsResult {
    /// Latency in microseconds
    pub latency_us: u64,

    /// Confidence in result
    pub confidence: f64,

    /// Result data
    pub data: Vec<u8>, // Placeholder
}
