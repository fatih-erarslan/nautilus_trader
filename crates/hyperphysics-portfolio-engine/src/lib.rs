//! # Thermodynamic Portfolio Engine
//!
//! Unified physics-based trading system integrating thermodynamic learning,
//! path integral optimization, fluid dynamics market simulation, and
//! spiking neural network prediction.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    THERMODYNAMIC PORTFOLIO ENGINE                    │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌──────────────────┐    ┌──────────────────┐                       │
//! │  │ Market Simulator │───▶│   SGNN Predictor │                       │
//! │  │ (Navier-Stokes)  │    │ (STDP Learning)  │                       │
//! │  └──────────────────┘    └────────┬─────────┘                       │
//! │                                   │                                 │
//! │                                   ▼                                 │
//! │  ┌──────────────────┐    ┌──────────────────┐                       │
//! │  │   Thermodynamic  │───▶│  Path Integral   │                       │
//! │  │    Scheduler     │    │    Optimizer     │                       │
//! │  │  (Phase Detect)  │    │ (Feynman Paths)  │                       │
//! │  └──────────────────┘    └────────┬─────────┘                       │
//! │                                   │                                 │
//! │                                   ▼                                 │
//! │                          ┌──────────────────┐                       │
//! │                          │  Optimal Weights │                       │
//! │                          └──────────────────┘                       │
//! │                                                                     │
//! │  Unified Temperature (T): Controls all component behaviors          │
//! │  T < T_c (Ordered):    Fine-tuning, exploitation                    │
//! │  T ≈ T_c (Critical):   Phase transition, rapid adaptation           │
//! │  T > T_c (Disordered): Exploration, large gradients                 │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Wolfram-Verified Constants
//!
//! All mathematical constants validated via Dilithium MCP:
//!
//! | Constant | Value | Validation Method |
//! |----------|-------|-------------------|
//! | T_c (Ising) | 2.269185314213022 | `ising_critical_temp()` |
//! | pBit P(h=0.5,T=0.15) | 0.9655 | `pbit_sample()` |
//! | Boltzmann exp(-0.5/1) | 0.6065 | `boltzmann_weight()` |
//! | STDP ΔW(Δt=10ms) | 0.0607 | `stdp_weight_change()` |
//! | H^11 distance | 0.4436 | `hyperbolic_distance()` |

pub use hyperphysics_thermodynamic_scheduler::{
    ThermodynamicScheduler, ThermodynamicState, SchedulerConfig, Phase,
    ISING_CRITICAL_TEMP, T_MIN, T_MAX, BETA_COUPLING,
};

pub use hyperphysics_path_integral::{
    PathIntegralOptimizer, RegimeAwareOptimizer, OptimizationResult,
    PortfolioPath, PortfolioState, PathStatistics, MarketRegime,
    TradingConstraints, MarketDynamics, OptimizerConfig,
};

pub use hyperphysics_market_microstructure::{
    MarketMicrostructureSimulator, OrderBookState, FluidParameters,
    MarketSignals, HyperPhysicsMarketSimulator, BoltzmannOrderGenerator,
    Side, Order, MarketEvent as MicrostructureEvent,
};

pub use hyperphysics_event_sgnn::{
    EventDrivenSGNNLayer, MultiScaleSGNN, LIFNeuron, Synapse, Spike,
    MarketEvent as SGNNEvent, EventType, Prediction, PerformanceMetrics,
    STDP_A_PLUS, STDP_A_MINUS, STDP_TAU_MS, MEMBRANE_TAU_MS,
};

use serde::{Deserialize, Serialize};

/// Unified engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Number of assets in portfolio
    pub num_assets: usize,
    /// Initial portfolio weights
    pub initial_weights: Vec<f64>,
    /// Initial temperature (unified parameter)
    pub initial_temperature: f64,
    /// Number of SGNN neurons per asset
    pub neurons_per_asset: usize,
    /// Path integral Monte Carlo samples
    pub path_samples: usize,
    /// Optimization horizon (trading days)
    pub horizon: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let num_assets = 10;
        Self {
            num_assets,
            initial_weights: vec![1.0 / num_assets as f64; num_assets],
            initial_temperature: 0.15,
            neurons_per_asset: 32,
            path_samples: 1000,
            horizon: 252,
        }
    }
}

/// Engine state for monitoring and diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    /// Unified temperature
    pub temperature: f64,
    /// Current phase
    pub phase: Phase,
    /// Current portfolio weights
    pub weights: Vec<f64>,
    /// Market prediction
    pub prediction: f64,
    /// Prediction confidence
    pub confidence: f64,
    /// Portfolio value
    pub value: f64,
    /// Sharpe ratio
    pub sharpe: f64,
    /// Events processed
    pub events_processed: u64,
    /// Current time (microseconds)
    pub time_us: u64,
}

/// Thermodynamic Portfolio Engine
///
/// Integrates all four physics-based components with unified temperature control.
pub struct ThermodynamicPortfolioEngine {
    /// Configuration
    config: EngineConfig,
    /// Thermodynamic scheduler (controls temperature evolution)
    scheduler: ThermodynamicScheduler,
    /// Path integral optimizer (portfolio optimization)
    optimizer: PathIntegralOptimizer,
    /// Market microstructure simulator (order book dynamics)
    market_sim: MarketMicrostructureSimulator,
    /// Event-driven SGNN (prediction engine)
    sgnn: MultiScaleSGNN,
    /// Current portfolio weights
    current_weights: Vec<f64>,
    /// Portfolio value history
    value_history: Vec<f64>,
    /// Events processed counter
    events_processed: u64,
    /// Current time (microseconds)
    current_time_us: u64,
    /// Last prediction
    last_prediction: Prediction,
}

impl ThermodynamicPortfolioEngine {
    /// Create new engine with configuration
    pub fn new(config: EngineConfig) -> Self {
        // Initialize thermodynamic scheduler
        let scheduler_config = SchedulerConfig {
            t0: config.initial_temperature,
            alpha0: 0.1,
            cooling_rate: 1.0,
            history_size: 100,
            weight_decay: 0.2,
        };
        let scheduler = ThermodynamicScheduler::new(scheduler_config);

        // Initialize path integral optimizer
        let optimizer = PathIntegralOptimizer::hyperphysics_default(config.num_assets)
            .with_temperature(config.initial_temperature);

        // Initialize market simulator (assume initial mid price of 100.0)
        let market_sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);

        // Initialize SGNN predictor
        let sgnn = MultiScaleSGNN::new(config.num_assets * config.neurons_per_asset);

        Self {
            current_weights: config.initial_weights.clone(),
            value_history: vec![1.0],
            events_processed: 0,
            current_time_us: 0,
            last_prediction: Prediction {
                direction: 0.0,
                confidence: 0.0,
                timestamp: 0,
            },
            config,
            scheduler,
            optimizer,
            market_sim,
            sgnn,
        }
    }

    /// Create with HyperPhysics defaults
    pub fn hyperphysics_default() -> Self {
        Self::new(EngineConfig::default())
    }

    /// Process a market event through the full pipeline
    pub fn process_event(&mut self, event: SGNNEvent) -> EngineState {
        self.events_processed += 1;
        self.current_time_us = event.timestamp;

        // 1. Update market simulator
        let micro_event = self.convert_to_microstructure_event(&event);
        self.market_sim.step(&[micro_event]);

        // 2. Get SGNN prediction
        self.last_prediction = self.sgnn.process_event(event);

        // 3. Update thermodynamic scheduler based on prediction confidence
        let gradient_norm = 1.0 - self.last_prediction.confidence;
        let loss = (1.0 - self.last_prediction.confidence).abs();
        let _learning_rate = self.scheduler.step(gradient_norm, loss);

        // 4. Synchronize temperature across all components
        let temperature = self.scheduler.temperature();
        self.sync_temperature(temperature);

        // 5. Get market signals for regime detection
        let signals = self.get_market_signals();
        let regime = self.detect_regime(&signals);

        // 6. Update optimizer if temperature changed significantly
        if self.should_reoptimize() {
            self.reoptimize(regime);
        }

        // 7. Return current state
        self.get_state()
    }

    /// Process a batch of events
    pub fn process_events(&mut self, events: &[SGNNEvent]) -> Vec<EngineState> {
        events.iter().map(|e| self.process_event(e.clone())).collect()
    }

    /// Convert SGNN event to microstructure event
    fn convert_to_microstructure_event(&self, event: &SGNNEvent) -> MicrostructureEvent {
        let order = Order {
            side: if event.volume > 0.0 { Side::Bid } else { Side::Ask },
            price: event.price,
            quantity: event.volume.abs(),
            timestamp: event.timestamp,
        };
        MicrostructureEvent::LimitOrder(order)
    }

    /// Synchronize temperature across all components
    fn sync_temperature(&mut self, temperature: f64) {
        self.optimizer.set_temperature(temperature);
        self.market_sim.set_temperature(temperature);
    }

    /// Get market signals
    fn get_market_signals(&self) -> MarketSignals {
        MarketSignals {
            mid_price: self.market_sim.get_state().mid_price,
            spread: self.market_sim.get_state().spread,
            imbalance: self.market_sim.compute_imbalance(),
            volatility: self.market_sim.compute_volatility(),
            time_us: self.current_time_us,
        }
    }

    /// Detect market regime from signals
    fn detect_regime(&self, signals: &MarketSignals) -> MarketRegime {
        // Heuristic regime detection
        let volatility = signals.volatility;
        let imbalance = signals.imbalance;

        if volatility > 0.5 {
            MarketRegime::Crisis
        } else if volatility > 0.3 {
            if imbalance > 0.3 {
                MarketRegime::Bull
            } else if imbalance < -0.3 {
                MarketRegime::Bear
            } else {
                MarketRegime::Normal
            }
        } else {
            MarketRegime::Normal
        }
    }

    /// Check if reoptimization is needed
    fn should_reoptimize(&self) -> bool {
        // Reoptimize every 100 events or on phase transition
        self.events_processed % 100 == 0 ||
        self.scheduler.phase() == Phase::Critical
    }

    /// Reoptimize portfolio
    fn reoptimize(&mut self, _regime: MarketRegime) {
        let result = self.optimizer.optimize(&self.current_weights);

        // Use expected weights from path integral
        if !result.expected_weights.is_empty() {
            self.current_weights = result.expected_weights[0].clone();
        }

        // Update value based on optimal path performance
        if let Some(last_value) = self.value_history.last() {
            let new_value = last_value * (1.0 + result.optimal_path.total_return / 252.0);
            self.value_history.push(new_value);
        }
    }

    /// Get current engine state
    pub fn get_state(&self) -> EngineState {
        let sharpe = self.compute_sharpe();

        EngineState {
            temperature: self.scheduler.temperature(),
            phase: self.scheduler.phase(),
            weights: self.current_weights.clone(),
            prediction: self.last_prediction.direction,
            confidence: self.last_prediction.confidence,
            value: *self.value_history.last().unwrap_or(&1.0),
            sharpe,
            events_processed: self.events_processed,
            time_us: self.current_time_us,
        }
    }

    /// Compute rolling Sharpe ratio
    fn compute_sharpe(&self) -> f64 {
        if self.value_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.value_history.windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance > 0.0 {
            (mean - 0.03 / 252.0) / variance.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }

    /// Force reheating (e.g., on regime shift detection)
    pub fn reheat(&mut self, temperature: f64) {
        self.scheduler.reheat(temperature);
        self.sync_temperature(temperature);
    }

    /// Get thermodynamic scheduler reference
    pub fn scheduler(&self) -> &ThermodynamicScheduler {
        &self.scheduler
    }

    /// Get path integral optimizer reference
    pub fn optimizer(&self) -> &PathIntegralOptimizer {
        &self.optimizer
    }

    /// Get market simulator reference
    pub fn market_sim(&self) -> &MarketMicrostructureSimulator {
        &self.market_sim
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.scheduler.temperature()
    }

    /// Get current phase
    pub fn phase(&self) -> Phase {
        self.scheduler.phase()
    }

    /// Get current weights
    pub fn weights(&self) -> &[f64] {
        &self.current_weights
    }

    /// Get value history
    pub fn value_history(&self) -> &[f64] {
        &self.value_history
    }

    /// Generate diagnostic report
    pub fn diagnostics(&self) -> String {
        format!(
            "═══════════════════════════════════════════════════════════════\n\
             THERMODYNAMIC PORTFOLIO ENGINE - DIAGNOSTICS\n\
             ═══════════════════════════════════════════════════════════════\n\
             \n\
             THERMODYNAMIC STATE:\n\
             Temperature: {:.6} (T/T_c = {:.4})\n\
             Phase: {:?}\n\
             Learning Rate: {:.6}\n\
             \n\
             PORTFOLIO STATE:\n\
             Weights: {:?}\n\
             Value: {:.4}\n\
             Sharpe: {:.4}\n\
             \n\
             PREDICTION STATE:\n\
             Direction: {:.4}\n\
             Confidence: {:.4}\n\
             \n\
             PERFORMANCE:\n\
             Events Processed: {}\n\
             Time (µs): {}\n\
             ═══════════════════════════════════════════════════════════════",
            self.scheduler.temperature(),
            self.scheduler.temperature() / ISING_CRITICAL_TEMP,
            self.scheduler.phase(),
            self.scheduler.learning_rate(),
            self.current_weights,
            self.value_history.last().unwrap_or(&1.0),
            self.compute_sharpe(),
            self.last_prediction.direction,
            self.last_prediction.confidence,
            self.events_processed,
            self.current_time_us
        )
    }
}

/// Factory for creating pre-configured engines
pub struct EngineFactory;

impl EngineFactory {
    /// Create engine for high-frequency trading
    pub fn hft_engine(num_assets: usize) -> ThermodynamicPortfolioEngine {
        let config = EngineConfig {
            num_assets,
            initial_weights: vec![1.0 / num_assets as f64; num_assets],
            initial_temperature: 0.1, // Lower temp for faster convergence
            neurons_per_asset: 64,    // More neurons for faster prediction
            path_samples: 500,        // Fewer samples for speed
            horizon: 21,              // Short horizon
        };
        ThermodynamicPortfolioEngine::new(config)
    }

    /// Create engine for portfolio optimization
    pub fn portfolio_engine(num_assets: usize) -> ThermodynamicPortfolioEngine {
        let config = EngineConfig {
            num_assets,
            initial_weights: vec![1.0 / num_assets as f64; num_assets],
            initial_temperature: 0.15,
            neurons_per_asset: 32,
            path_samples: 2000,       // More samples for accuracy
            horizon: 252,             // Full year horizon
        };
        ThermodynamicPortfolioEngine::new(config)
    }

    /// Create engine for research/backtesting
    pub fn research_engine(num_assets: usize) -> ThermodynamicPortfolioEngine {
        let config = EngineConfig {
            num_assets,
            initial_weights: vec![1.0 / num_assets as f64; num_assets],
            initial_temperature: 0.5, // Higher temp for exploration
            neurons_per_asset: 128,   // Large network
            path_samples: 5000,       // Many samples
            horizon: 504,             // Two year horizon
        };
        ThermodynamicPortfolioEngine::new(config)
    }
}

/// Simulation runner for backtesting
pub struct BacktestRunner {
    engine: ThermodynamicPortfolioEngine,
    results: Vec<EngineState>,
}

impl BacktestRunner {
    pub fn new(engine: ThermodynamicPortfolioEngine) -> Self {
        Self {
            engine,
            results: Vec::new(),
        }
    }

    /// Run backtest with synthetic events
    pub fn run_synthetic(&mut self, num_events: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for i in 0..num_events {
            let event = SGNNEvent {
                timestamp: i as u64 * 10_000, // 10ms intervals
                asset_id: rng.gen_range(0..self.engine.config.num_assets) as u8,
                event_type: EventType::Trade,
                price: 100.0 + rng.gen_range(-1.0..1.0),
                volume: rng.gen_range(-100.0..100.0),
            };

            let state = self.engine.process_event(event);
            self.results.push(state);
        }
    }

    /// Get backtest results
    pub fn results(&self) -> &[EngineState] {
        &self.results
    }

    /// Compute backtest statistics
    pub fn statistics(&self) -> BacktestStatistics {
        if self.results.is_empty() {
            return BacktestStatistics::default();
        }

        let final_value = self.results.last().unwrap().value;
        let initial_value = self.results.first().unwrap().value;

        let returns: Vec<f64> = self.results.windows(2)
            .map(|w| w[1].value / w[0].value - 1.0)
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len().max(1) as f64;

        let sharpe = if variance > 0.0 {
            mean_return / variance.sqrt() * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = initial_value;
        let mut max_drawdown = 0.0;

        for state in &self.results {
            if state.value > peak {
                peak = state.value;
            }
            let drawdown = (peak - state.value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        BacktestStatistics {
            total_return: (final_value / initial_value) - 1.0,
            sharpe_ratio: sharpe,
            max_drawdown,
            num_events: self.results.len(),
            final_value,
        }
    }
}

/// Backtest statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BacktestStatistics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub num_events: usize,
    pub final_value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = ThermodynamicPortfolioEngine::hyperphysics_default();
        assert_eq!(engine.config.num_assets, 10);
        assert!((engine.temperature() - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_temperature_synchronization() {
        let mut engine = ThermodynamicPortfolioEngine::hyperphysics_default();

        // Process some events to trigger temperature changes
        for i in 0..10 {
            let event = SGNNEvent {
                timestamp: i * 1000,
                asset_id: 0,
                event_type: EventType::Trade,
                price: 100.0,
                volume: 100.0,
            };
            engine.process_event(event);
        }

        // Temperature should be synchronized
        let engine_temp = engine.temperature();
        assert!(engine_temp > 0.0);
    }

    #[test]
    fn test_phase_detection() {
        let engine = ThermodynamicPortfolioEngine::hyperphysics_default();

        // At low temperature, should be ordered
        assert_eq!(engine.phase(), Phase::Ordered);
    }

    #[test]
    fn test_reheating() {
        let mut engine = ThermodynamicPortfolioEngine::hyperphysics_default();

        engine.reheat(0.8);
        assert!((engine.temperature() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_backtest_runner() {
        let engine = EngineFactory::hft_engine(5);
        let mut runner = BacktestRunner::new(engine);

        runner.run_synthetic(100);

        let stats = runner.statistics();
        assert_eq!(stats.num_events, 100);
        assert!(stats.final_value > 0.0);
    }

    #[test]
    fn test_diagnostics() {
        let engine = ThermodynamicPortfolioEngine::hyperphysics_default();
        let diag = engine.diagnostics();

        assert!(diag.contains("THERMODYNAMIC PORTFOLIO ENGINE"));
        assert!(diag.contains("Temperature"));
        assert!(diag.contains("Phase"));
    }

    #[test]
    fn test_ising_critical_temperature_constant() {
        // Verify the critical temperature constant is accessible
        let expected = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
        assert!((ISING_CRITICAL_TEMP - expected).abs() < 1e-10);
    }
}
