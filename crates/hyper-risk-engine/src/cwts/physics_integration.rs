//! Physics Engine Integration for CWTS Risk Management
//!
//! Maps market entities to physics simulation for dynamic risk analysis using
//! the Rapier physics engine.
//!
//! ## Market-Physics Mapping
//!
//! | Market Entity | Physics Entity | Property Mapping |
//! |--------------|----------------|------------------|
//! | Order | Rigid Body | Price→Y, Volume→Mass |
//! | Order Flow | Force | Direction, Magnitude |
//! | Liquidity | Friction | Resistance to price movement |
//! | Volatility | Energy | Kinetic energy in system |
//! | Support/Resistance | Constraint | Movement limits |
//!
//! ## Risk Applications
//!
//! - Simulate order book dynamics under stress
//! - Model flash crash propagation
//! - Identify unstable equilibrium states
//! - Calculate impact of large orders

use crate::core::{MarketRegime, RiskLevel, Symbol, Price, Quantity};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

// Re-export from rapier-hyperphysics when available
#[cfg(feature = "cwts-physics")]
use rapier_hyperphysics::{
    RapierHyperPhysicsAdapter, MarketMapper, SignalExtractor,
};

/// Current state of market physics simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPhysicsState {
    /// Total kinetic energy (volatility measure)
    pub kinetic_energy: f64,
    /// Potential energy (distance from equilibrium)
    pub potential_energy: f64,
    /// System momentum (directional bias)
    pub momentum: (f64, f64, f64), // (x, y, z) = (time, price, volume)
    /// Equilibrium stability
    pub stability: f64,
    /// Number of active bodies (orders)
    pub active_bodies: usize,
    /// Collision events (price interactions)
    pub collision_count: u32,
    /// Constraint violations (limit breaches)
    pub constraint_violations: u32,
}

impl MarketPhysicsState {
    /// Calculate total energy
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy + self.potential_energy
    }

    /// Get energy-based risk level
    #[must_use]
    pub fn energy_risk_level(&self) -> RiskLevel {
        let total = self.total_energy();
        if total > 1000.0 {
            RiskLevel::Critical
        } else if total > 500.0 {
            RiskLevel::High
        } else if total > 100.0 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }

    /// Get directional bias (positive = bullish, negative = bearish)
    #[must_use]
    pub fn directional_bias(&self) -> f64 {
        self.momentum.1 // Price momentum
    }
}

/// Order flow dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowDynamics {
    /// Net order flow (buy - sell)
    pub net_flow: f64,
    /// Flow imbalance ratio
    pub imbalance: f64,
    /// Flow acceleration
    pub acceleration: f64,
    /// Predicted impact
    pub predicted_impact: f64,
    /// Time to impact (ms)
    pub time_to_impact_ms: u64,
    /// Confidence in prediction
    pub confidence: f64,
}

/// Signal extracted from physics simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSignal {
    /// Signal type
    pub signal_type: PhysicsSignalType,
    /// Signal strength (0.0-1.0)
    pub strength: f64,
    /// Direction (positive or negative)
    pub direction: f64,
    /// Confidence
    pub confidence: f64,
    /// Time horizon (ms)
    pub horizon_ms: u64,
    /// Affected symbols
    pub affected_symbols: Vec<Symbol>,
}

/// Types of signals from physics simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsSignalType {
    /// Momentum signal
    Momentum,
    /// Energy accumulation
    EnergyAccumulation,
    /// Equilibrium instability
    InstabilityWarning,
    /// Cascade risk
    CascadeRisk,
    /// Liquidity vacuum
    LiquidityVacuum,
    /// Support/resistance approach
    LevelApproach,
    /// Oscillation detection
    Oscillation,
    /// Phase transition
    PhaseTransition,
}

impl PhysicsSignalType {
    /// Convert to risk level
    #[must_use]
    pub fn to_risk_level(&self, strength: f64) -> RiskLevel {
        match self {
            Self::CascadeRisk | Self::PhaseTransition => {
                if strength > 0.7 { RiskLevel::Critical }
                else if strength > 0.4 { RiskLevel::High }
                else { RiskLevel::Elevated }
            }
            Self::InstabilityWarning | Self::LiquidityVacuum => {
                if strength > 0.8 { RiskLevel::Critical }
                else if strength > 0.5 { RiskLevel::High }
                else { RiskLevel::Elevated }
            }
            _ => {
                if strength > 0.8 { RiskLevel::High }
                else if strength > 0.5 { RiskLevel::Elevated }
                else { RiskLevel::Normal }
            }
        }
    }
}

/// Configuration for physics risk adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Simulation time step (seconds)
    pub time_step: f32,
    /// Gravity (market forces) direction
    pub gravity: (f32, f32, f32),
    /// Price scale factor
    pub price_scale: f32,
    /// Volume scale factor
    pub volume_scale: f32,
    /// Maximum simulation substeps
    pub max_substeps: u32,
    /// Enable continuous collision detection
    pub enable_ccd: bool,
    /// Friction coefficient (liquidity)
    pub friction: f32,
    /// Restitution (bounce on support/resistance)
    pub restitution: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            time_step: 1.0 / 60.0, // 60 Hz
            gravity: (0.0, -9.81, 0.0),
            price_scale: 100.0,
            volume_scale: 0.001,
            max_substeps: 4,
            enable_ccd: true,
            friction: 0.5,
            restitution: 0.3,
        }
    }
}

/// Simulated order in physics space
#[derive(Debug, Clone)]
struct PhysicsOrder {
    id: u64,
    symbol: Symbol,
    side: OrderSide,
    price: f64,
    quantity: f64,
    body_handle: Option<usize>, // Rapier handle
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderSide {
    Buy,
    Sell,
}

/// Physics-based risk adapter
///
/// Uses physics simulation to analyze market dynamics:
/// - Maps order book to rigid body system
/// - Simulates order flow as forces
/// - Extracts risk signals from energy and momentum
pub struct PhysicsRiskAdapter {
    config: PhysicsConfig,
    state: RwLock<MarketPhysicsState>,
    signals: RwLock<Vec<SimulationSignal>>,
    orders: RwLock<HashMap<u64, PhysicsOrder>>,
    order_counter: AtomicU64,
    energy_history: RwLock<Vec<f64>>,
    momentum_history: RwLock<Vec<(f64, f64, f64)>>,
}

impl PhysicsRiskAdapter {
    /// Create a new physics risk adapter
    #[must_use]
    pub fn new(config: PhysicsConfig) -> Self {
        Self {
            config,
            state: RwLock::new(MarketPhysicsState {
                kinetic_energy: 0.0,
                potential_energy: 0.0,
                momentum: (0.0, 0.0, 0.0),
                stability: 1.0,
                active_bodies: 0,
                collision_count: 0,
                constraint_violations: 0,
            }),
            signals: RwLock::new(Vec::new()),
            orders: RwLock::new(HashMap::new()),
            order_counter: AtomicU64::new(0),
            energy_history: RwLock::new(Vec::with_capacity(1000)),
            momentum_history: RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Get current physics state
    #[must_use]
    pub fn state(&self) -> MarketPhysicsState {
        self.state.read().clone()
    }

    /// Get recent simulation signals
    #[must_use]
    pub fn signals(&self, limit: usize) -> Vec<SimulationSignal> {
        self.signals.read().iter().rev().take(limit).cloned().collect()
    }

    /// Add an order to the physics simulation
    pub fn add_order(
        &self,
        symbol: Symbol,
        side: bool, // true = buy, false = sell
        price: f64,
        quantity: f64,
    ) -> u64 {
        let id = self.order_counter.fetch_add(1, Ordering::SeqCst);

        let order = PhysicsOrder {
            id,
            symbol,
            side: if side { OrderSide::Buy } else { OrderSide::Sell },
            price,
            quantity,
            body_handle: None,
        };

        self.orders.write().insert(id, order);
        self.state.write().active_bodies += 1;

        id
    }

    /// Remove an order from the simulation
    pub fn remove_order(&self, id: u64) -> bool {
        if self.orders.write().remove(&id).is_some() {
            self.state.write().active_bodies -= 1;
            true
        } else {
            false
        }
    }

    /// Step the physics simulation
    pub fn step(&self, dt: f64) {
        // Calculate forces from order flow
        let (buy_force, sell_force) = self.calculate_order_forces();

        // Update momentum
        let net_force = buy_force - sell_force;
        let acceleration = net_force / self.state.read().active_bodies.max(1) as f64;

        {
            let mut state = self.state.write();
            state.momentum.1 += acceleration * dt;

            // Apply damping (liquidity friction)
            let damping = 1.0 - self.config.friction as f64 * dt;
            state.momentum.0 *= damping;
            state.momentum.1 *= damping;
            state.momentum.2 *= damping;

            // Update energy
            let velocity_sq = state.momentum.0.powi(2)
                + state.momentum.1.powi(2)
                + state.momentum.2.powi(2);
            state.kinetic_energy = 0.5 * velocity_sq * state.active_bodies as f64;

            // Calculate stability from energy oscillation
            let energy_history = self.energy_history.read();
            if energy_history.len() >= 10 {
                let recent = &energy_history[energy_history.len()-10..];
                let mean: f64 = recent.iter().sum::<f64>() / 10.0;
                let variance: f64 = recent.iter()
                    .map(|e| (e - mean).powi(2))
                    .sum::<f64>() / 10.0;
                state.stability = (1.0 - variance.sqrt() / mean.max(1.0)).clamp(0.0, 1.0);
            }
        }

        // Track history
        {
            let state = self.state.read();
            let mut energy_hist = self.energy_history.write();
            energy_hist.push(state.total_energy());
            if energy_hist.len() > 1000 {
                energy_hist.remove(0);
            }

            let mut momentum_hist = self.momentum_history.write();
            momentum_hist.push(state.momentum);
            if momentum_hist.len() > 1000 {
                momentum_hist.remove(0);
            }
        }

        // Extract signals
        self.extract_signals();
    }

    /// Calculate forces from order flow
    fn calculate_order_forces(&self) -> (f64, f64) {
        let orders = self.orders.read();

        let buy_force: f64 = orders.values()
            .filter(|o| matches!(o.side, OrderSide::Buy))
            .map(|o| o.quantity * self.config.volume_scale as f64)
            .sum();

        let sell_force: f64 = orders.values()
            .filter(|o| matches!(o.side, OrderSide::Sell))
            .map(|o| o.quantity * self.config.volume_scale as f64)
            .sum();

        (buy_force, sell_force)
    }

    /// Extract signals from simulation state
    fn extract_signals(&self) {
        let state = self.state.read();
        let mut signals = self.signals.write();

        // Momentum signal
        let momentum_mag = (state.momentum.0.powi(2)
            + state.momentum.1.powi(2)
            + state.momentum.2.powi(2)).sqrt();

        if momentum_mag > 10.0 {
            signals.push(SimulationSignal {
                signal_type: PhysicsSignalType::Momentum,
                strength: (momentum_mag / 100.0).min(1.0),
                direction: state.momentum.1.signum(),
                confidence: state.stability,
                horizon_ms: 1000,
                affected_symbols: Vec::new(),
            });
        }

        // Energy accumulation signal
        let energy_history = self.energy_history.read();
        if energy_history.len() >= 20 {
            let recent = &energy_history[energy_history.len()-10..];
            let older = &energy_history[energy_history.len()-20..energy_history.len()-10];

            let recent_avg: f64 = recent.iter().sum::<f64>() / 10.0;
            let older_avg: f64 = older.iter().sum::<f64>() / 10.0;

            if recent_avg > older_avg * 1.5 {
                signals.push(SimulationSignal {
                    signal_type: PhysicsSignalType::EnergyAccumulation,
                    strength: (recent_avg / older_avg.max(1.0) - 1.0).min(1.0),
                    direction: state.momentum.1.signum(),
                    confidence: 0.7,
                    horizon_ms: 5000,
                    affected_symbols: Vec::new(),
                });
            }
        }

        // Instability warning
        if state.stability < 0.3 {
            signals.push(SimulationSignal {
                signal_type: PhysicsSignalType::InstabilityWarning,
                strength: 1.0 - state.stability,
                direction: 0.0,
                confidence: 0.8,
                horizon_ms: 500,
                affected_symbols: Vec::new(),
            });
        }

        // Cascade risk (high energy + low stability)
        if state.total_energy() > 500.0 && state.stability < 0.5 {
            signals.push(SimulationSignal {
                signal_type: PhysicsSignalType::CascadeRisk,
                strength: (state.total_energy() / 1000.0).min(1.0) * (1.0 - state.stability),
                direction: state.momentum.1.signum(),
                confidence: state.stability,
                horizon_ms: 200,
                affected_symbols: Vec::new(),
            });
        }

        // Keep only recent signals
        if signals.len() > 100 {
            let excess = signals.len() - 100;
            signals.drain(0..excess);
        }
    }

    /// Analyze order flow dynamics
    pub fn analyze_order_flow(&self) -> OrderFlowDynamics {
        let (buy_force, sell_force) = self.calculate_order_forces();
        let net_flow = buy_force - sell_force;
        let total_flow = buy_force + sell_force;

        let imbalance = if total_flow > 0.0 {
            net_flow / total_flow
        } else {
            0.0
        };

        // Calculate acceleration from momentum history
        let momentum_history = self.momentum_history.read();
        let acceleration = if momentum_history.len() >= 2 {
            let curr = momentum_history.last().unwrap();
            let prev = &momentum_history[momentum_history.len() - 2];
            curr.1 - prev.1
        } else {
            0.0
        };

        // Predict impact based on flow and liquidity
        let state = self.state.read();
        let predicted_impact = net_flow * (1.0 - self.config.friction as f64)
            / state.active_bodies.max(1) as f64;

        OrderFlowDynamics {
            net_flow,
            imbalance,
            acceleration,
            predicted_impact,
            time_to_impact_ms: (100.0 / acceleration.abs().max(0.01)) as u64,
            confidence: state.stability,
        }
    }

    /// Simulate stress scenario
    pub fn simulate_stress(
        &self,
        shock_magnitude: f64,
        duration_steps: usize,
    ) -> Vec<MarketPhysicsState> {
        let mut results = Vec::with_capacity(duration_steps);

        // Store initial state
        let initial_state = self.state.read().clone();

        // Apply shock
        {
            let mut state = self.state.write();
            state.kinetic_energy += shock_magnitude;
            state.momentum.1 += shock_magnitude.sqrt() * -1.0; // Downward shock
        }

        // Run simulation
        for _ in 0..duration_steps {
            self.step(self.config.time_step as f64);
            results.push(self.state.read().clone());
        }

        // Restore initial state (this was a simulation)
        *self.state.write() = initial_state;

        results
    }

    /// Get risk level based on physics state
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        let state = self.state.read();

        // Combine energy and stability into risk
        let energy_risk = state.energy_risk_level();
        let stability_risk = if state.stability < 0.3 {
            RiskLevel::Critical
        } else if state.stability < 0.5 {
            RiskLevel::High
        } else if state.stability < 0.7 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        };

        // Return worst case
        match (energy_risk, stability_risk) {
            (RiskLevel::Critical, _) | (_, RiskLevel::Critical) => RiskLevel::Critical,
            (RiskLevel::High, _) | (_, RiskLevel::High) => RiskLevel::High,
            (RiskLevel::Elevated, _) | (_, RiskLevel::Elevated) => RiskLevel::Elevated,
            _ => RiskLevel::Normal,
        }
    }

    /// Assess physics-based risk for a portfolio.
    ///
    /// Returns a `SubsystemRisk` for integration with the CWTS coordinator.
    pub fn assess_physics_risk(&self, portfolio: &crate::core::Portfolio) -> super::coordinator::SubsystemRisk {
        use super::coordinator::{SubsystemRisk, SubsystemId};
        use crate::core::Timestamp;

        let start = std::time::Instant::now();

        // Step simulation to get current state
        self.step(self.config.time_step as f64);

        let state = self.state.read();
        let signals = self.signals(10);
        let flow = self.analyze_order_flow();

        // Calculate risk from physics state
        let energy_risk = state.total_energy() / 1000.0; // Normalize to 0-1
        let stability_risk = 1.0 - state.stability;
        let momentum_risk = (state.momentum.0.powi(2) + state.momentum.1.powi(2) + state.momentum.2.powi(2)).sqrt() / 100.0;
        let imbalance_risk = flow.imbalance.abs();

        // Check for critical signals
        let signal_risk = signals.iter()
            .filter(|s| matches!(s.signal_type, PhysicsSignalType::CascadeRisk | PhysicsSignalType::PhaseTransition))
            .map(|s| s.strength)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Combined risk score
        let combined_risk = (
            energy_risk * 0.25 +
            stability_risk * 0.30 +
            momentum_risk * 0.15 +
            imbalance_risk * 0.15 +
            signal_risk * 0.15
        ).clamp(0.0, 1.0);

        // Determine risk level
        let risk_level = if combined_risk > 0.8 || signal_risk > 0.7 {
            crate::core::RiskLevel::Critical
        } else if combined_risk > 0.6 || state.stability < 0.3 {
            crate::core::RiskLevel::High
        } else if combined_risk > 0.3 || state.stability < 0.5 {
            crate::core::RiskLevel::Elevated
        } else {
            crate::core::RiskLevel::Normal
        };

        // Position factor based on stability
        let position_factor = state.stability.clamp(0.3, 1.0);

        // Confidence based on simulation stability
        let confidence = state.stability * 0.7 + 0.3; // Base confidence of 0.3

        let latency_ns = start.elapsed().as_nanos() as u64;

        let reasoning = format!(
            "Physics: energy={:.2}, stability={:.3}, momentum={:.2}, flow_imbalance={:.2}%",
            state.total_energy(),
            state.stability,
            momentum_risk,
            imbalance_risk * 100.0
        );

        SubsystemRisk {
            subsystem: SubsystemId::Physics,
            risk_level,
            confidence,
            risk_score: combined_risk,
            position_factor,
            reasoning,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Reset simulation
    pub fn reset(&self) {
        *self.state.write() = MarketPhysicsState {
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            momentum: (0.0, 0.0, 0.0),
            stability: 1.0,
            active_bodies: 0,
            collision_count: 0,
            constraint_violations: 0,
        };
        self.orders.write().clear();
        self.signals.write().clear();
        self.energy_history.write().clear();
        self.momentum_history.write().clear();
    }
}

impl Default for PhysicsRiskAdapter {
    fn default() -> Self {
        Self::new(PhysicsConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_state_energy() {
        let state = MarketPhysicsState {
            kinetic_energy: 100.0,
            potential_energy: 50.0,
            momentum: (1.0, 2.0, 0.5),
            stability: 0.8,
            active_bodies: 10,
            collision_count: 0,
            constraint_violations: 0,
        };

        assert_eq!(state.total_energy(), 150.0);
        assert_eq!(state.directional_bias(), 2.0);
    }

    #[test]
    fn test_signal_risk_level() {
        assert_eq!(
            PhysicsSignalType::CascadeRisk.to_risk_level(0.8),
            RiskLevel::Critical
        );
        assert_eq!(
            PhysicsSignalType::Momentum.to_risk_level(0.3),
            RiskLevel::Normal
        );
    }

    #[test]
    fn test_adapter_creation() {
        let adapter = PhysicsRiskAdapter::default();
        let state = adapter.state();
        assert_eq!(state.active_bodies, 0);
        assert_eq!(state.stability, 1.0);
    }

    #[test]
    fn test_add_remove_orders() {
        let adapter = PhysicsRiskAdapter::default();

        let id1 = adapter.add_order(Symbol::new("BTC"), true, 50000.0, 1.0);
        let id2 = adapter.add_order(Symbol::new("BTC"), false, 50100.0, 0.5);

        assert_eq!(adapter.state().active_bodies, 2);

        assert!(adapter.remove_order(id1));
        assert_eq!(adapter.state().active_bodies, 1);

        assert!(!adapter.remove_order(id1)); // Already removed
    }

    #[test]
    fn test_step_simulation() {
        let adapter = PhysicsRiskAdapter::default();

        // Add imbalanced orders (more buys)
        for _ in 0..5 {
            adapter.add_order(Symbol::new("BTC"), true, 50000.0, 10.0);
        }
        adapter.add_order(Symbol::new("BTC"), false, 50100.0, 1.0);

        // Run simulation
        for _ in 0..10 {
            adapter.step(1.0 / 60.0);
        }

        let state = adapter.state();
        // With more buy orders, momentum should be positive
        assert!(state.momentum.1 > 0.0);
    }

    #[test]
    fn test_order_flow_analysis() {
        let adapter = PhysicsRiskAdapter::default();

        // Add orders
        for _ in 0..3 {
            adapter.add_order(Symbol::new("ETH"), true, 3000.0, 10.0);
        }
        adapter.add_order(Symbol::new("ETH"), false, 3010.0, 5.0);

        let flow = adapter.analyze_order_flow();

        assert!(flow.net_flow > 0.0); // More buy pressure
        assert!(flow.imbalance > 0.0);
    }

    #[test]
    fn test_stress_simulation() {
        let adapter = PhysicsRiskAdapter::default();

        // Add some orders
        for _ in 0..5 {
            adapter.add_order(Symbol::new("SOL"), true, 150.0, 100.0);
        }

        // Run stress test
        let results = adapter.simulate_stress(1000.0, 100);

        assert!(!results.is_empty());
        // Energy should eventually dissipate
        assert!(results.last().unwrap().kinetic_energy < 1000.0);
    }

    #[test]
    fn test_reset() {
        let adapter = PhysicsRiskAdapter::default();

        adapter.add_order(Symbol::new("BTC"), true, 50000.0, 1.0);
        adapter.step(1.0 / 60.0);

        adapter.reset();

        assert_eq!(adapter.state().active_bodies, 0);
        assert_eq!(adapter.state().kinetic_energy, 0.0);
    }
}
