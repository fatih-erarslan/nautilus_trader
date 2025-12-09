//! # Physics-Based Market Microstructure Simulator
//!
//! Models order book dynamics using Navier-Stokes fluid equations.
//!
//! ## Navier-Stokes Formulation
//!
//! ### Continuity Equation (Mass Conservation)
//! ```text
//! ∂ρ/∂t + ∂(ρv)/∂x = source
//! ```
//!
//! ### Momentum Equation
//! ```text
//! ∂v/∂t + v·∂v/∂x = -(1/ρ)·∂P/∂x + ν·∂²v/∂x² - κv
//! ```
//!
//! ### Equation of State
//! ```text
//! P = ρ·T + κ·(ρ_bid - ρ_ask)
//! ```
//!
//! ## Variables
//! - ρ: Liquidity density (shares per price level)
//! - v: Price momentum (velocity field)
//! - P: Market pressure (supply-demand imbalance)
//! - ν: Viscosity (transaction costs)
//! - T: Temperature (volatility)

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Order book state (fluid state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookState {
    /// Liquidity density (shares available per price level)
    pub bid_density: Vec<f64>,
    pub ask_density: Vec<f64>,
    /// Price momentum (velocity field)
    pub bid_velocity: Vec<f64>,
    pub ask_velocity: Vec<f64>,
    /// Market pressure (supply-demand imbalance)
    pub pressure: Vec<f64>,
    /// Price levels (grid)
    pub price_levels: Vec<f64>,
    /// Mid price
    pub mid_price: f64,
    /// Spread
    pub spread: f64,
    /// Time (microseconds)
    pub time_us: u64,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

/// Order (particle in the fluid)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Order {
    pub side: Side,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
}

/// Market event (force acting on the fluid)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    LimitOrder(Order),
    MarketOrder { side: Side, quantity: f64, timestamp: u64 },
    CancelOrder { side: Side, price: f64, quantity: f64, timestamp: u64 },
}

/// Fluid dynamics parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidParameters {
    /// Viscosity (transaction cost, friction)
    pub viscosity: f64,
    /// Compressibility (price elasticity)
    pub gamma: f64,
    /// Diffusion coefficient (order book spread)
    pub diffusion: f64,
    /// Mean reversion strength
    pub reversion_strength: f64,
    /// Temperature (market volatility)
    pub temperature: f64,
}

impl Default for FluidParameters {
    fn default() -> Self {
        Self {
            viscosity: 0.01,
            gamma: 1.4,
            diffusion: 0.05,
            reversion_strength: 0.3,
            temperature: 0.15, // Matches thermodynamic scheduler
        }
    }
}

/// Market Microstructure Simulator
pub struct MarketMicrostructureSimulator {
    /// Current order book state
    state: OrderBookState,
    /// Fluid parameters
    params: FluidParameters,
    /// Price grid resolution
    price_tick: f64,
    /// Number of price levels
    num_levels: usize,
    /// Event history
    event_history: VecDeque<MarketEvent>,
    /// Timestep (microseconds)
    dt_us: u64,
}

impl MarketMicrostructureSimulator {
    /// Create new simulator
    pub fn new(
        initial_mid_price: f64,
        price_tick: f64,
        num_levels: usize,
        params: FluidParameters,
    ) -> Self {
        // Initialize price grid
        let price_levels: Vec<f64> = (0..num_levels)
            .map(|i| initial_mid_price + (i as f64 - num_levels as f64 / 2.0) * price_tick)
            .collect();

        // Initialize uniform liquidity density
        let initial_density = 1000.0;
        let bid_density = vec![initial_density; num_levels];
        let ask_density = vec![initial_density; num_levels];

        // Zero initial velocity and pressure
        let bid_velocity = vec![0.0; num_levels];
        let ask_velocity = vec![0.0; num_levels];
        let pressure = vec![0.0; num_levels];

        let state = OrderBookState {
            bid_density,
            ask_density,
            bid_velocity,
            ask_velocity,
            pressure,
            price_levels,
            mid_price: initial_mid_price,
            spread: price_tick,
            time_us: 0,
        };

        Self {
            state,
            params,
            price_tick,
            num_levels,
            event_history: VecDeque::with_capacity(10000),
            dt_us: 10, // 10µs timestep
        }
    }

    /// HyperPhysics default configuration
    pub fn hyperphysics_default(initial_price: f64) -> Self {
        Self::new(initial_price, 0.01, 200, FluidParameters::default())
    }

    /// Step simulation forward
    pub fn step(&mut self, events: &[MarketEvent]) {
        // 1. Process market events (external forces)
        for event in events {
            self.apply_event(event);
        }

        // 2. Update fluid dynamics (Navier-Stokes)
        self.update_fluid_dynamics();

        // 3. Update market pressure (equation of state)
        self.update_pressure();

        // 4. Compute mid price and spread
        self.update_market_stats();

        // 5. Advance time
        self.state.time_us += self.dt_us;

        // 6. Record events
        for event in events {
            self.event_history.push_back(event.clone());
            if self.event_history.len() > 10000 {
                self.event_history.pop_front();
            }
        }
    }

    /// Apply market event (force on fluid)
    fn apply_event(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::LimitOrder(order) => {
                self.add_liquidity(order);
            }
            MarketEvent::MarketOrder { side, quantity, .. } => {
                self.consume_liquidity(*side, *quantity);
            }
            MarketEvent::CancelOrder { side, price, quantity, .. } => {
                self.remove_liquidity(*side, *price, *quantity);
            }
        }
    }

    /// Add liquidity (source term in continuity equation)
    fn add_liquidity(&mut self, order: &Order) {
        let level_idx = self.price_to_level(order.price);

        match order.side {
            Side::Bid => {
                self.state.bid_density[level_idx] += order.quantity;
            }
            Side::Ask => {
                self.state.ask_density[level_idx] += order.quantity;
            }
        }
    }

    /// Consume liquidity (market order impact)
    fn consume_liquidity(&mut self, side: Side, mut quantity: f64) {
        let levels = match side {
            Side::Bid => &mut self.state.ask_density,
            Side::Ask => &mut self.state.bid_density,
        };

        let start_idx = self.num_levels / 2;
        let direction: i32 = match side {
            Side::Bid => 1,
            Side::Ask => -1,
        };

        let mut idx = start_idx as i32;
        while quantity > 0.0 && idx >= 0 && idx < self.num_levels as i32 {
            let available = levels[idx as usize];
            let consumed = quantity.min(available);

            levels[idx as usize] -= consumed;
            quantity -= consumed;

            // Price impact (momentum transfer)
            let velocity = match side {
                Side::Bid => &mut self.state.ask_velocity,
                Side::Ask => &mut self.state.bid_velocity,
            };
            velocity[idx as usize] += consumed * direction as f64 * 0.01;

            idx += direction;
        }
    }

    /// Remove liquidity (cancel order)
    fn remove_liquidity(&mut self, side: Side, price: f64, quantity: f64) {
        let level_idx = self.price_to_level(price);

        match side {
            Side::Bid => {
                self.state.bid_density[level_idx] =
                    (self.state.bid_density[level_idx] - quantity).max(0.0);
            }
            Side::Ask => {
                self.state.ask_density[level_idx] =
                    (self.state.ask_density[level_idx] - quantity).max(0.0);
            }
        }
    }

    /// Update fluid dynamics (Navier-Stokes equations)
    fn update_fluid_dynamics(&mut self) {
        let dt = self.dt_us as f64 / 1_000_000.0;
        let dx = self.price_tick;
        let n = self.num_levels;

        // Bid side update
        let mut new_bid_density = self.state.bid_density.clone();
        let mut new_bid_velocity = self.state.bid_velocity.clone();

        // Ask side update
        let mut new_ask_density = self.state.ask_density.clone();
        let mut new_ask_velocity = self.state.ask_velocity.clone();

        // Interior points (finite difference)
        for i in 1..(n - 1) {
            // Bid side
            let flux_in = self.state.bid_density[i - 1] * self.state.bid_velocity[i - 1];
            let flux_out = self.state.bid_density[i] * self.state.bid_velocity[i];
            let drho_dt = -(flux_out - flux_in) / dx;
            new_bid_density[i] = (self.state.bid_density[i] + drho_dt * dt).max(0.0);

            let advection = self.state.bid_velocity[i] *
                (self.state.bid_velocity[i + 1] - self.state.bid_velocity[i - 1]) / (2.0 * dx);
            let pressure_grad = (self.state.pressure[i + 1] - self.state.pressure[i - 1]) / (2.0 * dx);
            let pressure_force = if self.state.bid_density[i] > 0.0 {
                -pressure_grad / self.state.bid_density[i]
            } else { 0.0 };
            let viscous = self.params.viscosity *
                (self.state.bid_velocity[i + 1] - 2.0 * self.state.bid_velocity[i] +
                 self.state.bid_velocity[i - 1]) / (dx * dx);
            let reversion = -self.params.reversion_strength * self.state.bid_velocity[i];

            let dv_dt = -advection + pressure_force + viscous + reversion;
            new_bid_velocity[i] = self.state.bid_velocity[i] + dv_dt * dt;

            // Ask side (similar)
            let flux_in = self.state.ask_density[i - 1] * self.state.ask_velocity[i - 1];
            let flux_out = self.state.ask_density[i] * self.state.ask_velocity[i];
            let drho_dt = -(flux_out - flux_in) / dx;
            new_ask_density[i] = (self.state.ask_density[i] + drho_dt * dt).max(0.0);

            let advection = self.state.ask_velocity[i] *
                (self.state.ask_velocity[i + 1] - self.state.ask_velocity[i - 1]) / (2.0 * dx);
            let pressure_force = if self.state.ask_density[i] > 0.0 {
                -pressure_grad / self.state.ask_density[i]
            } else { 0.0 };
            let viscous = self.params.viscosity *
                (self.state.ask_velocity[i + 1] - 2.0 * self.state.ask_velocity[i] +
                 self.state.ask_velocity[i - 1]) / (dx * dx);
            let reversion = -self.params.reversion_strength * self.state.ask_velocity[i];

            let dv_dt = -advection + pressure_force + viscous + reversion;
            new_ask_velocity[i] = self.state.ask_velocity[i] + dv_dt * dt;
        }

        // Boundary conditions (reflective)
        new_bid_density[0] = new_bid_density[1];
        new_bid_density[n - 1] = new_bid_density[n - 2];
        new_bid_velocity[0] = -new_bid_velocity[1];
        new_bid_velocity[n - 1] = -new_bid_velocity[n - 2];

        new_ask_density[0] = new_ask_density[1];
        new_ask_density[n - 1] = new_ask_density[n - 2];
        new_ask_velocity[0] = -new_ask_velocity[1];
        new_ask_velocity[n - 1] = -new_ask_velocity[n - 2];

        self.state.bid_density = new_bid_density;
        self.state.bid_velocity = new_bid_velocity;
        self.state.ask_density = new_ask_density;
        self.state.ask_velocity = new_ask_velocity;
    }

    /// Update market pressure (equation of state)
    fn update_pressure(&mut self) {
        for i in 0..self.num_levels {
            let bid_density = self.state.bid_density[i];
            let ask_density = self.state.ask_density[i];

            let imbalance = bid_density - ask_density;
            let density_avg = (bid_density + ask_density) / 2.0;

            // P = ρ·T + κ·imbalance
            self.state.pressure[i] = density_avg * self.params.temperature +
                                     self.params.reversion_strength * imbalance;
        }
    }

    /// Update market statistics
    fn update_market_stats(&mut self) {
        let mid_idx = self.num_levels / 2;

        // Find best bid
        let mut best_bid_idx = mid_idx;
        for i in (0..mid_idx).rev() {
            if self.state.bid_density[i] > 1.0 {
                best_bid_idx = i;
                break;
            }
        }

        // Find best ask
        let mut best_ask_idx = mid_idx;
        for i in mid_idx..self.num_levels {
            if self.state.ask_density[i] > 1.0 {
                best_ask_idx = i;
                break;
            }
        }

        let best_bid = self.state.price_levels[best_bid_idx];
        let best_ask = self.state.price_levels[best_ask_idx];

        self.state.mid_price = (best_bid + best_ask) / 2.0;
        self.state.spread = best_ask - best_bid;
    }

    /// Convert price to level index
    fn price_to_level(&self, price: f64) -> usize {
        let relative = (price - self.state.price_levels[0]) / self.price_tick;
        let idx = relative.round() as usize;
        idx.min(self.num_levels - 1)
    }

    /// Get current state
    pub fn get_state(&self) -> &OrderBookState {
        &self.state
    }

    /// Compute order book imbalance (predictive signal)
    pub fn compute_imbalance(&self) -> f64 {
        let mid_idx = self.num_levels / 2;
        let window = 10;

        let mut bid_volume = 0.0;
        let mut ask_volume = 0.0;

        for i in (mid_idx.saturating_sub(window))..(mid_idx + window).min(self.num_levels) {
            let distance_weight = 1.0 / (1.0 + (i as i32 - mid_idx as i32).abs() as f64);
            bid_volume += self.state.bid_density[i] * distance_weight;
            ask_volume += self.state.ask_density[i] * distance_weight;
        }

        (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6)
    }

    /// Compute price impact
    pub fn compute_price_impact(&self, side: Side, quantity: f64) -> f64 {
        let levels = match side {
            Side::Bid => &self.state.ask_density,
            Side::Ask => &self.state.bid_density,
        };

        let mid_idx = self.num_levels / 2;
        let mut cumulative_depth = 0.0;
        let mut levels_consumed = 0;

        for i in 0..10 {
            let idx = if matches!(side, Side::Bid) {
                mid_idx + i
            } else {
                mid_idx.saturating_sub(i)
            };

            if idx >= levels.len() { break; }

            cumulative_depth += levels[idx];
            levels_consumed += 1;

            if cumulative_depth >= quantity { break; }
        }

        let impact = levels_consumed as f64 * self.price_tick;

        match side {
            Side::Bid => impact,
            Side::Ask => -impact,
        }
    }

    /// Compute volatility (from pressure fluctuations)
    pub fn compute_volatility(&self) -> f64 {
        let n = self.state.pressure.len() as f64;
        let mean_pressure: f64 = self.state.pressure.iter().sum::<f64>() / n;
        let mean_pressure_sq: f64 = self.state.pressure.iter()
            .map(|p| p * p)
            .sum::<f64>() / n;

        (mean_pressure_sq - mean_pressure * mean_pressure).sqrt()
    }

    /// Set temperature parameter (for integration)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.params.temperature = temperature;
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.params.temperature
    }
}

/// Market signals extracted from simulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSignals {
    pub mid_price: f64,
    pub spread: f64,
    pub imbalance: f64,
    pub volatility: f64,
    pub time_us: u64,
}

/// Boltzmann order generator using statistical mechanics
pub struct BoltzmannOrderGenerator {
    temperature: f64,
    mean_arrival_rate: f64,
    mean_order_size: f64,
}

impl BoltzmannOrderGenerator {
    pub fn new(temperature: f64, arrival_rate: f64, order_size: f64) -> Self {
        Self {
            temperature,
            mean_arrival_rate: arrival_rate,
            mean_order_size: order_size,
        }
    }

    /// Generate next order (Poisson arrival, Boltzmann price distribution)
    pub fn generate_order(&self, mid_price: f64, time_us: u64) -> Option<MarketEvent> {
        let mut rng = rand::thread_rng();

        let dt = 1.0 / 1_000_000.0;
        let arrival_prob = self.mean_arrival_rate * dt;

        if rng.gen::<f64>() < arrival_prob {
            let side = if rng.gen::<f64>() < 0.5 { Side::Bid } else { Side::Ask };

            // Price offset from mid (Boltzmann distribution)
            let u: f64 = rng.gen();
            let offset = -self.temperature * u.ln() * if matches!(side, Side::Bid) { -1.0 } else { 1.0 };
            let price = mid_price + offset;

            // Quantity (exponential distribution)
            let u: f64 = rng.gen();
            let quantity = -self.mean_order_size * u.ln();

            // 90% limit orders, 10% market orders
            if rng.gen::<f64>() < 0.9 {
                Some(MarketEvent::LimitOrder(Order {
                    side,
                    price,
                    quantity,
                    timestamp: time_us,
                }))
            } else {
                Some(MarketEvent::MarketOrder {
                    side,
                    quantity,
                    timestamp: time_us,
                })
            }
        } else {
            None
        }
    }
}

/// Real-time market simulator for HyperPhysics
pub struct HyperPhysicsMarketSimulator {
    simulator: MarketMicrostructureSimulator,
    order_generator: BoltzmannOrderGenerator,
}

impl HyperPhysicsMarketSimulator {
    pub fn new(initial_price: f64, temperature: f64) -> Self {
        let simulator = MarketMicrostructureSimulator::hyperphysics_default(initial_price);
        let order_generator = BoltzmannOrderGenerator::new(temperature, 5000.0, 100.0);

        Self { simulator, order_generator }
    }

    /// Step simulation (10µs per call)
    pub fn step(&mut self) -> MarketSignals {
        let current_time = self.simulator.get_state().time_us;
        let mid_price = self.simulator.get_state().mid_price;

        let mut events = Vec::new();
        if let Some(event) = self.order_generator.generate_order(mid_price, current_time) {
            events.push(event);
        }

        self.simulator.step(&events);

        MarketSignals {
            mid_price: self.simulator.get_state().mid_price,
            spread: self.simulator.get_state().spread,
            imbalance: self.simulator.compute_imbalance(),
            volatility: self.simulator.compute_volatility(),
            time_us: self.simulator.get_state().time_us,
        }
    }

    /// Get current order book state
    pub fn get_order_book(&self) -> &OrderBookState {
        self.simulator.get_state()
    }

    /// Compute price impact for planned trade
    pub fn compute_impact(&self, side: Side, quantity: f64) -> f64 {
        self.simulator.compute_price_impact(side, quantity)
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.simulator.set_temperature(temperature);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_initialization() {
        let sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        assert_eq!(sim.state.mid_price, 100.0);
        assert!(sim.state.bid_density.iter().all(|&d| d > 0.0));
    }

    #[test]
    fn test_limit_order_addition() {
        let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let initial_density = sim.state.bid_density[100];

        let order = Order {
            side: Side::Bid,
            price: 100.0,
            quantity: 100.0,
            timestamp: 0,
        };

        sim.add_liquidity(&order);
        assert_eq!(sim.state.bid_density[100], initial_density + 100.0);
    }

    #[test]
    fn test_imbalance_computation() {
        let sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let imbalance = sim.compute_imbalance();
        assert!(imbalance.abs() <= 1.0);
    }

    #[test]
    fn test_physics_conservation() {
        let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let initial_total = sim.state.bid_density.iter().sum::<f64>() +
                           sim.state.ask_density.iter().sum::<f64>();

        // Step without events
        sim.step(&[]);

        let final_total = sim.state.bid_density.iter().sum::<f64>() +
                         sim.state.ask_density.iter().sum::<f64>();

        // Conservation (small numerical error ok)
        assert!((final_total - initial_total).abs() < 1.0);
    }
}
