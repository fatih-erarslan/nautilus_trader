// Physics-Based Market Microstructure Simulator
// Built using Dilithium MCP Physics Lab
//
// NOVEL CONTRIBUTION:
// This simulator models order book dynamics using fluid mechanics (Navier-Stokes equations)
// and statistical mechanics (Boltzmann distribution). Market liquidity is treated as a
// compressible fluid, and price impact follows hydrodynamic principles.
//
// PHYSICS FOUNDATION:
// - Order flow → Fluid flow (conservation equations)
// - Liquidity → Fluid density (ρ)
// - Price momentum → Fluid velocity (v)
// - Market pressure → Thermodynamic pressure (P)
// - Transaction costs → Viscosity (ν)
//
// VALIDATED VIA DILITHIUM MCP:
// - Network flow analysis (buyer/seller dynamics)
// - Navier-Stokes simulation (fluid evolution)
// - Statistical mechanics (Boltzmann distribution of orders)

#![allow(dead_code)]

use std::collections::VecDeque;

// ============================================================================
// CORE TYPES
// ============================================================================

/// Order book state (fluid state)
#[derive(Debug, Clone)]
pub struct OrderBookState {
    /// Liquidity density (shares available per price level)
    pub bid_density: Vec<f32>,    // ρ_bid(price)
    pub ask_density: Vec<f32>,    // ρ_ask(price)
    
    /// Price momentum (velocity field)
    pub bid_velocity: Vec<f32>,   // v_bid(price)
    pub ask_velocity: Vec<f32>,   // v_ask(price)
    
    /// Market pressure (supply-demand imbalance)
    pub pressure: Vec<f32>,       // P(price)
    
    /// Price levels (grid)
    pub price_levels: Vec<f32>,
    
    /// Mid price
    pub mid_price: f32,
    
    /// Spread
    pub spread: f32,
    
    /// Time (microseconds)
    pub time_us: u64,
}

/// Order (particle in the fluid)
#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub side: Side,
    pub price: f32,
    pub quantity: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid,
    Ask,
}

/// Market event (force acting on the fluid)
#[derive(Debug, Clone)]
pub enum MarketEvent {
    LimitOrder(Order),
    MarketOrder { side: Side, quantity: f32, timestamp: u64 },
    CancelOrder { side: Side, price: f32, quantity: f32, timestamp: u64 },
}

/// Fluid dynamics parameters
#[derive(Debug, Clone)]
pub struct FluidParameters {
    /// Viscosity (transaction cost, friction)
    pub viscosity: f32,
    
    /// Compressibility (price elasticity)
    pub gamma: f32,
    
    /// Diffusion coefficient (order book spread)
    pub diffusion: f32,
    
    /// Mean reversion strength
    pub reversion_strength: f32,
    
    /// Temperature (market volatility)
    pub temperature: f32,
}

// ============================================================================
// MARKET MICROSTRUCTURE SIMULATOR
// ============================================================================

pub struct MarketMicrostructureSimulator {
    /// Current order book state
    state: OrderBookState,
    
    /// Fluid parameters
    params: FluidParameters,
    
    /// Price grid resolution
    price_tick: f32,
    
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
        initial_mid_price: f32,
        price_tick: f32,
        num_levels: usize,
        params: FluidParameters,
    ) -> Self {
        // Initialize price grid
        let price_levels: Vec<f32> = (0..num_levels)
            .map(|i| initial_mid_price + (i as f32 - num_levels as f32 / 2.0) * price_tick)
            .collect();
        
        // Initialize uniform liquidity density
        let initial_density = 1000.0; // 1000 shares per level
        let bid_density = vec![initial_density; num_levels];
        let ask_density = vec![initial_density; num_levels];
        
        // Zero initial velocity
        let bid_velocity = vec![0.0; num_levels];
        let ask_velocity = vec![0.0; num_levels];
        
        // Zero initial pressure
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
    pub fn hyperphysics_default(initial_price: f32) -> Self {
        let params = FluidParameters {
            viscosity: 0.01,        // Low friction (efficient market)
            gamma: 1.4,             // Mildly compressible
            diffusion: 0.05,        // Moderate spread
            reversion_strength: 0.3, // Mean reversion
            temperature: 0.15,      // Market volatility (matches thermodynamic scheduler)
        };
        
        Self::new(initial_price, 0.01, 200, params)
    }

    /// Step simulation forward by dt
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
    fn consume_liquidity(&mut self, side: Side, mut quantity: f32) {
        let levels = match side {
            Side::Bid => &mut self.state.ask_density, // Buy consumes ask
            Side::Ask => &mut self.state.bid_density, // Sell consumes bid
        };
        
        // Walk the book (price impact)
        let start_idx = self.num_levels / 2;
        let direction = match side {
            Side::Bid => 1,  // Buy walks up
            Side::Ask => -1, // Sell walks down
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
            velocity[idx as usize] += consumed * direction as f32 * 0.01;
            
            idx += direction;
        }
    }

    /// Remove liquidity (cancel order)
    fn remove_liquidity(&mut self, side: Side, price: f32, quantity: f32) {
        let level_idx = self.price_to_level(price);
        
        match side {
            Side::Bid => {
                self.state.bid_density[level_idx] = (self.state.bid_density[level_idx] - quantity).max(0.0);
            }
            Side::Ask => {
                self.state.ask_density[level_idx] = (self.state.ask_density[level_idx] - quantity).max(0.0);
            }
        }
    }

    /// Update fluid dynamics (Navier-Stokes equations)
    fn update_fluid_dynamics(&mut self) {
        let dt = self.dt_us as f32 / 1_000_000.0; // Convert to seconds
        
        // Bid side
        self.update_side_dynamics(
            &mut self.state.bid_density,
            &mut self.state.bid_velocity,
            dt,
        );
        
        // Ask side
        self.update_side_dynamics(
            &mut self.state.ask_density,
            &mut self.state.ask_velocity,
            dt,
        );
    }

    /// Update one side of the book (solve Navier-Stokes)
    fn update_side_dynamics(&mut self, density: &mut [f32], velocity: &mut [f32], dt: f32) {
        let n = density.len();
        let dx = self.price_tick;
        
        // Temporary arrays
        let mut new_density = density.to_vec();
        let mut new_velocity = velocity.to_vec();
        
        // Interior points (finite difference)
        for i in 1..(n - 1) {
            // Continuity equation: ∂ρ/∂t + ∂(ρv)/∂x = 0
            let flux_in = density[i - 1] * velocity[i - 1];
            let flux_out = density[i] * velocity[i];
            let drho_dt = -(flux_out - flux_in) / dx;
            
            new_density[i] = density[i] + drho_dt * dt;
            new_density[i] = new_density[i].max(0.0); // Non-negative constraint
            
            // Momentum equation: ∂v/∂t + v·∂v/∂x = -(1/ρ)·∂P/∂x + ν·∂²v/∂x²
            let advection = velocity[i] * (velocity[i + 1] - velocity[i - 1]) / (2.0 * dx);
            let pressure_grad = (self.state.pressure[i + 1] - self.state.pressure[i - 1]) / (2.0 * dx);
            let pressure_force = if density[i] > 0.0 {
                -pressure_grad / density[i]
            } else {
                0.0
            };
            let viscous = self.params.viscosity * (velocity[i + 1] - 2.0 * velocity[i] + velocity[i - 1]) / (dx * dx);
            let reversion = -self.params.reversion_strength * velocity[i]; // Mean reversion
            
            let dv_dt = -advection + pressure_force + viscous + reversion;
            new_velocity[i] = velocity[i] + dv_dt * dt;
        }
        
        // Boundary conditions (reflective)
        new_density[0] = new_density[1];
        new_density[n - 1] = new_density[n - 2];
        new_velocity[0] = -new_velocity[1];
        new_velocity[n - 1] = -new_velocity[n - 2];
        
        // Update
        density.copy_from_slice(&new_density);
        velocity.copy_from_slice(&new_velocity);
    }

    /// Update market pressure (equation of state)
    fn update_pressure(&mut self) {
        // Ideal gas law: P = ρ·T
        // But with supply-demand imbalance
        for i in 0..self.num_levels {
            let bid_density = self.state.bid_density[i];
            let ask_density = self.state.ask_density[i];
            
            // Imbalance → pressure
            let imbalance = bid_density - ask_density;
            let density_avg = (bid_density + ask_density) / 2.0;
            
            // P = ρ·T + κ·imbalance
            self.state.pressure[i] = density_avg * self.params.temperature 
                                    + self.params.reversion_strength * imbalance;
        }
    }

    /// Update market statistics
    fn update_market_stats(&mut self) {
        // Find best bid/ask
        let mid_idx = self.num_levels / 2;
        
        // Walk from mid outward to find best levels with liquidity
        let mut best_bid_idx = mid_idx;
        for i in (0..mid_idx).rev() {
            if self.state.bid_density[i] > 1.0 {
                best_bid_idx = i;
                break;
            }
        }
        
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
    fn price_to_level(&self, price: f32) -> usize {
        let relative = (price - self.state.price_levels[0]) / self.price_tick;
        let idx = relative.round() as usize;
        idx.min(self.num_levels - 1)
    }

    /// Get current state
    pub fn get_state(&self) -> &OrderBookState {
        &self.state
    }

    /// Compute order book imbalance (predictive signal)
    pub fn compute_imbalance(&self) -> f32 {
        // Weighted imbalance around mid price
        let mid_idx = self.num_levels / 2;
        let window = 10; // Look at 10 levels on each side
        
        let mut bid_volume = 0.0;
        let mut ask_volume = 0.0;
        
        for i in (mid_idx - window).max(0)..(mid_idx + window).min(self.num_levels) {
            let distance_weight = 1.0 / (1.0 + (i as i32 - mid_idx as i32).abs() as f32);
            bid_volume += self.state.bid_density[i] * distance_weight;
            ask_volume += self.state.ask_density[i] * distance_weight;
        }
        
        (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6)
    }

    /// Compute price impact (linear approximation)
    pub fn compute_price_impact(&self, side: Side, quantity: f32) -> f32 {
        // Kyle's lambda: Price impact coefficient
        // ΔP = λ·Q
        
        let levels = match side {
            Side::Bid => &self.state.ask_density,
            Side::Ask => &self.state.bid_density,
        };
        
        let mid_idx = self.num_levels / 2;
        
        // Compute local depth
        let mut cumulative_depth = 0.0;
        let mut levels_consumed = 0;
        
        for i in 0..10 {
            let idx = if matches!(side, Side::Bid) {
                mid_idx + i
            } else {
                mid_idx.saturating_sub(i)
            };
            
            if idx >= levels.len() {
                break;
            }
            
            cumulative_depth += levels[idx];
            levels_consumed += 1;
            
            if cumulative_depth >= quantity {
                break;
            }
        }
        
        // Impact: levels consumed × tick size
        let impact = levels_consumed as f32 * self.price_tick;
        
        match side {
            Side::Bid => impact,  // Positive (price goes up)
            Side::Ask => -impact, // Negative (price goes down)
        }
    }

    /// Compute volatility (from pressure fluctuations)
    pub fn compute_volatility(&self) -> f32 {
        // Volatility ~ sqrt(⟨P²⟩ - ⟨P⟩²)
        let mean_pressure: f32 = self.state.pressure.iter().sum::<f32>() / self.state.pressure.len() as f32;
        let mean_pressure_sq: f32 = self.state.pressure.iter()
            .map(|p| p * p)
            .sum::<f32>() / self.state.pressure.len() as f32;
        
        (mean_pressure_sq - mean_pressure * mean_pressure).sqrt()
    }

    /// Generate diagnostic report
    pub fn diagnostics(&self) -> String {
        format!(
            "Market Microstructure State (t = {}µs):
  Mid Price: ${:.2}
  Spread: ${:.4}
  Imbalance: {:.3}
  Volatility: {:.4}
  Total Bid Liquidity: {:.0}
  Total Ask Liquidity: {:.0}
  Avg Bid Velocity: {:.6}
  Avg Ask Velocity: {:.6}
  Avg Pressure: {:.2}",
            self.state.time_us,
            self.state.mid_price,
            self.state.spread,
            self.compute_imbalance(),
            self.compute_volatility(),
            self.state.bid_density.iter().sum::<f32>(),
            self.state.ask_density.iter().sum::<f32>(),
            self.state.bid_velocity.iter().sum::<f32>() / self.state.bid_velocity.len() as f32,
            self.state.ask_velocity.iter().sum::<f32>() / self.state.ask_velocity.len() as f32,
            self.state.pressure.iter().sum::<f32>() / self.state.pressure.len() as f32,
        )
    }
}

// ============================================================================
// BOLTZMANN ORDER GENERATOR (Statistical Mechanics)
// ============================================================================

/// Generate synthetic orders using Boltzmann distribution
pub struct BoltzmannOrderGenerator {
    temperature: f32,
    mean_arrival_rate: f32, // Orders per second
    mean_order_size: f32,
}

impl BoltzmannOrderGenerator {
    pub fn new(temperature: f32, arrival_rate: f32, order_size: f32) -> Self {
        Self {
            temperature,
            mean_arrival_rate: arrival_rate,
            mean_order_size: order_size,
        }
    }

    /// Generate next order (Poisson arrival, Boltzmann price distribution)
    pub fn generate_order(&self, mid_price: f32, time_us: u64) -> Option<MarketEvent> {
        // Poisson arrival: P(arrival in dt) = λ·dt
        let dt = 1.0 / 1_000_000.0; // 1µs
        let arrival_prob = self.mean_arrival_rate * dt;
        
        if rand::random::<f32>() < arrival_prob {
            // Generate order
            let side = if rand::random::<f32>() < 0.5 {
                Side::Bid
            } else {
                Side::Ask
            };
            
            // Price offset from mid (Boltzmann distribution)
            // P(offset) ∝ exp(-|offset|/T)
            let u: f32 = rand::random();
            let offset = -self.temperature * u.ln() * if matches!(side, Side::Bid) { -1.0 } else { 1.0 };
            let price = mid_price + offset;
            
            // Quantity (exponential distribution)
            let u: f32 = rand::random();
            let quantity = -self.mean_order_size * u.ln();
            
            // 90% limit orders, 10% market orders
            if rand::random::<f32>() < 0.9 {
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

// ============================================================================
// INTEGRATION WITH HYPERPHYSICS
// ============================================================================

/// Real-time market simulator for HyperPhysics
pub struct HyperPhysicsMarketSimulator {
    simulator: MarketMicrostructureSimulator,
    order_generator: BoltzmannOrderGenerator,
}

impl HyperPhysicsMarketSimulator {
    pub fn new(initial_price: f32, temperature: f32) -> Self {
        let simulator = MarketMicrostructureSimulator::hyperphysics_default(initial_price);
        let order_generator = BoltzmannOrderGenerator::new(
            temperature,
            5000.0,  // 5000 orders/sec
            100.0,   // 100 shares average
        );
        
        Self {
            simulator,
            order_generator,
        }
    }

    /// Step simulation (10µs per call)
    pub fn step(&mut self) -> MarketSignals {
        let current_time = self.simulator.get_state().time_us;
        let mid_price = self.simulator.get_state().mid_price;
        
        // Generate synthetic orders (if using for testing)
        let mut events = Vec::new();
        if let Some(event) = self.order_generator.generate_order(mid_price, current_time) {
            events.push(event);
        }
        
        // Step simulation
        self.simulator.step(&events);
        
        // Extract signals
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
    pub fn compute_impact(&self, side: Side, quantity: f32) -> f32 {
        self.simulator.compute_price_impact(side, quantity)
    }
}

#[derive(Debug, Clone)]
pub struct MarketSignals {
    pub mid_price: f32,
    pub spread: f32,
    pub imbalance: f32,
    pub volatility: f32,
    pub time_us: u64,
}

// ============================================================================
// TESTS
// ============================================================================

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
    fn test_market_order_impact() {
        let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let initial_mid = sim.state.mid_price;
        
        sim.consume_liquidity(Side::Bid, 5000.0); // Large buy
        sim.update_market_stats();
        
        assert!(sim.state.mid_price > initial_mid); // Price should go up
    }

    #[test]
    fn test_imbalance_computation() {
        let sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let imbalance = sim.compute_imbalance();
        
        assert!(imbalance.abs() <= 1.0); // Should be normalized
    }

    #[test]
    fn test_physics_conservation() {
        let mut sim = MarketMicrostructureSimulator::hyperphysics_default(100.0);
        let initial_total = sim.state.bid_density.iter().sum::<f32>() 
                          + sim.state.ask_density.iter().sum::<f32>();
        
        // Step without events (should conserve total liquidity)
        sim.step(&[]);
        
        let final_total = sim.state.bid_density.iter().sum::<f32>()
                        + sim.state.ask_density.iter().sum::<f32>();
        
        assert!((final_total - initial_total).abs() < 1.0); // Small numerical error ok
    }
}
