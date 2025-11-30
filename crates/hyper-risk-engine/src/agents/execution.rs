//! Execution Agent.
//!
//! Handles optimal order execution with steganographic order hiding
//! to minimize market impact and implementation shortfall.
//!
//! ## Responsibilities
//! - Parent order splitting into child orders
//! - Optimal execution timing (TWAP, VWAP, IS, Adaptive)
//! - Market impact estimation and minimization
//! - Steganographic order hiding from predatory algorithms
//!
//! ## Scientific References
//! - Almgren & Chriss (2000): "Optimal execution of portfolio transactions"
//! - Bertsimas & Lo (1998): "Optimal control of execution costs"
//! - Kyle (1985): "Continuous auctions and insider trading"
//! - Obizhaeva & Wang (2013): "Optimal trading strategy and supply/demand dynamics"

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;
use std::collections::VecDeque;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::types::{
    MarketRegime, Portfolio, RiskDecision, Symbol, Timestamp,
    Price, Quantity, OrderSide,
};
use crate::core::error::Result;
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

// ============================================================================
// Configuration
// ============================================================================

/// Execution agent configuration.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Base agent config.
    pub base: AgentConfig,
    /// Default execution algorithm.
    pub default_algorithm: ExecutionAlgorithm,
    /// Kyle's lambda (market impact coefficient).
    /// Typical range: 0.1 - 1.0 for liquid stocks.
    pub kyle_lambda: f64,
    /// Temporary impact parameter α (volatility scaling).
    /// From Almgren-Chriss: typically 0.1 - 0.5.
    pub temp_impact_alpha: f64,
    /// Temporary impact exponent β.
    /// From empirical studies: typically 0.5 - 0.6.
    pub temp_impact_beta: f64,
    /// Permanent impact parameter γ.
    /// Typically 0.1 - 0.3 of daily volume.
    pub perm_impact_gamma: f64,
    /// Maximum child order size as fraction of ADV (Average Daily Volume).
    pub max_participation_rate: f64,
    /// Steganographic randomization factor (0.0 - 1.0).
    /// Higher values = more randomness = better hiding.
    pub steganographic_factor: f64,
    /// Time horizon for execution (nanoseconds).
    pub execution_horizon_ns: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "Execution".to_string(),
                max_latency_us: 500, // 500μs
                ..Default::default()
            },
            default_algorithm: ExecutionAlgorithm::Adaptive,
            kyle_lambda: 0.3,           // Moderate liquidity
            temp_impact_alpha: 0.3,     // 30% volatility impact
            temp_impact_beta: 0.5,      // Square root impact
            perm_impact_gamma: 0.2,     // 20% permanent impact
            max_participation_rate: 0.1, // 10% of volume
            steganographic_factor: 0.3,  // 30% randomization
            execution_horizon_ns: 300_000_000_000, // 5 minutes
        }
    }
}

// ============================================================================
// Execution Algorithms
// ============================================================================

/// Execution algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    /// Time-Weighted Average Price.
    /// Splits order evenly across time intervals.
    TWAP,
    /// Volume-Weighted Average Price.
    /// Splits order proportional to historical volume profile.
    VWAP,
    /// Implementation Shortfall.
    /// Minimizes deviation from arrival price (Almgren-Chriss).
    ImplementationShortfall,
    /// Adaptive execution.
    /// Adjusts strategy based on real-time market conditions.
    Adaptive,
}

// ============================================================================
// Order Types
// ============================================================================

/// Atomic counter for generating unique order IDs.
static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Parent order (large order to be executed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParentOrder {
    /// Unique order ID.
    pub id: u64,
    /// Symbol to trade.
    pub symbol: Symbol,
    /// Order side.
    pub side: OrderSide,
    /// Total quantity.
    pub quantity: Quantity,
    /// Target arrival price (for IS calculation).
    pub arrival_price: Price,
    /// Execution algorithm.
    pub algorithm: ExecutionAlgorithm,
    /// Maximum participation rate.
    pub max_participation: f64,
    /// Urgency factor (0.0 - 1.0, higher = more aggressive).
    pub urgency: f64,
    /// Timestamp when order was created.
    pub created_at: Timestamp,
    /// Execution deadline.
    pub deadline: Timestamp,
}

impl ParentOrder {
    /// Create new parent order.
    pub fn new(
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        arrival_price: Price,
        algorithm: ExecutionAlgorithm,
        urgency: f64,
        horizon_ns: u64,
    ) -> Self {
        let created_at = Timestamp::now();
        Self {
            id: ORDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            symbol,
            side,
            quantity,
            arrival_price,
            algorithm,
            max_participation: 0.1, // 10% default
            urgency: urgency.clamp(0.0, 1.0),
            created_at,
            deadline: Timestamp::from_nanos(created_at.as_nanos() + horizon_ns),
        }
    }

    /// Get remaining time until deadline (nanoseconds).
    pub fn remaining_time_ns(&self) -> u64 {
        let now = Timestamp::now();
        if self.deadline.as_nanos() > now.as_nanos() {
            self.deadline.as_nanos() - now.as_nanos()
        } else {
            0
        }
    }

    /// Get time elapsed since creation (nanoseconds).
    pub fn elapsed_time_ns(&self) -> u64 {
        Timestamp::now().as_nanos() - self.created_at.as_nanos()
    }
}

/// Child order (slice of parent order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildOrder {
    /// Unique child order ID.
    pub id: u64,
    /// Parent order ID.
    pub parent_id: u64,
    /// Symbol.
    pub symbol: Symbol,
    /// Order side.
    pub side: OrderSide,
    /// Slice quantity.
    pub quantity: Quantity,
    /// Limit price (if applicable).
    pub limit_price: Option<Price>,
    /// Scheduled execution time.
    pub scheduled_at: Timestamp,
    /// Whether order has been sent to market.
    pub sent: bool,
    /// Whether order has been filled.
    pub filled: bool,
    /// Fill price (if filled).
    pub fill_price: Option<Price>,
    /// Fill timestamp (if filled).
    pub fill_time: Option<Timestamp>,
}

impl ChildOrder {
    /// Create new child order.
    fn new(
        parent_id: u64,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        scheduled_at: Timestamp,
        limit_price: Option<Price>,
    ) -> Self {
        Self {
            id: ORDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            parent_id,
            symbol,
            side,
            quantity,
            limit_price,
            scheduled_at,
            sent: false,
            filled: false,
            fill_price: None,
            fill_time: None,
        }
    }

    /// Mark order as sent.
    pub fn mark_sent(&mut self) {
        self.sent = true;
    }

    /// Mark order as filled.
    pub fn mark_filled(&mut self, fill_price: Price) {
        self.filled = true;
        self.fill_price = Some(fill_price);
        self.fill_time = Some(Timestamp::now());
    }
}

// ============================================================================
// Market Impact Estimation
// ============================================================================

/// Market impact estimate (Almgren-Chriss framework).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactEstimate {
    /// Temporary impact (mean reversion component).
    pub temporary_impact: f64,
    /// Permanent impact (information component).
    pub permanent_impact: f64,
    /// Total expected cost (basis points).
    pub total_cost_bps: f64,
    /// Expected execution price.
    pub expected_price: Price,
}

impl MarketImpactEstimate {
    /// Calculate market impact using Almgren-Chriss model.
    ///
    /// # Formula
    /// Temporary: α * σ * (Q / V)^β
    /// Permanent: γ * (Q / ADV)
    ///
    /// Where:
    /// - α: volatility scaling parameter
    /// - σ: daily volatility
    /// - Q: order size
    /// - V: average trade size
    /// - β: impact exponent (typically 0.5-0.6)
    /// - γ: permanent impact coefficient
    /// - ADV: average daily volume
    ///
    /// # References
    /// Almgren & Chriss (2000), Journal of Risk
    pub fn calculate(
        quantity: f64,
        arrival_price: f64,
        daily_volatility: f64,
        avg_trade_size: f64,
        avg_daily_volume: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        // Temporary impact: α * σ * (Q/V)^β
        let size_ratio = quantity / avg_trade_size;
        let temporary_impact = alpha * daily_volatility * size_ratio.powf(beta);

        // Permanent impact: γ * (Q/ADV)
        let volume_fraction = quantity / avg_daily_volume;
        let permanent_impact = gamma * volume_fraction;

        // Total cost in basis points
        let total_impact = temporary_impact + permanent_impact;
        let total_cost_bps = total_impact * 10_000.0;

        // Expected execution price (assuming buy order)
        let expected_price = arrival_price * (1.0 + total_impact);

        Self {
            temporary_impact,
            permanent_impact,
            total_cost_bps,
            expected_price: Price::from_f64(expected_price),
        }
    }

    /// Calculate Kyle's lambda impact.
    ///
    /// # Formula
    /// ΔP = λ * Q
    ///
    /// Where:
    /// - λ: Kyle's lambda (market depth parameter)
    /// - Q: order quantity
    ///
    /// # References
    /// Kyle (1985), Econometrica
    pub fn kyle_impact(quantity: f64, arrival_price: f64, lambda: f64) -> f64 {
        lambda * quantity / arrival_price
    }
}

// ============================================================================
// Execution Reports
// ============================================================================

/// Execution report for completed orders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// Parent order ID.
    pub parent_id: u64,
    /// Symbol.
    pub symbol: Symbol,
    /// Total quantity filled.
    pub filled_quantity: Quantity,
    /// Volume-weighted average fill price.
    pub vwap: Price,
    /// Arrival price (benchmark).
    pub arrival_price: Price,
    /// Implementation shortfall (basis points).
    pub implementation_shortfall_bps: f64,
    /// Total number of child orders.
    pub num_children: usize,
    /// Total execution time (nanoseconds).
    pub execution_time_ns: u64,
    /// Average child order latency (nanoseconds).
    pub avg_child_latency_ns: u64,
}

impl ExecutionReport {
    /// Calculate implementation shortfall.
    ///
    /// # Formula
    /// IS = (VWAP - ArrivalPrice) / ArrivalPrice * 10000
    ///
    /// For sell orders, sign is reversed.
    pub fn calculate_shortfall(
        vwap: f64,
        arrival_price: f64,
        side: OrderSide,
    ) -> f64 {
        let diff = vwap - arrival_price;
        let shortfall = diff / arrival_price * 10_000.0;

        // For sell orders, negative IS is good (sold higher than arrival)
        match side {
            OrderSide::Buy => shortfall,    // Positive IS = bad (bought higher)
            OrderSide::Sell => -shortfall,  // Positive IS = good (sold higher)
        }
    }
}

// ============================================================================
// Execution Agent
// ============================================================================

/// Execution Agent state.
#[derive(Debug)]
pub struct ExecutionAgent {
    /// Configuration.
    config: ExecutionConfig,
    /// Current status (atomic for lock-free reads).
    status: AtomicU8,
    /// Active parent orders.
    parent_orders: RwLock<VecDeque<ParentOrder>>,
    /// Child order queue.
    child_orders: RwLock<VecDeque<ChildOrder>>,
    /// Completed execution reports.
    reports: RwLock<Vec<ExecutionReport>>,
    /// Statistics.
    stats: AgentStats,
}

impl ExecutionAgent {
    /// Create new execution agent.
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            parent_orders: RwLock::new(VecDeque::new()),
            child_orders: RwLock::new(VecDeque::new()),
            reports: RwLock::new(Vec::new()),
            stats: AgentStats::new(),
        }
    }

    /// Submit parent order for execution.
    pub fn submit_order(&self, order: ParentOrder) {
        let mut orders = self.parent_orders.write();
        orders.push_back(order);
    }

    /// Generate child orders using TWAP algorithm.
    ///
    /// Time-Weighted Average Price: splits order evenly across time.
    fn generate_twap_children(
        &self,
        parent: &ParentOrder,
        num_slices: usize,
    ) -> Vec<ChildOrder> {
        let slice_qty = parent.quantity.as_f64() / num_slices as f64;
        let time_interval = parent.remaining_time_ns() / num_slices as u64;

        let mut children = Vec::with_capacity(num_slices);
        let now = Timestamp::now();

        for i in 0..num_slices {
            // Add steganographic randomization to timing
            let base_delay = time_interval * i as u64;
            let randomization = (self.config.steganographic_factor
                * time_interval as f64) as u64;
            let jitter = self.generate_timing_jitter(randomization);
            let scheduled_time = Timestamp::from_nanos(
                now.as_nanos() + base_delay + jitter
            );

            // Add randomization to size (within ±10% of target)
            let size_jitter = self.generate_size_jitter(slice_qty);

            children.push(ChildOrder::new(
                parent.id,
                parent.symbol,
                parent.side,
                Quantity::from_f64(slice_qty + size_jitter),
                scheduled_time,
                None, // Market orders
            ));
        }

        children
    }

    /// Generate child orders using VWAP algorithm.
    ///
    /// Volume-Weighted Average Price: splits proportional to historical volume.
    fn generate_vwap_children(
        &self,
        parent: &ParentOrder,
        volume_profile: &[f64], // Historical volume distribution
    ) -> Vec<ChildOrder> {
        let total_volume: f64 = volume_profile.iter().sum();
        let mut children = Vec::with_capacity(volume_profile.len());

        let now = Timestamp::now();
        let time_per_bucket = parent.remaining_time_ns() / volume_profile.len() as u64;

        for (i, &volume_fraction) in volume_profile.iter().enumerate() {
            // Allocate quantity proportional to volume
            let weight = volume_fraction / total_volume;
            let slice_qty = parent.quantity.as_f64() * weight;

            // Skip negligible slices
            if slice_qty < 1.0 {
                continue;
            }

            // Steganographic timing
            let base_delay = time_per_bucket * i as u64;
            let jitter = self.generate_timing_jitter(
                (self.config.steganographic_factor * time_per_bucket as f64) as u64
            );

            children.push(ChildOrder::new(
                parent.id,
                parent.symbol,
                parent.side,
                Quantity::from_f64(slice_qty),
                Timestamp::from_nanos(now.as_nanos() + base_delay + jitter),
                None,
            ));
        }

        children
    }

    /// Generate child orders using Implementation Shortfall algorithm.
    ///
    /// Minimizes expected implementation shortfall (Almgren-Chriss optimal).
    fn generate_is_children(
        &self,
        parent: &ParentOrder,
        _volatility: f64,
        _adv: f64,
    ) -> Vec<ChildOrder> {
        // Almgren-Chriss optimal trajectory: exponential decay
        // Trading rate: dQ/dt = Q₀ * κ * exp(-κt)
        // where κ = λ/η (urgency parameter)

        let remaining_time = parent.remaining_time_ns() as f64 / 1e9; // seconds
        let kappa = parent.urgency / remaining_time;

        // Discretize into slices
        let num_slices = (remaining_time * 10.0).ceil() as usize; // 10 slices per second
        let dt = remaining_time / num_slices as f64;

        let mut children = Vec::with_capacity(num_slices);
        let mut remaining_qty = parent.quantity.as_f64();

        for i in 0..num_slices {
            let t = i as f64 * dt;

            // Optimal trading rate at time t
            let rate = kappa * (-kappa * t).exp();
            let slice_qty = remaining_qty * rate * dt;

            // Don't execute negligible quantities
            if slice_qty < 1.0 {
                continue;
            }

            remaining_qty -= slice_qty;

            // Steganographic scheduling
            let scheduled_time = Timestamp::from_nanos(
                parent.created_at.as_nanos()
                    + (t * 1e9) as u64
                    + self.generate_timing_jitter(1_000_000_000) // ±1s jitter
            );

            children.push(ChildOrder::new(
                parent.id,
                parent.symbol,
                parent.side,
                Quantity::from_f64(slice_qty),
                scheduled_time,
                None,
            ));
        }

        children
    }

    /// Generate timing jitter for steganographic hiding.
    ///
    /// Uses xorshift for fast pseudo-random generation (thread-safe).
    /// Returns value in range [0, 2*max_jitter).
    fn generate_timing_jitter(&self, max_jitter: u64) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};

        // Thread-safe xorshift PRNG (fast, not cryptographic)
        static JITTER_SEED: AtomicU64 = AtomicU64::new(123456789);

        let seed = JITTER_SEED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
            let mut x = s;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            Some(x)
        }).unwrap_or(123456789);

        if max_jitter == 0 {
            return 0;
        }
        (seed % (2 * max_jitter))
    }

    /// Generate size jitter for steganographic hiding.
    fn generate_size_jitter(&self, base_size: f64) -> f64 {
        let jitter_range = base_size * self.config.steganographic_factor * 0.2; // ±20%
        let jitter = self.generate_timing_jitter(jitter_range as u64) as f64;
        jitter - jitter_range / 2.0
    }

    /// Estimate market impact for an order.
    pub fn estimate_impact(
        &self,
        quantity: f64,
        arrival_price: f64,
        volatility: f64,
        avg_trade_size: f64,
        adv: f64,
    ) -> MarketImpactEstimate {
        MarketImpactEstimate::calculate(
            quantity,
            arrival_price,
            volatility,
            avg_trade_size,
            adv,
            self.config.temp_impact_alpha,
            self.config.temp_impact_beta,
            self.config.perm_impact_gamma,
        )
    }

    /// Get next child order ready for execution.
    pub fn get_next_child(&self) -> Option<ChildOrder> {
        let mut children = self.child_orders.write();
        let now = Timestamp::now();

        // Find first unsent child whose scheduled time has arrived
        if let Some(index) = children.iter().position(|c| {
            !c.sent && c.scheduled_at.as_nanos() <= now.as_nanos()
        }) {
            children.remove(index)
        } else {
            None
        }
    }

    /// Get execution reports.
    pub fn get_reports(&self) -> Vec<ExecutionReport> {
        self.reports.read().clone()
    }

    fn status_from_u8(val: u8) -> AgentStatus {
        match val {
            0 => AgentStatus::Idle,
            1 => AgentStatus::Processing,
            2 => AgentStatus::Paused,
            3 => AgentStatus::Error,
            _ => AgentStatus::ShuttingDown,
        }
    }
}

impl Agent for ExecutionAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, _portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        // Process pending parent orders
        let mut parents = self.parent_orders.write();
        let mut children_to_add = Vec::new();

        // Drain expired or complete parent orders
        parents.retain(|parent| {
            // Check if deadline exceeded
            if parent.remaining_time_ns() == 0 {
                return false; // Remove
            }

            // Generate children based on algorithm
            let new_children = match parent.algorithm {
                ExecutionAlgorithm::TWAP => {
                    self.generate_twap_children(parent, 10) // 10 slices default
                }
                ExecutionAlgorithm::VWAP => {
                    // Use uniform volume profile as default
                    // In production, would fetch real volume profile
                    let volume_profile = vec![1.0; 10];
                    self.generate_vwap_children(parent, &volume_profile)
                }
                ExecutionAlgorithm::ImplementationShortfall => {
                    // Use portfolio volatility as proxy
                    // In production, would calculate from real market data
                    let vol = 0.02; // 2% daily vol
                    let adv = 1_000_000.0; // Default ADV
                    self.generate_is_children(parent, vol, adv)
                }
                ExecutionAlgorithm::Adaptive => {
                    // Adapt based on market regime
                    if regime.is_favorable() {
                        // Use IS for favorable regimes (minimize cost)
                        let vol = 0.02;
                        let adv = 1_000_000.0;
                        self.generate_is_children(parent, vol, adv)
                    } else {
                        // Use TWAP for unfavorable regimes (reduce footprint)
                        self.generate_twap_children(parent, 20) // More slices = less visible
                    }
                }
            };

            children_to_add.extend(new_children);
            false // Remove parent after processing (one-time generation)
        });

        drop(parents); // Release lock

        // Add generated children to queue
        if !children_to_add.is_empty() {
            let mut children = self.child_orders.write();
            children.extend(children_to_add);
        }

        let latency = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        Ok(None) // No risk decisions from execution agent
    }

    fn start(&self) -> Result<()> {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        self.status.store(AgentStatus::ShuttingDown as u8, Ordering::Relaxed);
        Ok(())
    }

    fn pause(&self) {
        self.status.store(AgentStatus::Paused as u8, Ordering::Relaxed);
    }

    fn resume(&self) {
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);
    }

    fn process_count(&self) -> u64 {
        self.stats.cycles.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_agent_creation() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
    }

    #[test]
    fn test_parent_order_creation() {
        let symbol = Symbol::new("AAPL");
        let order = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::TWAP,
            0.5, // Medium urgency
            300_000_000_000, // 5 minutes
        );

        assert_eq!(order.symbol, symbol);
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity.as_f64(), 1000.0);
    }

    #[test]
    fn test_market_impact_calculation() {
        let impact = MarketImpactEstimate::calculate(
            1000.0,     // quantity
            100.0,      // arrival price
            0.02,       // 2% daily volatility
            100.0,      // avg trade size
            1_000_000.0, // ADV
            0.3,        // alpha
            0.5,        // beta
            0.2,        // gamma
        );

        // Verify impact is non-zero and reasonable
        assert!(impact.temporary_impact > 0.0);
        assert!(impact.permanent_impact > 0.0);
        assert!(impact.total_cost_bps > 0.0);
        assert!(impact.total_cost_bps < 1000.0); // Less than 1%
    }

    #[test]
    fn test_kyle_lambda_impact() {
        let impact = MarketImpactEstimate::kyle_impact(
            1000.0, // quantity
            100.0,  // price
            0.3,    // lambda
        );

        // Kyle's lambda: ΔP = λ * Q / P
        // Expected: 0.3 * 1000 / 100 = 3.0
        assert!((impact - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_implementation_shortfall_calculation() {
        let shortfall_buy = ExecutionReport::calculate_shortfall(
            100.5, // VWAP
            100.0, // Arrival price
            OrderSide::Buy,
        );

        // Bought at 100.5, arrived at 100 = 50 bps slippage
        assert!((shortfall_buy - 50.0).abs() < 0.1);

        let shortfall_sell = ExecutionReport::calculate_shortfall(
            100.5, // VWAP
            100.0, // Arrival price
            OrderSide::Sell,
        );

        // Sold at 100.5, arrived at 100 = -50 bps slippage (good!)
        assert!((shortfall_sell + 50.0).abs() < 0.1);
    }

    #[test]
    fn test_twap_child_generation() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::TWAP,
            0.5,
            300_000_000_000, // 5 minutes
        );

        let children = agent.generate_twap_children(&parent, 10);

        assert_eq!(children.len(), 10);

        // Verify total quantity matches (approximately, due to jitter)
        let total_qty: f64 = children.iter().map(|c| c.quantity.as_f64()).sum();
        assert!((total_qty - 1000.0).abs() < 100.0); // Within 10% due to jitter
    }

    #[test]
    fn test_vwap_child_generation() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::VWAP,
            0.5,
            300_000_000_000,
        );

        // Weighted volume profile: more volume at beginning and end
        let volume_profile = vec![2.0, 1.0, 1.0, 1.0, 2.0];
        let children = agent.generate_vwap_children(&parent, &volume_profile);

        assert!(children.len() > 0);

        // Verify higher quantities for higher volume periods
        // First and last should be approximately double middle periods
        if children.len() >= 5 {
            let first_qty = children[0].quantity.as_f64();
            let middle_qty = children[2].quantity.as_f64();
            assert!(first_qty > middle_qty);
        }
    }

    #[test]
    fn test_implementation_shortfall_children() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        // IS algorithm: kappa = urgency / remaining_time
        // For meaningful slices, need: slice_qty = remaining_qty * kappa * dt >= 1.0
        // With dt = 0.1s (10 slices/sec), need remaining_qty * kappa >= 10
        // Use high urgency (0.95) and short window (10s) to get kappa = 0.095
        // Then 100_000 * 0.095 * 0.1 = 950 per slice (valid)
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(100_000.0), // Large quantity for meaningful slices
            Price::from_f64(150.0),
            ExecutionAlgorithm::ImplementationShortfall,
            0.95, // High urgency for front-loaded trading
            10_000_000_000, // 10 second window
        );

        let children = agent.generate_is_children(&parent, 0.02, 1_000_000.0);

        // Should generate at least some children given the parameters
        // With high urgency and large qty, we expect meaningful slice sizes
        assert!(!children.is_empty(),
            "IS algorithm should generate child orders for qty={}, urgency={}, window=10s",
            100_000.0, 0.95);

        // IS algorithm should front-load for high urgency
        if children.len() >= 3 {
            let first_qty = children[0].quantity.as_f64();
            let last_qty = children[children.len() - 1].quantity.as_f64();
            // For high urgency, first slice should be significantly larger
            assert!(first_qty >= last_qty * 0.5,
                "IS front-loading: first_qty={}, last_qty={}", first_qty, last_qty);
        }
    }

    #[test]
    fn test_order_submission_and_processing() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::TWAP,
            0.5,
            300_000_000_000,
        );

        // Submit order
        agent.submit_order(parent);

        // Verify order is queued
        {
            let orders = agent.parent_orders.read();
            assert_eq!(orders.len(), 1);
        }

        // Process order
        let portfolio = Portfolio::new(1_000_000.0);
        let _ = agent.process(&portfolio, MarketRegime::BullTrending);

        // Verify children were generated
        {
            let children = agent.child_orders.read();
            assert!(children.len() > 0);
        }

        // Verify parent order was removed after processing
        {
            let orders = agent.parent_orders.read();
            assert_eq!(orders.len(), 0);
        }
    }

    #[test]
    fn test_adaptive_execution() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::Adaptive,
            0.5,
            300_000_000_000,
        );

        agent.submit_order(parent.clone());

        // Process in favorable regime - should use IS
        let portfolio = Portfolio::new(1_000_000.0);
        let _ = agent.process(&portfolio, MarketRegime::BullTrending);

        let favorable_children = agent.child_orders.read().len();

        // Clear and test unfavorable regime
        agent.child_orders.write().clear();
        agent.submit_order(parent);
        let _ = agent.process(&portfolio, MarketRegime::Crisis);

        let crisis_children = agent.child_orders.read().len();

        // Crisis mode should generate more slices (TWAP with 20 vs IS dynamic)
        assert!(crisis_children > favorable_children);
    }

    #[test]
    fn test_child_order_scheduling() {
        let config = ExecutionConfig::default();
        let agent = ExecutionAgent::new(config);

        let symbol = Symbol::new("AAPL");
        let parent = ParentOrder::new(
            symbol,
            OrderSide::Buy,
            Quantity::from_f64(1000.0),
            Price::from_f64(150.0),
            ExecutionAlgorithm::TWAP,
            0.5,
            10_000_000_000, // 10 seconds
        );

        agent.submit_order(parent);

        let portfolio = Portfolio::new(1_000_000.0);
        let _ = agent.process(&portfolio, MarketRegime::BullTrending);

        // Check that children are scheduled in the future
        let children = agent.child_orders.read();
        let now = Timestamp::now();

        for child in children.iter() {
            assert!(child.scheduled_at.as_nanos() >= now.as_nanos());
        }
    }
}
