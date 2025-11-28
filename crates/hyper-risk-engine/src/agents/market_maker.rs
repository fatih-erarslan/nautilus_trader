//! Market Maker Agent - Avellaneda-Stoikov Optimal Market Making
//!
//! Implements the optimal market making strategy from:
//! Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book."
//! Quantitative Finance, 8(3), 217-224.
//!
//! ## Model Overview
//!
//! The Avellaneda-Stoikov model provides optimal bid-ask spread and reservation price
//! for a market maker managing inventory risk and adverse selection.
//!
//! ### Key Formulas:
//!
//! 1. **Reservation Price** (indifference price):
//!    ```text
//!    r(s, q, t) = s - q * γ * σ² * (T - t)
//!    ```
//!    where:
//!    - s = mid-price
//!    - q = inventory position
//!    - γ = risk aversion parameter
//!    - σ = volatility
//!    - T - t = time to horizon
//!
//! 2. **Optimal Spread**:
//!    ```text
//!    δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
//!    ```
//!    where k = order arrival intensity
//!
//! 3. **Optimal Quotes**:
//!    ```text
//!    bid = r - δ/2
//!    ask = r + δ/2
//!    ```
//!
//! ## Adverse Selection Protection
//!
//! The agent monitors toxicity of order flow by tracking:
//! - Fill rate asymmetry (more fills on one side indicates toxic flow)
//! - Post-trade price movement (adverse selection indicator)
//! - Order book imbalance
//!
//! Spreads are widened dynamically when toxic flow is detected.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::types::{MarketRegime, Portfolio, RiskDecision, RiskLevel, Symbol, Timestamp, OrderSide, Price, Quantity, Position, PositionId};
use crate::core::error::Result;
use super::base::{Agent, AgentId, AgentStatus, AgentConfig, AgentStats};

// ============================================================================
// Core Types
// ============================================================================

/// Two-sided quote with bid and ask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    /// Bid price (market maker buys at this price).
    pub bid_price: f64,
    /// Bid size (shares).
    pub bid_size: f64,
    /// Ask price (market maker sells at this price).
    pub ask_price: f64,
    /// Ask size (shares).
    pub ask_size: f64,
    /// Quote generation timestamp.
    pub timestamp: Timestamp,
    /// Mid-price used for quote generation.
    pub mid_price: f64,
    /// Spread (ask - bid).
    pub spread: f64,
}

impl Quote {
    /// Calculate the mid-point of the quote.
    #[inline]
    pub fn mid(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    /// Calculate the spread as a percentage of mid.
    #[inline]
    pub fn spread_bps(&self) -> f64 {
        if self.mid_price > 0.0 {
            (self.spread / self.mid_price) * 10_000.0
        } else {
            0.0
        }
    }
}

/// Inventory state for a specific symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryState {
    /// Current position (positive = long, negative = short).
    pub position: f64,
    /// Average cost basis.
    pub avg_cost: f64,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Total volume traded.
    pub volume_traded: f64,
    /// Number of fills on bid side.
    pub bid_fills: u64,
    /// Number of fills on ask side.
    pub ask_fills: u64,
    /// Last update timestamp.
    pub updated_at: Timestamp,
}

impl Default for InventoryState {
    fn default() -> Self {
        Self {
            position: 0.0,
            avg_cost: 0.0,
            unrealized_pnl: 0.0,
            volume_traded: 0.0,
            bid_fills: 0,
            ask_fills: 0,
            updated_at: Timestamp::from_nanos(0),
        }
    }
}

impl InventoryState {
    /// Check if position is within limits.
    #[inline]
    pub fn is_within_limits(&self, max_inventory: f64) -> bool {
        self.position.abs() <= max_inventory
    }

    /// Calculate inventory skew (-1 = max short, 0 = neutral, +1 = max long).
    #[inline]
    pub fn inventory_skew(&self, max_inventory: f64) -> f64 {
        if max_inventory > 0.0 {
            (self.position / max_inventory).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Calculate fill rate imbalance (positive = more ask fills, negative = more bid fills).
    #[inline]
    pub fn fill_rate_imbalance(&self) -> f64 {
        let total_fills = self.bid_fills + self.ask_fills;
        if total_fills > 0 {
            (self.ask_fills as f64 - self.bid_fills as f64) / total_fills as f64
        } else {
            0.0
        }
    }
}

/// Toxicity score for adverse selection detection.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ToxicityScore(f64);

impl ToxicityScore {
    /// Create new toxicity score (clamped to [0.0, 1.0]).
    pub fn new(score: f64) -> Self {
        Self(score.clamp(0.0, 1.0))
    }

    /// Get score value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if flow is toxic (score > 0.5).
    #[inline]
    pub fn is_toxic(&self) -> bool {
        self.0 > 0.5
    }

    /// Get spread multiplier based on toxicity (1.0 to 3.0).
    #[inline]
    pub fn spread_multiplier(&self) -> f64 {
        1.0 + (self.0 * 2.0) // Linear scaling from 1.0 to 3.0
    }
}

impl Default for ToxicityScore {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Market maker configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakerConfig {
    /// Base agent config.
    pub base: AgentConfig,
    /// Risk aversion parameter γ (gamma).
    /// Higher values = more conservative, wider spreads.
    /// Typical range: 0.01 to 0.5
    pub gamma: f64,
    /// Maximum inventory position per symbol.
    pub max_inventory: f64,
    /// Minimum spread in basis points.
    pub min_spread_bps: f64,
    /// Maximum spread in basis points.
    pub max_spread_bps: f64,
    /// Order arrival intensity k (orders per second).
    pub order_arrival_rate: f64,
    /// Time horizon T in seconds.
    pub time_horizon_secs: f64,
    /// Default quote size.
    pub default_quote_size: f64,
    /// Enable adverse selection protection.
    pub enable_toxicity_detection: bool,
    /// Inventory mean-reversion speed.
    pub inventory_reversion_speed: f64,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            base: AgentConfig {
                name: "MarketMaker".to_string(),
                max_latency_us: 300, // 300μs target
                ..Default::default()
            },
            gamma: 0.1,                    // Moderate risk aversion
            max_inventory: 1000.0,         // 1000 shares max
            min_spread_bps: 1.0,           // 1 basis point minimum
            max_spread_bps: 50.0,          // 50 basis points maximum
            order_arrival_rate: 10.0,      // 10 orders per second
            time_horizon_secs: 60.0,       // 1 minute horizon
            default_quote_size: 100.0,     // 100 shares per quote
            enable_toxicity_detection: true,
            inventory_reversion_speed: 0.1, // 10% reversion per update
        }
    }
}

/// Recent fill information for toxicity detection.
#[derive(Debug, Clone)]
struct FillEvent {
    /// Fill price.
    price: f64,
    /// Fill side.
    side: OrderSide,
    /// Fill timestamp.
    timestamp: u64,
    /// Mid-price at fill time.
    mid_at_fill: f64,
}

// ============================================================================
// Market Maker Agent
// ============================================================================

/// Market Maker Agent implementing Avellaneda-Stoikov optimal market making.
#[derive(Debug)]
pub struct MarketMakerAgent {
    /// Configuration.
    config: MarketMakerConfig,
    /// Current status.
    status: AtomicU8,
    /// Inventory per symbol.
    inventory: RwLock<HashMap<Symbol, InventoryState>>,
    /// Current quotes per symbol.
    quotes: RwLock<HashMap<Symbol, Quote>>,
    /// Recent fills for toxicity detection (last 1000 fills).
    recent_fills: RwLock<Vec<FillEvent>>,
    /// Current toxicity scores per symbol.
    toxicity: RwLock<HashMap<Symbol, ToxicityScore>>,
    /// Volatility estimates per symbol.
    volatility: RwLock<HashMap<Symbol, f64>>,
    /// Statistics.
    stats: AgentStats,
}

impl MarketMakerAgent {
    /// Create new market maker agent.
    pub fn new(config: MarketMakerConfig) -> Self {
        Self {
            config,
            status: AtomicU8::new(AgentStatus::Idle as u8),
            inventory: RwLock::new(HashMap::new()),
            quotes: RwLock::new(HashMap::new()),
            recent_fills: RwLock::new(Vec::with_capacity(1000)),
            toxicity: RwLock::new(HashMap::new()),
            volatility: RwLock::new(HashMap::new()),
            stats: AgentStats::new(),
        }
    }

    /// Update inventory for a symbol.
    pub fn update_inventory(&self, symbol: Symbol, position: f64, avg_cost: f64, current_price: f64) {
        let mut inventory = self.inventory.write();
        let state = inventory.entry(symbol).or_insert_with(InventoryState::default);

        state.position = position;
        state.avg_cost = avg_cost;
        state.unrealized_pnl = position * (current_price - avg_cost);
        state.updated_at = Timestamp::now();
    }

    /// Update volatility estimate for a symbol.
    pub fn update_volatility(&self, symbol: Symbol, vol: f64) {
        let mut volatility = self.volatility.write();
        volatility.insert(symbol, vol);
    }

    /// Record a fill event for toxicity tracking.
    pub fn record_fill(&self, symbol: Symbol, price: f64, side: OrderSide, mid_at_fill: f64) {
        // Update fill counts
        {
            let mut inventory = self.inventory.write();
            let state = inventory.entry(symbol).or_insert_with(InventoryState::default);
            match side {
                OrderSide::Buy => state.bid_fills += 1,
                OrderSide::Sell => state.ask_fills += 1,
            }
        }

        // Record fill event
        {
            let mut fills = self.recent_fills.write();
            fills.push(FillEvent {
                price,
                side,
                timestamp: Timestamp::now().as_nanos(),
                mid_at_fill,
            });

            // Trim to last 1000 fills
            let len = fills.len();
            if len > 1000 {
                fills.drain(0..len - 1000);
            }
        }
    }

    /// Calculate reservation price using Avellaneda-Stoikov formula.
    ///
    /// r(s, q, t) = s - q * γ * σ² * (T - t)
    #[inline]
    fn calculate_reservation_price(&self, mid_price: f64, position: f64, volatility: f64, time_to_horizon: f64) -> f64 {
        let gamma = self.config.gamma;
        let inventory_adjustment = position * gamma * volatility.powi(2) * time_to_horizon;
        mid_price - inventory_adjustment
    }

    /// Calculate optimal spread using Avellaneda-Stoikov formula.
    ///
    /// δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
    #[inline]
    fn calculate_optimal_spread(&self, volatility: f64, time_to_horizon: f64) -> f64 {
        let gamma = self.config.gamma;
        let k = self.config.order_arrival_rate;

        // First term: risk component
        let risk_component = gamma * volatility.powi(2) * time_to_horizon;

        // Second term: adverse selection component
        let adverse_selection = if gamma > 0.0 && k > 0.0 {
            (2.0 / gamma) * (1.0 + gamma / k).ln()
        } else {
            0.0
        };

        risk_component + adverse_selection
    }

    /// Adjust spread based on inventory position.
    ///
    /// Widens the spread on the side we want to reduce.
    #[inline]
    fn adjust_for_inventory(&self, base_spread: f64, inventory_skew: f64) -> (f64, f64) {
        // inventory_skew: -1 (max short) to +1 (max long)
        // When long, widen ask spread to encourage selling
        // When short, widen bid spread to encourage buying

        let skew_multiplier = 1.0 + inventory_skew.abs() * 0.5; // Up to 50% wider

        let bid_spread = if inventory_skew > 0.0 {
            // Long position: narrow bid to discourage more buying
            base_spread / skew_multiplier
        } else {
            // Short position: widen bid to encourage buying
            base_spread * skew_multiplier
        };

        let ask_spread = if inventory_skew < 0.0 {
            // Short position: narrow ask to discourage more selling
            base_spread / skew_multiplier
        } else {
            // Long position: widen ask to encourage selling
            base_spread * skew_multiplier
        };

        (bid_spread, ask_spread)
    }

    /// Detect toxic order flow and calculate toxicity score.
    fn detect_toxic_flow(&self, symbol: &Symbol) -> ToxicityScore {
        if !self.config.enable_toxicity_detection {
            return ToxicityScore::default();
        }

        let fills = self.recent_fills.read();
        if fills.is_empty() {
            return ToxicityScore::default();
        }

        // Factor 1: Fill rate imbalance
        let inventory = self.inventory.read();
        let fill_imbalance = if let Some(state) = inventory.get(symbol) {
            state.fill_rate_imbalance().abs() // 0.0 to 1.0
        } else {
            0.0
        };

        // Factor 2: Adverse price movement after fills
        let mut adverse_movement_sum = 0.0;
        let mut count = 0;

        for fill in fills.iter().rev().take(100) {
            // Check if price moved against us after the fill
            // For a buy fill, adverse movement is price going down
            // For a sell fill, adverse movement is price going up
            let current_mid = fills.last().map(|f| f.mid_at_fill).unwrap_or(fill.mid_at_fill);
            let price_change = current_mid - fill.mid_at_fill;

            let adverse = match fill.side {
                OrderSide::Buy => -price_change.min(0.0),  // Negative = adverse for buy
                OrderSide::Sell => price_change.max(0.0),   // Positive = adverse for sell
            };

            if fill.mid_at_fill > 0.0 {
                adverse_movement_sum += (adverse / fill.mid_at_fill).abs();
                count += 1;
            }
        }

        let avg_adverse_movement = if count > 0 {
            (adverse_movement_sum / count as f64).min(1.0)
        } else {
            0.0
        };

        // Combine factors (weighted average)
        let toxicity = 0.6 * fill_imbalance + 0.4 * avg_adverse_movement;
        ToxicityScore::new(toxicity)
    }

    /// Generate two-sided quote for a symbol.
    #[allow(unused_variables)]
    pub fn generate_quotes(&self, symbol: Symbol, mid_price: f64, _current_time_secs: f64) -> Option<Quote> {
        // Get volatility
        let volatility = {
            let vol_map = self.volatility.read();
            *vol_map.get(&symbol).unwrap_or(&0.01) // Default 1% volatility
        };

        if volatility <= 0.0 || mid_price <= 0.0 {
            return None;
        }

        // Get inventory state
        let (position, inventory_skew) = {
            let inventory = self.inventory.read();
            if let Some(state) = inventory.get(&symbol) {
                (state.position, state.inventory_skew(self.config.max_inventory))
            } else {
                (0.0, 0.0)
            }
        };

        // Calculate time to horizon
        let time_to_horizon = self.config.time_horizon_secs.max(1.0);

        // Calculate reservation price
        let reservation_price = self.calculate_reservation_price(
            mid_price,
            position,
            volatility,
            time_to_horizon,
        );

        // Calculate optimal spread
        let optimal_spread = self.calculate_optimal_spread(volatility, time_to_horizon);

        // Adjust spread for inventory
        let (bid_spread_half, ask_spread_half) = self.adjust_for_inventory(optimal_spread / 2.0, inventory_skew);

        // Detect toxicity and adjust
        let toxicity = self.detect_toxic_flow(&symbol);
        {
            let mut tox_map = self.toxicity.write();
            tox_map.insert(symbol, toxicity);
        }

        let toxicity_mult = toxicity.spread_multiplier();
        let bid_spread_half = bid_spread_half * toxicity_mult;
        let ask_spread_half = ask_spread_half * toxicity_mult;

        // Generate quotes
        let mut bid_price = reservation_price - bid_spread_half;
        let mut ask_price = reservation_price + ask_spread_half;

        // Apply min/max spread constraints
        let total_spread = ask_price - bid_price;
        let min_spread = mid_price * self.config.min_spread_bps / 10_000.0;
        let max_spread = mid_price * self.config.max_spread_bps / 10_000.0;

        let constrained_spread = total_spread.clamp(min_spread, max_spread);

        // Re-center around reservation price
        bid_price = reservation_price - constrained_spread / 2.0;
        ask_price = reservation_price + constrained_spread / 2.0;

        // Ensure positive prices
        if bid_price <= 0.0 || ask_price <= 0.0 {
            return None;
        }

        Some(Quote {
            bid_price,
            bid_size: self.config.default_quote_size,
            ask_price,
            ask_size: self.config.default_quote_size,
            timestamp: Timestamp::now(),
            mid_price,
            spread: ask_price - bid_price,
        })
    }

    /// Get current quote for a symbol.
    pub fn get_quote(&self, symbol: &Symbol) -> Option<Quote> {
        self.quotes.read().get(symbol).cloned()
    }

    /// Get current inventory for a symbol.
    pub fn get_inventory(&self, symbol: &Symbol) -> Option<InventoryState> {
        self.inventory.read().get(symbol).cloned()
    }

    /// Get current toxicity score for a symbol.
    pub fn get_toxicity(&self, symbol: &Symbol) -> ToxicityScore {
        self.toxicity.read().get(symbol).copied().unwrap_or_default()
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

impl Agent for MarketMakerAgent {
    fn id(&self) -> AgentId {
        AgentId::new(&self.config.base.name)
    }

    fn status(&self) -> AgentStatus {
        Self::status_from_u8(self.status.load(Ordering::Relaxed))
    }

    fn process(&self, portfolio: &Portfolio, regime: MarketRegime) -> Result<Option<RiskDecision>> {
        let start = Instant::now();
        self.status.store(AgentStatus::Processing as u8, Ordering::Relaxed);

        let current_time_secs = Timestamp::now().as_nanos() as f64 / 1_000_000_000.0;

        // Process each position in portfolio
        let mut any_inventory_breach = false;
        let mut max_toxicity: f64 = 0.0;

        {
            let mut quotes = self.quotes.write();

            for position in &portfolio.positions {
                let symbol = position.symbol;
                let mid_price = position.current_price.as_f64();

                // Update inventory from portfolio
                self.update_inventory(
                    symbol,
                    position.quantity.as_f64(),
                    position.avg_entry_price.as_f64(),
                    mid_price,
                );

                // Generate new quotes
                if let Some(quote) = self.generate_quotes(symbol, mid_price, current_time_secs) {
                    quotes.insert(symbol, quote);
                }

                // Check inventory limits
                let inventory = self.inventory.read();
                if let Some(state) = inventory.get(&symbol) {
                    if !state.is_within_limits(self.config.max_inventory) {
                        any_inventory_breach = true;
                    }
                }

                // Track max toxicity
                let toxicity = self.get_toxicity(&symbol);
                max_toxicity = max_toxicity.max(toxicity.value());
            }
        }

        // Determine risk level
        let risk_level = if any_inventory_breach {
            RiskLevel::High
        } else if max_toxicity > 0.7 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        };

        // Apply regime adjustments
        let regime_mult = regime.risk_multiplier();

        let latency_ns = start.elapsed().as_nanos() as u64;
        self.stats.record_cycle(latency_ns);
        self.status.store(AgentStatus::Idle as u8, Ordering::Relaxed);

        // Return risk decision if needed
        if risk_level.is_restricted() || !regime.is_favorable() {
            Ok(Some(RiskDecision {
                allowed: false,
                risk_level,
                reason: format!(
                    "Market making risk: inventory_breach={}, toxicity={:.2}, regime={:?}",
                    any_inventory_breach, max_toxicity, regime
                ),
                size_adjustment: regime_mult,
                timestamp: Timestamp::now(),
                latency_ns,
            }))
        } else {
            Ok(None)
        }
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
    fn test_market_maker_creation() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);
        assert_eq!(agent.status(), AgentStatus::Idle);
    }

    #[test]
    fn test_reservation_price_formula() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        // Test with zero inventory - should equal mid price
        let mid_price = 100.0;
        let position = 0.0;
        let volatility = 0.02; // 2%
        let time_to_horizon = 60.0;

        let reservation = agent.calculate_reservation_price(mid_price, position, volatility, time_to_horizon);
        assert!((reservation - mid_price).abs() < 0.0001);

        // Test with long position - reservation should be below mid
        let position = 500.0; // Long 500 shares
        let reservation = agent.calculate_reservation_price(mid_price, position, volatility, time_to_horizon);
        assert!(reservation < mid_price);

        // Test with short position - reservation should be above mid
        let position = -500.0; // Short 500 shares
        let reservation = agent.calculate_reservation_price(mid_price, position, volatility, time_to_horizon);
        assert!(reservation > mid_price);
    }

    #[test]
    fn test_optimal_spread_formula() {
        let config = MarketMakerConfig {
            gamma: 0.1,
            order_arrival_rate: 10.0,
            ..Default::default()
        };
        let agent = MarketMakerAgent::new(config);

        let volatility = 0.02; // 2%
        let time_to_horizon = 60.0;

        let spread = agent.calculate_optimal_spread(volatility, time_to_horizon);

        // Spread should be positive
        assert!(spread > 0.0);

        // Higher volatility should lead to wider spread
        let high_vol_spread = agent.calculate_optimal_spread(0.05, time_to_horizon);
        assert!(high_vol_spread > spread);
    }

    #[test]
    fn test_inventory_adjustment() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        let base_spread = 0.10;

        // Test neutral inventory (no adjustment)
        let (bid_spread, ask_spread) = agent.adjust_for_inventory(base_spread, 0.0);
        assert!((bid_spread - base_spread).abs() < 0.0001);
        assert!((ask_spread - base_spread).abs() < 0.0001);

        // Test long inventory (wider ask)
        let (bid_spread, ask_spread) = agent.adjust_for_inventory(base_spread, 0.5);
        assert!(bid_spread < base_spread);
        assert!(ask_spread > base_spread);

        // Test short inventory (wider bid)
        let (bid_spread, ask_spread) = agent.adjust_for_inventory(base_spread, -0.5);
        assert!(bid_spread > base_spread);
        assert!(ask_spread < base_spread);
    }

    #[test]
    fn test_toxicity_score() {
        let score = ToxicityScore::new(0.3);
        assert_eq!(score.value(), 0.3);
        assert!(!score.is_toxic());

        let toxic = ToxicityScore::new(0.8);
        assert!(toxic.is_toxic());
        assert!(toxic.spread_multiplier() > 1.0);

        // Test clamping
        let clamped = ToxicityScore::new(1.5);
        assert_eq!(clamped.value(), 1.0);
    }

    #[test]
    fn test_inventory_state() {
        let mut state = InventoryState::default();
        state.position = 500.0;

        assert!(state.is_within_limits(1000.0));
        assert!(!state.is_within_limits(400.0));

        let skew = state.inventory_skew(1000.0);
        assert_eq!(skew, 0.5);

        state.bid_fills = 10;
        state.ask_fills = 30;
        let imbalance = state.fill_rate_imbalance();
        assert_eq!(imbalance, 0.5); // (30 - 10) / 40
    }

    #[test]
    fn test_quote_generation() {
        let config = MarketMakerConfig {
            gamma: 0.1,
            max_inventory: 1000.0,
            min_spread_bps: 5.0,
            max_spread_bps: 50.0,
            ..Default::default()
        };
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");
        let mid_price = 100.0;
        let volatility = 0.02;

        // Update volatility
        agent.update_volatility(symbol, volatility);

        // Generate quote with zero inventory
        let quote = agent.generate_quotes(symbol, mid_price, 0.0);
        assert!(quote.is_some());

        let quote = quote.unwrap();
        assert!(quote.bid_price > 0.0);
        assert!(quote.ask_price > quote.bid_price);
        assert_eq!(quote.spread, quote.ask_price - quote.bid_price);

        // Check spread constraints
        let spread_bps = (quote.spread / mid_price) * 10_000.0;
        assert!(spread_bps >= 5.0);
        assert!(spread_bps <= 50.0);
    }

    #[test]
    fn test_quote_with_inventory() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");
        let mid_price = 100.0;

        agent.update_volatility(symbol, 0.02);

        // Test with long position
        agent.update_inventory(symbol, 500.0, 99.0, mid_price);

        let quote = agent.generate_quotes(symbol, mid_price, 0.0).unwrap();

        // With long position, reservation price should be below mid
        // which means quotes are skewed lower
        let quote_mid = (quote.bid_price + quote.ask_price) / 2.0;
        assert!(quote_mid < mid_price);
    }

    #[test]
    fn test_fill_recording() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");

        // Record some fills
        agent.record_fill(symbol, 100.0, OrderSide::Buy, 99.95);
        agent.record_fill(symbol, 100.0, OrderSide::Sell, 100.05);

        let inventory = agent.get_inventory(&symbol).unwrap();
        assert_eq!(inventory.bid_fills, 1);
        assert_eq!(inventory.ask_fills, 1);

        let fills = agent.recent_fills.read();
        assert_eq!(fills.len(), 2);
    }

    #[test]
    fn test_toxicity_detection() {
        let config = MarketMakerConfig {
            enable_toxicity_detection: true,
            ..Default::default()
        };
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");

        // Simulate imbalanced fills (more buys than sells = toxic)
        for _ in 0..20 {
            agent.record_fill(symbol, 100.0, OrderSide::Buy, 100.0);
        }
        for _ in 0..5 {
            agent.record_fill(symbol, 100.0, OrderSide::Sell, 100.0);
        }

        let toxicity = agent.detect_toxic_flow(&symbol);
        assert!(toxicity.value() > 0.0); // Should detect imbalance
    }

    #[test]
    fn test_agent_process() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        let mut portfolio = Portfolio::new(100_000.0);

        // Add a position
        let symbol = Symbol::new("TEST");
        let position = crate::core::types::Position {
            id: crate::core::types::PositionId::new(),
            symbol,
            quantity: Quantity::from_f64(100.0),
            avg_entry_price: Price::from_f64(99.0),
            current_price: Price::from_f64(100.0),
            unrealized_pnl: 100.0,
            realized_pnl: 0.0,
            opened_at: Timestamp::now(),
            updated_at: Timestamp::now(),
        };
        portfolio.positions.push(position);

        agent.update_volatility(symbol, 0.02);

        // Process
        let result = agent.process(&portfolio, MarketRegime::BullTrending);
        assert!(result.is_ok());

        // Check that quote was generated
        let quote = agent.get_quote(&symbol);
        assert!(quote.is_some());

        // Check latency
        assert!(agent.avg_latency_ns() > 0);
        assert!(agent.avg_latency_ns() < 1_000_000); // Less than 1ms
    }

    #[test]
    fn test_spread_constraints() {
        let config = MarketMakerConfig {
            min_spread_bps: 10.0,
            max_spread_bps: 20.0,
            ..Default::default()
        };
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");
        let mid_price = 100.0;

        agent.update_volatility(symbol, 0.01); // Low volatility

        let quote = agent.generate_quotes(symbol, mid_price, 0.0).unwrap();
        let spread_bps = quote.spread_bps();

        // Should be within constraints
        assert!(spread_bps >= 10.0);
        assert!(spread_bps <= 20.0);
    }

    #[test]
    fn test_edge_cases() {
        let config = MarketMakerConfig::default();
        let agent = MarketMakerAgent::new(config);

        let symbol = Symbol::new("TEST");

        // Zero volatility should return None
        agent.update_volatility(symbol, 0.0);
        let quote = agent.generate_quotes(symbol, 100.0, 0.0);
        assert!(quote.is_none());

        // Negative price should return None
        agent.update_volatility(symbol, 0.02);
        let quote = agent.generate_quotes(symbol, -100.0, 0.0);
        assert!(quote.is_none());
    }
}
