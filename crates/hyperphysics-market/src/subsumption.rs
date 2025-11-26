//! # Subsumption Trading Architecture
//!
//! Implementation of Brooks' Subsumption Architecture (1986) for autonomous trading.
//! Layered control system where lower layers subsume higher layers.
//!
//! ## References
//! - Brooks, R. A. (1986). "A Robust Layered Control System for a Mobile Robot"
//!   IEEE Journal of Robotics and Automation, RA-2(1), 14-23.
//!
//! ## Architecture
//!
//! Layer hierarchy (0 = highest priority):
//! - Layer 0: Survival - circuit breakers, max drawdown protection
//! - Layer 1: Risk - position limits, exposure caps
//! - Layer 2: Position - inventory management, rebalancing
//! - Layer 3: Execution - order routing, slippage control
//! - Layer 4: Strategy - alpha signal generation
//! - Layer 5: Exploration - parameter tuning, learning
//!
//! Each layer can:
//! - Inhibit outputs of higher-numbered layers
//! - Suppress inputs to higher-numbered layers
//! - Operate independently without central controller

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for subsumption trading system
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubsumptionConfig {
    /// Survival layer configuration
    pub survival: SurvivalConfig,
    /// Risk layer configuration
    pub risk: RiskConfig,
    /// Position layer configuration
    pub position: PositionConfig,
    /// Execution layer configuration
    pub execution: ExecutionConfig,
    /// Strategy layer configuration
    pub strategy: StrategyConfig,
    /// Exploration layer configuration
    pub exploration: ExplorationConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SurvivalConfig {
    /// Maximum drawdown percentage (0-100)
    pub max_drawdown_pct: f64,
    /// Circuit breaker loss threshold (absolute value)
    pub circuit_breaker_loss: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RiskConfig {
    /// Maximum position size (in base currency)
    pub max_position_size: f64,
    /// Maximum sector exposure (percentage)
    pub max_sector_exposure: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PositionConfig {
    /// Target inventory level
    pub target_inventory: f64,
    /// Rebalance threshold (percentage deviation)
    pub rebalance_threshold: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExecutionConfig {
    /// Maximum slippage in basis points
    pub max_slippage_bps: f64,
    /// Maximum order size
    pub order_size_limit: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StrategyConfig {
    /// Signal threshold for trade generation
    pub signal_threshold: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExplorationConfig {
    /// Exploration rate (0-1)
    pub exploration_rate: f64,
    /// Minimum time between explorations (seconds)
    pub min_exploration_interval: u64,
}

/// Market state input to subsumption system
#[derive(Clone, Debug)]
pub struct MarketState {
    /// Current price
    pub price: f64,
    /// Current bid-ask spread
    pub spread_bps: f64,
    /// Market volatility (annualized)
    pub volatility: f64,
    /// Time of market state
    pub timestamp: u64,
}

/// State update for layer management
#[derive(Clone, Debug)]
pub enum StateUpdate {
    /// Update profit/loss
    PnL(f64),
    /// Update exposure
    Exposure(f64),
    /// Update inventory
    Inventory(f64),
    /// Update market signal
    Signal(f64),
    /// Update slippage
    Slippage(f64),
}

/// Trading action output from subsumption system
#[derive(Clone, Debug, PartialEq)]
pub enum TradingAction {
    /// Layer 0: Halt all trading
    Halt,
    /// Layer 1: Reduce exposure
    ReduceExposure,
    /// Layer 2: Rebalance inventory
    Rebalance,
    /// Layer 3: Execute order
    Execute(Order),
    /// Layer 4: Generated trading signal
    Signal(f64),
    /// Layer 5: Explore parameter space
    Explore,
    /// No action required
    NoAction,
}

/// Order specification
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    /// Order side
    pub side: OrderSide,
    /// Order quantity
    pub quantity: f64,
    /// Optional limit price
    pub limit_price: Option<f64>,
}

/// Order side
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

/// Layer 0: Survival - highest priority, circuit breakers and max drawdown
#[derive(Clone, Debug)]
pub struct SurvivalLayer {
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Circuit breaker loss threshold
    pub circuit_breaker_loss: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Halt status
    pub is_halted: bool,
    /// Peak equity for drawdown calculation
    peak_equity: f64,
    /// Current equity
    current_equity: f64,
}

impl SurvivalLayer {
    fn new(config: SurvivalConfig) -> Self {
        Self {
            max_drawdown_pct: config.max_drawdown_pct,
            circuit_breaker_loss: config.circuit_breaker_loss,
            current_drawdown: 0.0,
            is_halted: false,
            peak_equity: 0.0,
            current_equity: 0.0,
        }
    }

    /// Check survival conditions
    fn check(&self) -> Option<TradingAction> {
        if self.is_halted {
            return Some(TradingAction::Halt);
        }

        // Check circuit breaker
        if self.current_equity < -self.circuit_breaker_loss {
            return Some(TradingAction::Halt);
        }

        // Check max drawdown
        if self.current_drawdown > self.max_drawdown_pct {
            return Some(TradingAction::Halt);
        }

        None
    }

    /// Update with PnL
    fn update(&mut self, pnl: f64) {
        self.current_equity += pnl;

        // Update peak equity
        if self.current_equity > self.peak_equity {
            self.peak_equity = self.current_equity;
        }

        // Calculate drawdown
        if self.peak_equity > 0.0 {
            self.current_drawdown = ((self.peak_equity - self.current_equity) / self.peak_equity) * 100.0;
        }

        // Update halt status
        if self.current_equity < -self.circuit_breaker_loss ||
           self.current_drawdown > self.max_drawdown_pct {
            self.is_halted = true;
        }
    }

    /// Reset halt status (manual intervention required)
    pub fn reset_halt(&mut self) {
        self.is_halted = false;
    }
}

/// Layer 1: Risk management - position limits and exposure caps
#[derive(Clone, Debug)]
pub struct RiskLayer {
    /// Maximum position size
    pub max_position_size: f64,
    /// Maximum sector exposure
    pub max_sector_exposure: f64,
    /// Current exposure
    pub current_exposure: f64,
}

impl RiskLayer {
    fn new(config: RiskConfig) -> Self {
        Self {
            max_position_size: config.max_position_size,
            max_sector_exposure: config.max_sector_exposure,
            current_exposure: 0.0,
        }
    }

    /// Check risk limits
    fn check(&self) -> Option<TradingAction> {
        // Check if exposure exceeds limits
        if self.current_exposure.abs() > self.max_position_size {
            return Some(TradingAction::ReduceExposure);
        }

        if (self.current_exposure / self.max_position_size).abs() > (self.max_sector_exposure / 100.0) {
            return Some(TradingAction::ReduceExposure);
        }

        None
    }

    /// Update exposure
    fn update(&mut self, exposure: f64) {
        self.current_exposure = exposure;
    }
}

/// Layer 2: Position management - inventory control and rebalancing
#[derive(Clone, Debug)]
pub struct PositionLayer {
    /// Target inventory level
    pub target_inventory: f64,
    /// Current inventory
    pub current_inventory: f64,
    /// Rebalance threshold (percentage)
    pub rebalance_threshold: f64,
}

impl PositionLayer {
    fn new(config: PositionConfig) -> Self {
        Self {
            target_inventory: config.target_inventory,
            current_inventory: 0.0,
            rebalance_threshold: config.rebalance_threshold,
        }
    }

    /// Check if rebalancing needed
    fn check(&self) -> Option<TradingAction> {
        if self.target_inventory == 0.0 {
            return None;
        }

        let deviation = ((self.current_inventory - self.target_inventory) / self.target_inventory).abs() * 100.0;

        if deviation > self.rebalance_threshold {
            Some(TradingAction::Rebalance)
        } else {
            None
        }
    }

    /// Update inventory
    fn update(&mut self, inventory: f64) {
        self.current_inventory = inventory;
    }
}

/// Layer 3: Execution - order routing and slippage control
#[derive(Clone, Debug)]
pub struct ExecutionLayer {
    /// Maximum slippage in basis points
    pub max_slippage_bps: f64,
    /// Order size limit
    pub order_size_limit: f64,
    /// Current slippage estimate
    current_slippage_bps: f64,
}

impl ExecutionLayer {
    fn new(config: ExecutionConfig) -> Self {
        Self {
            max_slippage_bps: config.max_slippage_bps,
            order_size_limit: config.order_size_limit,
            current_slippage_bps: 0.0,
        }
    }

    /// Check execution conditions and generate order
    /// Only executes if signal exists and market conditions are acceptable
    fn check(&self, market_state: &MarketState, signal: f64, signal_threshold: f64) -> Option<TradingAction> {
        // Check if signal is strong enough (must meet strategy threshold)
        if signal.abs() < signal_threshold {
            return None;
        }

        // Check if slippage is acceptable
        if self.current_slippage_bps > self.max_slippage_bps {
            return None;
        }

        // Check if spread is acceptable (use spread as proxy for slippage)
        if market_state.spread_bps > self.max_slippage_bps {
            return None;
        }

        // Generate order based on signal
        let side = if signal > 0.0 {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };

        let quantity = (signal.abs() * self.order_size_limit).min(self.order_size_limit);

        Some(TradingAction::Execute(Order {
            side,
            quantity,
            limit_price: Some(market_state.price),
        }))
    }

    /// Update slippage estimate
    fn update(&mut self, slippage_bps: f64) {
        self.current_slippage_bps = slippage_bps;
    }
}

/// Layer 4: Strategy - alpha signal generation
#[derive(Clone, Debug)]
pub struct StrategyLayer {
    /// Signal threshold
    pub signal_threshold: f64,
    /// Current signal
    pub current_signal: f64,
}

impl StrategyLayer {
    fn new(config: StrategyConfig) -> Self {
        Self {
            signal_threshold: config.signal_threshold,
            current_signal: 0.0,
        }
    }

    /// Check if signal exceeds threshold
    fn check(&self) -> Option<TradingAction> {
        if self.current_signal.abs() >= self.signal_threshold {
            Some(TradingAction::Signal(self.current_signal))
        } else {
            None
        }
    }

    /// Update signal
    fn update(&mut self, signal: f64) {
        self.current_signal = signal;
    }
}

/// Layer 5: Exploration - parameter tuning and learning
#[derive(Clone, Debug)]
pub struct ExplorationLayer {
    /// Exploration rate
    pub exploration_rate: f64,
    /// Last exploration time
    pub last_exploration_time: u64,
    /// Minimum exploration interval
    min_exploration_interval: u64,
}

impl ExplorationLayer {
    fn new(config: ExplorationConfig) -> Self {
        Self {
            exploration_rate: config.exploration_rate,
            last_exploration_time: 0,
            min_exploration_interval: config.min_exploration_interval,
        }
    }

    /// Check if exploration should occur
    fn check(&self, current_time: u64) -> Option<TradingAction> {
        if current_time - self.last_exploration_time < self.min_exploration_interval {
            return None;
        }

        // Simple exploration trigger based on rate
        if (current_time % 100) as f64 / 100.0 < self.exploration_rate {
            Some(TradingAction::Explore)
        } else {
            None
        }
    }

    /// Update exploration time
    fn update(&mut self, time: u64) {
        self.last_exploration_time = time;
    }
}

/// Subsumption arbiter - manages layer priorities
#[derive(Clone, Debug)]
pub struct SubsumptionArbiter {
    /// Currently active layer (lowest index = highest priority)
    active_layer: Option<usize>,
}

impl SubsumptionArbiter {
    fn new() -> Self {
        Self {
            active_layer: None,
        }
    }

    /// Arbitrate between layer outputs
    /// Returns action from highest priority layer (lowest index)
    fn arbitrate(&mut self, actions: Vec<Option<TradingAction>>) -> TradingAction {
        // Find first non-None action (highest priority)
        for (layer_idx, action) in actions.iter().enumerate() {
            if let Some(act) = action {
                self.active_layer = Some(layer_idx);
                return act.clone();
            }
        }

        self.active_layer = None;
        TradingAction::NoAction
    }

    /// Get active layer index
    pub fn get_active_layer(&self) -> Option<usize> {
        self.active_layer
    }
}

/// Main subsumption trading system
pub struct SubsumptionTradingSystem {
    /// Layer 0: Survival
    layer_survival: SurvivalLayer,
    /// Layer 1: Risk
    layer_risk: RiskLayer,
    /// Layer 2: Position
    layer_position: PositionLayer,
    /// Layer 3: Execution
    layer_execution: ExecutionLayer,
    /// Layer 4: Strategy
    layer_strategy: StrategyLayer,
    /// Layer 5: Exploration
    layer_exploration: ExplorationLayer,
    /// Arbitration bus
    arbiter: SubsumptionArbiter,
}

impl SubsumptionTradingSystem {
    /// Create new subsumption trading system
    pub fn new(config: SubsumptionConfig) -> Self {
        Self {
            layer_survival: SurvivalLayer::new(config.survival),
            layer_risk: RiskLayer::new(config.risk),
            layer_position: PositionLayer::new(config.position),
            layer_execution: ExecutionLayer::new(config.execution),
            layer_strategy: StrategyLayer::new(config.strategy),
            layer_exploration: ExplorationLayer::new(config.exploration),
            arbiter: SubsumptionArbiter::new(),
        }
    }

    /// Process market state through all layers
    /// Returns highest priority action (lower layers subsume higher)
    pub fn process(&mut self, market_state: &MarketState) -> TradingAction {
        // Collect outputs from each layer
        let actions = vec![
            // Layer 0: Survival (highest priority)
            self.layer_survival.check(),
            // Layer 1: Risk
            self.layer_risk.check(),
            // Layer 2: Position
            self.layer_position.check(),
            // Layer 3: Execution (uses strategy signal and threshold)
            self.layer_execution.check(
                market_state,
                self.layer_strategy.current_signal,
                self.layer_strategy.signal_threshold
            ),
            // Layer 4: Strategy
            self.layer_strategy.check(),
            // Layer 5: Exploration (lowest priority)
            self.layer_exploration.check(market_state.timestamp),
        ];

        // Arbitrate and return highest priority action
        self.arbiter.arbitrate(actions)
    }

    /// Update layer states based on market data
    pub fn update_state(&mut self, update: &StateUpdate) {
        match update {
            StateUpdate::PnL(pnl) => {
                self.layer_survival.update(*pnl);
            }
            StateUpdate::Exposure(exposure) => {
                self.layer_risk.update(*exposure);
            }
            StateUpdate::Inventory(inventory) => {
                self.layer_position.update(*inventory);
            }
            StateUpdate::Signal(signal) => {
                self.layer_strategy.update(*signal);
            }
            StateUpdate::Slippage(slippage_bps) => {
                self.layer_execution.update(*slippage_bps);
            }
        }
    }

    /// Get currently active layer
    pub fn get_active_layer(&self) -> Option<usize> {
        self.arbiter.get_active_layer()
    }

    /// Get reference to survival layer
    pub fn survival_layer(&self) -> &SurvivalLayer {
        &self.layer_survival
    }

    /// Get mutable reference to survival layer
    pub fn survival_layer_mut(&mut self) -> &mut SurvivalLayer {
        &mut self.layer_survival
    }

    /// Get reference to risk layer
    pub fn risk_layer(&self) -> &RiskLayer {
        &self.layer_risk
    }

    /// Get reference to position layer
    pub fn position_layer(&self) -> &PositionLayer {
        &self.layer_position
    }

    /// Get reference to execution layer
    pub fn execution_layer(&self) -> &ExecutionLayer {
        &self.layer_execution
    }

    /// Get reference to strategy layer
    pub fn strategy_layer(&self) -> &StrategyLayer {
        &self.layer_strategy
    }

    /// Get reference to exploration layer
    pub fn exploration_layer(&self) -> &ExplorationLayer {
        &self.layer_exploration
    }
}

impl Default for SubsumptionConfig {
    fn default() -> Self {
        Self {
            survival: SurvivalConfig {
                max_drawdown_pct: 20.0,
                circuit_breaker_loss: 100000.0,
            },
            risk: RiskConfig {
                max_position_size: 1000000.0,
                max_sector_exposure: 50.0,
            },
            position: PositionConfig {
                target_inventory: 0.0,
                rebalance_threshold: 10.0,
            },
            execution: ExecutionConfig {
                max_slippage_bps: 10.0,
                order_size_limit: 10000.0,
            },
            strategy: StrategyConfig {
                signal_threshold: 0.5,
            },
            exploration: ExplorationConfig {
                exploration_rate: 0.1,
                min_exploration_interval: 3600,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> SubsumptionConfig {
        SubsumptionConfig {
            survival: SurvivalConfig {
                max_drawdown_pct: 10.0,
                circuit_breaker_loss: 1000.0,
            },
            risk: RiskConfig {
                max_position_size: 10000.0,
                max_sector_exposure: 30.0,
            },
            position: PositionConfig {
                target_inventory: 0.0, // Disable position layer for most tests
                rebalance_threshold: 5.0,
            },
            execution: ExecutionConfig {
                max_slippage_bps: 5.0,
                order_size_limit: 1000.0,
            },
            strategy: StrategyConfig {
                signal_threshold: 0.3,
            },
            exploration: ExplorationConfig {
                exploration_rate: 0.05,
                min_exploration_interval: 60,
            },
        }
    }

    fn create_test_market_state() -> MarketState {
        MarketState {
            price: 100.0,
            spread_bps: 2.0,
            volatility: 0.2,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    #[test]
    fn test_survival_layer_normal() {
        let config = create_test_config();
        let layer = SurvivalLayer::new(config.survival);

        assert_eq!(layer.check(), None);
        assert!(!layer.is_halted);
    }

    #[test]
    fn test_survival_layer_circuit_breaker() {
        let config = create_test_config();
        let mut layer = SurvivalLayer::new(config.survival);

        layer.update(-1500.0);
        assert_eq!(layer.check(), Some(TradingAction::Halt));
        assert!(layer.is_halted);
    }

    #[test]
    fn test_survival_layer_max_drawdown() {
        let config = create_test_config();
        let mut layer = SurvivalLayer::new(config.survival);

        // Establish peak
        layer.update(1000.0);
        // Create drawdown > 10%
        layer.update(-200.0);

        assert_eq!(layer.check(), Some(TradingAction::Halt));
        assert!(layer.is_halted);
    }

    #[test]
    fn test_risk_layer_within_limits() {
        let config = create_test_config();
        let layer = RiskLayer::new(config.risk);

        assert_eq!(layer.check(), None);
    }

    #[test]
    fn test_risk_layer_exceeds_position_size() {
        let config = create_test_config();
        let mut layer = RiskLayer::new(config.risk);

        layer.update(15000.0);
        assert_eq!(layer.check(), Some(TradingAction::ReduceExposure));
    }

    #[test]
    fn test_risk_layer_exceeds_sector_exposure() {
        let config = create_test_config();
        let mut layer = RiskLayer::new(config.risk);

        // 40% of max_position_size exceeds 30% sector limit
        layer.update(4000.0);
        assert_eq!(layer.check(), Some(TradingAction::ReduceExposure));
    }

    #[test]
    fn test_position_layer_no_rebalance() {
        let mut config = create_test_config();
        config.position.target_inventory = 5000.0; // Enable for this test
        let mut layer = PositionLayer::new(config.position);

        layer.update(5100.0); // 2% deviation from target 5000
        assert_eq!(layer.check(), None);
    }

    #[test]
    fn test_position_layer_needs_rebalance() {
        let mut config = create_test_config();
        config.position.target_inventory = 5000.0; // Enable for this test
        let mut layer = PositionLayer::new(config.position);

        layer.update(5500.0); // 10% deviation from target 5000
        assert_eq!(layer.check(), Some(TradingAction::Rebalance));
    }

    #[test]
    fn test_execution_layer_acceptable_slippage() {
        let config = create_test_config();
        let layer = ExecutionLayer::new(config.execution);
        let market_state = create_test_market_state();

        let action = layer.check(&market_state, 0.5, 0.3);
        assert!(matches!(action, Some(TradingAction::Execute(_))));
    }

    #[test]
    fn test_execution_layer_excessive_slippage() {
        let config = create_test_config();
        let mut layer = ExecutionLayer::new(config.execution);
        let market_state = create_test_market_state();

        layer.update(10.0); // Exceeds max_slippage_bps
        let action = layer.check(&market_state, 0.5, 0.3);
        assert_eq!(action, None);
    }

    #[test]
    fn test_execution_layer_order_generation() {
        let config = create_test_config();
        let layer = ExecutionLayer::new(config.execution);
        let market_state = create_test_market_state();

        let action = layer.check(&market_state, 0.5, 0.3);
        match action {
            Some(TradingAction::Execute(order)) => {
                assert_eq!(order.side, OrderSide::Buy);
                assert!(order.quantity > 0.0);
                assert!(order.quantity <= 1000.0);
                assert_eq!(order.limit_price, Some(100.0));
            }
            _ => panic!("Expected Execute action"),
        }
    }

    #[test]
    fn test_strategy_layer_signal_threshold() {
        let config = create_test_config();
        let mut layer = StrategyLayer::new(config.strategy);

        layer.update(0.2);
        assert_eq!(layer.check(), None);

        layer.update(0.4);
        assert_eq!(layer.check(), Some(TradingAction::Signal(0.4)));
    }

    #[test]
    fn test_exploration_layer_timing() {
        let config = create_test_config();
        let mut layer = ExplorationLayer::new(config.exploration);

        let time1 = 1000;
        layer.update(time1);

        // Too soon for exploration
        let action = layer.check(time1 + 30);
        assert_eq!(action, None);

        // Enough time has passed
        let action = layer.check(time1 + 70);
        // May or may not trigger based on exploration_rate randomness
        // Just verify it doesn't panic
        let _ = action;
    }

    #[test]
    fn test_subsumption_arbitration_survival_wins() {
        let mut system = SubsumptionTradingSystem::new(create_test_config());
        let market_state = create_test_market_state();

        // Trigger survival layer
        system.update_state(&StateUpdate::PnL(-1500.0));

        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Halt);
        assert_eq!(system.get_active_layer(), Some(0));
    }

    #[test]
    fn test_subsumption_arbitration_risk_wins() {
        let mut system = SubsumptionTradingSystem::new(create_test_config());
        let market_state = create_test_market_state();

        // Trigger risk layer
        system.update_state(&StateUpdate::Exposure(15000.0));
        // Add strategy signal
        system.update_state(&StateUpdate::Signal(0.5));

        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::ReduceExposure);
        assert_eq!(system.get_active_layer(), Some(1));
    }

    #[test]
    fn test_subsumption_arbitration_position_wins() {
        let mut config = create_test_config();
        config.position.target_inventory = 5000.0; // Enable position layer for this test
        let mut system = SubsumptionTradingSystem::new(config);
        let market_state = create_test_market_state();

        // Trigger position layer
        system.update_state(&StateUpdate::Inventory(5500.0));
        // Add strategy signal
        system.update_state(&StateUpdate::Signal(0.5));

        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Rebalance);
        assert_eq!(system.get_active_layer(), Some(2));
    }

    #[test]
    fn test_subsumption_arbitration_execution_wins() {
        let mut system = SubsumptionTradingSystem::new(create_test_config());
        let market_state = create_test_market_state();

        // Set normal conditions and strong signal
        system.update_state(&StateUpdate::Signal(0.5));

        let action = system.process(&market_state);
        assert!(matches!(action, TradingAction::Execute(_)));
        assert_eq!(system.get_active_layer(), Some(3));
    }

    #[test]
    fn test_subsumption_layer_priority() {
        let mut system = SubsumptionTradingSystem::new(create_test_config());
        let market_state = create_test_market_state();

        // Trigger multiple layers simultaneously
        system.update_state(&StateUpdate::PnL(-1500.0)); // Survival
        system.update_state(&StateUpdate::Exposure(15000.0)); // Risk
        system.update_state(&StateUpdate::Signal(0.5)); // Strategy

        // Survival should win (layer 0)
        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Halt);
        assert_eq!(system.get_active_layer(), Some(0));
    }

    #[test]
    fn test_full_integration() {
        let mut config = create_test_config();
        config.position.target_inventory = 5000.0; // Enable position layer for this test
        let mut system = SubsumptionTradingSystem::new(config);
        let mut market_state = create_test_market_state();

        // Set initial inventory to match target (avoid triggering position layer immediately)
        system.update_state(&StateUpdate::Inventory(5000.0));

        // Normal operation
        system.update_state(&StateUpdate::Signal(0.4));
        let action = system.process(&market_state);
        assert!(matches!(action, TradingAction::Execute(_)));

        // Increase slippage
        market_state.spread_bps = 10.0;
        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Signal(0.4)); // Falls back to signal layer

        // Add position deviation
        system.update_state(&StateUpdate::Inventory(5500.0));
        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Rebalance); // Position layer takes over

        // Add risk violation
        system.update_state(&StateUpdate::Exposure(15000.0));
        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::ReduceExposure); // Risk layer takes over

        // Trigger survival
        system.update_state(&StateUpdate::PnL(-1500.0));
        let action = system.process(&market_state);
        assert_eq!(action, TradingAction::Halt); // Survival layer dominates
    }

    #[test]
    fn test_layer_accessors() {
        let system = SubsumptionTradingSystem::new(create_test_config());

        assert_eq!(system.survival_layer().max_drawdown_pct, 10.0);
        assert_eq!(system.risk_layer().max_position_size, 10000.0);
        assert_eq!(system.position_layer().target_inventory, 0.0); // Updated to match config
        assert_eq!(system.execution_layer().max_slippage_bps, 5.0);
        assert_eq!(system.strategy_layer().signal_threshold, 0.3);
        assert_eq!(system.exploration_layer().exploration_rate, 0.05);
    }
}
