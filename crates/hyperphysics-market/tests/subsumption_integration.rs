//! Integration tests for Subsumption Trading System
//! Validates the complete Brooks' Subsumption Architecture implementation

use hyperphysics_market::subsumption::{
    SubsumptionTradingSystem,
    SubsumptionConfig,
    SurvivalConfig,
    RiskConfig,
    PositionConfig,
    ExecutionConfig,
    StrategyConfig,
    ExplorationConfig,
    MarketState,
    StateUpdate,
    TradingAction,
    OrderSide,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn create_production_config() -> SubsumptionConfig {
    SubsumptionConfig {
        survival: SurvivalConfig {
            max_drawdown_pct: 15.0,
            circuit_breaker_loss: 50000.0,
        },
        risk: RiskConfig {
            max_position_size: 100000.0,
            max_sector_exposure: 40.0,
        },
        position: PositionConfig {
            target_inventory: 0.0, // Disable position layer for most tests
            rebalance_threshold: 8.0,
        },
        execution: ExecutionConfig {
            max_slippage_bps: 8.0,
            order_size_limit: 5000.0,
        },
        strategy: StrategyConfig {
            signal_threshold: 0.4,
        },
        exploration: ExplorationConfig {
            exploration_rate: 0.03,
            min_exploration_interval: 300,
        },
    }
}

fn create_market_state(price: f64, spread_bps: f64, volatility: f64) -> MarketState {
    MarketState {
        price,
        spread_bps,
        volatility,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    }
}

#[test]
fn test_normal_trading_flow() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Normal trading with strong signal
    system.update_state(&StateUpdate::Signal(0.6));
    let action = system.process(&market_state);

    match action {
        TradingAction::Execute(order) => {
            assert_eq!(order.side, OrderSide::Buy);
            assert!(order.quantity > 0.0);
            assert!(order.quantity <= 5000.0);
        }
        _ => panic!("Expected Execute action in normal conditions"),
    }
}

#[test]
fn test_layer_hierarchy_enforcement() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Set all layers to trigger
    system.update_state(&StateUpdate::PnL(-60000.0)); // Survival
    system.update_state(&StateUpdate::Exposure(120000.0)); // Risk
    system.update_state(&StateUpdate::Inventory(60000.0)); // Position
    system.update_state(&StateUpdate::Signal(0.7)); // Strategy

    // Survival layer should win (highest priority)
    let action = system.process(&market_state);
    assert_eq!(action, TradingAction::Halt);
    assert_eq!(system.get_active_layer(), Some(0));
}

#[test]
fn test_risk_management_subsumption() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Trigger risk layer but not survival
    system.update_state(&StateUpdate::Exposure(120000.0));
    system.update_state(&StateUpdate::Signal(0.7));

    let action = system.process(&market_state);
    assert_eq!(action, TradingAction::ReduceExposure);
    assert_eq!(system.get_active_layer(), Some(1));
}

#[test]
fn test_position_rebalancing() {
    // Create custom config with position layer enabled
    let mut config = create_production_config();
    config.position.target_inventory = 50000.0;

    let mut system = SubsumptionTradingSystem::new(config);
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Trigger position layer with significant deviation
    system.update_state(&StateUpdate::Inventory(56000.0)); // 12% from target
    system.update_state(&StateUpdate::Signal(0.5));

    let action = system.process(&market_state);
    assert_eq!(action, TradingAction::Rebalance);
    assert_eq!(system.get_active_layer(), Some(2));
}

#[test]
fn test_high_slippage_prevents_execution() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 15.0, 0.40);

    // Strong signal but high slippage
    system.update_state(&StateUpdate::Signal(0.8));

    let action = system.process(&market_state);
    // Should fall back to signal layer without execution
    assert_eq!(action, TradingAction::Signal(0.8));
    assert_eq!(system.get_active_layer(), Some(4));
}

#[test]
fn test_survival_layer_recovery() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Trigger circuit breaker
    system.update_state(&StateUpdate::PnL(-60000.0));
    assert_eq!(system.process(&market_state), TradingAction::Halt);
    assert!(system.survival_layer().is_halted);

    // Manual recovery
    system.survival_layer_mut().reset_halt();
    assert!(!system.survival_layer().is_halted);
}

#[test]
fn test_multi_step_scenario() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Step 1: Normal trading
    system.update_state(&StateUpdate::Signal(0.5));
    let action = system.process(&market_state);
    assert!(matches!(action, TradingAction::Execute(_)));

    // Step 2: Market deteriorates, slippage increases
    let bad_market = create_market_state(148.0, 12.0, 0.50);
    let action = system.process(&bad_market);
    assert_eq!(action, TradingAction::Signal(0.5));

    // Step 3: Position grows too large
    system.update_state(&StateUpdate::Exposure(110000.0));
    let action = system.process(&bad_market);
    assert_eq!(action, TradingAction::ReduceExposure);

    // Step 4: Loss triggers circuit breaker
    system.update_state(&StateUpdate::PnL(-55000.0));
    let action = system.process(&bad_market);
    assert_eq!(action, TradingAction::Halt);
}

#[test]
fn test_default_configuration() {
    let mut system = SubsumptionTradingSystem::new(SubsumptionConfig::default());
    let market_state = create_market_state(100.0, 2.0, 0.20);

    // Verify default config creates valid system
    let action = system.process(&market_state);
    assert_eq!(action, TradingAction::NoAction); // No signal by default
}

#[test]
fn test_order_side_determination() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Positive signal -> Buy
    system.update_state(&StateUpdate::Signal(0.6));
    match system.process(&market_state) {
        TradingAction::Execute(order) => assert_eq!(order.side, OrderSide::Buy),
        _ => panic!("Expected Execute"),
    }

    // Negative signal -> Sell
    system.update_state(&StateUpdate::Signal(-0.6));
    match system.process(&market_state) {
        TradingAction::Execute(order) => assert_eq!(order.side, OrderSide::Sell),
        _ => panic!("Expected Execute"),
    }
}

#[test]
fn test_layer_state_persistence() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());

    // Update various layers
    system.update_state(&StateUpdate::PnL(1000.0));
    system.update_state(&StateUpdate::Exposure(50000.0));
    system.update_state(&StateUpdate::Inventory(52000.0));
    system.update_state(&StateUpdate::Signal(0.3));

    // Verify state persistence through accessors
    assert_eq!(system.survival_layer().current_drawdown, 0.0); // Profit, no drawdown
    assert_eq!(system.risk_layer().current_exposure, 50000.0);
    assert_eq!(system.position_layer().current_inventory, 52000.0);
    assert_eq!(system.strategy_layer().current_signal, 0.3);
}

#[test]
fn test_drawdown_calculation() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Establish peak
    system.update_state(&StateUpdate::PnL(10000.0));
    assert_eq!(system.survival_layer().current_drawdown, 0.0);

    // Create 10% drawdown (should be ok with 15% limit)
    system.update_state(&StateUpdate::PnL(-1000.0));
    let action = system.process(&market_state);
    assert_ne!(action, TradingAction::Halt);

    // Create 20% drawdown (exceeds 15% limit)
    system.update_state(&StateUpdate::PnL(-1200.0));
    let action = system.process(&market_state);
    assert_eq!(action, TradingAction::Halt);
}

#[test]
fn test_concurrent_layer_updates() {
    let mut system = SubsumptionTradingSystem::new(create_production_config());
    let market_state = create_market_state(150.0, 3.0, 0.25);

    // Simulate real trading scenario with multiple updates
    system.update_state(&StateUpdate::Signal(0.5));
    system.update_state(&StateUpdate::Exposure(30000.0)); // 30% of max, under 40% limit
    system.update_state(&StateUpdate::Slippage(4.0));

    // Should execute order in normal conditions
    let action = system.process(&market_state);
    assert!(matches!(action, TradingAction::Execute(_)));

    // Update with bad conditions
    system.update_state(&StateUpdate::Slippage(12.0));
    let action = system.process(&market_state);
    // High slippage prevents execution, should fall back to signal layer
    assert_eq!(action, TradingAction::Signal(0.5));
}
