//! Test fixtures and data generators

use q_star_core::*;
use q_star_trading::*;
use rand::prelude::*;

/// Generate random market state for testing
pub fn generate_random_market_state() -> MarketState {
    let mut rng = thread_rng();
    
    let num_prices = rng.gen_range(5..50);
    let base_price = rng.gen_range(50.0..200.0);
    
    let prices: Vec<f64> = (0..num_prices)
        .map(|_| base_price * (1.0 + rng.gen_range(-0.05..0.05)))
        .collect();
    
    let volumes: Vec<f64> = (0..num_prices)
        .map(|_| rng.gen_range(100.0..10000.0))
        .collect();
    
    let num_indicators = rng.gen_range(5..20);
    let technical_indicators: Vec<f64> = (0..num_indicators)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    
    MarketState {
        timestamp: chrono::Utc::now(),
        prices,
        volumes,
        technical_indicators,
        market_regime: match rng.gen_range(0..6) {
            0 => MarketRegime::Trending,
            1 => MarketRegime::Ranging,
            2 => MarketRegime::Volatile,
            3 => MarketRegime::Stable,
            4 => MarketRegime::Crisis,
            _ => MarketRegime::Recovery,
        },
        volatility: rng.gen_range(0.001..0.1),
        liquidity: rng.gen_range(0.1..1.0),
    }
}

/// Generate test action
pub fn generate_test_action() -> QStarAction {
    let mut rng = thread_rng();
    
    QStarAction {
        action_type: match rng.gen_range(0..10) {
            0 => ActionType::Buy,
            1 => ActionType::Sell,
            2 => ActionType::Hold,
            3 => ActionType::IncreasePosition,
            4 => ActionType::DecreasePosition,
            5 => ActionType::ClosePosition,
            6 => ActionType::StopLoss,
            7 => ActionType::TakeProfit,
            8 => ActionType::MarketMake,
            _ => ActionType::Hedge,
        },
        size: rng.gen_range(0.1..10.0),
        price: if rng.gen_bool(0.8) { Some(rng.gen_range(90.0..110.0)) } else { None },
        confidence: rng.gen_range(0.5..1.0),
        risk_level: match rng.gen_range(0..4) {
            0 => RiskLevel::VeryLow,
            1 => RiskLevel::Low,
            2 => RiskLevel::Medium,
            _ => RiskLevel::High,
        },
        priority: match rng.gen_range(0..4) {
            0 => ActionPriority::Low,
            1 => ActionPriority::Medium,
            2 => ActionPriority::High,
            _ => ActionPriority::Critical,
        },
        metadata: Default::default(),
    }
}

/// Generate market impact
pub fn generate_market_impact() -> MarketImpact {
    let mut rng = thread_rng();
    
    MarketImpact {
        price: rng.gen_range(95.0..105.0),
        volume: rng.gen_range(100.0..5000.0),
        slippage: rng.gen_range(0.0001..0.01),
        execution_time_ms: rng.gen_range(0..100),
    }
}

/// Generate test task
pub fn generate_test_task() -> QStarTask {
    QStarTask {
        id: uuid::Uuid::new_v4().to_string(),
        state: generate_random_market_state(),
        constraints: TaskConstraints {
            max_latency_us: rand::thread_rng().gen_range(10..100),
            required_confidence: rand::thread_rng().gen_range(0.7..0.95),
            risk_limit: rand::thread_rng().gen_range(0.01..0.05),
        },
        priority: match rand::thread_rng().gen_range(0..3) {
            0 => TaskPriority::Low,
            1 => TaskPriority::Medium,
            _ => TaskPriority::High,
        },
    }
}

/// Create stress test scenario
pub struct StressScenario {
    pub market_states: Vec<MarketState>,
    pub actions: Vec<QStarAction>,
    pub tasks: Vec<QStarTask>,
}

impl StressScenario {
    pub fn new(size: usize) -> Self {
        Self {
            market_states: (0..size).map(|_| generate_random_market_state()).collect(),
            actions: (0..size).map(|_| generate_test_action()).collect(),
            tasks: (0..size).map(|_| generate_test_task()).collect(),
        }
    }
}