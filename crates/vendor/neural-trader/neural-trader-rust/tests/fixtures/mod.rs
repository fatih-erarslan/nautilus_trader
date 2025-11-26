// Test fixtures and reusable test data
use rust_decimal::Decimal;
use std::str::FromStr;

/// Create a test portfolio with standard positions
pub fn create_test_portfolio() -> TestPortfolio {
    let mut portfolio = TestPortfolio {
        cash: Decimal::from_str("100000.0").unwrap(),
        positions: Vec::new(),
    };

    portfolio.add_position("AAPL", 100, "180.50");
    portfolio.add_position("MSFT", 50, "380.00");
    portfolio.add_position("GOOGL", 30, "140.00");

    portfolio
}

pub struct TestPortfolio {
    pub cash: Decimal,
    pub positions: Vec<TestPosition>,
}

impl TestPortfolio {
    pub fn add_position(&mut self, symbol: &str, quantity: i32, price: &str) {
        self.positions.push(TestPosition {
            symbol: symbol.to_string(),
            quantity,
            price: Decimal::from_str(price).unwrap(),
        });
    }

    pub fn total_value(&self) -> Decimal {
        let position_value: Decimal = self.positions.iter()
            .map(|p| Decimal::from(p.quantity) * p.price)
            .sum();
        self.cash + position_value
    }
}

pub struct TestPosition {
    pub symbol: String,
    pub quantity: i32,
    pub price: Decimal,
}

/// Generate mock OHLCV data for testing
pub fn generate_mock_bars(symbol: &str, days: usize) -> Vec<TestBar> {
    let mut bars = Vec::new();
    let mut price = 100.0;

    for i in 0..days {
        // Simulate price movement
        let change = ((i as f64 * 0.1).sin() * 2.0) + (i as f64 * 0.01);
        price += change;

        bars.push(TestBar {
            timestamp: format!("2024-01-{:02}", (i % 28) + 1),
            symbol: symbol.to_string(),
            open: price,
            high: price * 1.02,
            low: price * 0.98,
            close: price,
            volume: 1_000_000 + (i * 10_000) as i64,
        });
    }

    bars
}

pub struct TestBar {
    pub timestamp: String,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

/// Create test order with default values
pub fn create_test_order(symbol: &str, quantity: i32) -> TestOrder {
    TestOrder {
        id: format!("test-{}", uuid::Uuid::new_v4()),
        symbol: symbol.to_string(),
        quantity,
        side: "buy".to_string(),
        order_type: "market".to_string(),
        status: "new".to_string(),
    }
}

pub struct TestOrder {
    pub id: String,
    pub symbol: String,
    pub quantity: i32,
    pub side: String,
    pub order_type: String,
    pub status: String,
}

/// Generate test returns series
pub fn generate_test_returns(days: usize, mean: f64, volatility: f64) -> Vec<f64> {
    let mut returns = Vec::new();

    for i in 0..days {
        // Simple sine wave with noise for returns
        let trend = mean;
        let cycle = (i as f64 * 0.1).sin() * volatility;
        let noise = ((i as f64 * 13.0).sin() * 0.5) * volatility;

        returns.push(trend + cycle + noise);
    }

    returns
}

/// Create test market data snapshot
pub fn create_market_snapshot() -> TestMarketSnapshot {
    TestMarketSnapshot {
        symbols: vec![
            ("AAPL".to_string(), 180.50, 180.52),
            ("MSFT".to_string(), 380.00, 380.05),
            ("GOOGL".to_string(), 140.00, 140.03),
        ],
        timestamp: "2024-01-01T10:00:00Z".to_string(),
    }
}

pub struct TestMarketSnapshot {
    pub symbols: Vec<(String, f64, f64)>, // (symbol, bid, ask)
    pub timestamp: String,
}

/// Create test strategy configuration
pub fn create_strategy_config(strategy_type: &str) -> TestStrategyConfig {
    TestStrategyConfig {
        strategy_type: strategy_type.to_string(),
        symbols: vec!["AAPL".to_string(), "MSFT".to_string()],
        params: match strategy_type {
            "pairs" => vec![("window".to_string(), "20".to_string())],
            "momentum" => vec![("lookback".to_string(), "30".to_string())],
            _ => vec![],
        },
    }
}

pub struct TestStrategyConfig {
    pub strategy_type: String,
    pub symbols: Vec<String>,
    pub params: Vec<(String, String)>,
}

/// Create test risk limits
pub fn create_risk_limits() -> TestRiskLimits {
    TestRiskLimits {
        max_position_size: Decimal::from_str("0.10").unwrap(), // 10%
        max_portfolio_risk: Decimal::from_str("0.02").unwrap(), // 2%
        max_drawdown: Decimal::from_str("0.20").unwrap(), // 20%
        var_limit: Decimal::from_str("5000.0").unwrap(),
    }
}

pub struct TestRiskLimits {
    pub max_position_size: Decimal,
    pub max_portfolio_risk: Decimal,
    pub max_drawdown: Decimal,
    pub var_limit: Decimal,
}

// Re-export uuid for test fixtures
pub use uuid;
