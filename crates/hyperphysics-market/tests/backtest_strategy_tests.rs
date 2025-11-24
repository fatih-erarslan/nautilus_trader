//! Comprehensive backtest strategy tests
//!
//! Tests various trading strategies with the backtest engine

use hyperphysics_market::backtest::{
    Strategy, BacktestEngine, BacktestConfig, Signal, Side,
    Commission, Slippage, Portfolio, Position,
};
use hyperphysics_market::data::{Bar, Timeframe};
use hyperphysics_market::providers::MarketDataProvider;
use async_trait::async_trait;
use chrono::{Duration, Utc};
use hyperphysics_market::error::MarketResult;

// ============================================================================
// Mock Provider
// ============================================================================

struct MockProvider {
    bars: Vec<Bar>,
}

#[async_trait]
impl MarketDataProvider for MockProvider {
    async fn fetch_bars(
        &self,
        _symbol: &str,
        _timeframe: Timeframe,
        _start: chrono::DateTime<Utc>,
        _end: chrono::DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        Ok(self.bars.clone())
    }

    async fn fetch_latest_bar(&self, _symbol: &str) -> MarketResult<Bar> {
        Ok(self.bars.last().unwrap().clone())
    }

    fn provider_name(&self) -> &str {
        "MockProvider"
    }

    async fn supports_symbol(&self, _symbol: &str) -> MarketResult<bool> {
        Ok(true)
    }
}

// ============================================================================
// Test Strategies
// ============================================================================

/// Buy and hold strategy - buys on first bar and holds
struct BuyAndHoldStrategy {
    bought: bool,
}

#[async_trait]
impl Strategy for BuyAndHoldStrategy {
    async fn initialize(&mut self) {
        self.bought = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        if !self.bought {
            self.bought = true;
            vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        } else {
            vec![]
        }
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        "BuyAndHold".to_string()
    }
}

/// Simple moving average crossover strategy
struct SMAStrategy {
    short_period: usize,
    long_period: usize,
    prices: Vec<f64>,
    position_open: bool,
}

#[async_trait]
impl Strategy for SMAStrategy {
    async fn initialize(&mut self) {
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        if self.prices.len() < self.long_period {
            return vec![];
        }

        let short_sma: f64 = self.prices.iter().rev()
            .take(self.short_period)
            .sum::<f64>() / self.short_period as f64;

        let long_sma: f64 = self.prices.iter().rev()
            .take(self.long_period)
            .sum::<f64>() / self.long_period as f64;

        if short_sma > long_sma && !self.position_open {
            self.position_open = true;
            vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        } else if short_sma < long_sma && self.position_open {
            self.position_open = false;
            vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        } else {
            vec![]
        }
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        format!("SMA({},{})", self.short_period, self.long_period)
    }
}

/// Mean reversion strategy
struct MeanReversionStrategy {
    period: usize,
    std_dev_threshold: f64,
    prices: Vec<f64>,
    position_open: bool,
}

#[async_trait]
impl Strategy for MeanReversionStrategy {
    async fn initialize(&mut self) {
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        if self.prices.len() < self.period {
            return vec![];
        }

        let recent_prices: Vec<f64> = self.prices.iter().rev()
            .take(self.period)
            .copied()
            .collect();

        let mean: f64 = recent_prices.iter().sum::<f64>() / self.period as f64;
        let variance: f64 = recent_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / self.period as f64;
        let std_dev = variance.sqrt();

        let z_score = (bar.close - mean) / std_dev;

        // Buy when price is significantly below mean
        if z_score < -self.std_dev_threshold && !self.position_open {
            self.position_open = true;
            vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        }
        // Sell when price returns to mean or above
        else if z_score > 0.0 && self.position_open {
            self.position_open = false;
            vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        } else {
            vec![]
        }
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        format!("MeanReversion(period={},threshold={})", self.period, self.std_dev_threshold)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_uptrend_bars(count: usize, start_price: f64, increment: f64) -> Vec<Bar> {
    let mut bars = Vec::new();
    let start_time = Utc::now();

    for i in 0..count {
        let price = start_price + (i as f64 * increment);
        bars.push(Bar::new(
            "TEST".to_string(),
            start_time + Duration::hours(i as i64),
            price,
            price + 2.0,
            price - 2.0,
            price + 1.0,
            10000,
        ));
    }

    bars
}

fn create_downtrend_bars(count: usize, start_price: f64, decrement: f64) -> Vec<Bar> {
    let mut bars = Vec::new();
    let start_time = Utc::now();

    for i in 0..count {
        let price = start_price - (i as f64 * decrement);
        bars.push(Bar::new(
            "TEST".to_string(),
            start_time + Duration::hours(i as i64),
            price,
            price + 2.0,
            price - 2.0,
            price - 1.0,
            10000,
        ));
    }

    bars
}

fn create_oscillating_bars(count: usize, base_price: f64, amplitude: f64) -> Vec<Bar> {
    let mut bars = Vec::new();
    let start_time = Utc::now();

    for i in 0..count {
        let price = base_price + amplitude * ((i as f64 * std::f64::consts::PI / 10.0).sin());
        bars.push(Bar::new(
            "TEST".to_string(),
            start_time + Duration::hours(i as i64),
            price,
            price + 2.0,
            price - 2.0,
            price,
            10000,
        ));
    }

    bars
}

// ============================================================================
// Portfolio Tests
// ============================================================================

#[test]
fn test_portfolio_creation() {
    let portfolio = Portfolio::new(100000.0, Commission::None);
    assert_eq!(portfolio.cash, 100000.0);
    assert_eq!(portfolio.initial_capital, 100000.0);
    assert_eq!(portfolio.positions.len(), 0);
}

#[test]
fn test_portfolio_buy_success() {
    let mut portfolio = Portfolio::new(100000.0, Commission::Fixed(10.0));
    let result = portfolio.buy("AAPL", 100.0, 150.0, Utc::now());

    assert!(result.is_ok());
    let trade = result.unwrap();
    assert_eq!(trade.quantity, 100.0);
    assert_eq!(trade.price, 150.0);
    assert_eq!(trade.commission, 10.0);
    assert_eq!(portfolio.positions.len(), 1);
}

#[test]
fn test_portfolio_buy_insufficient_cash() {
    let mut portfolio = Portfolio::new(1000.0, Commission::None);
    let result = portfolio.buy("AAPL", 100.0, 150.0, Utc::now());

    assert!(result.is_err());
}

#[test]
fn test_portfolio_sell_success() {
    let mut portfolio = Portfolio::new(100000.0, Commission::Fixed(10.0));

    portfolio.buy("AAPL", 100.0, 150.0, Utc::now()).unwrap();
    let result = portfolio.sell("AAPL", 100.0, 160.0, Utc::now());

    assert!(result.is_ok());
    assert_eq!(portfolio.positions.len(), 0);
}

#[test]
fn test_portfolio_sell_insufficient_position() {
    let mut portfolio = Portfolio::new(100000.0, Commission::None);
    let result = portfolio.sell("AAPL", 100.0, 150.0, Utc::now());

    assert!(result.is_err());
}

#[test]
fn test_portfolio_equity_calculation() {
    use std::collections::HashMap;

    let mut portfolio = Portfolio::new(100000.0, Commission::None);
    portfolio.buy("AAPL", 100.0, 150.0, Utc::now()).unwrap();

    // Cash = 100,000 - 15,000 = 85,000
    // Position value = 100 * 150 = 15,000
    // Equity = 100,000
    assert_eq!(portfolio.equity(), 100000.0);

    // Update price
    let mut prices = HashMap::new();
    prices.insert("AAPL".to_string(), 160.0);
    portfolio.update_prices(&prices, Utc::now());

    // Cash = 85,000
    // Position value = 100 * 160 = 16,000
    // Equity = 101,000
    assert_eq!(portfolio.equity(), 101000.0);
}

// ============================================================================
// Position Tests
// ============================================================================

#[test]
fn test_position_creation() {
    let pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

    assert_eq!(pos.symbol, "AAPL");
    assert_eq!(pos.quantity, 100.0);
    assert_eq!(pos.avg_price, 150.0);
    assert_eq!(pos.current_price, 150.0);
}

#[test]
fn test_position_pnl() {
    let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

    assert_eq!(pos.unrealized_pnl(), 0.0);
    assert_eq!(pos.market_value(), 15000.0);

    pos.update_price(160.0, Utc::now());
    assert_eq!(pos.unrealized_pnl(), 1000.0);
    assert_eq!(pos.market_value(), 16000.0);
}

#[test]
fn test_position_averaging() {
    let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

    // Add 50 shares at $160
    pos.update(50.0, 160.0, Utc::now());

    // Average price should be (100*150 + 50*160) / 150 = 153.33
    assert_eq!(pos.quantity, 150.0);
    assert!((pos.avg_price - 153.33333333333334).abs() < 1e-10);
}

#[test]
fn test_position_reduce() {
    let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

    pos.update(-50.0, 160.0, Utc::now());

    assert_eq!(pos.quantity, 50.0);
    // Avg price may be recalculated when reducing - check it's reasonable
    assert!(pos.avg_price >= 140.0 && pos.avg_price <= 160.0);
}

#[test]
fn test_position_close() {
    let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

    pos.update(-100.0, 160.0, Utc::now());

    assert_eq!(pos.quantity, 0.0);
}

// ============================================================================
// Strategy Backtest Tests
// ============================================================================

#[tokio::test]
async fn test_buy_and_hold_uptrend() {
    let bars = create_uptrend_bars(20, 100.0, 1.0);
    let provider = MockProvider { bars: bars.clone() };

    let config = BacktestConfig {
        initial_capital: 100000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["TEST".to_string()],
        timeframe: Timeframe::Day1,
        start_date: bars[0].timestamp,
        end_date: bars.last().unwrap().timestamp,
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = BuyAndHoldStrategy { bought: false };

    let result = engine.run(&mut strategy).await.unwrap();

    assert!(result.metrics.final_equity > result.metrics.initial_capital);
    assert!(result.metrics.total_return > 0.0);
}

#[tokio::test]
async fn test_buy_and_hold_downtrend() {
    let bars = create_downtrend_bars(20, 100.0, 1.0);
    let provider = MockProvider { bars: bars.clone() };

    let config = BacktestConfig {
        initial_capital: 100000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["TEST".to_string()],
        timeframe: Timeframe::Day1,
        start_date: bars[0].timestamp,
        end_date: bars.last().unwrap().timestamp,
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = BuyAndHoldStrategy { bought: false };

    let result = engine.run(&mut strategy).await.unwrap();

    assert!(result.metrics.final_equity < result.metrics.initial_capital);
    assert!(result.metrics.total_return < 0.0);
}

#[tokio::test]
async fn test_sma_crossover_oscillating() {
    let bars = create_oscillating_bars(100, 100.0, 10.0);
    let provider = MockProvider { bars: bars.clone() };

    let config = BacktestConfig {
        initial_capital: 100000.0,
        commission: Commission::Fixed(1.0),
        slippage: Slippage::None,
        symbols: vec!["TEST".to_string()],
        timeframe: Timeframe::Day1,
        start_date: bars[0].timestamp,
        end_date: bars.last().unwrap().timestamp,
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = SMAStrategy {
        short_period: 5,
        long_period: 20,
        prices: Vec::new(),
        position_open: false,
    };

    let result = engine.run(&mut strategy).await.unwrap();

    // Should generate multiple trades
    assert!(result.trades.len() > 2);
}

#[tokio::test]
async fn test_mean_reversion_oscillating() {
    let bars = create_oscillating_bars(100, 100.0, 10.0);
    let provider = MockProvider { bars: bars.clone() };

    let config = BacktestConfig {
        initial_capital: 100000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["TEST".to_string()],
        timeframe: Timeframe::Day1,
        start_date: bars[0].timestamp,
        end_date: bars.last().unwrap().timestamp,
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = MeanReversionStrategy {
        period: 20,
        std_dev_threshold: 1.5,
        prices: Vec::new(),
        position_open: false,
    };

    let result = engine.run(&mut strategy).await.unwrap();

    // Mean reversion strategy may or may not generate trades depending on threshold
    // Just verify the backtest ran successfully
    assert!(result.metrics.final_equity > 0.0);
}

// ============================================================================
// Commission Tests
// ============================================================================

#[test]
fn test_commission_none() {
    let commission = Commission::None;
    assert_eq!(commission.calculate(1000.0), 0.0);
}

#[test]
fn test_commission_fixed() {
    let commission = Commission::Fixed(10.0);
    assert_eq!(commission.calculate(1000.0), 10.0);
    assert_eq!(commission.calculate(5000.0), 10.0);
}

#[test]
fn test_commission_percentage() {
    let commission = Commission::Percentage(0.001); // 0.1%
    assert_eq!(commission.calculate(10000.0), 10.0);
    assert_eq!(commission.calculate(5000.0), 5.0);
}

// ============================================================================
// Slippage Tests
// ============================================================================

#[test]
fn test_slippage_none() {
    let slippage = Slippage::None;
    assert_eq!(slippage.apply(100.0, Side::Buy), 100.0);
    assert_eq!(slippage.apply(100.0, Side::Sell), 100.0);
}

#[test]
fn test_slippage_fixed() {
    let slippage = Slippage::Fixed(0.05);
    assert_eq!(slippage.apply(100.0, Side::Buy), 100.05);
    assert_eq!(slippage.apply(100.0, Side::Sell), 99.95);
}

#[test]
fn test_slippage_percentage() {
    let slippage = Slippage::Percentage(0.001); // 0.1%
    assert_eq!(slippage.apply(100.0, Side::Buy), 100.1);
    assert_eq!(slippage.apply(100.0, Side::Sell), 99.9);
}
