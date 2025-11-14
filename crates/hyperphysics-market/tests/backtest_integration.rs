//! Integration tests for the backtesting framework
//!
//! These tests demonstrate various strategies and backtesting scenarios

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use hyperphysics_market::{
    backtest::{
        BacktestConfig, BacktestEngine, Commission, Signal, Slippage, Strategy,
    },
    data::{Bar, Timeframe},
    error::MarketResult,
    providers::MarketDataProvider,
};

// ============================================================================
// Mock Provider for Testing
// ============================================================================

#[derive(Clone)]
struct MockProvider {
    bars: Vec<Bar>,
}

impl MockProvider {
    fn new_with_trend(symbol: &str, days: usize, start_price: f64, daily_change: f64) -> Self {
        let mut bars = Vec::new();
        let start_time = Utc::now() - Duration::days(days as i64);

        for i in 0..days {
            let timestamp = start_time + Duration::days(i as i64);
            let price = start_price + (i as f64 * daily_change);
            let volatility = price * 0.02;

            bars.push(Bar::new(
                symbol.to_string(),
                timestamp,
                price,
                price + volatility,
                price - volatility,
                price + volatility * 0.5,
                100000 + (i * 1000) as u64,
            ));
        }

        Self { bars }
    }

    fn new_with_sine_wave(
        symbol: &str,
        days: usize,
        base_price: f64,
        amplitude: f64,
        period: f64,
    ) -> Self {
        let mut bars = Vec::new();
        let start_time = Utc::now() - Duration::days(days as i64);

        for i in 0..days {
            let timestamp = start_time + Duration::days(i as i64);
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / period;
            let price = base_price + amplitude * angle.sin();
            let volatility = price * 0.01;

            bars.push(Bar::new(
                symbol.to_string(),
                timestamp,
                price,
                price + volatility,
                price - volatility,
                price,
                100000,
            ));
        }

        Self { bars }
    }
}

#[async_trait]
impl MarketDataProvider for MockProvider {
    async fn fetch_bars(
        &self,
        _symbol: &str,
        _timeframe: Timeframe,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        Ok(self.bars.clone())
    }

    async fn fetch_latest_bar(&self, _symbol: &str) -> MarketResult<Bar> {
        Ok(self.bars.last().unwrap().clone())
    }

    fn provider_name(&self) -> &str {
        "Mock"
    }

    async fn supports_symbol(&self, _symbol: &str) -> MarketResult<bool> {
        Ok(true)
    }
}

// ============================================================================
// Example Strategies
// ============================================================================

/// Simple Moving Average Crossover Strategy
struct SMAStrategy {
    short_period: usize,
    long_period: usize,
    prices: Vec<f64>,
    position_open: bool,
}

impl SMAStrategy {
    fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
            prices: Vec::new(),
            position_open: false,
        }
    }

    fn calculate_sma(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period {
            return None;
        }

        let sum: f64 = self.prices.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }
}

#[async_trait]
impl Strategy for SMAStrategy {
    async fn initialize(&mut self) {
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let short_sma = self.calculate_sma(self.short_period);
        let long_sma = self.calculate_sma(self.long_period);

        if short_sma.is_none() || long_sma.is_none() {
            return vec![];
        }

        let short_sma = short_sma.unwrap();
        let long_sma = long_sma.unwrap();

        // Golden cross: buy signal
        if short_sma > long_sma && !self.position_open {
            self.position_open = true;
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Death cross: sell signal
        if short_sma < long_sma && self.position_open {
            self.position_open = false;
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        format!("SMA({}, {})", self.short_period, self.long_period)
    }
}

/// Mean Reversion Strategy
struct MeanReversionStrategy {
    period: usize,
    std_dev_threshold: f64,
    prices: Vec<f64>,
    position_open: bool,
}

impl MeanReversionStrategy {
    fn new(period: usize, std_dev_threshold: f64) -> Self {
        Self {
            period,
            std_dev_threshold,
            prices: Vec::new(),
            position_open: false,
        }
    }

    fn calculate_mean(&self) -> Option<f64> {
        if self.prices.len() < self.period {
            return None;
        }

        let sum: f64 = self.prices.iter().rev().take(self.period).sum();
        Some(sum / self.period as f64)
    }

    fn calculate_std_dev(&self, mean: f64) -> f64 {
        let variance: f64 = self
            .prices
            .iter()
            .rev()
            .take(self.period)
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / self.period as f64;

        variance.sqrt()
    }
}

#[async_trait]
impl Strategy for MeanReversionStrategy {
    async fn initialize(&mut self) {
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let mean = self.calculate_mean();
        if mean.is_none() {
            return vec![];
        }

        let mean = mean.unwrap();
        let std_dev = self.calculate_std_dev(mean);
        let z_score = (bar.close - mean) / std_dev;

        // Buy when price is below mean - threshold * std_dev
        if z_score < -self.std_dev_threshold && !self.position_open {
            self.position_open = true;
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Sell when price returns to mean or above
        if z_score > 0.0 && self.position_open {
            self.position_open = false;
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        format!("MeanReversion({}, {:.1}σ)", self.period, self.std_dev_threshold)
    }
}

/// Momentum Strategy based on rate of change
struct MomentumStrategy {
    period: usize,
    threshold: f64,
    prices: Vec<f64>,
    position_open: bool,
}

impl MomentumStrategy {
    fn new(period: usize, threshold: f64) -> Self {
        Self {
            period,
            threshold,
            prices: Vec::new(),
            position_open: false,
        }
    }

    fn calculate_roc(&self) -> Option<f64> {
        if self.prices.len() < self.period + 1 {
            return None;
        }

        let current = *self.prices.last().unwrap();
        let previous = self.prices[self.prices.len() - self.period - 1];

        Some((current - previous) / previous * 100.0)
    }
}

#[async_trait]
impl Strategy for MomentumStrategy {
    async fn initialize(&mut self) {
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let roc = self.calculate_roc();
        if roc.is_none() {
            return vec![];
        }

        let roc = roc.unwrap();

        // Buy on strong positive momentum
        if roc > self.threshold && !self.position_open {
            self.position_open = true;
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Sell on weak or negative momentum
        if roc < -self.threshold && self.position_open {
            self.position_open = false;
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }

    async fn finalize(&mut self) {}

    fn name(&self) -> String {
        format!("Momentum({}, {:.1}%)", self.period, self.threshold)
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_sma_strategy_uptrend() {
    let provider = MockProvider::new_with_trend("AAPL", 100, 100.0, 0.5);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Percentage(0.001),
        slippage: Slippage::Percentage(0.0005),
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(100),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = SMAStrategy::new(5, 20);

    let result = engine
        .run(&mut strategy)
        .await
        .expect("Backtest should succeed");

    // In an uptrend, SMA strategy should be profitable
    assert!(result.metrics.total_return > 0.0);
    assert!(result.metrics.final_equity > result.metrics.initial_capital);
    assert!(!result.trades.is_empty());

    println!("SMA Strategy Results:");
    println!("  Total Return: {:.2}%", result.metrics.total_return);
    println!("  Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
    println!("  Max Drawdown: {:.2}%", result.metrics.max_drawdown);
    println!("  Win Rate: {:.2}%", result.metrics.win_rate);
    println!("  Total Trades: {}", result.metrics.total_trades);
}

#[tokio::test]
async fn test_mean_reversion_sine_wave() {
    let provider = MockProvider::new_with_sine_wave("SPY", 200, 100.0, 10.0, 30.0);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Fixed(1.0),
        slippage: Slippage::None,
        symbols: vec!["SPY".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(200),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = MeanReversionStrategy::new(20, 1.5);

    let result = engine
        .run(&mut strategy)
        .await
        .expect("Backtest should succeed");

    // Mean reversion should work well with oscillating prices
    assert!(!result.trades.is_empty());
    println!("\nMean Reversion Strategy Results:");
    println!("  Total Return: {:.2}%", result.metrics.total_return);
    println!("  Win Rate: {:.2}%", result.metrics.win_rate);
    println!("  Profit Factor: {:.2}", result.metrics.profit_factor);
}

#[tokio::test]
async fn test_momentum_strategy() {
    let provider = MockProvider::new_with_trend("TSLA", 150, 200.0, 1.0);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Percentage(0.001),
        slippage: Slippage::Percentage(0.001),
        symbols: vec!["TSLA".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(150),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = MomentumStrategy::new(10, 5.0);

    let result = engine
        .run(&mut strategy)
        .await
        .expect("Backtest should succeed");

    println!("\nMomentum Strategy Results:");
    println!("  Total Return: {:.2}%", result.metrics.total_return);
    println!("  Annualized Return: {:.2}%", result.metrics.annualized_return);
    println!("  Max Drawdown: {:.2}%", result.metrics.max_drawdown);
    println!("  Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
}

#[tokio::test]
async fn test_multiple_timeframes() {
    // Test that different timeframes work correctly
    let provider = MockProvider::new_with_trend("BTC", 50, 30000.0, 100.0);

    for timeframe in &[
        Timeframe::Minute1,
        Timeframe::Minute5,
        Timeframe::Hour1,
        Timeframe::Day1,
    ] {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            commission: Commission::None,
            slippage: Slippage::None,
            symbols: vec!["BTC".to_string()],
            timeframe: *timeframe,
            start_date: Utc::now() - Duration::days(50),
            end_date: Utc::now(),
        };

        let engine = BacktestEngine::new(provider.clone(), config);
        let mut strategy = SMAStrategy::new(3, 10);

        let result = engine.run(&mut strategy).await.expect("Should succeed");
        assert!(!result.equity_curve.is_empty());
    }
}

#[tokio::test]
async fn test_commission_impact() {
    let provider = MockProvider::new_with_trend("AAPL", 100, 100.0, 0.5);

    // Test with no commission
    let config_no_commission = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(100),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider.clone(), config_no_commission.clone());
    let mut strategy = SMAStrategy::new(5, 20);
    let result_no_commission = engine.run(&mut strategy).await.unwrap();

    // Test with 1% commission
    let config_high_commission = BacktestConfig {
        commission: Commission::Percentage(0.01),
        ..config_no_commission
    };

    let engine = BacktestEngine::new(provider, config_high_commission);
    let mut strategy = SMAStrategy::new(5, 20);
    let result_high_commission = engine.run(&mut strategy).await.unwrap();

    // Commission should reduce returns
    assert!(
        result_no_commission.metrics.total_return
            > result_high_commission.metrics.total_return
    );
    assert!(
        result_high_commission.metrics.total_commission
            > result_no_commission.metrics.total_commission
    );
}

#[tokio::test]
async fn test_slippage_impact() {
    let provider = MockProvider::new_with_trend("AAPL", 100, 100.0, 0.5);

    // Test with no slippage
    let config_no_slippage = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(100),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider.clone(), config_no_slippage.clone());
    let mut strategy = SMAStrategy::new(5, 20);
    let result_no_slippage = engine.run(&mut strategy).await.unwrap();

    // Test with 1% slippage
    let config_high_slippage = BacktestConfig {
        slippage: Slippage::Percentage(0.01),
        ..config_no_slippage
    };

    let engine = BacktestEngine::new(provider, config_high_slippage);
    let mut strategy = SMAStrategy::new(5, 20);
    let result_high_slippage = engine.run(&mut strategy).await.unwrap();

    // Slippage should reduce returns
    assert!(
        result_no_slippage.metrics.total_return > result_high_slippage.metrics.total_return
    );
}

#[tokio::test]
async fn test_performance_metrics_calculation() {
    let provider = MockProvider::new_with_trend("AAPL", 365, 100.0, 0.2);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Fixed(5.0),
        slippage: Slippage::Percentage(0.001),
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(365),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = SMAStrategy::new(10, 30);

    let result = engine.run(&mut strategy).await.unwrap();

    // Verify all metrics are calculated
    assert!(result.metrics.total_return.is_finite());
    assert!(result.metrics.annualized_return.is_finite());
    assert!(result.metrics.sharpe_ratio.is_finite());
    assert!(result.metrics.max_drawdown >= 0.0);
    assert!(result.metrics.win_rate >= 0.0 && result.metrics.win_rate <= 100.0);
    assert!(result.metrics.profit_factor >= 0.0);

    // Verify consistency
    // Note: winning_trades + losing_trades counts round-trip trades (buy-sell pairs),
    // while total_trades counts all individual trades
    assert!(
        result.metrics.winning_trades + result.metrics.losing_trades <= result.metrics.total_trades
    );
    assert_eq!(result.metrics.initial_capital, 100_000.0);
}

#[tokio::test]
async fn test_equity_curve_generation() {
    let provider = MockProvider::new_with_trend("AAPL", 50, 100.0, 1.0);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(50),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = SMAStrategy::new(5, 15);

    let result = engine.run(&mut strategy).await.unwrap();

    // Verify equity curve
    assert!(!result.equity_curve.is_empty());
    assert!(result.equity_curve.len() <= 50); // Should have at most one point per bar

    // Verify equity curve is chronological
    for i in 1..result.equity_curve.len() {
        assert!(result.equity_curve[i].0 >= result.equity_curve[i - 1].0);
    }

    // Final equity in curve should match metrics
    if let Some((_, final_equity)) = result.equity_curve.last() {
        assert!((final_equity - result.metrics.final_equity).abs() < 0.01);
    }
}

#[tokio::test]
async fn test_close_all_signal() {
    let provider = MockProvider::new_with_trend("AAPL", 20, 100.0, 1.0);

    struct CloseAllStrategy {
        bar_count: usize,
    }

    #[async_trait]
    impl Strategy for CloseAllStrategy {
        async fn initialize(&mut self) {
            self.bar_count = 0;
        }

        async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
            self.bar_count += 1;

            if self.bar_count == 5 {
                // Buy on bar 5
                vec![Signal::Buy {
                    symbol: bar.symbol.clone(),
                    quantity: 100.0,
                    price: None,
                }]
            } else if self.bar_count == 10 {
                // Close all on bar 10
                vec![Signal::CloseAll]
            } else {
                vec![]
            }
        }

        async fn finalize(&mut self) {}
    }

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::None,
        slippage: Slippage::None,
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(20),
        end_date: Utc::now(),
    };

    let engine = BacktestEngine::new(provider, config);
    let mut strategy = CloseAllStrategy { bar_count: 0 };

    let result = engine.run(&mut strategy).await.unwrap();

    // Should have 2 trades: 1 buy, 1 sell (from CloseAll)
    assert_eq!(result.trades.len(), 2);
    assert!(result.portfolio.positions.is_empty());
}
