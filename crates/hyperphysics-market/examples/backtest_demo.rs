//! Comprehensive backtesting framework demonstration
//!
//! Run with: `cargo run --example backtest_demo --features=examples`

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
// Mock Data Provider
// ============================================================================

struct HistoricalProvider {
    bars: Vec<Bar>,
}

impl HistoricalProvider {
    fn new_trending_market(symbol: &str, days: usize, start_price: f64, trend: f64) -> Self {
        let mut bars = Vec::new();
        let start_time = Utc::now() - Duration::days(days as i64);

        for i in 0..days {
            let timestamp = start_time + Duration::days(i as i64);
            let base_price = start_price + (i as f64 * trend);

            // Add some randomness (simple sine wave for demo)
            let daily_volatility = base_price * 0.02;
            let noise = (i as f64).sin() * daily_volatility;

            let open = base_price + noise;
            let close = base_price + noise * 0.8;
            let high = base_price + daily_volatility;
            let low = base_price - daily_volatility;
            let volume = 1_000_000 + (i * 10_000) as u64;

            bars.push(Bar::new(
                symbol.to_string(),
                timestamp,
                open,
                high,
                low,
                close,
                volume,
            ));
        }

        Self { bars }
    }
}

#[async_trait]
impl MarketDataProvider for HistoricalProvider {
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
        "Historical"
    }

    async fn supports_symbol(&self, _symbol: &str) -> MarketResult<bool> {
        Ok(true)
    }
}

// ============================================================================
// Example Strategy: Dual Moving Average Crossover
// ============================================================================

struct DualMovingAverageStrategy {
    fast_period: usize,
    slow_period: usize,
    prices: Vec<f64>,
    position_open: bool,
}

impl DualMovingAverageStrategy {
    fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
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
impl Strategy for DualMovingAverageStrategy {
    async fn initialize(&mut self) {
        println!("Initializing Dual Moving Average Strategy");
        println!("  Fast MA: {} periods", self.fast_period);
        println!("  Slow MA: {} periods", self.slow_period);
        self.prices.clear();
        self.position_open = false;
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let fast_ma = self.calculate_sma(self.fast_period);
        let slow_ma = self.calculate_sma(self.slow_period);

        if fast_ma.is_none() || slow_ma.is_none() {
            return vec![];
        }

        let fast_ma = fast_ma.unwrap();
        let slow_ma = slow_ma.unwrap();

        // Golden Cross: Fast MA crosses above Slow MA
        if fast_ma > slow_ma && !self.position_open {
            println!(
                "  [{}] BUY SIGNAL: Fast MA ({:.2}) > Slow MA ({:.2}) @ ${:.2}",
                bar.timestamp.format("%Y-%m-%d"),
                fast_ma,
                slow_ma,
                bar.close
            );
            self.position_open = true;
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Death Cross: Fast MA crosses below Slow MA
        if fast_ma < slow_ma && self.position_open {
            println!(
                "  [{}] SELL SIGNAL: Fast MA ({:.2}) < Slow MA ({:.2}) @ ${:.2}",
                bar.timestamp.format("%Y-%m-%d"),
                fast_ma,
                slow_ma,
                bar.close
            );
            self.position_open = false;
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }

    async fn finalize(&mut self) {
        println!("Strategy execution complete");
    }

    fn name(&self) -> String {
        format!("DualMA({},{})", self.fast_period, self.slow_period)
    }
}

// ============================================================================
// Main Demo
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=".repeat(80));
    println!("HyperPhysics Backtesting Framework Demo");
    println!("=".repeat(80));
    println!();

    // Create historical data provider with an uptrend
    let provider = HistoricalProvider::new_trending_market("AAPL", 252, 150.0, 0.2);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Percentage(0.001), // 0.1% per trade
        slippage: Slippage::Percentage(0.0005),    // 0.05% slippage
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(252),
        end_date: Utc::now(),
    };

    println!("Backtest Configuration:");
    println!("  Initial Capital: ${:.2}", config.initial_capital);
    println!("  Commission: 0.1% per trade");
    println!("  Slippage: 0.05%");
    println!("  Timeframe: Daily");
    println!("  Duration: {} days", 252);
    println!();

    // Create strategy
    let mut strategy = DualMovingAverageStrategy::new(20, 50);

    // Run backtest
    println!("Running backtest...");
    println!();

    let engine = BacktestEngine::new(provider, config);
    let result = engine.run(&mut strategy).await?;

    // Display results
    println!();
    println!("=".repeat(80));
    println!("BACKTEST RESULTS");
    println!("=".repeat(80));
    println!();

    println!("Performance Metrics:");
    println!("  Initial Capital:    ${:>12.2}", result.metrics.initial_capital);
    println!("  Final Equity:       ${:>12.2}", result.metrics.final_equity);
    println!("  Total Return:       {:>12.2}%", result.metrics.total_return);
    println!("  Annualized Return:  {:>12.2}%", result.metrics.annualized_return);
    println!();

    println!("Risk Metrics:");
    println!("  Sharpe Ratio:       {:>12.2}", result.metrics.sharpe_ratio);
    println!("  Max Drawdown:       {:>12.2}%", result.metrics.max_drawdown);
    println!();

    println!("Trade Statistics:");
    println!("  Total Trades:       {:>12}", result.metrics.total_trades);
    println!("  Winning Trades:     {:>12}", result.metrics.winning_trades);
    println!("  Losing Trades:      {:>12}", result.metrics.losing_trades);
    println!("  Win Rate:           {:>12.2}%", result.metrics.win_rate);
    println!();

    println!("P&L Analysis:");
    println!("  Average Win:        ${:>12.2}", result.metrics.avg_win);
    println!("  Average Loss:       ${:>12.2}", result.metrics.avg_loss);
    println!("  Profit Factor:      {:>12.2}", result.metrics.profit_factor);
    println!("  Total Commission:   ${:>12.2}", result.metrics.total_commission);
    println!();

    // Display trade log
    println!("=".repeat(80));
    println!("TRADE LOG");
    println!("=".repeat(80));
    println!();

    for (i, trade) in result.trades.iter().enumerate() {
        let side_str = match trade.side {
            hyperphysics_market::backtest::Side::Buy => "BUY ",
            hyperphysics_market::backtest::Side::Sell => "SELL",
        };

        println!(
            "{:>3}. {} | {} | Qty: {:>6.0} @ ${:>8.2} | Commission: ${:>6.2}",
            i + 1,
            trade.timestamp.format("%Y-%m-%d"),
            side_str,
            trade.quantity,
            trade.price,
            trade.commission
        );
    }

    println!();

    // Display equity curve sample
    println!("=".repeat(80));
    println!("EQUITY CURVE (Sample - First 10 points)");
    println!("=".repeat(80));
    println!();

    for (i, (timestamp, equity)) in result.equity_curve.iter().take(10).enumerate() {
        let pnl_pct = (equity - result.metrics.initial_capital)
            / result.metrics.initial_capital * 100.0;
        println!(
            "{:>3}. {} | Equity: ${:>12.2} | P&L: {:>8.2}%",
            i + 1,
            timestamp.format("%Y-%m-%d"),
            equity,
            pnl_pct
        );
    }

    println!();
    println!("=".repeat(80));
    println!("Backtest Complete!");
    println!("=".repeat(80));

    Ok(())
}
