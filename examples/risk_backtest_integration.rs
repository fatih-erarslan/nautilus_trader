//! Example showing integration of risk management with backtesting framework
//!
//! This demonstrates how to use the RiskManager with the BacktestEngine
//! to implement risk-aware trading strategies.

use hyperphysics_market::risk::{
    RiskManager, RiskConfig, PositionSizingStrategy, StopLossType, Position as RiskPosition,
};
use hyperphysics_market::backtest::{
    Strategy, BacktestEngine, BacktestConfig, Signal, Bar, Side,
};
use hyperphysics_market::data::Timeframe;
use chrono::Utc;

/// A trend-following strategy with integrated risk management
struct RiskAwareTrendStrategy {
    risk_manager: RiskManager,
    lookback_period: usize,
    bars: Vec<Bar>,
}

impl RiskAwareTrendStrategy {
    fn new(initial_capital: f64) -> Self {
        let config = RiskConfig::default()
            .with_max_position_size(0.1)   // Max 10% per position
            .with_max_drawdown(0.2)        // 20% max drawdown
            .with_max_daily_loss(0.05)     // 5% max daily loss
            .with_risk_free_rate(0.02);    // 2% risk-free rate

        Self {
            risk_manager: RiskManager::new(initial_capital, config),
            lookback_period: 20,
            bars: Vec::new(),
        }
    }

    /// Calculate Average True Range (ATR)
    fn calculate_atr(&self, period: usize) -> f64 {
        if self.bars.len() < period + 1 {
            return 1.0; // Default ATR
        }

        let mut true_ranges = Vec::new();
        for i in 1..=period.min(self.bars.len() - 1) {
            let idx = self.bars.len() - i;
            let current = &self.bars[idx];
            let previous = &self.bars[idx - 1];

            let tr = (current.high - current.low)
                .max((current.high - previous.close).abs())
                .max((current.low - previous.close).abs());

            true_ranges.push(tr);
        }

        true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&self, period: usize) -> Option<f64> {
        if self.bars.len() < period {
            return None;
        }

        let sum: f64 = self.bars.iter()
            .rev()
            .take(period)
            .map(|b| b.close)
            .sum();

        Some(sum / period as f64)
    }

    /// Determine trend direction
    fn get_trend(&self) -> Option<Side> {
        let fast_ma = self.calculate_sma(5)?;
        let slow_ma = self.calculate_sma(20)?;

        if fast_ma > slow_ma {
            Some(Side::Long)
        } else if fast_ma < slow_ma {
            Some(Side::Short)
        } else {
            None
        }
    }
}

impl Strategy for RiskAwareTrendStrategy {
    fn on_bar(&mut self, bar: &Bar) -> Option<Signal> {
        // Store bar for calculations
        self.bars.push(bar.clone());
        if self.bars.len() > 100 {
            self.bars.remove(0);
        }

        // Need minimum bars for indicators
        if self.bars.len() < self.lookback_period {
            return None;
        }

        // 1. Check risk limits first
        let violations = self.risk_manager.check_risk_limits();
        if !violations.is_empty() {
            println!("Risk limits violated at {}: {:?}", bar.timestamp, violations);
            return None; // Don't trade when limits are breached
        }

        // 2. Get trend signal
        let trend = self.get_trend()?;

        // 3. Calculate ATR for volatility-based sizing and stops
        let atr = self.calculate_atr(14);

        // 4. Calculate position size using volatility-based method
        let position_sizing = PositionSizingStrategy::Volatility {
            atr,
            risk_per_trade: 500.0, // Risk $500 per trade
        };

        let shares = self.risk_manager.calculate_position_size(
            position_sizing,
            bar.close
        );

        // 5. Calculate stop loss using ATR
        let stop_type = StopLossType::AtrBased {
            atr,
            multiplier: 2.0, // 2 ATR stop
        };

        let is_long = matches!(trend, Side::Long);
        let stop_price = self.risk_manager.calculate_stop_loss(
            bar.close,
            stop_type,
            is_long
        );

        // 6. Calculate take profit (3:1 reward/risk ratio)
        let risk_per_share = (bar.close - stop_price).abs();
        let take_profit = if is_long {
            bar.close + (risk_per_share * 3.0)
        } else {
            bar.close - (risk_per_share * 3.0)
        };

        // 7. Add position to risk manager for tracking
        let risk_position = RiskPosition {
            symbol: bar.symbol.clone(),
            entry_price: bar.close,
            current_price: bar.close,
            quantity: shares,
            stop_loss: Some(stop_price),
            take_profit: Some(take_profit),
            entry_time: bar.timestamp,
        };

        self.risk_manager.add_position(risk_position);

        // 8. Return trading signal
        Some(Signal {
            side: trend,
            quantity: shares,
            stop_loss: Some(stop_price),
            take_profit: Some(take_profit),
        })
    }

    fn on_trade(&mut self, bar: &Bar, side: Side, quantity: f64, price: f64) {
        println!("{} - {} {} @ ${:.2}",
            bar.timestamp.format("%Y-%m-%d"),
            if matches!(side, Side::Long) { "BUY" } else { "SELL" },
            quantity,
            price
        );

        // Update position tracking
        if let Some(position) = self.risk_manager.get_positions().get(&bar.symbol) {
            println!("  Stop: ${:.2}, Target: ${:.2}",
                position.stop_loss.unwrap_or(0.0),
                position.take_profit.unwrap_or(0.0)
            );
        }
    }

    fn on_close(&mut self, bar: &Bar, side: Side, quantity: f64, price: f64, pnl: f64) {
        println!("{} - CLOSE {} @ ${:.2} (P&L: ${:.2})",
            bar.timestamp.format("%Y-%m-%d"),
            quantity,
            price,
            pnl
        );

        // Remove position from risk manager
        self.risk_manager.remove_position(&bar.symbol);

        // Update capital with realized P&L
        let current_capital = self.risk_manager.get_metrics().total_value;
        self.risk_manager.update_capital(current_capital + pnl);
    }

    fn on_day_end(&mut self, _date: chrono::NaiveDate) {
        // Reset daily tracking
        self.risk_manager.reset_daily();

        // Print daily metrics
        let metrics = self.risk_manager.get_metrics();
        println!("\nDaily Metrics:");
        println!("  Portfolio Value: ${:.2}", metrics.total_value);
        println!("  Daily P&L: ${:.2}", metrics.daily_pnl);
        println!("  Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
        println!("  Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
        println!("  Leverage: {:.2}x\n", metrics.leverage);
    }

    fn final_report(&self) {
        println!("\n=== Risk Management Report ===");

        let metrics = self.risk_manager.get_metrics();

        println!("\nPortfolio Metrics:");
        println!("  Final Value: ${:.2}", metrics.total_value);
        println!("  Total P&L: ${:.2}", metrics.total_pnl);
        println!("  Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);

        println!("\nRisk-Adjusted Performance:");
        println!("  Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
        println!("  Sortino Ratio: {:.3}", metrics.sortino_ratio);

        println!("\nRisk Metrics:");
        println!("  VaR (95%): ${:.2}", metrics.var_95);
        println!("  CVaR (95%): ${:.2}", metrics.cvar_95);
        println!("  Diversification Score: {:.3}", metrics.diversification_score);
        println!("  Final Leverage: {:.2}x", metrics.leverage);

        // Check for any final violations
        let violations = self.risk_manager.check_risk_limits();
        if violations.is_empty() {
            println!("\n✓ No risk limit violations");
        } else {
            println!("\n⚠ Risk Violations:");
            for violation in violations {
                println!("  - {}", violation);
            }
        }

        println!("\n=== End Report ===\n");
    }
}

fn main() {
    println!("=== Risk-Aware Backtesting Example ===\n");

    // Create strategy with risk management
    let strategy = RiskAwareTrendStrategy::new(100000.0);

    // Configure backtest
    let config = BacktestConfig {
        initial_capital: 100000.0,
        commission: hyperphysics_market::backtest::Commission::PerShare { rate: 0.005 },
        slippage: hyperphysics_market::backtest::Slippage::Fixed { amount: 0.01 },
    };

    // Create backtest engine
    let mut engine = BacktestEngine::new(strategy, config);

    // Generate sample data (in practice, load from data provider)
    println!("Generating sample market data...");
    let bars = generate_sample_data("AAPL", 252); // 1 year of daily data

    println!("Running backtest with risk management...\n");

    // Run backtest
    engine.run(&bars);

    // Get results
    let result = engine.get_result();

    println!("\n=== Backtest Results ===");
    println!("Total Trades: {}", result.metrics.total_trades);
    println!("Winning Trades: {}", result.metrics.winning_trades);
    println!("Win Rate: {:.2}%", result.metrics.win_rate * 100.0);
    println!("Profit Factor: {:.2}", result.metrics.profit_factor);
    println!("Average Win: ${:.2}", result.metrics.average_win);
    println!("Average Loss: ${:.2}", result.metrics.average_loss);
    println!("\nFinal Equity: ${:.2}", result.metrics.final_equity);
    println!("Total Return: {:.2}%", result.metrics.total_return * 100.0);
    println!("Max Drawdown: {:.2}%", result.metrics.max_drawdown * 100.0);
    println!("Sharpe Ratio: {:.3}", result.metrics.sharpe_ratio);

    println!("\n✓ Backtest complete with integrated risk management");
}

/// Generate sample market data for demonstration
fn generate_sample_data(symbol: &str, days: usize) -> Vec<Bar> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut bars = Vec::new();
    let mut price = 150.0;
    let start_date = Utc::now() - chrono::Duration::days(days as i64);

    for i in 0..days {
        let timestamp = start_date + chrono::Duration::days(i as i64);

        // Random walk with slight upward bias
        let change = rng.gen_range(-0.02..0.025);
        price *= 1.0 + change;

        let volatility = price * 0.02;
        let high = price + rng.gen_range(0.0..volatility);
        let low = price - rng.gen_range(0.0..volatility);
        let open = price + rng.gen_range(-volatility / 2.0..volatility / 2.0);
        let close = price;

        bars.push(Bar {
            symbol: symbol.to_string(),
            timestamp,
            open,
            high,
            low,
            close,
            volume: rng.gen_range(1000000..10000000),
            vwap: Some((high + low + close) / 3.0),
            trade_count: Some(rng.gen_range(5000..50000)),
        });
    }

    bars
}
