//! Backtest runner for HyperPhysics strategies.

use crate::backtest::{BacktestConfig, BacktestResults, MarketDataEvent, SlippageModel};
use crate::config::IntegrationConfig;
use crate::error::{IntegrationError, Result};
use crate::strategy::HyperPhysicsStrategy;
use crate::types::{HyperPhysicsOrderCommand, OrderSide};
use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, info};

/// Position tracking for backtest
#[derive(Debug, Clone, Default)]
struct Position {
    /// Current position size (positive = long, negative = short)
    size: f64,
    /// Average entry price
    avg_entry_price: f64,
    /// Unrealized PnL
    unrealized_pnl: f64,
    /// Realized PnL
    realized_pnl: f64,
}

/// Trade record
#[derive(Debug, Clone)]
struct TradeRecord {
    /// Entry timestamp
    entry_ts: u64,
    /// Exit timestamp
    exit_ts: u64,
    /// Entry price
    entry_price: f64,
    /// Exit price
    exit_price: f64,
    /// Trade size
    size: f64,
    /// Trade PnL
    pnl: f64,
    /// Commission paid
    commission: f64,
}

/// Backtest runner for HyperPhysics strategies.
///
/// This runner provides a simple backtest environment that can be used
/// standalone or integrated with Nautilus Trader's backtest engine.
pub struct BacktestRunner {
    /// Backtest configuration
    config: BacktestConfig,
    /// Strategy instance
    strategy: HyperPhysicsStrategy,
    /// Current equity
    equity: f64,
    /// Peak equity (for drawdown calculation)
    peak_equity: f64,
    /// Current position
    position: Position,
    /// Current market price
    current_price: f64,
    /// Trade history
    trades: Vec<TradeRecord>,
    /// Equity curve
    equity_curve: Vec<(u64, f64)>,
    /// Returns for Sharpe calculation
    returns: VecDeque<f64>,
    /// Last equity for return calculation
    last_equity: f64,
}

impl BacktestRunner {
    /// Create a new backtest runner
    pub async fn new(
        backtest_config: BacktestConfig,
        strategy_config: IntegrationConfig,
    ) -> Result<Self> {
        let strategy = HyperPhysicsStrategy::new(strategy_config).await?;

        Ok(Self {
            equity: backtest_config.initial_capital,
            peak_equity: backtest_config.initial_capital,
            last_equity: backtest_config.initial_capital,
            config: backtest_config,
            strategy,
            position: Position::default(),
            current_price: 0.0,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            returns: VecDeque::with_capacity(252), // ~1 year of daily returns
        })
    }

    /// Run backtest on market data events
    pub async fn run(&mut self, events: Vec<MarketDataEvent>) -> Result<BacktestResults> {
        let start_time = Instant::now();
        let total_events = events.len();

        info!(
            events = total_events,
            initial_capital = self.config.initial_capital,
            "Starting backtest"
        );

        // Set up strategy
        self.strategy.set_instrument("BACKTEST.SIM").await;
        self.strategy.start().await?;

        // Process events
        for (i, event) in events.into_iter().enumerate() {
            // Filter by time range
            let ts = event.timestamp();
            if ts < self.config.start_time_ns || ts > self.config.end_time_ns {
                continue;
            }

            // Process event
            let order = match event {
                MarketDataEvent::Quote(ref q) => {
                    self.current_price = (q.bid_price + q.ask_price) as f64 / 2.0
                        / 10_f64.powi(q.price_precision as i32);
                    self.strategy.on_quote(q).await?
                }
                MarketDataEvent::Trade(ref t) => {
                    self.current_price = t.price as f64 / 10_f64.powi(t.price_precision as i32);
                    self.strategy.on_trade(t).await?
                }
                MarketDataEvent::Bar(ref b) => {
                    self.current_price = b.close as f64 / 10_f64.powi(b.price_precision as i32);
                    self.strategy.on_bar(b).await?
                }
            };

            // Execute order if generated
            if let Some(cmd) = order {
                self.execute_order(&cmd, ts)?;
            }

            // Update unrealized PnL and equity
            self.update_equity(ts);

            // Log progress
            if self.config.verbose && i % 10000 == 0 {
                debug!(
                    progress = format!("{:.1}%", (i as f64 / total_events as f64) * 100.0),
                    equity = self.equity,
                    position = self.position.size,
                    "Backtest progress"
                );
            }
        }

        // Close any open position at end
        if self.position.size.abs() > 0.0001 {
            self.close_position(self.config.end_time_ns)?;
        }

        self.strategy.stop().await?;

        let runtime = start_time.elapsed().as_secs_f64();
        let results = self.calculate_results(runtime);

        info!(
            total_return = format!("{:.2}%", results.total_return * 100.0),
            sharpe = format!("{:.2}", results.sharpe_ratio),
            max_dd = format!("{:.2}%", results.max_drawdown * 100.0),
            trades = results.total_trades,
            runtime_secs = format!("{:.2}", runtime),
            "Backtest complete"
        );

        Ok(results)
    }

    /// Execute an order
    fn execute_order(&mut self, cmd: &HyperPhysicsOrderCommand, ts: u64) -> Result<()> {
        // Apply slippage
        let slippage = self.calculate_slippage();
        let exec_price = match cmd.side {
            OrderSide::Buy => self.current_price * (1.0 + slippage),
            OrderSide::Sell => self.current_price * (1.0 - slippage),
            OrderSide::NoSide => return Ok(()),
        };

        // Calculate commission
        let trade_value = exec_price * cmd.quantity;
        let commission = trade_value * self.config.commission_rate;

        // Update position
        let prev_size = self.position.size;
        let trade_size = match cmd.side {
            OrderSide::Buy => cmd.quantity,
            OrderSide::Sell => -cmd.quantity,
            OrderSide::NoSide => 0.0,
        };

        // Check if closing existing position
        if (prev_size > 0.0 && trade_size < 0.0) || (prev_size < 0.0 && trade_size > 0.0) {
            // Closing trade
            let close_size = trade_size.abs().min(prev_size.abs());
            let pnl = if prev_size > 0.0 {
                (exec_price - self.position.avg_entry_price) * close_size
            } else {
                (self.position.avg_entry_price - exec_price) * close_size
            };

            self.position.realized_pnl += pnl - commission;
            self.equity += pnl - commission;

            // Record trade
            self.trades.push(TradeRecord {
                entry_ts: 0, // Would need tracking
                exit_ts: ts,
                entry_price: self.position.avg_entry_price,
                exit_price: exec_price,
                size: close_size,
                pnl,
                commission,
            });

            // Update position size
            self.position.size += trade_size;
            if self.position.size.abs() < 0.0001 {
                self.position.size = 0.0;
                self.position.avg_entry_price = 0.0;
            }
        } else {
            // Opening or adding to position
            let new_size = prev_size + trade_size;
            if prev_size.abs() < 0.0001 {
                // New position
                self.position.avg_entry_price = exec_price;
            } else {
                // Adding to existing
                self.position.avg_entry_price =
                    (self.position.avg_entry_price * prev_size.abs() + exec_price * trade_size.abs())
                    / (prev_size.abs() + trade_size.abs());
            }
            self.position.size = new_size;
            self.equity -= commission;
        }

        debug!(
            side = ?cmd.side,
            size = cmd.quantity,
            price = exec_price,
            commission = commission,
            position = self.position.size,
            "Executed order"
        );

        Ok(())
    }

    /// Close open position
    fn close_position(&mut self, ts: u64) -> Result<()> {
        if self.position.size.abs() < 0.0001 {
            return Ok(());
        }

        let cmd = HyperPhysicsOrderCommand {
            client_order_id: format!("CLOSE-{}", ts),
            instrument_id: "BACKTEST.SIM".to_string(),
            side: if self.position.size > 0.0 { OrderSide::Sell } else { OrderSide::Buy },
            order_type: crate::types::OrderType::Market,
            quantity: self.position.size.abs(),
            price: None,
            time_in_force: crate::types::TimeInForce::IOC,
            reduce_only: true,
            post_only: false,
            hp_confidence: 1.0,
            hp_algorithm: "Close".to_string(),
            hp_latency_us: 0,
            hp_consensus_term: 0,
        };

        self.execute_order(&cmd, ts)
    }

    /// Calculate slippage based on model
    fn calculate_slippage(&self) -> f64 {
        match &self.config.slippage_model {
            SlippageModel::None => 0.0,
            SlippageModel::FixedBps(bps) => bps / 10000.0,
            SlippageModel::VolatilityBased { multiplier } => {
                // Would need volatility from strategy
                multiplier * 0.001 // Default assumption
            }
        }
    }

    /// Update equity and track returns
    fn update_equity(&mut self, ts: u64) {
        // Calculate unrealized PnL
        if self.position.size.abs() > 0.0001 {
            self.position.unrealized_pnl = if self.position.size > 0.0 {
                (self.current_price - self.position.avg_entry_price) * self.position.size
            } else {
                (self.position.avg_entry_price - self.current_price) * self.position.size.abs()
            };
        } else {
            self.position.unrealized_pnl = 0.0;
        }

        let current_equity = self.config.initial_capital
            + self.position.realized_pnl
            + self.position.unrealized_pnl;

        self.equity = current_equity;

        // Track peak for drawdown
        if self.equity > self.peak_equity {
            self.peak_equity = self.equity;
        }

        // Record equity curve point
        self.equity_curve.push((ts, self.equity));

        // Calculate return
        if self.last_equity > 0.0 {
            let ret = (self.equity - self.last_equity) / self.last_equity;
            self.returns.push_back(ret);
            if self.returns.len() > 252 {
                self.returns.pop_front();
            }
        }
        self.last_equity = self.equity;
    }

    /// Calculate final results
    fn calculate_results(&self, runtime: f64) -> BacktestResults {
        let total_return = (self.equity - self.config.initial_capital) / self.config.initial_capital;

        // Sharpe ratio
        let sharpe = if self.returns.len() > 1 {
            let mean: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
            let variance: f64 = self.returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (self.returns.len() - 1) as f64;
            let std = variance.sqrt();
            if std > 0.0 {
                (mean / std) * (252_f64).sqrt() // Annualized
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Max drawdown
        let max_drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - self.equity_curve.iter()
                .map(|(_, e)| *e)
                .fold(f64::INFINITY, f64::min))
                / self.peak_equity
        } else {
            0.0
        };

        // Trade statistics
        let (wins, losses): (Vec<_>, Vec<_>) = self.trades.iter()
            .partition(|t| t.pnl > 0.0);
        let win_rate = if !self.trades.is_empty() {
            wins.len() as f64 / self.trades.len() as f64
        } else {
            0.0
        };

        let total_profits: f64 = wins.iter().map(|t| t.pnl).sum();
        let total_losses: f64 = losses.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if total_losses > 0.0 {
            total_profits / total_losses
        } else if total_profits > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if !self.trades.is_empty() {
            self.trades.iter().map(|t| t.pnl).sum::<f64>() / self.trades.len() as f64
        } else {
            0.0
        };

        BacktestResults {
            total_return,
            sharpe_ratio: sharpe,
            max_drawdown,
            total_trades: self.trades.len() as u64,
            win_rate,
            avg_trade_return,
            profit_factor,
            final_equity: self.equity,
            avg_latency_us: 0.0, // Would come from strategy metrics
            backtest_runtime_secs: runtime,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NautilusQuoteTick;

    #[tokio::test]
    async fn test_backtest_runner_creation() {
        let bt_config = BacktestConfig::default();
        let strategy_config = IntegrationConfig::backtest();

        let runner = BacktestRunner::new(bt_config, strategy_config).await;
        assert!(runner.is_ok());
    }

    #[tokio::test]
    async fn test_empty_backtest() {
        let bt_config = BacktestConfig::default();
        let strategy_config = IntegrationConfig::backtest();

        let mut runner = BacktestRunner::new(bt_config.clone(), strategy_config).await.unwrap();
        let results = runner.run(vec![]).await.unwrap();

        assert_eq!(results.total_trades, 0);
        assert!((results.final_equity - bt_config.initial_capital).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_backtest_with_quotes() {
        let bt_config = BacktestConfig {
            initial_capital: 10000.0,
            ..Default::default()
        };
        let strategy_config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..IntegrationConfig::backtest()
        };

        let mut runner = BacktestRunner::new(bt_config, strategy_config).await.unwrap();

        // Create some test quotes
        let events: Vec<MarketDataEvent> = (0..10)
            .map(|i| {
                MarketDataEvent::Quote(NautilusQuoteTick {
                    instrument_id: 1,
                    bid_price: 10000 + i * 10,
                    ask_price: 10010 + i * 10,
                    bid_size: 100,
                    ask_size: 100,
                    price_precision: 2,
                    size_precision: 0,
                    ts_event: 1000000000 + i as u64 * 1000000,
                    ts_init: 1000000000 + i as u64 * 1000000,
                })
            })
            .collect();

        let results = runner.run(events).await.unwrap();

        // Results should be calculated
        assert!(results.backtest_runtime_secs > 0.0);
    }
}
