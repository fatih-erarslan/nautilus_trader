//! Backtesting Engine
//!
//! Comprehensive backtesting framework with:
//! - Historical data replay
//! - Realistic slippage modeling
//! - Commission calculation
//! - Performance metrics
//! - Walk-forward analysis

use crate::{Result, Strategy, Signal, StrategyError, Bar, MarketData, Portfolio, Direction, Position};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tracing::{debug, info};
use chrono::{DateTime, Utc};

use super::slippage::SlippageModel;
use super::performance::PerformanceMetrics;

/// Backtesting engine
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: Decimal,
    /// Commission per trade (percentage)
    commission_rate: f64,
    /// Slippage model
    slippage: SlippageModel,
    /// Track all trades
    trades: Vec<Trade>,
    /// Track equity curve
    equity_curve: Vec<EquityPoint>,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            initial_capital,
            commission_rate: 0.001, // 0.1% default
            slippage: SlippageModel::default(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
        }
    }

    /// Set commission rate
    pub fn with_commission(mut self, rate: f64) -> Self {
        self.commission_rate = rate;
        self
    }

    /// Set slippage model
    pub fn with_slippage(mut self, slippage: SlippageModel) -> Self {
        self.slippage = slippage;
        self
    }

    /// Run backtest on historical data
    pub async fn run<S: Strategy>(
        &mut self,
        strategy: &S,
        historical_data: HashMap<String, Vec<Bar>>,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<BacktestResult> {
        info!(
            "Starting backtest: {} from {} to {}",
            strategy.id(),
            start_date,
            end_date
        );

        // Initialize portfolio
        let mut portfolio = Portfolio::new(self.initial_capital);
        let mut current_time = start_date;

        // Record initial equity
        self.equity_curve.push(EquityPoint {
            timestamp: current_time,
            equity: self.initial_capital,
            cash: self.initial_capital,
            positions_value: Decimal::ZERO,
        });

        // Get all timestamps (union of all symbols)
        let mut all_timestamps = self.collect_timestamps(&historical_data);
        all_timestamps.sort();
        all_timestamps.retain(|t| *t >= start_date && *t <= end_date);

        info!("Backtesting over {} timestamps", all_timestamps.len());

        // Replay historical data
        for timestamp in all_timestamps {
            current_time = timestamp;

            // Process each symbol
            for (symbol, bars) in &historical_data {
                // Get bars up to current time
                let relevant_bars: Vec<Bar> = bars
                    .iter()
                    .filter(|b| b.timestamp <= current_time)
                    .cloned()
                    .collect();

                if relevant_bars.is_empty() {
                    continue;
                }

                // Create market data
                let current_bar = relevant_bars.last().unwrap().clone();
                let market_data = MarketData {
                    symbol: symbol.clone(),
                    timestamp: current_time,
                    price: Some(current_bar.close),
                    volume: Some(current_bar.volume),
                    bars: relevant_bars,
                };

                // Generate signals
                match strategy.process(&market_data, &portfolio).await {
                    Ok(signals) => {
                        for signal in signals {
                            self.execute_signal(&signal, &current_bar, &mut portfolio)?;
                        }
                    }
                    Err(e) => {
                        debug!("Strategy error for {}: {}", symbol, e);
                        continue;
                    }
                }
            }

            // Update portfolio value
            self.update_portfolio_value(&mut portfolio, &historical_data, current_time)?;

            // Record equity point
            self.equity_curve.push(EquityPoint {
                timestamp: current_time,
                equity: portfolio.total_value(),
                cash: portfolio.cash(),
                positions_value: portfolio.total_value() - portfolio.cash(),
            });
        }

        // Close all positions at end
        self.close_all_positions(&mut portfolio, &historical_data, end_date)?;

        // Calculate performance metrics
        let metrics = self.calculate_metrics(&portfolio);

        Ok(BacktestResult {
            initial_capital: self.initial_capital,
            final_equity: portfolio.total_value(),
            total_return: self.calculate_total_return(portfolio.total_value()),
            metrics,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            num_trades: self.trades.len(),
        })
    }

    /// Execute signal in backtest
    fn execute_signal(
        &mut self,
        signal: &Signal,
        current_bar: &Bar,
        portfolio: &mut Portfolio,
    ) -> Result<()> {
        let quantity = signal.quantity.unwrap_or(1) as i64;
        let price = signal.entry_price.unwrap_or(current_bar.close);

        // Apply slippage
        let execution_price = self.slippage.apply_slippage(
            price,
            signal.direction,
            quantity.unsigned_abs(),
            current_bar.volume.to_u64().unwrap_or(0),
        );

        // Calculate commission
        let trade_value = execution_price * Decimal::from(quantity.abs());
        let commission = trade_value * Decimal::from_f64_retain(self.commission_rate).unwrap();

        match signal.direction {
            Direction::Long => {
                let total_cost = trade_value + commission;
                if portfolio.cash() >= total_cost {
                    portfolio.update_cash(-total_cost);
                    let position = Position {
                        symbol: signal.symbol.clone(),
                        quantity,
                        avg_price: execution_price,
                        current_price: execution_price,
                        market_value: trade_value,
                        unrealized_pnl: Decimal::ZERO,
                    };
                    portfolio.update_position(signal.symbol.clone(), position);

                    self.trades.push(Trade {
                        timestamp: current_bar.timestamp,
                        symbol: signal.symbol.clone(),
                        side: TradeSide::Buy,
                        quantity: quantity as u32,
                        price: execution_price,
                        commission,
                        pnl: Decimal::ZERO,
                    });
                }
            }
            Direction::Short => {
                // For backtest, treat short as sell (close long position)
                let total_proceeds = trade_value - commission;
                portfolio.update_cash(total_proceeds);
                let position = Position {
                    symbol: signal.symbol.clone(),
                    quantity: -quantity,
                    avg_price: execution_price,
                    current_price: execution_price,
                    market_value: -trade_value,
                    unrealized_pnl: Decimal::ZERO,
                };
                portfolio.update_position(signal.symbol.clone(), position);

                self.trades.push(Trade {
                    timestamp: current_bar.timestamp,
                    symbol: signal.symbol.clone(),
                    side: TradeSide::Sell,
                    quantity: quantity as u32,
                    price: execution_price,
                    commission,
                    pnl: Decimal::ZERO, // Calculate later
                });
            }
            Direction::Close => {
                // Close existing position
                if let Some(position) = portfolio.get_position(&signal.symbol).cloned() {
                    let close_quantity = position.quantity.abs();
                    let side = if position.quantity > 0 {
                        TradeSide::Sell
                    } else {
                        TradeSide::Buy
                    };

                    let total_value = execution_price * Decimal::from(close_quantity);
                    let pnl = self.calculate_pnl(position.quantity, position.avg_price, execution_price);

                    if side == TradeSide::Sell {
                        portfolio.update_cash(total_value - commission);
                    } else {
                        portfolio.update_cash(-(total_value + commission));
                    }

                    let closed_position = Position {
                        symbol: signal.symbol.clone(),
                        quantity: 0,
                        avg_price: Decimal::ZERO,
                        current_price: execution_price,
                        market_value: Decimal::ZERO,
                        unrealized_pnl: Decimal::ZERO,
                    };
                    portfolio.update_position(signal.symbol.clone(), closed_position);

                    self.trades.push(Trade {
                        timestamp: current_bar.timestamp,
                        symbol: signal.symbol.clone(),
                        side,
                        quantity: close_quantity as u32,
                        price: execution_price,
                        commission,
                        pnl,
                    });
                }
            }
        }

        Ok(())
    }

    /// Calculate P&L
    fn calculate_pnl(&self, quantity: i64, entry_price: Decimal, exit_price: Decimal) -> Decimal {
        if quantity > 0 {
            // Long position
            (exit_price - entry_price) * Decimal::from(quantity)
        } else {
            // Short position
            (entry_price - exit_price) * Decimal::from(quantity.abs())
        }
    }

    /// Update portfolio value based on current prices
    fn update_portfolio_value(
        &self,
        portfolio: &mut Portfolio,
        historical_data: &HashMap<String, Vec<Bar>>,
        current_time: DateTime<Utc>,
    ) -> Result<()> {
        let symbols: Vec<_> = portfolio.positions().keys().cloned().collect();
        for symbol in symbols {
            if let Some(bars) = historical_data.get(&symbol) {
                if let Some(current_bar) = bars.iter().rfind(|b| b.timestamp <= current_time) {
                    portfolio.update_position_price(&symbol, current_bar.close);
                }
            }
        }
        Ok(())
    }

    /// Close all open positions
    fn close_all_positions(
        &mut self,
        portfolio: &mut Portfolio,
        historical_data: &HashMap<String, Vec<Bar>>,
        end_date: DateTime<Utc>,
    ) -> Result<()> {
        let positions: Vec<_> = portfolio.positions().keys().cloned().collect();

        for symbol in positions {
            if let Some(bars) = historical_data.get(&symbol) {
                if let Some(last_bar) = bars.iter().rfind(|b| b.timestamp <= end_date) {
                    if let Some(position) = portfolio.get_position(&symbol) {
                        let close_price = last_bar.close;
                        let pnl = self.calculate_pnl(position.quantity, position.avg_price, close_price);

                        self.trades.push(Trade {
                            timestamp: end_date,
                            symbol: symbol.clone(),
                            side: if position.quantity > 0 { TradeSide::Sell } else { TradeSide::Buy },
                            quantity: position.quantity.unsigned_abs() as u32,
                            price: close_price,
                            commission: Decimal::ZERO,
                            pnl,
                        });

                        let closed_position = Position {
                            symbol: symbol.clone(),
                            quantity: 0,
                            avg_price: Decimal::ZERO,
                            current_price: close_price,
                            market_value: Decimal::ZERO,
                            unrealized_pnl: Decimal::ZERO,
                        };
                        portfolio.update_position(symbol.clone(), closed_position);
                    }
                }
            }
        }

        Ok(())
    }

    /// Collect all unique timestamps
    fn collect_timestamps(&self, historical_data: &HashMap<String, Vec<Bar>>) -> Vec<DateTime<Utc>> {
        let mut timestamps = Vec::new();
        for bars in historical_data.values() {
            for bar in bars {
                timestamps.push(bar.timestamp);
            }
        }
        timestamps.sort();
        timestamps.dedup();
        timestamps
    }

    /// Calculate total return
    fn calculate_total_return(&self, final_equity: Decimal) -> f64 {
        let return_pct = (final_equity - self.initial_capital) / self.initial_capital;
        return_pct.to_f64().unwrap()
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, portfolio: &Portfolio) -> PerformanceMetrics {
        PerformanceMetrics::calculate(&self.equity_curve, &self.trades)
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub initial_capital: Decimal,
    pub final_equity: Decimal,
    pub total_return: f64,
    pub metrics: PerformanceMetrics,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<EquityPoint>,
    pub num_trades: usize,
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: u32,
    pub price: Decimal,
    pub commission: Decimal,
    pub pnl: Decimal,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Equity curve point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub timestamp: DateTime<Utc>,
    pub equity: Decimal,
    pub cash: Decimal,
    pub positions_value: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::momentum::MomentumStrategy;

    fn create_test_bars(symbol: &str, count: usize, start_price: f64) -> Vec<Bar> {
        (0..count)
            .map(|i| Bar {
                symbol: symbol.to_string(),
                timestamp: Utc::now() + chrono::Duration::days(i as i64),
                open: Decimal::from_f64_retain(start_price + i as f64).unwrap(),
                high: Decimal::from_f64_retain(start_price + i as f64 + 2.0).unwrap(),
                low: Decimal::from_f64_retain(start_price + i as f64 - 1.0).unwrap(),
                close: Decimal::from_f64_retain(start_price + i as f64 + 1.0).unwrap(),
                volume: 1000000,
            })
            .collect()
    }

    #[tokio::test]
    async fn test_backtest_basic() {
        let strategy = MomentumStrategy::new(vec!["TEST".to_string()], 10, 2.0, 0.5);

        let mut historical_data = HashMap::new();
        historical_data.insert("TEST".to_string(), create_test_bars("TEST", 100, 100.0));

        let start_date = Utc::now();
        let end_date = start_date + chrono::Duration::days(100);

        let mut engine = BacktestEngine::new(Decimal::from(100000));
        let result = engine
            .run(&strategy, historical_data, start_date, end_date)
            .await
            .unwrap();

        assert_eq!(result.initial_capital, Decimal::from(100000));
        assert!(!result.equity_curve.is_empty());
    }
}
