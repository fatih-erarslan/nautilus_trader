//! # Backtesting Framework
//!
//! Comprehensive event-driven backtesting engine for trading strategies.
//!
//! ## Features
//!
//! - Event-driven architecture with realistic order execution
//! - Portfolio management with position tracking
//! - Configurable slippage and commission models
//! - Performance metrics: Sharpe ratio, max drawdown, win rate, etc.
//! - Multiple timeframe support
//! - Trade log with detailed entry/exit tracking
//!
//! ## Example
//!
//! ```no_run
//! use hyperphysics_market::backtest::{Strategy, BacktestEngine, BacktestConfig, Signal};
//! use hyperphysics_market::data::Bar;
//! use async_trait::async_trait;
//!
//! struct SimpleMovingAverageStrategy {
//!     period: usize,
//!     prices: Vec<f64>,
//! }
//!
//! #[async_trait]
//! impl Strategy for SimpleMovingAverageStrategy {
//!     async fn initialize(&mut self) {
//!         self.prices.clear();
//!     }
//!
//!     async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
//!         self.prices.push(bar.close);
//!         if self.prices.len() < self.period {
//!             return vec![];
//!         }
//!
//!         let ma: f64 = self.prices.iter().rev().take(self.period).sum::<f64>()
//!             / self.period as f64;
//!
//!         if bar.close > ma {
//!             vec![Signal::Buy {
//!                 symbol: bar.symbol.clone(),
//!                 quantity: 100.0,
//!                 price: None, // Market order
//!             }]
//!         } else {
//!             vec![Signal::Sell {
//!                 symbol: bar.symbol.clone(),
//!                 quantity: 100.0,
//!                 price: None,
//!             }]
//!         }
//!     }
//!
//!     async fn finalize(&mut self) {}
//! }
//! ```

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::data::{Bar, Tick, Timeframe};
use crate::error::{MarketError, MarketResult};
use crate::providers::MarketDataProvider;

// ============================================================================
// Core Traits
// ============================================================================

/// Signal emitted by a trading strategy
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    /// Buy signal
    Buy {
        symbol: String,
        quantity: f64,
        price: Option<f64>, // None = market order
    },
    /// Sell signal
    Sell {
        symbol: String,
        quantity: f64,
        price: Option<f64>,
    },
    /// Close all positions for a symbol
    ClosePosition { symbol: String },
    /// Close all positions
    CloseAll,
}

/// Trading strategy interface
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Initialize strategy before backtesting starts
    async fn initialize(&mut self);

    /// Process a new bar and generate trading signals
    ///
    /// Called for each bar in chronological order during backtesting
    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal>;

    /// Process a new tick (optional, for tick-level strategies)
    ///
    /// Default implementation does nothing
    async fn on_tick(&mut self, _tick: &Tick) -> Vec<Signal> {
        vec![]
    }

    /// Finalize strategy after backtesting completes
    async fn finalize(&mut self);

    /// Get strategy name
    fn name(&self) -> String {
        "Strategy".to_string()
    }
}

// ============================================================================
// Portfolio Management
// ============================================================================

/// Represents a position in a single security
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Trading symbol
    pub symbol: String,
    /// Number of shares/units (positive for long, negative for short)
    pub quantity: f64,
    /// Average entry price
    pub avg_price: f64,
    /// Current market price
    pub current_price: f64,
    /// Timestamp when position was opened
    pub opened_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: String, quantity: f64, price: f64, timestamp: DateTime<Utc>) -> Self {
        Self {
            symbol,
            quantity,
            avg_price: price,
            current_price: price,
            opened_at: timestamp,
            updated_at: timestamp,
        }
    }

    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        self.quantity * (self.current_price - self.avg_price)
    }

    /// Calculate position value at current price
    pub fn market_value(&self) -> f64 {
        self.quantity * self.current_price
    }

    /// Calculate cost basis
    pub fn cost_basis(&self) -> f64 {
        self.quantity * self.avg_price
    }

    /// Update position with new trade
    pub fn update(&mut self, quantity: f64, price: f64, timestamp: DateTime<Utc>) {
        let new_quantity = self.quantity + quantity;

        if new_quantity.abs() < 1e-10 {
            // Position closed
            self.quantity = 0.0;
        } else if self.quantity.signum() != new_quantity.signum() {
            // Position reversed
            self.quantity = new_quantity;
            self.avg_price = price;
            self.opened_at = timestamp;
        } else {
            // Position increased
            let total_cost = self.quantity * self.avg_price + quantity * price;
            self.quantity = new_quantity;
            self.avg_price = total_cost / self.quantity;
        }

        self.current_price = price;
        self.updated_at = timestamp;
    }

    /// Update current market price
    pub fn update_price(&mut self, price: f64, timestamp: DateTime<Utc>) {
        self.current_price = price;
        self.updated_at = timestamp;
    }
}

/// Portfolio state tracking cash and positions
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Available cash
    pub cash: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Open positions
    pub positions: HashMap<String, Position>,
    /// Commission per trade (fixed amount or percentage)
    pub commission: Commission,
}

impl Portfolio {
    /// Create a new portfolio with initial capital
    pub fn new(initial_capital: f64, commission: Commission) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            commission,
        }
    }

    /// Get current equity (cash + position values)
    pub fn equity(&self) -> f64 {
        let position_value: f64 = self.positions.values().map(|p| p.market_value()).sum();
        self.cash + position_value
    }

    /// Get total unrealized P&L
    pub fn unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl()).sum()
    }

    /// Execute a buy order
    pub fn buy(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
    ) -> MarketResult<Trade> {
        let cost = quantity * price;
        let commission = self.commission.calculate(cost);
        let total_cost = cost + commission;

        if total_cost > self.cash {
            return Err(MarketError::ConfigError(format!(
                "Insufficient cash: need ${:.2}, have ${:.2}",
                total_cost, self.cash
            )));
        }

        self.cash -= total_cost;

        let position = self
            .positions
            .entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol.to_string(), 0.0, price, timestamp));

        position.update(quantity, price, timestamp);

        Ok(Trade {
            symbol: symbol.to_string(),
            side: Side::Buy,
            quantity,
            price,
            commission,
            timestamp,
        })
    }

    /// Execute a sell order
    pub fn sell(
        &mut self,
        symbol: &str,
        quantity: f64,
        price: f64,
        timestamp: DateTime<Utc>,
    ) -> MarketResult<Trade> {
        // Check if we have the position
        let position = self.positions.get(symbol);
        if position.is_none() || position.unwrap().quantity < quantity - 1e-10 {
            return Err(MarketError::ConfigError(format!(
                "Insufficient position: trying to sell {:.2} of {}",
                quantity, symbol
            )));
        }

        let proceeds = quantity * price;
        let commission = self.commission.calculate(proceeds);
        let net_proceeds = proceeds - commission;

        self.cash += net_proceeds;

        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.update(-quantity, price, timestamp);

            // Remove position if fully closed
            if pos.quantity.abs() < 1e-10 {
                self.positions.remove(symbol);
            }
        }

        Ok(Trade {
            symbol: symbol.to_string(),
            side: Side::Sell,
            quantity,
            price,
            commission,
            timestamp,
        })
    }

    /// Update all position prices
    pub fn update_prices(&mut self, prices: &HashMap<String, f64>, timestamp: DateTime<Utc>) {
        for (symbol, price) in prices {
            if let Some(position) = self.positions.get_mut(symbol) {
                position.update_price(*price, timestamp);
            }
        }
    }
}

// ============================================================================
// Order Execution
// ============================================================================

/// Commission model
#[derive(Debug, Clone)]
pub enum Commission {
    /// Fixed commission per trade
    Fixed(f64),
    /// Percentage of trade value
    Percentage(f64),
    /// No commission
    None,
}

impl Commission {
    /// Calculate commission for a trade
    pub fn calculate(&self, trade_value: f64) -> f64 {
        match self {
            Commission::Fixed(amount) => *amount,
            Commission::Percentage(pct) => trade_value * pct,
            Commission::None => 0.0,
        }
    }
}

/// Slippage model
#[derive(Debug, Clone)]
pub enum Slippage {
    /// Fixed slippage in price units
    Fixed(f64),
    /// Percentage of price
    Percentage(f64),
    /// No slippage
    None,
}

impl Slippage {
    /// Apply slippage to a price
    pub fn apply(&self, price: f64, side: Side) -> f64 {
        let slip = match self {
            Slippage::Fixed(amount) => *amount,
            Slippage::Percentage(pct) => price * pct,
            Slippage::None => 0.0,
        };

        match side {
            Side::Buy => price + slip,
            Side::Sell => price - slip,
        }
    }
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Executed trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub timestamp: DateTime<Utc>,
}

impl Trade {
    /// Calculate trade value (excluding commission)
    pub fn value(&self) -> f64 {
        self.quantity * self.price
    }

    /// Calculate net proceeds (for sells) or cost (for buys)
    pub fn net_value(&self) -> f64 {
        match self.side {
            Side::Buy => -(self.value() + self.commission),
            Side::Sell => self.value() - self.commission,
        }
    }
}

// ============================================================================
// Performance Metrics
// ============================================================================

/// Performance statistics for a backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return (percentage)
    pub total_return: f64,
    /// Annualized return (percentage)
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown (percentage)
    pub max_drawdown: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average profit per winning trade
    pub avg_win: f64,
    /// Average loss per losing trade
    pub avg_loss: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total commission paid
    pub total_commission: f64,
    /// Starting capital
    pub initial_capital: f64,
    /// Final equity
    pub final_equity: f64,
    /// Duration of backtest
    pub duration_days: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from equity curve and trades
    pub fn calculate(
        initial_capital: f64,
        final_equity: f64,
        equity_curve: &[(DateTime<Utc>, f64)],
        trades: &[Trade],
        duration_days: f64,
    ) -> Self {
        let total_return = (final_equity - initial_capital) / initial_capital * 100.0;
        let annualized_return = if duration_days > 0.0 {
            (final_equity / initial_capital).powf(365.25 / duration_days) - 1.0
        } else {
            0.0
        } * 100.0;

        // Calculate returns for Sharpe ratio
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1].1 - w[0].1) / w[0].1)
            .collect();

        let sharpe_ratio = if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                // Annualize assuming 252 trading days
                mean_return / std_dev * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Calculate max drawdown
        let max_drawdown = Self::calculate_max_drawdown(equity_curve);

        // Calculate trade statistics
        let mut winning_trades = 0;
        let mut losing_trades = 0;
        let mut total_profit = 0.0;
        let mut total_loss = 0.0;
        let mut total_commission = 0.0;

        // Group trades into round trips
        let mut trade_pnl: HashMap<String, Vec<f64>> = HashMap::new();

        for trade in trades {
            total_commission += trade.commission;

            // Track P&L per symbol
            let pnl_list = trade_pnl.entry(trade.symbol.clone()).or_default();
            match trade.side {
                Side::Buy => pnl_list.push(-trade.value()),
                Side::Sell => {
                    if let Some(cost) = pnl_list.pop() {
                        let pnl = trade.value() + cost - trade.commission;
                        if pnl > 0.0 {
                            winning_trades += 1;
                            total_profit += pnl;
                        } else {
                            losing_trades += 1;
                            total_loss += pnl.abs();
                        }
                    }
                }
            }
        }

        let win_rate = if winning_trades + losing_trades > 0 {
            winning_trades as f64 / (winning_trades + losing_trades) as f64 * 100.0
        } else {
            0.0
        };

        let avg_win = if winning_trades > 0 {
            total_profit / winning_trades as f64
        } else {
            0.0
        };

        let avg_loss = if losing_trades > 0 {
            total_loss / losing_trades as f64
        } else {
            0.0
        };

        let profit_factor = if total_loss > 1e-10 {
            total_profit / total_loss
        } else if total_profit > 1e-10 {
            f64::INFINITY
        } else {
            0.0
        };

        Self {
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            total_trades: trades.len(),
            winning_trades,
            losing_trades,
            avg_win,
            avg_loss,
            profit_factor,
            total_commission,
            initial_capital,
            final_equity,
            duration_days,
        }
    }

    /// Calculate maximum drawdown from equity curve
    fn calculate_max_drawdown(equity_curve: &[(DateTime<Utc>, f64)]) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = 0.0;

        for (_, equity) in equity_curve {
            if *equity > peak {
                peak = *equity;
            }

            let drawdown = (peak - equity) / peak * 100.0;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }
}

// ============================================================================
// Backtest Engine
// ============================================================================

/// Configuration for backtesting
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Commission model
    pub commission: Commission,
    /// Slippage model
    pub slippage: Slippage,
    /// Symbols to trade
    pub symbols: Vec<String>,
    /// Primary timeframe for bar data
    pub timeframe: Timeframe,
    /// Start date for backtest
    pub start_date: DateTime<Utc>,
    /// End date for backtest
    pub end_date: DateTime<Utc>,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            commission: Commission::Percentage(0.001), // 0.1%
            slippage: Slippage::Percentage(0.0005),    // 0.05%
            symbols: vec![],
            timeframe: Timeframe::Day1,
            start_date: Utc::now(),
            end_date: Utc::now(),
        }
    }
}

/// Backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Trade history
    pub trades: Vec<Trade>,
    /// Equity curve (timestamp, equity)
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
    /// Final portfolio state
    pub portfolio: Portfolio,
}

/// Event-driven backtesting engine
pub struct BacktestEngine<P: MarketDataProvider> {
    /// Market data provider
    provider: P,
    /// Backtest configuration
    config: BacktestConfig,
}

impl<P: MarketDataProvider> BacktestEngine<P> {
    /// Create a new backtest engine
    pub fn new(provider: P, config: BacktestConfig) -> Self {
        Self { provider, config }
    }

    /// Run backtest with the given strategy
    pub async fn run<S: Strategy>(&self, strategy: &mut S) -> MarketResult<BacktestResult> {
        // Initialize strategy
        strategy.initialize().await;

        // Initialize portfolio
        let mut portfolio = Portfolio::new(self.config.initial_capital, self.config.commission.clone());

        // Track trades and equity
        let mut trades = Vec::new();
        let mut equity_curve = Vec::new();

        // Fetch historical data for all symbols
        let mut all_bars: Vec<Bar> = Vec::new();

        for symbol in &self.config.symbols {
            let bars = self
                .provider
                .fetch_bars(
                    symbol,
                    self.config.timeframe,
                    self.config.start_date,
                    self.config.end_date,
                )
                .await?;

            all_bars.extend(bars);
        }

        // Sort bars by timestamp
        all_bars.sort_by_key(|b| b.timestamp);

        // Process bars chronologically
        for bar in all_bars {
            // Update position prices
            let mut prices = HashMap::new();
            prices.insert(bar.symbol.clone(), bar.close);
            portfolio.update_prices(&prices, bar.timestamp);

            // Record equity
            equity_curve.push((bar.timestamp, portfolio.equity()));

            // Generate signals
            let signals = strategy.on_bar(&bar).await;

            // Execute signals
            for signal in signals {
                match signal {
                    Signal::Buy {
                        symbol,
                        quantity,
                        price,
                    } => {
                        let exec_price = price.unwrap_or(bar.close);
                        let exec_price = self.config.slippage.apply(exec_price, Side::Buy);

                        match portfolio.buy(&symbol, quantity, exec_price, bar.timestamp) {
                            Ok(trade) => trades.push(trade),
                            Err(e) => {
                                tracing::warn!("Failed to execute buy: {}", e);
                            }
                        }
                    }
                    Signal::Sell {
                        symbol,
                        quantity,
                        price,
                    } => {
                        let exec_price = price.unwrap_or(bar.close);
                        let exec_price = self.config.slippage.apply(exec_price, Side::Sell);

                        match portfolio.sell(&symbol, quantity, exec_price, bar.timestamp) {
                            Ok(trade) => trades.push(trade),
                            Err(e) => {
                                tracing::warn!("Failed to execute sell: {}", e);
                            }
                        }
                    }
                    Signal::ClosePosition { symbol } => {
                        if let Some(position) = portfolio.positions.get(&symbol).cloned() {
                            let exec_price = self.config.slippage.apply(bar.close, Side::Sell);
                            match portfolio.sell(
                                &symbol,
                                position.quantity,
                                exec_price,
                                bar.timestamp,
                            ) {
                                Ok(trade) => trades.push(trade),
                                Err(e) => {
                                    tracing::warn!("Failed to close position: {}", e);
                                }
                            }
                        }
                    }
                    Signal::CloseAll => {
                        let positions: Vec<_> = portfolio.positions.values().cloned().collect();
                        for position in positions {
                            let exec_price = self.config.slippage.apply(bar.close, Side::Sell);
                            match portfolio.sell(
                                &position.symbol,
                                position.quantity,
                                exec_price,
                                bar.timestamp,
                            ) {
                                Ok(trade) => trades.push(trade),
                                Err(e) => {
                                    tracing::warn!("Failed to close position: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Finalize strategy
        strategy.finalize().await;

        // Calculate duration
        let duration_days = (self.config.end_date - self.config.start_date)
            .num_seconds() as f64
            / 86400.0;

        // Calculate performance metrics
        let metrics = PerformanceMetrics::calculate(
            self.config.initial_capital,
            portfolio.equity(),
            &equity_curve,
            &trades,
            duration_days,
        );

        Ok(BacktestResult {
            metrics,
            trades,
            equity_curve,
            portfolio,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_position_creation() {
        let pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

        assert_eq!(pos.symbol, "AAPL");
        assert_eq!(pos.quantity, 100.0);
        assert_eq!(pos.avg_price, 150.0);
        assert_relative_eq!(pos.market_value(), 15000.0);
        assert_relative_eq!(pos.unrealized_pnl(), 0.0);
    }

    #[test]
    fn test_position_update() {
        let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

        // Add to position
        pos.update(50.0, 160.0, Utc::now());
        assert_eq!(pos.quantity, 150.0);
        assert_relative_eq!(pos.avg_price, 153.33333333333334, epsilon = 1e-10);

        // Reduce position
        pos.update(-50.0, 170.0, Utc::now());
        assert_eq!(pos.quantity, 100.0);

        // Close position
        pos.update(-100.0, 180.0, Utc::now());
        assert_eq!(pos.quantity, 0.0);
    }

    #[test]
    fn test_position_unrealized_pnl() {
        let mut pos = Position::new("AAPL".to_string(), 100.0, 150.0, Utc::now());

        pos.update_price(160.0, Utc::now());
        assert_relative_eq!(pos.unrealized_pnl(), 1000.0);

        pos.update_price(140.0, Utc::now());
        assert_relative_eq!(pos.unrealized_pnl(), -1000.0);
    }

    #[test]
    fn test_portfolio_buy() {
        let mut portfolio = Portfolio::new(100_000.0, Commission::Fixed(10.0));

        let trade = portfolio
            .buy("AAPL", 100.0, 150.0, Utc::now())
            .expect("Buy should succeed");

        assert_eq!(trade.quantity, 100.0);
        assert_eq!(trade.price, 150.0);
        assert_eq!(trade.commission, 10.0);
        assert_relative_eq!(portfolio.cash, 100_000.0 - 15_000.0 - 10.0);
        assert_eq!(portfolio.positions.len(), 1);
    }

    #[test]
    fn test_portfolio_sell() {
        let mut portfolio = Portfolio::new(100_000.0, Commission::Fixed(10.0));

        portfolio
            .buy("AAPL", 100.0, 150.0, Utc::now())
            .expect("Buy should succeed");

        let trade = portfolio
            .sell("AAPL", 100.0, 160.0, Utc::now())
            .expect("Sell should succeed");

        assert_eq!(trade.quantity, 100.0);
        assert_eq!(trade.price, 160.0);
        assert_relative_eq!(portfolio.cash, 100_000.0 - 15_010.0 + 16_000.0 - 10.0);
        assert_eq!(portfolio.positions.len(), 0);
    }

    #[test]
    fn test_portfolio_insufficient_cash() {
        let mut portfolio = Portfolio::new(1000.0, Commission::None);

        let result = portfolio.buy("AAPL", 100.0, 150.0, Utc::now());
        assert!(result.is_err());
    }

    #[test]
    fn test_portfolio_insufficient_position() {
        let mut portfolio = Portfolio::new(100_000.0, Commission::None);

        let result = portfolio.sell("AAPL", 100.0, 150.0, Utc::now());
        assert!(result.is_err());
    }

    #[test]
    fn test_commission_calculation() {
        let fixed = Commission::Fixed(10.0);
        assert_eq!(fixed.calculate(1000.0), 10.0);

        let pct = Commission::Percentage(0.01);
        assert_eq!(pct.calculate(1000.0), 10.0);

        let none = Commission::None;
        assert_eq!(none.calculate(1000.0), 0.0);
    }

    #[test]
    fn test_slippage_application() {
        let fixed = Slippage::Fixed(0.05);
        assert_relative_eq!(fixed.apply(100.0, Side::Buy), 100.05);
        assert_relative_eq!(fixed.apply(100.0, Side::Sell), 99.95);

        let pct = Slippage::Percentage(0.001);
        assert_relative_eq!(pct.apply(100.0, Side::Buy), 100.1);
        assert_relative_eq!(pct.apply(100.0, Side::Sell), 99.9);
    }

    #[test]
    fn test_trade_value_calculation() {
        let trade = Trade {
            symbol: "AAPL".to_string(),
            side: Side::Buy,
            quantity: 100.0,
            price: 150.0,
            commission: 10.0,
            timestamp: Utc::now(),
        };

        assert_relative_eq!(trade.value(), 15000.0);
        assert_relative_eq!(trade.net_value(), -15010.0);

        let sell_trade = Trade {
            side: Side::Sell,
            ..trade
        };
        assert_relative_eq!(sell_trade.net_value(), 14990.0);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let equity_curve = vec![
            (Utc::now(), 100_000.0),
            (Utc::now(), 105_000.0),
            (Utc::now(), 103_000.0),
            (Utc::now(), 110_000.0),
        ];

        let trades = vec![
            Trade {
                symbol: "AAPL".to_string(),
                side: Side::Buy,
                quantity: 100.0,
                price: 100.0,
                commission: 10.0,
                timestamp: Utc::now(),
            },
            Trade {
                symbol: "AAPL".to_string(),
                side: Side::Sell,
                quantity: 100.0,
                price: 110.0,
                commission: 10.0,
                timestamp: Utc::now(),
            },
        ];

        let metrics = PerformanceMetrics::calculate(
            100_000.0,
            110_000.0,
            &equity_curve,
            &trades,
            365.0,
        );

        assert_relative_eq!(metrics.total_return, 10.0);
        assert_eq!(metrics.total_trades, 2);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.losing_trades, 0);
        assert_relative_eq!(metrics.win_rate, 100.0);
    }

    #[test]
    fn test_max_drawdown_calculation() {
        let equity_curve = vec![
            (Utc::now(), 100_000.0),
            (Utc::now(), 110_000.0),
            (Utc::now(), 90_000.0),
            (Utc::now(), 95_000.0),
            (Utc::now(), 105_000.0),
        ];

        let drawdown = PerformanceMetrics::calculate_max_drawdown(&equity_curve);
        assert_relative_eq!(drawdown, 18.181818181818183, epsilon = 1e-10);
    }

    // Mock provider for testing
    struct MockProvider {
        bars: Vec<Bar>,
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

    // Simple test strategy
    struct BuyAndHoldStrategy;

    #[async_trait]
    impl Strategy for BuyAndHoldStrategy {
        async fn initialize(&mut self) {}

        async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
            // Buy on first bar, hold otherwise
            vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        }

        async fn finalize(&mut self) {}
    }

    #[tokio::test]
    async fn test_backtest_engine_basic() {
        let bars = vec![
            Bar::new("AAPL".to_string(), Utc::now(), 100.0, 105.0, 99.0, 103.0, 1000),
            Bar::new("AAPL".to_string(), Utc::now(), 103.0, 108.0, 102.0, 107.0, 1200),
            Bar::new("AAPL".to_string(), Utc::now(), 107.0, 110.0, 105.0, 109.0, 1100),
        ];

        let provider = MockProvider { bars };

        let config = BacktestConfig {
            initial_capital: 100_000.0,
            commission: Commission::Fixed(1.0),
            slippage: Slippage::None,
            symbols: vec!["AAPL".to_string()],
            timeframe: Timeframe::Day1,
            start_date: Utc::now(),
            end_date: Utc::now(),
        };

        let engine = BacktestEngine::new(provider, config);
        let mut strategy = BuyAndHoldStrategy;

        let result = engine.run(&mut strategy).await.expect("Backtest should succeed");

        assert!(result.metrics.final_equity > result.metrics.initial_capital);
        assert!(!result.trades.is_empty());
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_signal_creation() {
        let buy_signal = Signal::Buy {
            symbol: "AAPL".to_string(),
            quantity: 100.0,
            price: Some(150.0),
        };

        match buy_signal {
            Signal::Buy { symbol, quantity, price } => {
                assert_eq!(symbol, "AAPL");
                assert_eq!(quantity, 100.0);
                assert_eq!(price, Some(150.0));
            }
            _ => panic!("Wrong signal type"),
        }
    }

    #[test]
    fn test_portfolio_equity() {
        let mut portfolio = Portfolio::new(100_000.0, Commission::None);

        portfolio.buy("AAPL", 100.0, 150.0, Utc::now()).unwrap();

        // Cash = 100,000 - 15,000 = 85,000
        // Position value = 100 * 150 = 15,000
        // Equity = 100,000
        assert_relative_eq!(portfolio.equity(), 100_000.0);

        // Update price
        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 160.0);
        portfolio.update_prices(&prices, Utc::now());

        // Cash = 85,000
        // Position value = 100 * 160 = 16,000
        // Equity = 101,000
        assert_relative_eq!(portfolio.equity(), 101_000.0);
    }
}
