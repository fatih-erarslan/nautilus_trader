//! Quantitative Analysis Framework for Neural Trading System
//!
//! This module provides comprehensive statistical validation, backtesting,
//! and performance attribution for neural trading strategies.

use std::collections::{HashMap, VecDeque};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::SQRT_2;

/// Comprehensive backtesting framework with realistic transaction costs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestEngine {
    /// Historical price data
    price_data: Vec<PricePoint>,
    /// Trading signals from neural network
    signals: Vec<TradingSignal>,
    /// Transaction cost model
    cost_model: TransactionCostModel,
    /// Portfolio state tracking
    portfolio: Portfolio,
    /// Performance metrics
    performance: PerformanceMetrics,
    /// Risk metrics tracking
    risk_metrics: RiskMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub bid: f64,
    pub ask: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub signal_strength: f64, // -1.0 to 1.0
    pub confidence: f64,      // 0.0 to 1.0
    pub prediction_horizon: Duration,
    pub expected_return: f64,
    pub predicted_volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCostModel {
    /// Bid-ask spread impact
    pub spread_cost: f64,
    /// Market impact model parameters
    pub market_impact: MarketImpact,
    /// Fixed commission per trade
    pub commission: f64,
    /// Borrowing costs for short positions
    pub borrow_cost: f64,
    /// Slippage model
    pub slippage: SlippageModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpact {
    /// Linear impact coefficient
    pub linear_coeff: f64,
    /// Square-root impact coefficient  
    pub sqrt_coeff: f64,
    /// Temporary impact decay
    pub decay_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Base slippage percentage
    pub base_slippage: f64,
    /// Volume-dependent slippage
    pub volume_impact: f64,
    /// Volatility multiplier
    pub volatility_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub cash: f64,
    pub positions: HashMap<String, Position>,
    pub total_value: f64,
    pub leverage: f64,
    pub margin_used: f64,
    pub pnl_history: VecDeque<f64>,
    pub trades: Vec<Trade>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub market_impact: f64,
    pub signal_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeSide {
    Buy,
    Sell,
    Short,
    Cover,
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Information ratio
    pub information_ratio: f64,
    /// Alpha vs benchmark
    pub alpha: f64,
    /// Beta vs benchmark
    pub beta: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Average drawdown duration
    pub avg_drawdown_duration: Duration,
    /// Win rate
    pub win_rate: f64,
    /// Average win/loss ratio
    pub win_loss_ratio: f64,
    /// Profit factor
    pub profit_factor: f64,
}

/// Risk metrics and VaR calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk (1%, 5%, 10%)
    pub var_1_day: HashMap<f64, f64>,
    /// Conditional Value at Risk
    pub cvar_1_day: HashMap<f64, f64>,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Current drawdown
    pub current_drawdown: f64,
    /// Tail ratio
    pub tail_ratio: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Downside deviation
    pub downside_deviation: f64,
    /// Beta to market
    pub market_beta: f64,
    /// Correlation to benchmark
    pub benchmark_correlation: f64,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(cost_model: TransactionCostModel) -> Self {
        Self {
            price_data: Vec::new(),
            signals: Vec::new(),
            cost_model,
            portfolio: Portfolio::new(),
            performance: PerformanceMetrics::default(),
            risk_metrics: RiskMetrics::default(),
        }
    }

    /// Add historical price data
    pub fn add_price_data(&mut self, data: Vec<PricePoint>) {
        self.price_data.extend(data);
        self.price_data.sort_by_key(|p| p.timestamp);
    }

    /// Add trading signals
    pub fn add_signals(&mut self, signals: Vec<TradingSignal>) {
        self.signals.extend(signals);
        self.signals.sort_by_key(|s| s.timestamp);
    }

    /// Run complete backtest
    pub fn run_backtest(&mut self, initial_capital: f64) -> BacktestResults {
        self.portfolio.cash = initial_capital;
        self.portfolio.total_value = initial_capital;

        // Execute trades based on signals
        for signal in &self.signals.clone() {
            self.execute_signal(signal);
            self.update_portfolio_value();
            self.calculate_metrics();
        }

        // Calculate final performance metrics
        self.calculate_final_performance();
        self.calculate_risk_metrics();

        BacktestResults {
            performance: self.performance.clone(),
            risk_metrics: self.risk_metrics.clone(),
            trades: self.portfolio.trades.clone(),
            equity_curve: self.portfolio.pnl_history.iter().cloned().collect(),
            drawdown_periods: self.calculate_drawdown_periods(),
        }
    }

    /// Execute trading signal with realistic costs
    fn execute_signal(&mut self, signal: &TradingSignal) {
        let current_price = self.get_current_price(&signal.symbol, signal.timestamp);
        if current_price.is_none() {
            return;
        }
        let price_data = current_price.unwrap();

        // Calculate position size based on signal strength and risk management
        let position_size = self.calculate_position_size(signal, &price_data);
        if position_size.abs() < 1e-6 {
            return;
        }

        // Calculate transaction costs
        let costs = self.calculate_transaction_costs(position_size, &price_data, signal);
        
        // Determine execution price
        let execution_price = if position_size > 0.0 {
            price_data.ask + costs.slippage
        } else {
            price_data.bid - costs.slippage
        };

        // Create trade
        let trade = Trade {
            timestamp: signal.timestamp,
            symbol: signal.symbol.clone(),
            side: if position_size > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
            quantity: position_size.abs(),
            price: execution_price,
            commission: costs.commission,
            slippage: costs.slippage,
            market_impact: costs.market_impact,
            signal_id: format!("{}-{}", signal.symbol, signal.timestamp.timestamp()),
        };

        // Update portfolio
        self.update_position(&trade);
        self.portfolio.trades.push(trade);
    }

    /// Calculate position size based on risk management
    fn calculate_position_size(&self, signal: &TradingSignal, price_data: &PricePoint) -> f64 {
        let signal_strength = signal.signal_strength;
        let confidence = signal.confidence;
        
        // Kelly criterion with confidence adjustment
        let kelly_fraction = self.calculate_kelly_fraction(signal);
        let risk_adjusted_kelly = kelly_fraction * confidence;
        
        // Maximum position size as percentage of portfolio
        let max_position_pct = 0.1; // 10% max per position
        let position_pct = (risk_adjusted_kelly * signal_strength.abs()).min(max_position_pct);
        
        // Calculate notional amount
        let notional = self.portfolio.total_value * position_pct;
        let position_size = notional / price_data.close;
        
        if signal_strength > 0.0 { position_size } else { -position_size }
    }

    /// Calculate Kelly fraction for optimal position sizing
    fn calculate_kelly_fraction(&self, signal: &TradingSignal) -> f64 {
        let expected_return = signal.expected_return;
        let volatility = signal.predicted_volatility;
        
        if volatility <= 0.0 {
            return 0.0;
        }
        
        // Kelly = (expected_return - risk_free_rate) / variance
        let risk_free_rate = 0.02; // 2% annual
        (expected_return - risk_free_rate) / (volatility * volatility)
    }

    /// Calculate realistic transaction costs
    fn calculate_transaction_costs(&self, position_size: f64, price_data: &PricePoint, signal: &TradingSignal) -> TransactionCosts {
        let notional = position_size.abs() * price_data.close;
        
        // Spread cost
        let spread_cost = (price_data.ask - price_data.bid) * 0.5;
        
        // Market impact based on square-root law
        let volume_pct = notional / (price_data.volume * price_data.close);
        let market_impact = self.cost_model.market_impact.linear_coeff * volume_pct +
                           self.cost_model.market_impact.sqrt_coeff * volume_pct.sqrt();
        
        // Slippage based on volatility and urgency
        let volatility_factor = signal.predicted_volatility * self.cost_model.slippage.volatility_multiplier;
        let slippage = self.cost_model.slippage.base_slippage + 
                      self.cost_model.slippage.volume_impact * volume_pct +
                      volatility_factor;
        
        TransactionCosts {
            commission: self.cost_model.commission,
            spread_cost,
            market_impact: market_impact * price_data.close,
            slippage: slippage * price_data.close,
            total_cost: self.cost_model.commission + spread_cost + market_impact * price_data.close + slippage * price_data.close,
        }
    }

    /// Get current price data for symbol at timestamp
    fn get_current_price(&self, symbol: &str, timestamp: DateTime<Utc>) -> Option<&PricePoint> {
        self.price_data.iter()
            .filter(|p| p.timestamp <= timestamp)
            .last()
    }

    /// Update portfolio position
    fn update_position(&mut self, trade: &Trade) {
        let position = self.portfolio.positions.entry(trade.symbol.clone()).or_insert(Position {
            symbol: trade.symbol.clone(),
            quantity: 0.0,
            avg_price: 0.0,
            market_value: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            duration: Duration::zero(),
        });

        let total_cost = trade.quantity * trade.price + trade.commission + trade.slippage + trade.market_impact;
        
        match trade.side {
            TradeSide::Buy => {
                if position.quantity >= 0.0 {
                    // Long position - add to position
                    let new_quantity = position.quantity + trade.quantity;
                    position.avg_price = (position.avg_price * position.quantity + total_cost) / new_quantity;
                    position.quantity = new_quantity;
                } else {
                    // Short position - covering
                    if trade.quantity >= position.quantity.abs() {
                        // Full cover plus new long
                        let pnl = position.quantity.abs() * (position.avg_price - trade.price);
                        position.realized_pnl += pnl;
                        position.quantity = trade.quantity - position.quantity.abs();
                        position.avg_price = trade.price;
                    } else {
                        // Partial cover
                        let pnl = trade.quantity * (position.avg_price - trade.price);
                        position.realized_pnl += pnl;
                        position.quantity += trade.quantity;
                    }
                }
                self.portfolio.cash -= total_cost;
            },
            TradeSide::Sell | TradeSide::Short => {
                if position.quantity <= 0.0 {
                    // Short position - add to short
                    let new_quantity = position.quantity - trade.quantity;
                    position.avg_price = (position.avg_price * position.quantity.abs() + total_cost) / new_quantity.abs();
                    position.quantity = new_quantity;
                } else {
                    // Long position - selling
                    if trade.quantity >= position.quantity {
                        // Full sale plus new short
                        let pnl = position.quantity * (trade.price - position.avg_price);
                        position.realized_pnl += pnl;
                        position.quantity = position.quantity - trade.quantity;
                        position.avg_price = trade.price;
                    } else {
                        // Partial sale
                        let pnl = trade.quantity * (trade.price - position.avg_price);
                        position.realized_pnl += pnl;
                        position.quantity -= trade.quantity;
                    }
                }
                self.portfolio.cash += trade.quantity * trade.price - trade.commission - trade.slippage - trade.market_impact;
            },
            TradeSide::Cover => {
                // Similar to Buy but specifically for covering shorts
                let pnl = trade.quantity * (position.avg_price - trade.price);
                position.realized_pnl += pnl;
                position.quantity += trade.quantity;
                self.portfolio.cash -= total_cost;
            },
        }
    }

    /// Update total portfolio value
    fn update_portfolio_value(&mut self) {
        let mut total_value = self.portfolio.cash;
        
        for (symbol, position) in &mut self.portfolio.positions {
            if let Some(current_price) = self.price_data.last() {
                position.market_value = position.quantity * current_price.close;
                position.unrealized_pnl = position.quantity * (current_price.close - position.avg_price);
                total_value += position.market_value;
            }
        }
        
        self.portfolio.total_value = total_value;
        self.portfolio.pnl_history.push_back(total_value);
        
        // Keep only last 252 days (1 year) for rolling calculations
        if self.portfolio.pnl_history.len() > 252 {
            self.portfolio.pnl_history.pop_front();
        }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&mut self) {
        if self.portfolio.pnl_history.len() < 2 {
            return;
        }

        let returns = self.calculate_returns();
        if returns.is_empty() {
            return;
        }

        // Basic metrics
        let total_return = (self.portfolio.total_value / self.portfolio.pnl_history[0] - 1.0) * 100.0;
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = self.calculate_volatility(&returns);
        
        // Risk-adjusted metrics
        let sharpe_ratio = if volatility > 0.0 {
            (mean_return * 252.0) / (volatility * (252.0_f64).sqrt())
        } else {
            0.0
        };
        
        let sortino_ratio = self.calculate_sortino_ratio(&returns);
        let max_drawdown = self.calculate_max_drawdown();
        
        self.performance.total_return = total_return;
        self.performance.annualized_return = mean_return * 252.0;
        self.performance.volatility = volatility * (252.0_f64).sqrt();
        self.performance.sharpe_ratio = sharpe_ratio;
        self.performance.sortino_ratio = sortino_ratio;
        self.performance.max_drawdown = max_drawdown;
    }

    /// Calculate returns from PnL history
    fn calculate_returns(&self) -> Vec<f64> {
        let mut returns = Vec::new();
        let pnl_vec: Vec<f64> = self.portfolio.pnl_history.iter().cloned().collect();
        
        for i in 1..pnl_vec.len() {
            if pnl_vec[i-1] > 0.0 {
                returns.push((pnl_vec[i] / pnl_vec[i-1]) - 1.0);
            }
        }
        
        returns
    }

    /// Calculate volatility
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Calculate Sortino ratio
    fn calculate_sortino_ratio(&self, returns: &[f64]) -> f64 {
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        
        if downside_returns.is_empty() {
            return f64::INFINITY;
        }
        
        let downside_deviation = self.calculate_volatility(&downside_returns);
        if downside_deviation > 0.0 {
            (mean_return * 252.0) / (downside_deviation * (252.0_f64).sqrt())
        } else {
            0.0
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self) -> f64 {
        let pnl_vec: Vec<f64> = self.portfolio.pnl_history.iter().cloned().collect();
        if pnl_vec.len() < 2 {
            return 0.0;
        }
        
        let mut max_drawdown = 0.0;
        let mut peak = pnl_vec[0];
        
        for &value in &pnl_vec[1..] {
            if value > peak {
                peak = value;
            }
            let drawdown = (peak - value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown * 100.0
    }

    /// Calculate final performance metrics
    fn calculate_final_performance(&mut self) {
        let returns = self.calculate_returns();
        if returns.is_empty() {
            return;
        }

        // Win rate and profit factor
        let winning_trades: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        let losing_trades: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        
        self.performance.win_rate = winning_trades.len() as f64 / returns.len() as f64 * 100.0;
        
        if !winning_trades.is_empty() && !losing_trades.is_empty() {
            let avg_win = winning_trades.iter().sum::<f64>() / winning_trades.len() as f64;
            let avg_loss = losing_trades.iter().sum::<f64>().abs() / losing_trades.len() as f64;
            self.performance.win_loss_ratio = avg_win / avg_loss;
            
            let total_wins = winning_trades.iter().sum::<f64>();
            let total_losses = losing_trades.iter().sum::<f64>().abs();
            self.performance.profit_factor = total_wins / total_losses;
        }
        
        // Calmar ratio
        if self.performance.max_drawdown > 0.0 {
            self.performance.calmar_ratio = self.performance.annualized_return / self.performance.max_drawdown;
        }
    }

    /// Calculate risk metrics including VaR and CVaR
    fn calculate_risk_metrics(&mut self) {
        let returns = self.calculate_returns();
        if returns.len() < 10 {
            return;
        }

        // Sort returns for percentile calculations
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR at different confidence levels
        let confidence_levels = vec![0.01, 0.05, 0.10];
        for &confidence in &confidence_levels {
            let index = (confidence * sorted_returns.len() as f64) as usize;
            let var = if index < sorted_returns.len() {
                -sorted_returns[index] * 100.0 // Convert to percentage loss
            } else {
                0.0
            };
            self.risk_metrics.var_1_day.insert(confidence * 100.0, var);
            
            // Calculate CVaR (Expected Shortfall)
            let tail_returns: Vec<f64> = sorted_returns.iter().take(index).cloned().collect();
            let cvar = if !tail_returns.is_empty() {
                -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64 * 100.0
            } else {
                0.0
            };
            self.risk_metrics.cvar_1_day.insert(confidence * 100.0, cvar);
        }
        
        // Higher order moments
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            // Skewness
            let skewness = returns.iter()
                .map(|r| ((r - mean_return) / std_dev).powi(3))
                .sum::<f64>() / returns.len() as f64;
            
            // Kurtosis
            let kurtosis = returns.iter()
                .map(|r| ((r - mean_return) / std_dev).powi(4))
                .sum::<f64>() / returns.len() as f64 - 3.0; // Excess kurtosis
            
            self.risk_metrics.skewness = skewness;
            self.risk_metrics.kurtosis = kurtosis;
        }
        
        // Downside deviation
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        self.risk_metrics.downside_deviation = self.calculate_volatility(&downside_returns) * (252.0_f64).sqrt();
        
        // Tail ratio (95th percentile return / 5th percentile return)
        let p95_index = (0.95 * sorted_returns.len() as f64) as usize;
        let p05_index = (0.05 * sorted_returns.len() as f64) as usize;
        if p95_index < sorted_returns.len() && p05_index < sorted_returns.len() && sorted_returns[p05_index] < 0.0 {
            self.risk_metrics.tail_ratio = sorted_returns[p95_index] / sorted_returns[p05_index].abs();
        }
        
        self.risk_metrics.max_drawdown = self.performance.max_drawdown;
        self.risk_metrics.current_drawdown = self.calculate_current_drawdown();
    }

    /// Calculate current drawdown
    fn calculate_current_drawdown(&self) -> f64 {
        let pnl_vec: Vec<f64> = self.portfolio.pnl_history.iter().cloned().collect();
        if pnl_vec.len() < 2 {
            return 0.0;
        }
        
        let current_value = pnl_vec.last().unwrap();
        let mut peak = pnl_vec[0];
        
        for &value in &pnl_vec {
            if value > peak {
                peak = value;
            }
        }
        
        if peak > 0.0 {
            (peak - current_value) / peak * 100.0
        } else {
            0.0
        }
    }

    /// Calculate drawdown periods
    fn calculate_drawdown_periods(&self) -> Vec<DrawdownPeriod> {
        let pnl_vec: Vec<f64> = self.portfolio.pnl_history.iter().cloned().collect();
        if pnl_vec.len() < 2 {
            return Vec::new();
        }
        
        let mut periods = Vec::new();
        let mut in_drawdown = false;
        let mut peak = pnl_vec[0];
        let mut peak_index = 0;
        let mut drawdown_start = 0;
        let mut max_drawdown_in_period = 0.0;
        
        for (i, &value) in pnl_vec.iter().enumerate() {
            if value > peak {
                if in_drawdown {
                    // End of drawdown period
                    periods.push(DrawdownPeriod {
                        start_index: drawdown_start,
                        end_index: i - 1,
                        peak_index,
                        trough_index: drawdown_start + pnl_vec[drawdown_start..i].iter()
                            .enumerate()
                            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap().0,
                        max_drawdown: max_drawdown_in_period,
                        duration_days: i - drawdown_start,
                        recovery_days: i - peak_index,
                    });
                    in_drawdown = false;
                }
                peak = value;
                peak_index = i;
            } else if value < peak {
                if !in_drawdown {
                    // Start of new drawdown
                    in_drawdown = true;
                    drawdown_start = i;
                    max_drawdown_in_period = 0.0;
                }
                let current_drawdown = (peak - value) / peak * 100.0;
                if current_drawdown > max_drawdown_in_period {
                    max_drawdown_in_period = current_drawdown;
                }
            }
        }
        
        // Handle ongoing drawdown
        if in_drawdown {
            let last_index = pnl_vec.len() - 1;
            periods.push(DrawdownPeriod {
                start_index: drawdown_start,
                end_index: last_index,
                peak_index,
                trough_index: drawdown_start + pnl_vec[drawdown_start..].iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0,
                max_drawdown: max_drawdown_in_period,
                duration_days: last_index - drawdown_start + 1,
                recovery_days: 0, // Still in drawdown
            });
        }
        
        periods
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCosts {
    pub commission: f64,
    pub spread_cost: f64,
    pub market_impact: f64,
    pub slippage: f64,
    pub total_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    pub performance: PerformanceMetrics,
    pub risk_metrics: RiskMetrics,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<f64>,
    pub drawdown_periods: Vec<DrawdownPeriod>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    pub start_index: usize,
    pub end_index: usize,
    pub peak_index: usize,
    pub trough_index: usize,
    pub max_drawdown: f64,
    pub duration_days: usize,
    pub recovery_days: usize,
}

impl Portfolio {
    pub fn new() -> Self {
        Self {
            cash: 0.0,
            positions: HashMap::new(),
            total_value: 0.0,
            leverage: 1.0,
            margin_used: 0.0,
            pnl_history: VecDeque::new(),
            trades: Vec::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            information_ratio: 0.0,
            alpha: 0.0,
            beta: 0.0,
            max_drawdown: 0.0,
            avg_drawdown_duration: Duration::zero(),
            win_rate: 0.0,
            win_loss_ratio: 0.0,
            profit_factor: 0.0,
        }
    }
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var_1_day: HashMap::new(),
            cvar_1_day: HashMap::new(),
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            tail_ratio: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            downside_deviation: 0.0,
            market_beta: 0.0,
            benchmark_correlation: 0.0,
        }
    }
}
