//! # Q* Trading Reward Functions
//! 
//! Sophisticated reward functions and market mechanics for Q* algorithm
//! with multi-dimensional optimization for cryptocurrency trading.
//!
//! ## Reward Dimensions
//!
//! - **Profit**: Risk-adjusted returns with Sharpe optimization
//! - **Risk**: Downside protection and volatility management
//! - **Efficiency**: Transaction cost and slippage minimization
//! - **Timing**: Entry/exit precision and trend alignment
//! - **Diversification**: Cross-asset correlation management
//!
//! ## Performance Targets
//!
//! - Calculation: <1μs per reward computation
//! - Accuracy: >99.99% numerical precision
//! - Risk Metrics: Real-time VaR, CVaR, Sortino
//! - Optimization: Multi-objective Pareto frontier

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use ndarray::{Array1, Array2};
use ordered_float::OrderedFloat;
use q_star_core::{MarketState, QStarAction, Experience, QStarError};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

pub mod profit;
pub mod risk;
pub mod efficiency;
pub mod timing;
pub mod portfolio;
pub mod market_impact;
pub mod advanced_metrics;
pub mod reinforcement;

pub use profit::*;
pub use risk::*;
pub use efficiency::*;
pub use timing::*;
pub use portfolio::*;
pub use market_impact::*;
pub use advanced_metrics::*;
pub use reinforcement::*;

/// Trading-specific errors
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Invalid position: {0}")]
    InvalidPosition(String),
    
    #[error("Insufficient balance: {0}")]
    InsufficientBalance(String),
    
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    
    #[error("Market impact too high: {0}")]
    MarketImpactError(String),
    
    #[error("Reward calculation failed: {0}")]
    RewardCalculationError(String),
    
    #[error("Portfolio constraint violated: {0}")]
    PortfolioConstraintError(String),
    
    #[error("Q* error: {0}")]
    QStarError(#[from] QStarError),
}

/// Trading reward configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingRewardConfig {
    /// Weight for profit component (0.0 to 1.0)
    pub profit_weight: f64,
    
    /// Weight for risk component (0.0 to 1.0)
    pub risk_weight: f64,
    
    /// Weight for efficiency component (0.0 to 1.0)
    pub efficiency_weight: f64,
    
    /// Weight for timing component (0.0 to 1.0)
    pub timing_weight: f64,
    
    /// Risk-free rate for Sharpe calculation
    pub risk_free_rate: f64,
    
    /// Target Sharpe ratio
    pub target_sharpe: f64,
    
    /// Maximum drawdown tolerance
    pub max_drawdown: f64,
    
    /// Transaction cost (percentage)
    pub transaction_cost: f64,
    
    /// Slippage model
    pub slippage_model: SlippageModel,
    
    /// Risk metrics to calculate
    pub risk_metrics: Vec<RiskMetricType>,
}

/// Slippage models for realistic execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    /// Fixed percentage slippage
    Fixed { rate: f64 },
    
    /// Linear market impact
    Linear { impact_coefficient: f64 },
    
    /// Square-root market impact (Almgren-Chriss)
    SquareRoot { 
        temporary_impact: f64,
        permanent_impact: f64,
    },
    
    /// Advanced microstructure model
    Microstructure {
        spread: f64,
        depth: f64,
        volatility_factor: f64,
    },
}

/// Risk metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskMetricType {
    VaR,           // Value at Risk
    CVaR,          // Conditional Value at Risk
    Sortino,       // Sortino ratio
    Calmar,        // Calmar ratio
    MaxDrawdown,   // Maximum drawdown
    Volatility,    // Standard deviation
    Beta,          // Market beta
    Correlation,   // Cross-asset correlation
}

impl Default for TradingRewardConfig {
    fn default() -> Self {
        Self {
            profit_weight: 0.4,
            risk_weight: 0.3,
            efficiency_weight: 0.2,
            timing_weight: 0.1,
            risk_free_rate: 0.02, // 2% annual
            target_sharpe: 2.0,
            max_drawdown: 0.2, // 20% max drawdown
            transaction_cost: 0.001, // 0.1%
            slippage_model: SlippageModel::Linear { impact_coefficient: 0.0001 },
            risk_metrics: vec![
                RiskMetricType::VaR,
                RiskMetricType::Sortino,
                RiskMetricType::MaxDrawdown,
            ],
        }
    }
}

/// Trading position tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPosition {
    /// Asset symbol
    pub symbol: String,
    
    /// Position size (positive = long, negative = short)
    pub size: f64,
    
    /// Entry price
    pub entry_price: f64,
    
    /// Current price
    pub current_price: f64,
    
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    
    /// Realized P&L
    pub realized_pnl: f64,
    
    /// Position risk metrics
    pub risk_metrics: HashMap<RiskMetricType, f64>,
}

impl TradingPosition {
    /// Create new position
    pub fn new(symbol: String, size: f64, entry_price: f64) -> Self {
        Self {
            symbol,
            size,
            entry_price,
            current_price: entry_price,
            entry_time: Utc::now(),
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            risk_metrics: HashMap::new(),
        }
    }
    
    /// Update position with current price
    pub fn update_price(&mut self, current_price: f64) {
        self.current_price = current_price;
        self.unrealized_pnl = self.calculate_unrealized_pnl();
    }
    
    /// Calculate unrealized P&L
    fn calculate_unrealized_pnl(&self) -> f64 {
        self.size * (self.current_price - self.entry_price)
    }
    
    /// Get position value
    pub fn get_value(&self) -> f64 {
        self.size.abs() * self.current_price
    }
    
    /// Check if position is profitable
    pub fn is_profitable(&self) -> bool {
        self.unrealized_pnl > 0.0
    }
}

/// Portfolio state for Q* trading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPortfolio {
    /// Current positions
    pub positions: HashMap<String, TradingPosition>,
    
    /// Cash balance
    pub cash_balance: f64,
    
    /// Total portfolio value
    pub total_value: f64,
    
    /// Historical returns
    pub returns_history: VecDeque<f64>,
    
    /// Risk metrics
    pub portfolio_risk_metrics: HashMap<RiskMetricType, f64>,
    
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub total_trades: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            total_trades: 0,
        }
    }
}

/// Main trading reward calculator
pub struct TradingRewardCalculator {
    /// Configuration
    config: TradingRewardConfig,
    
    /// Current portfolio state
    portfolio: Arc<RwLock<TradingPortfolio>>,
    
    /// Market data history
    market_history: Arc<RwLock<MarketHistory>>,
    
    /// Risk calculator
    risk_calculator: RiskCalculator,
    
    /// Market impact model
    market_impact_model: MarketImpactModel,
}

/// Market history for calculations
#[derive(Debug, Clone)]
pub struct MarketHistory {
    /// Price history
    pub price_history: VecDeque<(DateTime<Utc>, f64)>,
    
    /// Volume history
    pub volume_history: VecDeque<(DateTime<Utc>, f64)>,
    
    /// Volatility history
    pub volatility_history: VecDeque<(DateTime<Utc>, f64)>,
    
    /// Maximum history size
    pub max_size: usize,
}

impl TradingRewardCalculator {
    /// Create new reward calculator
    pub fn new(config: TradingRewardConfig, initial_balance: f64) -> Self {
        let portfolio = TradingPortfolio {
            positions: HashMap::new(),
            cash_balance: initial_balance,
            total_value: initial_balance,
            returns_history: VecDeque::with_capacity(1000),
            portfolio_risk_metrics: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
        };
        
        let market_history = MarketHistory {
            price_history: VecDeque::with_capacity(10000),
            volume_history: VecDeque::with_capacity(10000),
            volatility_history: VecDeque::with_capacity(10000),
            max_size: 10000,
        };
        
        Self {
            config: config.clone(),
            portfolio: Arc::new(RwLock::new(portfolio)),
            market_history: Arc::new(RwLock::new(market_history)),
            risk_calculator: RiskCalculator::new(config.clone()),
            market_impact_model: MarketImpactModel::new(config.slippage_model.clone()),
        }
    }
    
    /// Calculate reward for state-action-next_state transition
    pub async fn calculate_reward(
        &self,
        state: &MarketState,
        action: &QStarAction,
        next_state: &MarketState,
    ) -> Result<f64, TradingError> {
        let start_time = std::time::Instant::now();
        
        // Update market history
        self.update_market_history(state).await?;
        
        // Calculate reward components
        let profit_reward = self.calculate_profit_reward(state, action, next_state).await?;
        let risk_penalty = self.calculate_risk_penalty(state, action).await?;
        let efficiency_reward = self.calculate_efficiency_reward(action, state).await?;
        let timing_reward = self.calculate_timing_reward(state, action, next_state).await?;
        
        // Combine rewards with configured weights
        let total_reward = self.config.profit_weight * profit_reward
            - self.config.risk_weight * risk_penalty
            + self.config.efficiency_weight * efficiency_reward
            + self.config.timing_weight * timing_reward;
        
        // Ensure sub-microsecond performance
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 1 {
            log::warn!("Reward calculation took {}μs, exceeding 1μs target", elapsed.as_micros());
        }
        
        Ok(total_reward)
    }
    
    /// Calculate profit reward component
    async fn calculate_profit_reward(
        &self,
        state: &MarketState,
        action: &QStarAction,
        next_state: &MarketState,
    ) -> Result<f64, TradingError> {
        let mut portfolio = self.portfolio.write().await;
        
        let price_change = (next_state.price - state.price) / state.price;
        let mut profit = 0.0;
        
        match action {
            QStarAction::Buy { amount } => {
                // Calculate profit from buying
                let position_value = portfolio.cash_balance * amount;
                let shares = position_value / state.price;
                profit = shares * (next_state.price - state.price);
                
                // Apply transaction costs
                let transaction_cost = position_value * self.config.transaction_cost;
                profit -= transaction_cost;
                
                // Apply slippage
                let slippage = self.market_impact_model.calculate_slippage(
                    position_value,
                    state.volume,
                    state.volatility,
                );
                profit -= slippage;
            }
            
            QStarAction::Sell { amount } => {
                // Calculate profit from selling
                if let Some(position) = portfolio.positions.get(&state.symbol.unwrap_or_default()) {
                    let sell_value = position.size * amount * next_state.price;
                    profit = sell_value - (position.size * amount * position.entry_price);
                    
                    // Apply transaction costs
                    let transaction_cost = sell_value * self.config.transaction_cost;
                    profit -= transaction_cost;
                    
                    // Apply slippage
                    let slippage = self.market_impact_model.calculate_slippage(
                        sell_value,
                        state.volume,
                        state.volatility,
                    );
                    profit -= slippage;
                }
            }
            
            QStarAction::Hold => {
                // Calculate opportunity cost or unrealized P&L change
                for position in portfolio.positions.values_mut() {
                    position.update_price(next_state.price);
                    profit += position.unrealized_pnl - position.size * (position.current_price - position.entry_price);
                }
            }
            
            _ => {} // Other actions have different reward structures
        }
        
        // Risk-adjust the profit
        let risk_adjusted_profit = profit / (1.0 + state.volatility);
        
        // Normalize to [0, 1] range using sigmoid
        let normalized_profit = 1.0 / (1.0 + (-risk_adjusted_profit * 0.01).exp());
        
        Ok(normalized_profit)
    }
    
    /// Calculate risk penalty component
    async fn calculate_risk_penalty(
        &self,
        state: &MarketState,
        action: &QStarAction,
    ) -> Result<f64, TradingError> {
        let portfolio = self.portfolio.read().await;
        
        let mut risk_penalty = 0.0;
        
        // Calculate position risk
        let position_risk = match action {
            QStarAction::Buy { amount } => {
                let new_position_value = portfolio.cash_balance * amount;
                let concentration_risk = new_position_value / portfolio.total_value;
                concentration_risk * state.volatility
            }
            QStarAction::Sell { .. } => {
                // Selling reduces risk
                -0.1
            }
            _ => 0.0,
        };
        
        // Calculate portfolio risk metrics
        let var = self.risk_calculator.calculate_var(&portfolio, 0.95).await?;
        let max_drawdown = self.calculate_current_drawdown(&portfolio).await?;
        
        // Combine risk factors
        risk_penalty = position_risk + (var / portfolio.total_value) + (max_drawdown / self.config.max_drawdown);
        
        // Add penalty for exceeding risk limits
        if max_drawdown > self.config.max_drawdown {
            risk_penalty += 1.0; // Heavy penalty
        }
        
        Ok(risk_penalty.max(0.0).min(1.0))
    }
    
    /// Calculate efficiency reward component
    async fn calculate_efficiency_reward(
        &self,
        action: &QStarAction,
        state: &MarketState,
    ) -> Result<f64, TradingError> {
        let portfolio = self.portfolio.read().await;
        
        let efficiency = match action {
            QStarAction::Buy { amount } | QStarAction::Sell { amount } => {
                // Reward efficient use of capital
                let capital_efficiency = amount; // Using the requested amount efficiently
                
                // Penalize during high volatility (higher costs)
                let volatility_factor = 1.0 / (1.0 + state.volatility * 10.0);
                
                // Reward good liquidity conditions
                let liquidity_factor = (state.volume / 1_000_000.0).min(1.0);
                
                capital_efficiency * volatility_factor * liquidity_factor
            }
            
            QStarAction::Hold => {
                // Small reward for avoiding unnecessary trades
                0.1
            }
            
            QStarAction::StopLoss { .. } | QStarAction::TakeProfit { .. } => {
                // Risk management actions are efficient
                0.5
            }
            
            _ => 0.0,
        };
        
        Ok(efficiency)
    }
    
    /// Calculate timing reward component
    async fn calculate_timing_reward(
        &self,
        state: &MarketState,
        action: &QStarAction,
        next_state: &MarketState,
    ) -> Result<f64, TradingError> {
        let market_history = self.market_history.read().await;
        
        // Calculate trend alignment
        let trend_direction = if next_state.price > state.price { 1.0 } else { -1.0 };
        
        let timing_score = match action {
            QStarAction::Buy { .. } => {
                // Reward buying before price increase
                if trend_direction > 0.0 {
                    0.8 + 0.2 * ((next_state.price - state.price) / state.price).min(0.2)
                } else {
                    0.0 // Penalize buying before price decrease
                }
            }
            
            QStarAction::Sell { .. } => {
                // Reward selling before price decrease
                if trend_direction < 0.0 {
                    0.8 + 0.2 * ((state.price - next_state.price) / state.price).min(0.2)
                } else {
                    0.0 // Penalize selling before price increase
                }
            }
            
            QStarAction::Hold => {
                // Neutral timing for holding
                0.5
            }
            
            _ => 0.5,
        };
        
        // Adjust for momentum
        let momentum_factor = self.calculate_momentum_factor(&market_history).await;
        
        Ok(timing_score * momentum_factor)
    }
    
    /// Update market history
    async fn update_market_history(&self, state: &MarketState) -> Result<(), TradingError> {
        let mut history = self.market_history.write().await;
        
        let timestamp = Utc::now();
        
        // Add new data
        history.price_history.push_back((timestamp, state.price));
        history.volume_history.push_back((timestamp, state.volume));
        history.volatility_history.push_back((timestamp, state.volatility));
        
        // Trim old data
        while history.price_history.len() > history.max_size {
            history.price_history.pop_front();
            history.volume_history.pop_front();
            history.volatility_history.pop_front();
        }
        
        Ok(())
    }
    
    /// Calculate current drawdown
    async fn calculate_current_drawdown(&self, portfolio: &TradingPortfolio) -> Result<f64, TradingError> {
        if portfolio.returns_history.is_empty() {
            return Ok(0.0);
        }
        
        let mut peak = portfolio.total_value;
        let mut max_drawdown = 0.0;
        
        for &value in &portfolio.returns_history {
            peak = peak.max(value);
            let drawdown = (peak - value) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate momentum factor
    async fn calculate_momentum_factor(&self, history: &MarketHistory) -> f64 {
        if history.price_history.len() < 10 {
            return 1.0; // Neutral momentum
        }
        
        // Simple momentum: compare recent average to older average
        let recent_prices: Vec<f64> = history.price_history.iter()
            .rev()
            .take(5)
            .map(|(_, p)| *p)
            .collect();
        
        let older_prices: Vec<f64> = history.price_history.iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|(_, p)| *p)
            .collect();
        
        if recent_prices.is_empty() || older_prices.is_empty() {
            return 1.0;
        }
        
        let recent_avg = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let older_avg = older_prices.iter().sum::<f64>() / older_prices.len() as f64;
        
        let momentum = (recent_avg - older_avg) / older_avg;
        
        // Normalize to [0.5, 1.5] range
        (1.0 + momentum).max(0.5).min(1.5)
    }
    
    /// Get current portfolio state
    pub async fn get_portfolio(&self) -> TradingPortfolio {
        self.portfolio.read().await.clone()
    }
    
    /// Execute trade and update portfolio
    pub async fn execute_trade(
        &self,
        action: &QStarAction,
        state: &MarketState,
    ) -> Result<(), TradingError> {
        let mut portfolio = self.portfolio.write().await;
        
        match action {
            QStarAction::Buy { amount } => {
                let symbol = state.symbol.clone().unwrap_or_else(|| "BTC/USDT".to_string());
                let position_value = portfolio.cash_balance * amount;
                
                if position_value > portfolio.cash_balance {
                    return Err(TradingError::InsufficientBalance(
                        format!("Need {} but have {}", position_value, portfolio.cash_balance)
                    ));
                }
                
                let shares = position_value / state.price;
                
                // Create or update position
                if let Some(position) = portfolio.positions.get_mut(&symbol) {
                    // Average up/down
                    let total_shares = position.size + shares;
                    let total_cost = position.size * position.entry_price + shares * state.price;
                    position.entry_price = total_cost / total_shares;
                    position.size = total_shares;
                } else {
                    portfolio.positions.insert(
                        symbol.clone(),
                        TradingPosition::new(symbol, shares, state.price),
                    );
                }
                
                portfolio.cash_balance -= position_value;
                portfolio.performance_metrics.total_trades += 1;
            }
            
            QStarAction::Sell { amount } => {
                let symbol = state.symbol.clone().unwrap_or_else(|| "BTC/USDT".to_string());
                
                if let Some(position) = portfolio.positions.get_mut(&symbol) {
                    let sell_shares = position.size * amount;
                    let sell_value = sell_shares * state.price;
                    
                    // Update position
                    position.size -= sell_shares;
                    position.realized_pnl += sell_shares * (state.price - position.entry_price);
                    
                    // Remove position if fully closed
                    if position.size.abs() < 1e-8 {
                        portfolio.positions.remove(&symbol);
                    }
                    
                    portfolio.cash_balance += sell_value;
                    portfolio.performance_metrics.total_trades += 1;
                    
                    // Update win/loss statistics
                    let trade_pnl = sell_shares * (state.price - position.entry_price);
                    if trade_pnl > 0.0 {
                        portfolio.performance_metrics.avg_win = 
                            (portfolio.performance_metrics.avg_win * portfolio.performance_metrics.win_rate + trade_pnl) 
                            / (portfolio.performance_metrics.win_rate + 1.0);
                        portfolio.performance_metrics.win_rate += 1.0;
                    } else {
                        portfolio.performance_metrics.avg_loss = 
                            (portfolio.performance_metrics.avg_loss * (1.0 - portfolio.performance_metrics.win_rate) + trade_pnl.abs()) 
                            / (2.0 - portfolio.performance_metrics.win_rate);
                    }
                } else {
                    return Err(TradingError::InvalidPosition(
                        format!("No position to sell for {}", symbol)
                    ));
                }
            }
            
            _ => {} // Other actions don't directly modify portfolio
        }
        
        // Update total portfolio value
        portfolio.total_value = portfolio.cash_balance;
        for position in portfolio.positions.values() {
            portfolio.total_value += position.get_value();
        }
        
        // Record return
        portfolio.returns_history.push_back(portfolio.total_value);
        while portfolio.returns_history.len() > 1000 {
            portfolio.returns_history.pop_front();
        }
        
        Ok(())
    }
}

/// Risk calculator for portfolio risk metrics
pub struct RiskCalculator {
    config: TradingRewardConfig,
}

impl RiskCalculator {
    pub fn new(config: TradingRewardConfig) -> Self {
        Self { config }
    }
    
    /// Calculate Value at Risk
    pub async fn calculate_var(
        &self,
        portfolio: &TradingPortfolio,
        confidence_level: f64,
    ) -> Result<f64, TradingError> {
        if portfolio.returns_history.len() < 20 {
            return Ok(0.0); // Not enough data
        }
        
        // Calculate returns
        let returns: Vec<f64> = portfolio.returns_history.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate mean and std dev
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        // Calculate VaR using normal distribution
        let normal = Normal::new(mean, std_dev)
            .map_err(|e| TradingError::RewardCalculationError(format!("Normal distribution error: {}", e)))?;
        
        let var = normal.inverse_cdf(1.0 - confidence_level);
        
        Ok(-var * portfolio.total_value)
    }
    
    /// Calculate Sortino ratio
    pub async fn calculate_sortino(
        &self,
        returns: &[f64],
        target_return: f64,
    ) -> Result<f64, TradingError> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let excess_returns: Vec<f64> = returns.iter()
            .map(|r| r - target_return)
            .collect();
        
        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        
        // Calculate downside deviation
        let downside_variance = excess_returns.iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>() / excess_returns.len() as f64;
        
        let downside_deviation = downside_variance.sqrt();
        
        if downside_deviation > 0.0 {
            Ok(mean_excess / downside_deviation)
        } else {
            Ok(0.0)
        }
    }
}

/// Market impact model for realistic execution
pub struct MarketImpactModel {
    slippage_model: SlippageModel,
}

impl MarketImpactModel {
    pub fn new(slippage_model: SlippageModel) -> Self {
        Self { slippage_model }
    }
    
    /// Calculate slippage for a trade
    pub fn calculate_slippage(
        &self,
        trade_value: f64,
        market_volume: f64,
        volatility: f64,
    ) -> f64 {
        match &self.slippage_model {
            SlippageModel::Fixed { rate } => trade_value * rate,
            
            SlippageModel::Linear { impact_coefficient } => {
                let volume_fraction = trade_value / market_volume;
                trade_value * volume_fraction * impact_coefficient
            }
            
            SlippageModel::SquareRoot { temporary_impact, permanent_impact } => {
                let volume_fraction = trade_value / market_volume;
                let temp_impact = temporary_impact * volume_fraction.sqrt() * volatility;
                let perm_impact = permanent_impact * volume_fraction;
                trade_value * (temp_impact + perm_impact)
            }
            
            SlippageModel::Microstructure { spread, depth, volatility_factor } => {
                let spread_cost = trade_value * spread / 2.0;
                let depth_impact = trade_value / depth;
                let vol_impact = volatility * volatility_factor * trade_value;
                spread_cost + depth_impact + vol_impact
            }
        }
    }
}

/// Factory functions for creating reward calculators
pub mod factory {
    use super::*;
    
    /// Create optimized reward calculator for crypto trading
    pub fn create_crypto_reward_calculator(initial_balance: f64) -> TradingRewardCalculator {
        let config = TradingRewardConfig {
            profit_weight: 0.35,
            risk_weight: 0.35,
            efficiency_weight: 0.20,
            timing_weight: 0.10,
            risk_free_rate: 0.05, // 5% for crypto
            target_sharpe: 1.5, // Lower target for crypto volatility
            max_drawdown: 0.3, // 30% for crypto
            transaction_cost: 0.001, // 0.1% typical exchange fee
            slippage_model: SlippageModel::SquareRoot {
                temporary_impact: 0.01,
                permanent_impact: 0.001,
            },
            risk_metrics: vec![
                RiskMetricType::VaR,
                RiskMetricType::CVaR,
                RiskMetricType::Sortino,
                RiskMetricType::MaxDrawdown,
            ],
        };
        
        TradingRewardCalculator::new(config, initial_balance)
    }
    
    /// Create high-frequency trading reward calculator
    pub fn create_hft_reward_calculator(initial_balance: f64) -> TradingRewardCalculator {
        let config = TradingRewardConfig {
            profit_weight: 0.25,
            risk_weight: 0.25,
            efficiency_weight: 0.35, // High weight on efficiency for HFT
            timing_weight: 0.15,
            risk_free_rate: 0.02,
            target_sharpe: 3.0, // High Sharpe target for HFT
            max_drawdown: 0.1, // Tight drawdown control
            transaction_cost: 0.0001, // Ultra-low fees for HFT
            slippage_model: SlippageModel::Microstructure {
                spread: 0.0001,
                depth: 1_000_000.0,
                volatility_factor: 0.001,
            },
            ..Default::default()
        };
        
        TradingRewardCalculator::new(config, initial_balance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_star_core::MarketRegime;
    
    fn create_test_state(price: f64) -> MarketState {
        MarketState {
            price,
            volume: 1_000_000.0,
            volatility: 0.02,
            rsi: 0.5,
            macd: 0.001,
            market_regime: MarketRegime::Trending,
            timestamp: Utc::now(),
            features: vec![0.1],
            symbol: Some("BTC/USDT".to_string()),
        }
    }
    
    #[tokio::test]
    async fn test_reward_calculator_creation() {
        let calculator = factory::create_crypto_reward_calculator(10000.0);
        let portfolio = calculator.get_portfolio().await;
        assert_eq!(portfolio.cash_balance, 10000.0);
        assert_eq!(portfolio.total_value, 10000.0);
    }
    
    #[tokio::test]
    async fn test_profit_reward_calculation() {
        let calculator = factory::create_crypto_reward_calculator(10000.0);
        
        let state = create_test_state(50000.0);
        let next_state = create_test_state(51000.0); // 2% increase
        let action = QStarAction::Buy { amount: 0.5 };
        
        let reward = calculator.calculate_profit_reward(&state, &action, &next_state).await;
        assert!(reward.is_ok());
        assert!(reward.unwrap() > 0.5); // Should be positive for profitable trade
    }
    
    #[tokio::test]
    async fn test_execute_trade() {
        let calculator = factory::create_crypto_reward_calculator(10000.0);
        let state = create_test_state(50000.0);
        
        // Test buy trade
        let buy_action = QStarAction::Buy { amount: 0.5 };
        let result = calculator.execute_trade(&buy_action, &state).await;
        assert!(result.is_ok());
        
        let portfolio = calculator.get_portfolio().await;
        assert!(portfolio.cash_balance < 10000.0);
        assert_eq!(portfolio.positions.len(), 1);
        
        // Test sell trade
        let sell_action = QStarAction::Sell { amount: 0.5 };
        let result = calculator.execute_trade(&sell_action, &state).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_risk_calculation() {
        let calculator = factory::create_crypto_reward_calculator(10000.0);
        let portfolio = calculator.get_portfolio().await;
        
        let var = calculator.risk_calculator.calculate_var(&portfolio, 0.95).await;
        assert!(var.is_ok());
    }
    
    #[tokio::test]
    async fn test_market_impact() {
        let model = MarketImpactModel::new(SlippageModel::Linear { impact_coefficient: 0.001 });
        let slippage = model.calculate_slippage(1000.0, 1_000_000.0, 0.02);
        assert!(slippage > 0.0);
        assert!(slippage < 10.0); // Reasonable slippage
    }
}