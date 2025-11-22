//! Portfolio management implementation

use crate::prelude::*;
use crate::models::{Position, Order, OrderSide, MarketData, Trade};
use chrono::{DateTime, Utc, Duration};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Portfolio manager for comprehensive portfolio management
#[derive(Debug)]
pub struct PortfolioManager {
    /// Manager configuration
    config: PortfolioManagerConfig,
    
    /// Current portfolio state
    portfolio_state: Arc<RwLock<PortfolioState>>,
    
    /// Position tracking
    position_tracker: PositionTracker,
    
    /// Performance tracking
    performance_tracker: PerformanceTracker,
    
    /// Trade history
    trade_history: VecDeque<TradeRecord>,
    
    /// Risk metrics
    risk_metrics: RiskMetrics,
}

#[derive(Debug, Clone)]
pub struct PortfolioManagerConfig {
    /// Base currency for portfolio valuation
    pub base_currency: String,
    
    /// Performance calculation frequency
    pub performance_calc_frequency_seconds: u32,
    
    /// Maximum trade history to retain
    pub max_trade_history: usize,
    
    /// Enable real-time performance tracking
    pub enable_realtime_tracking: bool,
    
    /// Position sizing parameters
    pub position_sizing: PositionSizingConfig,
    
    /// Risk management settings
    pub risk_management: RiskManagementConfig,
    
    /// Rebalancing settings
    pub rebalancing: RebalancingConfig,
}

#[derive(Debug, Clone)]
pub struct PositionSizingConfig {
    /// Default position sizing method
    pub default_method: String,
    
    /// Maximum position size as percentage of portfolio
    pub max_position_pct: f64,
    
    /// Minimum position size in base currency
    pub min_position_size: Decimal,
    
    /// Position concentration limits
    pub concentration_limits: ConcentrationLimits,
}

#[derive(Debug, Clone)]
pub struct ConcentrationLimits {
    /// Maximum single position percentage
    pub max_single_position_pct: f64,
    
    /// Maximum sector concentration
    pub max_sector_pct: f64,
    
    /// Maximum geographic concentration
    pub max_geographic_pct: f64,
    
    /// Maximum asset class concentration
    pub max_asset_class_pct: f64,
}

#[derive(Debug, Clone)]
pub struct RiskManagementConfig {
    /// Enable automated risk management
    pub enabled: bool,
    
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    
    /// Take profit percentage
    pub take_profit_pct: f64,
    
    /// Maximum daily loss percentage
    pub max_daily_loss_pct: f64,
    
    /// Risk monitoring frequency in seconds
    pub monitoring_frequency_seconds: u32,
}

#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Enable automatic rebalancing
    pub enabled: bool,
    
    /// Rebalancing frequency in hours
    pub frequency_hours: u32,
    
    /// Minimum drift threshold for rebalancing
    pub drift_threshold_pct: f64,
    
    /// Target allocation method
    pub target_allocation_method: String,
}

#[derive(Debug, Clone, Default)]
struct PortfolioState {
    /// Total portfolio value in base currency
    total_value: Decimal,
    
    /// Available cash
    available_cash: Decimal,
    
    /// Current positions
    positions: HashMap<String, Position>,
    
    /// Unrealized P&L
    unrealized_pnl: Decimal,
    
    /// Realized P&L
    realized_pnl: Decimal,
    
    /// Portfolio inception date
    inception_date: Option<DateTime<Utc>>,
    
    /// Last update timestamp
    last_updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
struct PositionTracker {
    /// Position history
    position_history: VecDeque<PositionSnapshot>,
    
    /// Position analytics
    position_analytics: HashMap<String, PositionAnalytics>,
    
    /// Sector allocations
    sector_allocations: HashMap<String, Decimal>,
    
    /// Geographic allocations
    geographic_allocations: HashMap<String, Decimal>,
    
    /// Asset class allocations
    asset_class_allocations: HashMap<String, Decimal>,
}

#[derive(Debug, Clone)]
struct PositionSnapshot {
    timestamp: DateTime<Utc>,
    positions: HashMap<String, Position>,
    total_value: Decimal,
    sector_breakdown: HashMap<String, f64>,
    geographic_breakdown: HashMap<String, f64>,
}

#[derive(Debug, Clone, Default)]
struct PositionAnalytics {
    symbol: String,
    entry_date: Option<DateTime<Utc>>,
    average_entry_price: Decimal,
    total_cost_basis: Decimal,
    current_value: Decimal,
    unrealized_pnl: Decimal,
    realized_pnl: Decimal,
    holding_period_days: i64,
    number_of_trades: u32,
    win_rate: f64,
    largest_gain: Decimal,
    largest_loss: Decimal,
}

#[derive(Debug, Clone, Default)]
struct PerformanceTracker {
    /// Daily returns
    daily_returns: VecDeque<DailyReturn>,
    
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
    
    /// Benchmark comparison
    benchmark_comparison: Option<BenchmarkComparison>,
    
    /// Attribution analysis
    attribution_analysis: AttributionAnalysis,
}

#[derive(Debug, Clone)]
struct DailyReturn {
    date: DateTime<Utc>,
    portfolio_value: Decimal,
    daily_return: f64,
    cumulative_return: f64,
    benchmark_return: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct PerformanceMetrics {
    total_return: f64,
    annualized_return: f64,
    volatility: f64,
    sharpe_ratio: f64,
    sortino_ratio: f64,
    max_drawdown: f64,
    calmar_ratio: f64,
    win_rate: f64,
    profit_factor: f64,
    information_ratio: Option<f64>,
    tracking_error: Option<f64>,
}

#[derive(Debug, Clone)]
struct BenchmarkComparison {
    benchmark_symbol: String,
    alpha: f64,
    beta: f64,
    correlation: f64,
    up_capture: f64,
    down_capture: f64,
}

#[derive(Debug, Clone, Default)]
struct AttributionAnalysis {
    sector_attribution: HashMap<String, f64>,
    security_selection: f64,
    allocation_effect: f64,
    interaction_effect: f64,
}

#[derive(Debug, Clone)]
struct TradeRecord {
    trade_id: uuid::Uuid,
    order_id: uuid::Uuid,
    symbol: String,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    timestamp: DateTime<Utc>,
    fees: Decimal,
    impact_on_portfolio: TradeImpact,
}

#[derive(Debug, Clone)]
struct TradeImpact {
    position_change: Decimal,
    cash_change: Decimal,
    realized_pnl: Decimal,
    new_position_size: Decimal,
    sector_allocation_change: HashMap<String, f64>,
}

#[derive(Debug, Clone, Default)]
struct RiskMetrics {
    portfolio_var_95: f64,
    portfolio_var_99: f64,
    expected_shortfall: f64,
    portfolio_beta: f64,
    concentration_risk: f64,
    sector_concentration: HashMap<String, f64>,
    correlation_risk: f64,
    leverage_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioSummary {
    pub total_value: Decimal,
    pub available_cash: Decimal,
    pub total_positions: usize,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub daily_pnl: Decimal,
    pub performance_metrics: PerformanceMetrics,
    pub risk_metrics: RiskMetrics,
    pub position_breakdown: HashMap<String, PositionBreakdown>,
    pub sector_allocation: HashMap<String, f64>,
    pub geographic_allocation: HashMap<String, f64>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PositionBreakdown {
    pub symbol: String,
    pub quantity: Decimal,
    pub market_value: Decimal,
    pub cost_basis: Decimal,
    pub unrealized_pnl: Decimal,
    pub allocation_pct: f64,
    pub days_held: i64,
}

impl Default for PortfolioManagerConfig {
    fn default() -> Self {
        Self {
            base_currency: "USD".to_string(),
            performance_calc_frequency_seconds: 300, // 5 minutes
            max_trade_history: 10000,
            enable_realtime_tracking: true,
            position_sizing: PositionSizingConfig {
                default_method: "equal_weight".to_string(),
                max_position_pct: 0.10,
                min_position_size: Decimal::from(100),
                concentration_limits: ConcentrationLimits {
                    max_single_position_pct: 0.15,
                    max_sector_pct: 0.30,
                    max_geographic_pct: 0.40,
                    max_asset_class_pct: 0.60,
                },
            },
            risk_management: RiskManagementConfig {
                enabled: true,
                stop_loss_pct: 0.10,
                take_profit_pct: 0.20,
                max_daily_loss_pct: 0.05,
                monitoring_frequency_seconds: 60,
            },
            rebalancing: RebalancingConfig {
                enabled: false,
                frequency_hours: 168, // Weekly
                drift_threshold_pct: 0.05,
                target_allocation_method: "equal_weight".to_string(),
            },
        }
    }
}

impl PortfolioManager {
    /// Create a new portfolio manager
    pub fn new(config: PortfolioManagerConfig) -> Self {
        let mut portfolio_state = PortfolioState::default();
        portfolio_state.inception_date = Some(Utc::now());

        Self {
            config,
            portfolio_state: Arc::new(RwLock::new(portfolio_state)),
            position_tracker: PositionTracker::default(),
            performance_tracker: PerformanceTracker::default(),
            trade_history: VecDeque::new(),
            risk_metrics: RiskMetrics::default(),
        }
    }

    /// Initialize portfolio with starting capital
    pub async fn initialize_portfolio(&mut self, starting_capital: Decimal) -> Result<()> {
        let mut state = self.portfolio_state.write().await;
        state.available_cash = starting_capital;
        state.total_value = starting_capital;
        state.inception_date = Some(Utc::now());
        state.last_updated = Some(Utc::now());
        
        info!("Portfolio initialized with {} {} starting capital", starting_capital, self.config.base_currency);
        Ok(())
    }

    /// Process a new trade
    pub async fn process_trade(&mut self, trade: Trade) -> Result<()> {
        // First, get a copy of the position data for trade impact calculation
        let position_data = {
            let state = self.portfolio_state.read().await;
            state.positions.get(&trade.symbol).cloned()
        };
        
        // Calculate trade impact with the copy (outside of the lock)
        let default_position = Position {
            symbol: trade.symbol.clone(),
            side: if trade.quantity > Decimal::ZERO { crate::models::PositionSide::Long } else { crate::models::PositionSide::Short },
            quantity: Decimal::ZERO,
            entry_price: Decimal::ZERO,
            mark_price: trade.price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let position_for_impact = position_data.as_ref().unwrap_or(&default_position);
        let trade_impact = self.calculate_trade_impact(&trade, position_for_impact).await?;
        
        // Now update the actual position with the write lock
        let mut state = self.portfolio_state.write().await;
        
        // Update position
        let mut realized_pnl_update = Decimal::ZERO;
        
        let position = state.positions.entry(trade.symbol.clone()).or_insert_with(|| Position {
            symbol: trade.symbol.clone(),
            side: if trade.quantity > Decimal::ZERO { crate::models::PositionSide::Long } else { crate::models::PositionSide::Short },
            quantity: Decimal::ZERO,
            entry_price: Decimal::ZERO,
            mark_price: trade.price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        });
        
        // Update position based on trade
        if matches!(trade.side, OrderSide::Buy) {
            let new_quantity = position.quantity + trade.quantity;
            let new_cost_basis = position.entry_price * position.quantity + trade.price * trade.quantity;
            position.entry_price = if new_quantity > Decimal::ZERO {
                new_cost_basis / new_quantity
            } else {
                Decimal::ZERO
            };
            position.quantity = new_quantity;
        } else {
            // Sell trade
            let sell_quantity = trade.quantity.min(position.quantity);
            position.quantity -= sell_quantity;
            
            // Calculate realized P&L
            let realized_pnl = (trade.price - position.entry_price) * sell_quantity - trade.fee;
            position.realized_pnl += realized_pnl;
            realized_pnl_update = realized_pnl;
        }

        position.mark_price = trade.price;
        position.updated_at = Utc::now();

        // Update realized P&L
        state.realized_pnl += realized_pnl_update;

        // Update cash position
        let cash_impact = if matches!(trade.side, OrderSide::Buy) {
            -(trade.quantity * trade.price + trade.fee)
        } else {
            trade.quantity * trade.price - trade.fee
        };
        state.available_cash += cash_impact;

        // Create trade record (all values are cloned/copied before this point)
        let trade_record = TradeRecord {
            trade_id: trade.id,
            order_id: trade.order_id,
            symbol: trade.symbol.clone(),
            side: trade.side.clone(),
            quantity: trade.quantity,
            price: trade.price,
            timestamp: trade.timestamp,
            fees: trade.fee,
            impact_on_portfolio: trade_impact,
        };

        // Update last_updated before dropping state
        state.last_updated = Some(Utc::now());
        
        // Store symbol for analytics update after releasing the lock
        let symbol = trade.symbol.clone();
        let trade_info = (trade.side.clone(), trade.quantity, trade.symbol.clone(), trade.price);
        
        // Drop the state lock before pushing to trade history
        drop(state);
        
        self.trade_history.push_back(trade_record);
        
        // Maintain trade history size
        while self.trade_history.len() > self.config.max_trade_history {
            self.trade_history.pop_front();
        }
        
        // Update portfolio analytics
        self.update_position_analytics(&symbol).await?;
        
        info!("Processed trade: {:?} {} {} at {}", trade_info.0, trade_info.1, trade_info.2, trade_info.3);
        Ok(())
    }

    /// Update market prices for all positions
    pub async fn update_market_prices(&mut self, market_data: &HashMap<String, MarketData>) -> Result<()> {
        let mut state = self.portfolio_state.write().await;
        let mut total_unrealized_pnl = Decimal::ZERO;
        let mut total_value = state.available_cash;

        for (symbol, position) in state.positions.iter_mut() {
            if let Some(market_price) = market_data.get(symbol) {
                position.mark_price = market_price.mid;
                
                // Calculate unrealized P&L
                let unrealized_pnl = (market_price.mid - position.entry_price) * position.quantity;
                position.unrealized_pnl = unrealized_pnl;
                total_unrealized_pnl += unrealized_pnl;
                
                // Add position value to total
                let position_value = position.quantity * market_price.mid;
                total_value += position_value;
                
                position.updated_at = Utc::now();
            }
        }

        state.unrealized_pnl = total_unrealized_pnl;
        state.total_value = total_value;
        state.last_updated = Some(Utc::now());

        // Check if we need to update performance tracking
        let should_update_performance = self.config.enable_realtime_tracking;
        
        // Drop the state lock before calling performance tracking
        drop(state);

        // Update performance tracking
        if should_update_performance {
            self.update_performance_tracking().await?;
        }

        Ok(())
    }

    /// Get current portfolio summary
    pub async fn get_portfolio_summary(&self) -> Result<PortfolioSummary> {
        let state = self.portfolio_state.read().await;
        
        // Calculate position breakdown
        let mut position_breakdown = HashMap::new();
        for (symbol, position) in &state.positions {
            if position.quantity > Decimal::ZERO {
                let market_value = position.quantity * position.mark_price;
                let allocation_pct = if state.total_value > Decimal::ZERO {
                    (market_value / state.total_value).to_f64().unwrap_or(0.0)
                } else {
                    0.0
                };

                // Calculate days held (simplified)
                let days_held = if let Some(analytics) = self.position_tracker.position_analytics.get(symbol) {
                    analytics.holding_period_days
                } else {
                    0
                };

                position_breakdown.insert(symbol.clone(), PositionBreakdown {
                    symbol: symbol.clone(),
                    quantity: position.quantity,
                    market_value,
                    cost_basis: position.entry_price * position.quantity,
                    unrealized_pnl: position.unrealized_pnl,
                    allocation_pct,
                    days_held,
                });
            }
        }

        // Calculate daily P&L (simplified)
        let daily_pnl = if let Some(last_return) = self.performance_tracker.daily_returns.back() {
            (state.total_value - last_return.portfolio_value)
        } else {
            Decimal::ZERO
        };

        Ok(PortfolioSummary {
            total_value: state.total_value,
            available_cash: state.available_cash,
            total_positions: state.positions.len(),
            unrealized_pnl: state.unrealized_pnl,
            realized_pnl: state.realized_pnl,
            daily_pnl,
            performance_metrics: self.performance_tracker.performance_metrics.clone(),
            risk_metrics: self.risk_metrics.clone(),
            position_breakdown,
            sector_allocation: self.calculate_sector_allocation().await,
            geographic_allocation: self.calculate_geographic_allocation().await,
            generated_at: Utc::now(),
        })
    }

    /// Get position analytics for a specific symbol
    pub async fn get_position_analytics(&self, symbol: &str) -> Option<PositionAnalytics> {
        self.position_tracker.position_analytics.get(symbol).cloned()
    }

    /// Get portfolio performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_tracker.performance_metrics.clone()
    }

    /// Get trade history
    pub async fn get_trade_history(&self, limit: Option<usize>) -> Vec<TradeRecord> {
        let limit = limit.unwrap_or(100);
        self.trade_history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Execute rebalancing if needed
    pub async fn check_and_rebalance(&mut self, target_allocations: &HashMap<String, f64>) -> Result<Vec<Order>> {
        if !self.config.rebalancing.enabled {
            return Ok(Vec::new());
        }

        let current_allocations = self.calculate_current_allocations().await?;
        let mut rebalance_orders = Vec::new();

        for (symbol, target_pct) in target_allocations {
            let current_pct = current_allocations.get(symbol).copied().unwrap_or(0.0);
            let drift = (current_pct - target_pct).abs();

            if drift > self.config.rebalancing.drift_threshold_pct {
                let order = self.create_rebalancing_order(symbol, target_pct, current_pct).await?;
                if let Some(order) = order {
                    rebalance_orders.push(order);
                }
            }
        }

        if !rebalance_orders.is_empty() {
            info!("Generated {} rebalancing orders", rebalance_orders.len());
        }

        Ok(rebalance_orders)
    }

    async fn calculate_trade_impact(&self, trade: &Trade, position: &Position) -> Result<TradeImpact> {
        let position_change = if matches!(trade.side, OrderSide::Buy) {
            trade.quantity
        } else {
            -trade.quantity.min(position.quantity)
        };

        let cash_change = if matches!(trade.side, OrderSide::Buy) {
            -(trade.quantity * trade.price + trade.fee)
        } else {
            trade.quantity * trade.price - trade.fee
        };

        let realized_pnl = if matches!(trade.side, OrderSide::Sell) {
            (trade.price - position.entry_price) * trade.quantity.min(position.quantity)
        } else {
            Decimal::ZERO
        };

        let new_position_size = if matches!(trade.side, OrderSide::Buy) {
            position.quantity + trade.quantity
        } else {
            position.quantity - trade.quantity.min(position.quantity)
        };

        // Simplified sector allocation change
        let mut sector_allocation_change = HashMap::new();
        sector_allocation_change.insert("Technology".to_string(), 0.01); // Placeholder

        Ok(TradeImpact {
            position_change,
            cash_change,
            realized_pnl,
            new_position_size,
            sector_allocation_change,
        })
    }

    async fn update_position_analytics(&mut self, symbol: &str) -> Result<()> {
        let state = self.portfolio_state.read().await;
        let position = state.positions.get(symbol);
        
        if let Some(position) = position {
            let analytics = self.position_tracker.position_analytics.entry(symbol.to_string()).or_insert_with(|| {
                PositionAnalytics {
                    symbol: symbol.to_string(),
                    entry_date: Some(Utc::now()),
                    ..Default::default()
                }
            });

            analytics.current_value = position.quantity * position.mark_price;
            analytics.total_cost_basis = position.entry_price * position.quantity;
            analytics.unrealized_pnl = position.unrealized_pnl;
            analytics.realized_pnl = position.realized_pnl;
            analytics.average_entry_price = position.entry_price;

            // Calculate holding period
            if let Some(entry_date) = analytics.entry_date {
                analytics.holding_period_days = (Utc::now() - entry_date).num_days();
            }

            // Update trade count
            let trade_count = self.trade_history.iter()
                .filter(|t| t.symbol == *symbol)
                .count() as u32;
            analytics.number_of_trades = trade_count;

            // Calculate win rate (simplified)
            let winning_trades = self.trade_history.iter()
                .filter(|t| t.symbol == *symbol && t.impact_on_portfolio.realized_pnl > Decimal::ZERO)
                .count();
            
            if trade_count > 0 {
                analytics.win_rate = winning_trades as f64 / trade_count as f64;
            }
        }

        Ok(())
    }

    async fn update_performance_tracking(&mut self) -> Result<()> {
        // Extract needed values from state before dropping the lock
        let (total_value, inception_date) = {
            let state = self.portfolio_state.read().await;
            (state.total_value, state.inception_date)
        };
        
        // Calculate daily return
        let daily_return = if let Some(last_return) = self.performance_tracker.daily_returns.back() {
            let return_value = if last_return.portfolio_value > Decimal::ZERO {
                ((total_value - last_return.portfolio_value) / last_return.portfolio_value).to_f64().unwrap_or(0.0)
            } else {
                0.0
            };
            return_value
        } else {
            0.0
        };

        // Calculate cumulative return
        let cumulative_return = if let Some(_inception_value) = inception_date {
            // Simplified - would use initial capital
            let initial_capital = Decimal::from(100000); // Placeholder
            if initial_capital > Decimal::ZERO {
                ((total_value - initial_capital) / initial_capital).to_f64().unwrap_or(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let daily_return_record = DailyReturn {
            date: Utc::now(),
            portfolio_value: total_value,
            daily_return,
            cumulative_return,
            benchmark_return: None, // Would fetch from benchmark data
        };

        self.performance_tracker.daily_returns.push_back(daily_return_record);

        // Maintain history size (keep last 2 years of daily data)
        while self.performance_tracker.daily_returns.len() > 730 {
            self.performance_tracker.daily_returns.pop_front();
        }

        // Update performance metrics
        self.calculate_performance_metrics().await?;

        Ok(())
    }

    async fn calculate_performance_metrics(&mut self) -> Result<()> {
        let returns: Vec<f64> = self.performance_tracker.daily_returns
            .iter()
            .map(|r| r.daily_return)
            .collect();

        if returns.len() < 2 {
            return Ok(());
        }

        // Calculate basic metrics
        let total_return = self.performance_tracker.daily_returns.back()
            .map(|r| r.cumulative_return)
            .unwrap_or(0.0);

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        let annualized_return = mean_return * 252.0;
        let risk_free_rate = 0.02; // 2% risk-free rate

        let sharpe_ratio = if volatility > 0.0 {
            (annualized_return - risk_free_rate) / volatility
        } else {
            0.0
        };

        // Calculate Sortino ratio
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !negative_returns.is_empty() {
            negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * (252.0_f64).sqrt();

        let sortino_ratio = if downside_deviation > 0.0 {
            (annualized_return - risk_free_rate) / downside_deviation
        } else {
            0.0
        };

        // Calculate maximum drawdown
        let mut peak = 0.0;
        let mut max_drawdown = 0.0;
        
        for daily_return in &self.performance_tracker.daily_returns {
            let value = daily_return.cumulative_return;
            if value > peak {
                peak = value;
            } else {
                let drawdown = (peak - value) / (1.0 + peak).max(0.01);
                max_drawdown = f64::max(max_drawdown, drawdown);
            }
        }

        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Calculate win rate
        let winning_days = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = winning_days as f64 / returns.len() as f64;

        // Calculate profit factor
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            0.0
        };

        self.performance_tracker.performance_metrics = PerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            information_ratio: None, // Would calculate vs benchmark
            tracking_error: None,    // Would calculate vs benchmark
        };

        Ok(())
    }

    async fn calculate_sector_allocation(&self) -> HashMap<String, f64> {
        // Simplified sector allocation - would use actual sector data
        let mut allocations = HashMap::new();
        allocations.insert("Technology".to_string(), 0.40);
        allocations.insert("Finance".to_string(), 0.25);
        allocations.insert("Healthcare".to_string(), 0.20);
        allocations.insert("Other".to_string(), 0.15);
        allocations
    }

    async fn calculate_geographic_allocation(&self) -> HashMap<String, f64> {
        // Simplified geographic allocation - would use actual geographic data
        let mut allocations = HashMap::new();
        allocations.insert("United States".to_string(), 0.60);
        allocations.insert("Europe".to_string(), 0.25);
        allocations.insert("Asia".to_string(), 0.15);
        allocations
    }

    async fn calculate_current_allocations(&self) -> Result<HashMap<String, f64>> {
        let state = self.portfolio_state.read().await;
        let mut allocations = HashMap::new();

        if state.total_value <= Decimal::ZERO {
            return Ok(allocations);
        }

        for (symbol, position) in &state.positions {
            if position.quantity > Decimal::ZERO {
                let position_value = position.quantity * position.mark_price;
                let allocation_pct = (position_value / state.total_value).to_f64().unwrap_or(0.0);
                allocations.insert(symbol.clone(), allocation_pct);
            }
        }

        Ok(allocations)
    }

    async fn create_rebalancing_order(&self, symbol: &str, target_pct: &f64, current_pct: f64) -> Result<Option<Order>> {
        let state = self.portfolio_state.read().await;
        
        if state.total_value <= Decimal::ZERO {
            return Ok(None);
        }

        let target_value = state.total_value * Decimal::from_f64_retain(*target_pct).unwrap_or_default();
        let current_value = state.total_value * Decimal::from_f64_retain(current_pct).unwrap_or_default();
        let difference = target_value - current_value;

        if difference.abs() < self.config.position_sizing.min_position_size {
            return Ok(None);
        }

        let current_position = state.positions.get(symbol);
        let current_price = current_position.map(|p| p.mark_price).unwrap_or(Decimal::from(100)); // Fallback price

        let quantity_change = difference / current_price;
        let side = if quantity_change > Decimal::ZERO { OrderSide::Buy } else { OrderSide::Sell };

        let order = Order {
            id: uuid::Uuid::new_v4(),
            symbol: symbol.to_string(),
            side,
            order_type: crate::models::OrderType::Market,
            quantity: quantity_change.abs(),
            price: Some(current_price),
            time_in_force: crate::models::TimeInForce::IOC,
            status: crate::models::OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        Ok(Some(order))
    }

    /// Get portfolio risk metrics
    pub async fn get_risk_metrics(&self) -> RiskMetrics {
        self.risk_metrics.clone()
    }

    /// Update risk metrics based on current portfolio
    pub async fn update_risk_metrics(&mut self, _market_data: &HashMap<String, MarketData>) -> Result<()> {
        let state = self.portfolio_state.read().await;
        
        // Calculate portfolio VaR (simplified)
        let mut portfolio_volatility = 0.0;
        let mut total_weight_squared = 0.0;
        
        for (symbol, position) in &state.positions {
            if position.quantity > Decimal::ZERO && state.total_value > Decimal::ZERO {
                let weight = ((position.quantity * position.mark_price) / state.total_value).to_f64().unwrap_or(0.0);
                let volatility = 0.25; // Simplified - would calculate from market data
                portfolio_volatility += weight * weight * volatility * volatility;
                total_weight_squared += weight * weight;
            }
        }
        
        portfolio_volatility = portfolio_volatility.sqrt();
        
        self.risk_metrics.portfolio_var_95 = portfolio_volatility * 1.645; // 95% VaR
        self.risk_metrics.portfolio_var_99 = portfolio_volatility * 2.326; // 99% VaR
        self.risk_metrics.expected_shortfall = self.risk_metrics.portfolio_var_95 * 1.3;
        self.risk_metrics.concentration_risk = total_weight_squared.sqrt();
        
        // Calculate leverage ratio
        let total_exposure: Decimal = state.positions.values()
            .map(|p| (p.quantity * p.mark_price).abs())
            .sum();
        self.risk_metrics.leverage_ratio = if state.total_value > Decimal::ZERO {
            (total_exposure / state.total_value).to_f64().unwrap_or(1.0)
        } else {
            1.0
        };

        Ok(())
    }
}