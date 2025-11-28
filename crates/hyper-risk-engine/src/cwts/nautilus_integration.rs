//! Nautilus Trader Integration for CWTS Risk Management
//!
//! Provides execution risk management and backtesting integration with
//! NautilusTrader's production-grade trading infrastructure.
//!
//! ## Features
//!
//! - **Execution Risk**: Monitor slippage, fill rates, market impact
//! - **Backtest Risk**: Historical risk metrics and scenario analysis
//! - **Live Trading Guard**: Real-time position and exposure limits
//!
//! ## Risk Applications
//!
//! - Pre-trade execution risk assessment
//! - Post-trade execution quality analysis
//! - Backtest risk metrics validation
//! - Live trading risk barriers

use crate::core::{RiskLevel, Symbol, Price, Quantity, Order, OrderSide};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration};

// Re-export from hyperphysics-nautilus when available
#[cfg(feature = "cwts-nautilus")]
use hyperphysics_nautilus::prelude::*;

/// Execution risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRisk {
    /// Expected slippage (bps)
    pub expected_slippage_bps: f64,
    /// Worst case slippage (bps)
    pub worst_case_slippage_bps: f64,
    /// Expected fill rate
    pub expected_fill_rate: f64,
    /// Market impact estimate
    pub market_impact: f64,
    /// Time to fill estimate (ms)
    pub time_to_fill_ms: u64,
    /// Execution risk score (0.0-1.0)
    pub risk_score: f64,
    /// Recommended execution strategy
    pub recommended_strategy: ExecutionStrategy,
}

impl ExecutionRisk {
    /// Convert to risk level
    #[must_use]
    pub fn to_risk_level(&self) -> RiskLevel {
        if self.risk_score > 0.8 {
            RiskLevel::Critical
        } else if self.risk_score > 0.6 {
            RiskLevel::High
        } else if self.risk_score > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
}

/// Recommended execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Immediate execution (market order)
    Immediate,
    /// Time-weighted average price
    TWAP,
    /// Volume-weighted average price
    VWAP,
    /// Implementation shortfall minimization
    IS,
    /// Participation rate
    POV,
    /// Arrival price
    ArrivalPrice,
    /// Iceberg order
    Iceberg,
    /// Sniper (opportunistic)
    Sniper,
}

impl ExecutionStrategy {
    /// Get expected slippage for this strategy
    #[must_use]
    pub fn expected_slippage_reduction(&self) -> f64 {
        match self {
            Self::Immediate => 0.0,
            Self::TWAP => 0.3,
            Self::VWAP => 0.4,
            Self::IS => 0.5,
            Self::POV => 0.35,
            Self::ArrivalPrice => 0.45,
            Self::Iceberg => 0.25,
            Self::Sniper => 0.6,
        }
    }
}

/// Backtest risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRiskMetrics {
    /// Total return
    pub total_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Value at Risk (99%)
    pub var_99: f64,
    /// Conditional VaR (95%)
    pub cvar_95: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average trade duration
    pub avg_trade_duration_ms: u64,
    /// Number of trades
    pub trade_count: u64,
    /// Maximum consecutive losses
    pub max_consecutive_losses: u32,
    /// Regime-specific metrics
    pub regime_metrics: HashMap<String, RegimeMetrics>,
}

impl BacktestRiskMetrics {
    /// Calculate overall risk score
    #[must_use]
    pub fn risk_score(&self) -> f64 {
        // Weight different metrics
        let drawdown_risk = self.max_drawdown.abs().min(1.0);
        let var_risk = self.var_99.abs().min(1.0);
        let sharpe_risk = (2.0 - self.sharpe_ratio.max(0.0)).max(0.0) / 2.0;

        (drawdown_risk * 0.4 + var_risk * 0.3 + sharpe_risk * 0.3).min(1.0)
    }

    /// Get risk level from backtest
    #[must_use]
    pub fn to_risk_level(&self) -> RiskLevel {
        let score = self.risk_score();
        if score > 0.8 {
            RiskLevel::Critical
        } else if score > 0.6 {
            RiskLevel::High
        } else if score > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }

    /// Check if backtest passes risk requirements
    #[must_use]
    pub fn passes_requirements(&self, requirements: &BacktestRequirements) -> bool {
        self.max_drawdown.abs() <= requirements.max_drawdown &&
        self.sharpe_ratio >= requirements.min_sharpe &&
        self.var_99.abs() <= requirements.max_var_99 &&
        self.win_rate >= requirements.min_win_rate
    }
}

/// Regime-specific backtest metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeMetrics {
    /// Return in this regime
    pub return_pct: f64,
    /// Win rate in this regime
    pub win_rate: f64,
    /// Average trade in this regime
    pub avg_trade: f64,
    /// Trade count in this regime
    pub trade_count: u32,
}

/// Requirements for backtest validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequirements {
    /// Maximum acceptable drawdown
    pub max_drawdown: f64,
    /// Minimum Sharpe ratio
    pub min_sharpe: f64,
    /// Maximum VaR 99%
    pub max_var_99: f64,
    /// Minimum win rate
    pub min_win_rate: f64,
    /// Minimum trade count for statistical significance
    pub min_trades: u64,
    /// Minimum backtest duration (days)
    pub min_duration_days: u32,
}

impl Default for BacktestRequirements {
    fn default() -> Self {
        Self {
            max_drawdown: 0.20,
            min_sharpe: 1.0,
            max_var_99: 0.05,
            min_win_rate: 0.45,
            min_trades: 100,
            min_duration_days: 252, // 1 year
        }
    }
}

/// Live trading guard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveTradingGuard {
    /// Maximum position size per symbol
    pub max_position_size: HashMap<Symbol, f64>,
    /// Maximum total exposure
    pub max_total_exposure: f64,
    /// Maximum daily loss limit
    pub max_daily_loss: f64,
    /// Maximum orders per second
    pub max_orders_per_second: u32,
    /// Maximum open orders
    pub max_open_orders: u32,
    /// Enabled kill switch
    pub kill_switch_enabled: bool,
    /// Current daily P&L
    current_daily_pnl: RwLock<f64>,
    /// Order rate counter
    order_count_window: RwLock<Vec<DateTime<Utc>>>,
    /// Open order count
    open_orders: AtomicU64,
    /// Trading halted flag
    halted: AtomicBool,
}

impl LiveTradingGuard {
    /// Create a new trading guard
    #[must_use]
    pub fn new(max_exposure: f64, max_daily_loss: f64) -> Self {
        Self {
            max_position_size: HashMap::new(),
            max_total_exposure: max_exposure,
            max_daily_loss,
            max_orders_per_second: 10,
            max_open_orders: 100,
            kill_switch_enabled: true,
            current_daily_pnl: RwLock::new(0.0),
            order_count_window: RwLock::new(Vec::new()),
            open_orders: AtomicU64::new(0),
            halted: AtomicBool::new(false),
        }
    }

    /// Set position limit for a symbol
    pub fn set_position_limit(&mut self, symbol: Symbol, limit: f64) {
        self.max_position_size.insert(symbol, limit);
    }

    /// Check if order is allowed
    pub fn check_order(&self, symbol: &Symbol, quantity: f64, current_position: f64) -> OrderCheckResult {
        // Check if trading is halted
        if self.halted.load(Ordering::SeqCst) {
            return OrderCheckResult::Rejected(RejectReason::TradingHalted);
        }

        // Check position limits
        if let Some(&limit) = self.max_position_size.get(symbol) {
            if (current_position + quantity).abs() > limit {
                return OrderCheckResult::Rejected(RejectReason::PositionLimitExceeded);
            }
        }

        // Check order rate
        {
            let mut window = self.order_count_window.write();
            let now = Utc::now();
            window.retain(|t| now - *t < Duration::seconds(1));
            if window.len() >= self.max_orders_per_second as usize {
                return OrderCheckResult::Rejected(RejectReason::RateLimitExceeded);
            }
            window.push(now);
        }

        // Check open orders
        if self.open_orders.load(Ordering::SeqCst) >= self.max_open_orders as u64 {
            return OrderCheckResult::Rejected(RejectReason::TooManyOpenOrders);
        }

        // Check daily loss
        if *self.current_daily_pnl.read() < -self.max_daily_loss {
            return OrderCheckResult::Rejected(RejectReason::DailyLossLimitReached);
        }

        OrderCheckResult::Approved
    }

    /// Update daily P&L
    pub fn update_pnl(&self, pnl_change: f64) {
        let mut pnl = self.current_daily_pnl.write();
        *pnl += pnl_change;

        // Trigger kill switch if enabled and limit exceeded
        if self.kill_switch_enabled && *pnl < -self.max_daily_loss {
            self.halt_trading();
        }
    }

    /// Increment open order count
    pub fn order_sent(&self) {
        self.open_orders.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement open order count
    pub fn order_filled_or_cancelled(&self) {
        self.open_orders.fetch_sub(1, Ordering::SeqCst);
    }

    /// Halt all trading
    pub fn halt_trading(&self) {
        self.halted.store(true, Ordering::SeqCst);
    }

    /// Resume trading
    pub fn resume_trading(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }

    /// Check if trading is halted
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::SeqCst)
    }

    /// Reset daily metrics (call at start of day)
    pub fn reset_daily(&self) {
        *self.current_daily_pnl.write() = 0.0;
        self.order_count_window.write().clear();
    }

    /// Get current status
    #[must_use]
    pub fn status(&self) -> GuardStatus {
        GuardStatus {
            is_halted: self.is_halted(),
            daily_pnl: *self.current_daily_pnl.read(),
            open_orders: self.open_orders.load(Ordering::SeqCst),
            pnl_utilization: (*self.current_daily_pnl.read() / -self.max_daily_loss).clamp(-1.0, 1.0),
        }
    }
}

/// Result of order check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderCheckResult {
    /// Order approved
    Approved,
    /// Order rejected with reason
    Rejected(RejectReason),
}

/// Reasons for order rejection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// Trading is halted
    TradingHalted,
    /// Position limit exceeded
    PositionLimitExceeded,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Too many open orders
    TooManyOpenOrders,
    /// Daily loss limit reached
    DailyLossLimitReached,
    /// Exposure limit exceeded
    ExposureLimitExceeded,
}

/// Guard status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardStatus {
    /// Is trading halted
    pub is_halted: bool,
    /// Current daily P&L
    pub daily_pnl: f64,
    /// Open order count
    pub open_orders: u64,
    /// P&L utilization (-1 to 1, negative = loss)
    pub pnl_utilization: f64,
}

/// Configuration for Nautilus risk adapter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NautilusConfig {
    /// Execution cost model
    pub cost_model: CostModel,
    /// Slippage model
    pub slippage_model: SlippageModel,
    /// Backtest requirements
    pub backtest_requirements: BacktestRequirements,
    /// Enable execution analytics
    pub enable_analytics: bool,
}

impl Default for NautilusConfig {
    fn default() -> Self {
        Self {
            cost_model: CostModel::default(),
            slippage_model: SlippageModel::default(),
            backtest_requirements: BacktestRequirements::default(),
            enable_analytics: true,
        }
    }
}

/// Execution cost model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Fixed cost per trade
    pub fixed_cost: f64,
    /// Variable cost (bps)
    pub variable_cost_bps: f64,
    /// Exchange fee (bps)
    pub exchange_fee_bps: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            fixed_cost: 0.0,
            variable_cost_bps: 0.5,
            exchange_fee_bps: 2.0,
        }
    }
}

/// Slippage model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Base slippage (bps)
    pub base_slippage_bps: f64,
    /// Volume impact factor
    pub volume_impact: f64,
    /// Volatility impact factor
    pub volatility_impact: f64,
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self {
            base_slippage_bps: 1.0,
            volume_impact: 0.1,
            volatility_impact: 0.5,
        }
    }
}

/// Nautilus-based risk adapter
///
/// Provides execution risk management for live trading:
/// - Pre-trade execution risk assessment
/// - Live trading guards and limits
/// - Post-trade execution quality analysis
pub struct NautilusRiskAdapter {
    config: NautilusConfig,
    guard: RwLock<LiveTradingGuard>,
    execution_history: RwLock<Vec<ExecutionRecord>>,
    backtest_cache: RwLock<HashMap<String, BacktestRiskMetrics>>,
}

/// Record of an execution
#[derive(Debug, Clone)]
struct ExecutionRecord {
    timestamp: DateTime<Utc>,
    symbol: Symbol,
    side: OrderSide,
    requested_qty: f64,
    filled_qty: f64,
    expected_price: f64,
    actual_price: f64,
    slippage_bps: f64,
    time_to_fill_ms: u64,
}

impl NautilusRiskAdapter {
    /// Create a new Nautilus risk adapter
    #[must_use]
    pub fn new(config: NautilusConfig) -> Self {
        Self {
            config,
            guard: RwLock::new(LiveTradingGuard::new(1_000_000.0, 50_000.0)),
            execution_history: RwLock::new(Vec::new()),
            backtest_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Get reference to trading guard
    pub fn guard(&self) -> impl std::ops::Deref<Target = LiveTradingGuard> + '_ {
        self.guard.read()
    }

    /// Get mutable reference to trading guard
    pub fn guard_mut(&self) -> impl std::ops::DerefMut<Target = LiveTradingGuard> + '_ {
        self.guard.write()
    }

    /// Assess execution risk for an order
    pub fn assess_execution_risk(
        &self,
        symbol: &Symbol,
        quantity: f64,
        price: f64,
        side: OrderSide,
        market_volatility: f64,
        available_liquidity: f64,
    ) -> ExecutionRisk {
        // Calculate slippage estimate
        let volume_ratio = quantity / available_liquidity.max(1.0);
        let base_slippage = self.config.slippage_model.base_slippage_bps;
        let vol_slippage = market_volatility * self.config.slippage_model.volatility_impact * 100.0;
        let volume_slippage = volume_ratio * self.config.slippage_model.volume_impact * 100.0;

        let expected_slippage = base_slippage + vol_slippage + volume_slippage;
        let worst_case_slippage = expected_slippage * 2.5;

        // Estimate fill rate
        let fill_rate = (1.0 - volume_ratio.min(0.5)).max(0.5);

        // Market impact (square root law)
        let market_impact = (quantity / available_liquidity.max(1.0)).sqrt() * market_volatility;

        // Time to fill estimate
        let time_to_fill = if volume_ratio < 0.1 {
            100 // Immediate
        } else if volume_ratio < 0.3 {
            (volume_ratio * 5000.0) as u64
        } else {
            (volume_ratio * 20000.0) as u64
        };

        // Overall risk score
        let risk_score = (expected_slippage / 50.0 + volume_ratio + market_impact).clamp(0.0, 1.0);

        // Recommend execution strategy
        let strategy = if volume_ratio < 0.05 {
            ExecutionStrategy::Immediate
        } else if volume_ratio < 0.15 {
            ExecutionStrategy::TWAP
        } else if volume_ratio < 0.3 {
            ExecutionStrategy::VWAP
        } else if market_volatility > 0.03 {
            ExecutionStrategy::IS
        } else {
            ExecutionStrategy::Iceberg
        };

        ExecutionRisk {
            expected_slippage_bps: expected_slippage,
            worst_case_slippage_bps: worst_case_slippage,
            expected_fill_rate: fill_rate,
            market_impact,
            time_to_fill_ms: time_to_fill,
            risk_score,
            recommended_strategy: strategy,
        }
    }

    /// Record an execution
    pub fn record_execution(
        &self,
        symbol: Symbol,
        side: OrderSide,
        requested_qty: f64,
        filled_qty: f64,
        expected_price: f64,
        actual_price: f64,
        time_to_fill_ms: u64,
    ) {
        let slippage_bps = match side {
            OrderSide::Buy => (actual_price - expected_price) / expected_price * 10000.0,
            OrderSide::Sell => (expected_price - actual_price) / expected_price * 10000.0,
        };

        let record = ExecutionRecord {
            timestamp: Utc::now(),
            symbol,
            side,
            requested_qty,
            filled_qty,
            expected_price,
            actual_price,
            slippage_bps,
            time_to_fill_ms,
        };

        let mut history = self.execution_history.write();
        history.push(record);

        // Keep last 10000 executions
        if history.len() > 10000 {
            history.remove(0);
        }
    }

    /// Get execution quality metrics
    #[must_use]
    pub fn execution_quality(&self) -> ExecutionQualityMetrics {
        let history = self.execution_history.read();

        if history.is_empty() {
            return ExecutionQualityMetrics::default();
        }

        let avg_slippage = history.iter()
            .map(|r| r.slippage_bps)
            .sum::<f64>() / history.len() as f64;

        let fill_rate = history.iter()
            .map(|r| r.filled_qty / r.requested_qty)
            .sum::<f64>() / history.len() as f64;

        let avg_time_to_fill = history.iter()
            .map(|r| r.time_to_fill_ms)
            .sum::<u64>() / history.len() as u64;

        ExecutionQualityMetrics {
            avg_slippage_bps: avg_slippage,
            fill_rate,
            avg_time_to_fill_ms: avg_time_to_fill,
            execution_count: history.len() as u64,
        }
    }

    /// Store backtest results
    pub fn store_backtest(&self, name: &str, metrics: BacktestRiskMetrics) {
        self.backtest_cache.write().insert(name.to_string(), metrics);
    }

    /// Get cached backtest results
    #[must_use]
    pub fn get_backtest(&self, name: &str) -> Option<BacktestRiskMetrics> {
        self.backtest_cache.read().get(name).cloned()
    }

    /// Validate backtest meets requirements
    #[must_use]
    pub fn validate_backtest(&self, metrics: &BacktestRiskMetrics) -> BacktestValidation {
        let requirements = &self.config.backtest_requirements;

        let mut violations = Vec::new();

        if metrics.max_drawdown.abs() > requirements.max_drawdown {
            violations.push(format!(
                "Drawdown {:.2}% exceeds limit {:.2}%",
                metrics.max_drawdown * 100.0,
                requirements.max_drawdown * 100.0
            ));
        }

        if metrics.sharpe_ratio < requirements.min_sharpe {
            violations.push(format!(
                "Sharpe {:.2} below minimum {:.2}",
                metrics.sharpe_ratio,
                requirements.min_sharpe
            ));
        }

        if metrics.var_99.abs() > requirements.max_var_99 {
            violations.push(format!(
                "VaR99 {:.2}% exceeds limit {:.2}%",
                metrics.var_99 * 100.0,
                requirements.max_var_99 * 100.0
            ));
        }

        if metrics.trade_count < requirements.min_trades {
            violations.push(format!(
                "Trade count {} below minimum {}",
                metrics.trade_count,
                requirements.min_trades
            ));
        }

        BacktestValidation {
            passed: violations.is_empty(),
            violations,
            risk_score: metrics.risk_score(),
        }
    }

    /// Assess execution risk for a portfolio position on a specific symbol.
    ///
    /// Returns a `SubsystemRisk` for integration with the CWTS coordinator.
    /// This method aggregates execution risk, guard status, and recent execution quality.
    pub fn assess_portfolio_execution_risk(
        &self,
        portfolio: &crate::core::Portfolio,
        symbol: &Symbol,
    ) -> super::coordinator::SubsystemRisk {
        use super::coordinator::{SubsystemRisk, SubsystemId};
        use crate::core::Timestamp;

        let start = std::time::Instant::now();

        // Get guard status
        let guard_status = self.guard.read().status();

        // Get execution quality metrics
        let quality = self.execution_quality();

        // Calculate position size and risk for this symbol
        let position = portfolio.positions.iter().find(|p| &p.symbol == symbol);
        let position_value = position
            .map(|p| p.quantity.as_f64() * p.avg_entry_price.as_f64())
            .unwrap_or(0.0);

        // Estimate market conditions (simplified - in production, use real market data)
        let volatility = 0.02; // 2% assumed volatility
        let liquidity = 100_000.0; // Assumed liquidity

        // Assess execution risk for a hypothetical trade
        let exec_risk = self.assess_execution_risk(
            symbol,
            position_value.abs() / 1000.0, // Convert to approximate quantity
            position
                .map(|p| p.avg_entry_price.as_f64())
                .unwrap_or(100.0),
            crate::core::OrderSide::Buy, // Direction doesn't matter for risk
            volatility,
            liquidity,
        );

        // Calculate combined risk score
        let pnl_risk = (-guard_status.pnl_utilization).max(0.0); // 0 to 1
        let slippage_risk = (quality.avg_slippage_bps / 50.0).min(1.0); // 50 bps = 1.0 risk
        let fill_risk = 1.0 - quality.fill_rate;
        let exec_score_risk = exec_risk.risk_score;

        let combined_risk = (
            pnl_risk * 0.35 +
            slippage_risk * 0.20 +
            fill_risk * 0.15 +
            exec_score_risk * 0.30
        ).clamp(0.0, 1.0);

        // Determine risk level
        let risk_level = if guard_status.is_halted {
            crate::core::RiskLevel::Critical
        } else if combined_risk > 0.8 || pnl_risk > 0.9 {
            crate::core::RiskLevel::Critical
        } else if combined_risk > 0.6 || pnl_risk > 0.7 {
            crate::core::RiskLevel::High
        } else if combined_risk > 0.3 {
            crate::core::RiskLevel::Elevated
        } else {
            crate::core::RiskLevel::Normal
        };

        // Position factor based on execution conditions
        let position_factor = if guard_status.is_halted {
            0.0
        } else {
            (1.0 - combined_risk * 0.5).clamp(0.3, 1.0)
        };

        // Confidence based on execution history
        let confidence = if quality.execution_count < 10 {
            0.5 // Low confidence with limited history
        } else {
            (quality.fill_rate * 0.7 + 0.3).min(0.95)
        };

        let latency_ns = start.elapsed().as_nanos() as u64;

        let reasoning = format!(
            "Nautilus: daily_pnl=${:.0}, slippage={:.1}bps, fill_rate={:.1}%, exec_risk={:.2}",
            guard_status.daily_pnl,
            quality.avg_slippage_bps,
            quality.fill_rate * 100.0,
            exec_risk.risk_score
        );

        SubsystemRisk {
            subsystem: SubsystemId::Nautilus,
            risk_level,
            confidence,
            risk_score: combined_risk,
            position_factor,
            reasoning,
            timestamp: Timestamp::now(),
            latency_ns,
        }
    }

    /// Get overall risk level
    #[must_use]
    pub fn risk_level(&self) -> RiskLevel {
        let guard_status = self.guard.read().status();

        if guard_status.is_halted {
            return RiskLevel::Critical;
        }

        let pnl_risk = (-guard_status.pnl_utilization).max(0.0);

        if pnl_risk > 0.8 {
            RiskLevel::Critical
        } else if pnl_risk > 0.6 {
            RiskLevel::High
        } else if pnl_risk > 0.3 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
}

impl Default for NautilusRiskAdapter {
    fn default() -> Self {
        Self::new(NautilusConfig::default())
    }
}

/// Execution quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionQualityMetrics {
    /// Average slippage in bps
    pub avg_slippage_bps: f64,
    /// Fill rate (0.0-1.0)
    pub fill_rate: f64,
    /// Average time to fill in ms
    pub avg_time_to_fill_ms: u64,
    /// Total execution count
    pub execution_count: u64,
}

/// Backtest validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestValidation {
    /// Whether backtest passed
    pub passed: bool,
    /// List of violations
    pub violations: Vec<String>,
    /// Risk score
    pub risk_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_risk_assessment() {
        let adapter = NautilusRiskAdapter::default();

        let risk = adapter.assess_execution_risk(
            &Symbol::from("BTC"),
            10.0,
            50000.0,
            OrderSide::Buy,
            0.02, // 2% volatility
            100.0, // Available liquidity
        );

        assert!(risk.expected_slippage_bps > 0.0);
        assert!(risk.expected_fill_rate > 0.0 && risk.expected_fill_rate <= 1.0);
        assert!(risk.risk_score >= 0.0 && risk.risk_score <= 1.0);
    }

    #[test]
    fn test_execution_strategy_selection() {
        let adapter = NautilusRiskAdapter::default();

        // Small order should get immediate execution
        let small_risk = adapter.assess_execution_risk(
            &Symbol::from("ETH"),
            1.0,
            3000.0,
            OrderSide::Buy,
            0.01,
            1000.0,
        );
        assert_eq!(small_risk.recommended_strategy, ExecutionStrategy::Immediate);

        // Large order should get VWAP or similar
        let large_risk = adapter.assess_execution_risk(
            &Symbol::from("ETH"),
            200.0,
            3000.0,
            OrderSide::Buy,
            0.01,
            1000.0,
        );
        assert!(matches!(
            large_risk.recommended_strategy,
            ExecutionStrategy::VWAP | ExecutionStrategy::TWAP | ExecutionStrategy::Iceberg
        ));
    }

    #[test]
    fn test_trading_guard() {
        let guard = LiveTradingGuard::new(100_000.0, 5_000.0);

        // Should approve normal order
        let result = guard.check_order(&Symbol::from("BTC"), 1.0, 0.0);
        assert_eq!(result, OrderCheckResult::Approved);

        // Halt and check rejection
        guard.halt_trading();
        let result = guard.check_order(&Symbol::from("BTC"), 1.0, 0.0);
        assert!(matches!(result, OrderCheckResult::Rejected(RejectReason::TradingHalted)));
    }

    #[test]
    fn test_backtest_validation() {
        let adapter = NautilusRiskAdapter::default();

        let good_metrics = BacktestRiskMetrics {
            total_return: 0.50,
            sharpe_ratio: 2.0,
            sortino_ratio: 2.5,
            max_drawdown: -0.10,
            calmar_ratio: 5.0,
            var_95: -0.02,
            var_99: -0.03,
            cvar_95: -0.025,
            win_rate: 0.55,
            profit_factor: 1.8,
            avg_trade_duration_ms: 3600000,
            trade_count: 500,
            max_consecutive_losses: 5,
            regime_metrics: HashMap::new(),
        };

        let validation = adapter.validate_backtest(&good_metrics);
        assert!(validation.passed);

        let bad_metrics = BacktestRiskMetrics {
            max_drawdown: -0.50, // Too high
            sharpe_ratio: 0.5,    // Too low
            ..good_metrics.clone()
        };

        let validation = adapter.validate_backtest(&bad_metrics);
        assert!(!validation.passed);
        assert!(!validation.violations.is_empty());
    }

    #[test]
    fn test_execution_recording() {
        let adapter = NautilusRiskAdapter::default();

        adapter.record_execution(
            Symbol::from("BTC"),
            OrderSide::Buy,
            1.0,
            1.0,
            50000.0,
            50010.0,
            50,
        );

        let quality = adapter.execution_quality();
        assert_eq!(quality.execution_count, 1);
        assert!(quality.avg_slippage_bps > 0.0); // Should have positive slippage for buy
    }
}
