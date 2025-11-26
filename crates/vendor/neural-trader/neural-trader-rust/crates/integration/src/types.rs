//! Common types used across the integration layer.

use serde::{Serialize, Deserialize};
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};

/// Result of executing a trading strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub strategy_name: String,
    pub timestamp: DateTime<Utc>,
    pub orders: Vec<OrderResult>,
    pub total_value: Decimal,
    pub profit_loss: Decimal,
    pub metadata: serde_json::Value,
}

/// Result of placing an order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderResult {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub price: Decimal,
    pub status: OrderStatus,
    pub filled_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Filled,
    PartiallyFilled,
    Cancelled,
    Rejected,
}

/// Portfolio state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub total_value: Decimal,
    pub cash: Decimal,
    pub positions: Vec<Position>,
    pub updated_at: DateTime<Utc>,
}

/// Individual position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: Decimal,
    pub average_price: Decimal,
    pub current_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pl: Decimal,
    pub unrealized_pl_percent: Decimal,
}

/// Risk analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReport {
    pub timestamp: DateTime<Utc>,
    pub var_95: Decimal,
    pub var_99: Decimal,
    pub cvar_95: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub beta: Decimal,
    pub position_risks: Vec<PositionRisk>,
    pub alerts: Vec<RiskAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRisk {
    pub symbol: String,
    pub exposure: Decimal,
    pub risk_score: f64,
    pub concentration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAlert {
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

/// Model training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    pub model_type: String,
    pub training_data: String,
    pub parameters: serde_json::Value,
    pub validation_split: f64,
    pub epochs: usize,
}

/// Performance report for a time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub period: TimePeriod,
    pub total_return: Decimal,
    pub annualized_return: Decimal,
    pub sharpe_ratio: Decimal,
    pub max_drawdown: Decimal,
    pub win_rate: f64,
    pub profit_factor: Decimal,
    pub trades: usize,
    pub winners: usize,
    pub losers: usize,
}

/// Time period for analysis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimePeriod {
    Day,
    Week,
    Month,
    Quarter,
    Year,
    AllTime,
    Custom { start: DateTime<Utc>, end: DateTime<Utc> },
}

/// System health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub broker_pool: ComponentHealth,
    pub strategy_manager: ComponentHealth,
    pub model_registry: ComponentHealth,
    pub trading_service: ComponentHealth,
    pub risk_service: ComponentHealth,
    pub neural_service: ComponentHealth,
    pub analytics_service: ComponentHealth,
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        self.broker_pool.is_healthy()
            && self.strategy_manager.is_healthy()
            && self.model_registry.is_healthy()
            && self.trading_service.is_healthy()
            && self.risk_service.is_healthy()
            && self.neural_service.is_healthy()
            && self.analytics_service.is_healthy()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatusEnum,
    pub message: Option<String>,
    pub last_check: DateTime<Utc>,
    pub uptime: std::time::Duration,
}

impl ComponentHealth {
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, HealthStatusEnum::Healthy)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatusEnum {
    Healthy,
    Degraded,
    Unhealthy,
}
