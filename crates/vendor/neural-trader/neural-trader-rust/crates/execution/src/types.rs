use serde::{Deserialize, Serialize};
use std::fmt;

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderSide {
    Buy,
    Sell,
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "buy"),
            OrderSide::Sell => write!(f, "sell"),
        }
    }
}

impl OrderSide {
    pub fn from_str(s: &str) -> Result<Self, ExecutionError> {
        match s.to_lowercase().as_str() {
            "buy" | "long" => Ok(OrderSide::Buy),
            "sell" | "short" => Ok(OrderSide::Sell),
            _ => Err(ExecutionError::InvalidOrderSide(s.to_string())),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "market"),
            OrderType::Limit => write!(f, "limit"),
            OrderType::Stop => write!(f, "stop"),
            OrderType::StopLimit => write!(f, "stop_limit"),
        }
    }
}

impl OrderType {
    pub fn from_str(s: &str) -> Result<Self, ExecutionError> {
        match s.to_lowercase().as_str() {
            "market" => Ok(OrderType::Market),
            "limit" => Ok(OrderType::Limit),
            "stop" => Ok(OrderType::Stop),
            "stop_limit" => Ok(OrderType::StopLimit),
            _ => Err(ExecutionError::InvalidOrderType(s.to_string())),
        }
    }
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Pending => write!(f, "pending"),
            OrderStatus::Submitted => write!(f, "submitted"),
            OrderStatus::PartiallyFilled => write!(f, "partially_filled"),
            OrderStatus::Filled => write!(f, "filled"),
            OrderStatus::Cancelled => write!(f, "cancelled"),
            OrderStatus::Rejected => write!(f, "rejected"),
            OrderStatus::Expired => write!(f, "expired"),
        }
    }
}

/// Order validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderValidation {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl OrderValidation {
    pub fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.valid = false;
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.valid && self.errors.is_empty()
    }
}

impl Default for OrderValidation {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution errors
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Invalid order side: {0}")]
    InvalidOrderSide(String),

    #[error("Invalid order type: {0}")]
    InvalidOrderType(String),

    #[error("Invalid symbol format: {0}")]
    InvalidSymbol(String),

    #[error("Invalid quantity: {0}")]
    InvalidQuantity(i32),

    #[error("Limit price required for limit orders")]
    LimitPriceRequired,

    #[error("Live trading is disabled. Set ENABLE_LIVE_TRADING=true to enable")]
    LiveTradingDisabled,

    #[error("Order validation failed: {0}")]
    ValidationFailed(String),

    #[error("Broker API error: {0}")]
    BrokerApiError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Environment variable error: {0}")]
    EnvError(#[from] std::env::VarError),

    #[error("HTTP request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ExecutionError>;
