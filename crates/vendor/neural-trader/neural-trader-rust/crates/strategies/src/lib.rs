//! Neural Trader Strategy Integration
//!
//! Complete integration of 7 trading strategies with:
//! - Broker execution (Agent 3)
//! - Neural predictions (Agent 4)
//! - Risk management (Agent 6)
//! - Backtesting framework
//! - Strategy orchestration

// Core types
mod base;
mod config;

// Simple registry for MCP tool compatibility
pub mod registry;

// Strategies
pub mod momentum;
pub mod mean_reversion;
pub mod pairs;
pub mod enhanced_momentum;
pub mod neural_trend;
pub mod neural_sentiment;
pub mod neural_arbitrage;
pub mod mirror;
pub mod ensemble;
pub mod pattern_matcher;

// Integration layers
pub mod integration;
pub mod backtest;
pub mod orchestrator;

// Re-exports
pub use base::*;
pub use config::*;
pub use registry::{StrategyRegistry, StrategyMetadataSimple};
pub use integration::{
    BrokerClient, StrategyExecutor, ExecutionResult,
    NeuralPredictor, PricePrediction, MarketRegime,
    RiskManager, ValidationResult, RiskWarning,
};
pub use backtest::{BacktestEngine, BacktestResult, SlippageModel, PerformanceMetrics};
pub use orchestrator::{StrategyOrchestrator, AllocationMode, StrategyPerformance};

// Common dependencies
pub use async_trait::async_trait;
pub use chrono;
pub use rust_decimal::Decimal;
pub use serde::{Serialize, Deserialize};
pub use thiserror::Error;
pub use tracing;

// Re-export from other crates
pub use nt_core::types::{Symbol, Bar};
pub use nt_execution::{OrderRequest, OrderResponse, OrderType, OrderSide, TimeInForce};

// MarketData and Portfolio types defined locally until core integration

/// Market data container for strategies
#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub bars: Vec<Bar>,
    pub price: Option<Decimal>,
    pub volume: Option<Decimal>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MarketData {
    pub fn new(symbol: String, bars: Vec<Bar>) -> Self {
        let price = bars.last().map(|b| b.close);
        let volume = bars.last().map(|b| b.volume);
        Self {
            symbol,
            bars,
            price,
            volume,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn last(&self) -> Option<&Bar> {
        self.bars.last()
    }
}

/// Portfolio state for strategy processing
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub cash: Decimal,
    pub positions: std::collections::HashMap<String, Position>,
    pub total_value: Decimal,
}

impl Portfolio {
    pub fn new(cash: Decimal) -> Self {
        Self {
            cash,
            positions: std::collections::HashMap::new(),
            total_value: cash,
        }
    }

    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    pub fn positions(&self) -> &std::collections::HashMap<String, Position> {
        &self.positions
    }

    pub fn total_value(&self) -> Decimal {
        self.total_value
    }

    pub fn cash(&self) -> Decimal {
        self.cash
    }

    pub fn update_cash(&mut self, amount: Decimal) {
        self.cash += amount;
        self.recalculate_total_value();
    }

    pub fn update_position(&mut self, symbol: String, position: Position) {
        self.positions.insert(symbol, position);
        self.recalculate_total_value();
    }

    pub fn update_position_price(&mut self, symbol: &str, new_price: Decimal) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.current_price = new_price;
            pos.market_value = new_price * Decimal::from(pos.quantity.abs());
            pos.unrealized_pnl = pos.market_value - pos.avg_price * Decimal::from(pos.quantity.abs());
        }
        self.recalculate_total_value();
    }

    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    fn recalculate_total_value(&mut self) {
        let positions_value: Decimal = self.positions.values()
            .map(|p| p.market_value)
            .sum();
        self.total_value = self.cash + positions_value;
    }
}

/// Position in portfolio
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub avg_price: Decimal,
    pub current_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pnl: Decimal,
}

/// Strategy result type
pub type Result<T> = std::result::Result<T, StrategyError>;

/// Strategy error types
#[derive(Debug, Error)]
pub enum StrategyError {
    #[error("Insufficient data: needed {needed}, available {available}")]
    InsufficientData { needed: usize, available: usize },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Trading direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
    Close,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Long => write!(f, "LONG"),
            Direction::Short => write!(f, "SHORT"),
            Direction::Close => write!(f, "CLOSE"),
        }
    }
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub strategy_id: String,
    pub symbol: String,
    pub direction: Direction,
    pub confidence: Option<f64>,
    pub quantity: Option<u32>,
    pub entry_price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub reasoning: Option<String>,
    pub features: Vec<f64>,
}

impl Signal {
    pub fn new(strategy_id: String, symbol: String, direction: Direction) -> Self {
        Self {
            strategy_id,
            symbol,
            direction,
            confidence: None,
            quantity: None,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
            reasoning: None,
            features: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_quantity(mut self, quantity: u32) -> Self {
        self.quantity = Some(quantity);
        self
    }

    pub fn with_entry_price(mut self, price: Decimal) -> Self {
        self.entry_price = Some(price);
        self
    }

    pub fn with_stop_loss(mut self, price: Decimal) -> Self {
        self.stop_loss = Some(price);
        self
    }

    pub fn with_take_profit(mut self, price: Decimal) -> Self {
        self.take_profit = Some(price);
        self
    }

    pub fn with_reasoning(mut self, reasoning: String) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = features;
        self
    }
}

/// Strategy trait
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Get strategy ID
    fn id(&self) -> &str;

    /// Get strategy metadata
    fn metadata(&self) -> StrategyMetadata;

    /// Process market data and generate signals
    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>>;

    /// Validate strategy configuration
    fn validate_config(&self) -> Result<()>;

    /// Get risk parameters
    fn risk_parameters(&self) -> RiskParameters;
}

/// Strategy metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadata {
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub tags: Vec<String>,
    pub min_capital: Decimal,
    pub max_drawdown_threshold: f64,
}

/// Risk parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    pub max_position_size: Decimal,
    pub max_leverage: f64,
    pub stop_loss_percentage: f64,
    pub take_profit_percentage: f64,
    pub max_daily_loss: Decimal,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size: Decimal::from(10000),
            max_leverage: 2.0,
            stop_loss_percentage: 0.02,
            take_profit_percentage: 0.05,
            max_daily_loss: Decimal::from(5000),
        }
    }
}
