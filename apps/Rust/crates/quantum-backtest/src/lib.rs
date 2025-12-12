//! Quantum-Enhanced Backtesting Engine
//! 
//! A comprehensive backtesting framework that integrates quantum pattern recognition
//! with traditional trading strategies for cryptocurrency markets.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod engine;
pub mod strategy;
pub mod portfolio;
pub mod metrics;
pub mod data;
pub mod quantum;

pub use engine::BacktestEngine;
pub use strategy::TengriQuantumStrategy;
pub use portfolio::{Portfolio, Position, Trade};
pub use metrics::{PerformanceMetrics, RiskMetrics, DrawdownMetrics};

/// Backtesting error types
#[derive(Error, Debug)]
pub enum BacktestError {
    #[error("Data error: {0}")]
    Data(String),
    
    #[error("Strategy error: {0}")]
    Strategy(String),
    
    #[error("Portfolio error: {0}")]
    Portfolio(String),
    
    #[error("Quantum computation error: {0}")]
    Quantum(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, BacktestError>;

/// Backtesting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Start date for backtesting
    pub start_date: DateTime<Utc>,
    /// End date for backtesting
    pub end_date: DateTime<Utc>,
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (as decimal, e.g., 0.001 for 0.1%)
    pub commission_rate: f64,
    /// Slippage rate (as decimal)
    pub slippage_rate: f64,
    /// Assets to trade
    pub assets: Vec<String>,
    /// Data timeframe (e.g., "1h", "4h", "1d")
    pub timeframe: String,
    /// Enable quantum pattern recognition
    pub enable_quantum: bool,
    /// Risk management settings
    pub risk_management: RiskConfig,
    /// Quantum pattern thresholds
    pub quantum_thresholds: QuantumThresholds,
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size as percentage of portfolio
    pub max_position_size: f64,
    /// Stop loss percentage
    pub stop_loss: f64,
    /// Take profit percentage
    pub take_profit: f64,
    /// Maximum drawdown before stopping
    pub max_drawdown: f64,
    /// Maximum number of concurrent positions
    pub max_positions: usize,
}

/// Quantum pattern recognition thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumThresholds {
    /// Minimum coherence time for quantum patterns
    pub min_coherence_time: f64,
    /// Entanglement correlation threshold
    pub entanglement_threshold: f64,
    /// Superposition confidence threshold
    pub superposition_threshold: f64,
    /// Fourier pattern strength threshold
    pub fourier_threshold: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        let end_date = Utc::now();
        let start_date = end_date - chrono::Duration::days(5 * 365); // 5 years
        
        Self {
            start_date,
            end_date,
            initial_capital: 100_000.0,
            commission_rate: 0.001, // 0.1%
            slippage_rate: 0.0005,  // 0.05%
            assets: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "ADAUSDT".to_string(),
                "SOLUSDT".to_string(),
            ],
            timeframe: "1h".to_string(),
            enable_quantum: true,
            risk_management: RiskConfig {
                max_position_size: 0.25, // 25% max per position
                stop_loss: 0.05,         // 5% stop loss
                take_profit: 0.15,       // 15% take profit
                max_drawdown: 0.20,      // 20% max drawdown
                max_positions: 4,
            },
            quantum_thresholds: QuantumThresholds {
                min_coherence_time: 100.0,
                entanglement_threshold: 0.7,
                superposition_threshold: 0.8,
                fourier_threshold: 0.75,
            },
        }
    }
}

/// Market data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
}

/// Signal types from quantum analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    /// Strong buy signal
    StrongBuy,
    /// Weak buy signal
    Buy,
    /// Hold position
    Hold,
    /// Weak sell signal
    Sell,
    /// Strong sell signal
    StrongSell,
}

/// Trading signal with quantum confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub signal_type: SignalType,
    pub confidence: f64,
    pub quantum_patterns: Vec<String>,
    pub price_target: Option<f64>,
    pub stop_loss: Option<f64>,
}

/// Backtest result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub config: BacktestConfig,
    pub performance: PerformanceMetrics,
    pub risk_metrics: RiskMetrics,
    pub drawdown_metrics: DrawdownMetrics,
    pub trades: Vec<Trade>,
    pub quantum_signals: Vec<TradingSignal>,
    pub portfolio_value_history: Vec<(DateTime<Utc>, f64)>,
    pub execution_time_ms: u128,
}

/// Initialize the backtesting system
pub async fn init() -> Result<()> {
    tracing::info!("Initializing Quantum-Enhanced Backtesting Engine");
    
    // Initialize quantum pattern engine
    quantum::init().await?;
    
    tracing::info!("Backtesting engine initialized successfully");
    Ok(())
}