//! Integration modules connecting strategies to execution infrastructure
//!
//! This module provides the bridge between trading strategies and:
//! - Agent 3: Broker clients for order execution
//! - Agent 4: Neural models for predictions
//! - Agent 6: Risk management and position sizing

pub mod broker;
pub mod neural;
pub mod risk;

pub use broker::{StrategyExecutor, ExecutionResult, BrokerClient};
pub use neural::{NeuralPredictor, PricePrediction, VolatilityPrediction, MarketRegime, SentimentSignal};
pub use risk::{RiskManager, ValidationResult, RiskWarning, RiskLevel};
