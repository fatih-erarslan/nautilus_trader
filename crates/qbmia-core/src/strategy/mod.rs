//! Strategic analysis and decision-making modules

pub mod machiavellian;
pub mod robin_hood;
pub mod temporal_nash;
pub mod antifragile_coalition;

// Re-export main types
pub use machiavellian::{MachiavellianFramework, ManipulationDetectionResult};
pub use robin_hood::{RobinHoodProtocol, WealthDistributionAnalysis};
pub use temporal_nash::{TemporalBiologicalNash, TemporalEquilibrium};
pub use antifragile_coalition::{AntifragileCoalition, CoalitionAnalysis};

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common strategy types and utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyAction {
    Buy,
    Sell,
    Hold,
    Wait,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRecommendation {
    pub action: StrategyAction,
    pub confidence: f64,
    pub reasoning: String,
    pub tactics: Vec<String>,
}

/// Market manipulation patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub characteristics: HashMap<String, f64>,
}

/// Order flow data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    pub timestamp: f64,
    pub side: String, // "buy" or "sell"
    pub size: f64,
    pub price: f64,
    pub cancelled: bool,
}

/// Market conditions for strategy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub trend: f64,
    pub volume: f64,
    pub participant_count: usize,
    pub crisis_indicators: HashMap<String, f64>,
}

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            volatility: 0.0,
            trend: 0.0,
            volume: 0.0,
            participant_count: 0,
            crisis_indicators: HashMap::new(),
        }
    }
}