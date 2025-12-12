//! Core types and data structures for Quantum Agentic Reasoning
//!
//! This module defines the fundamental types, enums, and data structures
//! used throughout the QAR system.

use std::collections::HashMap;
use std::fmt;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod types;
pub mod traits;
pub mod constants;

pub use types::*;
pub use traits::*;
pub use constants::*;

// Re-export our own error types
pub use crate::error::{QarError, QarResult};

// Re-export quantum types from quantum-core for convenience  
pub use quantum_core::{
    QuantumState as CoreQuantumState, 
    QuantumGate as Gate, 
    QuantumError, 
    HardwareConfig,
    ComplexAmplitude,
};

// Import types from specific modules to avoid conflicts
pub use quantum_core::traits::{
    QuantumCircuit as CoreQuantumCircuit,
    HardwareInterface as CoreHardwareInterface,
};

// Re-export the local types to avoid conflicts
pub use crate::core::types::{
    CircuitParams,
    ExecutionContext,
    PatternMatch,
    HardwareMetrics,
    DecisionOptimization,
    DecisionMetrics,
    DecisionOutcome,
    PatternData,
    RegimeAnalysis,
};

// Trait definition for DecisionEngine
#[async_trait::async_trait]
pub trait DecisionEngine {
    async fn make_decision(
        &self,
        factors: &FactorMap,
        context: &MarketContext,
    ) -> crate::error::QarResult<TradingDecision>;
    
    async fn update_with_feedback(
        &self,
        outcome: DecisionOutcome,
    ) -> crate::error::QarResult<()>;
    
    fn confidence_threshold(&self) -> f64;
    
    async fn set_confidence_threshold(&self, threshold: f64) -> crate::error::QarResult<()>;
    
    fn get_metrics(&self) -> DecisionMetrics;
}

// Use quantum-core's QuantumResult (avoid name conflict with lib.rs)
pub use quantum_core::QuantumResult;

/// Standard factors used in quantum agentic reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StandardFactors {
    /// Market trend strength and direction
    Trend,
    /// Market volatility and uncertainty
    Volatility,
    /// Price momentum indicators
    Momentum,
    /// Market sentiment and psychology
    Sentiment,
    /// Market liquidity conditions
    Liquidity,
    /// Asset correlation patterns
    Correlation,
    /// Market cycle position
    Cycle,
    /// Anomaly detection signals
    Anomaly,
}

impl StandardFactors {
    /// Get all standard factors as a vector
    pub fn all() -> Vec<StandardFactors> {
        vec![
            StandardFactors::Trend,
            StandardFactors::Volatility,
            StandardFactors::Momentum,
            StandardFactors::Sentiment,
            StandardFactors::Liquidity,
            StandardFactors::Correlation,
            StandardFactors::Cycle,
            StandardFactors::Anomaly,
        ]
    }

    /// Get the string representation of the factor
    pub fn as_str(&self) -> &'static str {
        match self {
            StandardFactors::Trend => "trend",
            StandardFactors::Volatility => "volatility",
            StandardFactors::Momentum => "momentum",
            StandardFactors::Sentiment => "sentiment",
            StandardFactors::Liquidity => "liquidity",
            StandardFactors::Correlation => "correlation",
            StandardFactors::Cycle => "cycle",
            StandardFactors::Anomaly => "anomaly",
        }
    }

    /// Parse a string into a StandardFactors enum
    pub fn from_str(s: &str) -> Result<Self, QarError> {
        match s.to_lowercase().as_str() {
            "trend" => Ok(StandardFactors::Trend),
            "volatility" => Ok(StandardFactors::Volatility),
            "momentum" => Ok(StandardFactors::Momentum),
            "sentiment" => Ok(StandardFactors::Sentiment),
            "liquidity" => Ok(StandardFactors::Liquidity),
            "correlation" => Ok(StandardFactors::Correlation),
            "cycle" => Ok(StandardFactors::Cycle),
            "anomaly" => Ok(StandardFactors::Anomaly),
            _ => Err(QarError::InvalidFactor(s.to_string())),
        }
    }
}

impl fmt::Display for StandardFactors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Types of trading decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionType {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold position
    Hold,
    /// Exit position
    Exit,
    /// Hedge position
    Hedge,
    /// Increase position size
    Increase,
    /// Decrease position size
    Decrease,
}

impl DecisionType {
    /// Check if this is an actionable decision (not Hold)
    pub fn is_actionable(&self) -> bool {
        !matches!(self, DecisionType::Hold)
    }

    /// Check if this is a buy-side decision
    pub fn is_buy_side(&self) -> bool {
        matches!(self, DecisionType::Buy | DecisionType::Increase)
    }

    /// Check if this is a sell-side decision
    pub fn is_sell_side(&self) -> bool {
        matches!(self, DecisionType::Sell | DecisionType::Exit | DecisionType::Decrease)
    }
}

impl fmt::Display for DecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecisionType::Buy => write!(f, "BUY"),
            DecisionType::Sell => write!(f, "SELL"),
            DecisionType::Hold => write!(f, "HOLD"),
            DecisionType::Exit => write!(f, "EXIT"),
            DecisionType::Hedge => write!(f, "HEDGE"),
            DecisionType::Increase => write!(f, "INCREASE"),
            DecisionType::Decrease => write!(f, "DECREASE"),
        }
    }
}

/// Market phases based on Panarchy theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketPhase {
    /// Growth phase - increasing complexity and rigidity
    Growth,
    /// Conservation phase - stability and efficiency
    Conservation,
    /// Release phase - creative destruction and collapse
    Release,
    /// Reorganization phase - innovation and renewal
    Reorganization,
    /// Unknown phase - uncertain or transitional
    Unknown,
}

impl MarketPhase {
    /// Get the next phase in the cycle
    pub fn next(&self) -> Self {
        match self {
            MarketPhase::Growth => MarketPhase::Conservation,
            MarketPhase::Conservation => MarketPhase::Release,
            MarketPhase::Release => MarketPhase::Reorganization,
            MarketPhase::Reorganization => MarketPhase::Growth,
            MarketPhase::Unknown => MarketPhase::Unknown,
        }
    }

    /// Get the previous phase in the cycle
    pub fn previous(&self) -> Self {
        match self {
            MarketPhase::Growth => MarketPhase::Reorganization,
            MarketPhase::Conservation => MarketPhase::Growth,
            MarketPhase::Release => MarketPhase::Conservation,
            MarketPhase::Reorganization => MarketPhase::Release,
            MarketPhase::Unknown => MarketPhase::Unknown,
        }
    }

    /// Check if this is a stable phase
    pub fn is_stable(&self) -> bool {
        matches!(self, MarketPhase::Growth | MarketPhase::Conservation)
    }

    /// Check if this is a chaotic phase
    pub fn is_chaotic(&self) -> bool {
        matches!(self, MarketPhase::Release | MarketPhase::Reorganization)
    }
}

impl fmt::Display for MarketPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarketPhase::Growth => write!(f, "GROWTH"),
            MarketPhase::Conservation => write!(f, "CONSERVATION"),
            MarketPhase::Release => write!(f, "RELEASE"),
            MarketPhase::Reorganization => write!(f, "REORGANIZATION"),
            MarketPhase::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Trading decision with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    /// Unique identifier for this decision
    pub id: Uuid,
    /// Type of decision (Buy, Sell, Hold, etc.)
    pub decision_type: DecisionType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable reasoning for the decision
    pub reasoning: String,
    /// Timestamp when the decision was made
    pub timestamp: DateTime<Utc>,
    /// Decision parameters and factors
    pub parameters: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TradingDecision {
    /// Create a new trading decision
    pub fn new(
        decision_type: DecisionType,
        confidence: f64,
        reasoning: String,
        parameters: HashMap<String, f64>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            decision_type,
            confidence,
            reasoning,
            timestamp: Utc::now(),
            parameters,
            metadata: HashMap::new(),
        }
    }

    /// Check if this decision meets the confidence threshold
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Add metadata to the decision
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Map of factor names to their values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMap {
    factors: HashMap<String, f64>,
}

impl FactorMap {
    /// Create a new factor map
    pub fn new(factors: HashMap<String, f64>) -> Result<Self, QarError> {
        // Validate that all values are finite
        for (key, value) in &factors {
            if !value.is_finite() {
                return Err(QarError::InvalidFactorValue {
                    factor: key.clone(),
                    value: *value,
                });
            }
        }
        Ok(Self { factors })
    }

    /// Create a factor map from standard factors
    pub fn from_standard_factors(values: [f64; 8]) -> Result<Self, QarError> {
        let mut factors = HashMap::new();
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(QarError::InvalidFactorValue {
                    factor: format!("standard_factor_{}", i),
                    value,
                });
            }
            let factor = StandardFactors::all()[i];
            factors.insert(factor.as_str().to_string(), value);
        }
        Self::new(factors)
    }

    /// Get a factor value by name
    pub fn get(&self, name: &str) -> Option<f64> {
        self.factors.get(name).copied()
    }

    /// Get a factor value by StandardFactors enum
    pub fn get_standard(&self, factor: StandardFactors) -> Option<f64> {
        self.factors.get(factor.as_str()).copied()
    }

    /// Get all factors as a HashMap
    pub fn factors(&self) -> &HashMap<String, f64> {
        &self.factors
    }

    /// Get factors as a vector in standard order
    pub fn as_standard_vector(&self) -> Vec<f64> {
        StandardFactors::all()
            .iter()
            .map(|f| self.get_standard(*f).unwrap_or(0.5))
            .collect()
    }

    /// Number of factors
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Validate that all standard factors are present
    pub fn validate_standard_factors(&self) -> Result<(), QarError> {
        for factor in StandardFactors::all() {
            if !self.factors.contains_key(factor.as_str()) {
                return Err(QarError::MissingFactor(factor.as_str().to_string()));
            }
        }
        Ok(())
    }
}

/// Market context for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketContext {
    /// Current market phase
    pub phase: MarketPhase,
    /// Market volatility regime
    pub volatility_regime: f64,
    /// Market trend strength
    pub trend_strength: f64,
    /// Liquidity conditions
    pub liquidity: f64,
    /// Additional context data
    pub context: HashMap<String, f64>,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            phase: MarketPhase::Unknown,
            volatility_regime: 0.5,
            trend_strength: 0.5,
            liquidity: 0.5,
            context: HashMap::new(),
        }
    }
}

impl MarketContext {
    /// Create a new market context
    pub fn new(
        phase: MarketPhase,
        volatility_regime: f64,
        trend_strength: f64,
        liquidity: f64,
    ) -> Self {
        Self {
            phase,
            volatility_regime,
            trend_strength,
            liquidity,
            context: HashMap::new(),
        }
    }

    /// Add additional context data
    pub fn with_context(mut self, key: String, value: f64) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Get context value
    pub fn get_context(&self, key: &str) -> Option<f64> {
        self.context.get(key).copied()
    }
}

// QarError and QarResult are now re-exported from the error module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_factors_all() {
        let factors = StandardFactors::all();
        assert_eq!(factors.len(), 8);
        assert!(factors.contains(&StandardFactors::Trend));
        assert!(factors.contains(&StandardFactors::Volatility));
    }

    #[test]
    fn test_standard_factors_string_conversion() {
        assert_eq!(StandardFactors::Trend.as_str(), "trend");
        assert_eq!(StandardFactors::from_str("trend").unwrap(), StandardFactors::Trend);
        assert!(StandardFactors::from_str("invalid").is_err());
    }

    #[test]
    fn test_decision_type_properties() {
        assert!(DecisionType::Buy.is_actionable());
        assert!(DecisionType::Buy.is_buy_side());
        assert!(!DecisionType::Buy.is_sell_side());
        
        assert!(!DecisionType::Hold.is_actionable());
        assert!(!DecisionType::Hold.is_buy_side());
        assert!(!DecisionType::Hold.is_sell_side());
        
        assert!(DecisionType::Sell.is_actionable());
        assert!(!DecisionType::Sell.is_buy_side());
        assert!(DecisionType::Sell.is_sell_side());
    }

    #[test]
    fn test_market_phase_cycle() {
        assert_eq!(MarketPhase::Growth.next(), MarketPhase::Conservation);
        assert_eq!(MarketPhase::Conservation.next(), MarketPhase::Release);
        assert_eq!(MarketPhase::Release.next(), MarketPhase::Reorganization);
        assert_eq!(MarketPhase::Reorganization.next(), MarketPhase::Growth);
        
        assert_eq!(MarketPhase::Growth.previous(), MarketPhase::Reorganization);
        assert_eq!(MarketPhase::Conservation.previous(), MarketPhase::Growth);
    }

    #[test]
    fn test_factor_map_creation() {
        let mut factors = HashMap::new();
        factors.insert("trend".to_string(), 0.75);
        factors.insert("volatility".to_string(), 0.45);
        
        let factor_map = FactorMap::new(factors).unwrap();
        assert_eq!(factor_map.get("trend"), Some(0.75));
        assert_eq!(factor_map.get("volatility"), Some(0.45));
        assert_eq!(factor_map.get("momentum"), None);
    }

    #[test]
    fn test_factor_map_from_standard() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let factor_map = FactorMap::from_standard_factors(values).unwrap();
        
        assert_eq!(factor_map.get_standard(StandardFactors::Trend), Some(0.1));
        assert_eq!(factor_map.get_standard(StandardFactors::Volatility), Some(0.2));
        assert_eq!(factor_map.get_standard(StandardFactors::Anomaly), Some(0.8));
    }

    #[test]
    fn test_trading_decision_creation() {
        let mut params = HashMap::new();
        params.insert("confidence".to_string(), 0.85);
        
        let decision = TradingDecision::new(
            DecisionType::Buy,
            0.85,
            "Strong bullish trend detected".to_string(),
            params,
        );
        
        assert_eq!(decision.decision_type, DecisionType::Buy);
        assert_eq!(decision.confidence, 0.85);
        assert!(decision.meets_threshold(0.8));
        assert!(!decision.meets_threshold(0.9));
    }

    #[test]
    fn test_invalid_factor_values() {
        let mut factors = HashMap::new();
        factors.insert("trend".to_string(), f64::NAN);
        
        let result = FactorMap::new(factors);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QarError::InvalidFactorValue { .. }));
    }
}