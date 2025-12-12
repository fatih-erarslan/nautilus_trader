//! # PADS Unified Types
//!
//! This module contains all the type definitions used throughout the PADS system.
//! These types maintain 100% compatibility with the original Python PADS implementation
//! while providing enhanced type safety and performance.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use std::fmt;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Market phase enum (from Python PADS MarketPhase)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketPhase {
    /// Growth phase - expanding markets
    Growth,
    /// Conservation phase - mature, stable markets
    Conservation,
    /// Release phase - market corrections or crashes
    Release,
    /// Reorganization phase - recovery and rebuilding
    Reorganization,
    /// Unknown phase - fallback state
    Unknown,
}

impl MarketPhase {
    /// Create MarketPhase from string (matches Python from_string method)
    pub fn from_string(phase_str: &str) -> Self {
        match phase_str.to_lowercase().as_str() {
            "growth" => Self::Growth,
            "conservation" => Self::Conservation,
            "release" => Self::Release,
            "reorganization" => Self::Reorganization,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for MarketPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Growth => write!(f, "growth"),
            Self::Conservation => write!(f, "conservation"),
            Self::Release => write!(f, "release"),
            Self::Reorganization => write!(f, "reorganization"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Decision type enum (from Python PADS DecisionType)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionType {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold position
    Hold,
    /// Exit position completely
    Exit,
    /// Hedge position
    Hedge,
    /// Increase position size
    Increase,
    /// Decrease position size
    Decrease,
}

impl fmt::Display for DecisionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buy => write!(f, "BUY"),
            Self::Sell => write!(f, "SELL"),
            Self::Hold => write!(f, "HOLD"),
            Self::Exit => write!(f, "EXIT"),
            Self::Hedge => write!(f, "HEDGE"),
            Self::Increase => write!(f, "INCREASE"),
            Self::Decrease => write!(f, "DECREASE"),
        }
    }
}

/// Trading decision structure (from Python PADS TradingDecision)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    /// Type of decision
    pub decision_type: DecisionType,
    /// Action string (for lib.rs compatibility)
    pub action: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning behind the decision
    pub reasoning: String,
    /// Timestamp of decision
    pub timestamp: DateTime<Utc>,
    /// Additional parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Metadata for execution info
    pub metadata: HashMap<String, serde_json::Value>,
    /// Unique decision identifier
    pub id: String,
}

impl TradingDecision {
    /// Create a new trading decision
    pub fn new(
        decision_type: DecisionType,
        confidence: f64,
        reasoning: String,
    ) -> Self {
        Self {
            decision_type,
            action: format!("{}", decision_type),
            confidence,
            reasoning,
            timestamp: Utc::now(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            id: Uuid::new_v4().to_string(),
        }
    }

    /// Add parameter to decision
    pub fn with_parameter(mut self, key: String, value: serde_json::Value) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Add metadata to decision
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for TradingDecision {
    fn default() -> Self {
        Self::new(
            DecisionType::Hold,
            0.5,
            "Default decision".to_string(),
        )
    }
}

/// Market state structure (enhanced version from Python PADS)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    /// Current price
    pub price: f64,
    /// Trading volume
    pub volume: f64,
    /// Volatility measure
    pub volatility: f64,
    /// Trend direction and strength
    pub trend: f64,
    /// Momentum indicator
    pub momentum: f64,
    /// Current market phase
    pub phase: MarketPhase,
    /// Market regime (normal, stress, etc.)
    pub regime: String,
    /// Self-Organized Criticality index
    pub soc_index: f64,
    /// Black swan risk assessment
    pub black_swan_risk: f64,
    /// Multi-scale phases
    pub micro_phase: MarketPhase,
    pub meso_phase: MarketPhase,
    pub macro_phase: MarketPhase,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Currency pair
    pub pair: String,
    /// Additional market data
    pub additional_data: HashMap<String, f64>,
}

impl MarketState {
    /// Create new market state with basic parameters
    pub fn new(price: f64, volume: f64, volatility: f64) -> Self {
        Self {
            price,
            volume,
            volatility,
            trend: 0.0,
            momentum: 0.0,
            phase: MarketPhase::Conservation,
            regime: "normal".to_string(),
            soc_index: 0.5,
            black_swan_risk: 0.1,
            micro_phase: MarketPhase::Growth,
            meso_phase: MarketPhase::Growth,
            macro_phase: MarketPhase::Growth,
            timestamp: Utc::now(),
            pair: "UNKNOWN".to_string(),
            additional_data: HashMap::new(),
        }
    }

    /// Update market data
    pub fn update(&mut self, price: f64, volume: f64, volatility: f64) {
        self.price = price;
        self.volume = volume;
        self.volatility = volatility;
        self.timestamp = Utc::now();
    }

    /// Get additional data value
    pub fn get_additional(&self, key: &str) -> Option<f64> {
        self.additional_data.get(key).copied()
    }

    /// Set additional data value
    pub fn set_additional(&mut self, key: String, value: f64) {
        self.additional_data.insert(key, value);
    }
}

/// Position state structure (from Python PADS usage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionState {
    /// Whether position is open
    pub position_open: bool,
    /// Position direction (1 for long, -1 for short, 0 for none)
    pub position_direction: i32,
    /// Position size
    pub position_size: f64,
    /// Current profit/loss
    pub current_profit: f64,
    /// Entry price
    pub entry_price: f64,
    /// Stop loss level
    pub stop_loss: Option<f64>,
    /// Take profit level
    pub take_profit: Option<f64>,
    /// Position duration
    pub duration: chrono::Duration,
    /// Additional position data
    pub additional_data: HashMap<String, f64>,
}

impl PositionState {
    /// Create new empty position state
    pub fn new() -> Self {
        Self {
            position_open: false,
            position_direction: 0,
            position_size: 0.0,
            current_profit: 0.0,
            entry_price: 0.0,
            stop_loss: None,
            take_profit: None,
            duration: chrono::Duration::zero(),
            additional_data: HashMap::new(),
        }
    }

    /// Open a new position
    pub fn open_position(&mut self, direction: i32, size: f64, entry_price: f64) {
        self.position_open = true;
        self.position_direction = direction;
        self.position_size = size;
        self.entry_price = entry_price;
        self.current_profit = 0.0;
    }

    /// Close the position
    pub fn close_position(&mut self) {
        self.position_open = false;
        self.position_direction = 0;
        self.position_size = 0.0;
        self.current_profit = 0.0;
    }

    /// Update current profit
    pub fn update_profit(&mut self, current_price: f64) {
        if self.position_open {
            self.current_profit = (current_price - self.entry_price) 
                * self.position_direction as f64 
                * self.position_size;
        }
    }
}

impl Default for PositionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Factor values structure (from Python PADS factor_values usage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorValues {
    /// Factor value map
    pub values: HashMap<String, f64>,
}

impl FactorValues {
    /// Create new factor values
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Get factor value
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }

    /// Set factor value
    pub fn set(&mut self, key: String, value: f64) {
        self.values.insert(key, value);
    }

    /// Get factor value with default
    pub fn get_or_default(&self, key: &str, default: f64) -> f64 {
        self.values.get(key).copied().unwrap_or(default)
    }
}

impl Default for FactorValues {
    fn default() -> Self {
        Self::new()
    }
}

/// Board member structure (from Python PADS board_members)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardMember {
    /// Member name/identifier
    pub name: String,
    /// Voting weight (0.0 to 1.0)
    pub weight: f64,
    /// Reputation score (0.0 to 1.0)
    pub reputation: f64,
    /// Member capabilities
    pub capabilities: Vec<String>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl BoardMember {
    /// Create new board member
    pub fn new(name: String, weight: f64) -> Self {
        Self {
            name,
            weight,
            reputation: 0.5,
            capabilities: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    /// Update reputation score
    pub fn update_reputation(&mut self, score: f64) {
        self.reputation = score.clamp(0.0, 1.0);
    }

    /// Add capability
    pub fn add_capability(&mut self, capability: String) {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
    }
}

/// Component vote structure (from Python PADS component_votes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentVote {
    /// Vote value (-1.0 to 1.0)
    pub vote_value: Option<f64>,
    /// Confidence in vote (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning behind vote
    pub reasoning: String,
    /// Raw decision/prediction data
    pub raw_data: HashMap<String, serde_json::Value>,
    /// Error message if vote failed
    pub error: Option<String>,
}

impl ComponentVote {
    /// Create new component vote
    pub fn new(vote_value: Option<f64>, confidence: f64, reasoning: String) -> Self {
        Self {
            vote_value,
            confidence,
            reasoning,
            raw_data: HashMap::new(),
            error: None,
        }
    }

    /// Create error vote
    pub fn error(error: String) -> Self {
        Self {
            vote_value: None,
            confidence: 0.0,
            reasoning: String::new(),
            raw_data: HashMap::new(),
            error: Some(error),
        }
    }

    /// Check if vote is valid
    pub fn is_valid(&self) -> bool {
        self.vote_value.is_some() && self.error.is_none()
    }
}

/// Board state structure (from Python PADS board_state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardState {
    /// Agreement level among components
    pub consensus_level: f64,
    /// Strength of belief in decision
    pub conviction_level: f64,
    /// Current risk appetite
    pub risk_appetite: f64,
    /// Perceived opportunity score
    pub opportunity_score: f64,
    /// Votes accumulated
    pub voting_quorum: f64,
    /// Level of disagreement
    pub dissent_level: f64,
    /// Current decision strategy
    pub current_strategy: String,
    /// Information value from board
    pub information_value: f64,
}

impl BoardState {
    /// Create new board state
    pub fn new() -> Self {
        Self {
            consensus_level: 0.5,
            conviction_level: 0.5,
            risk_appetite: 0.5,
            opportunity_score: 0.5,
            voting_quorum: 0.0,
            dissent_level: 0.0,
            current_strategy: "balanced".to_string(),
            information_value: 0.0,
        }
    }
}

impl Default for BoardState {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics structure (from Python PADS performance_metrics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of decisions made
    pub total_decisions: u64,
    /// Number of successful decisions
    pub successful_decisions: u64,
    /// Risk-adjusted return
    pub risk_adjusted_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate percentage
    pub win_rate: f64,
    /// Decisions by phase
    pub decisions_by_phase: HashMap<String, u64>,
    /// Average decision latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Quantum advantage metric
    pub quantum_advantage: f64,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            total_decisions: 0,
            successful_decisions: 0,
            risk_adjusted_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            decisions_by_phase: HashMap::new(),
            avg_latency_ns: 0,
            quantum_advantage: 1.0,
        }
    }

    /// Update metrics with new decision
    pub fn update_decision(&mut self, decision: &TradingDecision, success: bool, latency_ns: u64) {
        self.total_decisions += 1;
        if success {
            self.successful_decisions += 1;
        }
        
        // Update win rate
        self.win_rate = (self.successful_decisions as f64) / (self.total_decisions as f64) * 100.0;
        
        // Update rolling average latency
        let current_avg = self.avg_latency_ns;
        self.avg_latency_ns = (current_avg * 99 + latency_ns) / 100;
        
        // Update decisions by phase if metadata contains phase info
        if let Some(phase) = decision.metadata.get("PADS_phase") {
            if let Some(phase_str) = phase.as_str() {
                *self.decisions_by_phase.entry(phase_str.to_string()).or_insert(0) += 1;
            }
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk advice structure (from Python PADS get_risk_advice)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAdvice {
    /// Stop loss adjustment factor
    pub stoploss_adjustment: f64,
    /// Position sizing adjustment factor
    pub position_sizing: f64,
    /// Take profit level
    pub take_profit: Option<f64>,
    /// Confidence in advice
    pub confidence: f64,
    /// Reasoning behind advice
    pub reasons: Vec<String>,
}

impl RiskAdvice {
    /// Create new risk advice
    pub fn new() -> Self {
        Self {
            stoploss_adjustment: 1.0,
            position_sizing: 1.0,
            take_profit: None,
            confidence: 0.5,
            reasons: Vec::new(),
        }
    }

    /// Add reason to advice
    pub fn add_reason(&mut self, reason: String) {
        self.reasons.push(reason);
    }

    /// Apply conservative adjustments
    pub fn apply_conservative(&mut self, factor: f64) {
        self.stoploss_adjustment *= factor;
        self.position_sizing *= factor;
        self.add_reason(format!("Applied conservative factor: {:.2}", factor));
    }

    /// Apply aggressive adjustments
    pub fn apply_aggressive(&mut self, factor: f64) {
        self.stoploss_adjustment *= (1.0 + factor);
        self.position_sizing *= (1.0 + factor);
        self.add_reason(format!("Applied aggressive factor: {:.2}", factor));
    }
}

impl Default for RiskAdvice {
    fn default() -> Self {
        Self::new()
    }
}

/// System summary structure (from Python PADS create_system_summary)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    /// Timestamp of summary
    pub timestamp: DateTime<Utc>,
    /// Latest decision information
    pub latest_decision: Option<TradingDecision>,
    /// Market regime information
    pub regime_info: HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Board member information
    pub board_members: HashMap<String, BoardMember>,
}

impl SystemSummary {
    /// Create new system summary
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            latest_decision: None,
            regime_info: HashMap::new(),
            performance_metrics: PerformanceMetrics::new(),
            board_members: HashMap::new(),
        }
    }
}

impl Default for SystemSummary {
    fn default() -> Self {
        Self::new()
    }
}

/// PADS decision structure (main decision type)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsDecision {
    /// Trading decision
    pub trading_decision: TradingDecision,
    /// Market analysis
    pub market_analysis: MarketAnalysis,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    /// Agent contributions
    pub agent_contributions: HashMap<String, f64>,
    /// Decision metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PadsDecision {
    /// Create new PADS decision
    pub fn new(trading_decision: TradingDecision) -> Self {
        Self {
            trading_decision,
            market_analysis: MarketAnalysis::default(),
            risk_assessment: RiskAssessment::default(),
            agent_contributions: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Market analysis structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketAnalysis {
    /// Current market phase
    pub phase: MarketPhase,
    /// Trend strength
    pub trend_strength: f64,
    /// Volatility assessment
    pub volatility_assessment: f64,
    /// Pattern recognition results
    pub patterns: Vec<String>,
    /// Confidence in analysis
    pub confidence: f64,
}

impl Default for MarketAnalysis {
    fn default() -> Self {
        Self {
            phase: MarketPhase::Unknown,
            trend_strength: 0.0,
            volatility_assessment: 0.0,
            patterns: Vec::new(),
            confidence: 0.0,
        }
    }
}

/// Risk assessment structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Risk score (0.0 to 1.0)
    pub risk_score: f64,
    /// Confidence in assessment
    pub confidence: f64,
    /// Risk factors identified
    pub risk_factors: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            risk_score: 0.5,
            confidence: 0.0,
            risk_factors: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

/// Panarchy state structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyState {
    /// Current phase
    pub current_phase: String,
    /// Resilience score
    pub resilience_score: f64,
    /// Potential score
    pub potential_score: f64,
    /// Connectedness score
    pub connectedness_score: f64,
    /// Phase transition probability
    pub transition_probability: f64,
}

impl Default for PanarchyState {
    fn default() -> Self {
        Self {
            current_phase: "growth".to_string(),
            resilience_score: 0.5,
            potential_score: 0.5,
            connectedness_score: 0.5,
            transition_probability: 0.1,
        }
    }
}

/// Main system summary type that matches lib.rs usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummaryMain {
    /// System version
    pub version: String,
    /// Decision count
    pub decision_count: usize,
    /// Average decision latency
    pub avg_decision_latency: Duration,
    /// Agent count
    pub agent_count: usize,
    /// Active features
    pub active_features: Vec<String>,
    /// System health
    pub system_health: f64,
    /// Last update
    pub last_update: SystemTime,
}

impl Default for SystemSummaryMain {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            decision_count: 0,
            avg_decision_latency: Duration::from_nanos(0),
            agent_count: 12,
            active_features: vec!["core".to_string()],
            system_health: 1.0,
            last_update: SystemTime::now(),
        }
    }
}

/// Board decision structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardDecision {
    /// Decision confidence
    pub confidence: f64,
    /// Decision type
    pub decision_type: DecisionType,
    /// Board consensus level
    pub consensus_level: f64,
    /// Individual member votes
    pub member_votes: HashMap<String, ComponentVote>,
    /// Decision rationale
    pub rationale: String,
}

impl Default for BoardDecision {
    fn default() -> Self {
        Self {
            confidence: 0.5,
            decision_type: DecisionType::Hold,
            consensus_level: 0.0,
            member_votes: HashMap::new(),
            rationale: String::new(),
        }
    }
}

/// Market regime structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegime {
    /// Regime type
    pub regime_type: String,
    /// Regime confidence
    pub confidence: f64,
    /// Regime characteristics
    pub characteristics: Vec<String>,
    /// Regime duration
    pub duration: Duration,
}

impl Default for MarketRegime {
    fn default() -> Self {
        Self {
            regime_type: "normal".to_string(),
            confidence: 0.5,
            characteristics: vec!["stable".to_string()],
            duration: Duration::from_secs(0),
        }
    }
}

/// Risk analysis structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAnalysis {
    /// Overall risk score
    pub risk_score: f64,
    /// Risk factors
    pub risk_factors: Vec<String>,
    /// Risk recommendations
    pub recommendations: Vec<String>,
    /// Confidence in analysis
    pub confidence: f64,
}

impl Default for RiskAnalysis {
    fn default() -> Self {
        Self {
            risk_score: 0.5,
            risk_factors: Vec::new(),
            recommendations: Vec::new(),
            confidence: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_phase_from_string() {
        assert_eq!(MarketPhase::from_string("growth"), MarketPhase::Growth);
        assert_eq!(MarketPhase::from_string("CONSERVATION"), MarketPhase::Conservation);
        assert_eq!(MarketPhase::from_string("invalid"), MarketPhase::Unknown);
    }

    #[test]
    fn test_trading_decision_creation() {
        let decision = TradingDecision::new(
            DecisionType::Buy,
            0.8,
            "Test decision".to_string(),
        );
        
        assert_eq!(decision.decision_type, DecisionType::Buy);
        assert_eq!(decision.confidence, 0.8);
        assert_eq!(decision.reasoning, "Test decision");
    }

    #[test]
    fn test_market_state_creation() {
        let market = MarketState::new(100.0, 1000.0, 0.2);
        
        assert_eq!(market.price, 100.0);
        assert_eq!(market.volume, 1000.0);
        assert_eq!(market.volatility, 0.2);
    }

    #[test]
    fn test_position_state_operations() {
        let mut position = PositionState::new();
        assert!(!position.position_open);
        
        position.open_position(1, 100.0, 50.0);
        assert!(position.position_open);
        assert_eq!(position.position_direction, 1);
        assert_eq!(position.position_size, 100.0);
        
        position.close_position();
        assert!(!position.position_open);
    }

    #[test]
    fn test_factor_values_operations() {
        let mut factors = FactorValues::new();
        
        factors.set("test_factor".to_string(), 0.5);
        assert_eq!(factors.get("test_factor"), Some(0.5));
        assert_eq!(factors.get_or_default("missing_factor", 0.0), 0.0);
    }

    #[test]
    fn test_component_vote_validity() {
        let valid_vote = ComponentVote::new(Some(0.5), 0.8, "Valid vote".to_string());
        assert!(valid_vote.is_valid());
        
        let error_vote = ComponentVote::error("Test error".to_string());
        assert!(!error_vote.is_valid());
    }

    #[test]
    fn test_performance_metrics_update() {
        let mut metrics = PerformanceMetrics::new();
        let decision = TradingDecision::new(
            DecisionType::Buy,
            0.8,
            "Test decision".to_string(),
        );
        
        metrics.update_decision(&decision, true, 5000);
        assert_eq!(metrics.total_decisions, 1);
        assert_eq!(metrics.successful_decisions, 1);
        assert_eq!(metrics.win_rate, 100.0);
        assert_eq!(metrics.avg_latency_ns, 5000);
    }

    #[test]
    fn test_risk_advice_operations() {
        let mut advice = RiskAdvice::new();
        
        advice.apply_conservative(0.8);
        assert_eq!(advice.stoploss_adjustment, 0.8);
        assert_eq!(advice.position_sizing, 0.8);
        assert_eq!(advice.reasons.len(), 1);
        
        advice.apply_aggressive(0.2);
        assert_eq!(advice.stoploss_adjustment, 0.96); // 0.8 * 1.2
        assert_eq!(advice.position_sizing, 0.96);
        assert_eq!(advice.reasons.len(), 2);
    }
}