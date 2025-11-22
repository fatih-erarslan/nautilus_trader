//! Decision context for prospect theory operations
//!
//! This module provides context structures that capture the complete environment
//! and state needed for prospect theory decision making.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{MarketData, Position, BehavioralFactors, FramingContext, Result};

/// Comprehensive decision context for prospect theory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Current market data and conditions
    pub market_data: MarketData,
    /// Current position (if any)
    pub position: Option<Position>,
    /// Behavioral factors affecting decision
    pub behavioral_factors: BehavioralFactors,
    /// Framing context for the decision
    pub framing_context: FramingContext,
    /// Historical context and patterns
    pub historical_context: HistoricalContext,
    /// Risk constraints and preferences
    pub risk_constraints: RiskConstraints,
    /// Decision metadata
    pub metadata: DecisionMetadata,
}

/// Historical context for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalContext {
    /// Recent trading decisions and outcomes
    pub recent_decisions: Vec<DecisionOutcome>,
    /// Performance metrics over time
    pub performance_metrics: PerformanceHistory,
    /// Learned patterns and insights
    pub learned_patterns: Vec<Pattern>,
    /// Market regime classification
    pub market_regime: MarketRegime,
}

/// Decision outcome tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Decision timestamp
    pub timestamp: u64,
    /// Action taken
    pub action: String,
    /// Expected value at decision time
    pub expected_value: f64,
    /// Actual outcome value
    pub actual_outcome: Option<f64>,
    /// Confidence level
    pub confidence: f64,
    /// Contextual factors
    pub context_factors: HashMap<String, f64>,
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Win rate over recent decisions
    pub win_rate: f64,
    /// Average return per trade
    pub average_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Consistency score
    pub consistency_score: f64,
}

/// Learned pattern representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern identifier
    pub id: String,
    /// Pattern description
    pub description: String,
    /// Confidence in pattern
    pub confidence: f64,
    /// Success rate when pattern is detected
    pub success_rate: f64,
    /// Pattern features
    pub features: Vec<f64>,
    /// Last occurrence timestamp
    pub last_seen: u64,
}

/// Market regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Trending upward market
    BullTrend,
    /// Trending downward market
    BearTrend,
    /// Sideways/ranging market
    Sideways,
    /// High volatility market
    HighVolatility,
    /// Low volatility market
    LowVolatility,
    /// Uncertain/transitional market
    Uncertain,
}

/// Risk constraints and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConstraints {
    /// Maximum position size as fraction of portfolio
    pub max_position_size: f64,
    /// Maximum risk per trade
    pub max_risk_per_trade: f64,
    /// Stop loss threshold
    pub stop_loss_threshold: f64,
    /// Take profit threshold
    pub take_profit_threshold: f64,
    /// Risk tolerance level (0.0 to 1.0)
    pub risk_tolerance: f64,
    /// Preferred holding period
    pub preferred_holding_period: Option<u64>,
}

/// Decision metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetadata {
    /// Decision ID for tracking
    pub decision_id: String,
    /// Session ID
    pub session_id: String,
    /// Algorithm version
    pub algorithm_version: String,
    /// Computation time in nanoseconds
    pub computation_time_ns: u64,
    /// Quality score of the decision
    pub quality_score: f64,
    /// Additional context tags
    pub tags: HashMap<String, String>,
}

impl DecisionContext {
    /// Create a new decision context
    pub fn new(
        market_data: MarketData,
        position: Option<Position>,
        behavioral_factors: BehavioralFactors,
        framing_context: FramingContext,
    ) -> Self {
        Self {
            market_data,
            position,
            behavioral_factors,
            framing_context,
            historical_context: HistoricalContext::default(),
            risk_constraints: RiskConstraints::default(),
            metadata: DecisionMetadata::default(),
        }
    }
    
    /// Update historical context with new decision outcome
    pub fn update_history(&mut self, outcome: DecisionOutcome) {
        self.historical_context.recent_decisions.push(outcome);
        
        // Keep only recent decisions (last 100)
        if self.historical_context.recent_decisions.len() > 100 {
            self.historical_context.recent_decisions.remove(0);
        }
        
        // Update performance metrics
        self.update_performance_metrics();
    }
    
    /// Add a learned pattern
    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.historical_context.learned_patterns.push(pattern);
    }
    
    /// Get relevant patterns for current market conditions
    pub fn get_relevant_patterns(&self) -> Vec<&Pattern> {
        self.historical_context.learned_patterns
            .iter()
            .filter(|pattern| pattern.confidence > 0.7)
            .collect()
    }
    
    /// Update market regime based on current conditions
    pub fn update_market_regime(&mut self) {
        // Simplified regime detection based on volatility and trend
        let volatility = self.market_data.current_price * 0.02; // Placeholder calculation
        
        self.historical_context.market_regime = if volatility > 0.05 {
            MarketRegime::HighVolatility
        } else if volatility < 0.01 {
            MarketRegime::LowVolatility
        } else {
            MarketRegime::Uncertain
        };
    }
    
    /// Check if risk constraints are satisfied for a potential action
    pub fn check_risk_constraints(&self, action: &str, position_size: f64) -> bool {
        if position_size > self.risk_constraints.max_position_size {
            return false;
        }
        
        // Additional risk checks would go here
        true
    }
    
    /// Calculate overall context score for decision quality
    pub fn context_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;
        
        // Market data quality
        if !self.market_data.possible_outcomes.is_empty() {
            score += 0.2;
        }
        factors += 1;
        
        // Historical context richness
        if !self.historical_context.recent_decisions.is_empty() {
            score += 0.3;
        }
        factors += 1;
        
        // Pattern availability
        if !self.historical_context.learned_patterns.is_empty() {
            score += 0.2;
        }
        factors += 1;
        
        // Behavioral factors completeness
        score += 0.3; // Always available
        factors += 1;
        
        score / factors as f64
    }
    
    /// Update performance metrics based on recent decisions
    fn update_performance_metrics(&mut self) {
        let decisions = &self.historical_context.recent_decisions;
        if decisions.is_empty() {
            return;
        }
        
        // Calculate win rate
        let wins = decisions.iter()
            .filter_map(|d| d.actual_outcome)
            .filter(|&outcome| outcome > 0.0)
            .count();
        
        let total_with_outcomes = decisions.iter()
            .filter(|d| d.actual_outcome.is_some())
            .count();
        
        if total_with_outcomes > 0 {
            self.historical_context.performance_metrics.win_rate = 
                wins as f64 / total_with_outcomes as f64;
        }
        
        // Calculate average return
        let total_return: f64 = decisions.iter()
            .filter_map(|d| d.actual_outcome)
            .sum();
        
        if total_with_outcomes > 0 {
            self.historical_context.performance_metrics.average_return = 
                total_return / total_with_outcomes as f64;
        }
        
        // Update consistency score based on variance of outcomes
        if total_with_outcomes > 1 {
            let returns: Vec<f64> = decisions.iter()
                .filter_map(|d| d.actual_outcome)
                .collect();
            
            let mean = self.historical_context.performance_metrics.average_return;
            let variance = returns.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            
            // Consistency is inverse of normalized variance
            self.historical_context.performance_metrics.consistency_score = 
                1.0 / (1.0 + variance);
        }
    }
}

impl Default for HistoricalContext {
    fn default() -> Self {
        Self {
            recent_decisions: Vec::new(),
            performance_metrics: PerformanceHistory::default(),
            learned_patterns: Vec::new(),
            market_regime: MarketRegime::Uncertain,
        }
    }
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self {
            win_rate: 0.0,
            average_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            consistency_score: 0.0,
        }
    }
}

impl Default for RiskConstraints {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,  // 10% of portfolio
            max_risk_per_trade: 0.02, // 2% risk per trade
            stop_loss_threshold: -0.05, // 5% stop loss
            take_profit_threshold: 0.10, // 10% take profit
            risk_tolerance: 0.5, // Moderate risk tolerance
            preferred_holding_period: Some(24 * 60 * 60 * 1000), // 1 day in ms
        }
    }
}

impl Default for DecisionMetadata {
    fn default() -> Self {
        Self {
            decision_id: uuid::Uuid::new_v4().to_string(),
            session_id: "default".to_string(),
            algorithm_version: "1.0.0".to_string(),
            computation_time_ns: 0,
            quality_score: 0.0,
            tags: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameType, test_utils::*};
    
    #[test]
    fn test_decision_context_creation() {
        let market_data = create_test_market_data();
        let behavioral_factors = create_test_behavioral_factors();
        let framing_context = FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        };
        
        let context = DecisionContext::new(
            market_data,
            None,
            behavioral_factors,
            framing_context,
        );
        
        assert_eq!(context.historical_context.recent_decisions.len(), 0);
        assert!(context.context_score() > 0.0);
    }
    
    #[test]
    fn test_historical_context_updates() {
        let mut context = create_test_decision_context();
        
        let outcome = DecisionOutcome {
            timestamp: 1640995200000,
            action: "BUY".to_string(),
            expected_value: 100.0,
            actual_outcome: Some(110.0),
            confidence: 0.8,
            context_factors: HashMap::new(),
        };
        
        context.update_history(outcome);
        
        assert_eq!(context.historical_context.recent_decisions.len(), 1);
        assert!(context.historical_context.performance_metrics.win_rate > 0.0);
    }
    
    #[test]
    fn test_pattern_management() {
        let mut context = create_test_decision_context();
        
        let pattern = Pattern {
            id: "trend_reversal".to_string(),
            description: "Trend reversal pattern".to_string(),
            confidence: 0.85,
            success_rate: 0.75,
            features: vec![1.0, 2.0, 3.0],
            last_seen: 1640995200000,
        };
        
        context.add_pattern(pattern);
        
        let relevant_patterns = context.get_relevant_patterns();
        assert_eq!(relevant_patterns.len(), 1);
        assert_eq!(relevant_patterns[0].id, "trend_reversal");
    }
    
    #[test]
    fn test_risk_constraints() {
        let context = create_test_decision_context();
        
        assert!(context.check_risk_constraints("BUY", 0.05)); // 5% position
        assert!(!context.check_risk_constraints("BUY", 0.15)); // 15% position exceeds 10% limit
    }
    
    #[test]
    fn test_context_score() {
        let context = create_test_decision_context();
        let score = context.context_score();
        
        assert!(score >= 0.0 && score <= 1.0);
        assert!(score > 0.5); // Should be decent score with test data
    }
}