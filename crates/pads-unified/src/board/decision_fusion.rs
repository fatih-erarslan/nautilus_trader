//! # Decision Fusion System
//!
//! Advanced decision fusion system implementing multiple decision styles and strategies
//! for the PADS trading system. Combines various decision-making approaches with
//! sophisticated aggregation and consensus mechanisms.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::lmsr::{LMSRMarket, LMSRConfig, MarketState, ScoringResult};

/// Decision styles for different market conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionStyle {
    /// Seeks broad agreement across all agents
    Consensus,
    /// Takes advantage of immediate opportunities
    Opportunistic,
    /// Prioritizes risk management and capital preservation
    Defensive,
    /// Balanced approach with calculated position sizing
    CalculatedRisk,
    /// Contrarian approach - goes against market sentiment
    Contrarian,
    /// Follows strong trends and momentum
    MomentumFollowing,
}

/// Market condition types for style selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketCondition {
    Bullish,
    Bearish,
    Sideways,
    Volatile,
    Trending,
    Consolidating,
}

/// Decision fusion configuration
#[derive(Debug, Clone)]
pub struct DecisionFusionConfig {
    /// Weights for different decision styles
    pub style_weights: HashMap<DecisionStyle, f64>,
    /// Minimum consensus threshold
    pub consensus_threshold: f64,
    /// Maximum position risk per decision
    pub max_position_risk: f64,
    /// Style adaptation enabled
    pub adaptive_styles: bool,
    /// Market condition detection enabled
    pub market_condition_detection: bool,
}

impl Default for DecisionFusionConfig {
    fn default() -> Self {
        let mut style_weights = HashMap::new();
        style_weights.insert(DecisionStyle::Consensus, 1.0);
        style_weights.insert(DecisionStyle::Opportunistic, 0.8);
        style_weights.insert(DecisionStyle::Defensive, 1.2);
        style_weights.insert(DecisionStyle::CalculatedRisk, 1.0);
        style_weights.insert(DecisionStyle::Contrarian, 0.6);
        style_weights.insert(DecisionStyle::MomentumFollowing, 0.9);
        
        Self {
            style_weights,
            consensus_threshold: 0.7,
            max_position_risk: 0.25,
            adaptive_styles: true,
            market_condition_detection: true,
        }
    }
}

/// Individual decision from a style
#[derive(Debug, Clone)]
pub struct StyleDecision {
    pub style: DecisionStyle,
    pub action: DecisionAction,
    pub confidence: f64,
    pub position_size: f64,
    pub reasoning: String,
    pub risk_score: f64,
    pub timestamp: Instant,
}

/// Decision actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionAction {
    Buy,
    Sell,
    Hold,
    Increase(f64),
    Decrease(f64),
    Emergency,
}

/// Fused decision result
#[derive(Debug, Clone)]
pub struct FusedDecision {
    pub final_action: DecisionAction,
    pub confidence: f64,
    pub consensus_score: f64,
    pub style_votes: HashMap<DecisionStyle, StyleDecision>,
    pub market_condition: MarketCondition,
    pub risk_assessment: f64,
    pub reasoning: Vec<String>,
    pub metadata: HashMap<String, f64>,
}

/// Decision fusion engine
pub struct DecisionFusionEngine {
    config: DecisionFusionConfig,
    lmsr_market: Arc<LMSRMarket>,
    style_engines: HashMap<DecisionStyle, Box<dyn DecisionStyleEngine + Send + Sync>>,
    performance_tracker: Arc<RwLock<StylePerformanceTracker>>,
    market_condition_detector: Arc<MarketConditionDetector>,
}

/// Performance tracking for decision styles
#[derive(Debug, Clone)]
pub struct StylePerformanceTracker {
    pub style_performance: HashMap<DecisionStyle, StylePerformance>,
    pub total_decisions: u64,
    pub correct_predictions: u64,
    pub average_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct StylePerformance {
    pub accuracy: f64,
    pub average_confidence: f64,
    pub risk_adjusted_return: f64,
    pub total_trades: u64,
    pub winning_trades: u64,
    pub drawdown: f64,
}

impl Default for StylePerformanceTracker {
    fn default() -> Self {
        Self {
            style_performance: HashMap::new(),
            total_decisions: 0,
            correct_predictions: 0,
            average_confidence: 0.0,
        }
    }
}

/// Trait for decision style engines
pub trait DecisionStyleEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision;
    fn get_style(&self) -> DecisionStyle;
    fn update_performance(&mut self, actual_outcome: f64, predicted_outcome: f64);
}

/// Market condition detector
pub struct MarketConditionDetector {
    trend_threshold: f64,
    volatility_threshold: f64,
    momentum_threshold: f64,
}

impl MarketConditionDetector {
    pub fn new() -> Self {
        Self {
            trend_threshold: 0.15,
            volatility_threshold: 0.4,
            momentum_threshold: 0.2,
        }
    }
    
    pub fn detect_condition(&self, market: &MarketState) -> MarketCondition {
        let trend_abs = market.trend.abs();
        let is_trending = trend_abs > self.trend_threshold;
        let is_volatile = market.volatility > self.volatility_threshold;
        let strong_momentum = market.momentum.abs() > self.momentum_threshold;
        
        if is_volatile && !is_trending {
            MarketCondition::Volatile
        } else if is_trending && strong_momentum {
            if market.trend > 0.0 {
                MarketCondition::Bullish
            } else {
                MarketCondition::Bearish
            }
        } else if is_trending {
            MarketCondition::Trending
        } else if market.volatility < 0.1 && trend_abs < 0.05 {
            MarketCondition::Consolidating
        } else {
            MarketCondition::Sideways
        }
    }
}

impl DecisionFusionEngine {
    pub fn new(config: DecisionFusionConfig) -> Self {
        let lmsr_config = LMSRConfig::default();
        let lmsr_market = Arc::new(LMSRMarket::new(lmsr_config));
        let performance_tracker = Arc::new(RwLock::new(StylePerformanceTracker::default()));
        let market_condition_detector = Arc::new(MarketConditionDetector::new());
        
        let mut style_engines: HashMap<DecisionStyle, Box<dyn DecisionStyleEngine + Send + Sync>> = HashMap::new();
        style_engines.insert(DecisionStyle::Consensus, Box::new(ConsensusEngine::new()));
        style_engines.insert(DecisionStyle::Opportunistic, Box::new(OpportunisticEngine::new()));
        style_engines.insert(DecisionStyle::Defensive, Box::new(DefensiveEngine::new()));
        style_engines.insert(DecisionStyle::CalculatedRisk, Box::new(CalculatedRiskEngine::new()));
        style_engines.insert(DecisionStyle::Contrarian, Box::new(ContrarianEngine::new()));
        style_engines.insert(DecisionStyle::MomentumFollowing, Box::new(MomentumFollowingEngine::new()));
        
        Self {
            config,
            lmsr_market,
            style_engines,
            performance_tracker,
            market_condition_detector,
        }
    }
    
    /// Fuse decisions from all styles
    pub async fn fuse_decisions(&self, market: &MarketState) -> FusedDecision {
        let start = Instant::now();
        
        // Get LMSR scoring first
        let lmsr_result = self.lmsr_market.calculate_score(market).await;
        
        // Detect market condition
        let market_condition = if self.config.market_condition_detection {
            self.market_condition_detector.detect_condition(market)
        } else {
            MarketCondition::Sideways
        };
        
        // Get decisions from all styles
        let mut style_votes = HashMap::new();
        for (style, engine) in &self.style_engines {
            let decision = engine.make_decision(market, &lmsr_result);
            style_votes.insert(*style, decision);
        }
        
        // Apply adaptive weights based on market condition
        let adaptive_weights = if self.config.adaptive_styles {
            self.calculate_adaptive_weights(market_condition).await
        } else {
            self.config.style_weights.clone()
        };
        
        // Calculate weighted consensus
        let consensus_result = self.calculate_weighted_consensus(&style_votes, &adaptive_weights).await;
        
        // Risk assessment
        let risk_assessment = self.assess_risk(market, &style_votes, &lmsr_result).await;
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(&style_votes, &consensus_result, market_condition);
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("lmsr_confidence".to_string(), lmsr_result.confidence);
        metadata.insert("lmsr_info_gain".to_string(), lmsr_result.information_gain);
        metadata.insert("processing_time_ms".to_string(), start.elapsed().as_millis() as f64);
        metadata.insert("risk_score".to_string(), risk_assessment);
        
        FusedDecision {
            final_action: consensus_result.action,
            confidence: consensus_result.confidence,
            consensus_score: consensus_result.consensus_score,
            style_votes,
            market_condition,
            risk_assessment,
            reasoning,
            metadata,
        }
    }
    
    /// Calculate adaptive weights based on market condition
    async fn calculate_adaptive_weights(&self, condition: MarketCondition) -> HashMap<DecisionStyle, f64> {
        let mut weights = self.config.style_weights.clone();
        
        match condition {
            MarketCondition::Bullish => {
                weights.insert(DecisionStyle::MomentumFollowing, 1.3);
                weights.insert(DecisionStyle::Opportunistic, 1.2);
                weights.insert(DecisionStyle::Defensive, 0.8);
            }
            MarketCondition::Bearish => {
                weights.insert(DecisionStyle::Defensive, 1.4);
                weights.insert(DecisionStyle::Contrarian, 1.1);
                weights.insert(DecisionStyle::MomentumFollowing, 0.7);
            }
            MarketCondition::Volatile => {
                weights.insert(DecisionStyle::Defensive, 1.5);
                weights.insert(DecisionStyle::CalculatedRisk, 1.2);
                weights.insert(DecisionStyle::Opportunistic, 0.6);
            }
            MarketCondition::Trending => {
                weights.insert(DecisionStyle::MomentumFollowing, 1.4);
                weights.insert(DecisionStyle::Consensus, 1.1);
                weights.insert(DecisionStyle::Contrarian, 0.5);
            }
            MarketCondition::Consolidating => {
                weights.insert(DecisionStyle::Consensus, 1.3);
                weights.insert(DecisionStyle::CalculatedRisk, 1.1);
                weights.insert(DecisionStyle::MomentumFollowing, 0.6);
            }
            MarketCondition::Sideways => {
                // Use default weights
            }
        }
        
        weights
    }
    
    /// Calculate weighted consensus from style votes
    async fn calculate_weighted_consensus(
        &self,
        style_votes: &HashMap<DecisionStyle, StyleDecision>,
        weights: &HashMap<DecisionStyle, f64>,
    ) -> ConsensusResult {
        let mut action_scores: HashMap<DecisionAction, f64> = HashMap::new();
        let mut total_weight = 0.0;
        let mut total_confidence = 0.0;
        
        for (style, decision) in style_votes {
            if let Some(&weight) = weights.get(style) {
                let score = decision.confidence * weight;
                let action = decision.action;
                
                *action_scores.entry(action).or_insert(0.0) += score;
                total_weight += weight;
                total_confidence += decision.confidence * weight;
            }
        }
        
        // Find highest scoring action
        let (best_action, best_score) = action_scores
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((DecisionAction::Hold, 0.0));
        
        let consensus_score = if total_weight > 0.0 {
            best_score / total_weight
        } else {
            0.0
        };
        
        let confidence = if total_weight > 0.0 {
            total_confidence / total_weight
        } else {
            0.0
        };
        
        ConsensusResult {
            action: best_action,
            confidence,
            consensus_score,
        }
    }
    
    /// Assess overall risk of the decision
    async fn assess_risk(
        &self,
        market: &MarketState,
        style_votes: &HashMap<DecisionStyle, StyleDecision>,
        lmsr_result: &ScoringResult,
    ) -> f64 {
        let mut risk_factors = Vec::new();
        
        // Market volatility risk
        risk_factors.push(market.volatility);
        
        // LMSR emergency risk
        risk_factors.push(lmsr_result.emergency_score);
        
        // Style disagreement risk
        let mut actions: Vec<DecisionAction> = style_votes.values().map(|d| d.action).collect();
        actions.sort_by(|a, b| std::mem::discriminant(a).cmp(&std::mem::discriminant(b)));
        let unique_actions = actions.len();
        let disagreement_risk = if unique_actions > 1 {
            (unique_actions - 1) as f64 / style_votes.len() as f64
        } else {
            0.0
        };
        risk_factors.push(disagreement_risk);
        
        // Position size risk
        let avg_position_size = style_votes.values()
            .map(|d| d.position_size.abs())
            .sum::<f64>() / style_votes.len() as f64;
        risk_factors.push(avg_position_size / self.config.max_position_risk);
        
        // Calculate weighted risk
        let weights = vec![0.3, 0.3, 0.2, 0.2];
        risk_factors.iter()
            .zip(weights.iter())
            .map(|(risk, weight)| risk * weight)
            .sum::<f64>()
            .min(1.0)
    }
    
    /// Generate reasoning for the decision
    fn generate_reasoning(
        &self,
        style_votes: &HashMap<DecisionStyle, StyleDecision>,
        consensus: &ConsensusResult,
        market_condition: MarketCondition,
    ) -> Vec<String> {
        let mut reasoning = Vec::new();
        
        reasoning.push(format!("Market condition: {:?}", market_condition));
        reasoning.push(format!("Consensus confidence: {:.2}", consensus.confidence));
        reasoning.push(format!("Final action: {:?}", consensus.action));
        
        // Top contributing styles
        let mut sorted_votes: Vec<_> = style_votes.iter().collect();
        sorted_votes.sort_by(|a, b| b.1.confidence.partial_cmp(&a.1.confidence).unwrap());
        
        for (style, decision) in sorted_votes.iter().take(3) {
            reasoning.push(format!("{:?}: {} (confidence: {:.2})", 
                                 style, decision.reasoning, decision.confidence));
        }
        
        reasoning
    }
}

/// Consensus calculation result
#[derive(Debug, Clone)]
struct ConsensusResult {
    action: DecisionAction,
    confidence: f64,
    consensus_score: f64,
}

// Decision style engine implementations

/// Consensus-seeking engine
struct ConsensusEngine;

impl ConsensusEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for ConsensusEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let confidence = lmsr_result.confidence * (1.0 - lmsr_result.variance);
        let action = if confidence > 0.8 {
            match lmsr_result.decision {
                crate::lmsr::LMSRDecision::Buy => DecisionAction::Buy,
                crate::lmsr::LMSRDecision::Sell => DecisionAction::Sell,
                crate::lmsr::LMSRDecision::Increase(n) => DecisionAction::Increase(n as f64 * 0.5),
                crate::lmsr::LMSRDecision::Decrease(n) => DecisionAction::Decrease(n as f64 * 0.5),
                _ => DecisionAction::Hold,
            }
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::Consensus,
            action,
            confidence,
            position_size: 0.1,
            reasoning: "Seeking broad consensus".to_string(),
            risk_score: lmsr_result.variance,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::Consensus
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

/// Opportunistic engine
struct OpportunisticEngine;

impl OpportunisticEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for OpportunisticEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let opportunity_score = lmsr_result.information_gain * (1.0 - market.volatility);
        let confidence = lmsr_result.confidence * opportunity_score;
        
        let action = if opportunity_score > 0.5 {
            match lmsr_result.decision {
                crate::lmsr::LMSRDecision::Buy => DecisionAction::Increase(0.2),
                crate::lmsr::LMSRDecision::Sell => DecisionAction::Decrease(0.2),
                crate::lmsr::LMSRDecision::Increase(n) => DecisionAction::Increase(n as f64 * 1.5),
                crate::lmsr::LMSRDecision::Decrease(n) => DecisionAction::Decrease(n as f64 * 1.5),
                _ => DecisionAction::Hold,
            }
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::Opportunistic,
            action,
            confidence,
            position_size: 0.15,
            reasoning: "Exploiting market opportunity".to_string(),
            risk_score: market.volatility,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::Opportunistic
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

/// Defensive engine
struct DefensiveEngine;

impl DefensiveEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for DefensiveEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let risk_factor = market.volatility + lmsr_result.emergency_score;
        let confidence = lmsr_result.confidence * (1.0 - risk_factor);
        
        let action = if risk_factor > 0.7 {
            DecisionAction::Decrease(0.3)
        } else if confidence > 0.8 && risk_factor < 0.3 {
            match lmsr_result.decision {
                crate::lmsr::LMSRDecision::Buy => DecisionAction::Buy,
                crate::lmsr::LMSRDecision::Sell => DecisionAction::Sell,
                _ => DecisionAction::Hold,
            }
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::Defensive,
            action,
            confidence,
            position_size: 0.05,
            reasoning: "Prioritizing capital preservation".to_string(),
            risk_score: risk_factor,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::Defensive
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

/// Calculated risk engine
struct CalculatedRiskEngine;

impl CalculatedRiskEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for CalculatedRiskEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let risk_reward = lmsr_result.information_gain / (market.volatility + 0.01);
        let confidence = lmsr_result.confidence * risk_reward.min(1.0);
        
        let position_size = (confidence * 0.2).min(0.15);
        let action = if confidence > 0.6 {
            match lmsr_result.decision {
                crate::lmsr::LMSRDecision::Buy => DecisionAction::Increase(position_size),
                crate::lmsr::LMSRDecision::Sell => DecisionAction::Decrease(position_size),
                _ => DecisionAction::Hold,
            }
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::CalculatedRisk,
            action,
            confidence,
            position_size,
            reasoning: format!("Risk-reward ratio: {:.2}", risk_reward),
            risk_score: market.volatility,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::CalculatedRisk
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

/// Contrarian engine
struct ContrarianEngine;

impl ContrarianEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for ContrarianEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let contrarian_signal = -market.momentum; // Opposite of momentum
        let confidence = lmsr_result.confidence * contrarian_signal.abs();
        
        let action = if contrarian_signal > 0.3 {
            DecisionAction::Buy
        } else if contrarian_signal < -0.3 {
            DecisionAction::Sell
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::Contrarian,
            action,
            confidence,
            position_size: 0.08,
            reasoning: "Contrarian market position".to_string(),
            risk_score: market.volatility,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::Contrarian
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

/// Momentum following engine
struct MomentumFollowingEngine;

impl MomentumFollowingEngine {
    fn new() -> Self {
        Self
    }
}

impl DecisionStyleEngine for MomentumFollowingEngine {
    fn make_decision(&self, market: &MarketState, lmsr_result: &ScoringResult) -> StyleDecision {
        let momentum_strength = market.momentum.abs();
        let confidence = lmsr_result.confidence * momentum_strength;
        
        let action = if market.momentum > 0.2 {
            DecisionAction::Increase(0.15)
        } else if market.momentum < -0.2 {
            DecisionAction::Decrease(0.15)
        } else {
            DecisionAction::Hold
        };
        
        StyleDecision {
            style: DecisionStyle::MomentumFollowing,
            action,
            confidence,
            position_size: 0.12,
            reasoning: format!("Following momentum: {:.2}", market.momentum),
            risk_score: 1.0 - momentum_strength,
            timestamp: Instant::now(),
        }
    }
    
    fn get_style(&self) -> DecisionStyle {
        DecisionStyle::MomentumFollowing
    }
    
    fn update_performance(&mut self, _actual_outcome: f64, _predicted_outcome: f64) {
        // Update performance metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_decision_fusion_engine() {
        let config = DecisionFusionConfig::default();
        let engine = DecisionFusionEngine::new(config);
        
        let market_state = MarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.3,
            trend: 0.2,
            momentum: 0.1,
            timestamp: 1234567890,
            liquidity: 0.8,
        };
        
        let result = engine.fuse_decisions(&market_state).await;
        
        assert!(result.confidence > 0.0);
        assert!(result.consensus_score >= 0.0);
        assert!(result.risk_assessment >= 0.0);
        assert!(!result.reasoning.is_empty());
    }
    
    #[test]
    fn test_market_condition_detection() {
        let detector = MarketConditionDetector::new();
        
        // Test bullish condition
        let bullish_market = MarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.2,
            trend: 0.3,
            momentum: 0.25,
            timestamp: 1234567890,
            liquidity: 0.8,
        };
        
        let condition = detector.detect_condition(&bullish_market);
        assert!(matches!(condition, MarketCondition::Bullish));
        
        // Test volatile condition
        let volatile_market = MarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.8,
            trend: 0.05,
            momentum: 0.1,
            timestamp: 1234567890,
            liquidity: 0.8,
        };
        
        let condition = detector.detect_condition(&volatile_market);
        assert!(matches!(condition, MarketCondition::Volatile));
    }
    
    #[test]
    fn test_consensus_engine() {
        let engine = ConsensusEngine::new();
        
        let market_state = MarketState {
            price: 100.0,
            volume: 1000.0,
            volatility: 0.3,
            trend: 0.2,
            momentum: 0.1,
            timestamp: 1234567890,
            liquidity: 0.8,
        };
        
        let lmsr_result = ScoringResult {
            decision: crate::lmsr::LMSRDecision::Buy,
            confidence: 0.8,
            information_gain: 0.1,
            market_probability: 0.6,
            cost_differential: -0.05,
            variance: 0.1,
            emergency_score: 0.0,
        };
        
        let decision = engine.make_decision(&market_state, &lmsr_result);
        
        assert!(matches!(decision.style, DecisionStyle::Consensus));
        assert!(decision.confidence > 0.0);
        assert!(decision.position_size > 0.0);
    }
}