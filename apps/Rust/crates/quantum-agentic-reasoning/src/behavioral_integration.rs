//! Behavioral Integration Module
//! 
//! Integrates Quantum Prospect Theory with existing QAR decision systems.
//! Provides seamless integration between behavioral economics and quantum algorithms.

use crate::core::{QarResult, FactorMap, TradingDecision, DecisionType};
use crate::analysis::AnalysisResult;
use prospect_theory::{
    QuantumProspectTheory, MarketData, Position, FramingContext, FrameType, TradingAction
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Behavioral integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralConfig {
    /// Weight for prospect theory decisions
    pub prospect_theory_weight: f64,
    /// Enable framing effects
    pub enable_framing: bool,
    /// Enable loss aversion adjustments
    pub enable_loss_aversion: bool,
    /// Enable probability weighting
    pub enable_probability_weighting: bool,
    /// Enable mental accounting
    pub enable_mental_accounting: bool,
    /// Reference point adaptation rate
    pub reference_adaptation_rate: f64,
}

impl Default for BehavioralConfig {
    fn default() -> Self {
        Self {
            prospect_theory_weight: 0.4,
            enable_framing: true,
            enable_loss_aversion: true,
            enable_probability_weighting: true,
            enable_mental_accounting: true,
            reference_adaptation_rate: 0.1,
        }
    }
}

/// Behavioral decision enhancer
#[derive(Debug)]
pub struct BehavioralDecisionEnhancer {
    config: BehavioralConfig,
    prospect_theory: QuantumProspectTheory,
    reference_points: HashMap<String, f64>,
    framing_history: Vec<FramingContext>,
}

impl BehavioralDecisionEnhancer {
    /// Create new behavioral decision enhancer
    pub fn new(config: BehavioralConfig, prospect_theory: QuantumProspectTheory) -> Self {
        Self {
            config,
            prospect_theory,
            reference_points: HashMap::new(),
            framing_history: Vec::new(),
        }
    }

    /// Enhance trading decision with behavioral factors
    pub fn enhance_decision(&mut self, 
                           base_decision: &TradingDecision,
                           market_factors: &FactorMap,
                           analysis_result: &AnalysisResult,
                           symbol: &str) -> QarResult<EnhancedBehavioralDecision> {
        
        // Extract market data for prospect theory
        let market_data = self.create_market_data_from_factors(market_factors, symbol)?;
        
        // Get current position context
        let position = self.get_position_context(symbol, base_decision);
        
        // Make prospect theory decision
        let pt_decision = self.prospect_theory.make_trading_decision(&market_data, position.as_ref())
            .map_err(|e| crate::core::QarError::External(format!("Prospect Theory error: {}", e)))?;
        
        // Apply behavioral enhancements
        let enhanced_confidence = self.apply_confidence_enhancement(
            base_decision.confidence, 
            pt_decision.confidence,
            &pt_decision.behavioral_factors
        );
        
        let enhanced_expected_return = self.apply_return_enhancement(
            base_decision.expected_return,
            pt_decision.expected_value,
            &pt_decision.behavioral_factors
        );
        
        let enhanced_risk_assessment = self.apply_risk_enhancement(
            base_decision.risk_assessment,
            pt_decision.risk_metric,
            &pt_decision.behavioral_factors
        );
        
        // Determine final decision type
        let enhanced_decision_type = self.blend_decision_types(
            &base_decision.decision_type,
            &pt_decision.action
        );
        
        // Update reference points
        self.update_reference_point(symbol, &market_data);
        
        // Store framing context
        self.framing_history.push(market_data.frame.clone());
        if self.framing_history.len() > 100 {
            self.framing_history.remove(0);
        }
        
        Ok(EnhancedBehavioralDecision {
            original_decision: base_decision.clone(),
            prospect_theory_decision: pt_decision.clone(),
            enhanced_decision: TradingDecision {
                decision_type: enhanced_decision_type,
                confidence: enhanced_confidence,
                expected_return: enhanced_expected_return,
                risk_assessment: enhanced_risk_assessment,
                urgency_score: base_decision.urgency_score, // Keep original urgency
                reasoning: format!("Behavioral enhancement: PT={:.3}, base={:.3}, blend={:.3}", 
                                 pt_decision.confidence, base_decision.confidence, enhanced_confidence),
                timestamp: chrono::Utc::now(),
            },
            behavioral_factors: pt_decision.behavioral_factors,
            reference_point: self.reference_points.get(symbol).copied(),
            framing_effect: self.calculate_framing_effect(&market_data.frame),
        })
    }

    /// Create market data from factor map
    fn create_market_data_from_factors(&self, factors: &FactorMap, symbol: &str) -> QarResult<MarketData> {
        use crate::core::StandardFactors;
        
        let trend = factors.get_factor(&StandardFactors::Trend)?;
        let volatility = factors.get_factor(&StandardFactors::Volatility)?;
        let momentum = factors.get_factor(&StandardFactors::Momentum)?;
        let sentiment = factors.get_factor(&StandardFactors::Sentiment)?;
        
        // Get reference point or use default
        let reference_point = self.reference_points.get(symbol).copied().unwrap_or(100.0);
        let current_price = reference_point * (1.0 + (trend - 0.5) * 0.1);
        
        // Create outcome scenarios based on volatility
        let volatility_range = volatility * current_price * 0.2;
        let possible_outcomes = vec![
            current_price + volatility_range * 2.0,
            current_price + volatility_range,
            current_price,
            current_price - volatility_range,
            current_price - volatility_range * 2.0,
        ];
        
        // Create probabilities based on trend and momentum
        let bullish_bias = (trend + momentum) / 2.0;
        let buy_probabilities = vec![
            bullish_bias * 0.4,
            bullish_bias * 0.3,
            0.2,
            (1.0 - bullish_bias) * 0.2,
            (1.0 - bullish_bias) * 0.1,
        ];
        
        let sell_probabilities = vec![
            (1.0 - bullish_bias) * 0.1,
            (1.0 - bullish_bias) * 0.2,
            0.2,
            bullish_bias * 0.3,
            bullish_bias * 0.4,
        ];
        
        let hold_probabilities = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        
        // Determine framing based on sentiment and recent performance
        let frame_type = if sentiment > 0.7 {
            FrameType::Gain
        } else if sentiment < 0.3 {
            FrameType::Loss
        } else {
            FrameType::Neutral
        };
        
        Ok(MarketData {
            symbol: symbol.to_string(),
            current_price,
            possible_outcomes,
            buy_probabilities,
            sell_probabilities,
            hold_probabilities,
            frame: FramingContext {
                frame_type,
                emphasis: sentiment,
            },
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
        })
    }

    /// Get position context for decision making
    fn get_position_context(&self, symbol: &str, base_decision: &TradingDecision) -> Option<Position> {
        // Create a position based on the current decision context
        // In a real implementation, this would come from portfolio state
        match base_decision.decision_type {
            DecisionType::Buy => Some(Position {
                symbol: symbol.to_string(),
                quantity: 1.0,
                entry_price: 100.0, // Would be actual entry price
                current_value: 100.0,
                unrealized_pnl: 0.0,
            }),
            DecisionType::Sell => Some(Position {
                symbol: symbol.to_string(),
                quantity: -1.0,
                entry_price: 100.0,
                current_value: 100.0,
                unrealized_pnl: 0.0,
            }),
            DecisionType::Hold => None,
        }
    }

    /// Apply confidence enhancement using behavioral factors
    fn apply_confidence_enhancement(&self, 
                                   base_confidence: f64,
                                   pt_confidence: f64,
                                   behavioral_factors: &prospect_theory::BehavioralFactors) -> f64 {
        if !self.config.enable_loss_aversion {
            return base_confidence;
        }

        let pt_weight = self.config.prospect_theory_weight;
        let base_weight = 1.0 - pt_weight;
        
        // Apply loss aversion bias to confidence
        let loss_aversion_adjustment = if behavioral_factors.loss_aversion_impact > 0.0 {
            1.0 - behavioral_factors.loss_aversion_impact * 0.2 // Reduce confidence for losses
        } else {
            1.0 + behavioral_factors.loss_aversion_impact.abs() * 0.1 // Increase confidence for gains
        };
        
        let blended_confidence = (base_confidence * base_weight + pt_confidence * pt_weight) * loss_aversion_adjustment;
        blended_confidence.max(0.0).min(1.0)
    }

    /// Apply return enhancement using behavioral factors
    fn apply_return_enhancement(&self,
                               base_return: Option<f64>,
                               pt_expected_value: f64,
                               behavioral_factors: &prospect_theory::BehavioralFactors) -> Option<f64> {
        let base = base_return.unwrap_or(0.0);
        let pt_weight = self.config.prospect_theory_weight;
        
        if !self.config.enable_probability_weighting {
            return Some(base);
        }
        
        // Apply probability weighting bias
        let probability_adjustment = 1.0 + behavioral_factors.probability_weighting_bias * 0.1;
        let blended_return = (base * (1.0 - pt_weight) + pt_expected_value * pt_weight) * probability_adjustment;
        
        Some(blended_return)
    }

    /// Apply risk enhancement using behavioral factors
    fn apply_risk_enhancement(&self,
                             base_risk: Option<f64>,
                             pt_risk: f64,
                             behavioral_factors: &prospect_theory::BehavioralFactors) -> Option<f64> {
        let base = base_risk.unwrap_or(0.5);
        let pt_weight = self.config.prospect_theory_weight;
        
        // Mental accounting can affect risk perception
        let mental_accounting_adjustment = if self.config.enable_mental_accounting {
            1.0 + behavioral_factors.mental_accounting_bias * 0.15
        } else {
            1.0
        };
        
        let blended_risk = (base * (1.0 - pt_weight) + pt_risk * pt_weight) * mental_accounting_adjustment;
        Some(blended_risk.max(0.0).min(1.0))
    }

    /// Blend decision types from base and prospect theory
    fn blend_decision_types(&self, base_type: &DecisionType, pt_action: &TradingAction) -> DecisionType {
        let pt_weight = self.config.prospect_theory_weight;
        
        // If prospect theory has high weight, prefer its decision
        if pt_weight > 0.5 {
            match pt_action {
                TradingAction::Buy => DecisionType::Buy,
                TradingAction::Sell => DecisionType::Sell,
                TradingAction::Hold => DecisionType::Hold,
            }
        } else {
            base_type.clone()
        }
    }

    /// Update reference point using adaptive learning
    fn update_reference_point(&mut self, symbol: &str, market_data: &MarketData) {
        let current_ref = self.reference_points.get(symbol).copied().unwrap_or(market_data.current_price);
        let new_ref = current_ref * (1.0 - self.config.reference_adaptation_rate) + 
                     market_data.current_price * self.config.reference_adaptation_rate;
        
        self.reference_points.insert(symbol.to_string(), new_ref);
    }

    /// Calculate framing effect impact
    fn calculate_framing_effect(&self, frame: &FramingContext) -> f64 {
        if !self.config.enable_framing {
            return 0.0;
        }

        match frame.frame_type {
            FrameType::Gain => frame.emphasis * 0.1,  // Positive framing effect
            FrameType::Loss => -frame.emphasis * 0.15, // Negative framing effect (stronger)
            FrameType::Neutral => 0.0,
        }
    }

    /// Get behavioral performance metrics
    pub fn get_behavioral_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        metrics.insert("reference_points_tracked".to_string(), self.reference_points.len() as f64);
        metrics.insert("framing_history_length".to_string(), self.framing_history.len() as f64);
        
        // Analyze framing distribution
        let gain_frames = self.framing_history.iter().filter(|f| matches!(f.frame_type, FrameType::Gain)).count();
        let loss_frames = self.framing_history.iter().filter(|f| matches!(f.frame_type, FrameType::Loss)).count();
        let neutral_frames = self.framing_history.iter().filter(|f| matches!(f.frame_type, FrameType::Neutral)).count();
        
        let total_frames = self.framing_history.len() as f64;
        if total_frames > 0.0 {
            metrics.insert("gain_frame_ratio".to_string(), gain_frames as f64 / total_frames);
            metrics.insert("loss_frame_ratio".to_string(), loss_frames as f64 / total_frames);
            metrics.insert("neutral_frame_ratio".to_string(), neutral_frames as f64 / total_frames);
        }
        
        // Average reference point
        if !self.reference_points.is_empty() {
            let avg_ref_point = self.reference_points.values().sum::<f64>() / self.reference_points.len() as f64;
            metrics.insert("average_reference_point".to_string(), avg_ref_point);
        }
        
        metrics
    }

    /// Reset behavioral state
    pub fn reset(&mut self) {
        self.reference_points.clear();
        self.framing_history.clear();
    }
}

/// Enhanced decision with behavioral factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedBehavioralDecision {
    /// Original base decision
    pub original_decision: TradingDecision,
    /// Prospect theory decision
    pub prospect_theory_decision: prospect_theory::TradingDecision,
    /// Final enhanced decision
    pub enhanced_decision: TradingDecision,
    /// Behavioral factors that influenced the decision
    pub behavioral_factors: prospect_theory::BehavioralFactors,
    /// Reference point used
    pub reference_point: Option<f64>,
    /// Framing effect impact
    pub framing_effect: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prospect_theory::QuantumProspectTheoryConfig;

    #[test]
    fn test_behavioral_enhancer_creation() {
        let config = BehavioralConfig::default();
        let pt_config = QuantumProspectTheoryConfig::default();
        let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
        
        let enhancer = BehavioralDecisionEnhancer::new(config, prospect_theory);
        assert_eq!(enhancer.reference_points.len(), 0);
        assert_eq!(enhancer.framing_history.len(), 0);
    }

    #[test]
    fn test_market_data_creation() {
        let config = BehavioralConfig::default();
        let pt_config = QuantumProspectTheoryConfig::default();
        let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
        let enhancer = BehavioralDecisionEnhancer::new(config, prospect_theory);
        
        let mut factors = std::collections::HashMap::new();
        factors.insert("trend".to_string(), 0.7);
        factors.insert("volatility".to_string(), 0.3);
        factors.insert("momentum".to_string(), 0.6);
        factors.insert("sentiment".to_string(), 0.8);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let market_data = enhancer.create_market_data_from_factors(&factor_map, "BTC/USDT");
        
        assert!(market_data.is_ok());
        let data = market_data.unwrap();
        assert_eq!(data.symbol, "BTC/USDT");
        assert_eq!(data.possible_outcomes.len(), 5);
        assert!(matches!(data.frame.frame_type, FrameType::Gain)); // High sentiment should create gain frame
    }

    #[test]
    fn test_confidence_enhancement() {
        let config = BehavioralConfig::default();
        let pt_config = QuantumProspectTheoryConfig::default();
        let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
        let enhancer = BehavioralDecisionEnhancer::new(config, prospect_theory);
        
        let behavioral_factors = prospect_theory::BehavioralFactors {
            loss_aversion_impact: -0.2, // Gain scenario
            probability_weighting_bias: 0.1,
            mental_accounting_bias: 0.05,
        };
        
        let enhanced = enhancer.apply_confidence_enhancement(0.7, 0.8, &behavioral_factors);
        
        // Should be between the two input confidences, adjusted for behavioral factors
        assert!(enhanced > 0.6 && enhanced < 0.9);
    }

    #[test]
    fn test_decision_type_blending() {
        let config = BehavioralConfig {
            prospect_theory_weight: 0.6, // High PT weight
            ..Default::default()
        };
        let pt_config = QuantumProspectTheoryConfig::default();
        let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
        let enhancer = BehavioralDecisionEnhancer::new(config, prospect_theory);
        
        // PT should dominate with high weight
        let blended = enhancer.blend_decision_types(&DecisionType::Hold, &TradingAction::Buy);
        assert_eq!(blended, DecisionType::Buy);
        
        // Base should dominate with low PT weight
        let config_low_pt = BehavioralConfig {
            prospect_theory_weight: 0.3,
            ..Default::default()
        };
        let enhancer_low_pt = BehavioralDecisionEnhancer::new(config_low_pt, prospect_theory);
        let blended_low = enhancer_low_pt.blend_decision_types(&DecisionType::Hold, &TradingAction::Buy);
        assert_eq!(blended_low, DecisionType::Hold);
    }

    #[test]
    fn test_framing_effect_calculation() {
        let config = BehavioralConfig::default();
        let pt_config = QuantumProspectTheoryConfig::default();
        let prospect_theory = QuantumProspectTheory::new(pt_config).unwrap();
        let enhancer = BehavioralDecisionEnhancer::new(config, prospect_theory);
        
        let gain_frame = FramingContext {
            frame_type: FrameType::Gain,
            emphasis: 0.8,
        };
        let gain_effect = enhancer.calculate_framing_effect(&gain_frame);
        assert!(gain_effect > 0.0);
        
        let loss_frame = FramingContext {
            frame_type: FrameType::Loss,
            emphasis: 0.6,
        };
        let loss_effect = enhancer.calculate_framing_effect(&loss_frame);
        assert!(loss_effect < 0.0);
        assert!(loss_effect.abs() > gain_effect); // Loss frame should have stronger effect
        
        let neutral_frame = FramingContext {
            frame_type: FrameType::Neutral,
            emphasis: 0.5,
        };
        let neutral_effect = enhancer.calculate_framing_effect(&neutral_frame);
        assert_eq!(neutral_effect, 0.0);
    }
}