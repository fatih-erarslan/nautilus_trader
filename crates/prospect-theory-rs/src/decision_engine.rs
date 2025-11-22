//! Decision Engine for Prospect Theory
use crate::{QuantumProspectTheoryConfig, MarketData, TradingDecision, TradingAction, RiskAssessment, BehavioralFactors, AccountWeights, ProspectTheoryError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct DecisionEngine {
    config: QuantumProspectTheoryConfig,
    decision_history: Vec<TradingDecision>,
}

impl DecisionEngine {
    pub fn new(config: &QuantumProspectTheoryConfig) -> Result<Self> { 
        Ok(Self {
            config: config.clone(),
            decision_history: Vec::new(),
        })
    }
    
    pub fn make_decision(&mut self, buy: f64, sell: f64, hold: f64, weights: &AccountWeights, market: &MarketData) -> Result<TradingDecision> {
        // Apply mental accounting weights
        let weighted_buy = buy * weights.trading;
        let weighted_sell = sell * weights.investment;
        let weighted_hold = hold * weights.speculation;
        
        let action = if weighted_buy > weighted_sell && weighted_buy > weighted_hold {
            TradingAction::Buy { 
                quantity: 1.0, 
                max_price: market.current_price * 1.01 
            }
        } else if weighted_sell > weighted_hold {
            TradingAction::Sell { 
                quantity: 1.0, 
                min_price: market.current_price * 0.99 
            }
        } else {
            TradingAction::Hold
        };
        
        let prospect_value = weighted_buy.max(weighted_sell).max(weighted_hold);
        let confidence = (prospect_value.abs() + 0.5).min(1.0);
        
        let behavioral_factors = BehavioralFactors {
            loss_aversion_impact: if prospect_value < 0.0 { -0.2 } else { 0.1 },
            probability_weighting_bias: (prospect_value * 0.1).clamp(-0.5, 0.5),
            framing_effect: 0.0,
            mental_accounting_influence: (weights.trading - 1.0) * 0.1,
            reference_point_shift: 0.0,
        };
        
        let risk_assessment = RiskAssessment { 
            value_at_risk: market.current_price * 0.02,
            expected_shortfall: market.current_price * 0.03,
            loss_probability: (1.0 - confidence) * 0.5,
            maximum_loss: market.current_price * 0.1,
        };
        
        let decision = TradingDecision {
            action: action.clone(),
            confidence,
            prospect_value,
            reasoning: vec![
                format!("Prospect value: {:.3}", prospect_value),
                format!("Confidence: {:.3}", confidence),
                format!("Action: {:?}", action),
            ],
            risk_assessment,
            behavioral_factors,
            quantum_advantage: Some(0.15),
        };
        
        // Store in history
        self.decision_history.push(decision.clone());
        if self.decision_history.len() > 1000 {
            self.decision_history.remove(0);
        }
        
        Ok(decision)
    }
    
    pub fn get_history(&self) -> &[TradingDecision] {
        &self.decision_history
    }
    
    pub fn clear_history(&mut self) {
        self.decision_history.clear();
    }
}