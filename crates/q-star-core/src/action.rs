//! Q* Action Space Definition
//! 
//! Defines the action space for Q* algorithm optimized for trading decisions
//! with precise risk management and execution parameters.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::QStarError;

/// Q* action enumeration for trading decisions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QStarAction {
    /// Hold current position
    Hold,
    
    /// Buy with specified amount (0.0 to 1.0 of available capital)
    Buy { amount: f64 },
    
    /// Sell with specified amount (0.0 to 1.0 of current position)
    Sell { amount: f64 },
    
    /// Set stop loss at threshold (percentage from current price)
    StopLoss { threshold: f64 },
    
    /// Set take profit at threshold (percentage from current price)
    TakeProfit { threshold: f64 },
    
    /// Close all positions immediately
    CloseAll,
    
    /// Rebalance portfolio with new weights
    Rebalance { weights: Vec<f64> },
    
    /// Scale position size up or down
    Scale { factor: f64 },
    
    /// Hedge position with opposite trade
    Hedge { ratio: f64 },
    
    /// Wait for better entry (skip this time step)
    Wait,
}

impl Hash for QStarAction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Create deterministic hash for action matching
        match self {
            QStarAction::Hold => {
                0u8.hash(state);
            }
            QStarAction::Buy { amount } => {
                1u8.hash(state);
                ((amount * 1000.0) as u32).hash(state);
            }
            QStarAction::Sell { amount } => {
                2u8.hash(state);
                ((amount * 1000.0) as u32).hash(state);
            }
            QStarAction::StopLoss { threshold } => {
                3u8.hash(state);
                ((threshold * 10000.0) as u32).hash(state);
            }
            QStarAction::TakeProfit { threshold } => {
                4u8.hash(state);
                ((threshold * 10000.0) as u32).hash(state);
            }
            QStarAction::CloseAll => {
                5u8.hash(state);
            }
            QStarAction::Rebalance { weights } => {
                6u8.hash(state);
                for weight in weights {
                    ((weight * 1000.0) as u32).hash(state);
                }
            }
            QStarAction::Scale { factor } => {
                7u8.hash(state);
                ((factor * 1000.0) as u32).hash(state);
            }
            QStarAction::Hedge { ratio } => {
                8u8.hash(state);
                ((ratio * 1000.0) as u32).hash(state);
            }
            QStarAction::Wait => {
                9u8.hash(state);
            }
        }
    }
}

impl fmt::Display for QStarAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QStarAction::Hold => write!(f, "HOLD"),
            QStarAction::Buy { amount } => write!(f, "BUY({:.2})", amount),
            QStarAction::Sell { amount } => write!(f, "SELL({:.2})", amount),
            QStarAction::StopLoss { threshold } => write!(f, "STOP_LOSS({:.2}%)", threshold * 100.0),
            QStarAction::TakeProfit { threshold } => write!(f, "TAKE_PROFIT({:.2}%)", threshold * 100.0),
            QStarAction::CloseAll => write!(f, "CLOSE_ALL"),
            QStarAction::Rebalance { weights } => {
                write!(f, "REBALANCE([{}])", 
                    weights.iter().map(|w| format!("{:.2}", w))
                           .collect::<Vec<_>>().join(", "))
            }
            QStarAction::Scale { factor } => write!(f, "SCALE({:.2}x)", factor),
            QStarAction::Hedge { ratio } => write!(f, "HEDGE({:.2})", ratio),
            QStarAction::Wait => write!(f, "WAIT"),
        }
    }
}

impl QStarAction {
    /// Validate action parameters
    pub fn validate(&self) -> Result<(), QStarError> {
        match self {
            QStarAction::Buy { amount } => {
                if *amount < 0.0 || *amount > 1.0 {
                    return Err(QStarError::ActionError(
                        format!("Buy amount {} must be between 0.0 and 1.0", amount)
                    ));
                }
            }
            QStarAction::Sell { amount } => {
                if *amount < 0.0 || *amount > 1.0 {
                    return Err(QStarError::ActionError(
                        format!("Sell amount {} must be between 0.0 and 1.0", amount)
                    ));
                }
            }
            QStarAction::StopLoss { threshold } => {
                if *threshold < 0.0 || *threshold > 0.5 {
                    return Err(QStarError::ActionError(
                        format!("Stop loss threshold {} must be between 0.0 and 0.5", threshold)
                    ));
                }
            }
            QStarAction::TakeProfit { threshold } => {
                if *threshold < 0.0 || *threshold > 2.0 {
                    return Err(QStarError::ActionError(
                        format!("Take profit threshold {} must be between 0.0 and 2.0", threshold)
                    ));
                }
            }
            QStarAction::Rebalance { weights } => {
                let sum: f64 = weights.iter().sum();
                if (sum - 1.0).abs() > 0.01 {
                    return Err(QStarError::ActionError(
                        format!("Rebalance weights sum to {}, must sum to 1.0", sum)
                    ));
                }
                if weights.iter().any(|&w| w < 0.0 || w > 1.0) {
                    return Err(QStarError::ActionError(
                        "All weights must be between 0.0 and 1.0".to_string()
                    ));
                }
            }
            QStarAction::Scale { factor } => {
                if *factor < 0.1 || *factor > 10.0 {
                    return Err(QStarError::ActionError(
                        format!("Scale factor {} must be between 0.1 and 10.0", factor)
                    ));
                }
            }
            QStarAction::Hedge { ratio } => {
                if *ratio < 0.0 || *ratio > 1.0 {
                    return Err(QStarError::ActionError(
                        format!("Hedge ratio {} must be between 0.0 and 1.0", ratio)
                    ));
                }
            }
            _ => {} // No validation needed for Hold, CloseAll, Wait
        }
        Ok(())
    }
    
    /// Get action category for grouping similar actions
    pub fn category(&self) -> ActionCategory {
        match self {
            QStarAction::Hold | QStarAction::Wait => ActionCategory::Passive,
            QStarAction::Buy { .. } => ActionCategory::Entry,
            QStarAction::Sell { .. } => ActionCategory::Exit,
            QStarAction::StopLoss { .. } | QStarAction::TakeProfit { .. } => ActionCategory::RiskManagement,
            QStarAction::CloseAll => ActionCategory::Emergency,
            QStarAction::Rebalance { .. } => ActionCategory::Optimization,
            QStarAction::Scale { .. } | QStarAction::Hedge { .. } => ActionCategory::Adjustment,
        }
    }
    
    /// Get risk level of action (0.0 = no risk, 1.0 = high risk)
    pub fn risk_level(&self) -> f64 {
        match self {
            QStarAction::Hold | QStarAction::Wait => 0.0,
            QStarAction::StopLoss { .. } => 0.1,
            QStarAction::TakeProfit { .. } => 0.2,
            QStarAction::Buy { amount } | QStarAction::Sell { amount } => *amount * 0.5,
            QStarAction::Scale { factor } => {
                if *factor > 1.0 {
                    (*factor - 1.0) * 0.3
                } else {
                    (1.0 - *factor) * 0.2
                }
            }
            QStarAction::Hedge { ratio } => *ratio * 0.3,
            QStarAction::Rebalance { weights } => {
                // Risk based on portfolio concentration
                let max_weight = weights.iter().fold(0.0_f64, |a, &b| a.max(b));
                max_weight * 0.4
            }
            QStarAction::CloseAll => 0.8, // High risk action
        }
    }
    
    /// Get expected impact on portfolio (positive = increase value, negative = decrease)
    pub fn expected_impact(&self) -> f64 {
        match self {
            QStarAction::Hold | QStarAction::Wait => 0.0,
            QStarAction::Buy { amount } => *amount * 0.1, // Expected positive return
            QStarAction::Sell { amount } => -*amount * 0.05, // Opportunity cost
            QStarAction::StopLoss { threshold } => -*threshold, // Realized loss
            QStarAction::TakeProfit { threshold } => *threshold * 0.8, // Realized gain
            QStarAction::CloseAll => -0.02, // Transaction costs
            QStarAction::Rebalance { .. } => 0.01, // Slight efficiency gain
            QStarAction::Scale { factor } => (*factor - 1.0) * 0.05,
            QStarAction::Hedge { ratio } => -*ratio * 0.01, // Hedging cost
        }
    }
    
    /// Check if action requires margin/leverage
    pub fn requires_margin(&self) -> bool {
        match self {
            QStarAction::Buy { amount } => *amount > 0.9,
            QStarAction::Scale { factor } => *factor > 2.0,
            QStarAction::Hedge { .. } => true,
            _ => false,
        }
    }
    
    /// Get execution priority (higher = more urgent)
    pub fn execution_priority(&self) -> u8 {
        match self {
            QStarAction::StopLoss { .. } => 10, // Highest priority
            QStarAction::CloseAll => 9,
            QStarAction::TakeProfit { .. } => 8,
            QStarAction::Sell { .. } => 7,
            QStarAction::Buy { .. } => 6,
            QStarAction::Scale { .. } => 5,
            QStarAction::Hedge { .. } => 4,
            QStarAction::Rebalance { .. } => 3,
            QStarAction::Hold => 2,
            QStarAction::Wait => 1, // Lowest priority
        }
    }
    
    /// Get estimated execution time in microseconds
    pub fn execution_time_us(&self) -> u64 {
        match self {
            QStarAction::Hold | QStarAction::Wait => 1,
            QStarAction::Buy { .. } | QStarAction::Sell { .. } => 50,
            QStarAction::StopLoss { .. } | QStarAction::TakeProfit { .. } => 30,
            QStarAction::CloseAll => 100,
            QStarAction::Scale { .. } => 40,
            QStarAction::Hedge { .. } => 80,
            QStarAction::Rebalance { .. } => 200,
        }
    }
    
    /// Convert action to feature vector for neural networks
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let mut features = vec![0.0; 16]; // Base feature vector size
        
        match self {
            QStarAction::Hold => features[0] = 1.0,
            QStarAction::Buy { amount } => {
                features[1] = 1.0;
                features[10] = *amount;
            }
            QStarAction::Sell { amount } => {
                features[2] = 1.0;
                features[10] = *amount;
            }
            QStarAction::StopLoss { threshold } => {
                features[3] = 1.0;
                features[11] = *threshold;
            }
            QStarAction::TakeProfit { threshold } => {
                features[4] = 1.0;
                features[11] = *threshold;
            }
            QStarAction::CloseAll => features[5] = 1.0,
            QStarAction::Rebalance { weights } => {
                features[6] = 1.0;
                // Use first few weights (limited by feature vector size)
                for (i, &weight) in weights.iter().take(4).enumerate() {
                    features[12 + i] = weight;
                }
            }
            QStarAction::Scale { factor } => {
                features[7] = 1.0;
                features[10] = *factor;
            }
            QStarAction::Hedge { ratio } => {
                features[8] = 1.0;
                features[11] = *ratio;
            }
            QStarAction::Wait => features[9] = 1.0,
        }
        
        features
    }
    
    /// Create action from feature vector
    pub fn from_feature_vector(features: &[f64]) -> Result<Self, QStarError> {
        if features.len() < 16 {
            return Err(QStarError::ActionError(
                "Insufficient features for action reconstruction".to_string()
            ));
        }
        
        // Find the action type (one-hot encoded in first 10 positions)
        let action_type = features[0..10].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| QStarError::ActionError("No action type found".to_string()))?;
        
        let action = match action_type {
            0 => QStarAction::Hold,
            1 => QStarAction::Buy { amount: features[10] },
            2 => QStarAction::Sell { amount: features[10] },
            3 => QStarAction::StopLoss { threshold: features[11] },
            4 => QStarAction::TakeProfit { threshold: features[11] },
            5 => QStarAction::CloseAll,
            6 => {
                let weights = features[12..16].to_vec();
                QStarAction::Rebalance { weights }
            }
            7 => QStarAction::Scale { factor: features[10] },
            8 => QStarAction::Hedge { ratio: features[11] },
            9 => QStarAction::Wait,
            _ => return Err(QStarError::ActionError(
                format!("Invalid action type index: {}", action_type)
            )),
        };
        
        action.validate()?;
        Ok(action)
    }
}

/// Action category for grouping and analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionCategory {
    /// Passive actions (hold, wait)
    Passive,
    
    /// Entry actions (buy)
    Entry,
    
    /// Exit actions (sell)
    Exit,
    
    /// Risk management (stop loss, take profit)
    RiskManagement,
    
    /// Emergency actions (close all)
    Emergency,
    
    /// Portfolio optimization (rebalance)
    Optimization,
    
    /// Position adjustment (scale, hedge)
    Adjustment,
}

/// Action space configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSpaceConfig {
    /// Allowed buy amounts
    pub buy_amounts: Vec<f64>,
    
    /// Allowed sell amounts
    pub sell_amounts: Vec<f64>,
    
    /// Stop loss thresholds
    pub stop_loss_thresholds: Vec<f64>,
    
    /// Take profit thresholds
    pub take_profit_thresholds: Vec<f64>,
    
    /// Scale factors
    pub scale_factors: Vec<f64>,
    
    /// Hedge ratios
    pub hedge_ratios: Vec<f64>,
    
    /// Enable advanced actions
    pub enable_rebalance: bool,
    pub enable_hedging: bool,
    pub enable_scaling: bool,
}

impl Default for ActionSpaceConfig {
    fn default() -> Self {
        Self {
            buy_amounts: vec![0.1, 0.25, 0.5, 1.0],
            sell_amounts: vec![0.1, 0.25, 0.5, 1.0],
            stop_loss_thresholds: vec![0.01, 0.02, 0.05, 0.1],
            take_profit_thresholds: vec![0.02, 0.05, 0.1, 0.2],
            scale_factors: vec![0.5, 1.5, 2.0],
            hedge_ratios: vec![0.25, 0.5, 0.75],
            enable_rebalance: true,
            enable_hedging: true,
            enable_scaling: true,
        }
    }
}

impl ActionSpaceConfig {
    /// Generate all possible actions based on configuration
    pub fn generate_action_space(&self) -> Vec<QStarAction> {
        let mut actions = Vec::new();
        
        // Basic actions
        actions.push(QStarAction::Hold);
        actions.push(QStarAction::Wait);
        actions.push(QStarAction::CloseAll);
        
        // Buy actions
        for &amount in &self.buy_amounts {
            actions.push(QStarAction::Buy { amount });
        }
        
        // Sell actions
        for &amount in &self.sell_amounts {
            actions.push(QStarAction::Sell { amount });
        }
        
        // Stop loss actions
        for &threshold in &self.stop_loss_thresholds {
            actions.push(QStarAction::StopLoss { threshold });
        }
        
        // Take profit actions
        for &threshold in &self.take_profit_thresholds {
            actions.push(QStarAction::TakeProfit { threshold });
        }
        
        // Scale actions (if enabled)
        if self.enable_scaling {
            for &factor in &self.scale_factors {
                actions.push(QStarAction::Scale { factor });
            }
        }
        
        // Hedge actions (if enabled)
        if self.enable_hedging {
            for &ratio in &self.hedge_ratios {
                actions.push(QStarAction::Hedge { ratio });
            }
        }
        
        // Rebalance actions (if enabled)
        if self.enable_rebalance {
            // Add some common rebalancing patterns
            actions.push(QStarAction::Rebalance { weights: vec![1.0] });
            actions.push(QStarAction::Rebalance { weights: vec![0.6, 0.4] });
            actions.push(QStarAction::Rebalance { weights: vec![0.5, 0.3, 0.2] });
        }
        
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_action_validation() {
        // Valid actions
        assert!(QStarAction::Buy { amount: 0.5 }.validate().is_ok());
        assert!(QStarAction::Sell { amount: 0.3 }.validate().is_ok());
        assert!(QStarAction::StopLoss { threshold: 0.05 }.validate().is_ok());
        
        // Invalid actions
        assert!(QStarAction::Buy { amount: 1.5 }.validate().is_err());
        assert!(QStarAction::Sell { amount: -0.1 }.validate().is_err());
        assert!(QStarAction::StopLoss { threshold: 0.6 }.validate().is_err());
    }
    
    #[test]
    fn test_action_categories() {
        assert_eq!(QStarAction::Hold.category(), ActionCategory::Passive);
        assert_eq!(QStarAction::Buy { amount: 0.5 }.category(), ActionCategory::Entry);
        assert_eq!(QStarAction::StopLoss { threshold: 0.05 }.category(), ActionCategory::RiskManagement);
    }
    
    #[test]
    fn test_action_risk_levels() {
        assert_eq!(QStarAction::Hold.risk_level(), 0.0);
        assert!(QStarAction::Buy { amount: 1.0 }.risk_level() > 0.0);
        assert!(QStarAction::CloseAll.risk_level() > 0.5);
    }
    
    #[test]
    fn test_feature_vector_conversion() {
        let action = QStarAction::Buy { amount: 0.75 };
        let features = action.to_feature_vector();
        
        assert_eq!(features.len(), 16);
        assert_eq!(features[1], 1.0); // Buy action indicator
        assert_eq!(features[10], 0.75); // Amount
        
        // Test round-trip conversion
        let reconstructed = QStarAction::from_feature_vector(&features).unwrap();
        match reconstructed {
            QStarAction::Buy { amount } => assert!((amount - 0.75).abs() < 1e-10),
            _ => panic!("Wrong action type reconstructed"),
        }
    }
    
    #[test]
    fn test_action_space_generation() {
        let config = ActionSpaceConfig::default();
        let actions = config.generate_action_space();
        
        assert!(!actions.is_empty());
        assert!(actions.contains(&QStarAction::Hold));
        assert!(actions.iter().any(|a| matches!(a, QStarAction::Buy { .. })));
    }
    
    #[test]
    fn test_execution_priority() {
        assert!(QStarAction::StopLoss { threshold: 0.05 }.execution_priority() > 
                QStarAction::Buy { amount: 0.5 }.execution_priority());
        assert!(QStarAction::CloseAll.execution_priority() > 
                QStarAction::Hold.execution_priority());
    }
}