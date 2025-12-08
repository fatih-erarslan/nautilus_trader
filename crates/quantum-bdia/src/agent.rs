//! BDIA Agent implementation

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;
use tracing::{info, debug};

use crate::{
    factors::{StandardFactors, MarketData, FactorWeights},
    prospect::{ProspectTheory, ProspectValue},
    DecisionType,
};

/// Belief state of an agent
#[derive(Debug, Clone)]
pub struct Belief {
    /// Beliefs about each market factor
    pub factor_beliefs: DashMap<StandardFactors, f64>,
    /// Confidence in beliefs
    pub confidence: f64,
    /// Timestamp of belief formation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Belief {
    /// Create new belief from market data
    pub fn from_market_data(market_data: &MarketData) -> Self {
        let factor_beliefs = DashMap::new();
        
        // Form beliefs based on market data with some noise/interpretation
        for factor in StandardFactors::all() {
            let raw_value = market_data.get_factor(factor);
            // Add small random variation to simulate belief formation
            let belief_value = raw_value + (rand::random::<f64>() - 0.5) * 0.1;
            factor_beliefs.insert(factor, belief_value.clamp(-1.0, 1.0));
        }
        
        // Confidence based on data quality/volatility
        let confidence = 1.0 - market_data.volatility * 0.5;
        
        Self {
            factor_beliefs,
            confidence: confidence.clamp(0.1, 1.0),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Get belief for a specific factor
    pub fn get_factor_belief(&self, factor: StandardFactors) -> f64 {
        self.factor_beliefs.get(&factor).map(|v| *v).unwrap_or(0.0)
    }
}

/// Desire state of an agent
#[derive(Debug, Clone)]
pub struct Desire {
    /// Intrinsic risk appetite/desire parameter
    pub intrinsic_desire: f64,
    /// Goal state (profit target)
    pub goal: f64,
    /// Risk tolerance
    pub risk_tolerance: f64,
}

impl Desire {
    /// Create new desire state
    pub fn new(intrinsic_desire: f64, goal: f64, risk_tolerance: f64) -> Self {
        Self {
            intrinsic_desire: intrinsic_desire.clamp(-1.0, 1.0),
            goal,
            risk_tolerance: risk_tolerance.clamp(0.0, 1.0),
        }
    }
}

/// Intention formed from beliefs and desires
#[derive(Debug, Clone)]
pub struct Intention {
    /// Raw intention signal
    pub signal: f64,
    /// Components of intention
    pub cadm_signal: f64,
    pub prospect_adjustment: f64,
    pub desire_component: f64,
    /// Reasoning trace
    pub reasoning: Vec<String>,
}

/// BDIA Agent that forms beliefs, has desires, creates intentions, and takes actions
pub struct BDIAAgent {
    /// Agent identifier
    pub name: String,
    /// Factor weights for CADM
    pub weights: Arc<RwLock<FactorWeights>>,
    /// Current belief state
    pub belief: Arc<RwLock<Belief>>,
    /// Desire state
    pub desire: Arc<RwLock<Desire>>,
    /// Prospect theory parameters
    pub prospect_theory: ProspectTheory,
    /// Performance history
    pub performance_history: Arc<RwLock<Vec<f64>>>,
    /// Configuration
    pub config: AgentConfig,
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum history size
    pub max_history: usize,
    /// Enable logging
    pub enable_logging: bool,
    /// Learning rate for updates
    pub learning_rate: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_history: 100,
            enable_logging: true,
            learning_rate: 0.01,
        }
    }
}

impl BDIAAgent {
    /// Create new agent
    pub fn new(name: String, weights: FactorWeights, desire: Desire) -> Self {
        Self::with_config(name, weights, desire, AgentConfig::default())
    }
    
    /// Create new agent with custom configuration
    pub fn with_config(
        name: String,
        weights: FactorWeights,
        desire: Desire,
        config: AgentConfig,
    ) -> Self {
        info!("Creating BDIA agent: {}", name);
        
        // Create initial belief (neutral)
        let belief = Belief {
            factor_beliefs: DashMap::new(),
            confidence: 0.5,
            timestamp: chrono::Utc::now(),
        };
        
        Self {
            name,
            weights: Arc::new(RwLock::new(weights)),
            belief: Arc::new(RwLock::new(belief)),
            desire: Arc::new(RwLock::new(desire)),
            prospect_theory: ProspectTheory::default(),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }
    
    /// Update beliefs based on market data
    pub fn update_beliefs(&self, market_data: &MarketData) {
        let new_belief = Belief::from_market_data(market_data);
        *self.belief.write() = new_belief;
        
        if self.config.enable_logging {
            debug!("[{}] Updated beliefs with confidence: {:.3}", 
                   self.name, self.belief.read().confidence);
        }
    }
    
    /// Compute intention from current beliefs and desires
    pub fn compute_intention(
        &self,
        expected_outcome: f64,
        probability: f64,
    ) -> Intention {
        let belief = self.belief.read();
        let weights = self.weights.read();
        let desire = self.desire.read();
        
        // Compute CADM signal (weighted sum of beliefs)
        let mut cadm_signal = 0.0;
        for factor in StandardFactors::all() {
            let belief_value = belief.get_factor_belief(factor);
            let weight = weights.get(factor);
            cadm_signal += belief_value * weight;
        }
        
        // Apply prospect theory adjustment
        let prospect_value = self.prospect_theory.prospect_value(expected_outcome, probability);
        let prospect_adjustment = prospect_value * desire.risk_tolerance;
        
        // Combine with intrinsic desire
        let signal = cadm_signal + prospect_adjustment + desire.intrinsic_desire;
        
        let mut reasoning = vec![
            format!("CADM signal: {:.3}", cadm_signal),
            format!("Prospect adjustment: {:.3}", prospect_adjustment),
            format!("Desire component: {:.3}", desire.intrinsic_desire),
            format!("Total intention: {:.3}", signal),
        ];
        
        if self.config.enable_logging {
            info!("[{}] Intention: {:.3} (CADM: {:.3}, Prospect: {:.3}, Desire: {:.3})",
                  self.name, signal, cadm_signal, prospect_adjustment, desire.intrinsic_desire);
        }
        
        Intention {
            signal,
            cadm_signal,
            prospect_adjustment,
            desire_component: desire.intrinsic_desire,
            reasoning,
        }
    }
    
    /// Make decision based on intention (without quantum fusion)
    pub fn decide_classical(&self, intention: &Intention) -> DecisionType {
        // Simple threshold-based decision
        if intention.signal > 0.5 {
            DecisionType::Buy
        } else if intention.signal < -0.5 {
            DecisionType::Sell
        } else if intention.signal > 0.2 {
            DecisionType::Increase(5)
        } else if intention.signal < -0.2 {
            DecisionType::Decrease(5)
        } else {
            DecisionType::Hold
        }
    }
    
    /// Update agent parameters based on feedback (cognitive reappraisal)
    pub fn cognitive_reappraisal(
        &self,
        market_data: &MarketData,
        predicted_return: f64,
        actual_return: f64,
    ) {
        let error = actual_return - predicted_return;
        let learning_rate = self.config.learning_rate;
        
        // Update desire based on performance
        {
            let mut desire = self.desire.write();
            desire.intrinsic_desire += learning_rate * error;
            desire.intrinsic_desire = desire.intrinsic_desire.clamp(-1.0, 1.0);
            
            // Adjust risk tolerance based on success
            if error > 0.0 {
                desire.risk_tolerance = (desire.risk_tolerance * 1.05).min(1.0);
            } else {
                desire.risk_tolerance = (desire.risk_tolerance * 0.95).max(0.1);
            }
        }
        
        // Update weights based on market feedback
        {
            let mut weights = self.weights.write();
            for factor in StandardFactors::all() {
                let signal = market_data.get_factor(factor);
                let update = learning_rate * error * signal;
                weights.update(factor, update);
            }
            // Don't normalize weights to preserve learning
        }
        
        // Update performance history
        {
            let mut history = self.performance_history.write();
            history.push(actual_return);
            if history.len() > self.config.max_history {
                history.remove(0);
            }
        }
        
        if self.config.enable_logging {
            info!("[{}] Cognitive reappraisal: error={:.3}, new desire={:.3}",
                  self.name, error, self.desire.read().intrinsic_desire);
        }
    }
    
    /// Get agent performance metrics
    pub fn get_performance(&self) -> AgentPerformance {
        let history = self.performance_history.read();
        
        let avg_return = if history.is_empty() {
            0.0
        } else {
            history.iter().sum::<f64>() / history.len() as f64
        };
        
        let volatility = if history.len() > 1 {
            let variance = history.iter()
                .map(|&r| (r - avg_return).powi(2))
                .sum::<f64>() / (history.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        
        AgentPerformance {
            name: self.name.clone(),
            avg_return,
            volatility,
            trades: history.len(),
            current_desire: self.desire.read().intrinsic_desire,
        }
    }
}

/// Agent performance metrics
#[derive(Debug, Clone)]
pub struct AgentPerformance {
    pub name: String,
    pub avg_return: f64,
    pub volatility: f64,
    pub trades: usize,
    pub current_desire: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_belief_formation() {
        let market_data = MarketData::random();
        let belief = Belief::from_market_data(&market_data);
        
        assert!(belief.confidence > 0.0 && belief.confidence <= 1.0);
        assert_eq!(belief.factor_beliefs.len(), StandardFactors::all().len());
    }
    
    #[test]
    fn test_agent_creation() {
        let weights = FactorWeights::default();
        let desire = Desire::new(0.5, 0.1, 0.7);
        let agent = BDIAAgent::new("test_agent".to_string(), weights, desire);
        
        assert_eq!(agent.name, "test_agent");
        assert_eq!(agent.desire.read().intrinsic_desire, 0.5);
    }
    
    #[test]
    fn test_intention_computation() {
        let weights = FactorWeights::default();
        let desire = Desire::new(0.3, 0.1, 0.7);
        let agent = BDIAAgent::new("test_agent".to_string(), weights, desire);
        
        let market_data = MarketData::random();
        agent.update_beliefs(&market_data);
        
        let intention = agent.compute_intention(0.05, 0.7);
        
        assert!(!intention.reasoning.is_empty());
        assert!(intention.signal.is_finite());
    }
    
    #[test]
    fn test_cognitive_reappraisal() {
        let weights = FactorWeights::default();
        let desire = Desire::new(0.0, 0.1, 0.5);
        let agent = BDIAAgent::new("test_agent".to_string(), weights, desire);
        
        let market_data = MarketData::random();
        let initial_desire = agent.desire.read().intrinsic_desire;
        
        // Positive feedback should increase desire
        agent.cognitive_reappraisal(&market_data, 0.02, 0.05);
        
        let new_desire = agent.desire.read().intrinsic_desire;
        assert!(new_desire > initial_desire);
        
        // Check performance history
        let perf = agent.get_performance();
        assert_eq!(perf.trades, 1);
        assert_eq!(perf.avg_return, 0.05);
    }
}