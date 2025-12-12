//! Multi-agent BDIA network with consensus mechanisms

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use anyhow::Result;
use tracing::{info, debug, warn};

use crate::{
    agent::{BDIAAgent, AgentConfig, Desire, AgentPerformance},
    factors::{MarketData, FactorWeights},
    market_phase::MarketPhase,
    cognitive::CognitiveReappraisal,
    DecisionType,
};

/// Multi-agent BDIA network
pub struct BDIANetwork {
    /// Agents in the network
    agents: Vec<Arc<BDIAAgent>>,
    
    /// Agent performance weights for consensus
    agent_weights: Arc<RwLock<DashMap<String, f64>>>,
    
    /// Performance history for adaptive weighting
    performance_history: Arc<RwLock<DashMap<String, Vec<f64>>>>,
    
    /// Network-level decision history
    decision_history: Arc<RwLock<Vec<ConsensusDecision>>>,
    
    /// Configuration
    config: NetworkConfig,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Number of agents
    pub agent_count: usize,
    /// Enable adaptive agent weighting
    pub adaptive_weighting: bool,
    /// Maximum decision history
    pub max_history: usize,
    /// Diversity factor for agent initialization
    pub diversity_factor: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            agent_count: 5,
            adaptive_weighting: true,
            max_history: 1000,
            diversity_factor: 0.2,
        }
    }
}

/// Consensus decision from the network
#[derive(Debug, Clone)]
pub struct ConsensusDecision {
    /// Final decision
    pub decision: DecisionType,
    /// Weighted intention signal
    pub weighted_intention: f64,
    /// Individual agent decisions
    pub agent_decisions: Vec<(String, DecisionType, f64)>, // (name, decision, intention)
    /// Consensus confidence
    pub confidence: f64,
    /// Market phase at decision time
    pub market_phase: MarketPhase,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl BDIANetwork {
    /// Create new network with default agents
    pub fn new(config: NetworkConfig) -> Result<Self> {
        info!("Creating BDIA network with {} agents", config.agent_count);
        
        let mut agents = Vec::new();
        let agent_weights = Arc::new(RwLock::new(DashMap::new()));
        let performance_history = Arc::new(RwLock::new(DashMap::new()));
        
        // Create diverse agents
        for i in 0..config.agent_count {
            let agent = Self::create_agent(i, &config)?;
            
            // Initialize equal weights
            agent_weights.write().insert(
                agent.name.clone(),
                1.0 / config.agent_count as f64
            );
            
            // Initialize empty performance history
            performance_history.write().insert(
                agent.name.clone(),
                Vec::new()
            );
            
            agents.push(Arc::new(agent));
        }
        
        Ok(Self {
            agents,
            agent_weights,
            performance_history,
            decision_history: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }
    
    /// Create an agent with variation
    fn create_agent(index: usize, config: &NetworkConfig) -> Result<BDIAAgent> {
        let name = format!("Agent_{}", index);
        
        // Create weights with variation
        let mut weights = FactorWeights::default();
        
        // Add random variation to weights
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for factor in crate::factors::StandardFactors::all() {
            let base_weight = weights.get(factor);
            let variation = (rng.gen::<f64>() - 0.5) * config.diversity_factor;
            weights.set(factor, (base_weight + variation).max(0.0));
        }
        
        // Create desire with variation
        let intrinsic_desire = (index as f64 / config.agent_count as f64) * 2.0 - 1.0; // Range [-1, 1]
        let goal = 0.1 + rng.gen::<f64>() * 0.1; // Goal between 0.1 and 0.2
        let risk_tolerance = 0.3 + rng.gen::<f64>() * 0.4; // Risk tolerance between 0.3 and 0.7
        
        let desire = Desire::new(intrinsic_desire, goal, risk_tolerance);
        
        // Create agent config
        let agent_config = AgentConfig {
            max_history: 100,
            enable_logging: false, // Disable individual agent logging for network
            learning_rate: 0.01,
        };
        
        Ok(BDIAAgent::with_config(name, weights, desire, agent_config))
    }
    
    /// Make consensus decision
    pub async fn consensus_decision(
        &self,
        market_data: &MarketData,
        market_phase: MarketPhase,
    ) -> Result<ConsensusDecision> {
        debug!("Computing network consensus decision");
        
        // Update all agent beliefs
        for agent in &self.agents {
            agent.update_beliefs(market_data);
        }
        
        // Collect agent decisions and intentions
        let mut agent_decisions = Vec::new();
        let mut weighted_intention = 0.0;
        let agent_weights = self.agent_weights.read();
        
        for agent in &self.agents {
            // Compute intention with phase-adjusted parameters
            let (expected_outcome, probability) = self.phase_adjusted_expectations(market_phase);
            let intention = agent.compute_intention(expected_outcome, probability);
            
            // Get agent's classical decision
            let decision = agent.decide_classical(&intention);
            
            // Get agent weight
            let weight = agent_weights.get(&agent.name)
                .copied()
                .unwrap_or(1.0 / self.agents.len() as f64);
            
            // Add to weighted intention
            weighted_intention += intention.signal * weight;
            
            agent_decisions.push((agent.name.clone(), decision, intention.signal));
        }
        
        // Determine consensus decision based on weighted intention
        let consensus = self.determine_consensus(weighted_intention, &agent_decisions);
        
        // Calculate confidence
        let confidence = self.calculate_confidence(&agent_decisions, consensus);
        
        let decision = ConsensusDecision {
            decision: consensus,
            weighted_intention,
            agent_decisions,
            confidence,
            market_phase,
            timestamp: chrono::Utc::now(),
        };
        
        // Store in history
        {
            let mut history = self.decision_history.write();
            history.push(decision.clone());
            if history.len() > self.config.max_history {
                history.remove(0);
            }
        }
        
        info!("Network consensus: {:?} (confidence: {:.3})", consensus, confidence);
        
        Ok(decision)
    }
    
    /// Get phase-adjusted expected outcome and probability
    fn phase_adjusted_expectations(&self, phase: MarketPhase) -> (f64, f64) {
        match phase {
            MarketPhase::Growth => (0.05, 0.7),      // Positive expectation, high probability
            MarketPhase::Conservation => (0.02, 0.6), // Small positive, moderate probability
            MarketPhase::Release => (-0.05, 0.6),     // Negative expectation
            MarketPhase::Reorganization => (0.03, 0.5), // Uncertain
        }
    }
    
    /// Determine consensus from weighted intention
    fn determine_consensus(
        &self,
        weighted_intention: f64,
        agent_decisions: &[(String, DecisionType, f64)],
    ) -> DecisionType {
        // Primary decision based on weighted intention
        let primary = if weighted_intention > 0.6 {
            DecisionType::Buy
        } else if weighted_intention < -0.6 {
            DecisionType::Sell
        } else if weighted_intention > 0.3 {
            DecisionType::Increase(10)
        } else if weighted_intention < -0.3 {
            DecisionType::Decrease(10)
        } else if weighted_intention.abs() < 0.1 {
            DecisionType::Hold
        } else if weighted_intention > 0.0 {
            DecisionType::Hedge
        } else {
            DecisionType::Exit
        };
        
        // Count agent votes as tiebreaker
        use std::collections::HashMap;
        let mut vote_counts: HashMap<DecisionType, usize> = HashMap::new();
        
        for (_, decision, _) in agent_decisions {
            *vote_counts.entry(*decision).or_insert(0) += 1;
        }
        
        // If primary decision has support, use it
        if vote_counts.get(&primary).copied().unwrap_or(0) > 0 {
            primary
        } else {
            // Otherwise use most common decision
            vote_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(decision, _)| decision)
                .unwrap_or(DecisionType::Hold)
        }
    }
    
    /// Calculate consensus confidence
    fn calculate_confidence(
        &self,
        agent_decisions: &[(String, DecisionType, f64)],
        consensus: DecisionType,
    ) -> f64 {
        let total_agents = agent_decisions.len() as f64;
        
        // Count agents agreeing with consensus
        let agreeing_agents = agent_decisions
            .iter()
            .filter(|(_, decision, _)| *decision == consensus)
            .count() as f64;
        
        // Calculate intention variance
        let intentions: Vec<f64> = agent_decisions
            .iter()
            .map(|(_, _, intention)| *intention)
            .collect();
        
        let mean_intention = intentions.iter().sum::<f64>() / intentions.len() as f64;
        let variance = intentions
            .iter()
            .map(|&i| (i - mean_intention).powi(2))
            .sum::<f64>() / intentions.len() as f64;
        
        // Confidence based on agreement and low variance
        let agreement_score = agreeing_agents / total_agents;
        let coherence_score = 1.0 / (1.0 + variance);
        
        (agreement_score * 0.7 + coherence_score * 0.3).clamp(0.0, 1.0)
    }
    
    /// Update all agents based on feedback
    pub async fn update_all_agents(
        &mut self,
        market_data: &MarketData,
        predicted_return: f64,
        actual_return: f64,
        cognitive_engine: &CognitiveReappraisal,
    ) -> Result<()> {
        info!("Updating network agents with feedback");
        
        // Calculate performance metric
        let error = (predicted_return - actual_return).abs();
        let performance = 1.0 / (1.0 + error); // Higher is better
        
        // Update each agent
        for agent in &self.agents {
            // Use cognitive engine for sophisticated updates
            let learning_rate = cognitive_engine.adaptive_learning_rate(
                &agent.performance_history.read()
            );
            
            // Update agent
            agent.cognitive_reappraisal(market_data, predicted_return, actual_return);
            
            // Update performance history
            self.performance_history
                .write()
                .get_mut(&agent.name)
                .map(|history| {
                    history.push(performance);
                    if history.len() > 100 {
                        history.remove(0);
                    }
                });
        }
        
        // Update agent weights if adaptive weighting is enabled
        if self.config.adaptive_weighting {
            self.update_agent_weights();
        }
        
        Ok(())
    }
    
    /// Update agent weights based on performance
    fn update_agent_weights(&self) {
        let performance_history = self.performance_history.read();
        let mut new_weights = DashMap::new();
        
        // Calculate average performance for each agent
        let mut avg_performances = Vec::new();
        
        for agent in &self.agents {
            if let Some(history) = performance_history.get(&agent.name) {
                if !history.is_empty() {
                    let recent = history.iter().rev().take(20).copied().collect::<Vec<_>>();
                    let avg_perf = recent.iter().sum::<f64>() / recent.len() as f64;
                    avg_performances.push((agent.name.clone(), avg_perf));
                } else {
                    avg_performances.push((agent.name.clone(), 0.5));
                }
            } else {
                avg_performances.push((agent.name.clone(), 0.5));
            }
        }
        
        // Normalize to weights summing to 1.0
        let total_perf: f64 = avg_performances
            .iter()
            .map(|(_, p)| p.max(0.1)) // Minimum weight
            .sum();
        
        for (name, perf) in avg_performances {
            let weight = perf.max(0.1) / total_perf;
            new_weights.insert(name, weight);
        }
        
        // Update weights
        *self.agent_weights.write() = new_weights;
        
        debug!("Updated agent weights based on performance");
    }
    
    /// Get network state
    pub fn get_network_state(&self) -> NetworkState {
        let agent_weights = self.agent_weights.read();
        let decision_history = self.decision_history.read();
        
        let agent_states: Vec<AgentState> = self.agents
            .iter()
            .map(|agent| {
                let perf = agent.get_performance();
                let weight = agent_weights
                    .get(&agent.name)
                    .copied()
                    .unwrap_or(0.0);
                
                AgentState {
                    name: agent.name.clone(),
                    weight,
                    performance: perf,
                }
            })
            .collect();
        
        let recent_decisions: Vec<DecisionType> = decision_history
            .iter()
            .rev()
            .take(10)
            .map(|d| d.decision)
            .collect();
        
        NetworkState {
            agent_count: self.agents.len(),
            agent_states,
            recent_decisions,
            total_decisions: decision_history.len(),
        }
    }
    
    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

/// Network state information
#[derive(Debug, Clone)]
pub struct NetworkState {
    pub agent_count: usize,
    pub agent_states: Vec<AgentState>,
    pub recent_decisions: Vec<DecisionType>,
    pub total_decisions: usize,
}

/// Individual agent state
#[derive(Debug, Clone)]
pub struct AgentState {
    pub name: String,
    pub weight: f64,
    pub performance: AgentPerformance,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_network_creation() {
        let config = NetworkConfig {
            agent_count: 3,
            ..Default::default()
        };
        
        let network = BDIANetwork::new(config).unwrap();
        assert_eq!(network.agents.len(), 3);
        
        // Check agent weights sum to 1
        let weights = network.agent_weights.read();
        let sum: f64 = weights.iter().map(|e| *e.value()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[tokio::test]
    async fn test_consensus_decision() {
        let network = BDIANetwork::new(NetworkConfig::default()).unwrap();
        let market_data = MarketData::random();
        
        let decision = network.consensus_decision(&market_data, MarketPhase::Growth)
            .await
            .unwrap();
        
        assert!(!decision.agent_decisions.is_empty());
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    }
    
    #[tokio::test]
    async fn test_agent_updates() {
        let mut network = BDIANetwork::new(NetworkConfig::default()).unwrap();
        let market_data = MarketData::random();
        let cognitive_engine = CognitiveReappraisal::default();
        
        // Make initial decision
        let decision = network.consensus_decision(&market_data, MarketPhase::Growth)
            .await
            .unwrap();
        
        // Update with feedback
        network.update_all_agents(&market_data, 0.02, 0.03, &cognitive_engine)
            .await
            .unwrap();
        
        // Check performance history was updated
        let perf_history = network.performance_history.read();
        for agent in &network.agents {
            assert!(perf_history.get(&agent.name).unwrap().len() > 0);
        }
    }
}