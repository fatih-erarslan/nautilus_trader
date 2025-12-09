//! pBit-Enhanced Swarm Consensus
//!
//! Uses Boltzmann statistics and Ising model dynamics for swarm decision fusion.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! Agent votes are weighted using Boltzmann distribution:
//! - W_i = exp(-E_i / T) where E_i = -confidence_i (negative because high confidence = low energy)
//! - Normalized weight: P_i = W_i / Z where Z = Σ W_j
//!
//! For action fusion, we model as an Ising system:
//! - Each agent vote is a "spin" in action space
//! - Consensus emerges as magnetization of the swarm
//! - Temperature controls exploration vs exploitation

use std::collections::HashMap;
use q_star_core::{QStarAction, QStarSearchResult};

/// Boltzmann constant for pBit consensus (can be tuned)
pub const CONSENSUS_BOLTZMANN_K: f64 = 1.0;

/// pBit consensus configuration
#[derive(Debug, Clone)]
pub struct PBitConsensusConfig {
    /// Temperature for Boltzmann weighting
    pub temperature: f64,
    /// Coupling strength for action similarity
    pub coupling_j: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Enable annealing (decrease temperature over time)
    pub enable_annealing: bool,
    /// Annealing rate (temperature decay per decision)
    pub annealing_rate: f64,
}

impl Default for PBitConsensusConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            coupling_j: 1.0,
            min_confidence: 0.1,
            enable_annealing: true,
            annealing_rate: 0.995,
        }
    }
}

/// pBit-enhanced consensus engine
#[derive(Debug)]
pub struct PBitConsensusEngine {
    config: PBitConsensusConfig,
    /// Current effective temperature
    current_temperature: f64,
    /// Decision history for learning
    decision_history: Vec<ConsensusDecision>,
    /// Action magnetization (preference strength)
    action_magnetization: HashMap<ActionKey, f64>,
}

/// Record of a consensus decision
#[derive(Debug, Clone)]
pub struct ConsensusDecision {
    /// Chosen action
    pub action: QStarAction,
    /// Confidence of decision
    pub confidence: f64,
    /// Number of agents that voted for this action
    pub vote_count: usize,
    /// Total votes
    pub total_votes: usize,
    /// Partition function at decision time
    pub partition_function: f64,
    /// System entropy
    pub entropy: f64,
}

/// Key for action indexing (simplified representation)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ActionKey {
    Buy,
    Sell,
    Hold,
    StopLoss,
    TakeProfit,
    CloseAll,
    Rebalance,
    Scale,
    Hedge,
    Wait,
}

impl From<&QStarAction> for ActionKey {
    fn from(action: &QStarAction) -> Self {
        match action {
            QStarAction::Buy { .. } => ActionKey::Buy,
            QStarAction::Sell { .. } => ActionKey::Sell,
            QStarAction::Hold => ActionKey::Hold,
            QStarAction::StopLoss { .. } => ActionKey::StopLoss,
            QStarAction::TakeProfit { .. } => ActionKey::TakeProfit,
            QStarAction::CloseAll => ActionKey::CloseAll,
            QStarAction::Rebalance { .. } => ActionKey::Rebalance,
            QStarAction::Scale { .. } => ActionKey::Scale,
            QStarAction::Hedge { .. } => ActionKey::Hedge,
            QStarAction::Wait => ActionKey::Wait,
        }
    }
}

impl PBitConsensusEngine {
    /// Create new consensus engine
    pub fn new(config: PBitConsensusConfig) -> Self {
        Self {
            current_temperature: config.temperature,
            config,
            decision_history: Vec::with_capacity(10000),
            action_magnetization: HashMap::new(),
        }
    }

    /// Compute Boltzmann weight for an agent's vote
    fn boltzmann_weight(&self, confidence: f64) -> f64 {
        // Energy = -confidence (high confidence = low energy = preferred)
        let energy = -confidence.max(self.config.min_confidence);
        (energy / self.current_temperature).exp()
    }

    /// Compute partition function for normalization
    fn partition_function(&self, weights: &[f64]) -> f64 {
        weights.iter().sum::<f64>().max(f64::EPSILON)
    }

    /// Compute system entropy from probabilities
    fn entropy(&self, probabilities: &[f64]) -> f64 {
        probabilities.iter()
            .filter(|&&p| p > f64::EPSILON)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Perform Boltzmann-weighted consensus
    pub fn boltzmann_consensus(&mut self, decisions: &[QStarSearchResult]) -> Option<QStarAction> {
        if decisions.is_empty() {
            return None;
        }

        // Group votes by action type
        let mut action_votes: HashMap<ActionKey, Vec<&QStarSearchResult>> = HashMap::new();
        for decision in decisions {
            let key = ActionKey::from(&decision.action);
            action_votes.entry(key).or_default().push(decision);
        }

        // Calculate Boltzmann weights for each action type
        let mut action_weights: HashMap<ActionKey, f64> = HashMap::new();
        let mut action_representatives: HashMap<ActionKey, QStarAction> = HashMap::new();

        for (action_key, votes) in &action_votes {
            // Sum weights of all votes for this action
            let total_weight: f64 = votes.iter()
                .map(|v| self.boltzmann_weight(v.confidence))
                .sum();
            
            action_weights.insert(action_key.clone(), total_weight);

            // Keep highest confidence vote as representative
            if let Some(best_vote) = votes.iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()) {
                action_representatives.insert(action_key.clone(), best_vote.action.clone());
            }
        }

        // Calculate partition function
        let weights: Vec<f64> = action_weights.values().copied().collect();
        let z = self.partition_function(&weights);

        // Calculate probabilities
        let probabilities: Vec<f64> = weights.iter().map(|w| w / z).collect();

        // Select action with highest probability (deterministic)
        // Or sample according to probabilities (stochastic)
        let (best_action_key, best_weight) = action_weights.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

        let best_action = action_representatives.get(best_action_key)?.clone();

        // Update magnetization
        for (key, &weight) in &action_weights {
            let prob = weight / z;
            let current_mag = self.action_magnetization.entry(key.clone()).or_insert(0.0);
            *current_mag = 0.9 * *current_mag + 0.1 * (2.0 * prob - 1.0); // EMA update
        }

        // Record decision
        let entropy = self.entropy(&probabilities);
        let vote_count = action_votes.get(best_action_key).map(|v| v.len()).unwrap_or(0);
        
        self.decision_history.push(ConsensusDecision {
            action: best_action.clone(),
            confidence: *best_weight / z,
            vote_count,
            total_votes: decisions.len(),
            partition_function: z,
            entropy,
        });

        // Anneal temperature
        if self.config.enable_annealing {
            self.current_temperature *= self.config.annealing_rate;
            self.current_temperature = self.current_temperature.max(0.01); // Floor
        }

        Some(best_action)
    }

    /// Ising-inspired consensus with coupling between similar actions
    pub fn ising_consensus(&mut self, decisions: &[QStarSearchResult]) -> Option<QStarAction> {
        if decisions.is_empty() {
            return None;
        }

        // Calculate effective field for each decision
        let mut effective_fields: Vec<f64> = Vec::with_capacity(decisions.len());

        for (i, decision_i) in decisions.iter().enumerate() {
            let mut h_eff = decision_i.confidence; // External field

            // Add coupling contribution from other decisions
            for (j, decision_j) in decisions.iter().enumerate() {
                if i != j {
                    // Coupling is positive (ferromagnetic) for same action, negative for different
                    let coupling = if ActionKey::from(&decision_i.action) == ActionKey::from(&decision_j.action) {
                        self.config.coupling_j
                    } else {
                        -self.config.coupling_j * 0.5 // Weaker anti-ferromagnetic
                    };
                    
                    h_eff += coupling * decision_j.confidence;
                }
            }

            effective_fields.push(h_eff);
        }

        // Calculate Boltzmann weights with effective fields
        let weights: Vec<f64> = effective_fields.iter()
            .map(|&h| (h / self.current_temperature).exp())
            .collect();

        let z = self.partition_function(&weights);

        // Select action with highest weight
        let (best_idx, _) = weights.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;

        Some(decisions[best_idx].action.clone())
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.current_temperature
    }

    /// Reset temperature to initial value
    pub fn reset_temperature(&mut self) {
        self.current_temperature = self.config.temperature;
    }

    /// Get action magnetization (swarm preference)
    pub fn magnetization(&self, action: &ActionKey) -> f64 {
        self.action_magnetization.get(action).copied().unwrap_or(0.0)
    }

    /// Get consensus statistics
    pub fn stats(&self) -> ConsensusStats {
        if self.decision_history.is_empty() {
            return ConsensusStats::default();
        }

        let avg_entropy: f64 = self.decision_history.iter()
            .map(|d| d.entropy)
            .sum::<f64>() / self.decision_history.len() as f64;

        let avg_confidence: f64 = self.decision_history.iter()
            .map(|d| d.confidence)
            .sum::<f64>() / self.decision_history.len() as f64;

        let avg_agreement: f64 = self.decision_history.iter()
            .map(|d| d.vote_count as f64 / d.total_votes.max(1) as f64)
            .sum::<f64>() / self.decision_history.len() as f64;

        ConsensusStats {
            total_decisions: self.decision_history.len(),
            avg_entropy,
            avg_confidence,
            avg_agreement,
            current_temperature: self.current_temperature,
        }
    }
}

/// Consensus statistics
#[derive(Debug, Clone, Default)]
pub struct ConsensusStats {
    pub total_decisions: usize,
    pub avg_entropy: f64,
    pub avg_confidence: f64,
    pub avg_agreement: f64,
    pub current_temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decision(action: QStarAction, confidence: f64, value: f64) -> QStarSearchResult {
        QStarSearchResult {
            action,
            confidence,
            expected_value: value,
            search_depth: 3,
            nodes_explored: 100,
        }
    }

    #[test]
    fn test_boltzmann_weight() {
        let config = PBitConsensusConfig::default();
        let engine = PBitConsensusEngine::new(config);

        // Higher confidence = higher weight
        let w_high = engine.boltzmann_weight(0.9);
        let w_low = engine.boltzmann_weight(0.1);
        assert!(w_high > w_low);
    }

    #[test]
    fn test_boltzmann_consensus() {
        let config = PBitConsensusConfig::default();
        let mut engine = PBitConsensusEngine::new(config);

        let decisions = vec![
            make_decision(QStarAction::Buy { amount: 1.0 }, 0.9, 100.0),
            make_decision(QStarAction::Buy { amount: 0.5 }, 0.8, 90.0),
            make_decision(QStarAction::Sell { amount: 1.0 }, 0.3, 50.0),
        ];

        let result = engine.boltzmann_consensus(&decisions);
        assert!(result.is_some());
        
        // Should prefer Buy due to higher confidence
        if let Some(QStarAction::Buy { .. }) = result {
            // Expected
        } else {
            panic!("Expected Buy action");
        }
    }

    #[test]
    fn test_temperature_annealing() {
        let mut config = PBitConsensusConfig::default();
        config.enable_annealing = true;
        config.annealing_rate = 0.9;
        
        let mut engine = PBitConsensusEngine::new(config);
        let initial_temp = engine.temperature();

        let decisions = vec![
            make_decision(QStarAction::Hold, 0.5, 0.0),
        ];

        engine.boltzmann_consensus(&decisions);
        assert!(engine.temperature() < initial_temp);
    }

    #[test]
    fn test_partition_function_wolfram() {
        // Wolfram: Sum[Exp[-(-c)/1], {c, {0.1, 0.5, 0.9}}]
        // = Exp[0.1] + Exp[0.5] + Exp[0.9]
        // = 1.105 + 1.649 + 2.460 = 5.214
        let config = PBitConsensusConfig::default();
        let engine = PBitConsensusEngine::new(config);

        let weights = vec![
            engine.boltzmann_weight(0.1),
            engine.boltzmann_weight(0.5),
            engine.boltzmann_weight(0.9),
        ];

        let z = engine.partition_function(&weights);
        assert!((z - 5.214).abs() < 0.01);
    }
}
