//! pBit Lattice Coordination
//!
//! Integrates the pBit probabilistic computing layer from quantum-core
//! with the unified quantum agent coordination system.
//!
//! ## Features
//!
//! - Agent states mapped to pBit lattice nodes
//! - STDP-based learning for agent interaction weights
//! - Simulated annealing for coordination optimization
//! - Entanglement-like couplings for agent correlations
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │       SwarmEnhancedCoordinator          │
//! └──────────────────┬──────────────────────┘
//!                    │
//! ┌──────────────────▼──────────────────────┐
//! │     PBitLatticeCoordinator              │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
//! │  │ Agent 1 │──│ Agent 2 │──│ Agent 3 │  │
//! │  │  (pBit) │  │  (pBit) │  │  (pBit) │  │
//! │  └────┬────┘  └────┬────┘  └────┬────┘  │
//! │       │            │            │       │
//! │  ┌────▼────────────▼────────────▼────┐  │
//! │  │      Coupling Network (STDP)      │  │
//! │  └───────────────────────────────────┘  │
//! └─────────────────────────────────────────┘
//! ```

use quantum_core::{
    PBitState, PBitConfig, PBitCircuit, PBitBackend, PBitBackendConfig,
    LatticeState, LatticeBridgeConfig, PBitCoupling,
    QuantumResult, QuantumError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};

/// Configuration for pBit lattice coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitCoordinationConfig {
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Initial coupling strength between agents
    pub initial_coupling: f64,
    /// STDP learning rate (potentiation)
    pub stdp_a_plus: f64,
    /// STDP depression rate
    pub stdp_a_minus: f64,
    /// STDP time constant (ms)
    pub stdp_tau: f64,
    /// Number of equilibration sweeps per coordination step
    pub sweeps_per_step: usize,
    /// Enable automatic annealing
    pub auto_anneal: bool,
    /// Annealing cooling rate
    pub annealing_rate: f64,
    /// Minimum temperature for annealing
    pub min_temperature: f64,
    /// Maximum number of agents supported
    pub max_agents: usize,
}

impl Default for PBitCoordinationConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            initial_coupling: 0.5,
            stdp_a_plus: 0.01,
            stdp_a_minus: 0.005,
            stdp_tau: 20.0,
            sweeps_per_step: 10,
            auto_anneal: true,
            annealing_rate: 0.99,
            min_temperature: 0.01,
            max_agents: 32,
        }
    }
}

/// Agent node in the pBit coordination lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNode {
    /// Agent identifier
    pub agent_id: String,
    /// pBit index in the lattice
    pub pbit_index: usize,
    /// Current activation state (spin)
    pub activation: f64,
    /// Confidence level (probability_up)
    pub confidence: f64,
    /// Local bias (external influence)
    pub bias: f64,
    /// Last spike time for STDP
    pub last_spike_time: f64,
    /// Agent weight in ensemble
    pub weight: f64,
    /// Performance score
    pub performance_score: f64,
}

/// Coupling between agents in the coordination lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCoupling {
    /// Source agent
    pub from_agent: String,
    /// Target agent
    pub to_agent: String,
    /// Coupling strength (positive = cooperative, negative = competitive)
    pub strength: f64,
    /// STDP trace for learning
    pub trace: f64,
    /// Correlation history
    pub correlation: f64,
}

/// Coordination signal emitted by the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSignal {
    /// Signal type
    pub signal_type: CoordinationSignalType,
    /// Participating agents
    pub agents: Vec<String>,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Suggested action
    pub action: CoordinationAction,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Types of coordination signals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoordinationSignalType {
    /// Agents should align (consensus)
    Consensus,
    /// Agents should diversify (exploration)
    Diversify,
    /// Specific agent should lead
    LeaderElection,
    /// Risk alert from correlated agents
    RiskAlert,
    /// Opportunity detected by correlated agents
    OpportunityAlert,
}

/// Actions suggested by coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationAction {
    /// Increase agent weights
    IncreaseWeight(Vec<String>),
    /// Decrease agent weights
    DecreaseWeight(Vec<String>),
    /// Strengthen coupling between agents
    StrengthenCoupling(String, String),
    /// Weaken coupling between agents
    WeakenCoupling(String, String),
    /// Trigger collective action
    CollectiveAction(String),
    /// No action needed
    NoAction,
}

/// pBit lattice coordinator for agent coordination
pub struct PBitLatticeCoordinator {
    /// Configuration
    config: PBitCoordinationConfig,
    /// Underlying pBit state
    pbit_state: PBitState,
    /// Agent nodes mapped to pBit indices
    agents: HashMap<String, AgentNode>,
    /// Agent couplings
    couplings: HashMap<(String, String), AgentCoupling>,
    /// Current simulation time
    time: f64,
    /// Temperature for annealing
    current_temperature: f64,
    /// Coordination history
    history: Vec<CoordinationSignal>,
}

impl PBitLatticeCoordinator {
    /// Create a new pBit lattice coordinator
    pub fn new(config: PBitCoordinationConfig) -> Result<Self, QuantumError> {
        let pbit_config = PBitConfig {
            temperature: config.temperature,
            coupling_strength: config.initial_coupling,
            external_field: 0.0,
            seed: None,
        };

        // Start with capacity for max agents
        let pbit_state = PBitState::with_config(config.max_agents.min(32), pbit_config)?;

        Ok(Self {
            current_temperature: config.temperature,
            config,
            pbit_state,
            agents: HashMap::new(),
            couplings: HashMap::new(),
            time: 0.0,
            history: Vec::new(),
        })
    }

    /// Register an agent in the coordination lattice
    pub fn register_agent(&mut self, agent_id: &str, initial_weight: f64) -> Result<usize, QuantumError> {
        if self.agents.len() >= self.config.max_agents {
            return Err(QuantumError::computation_error(
                "register_agent",
                "Maximum agent capacity reached",
            ));
        }

        let pbit_index = self.agents.len();

        let agent_node = AgentNode {
            agent_id: agent_id.to_string(),
            pbit_index,
            activation: 1.0, // Start active
            confidence: 0.5, // Start uncertain
            bias: 0.0,
            last_spike_time: 0.0,
            weight: initial_weight,
            performance_score: 0.5,
        };

        // Initialize pBit for this agent
        if let Some(pbit) = self.pbit_state.get_pbit_mut(pbit_index) {
            pbit.probability_up = 0.5;
            pbit.spin = 1.0;
            pbit.bias = 0.0;
        }

        self.agents.insert(agent_id.to_string(), agent_node);

        // Create initial couplings with all existing agents
        for (existing_id, existing_node) in &self.agents {
            if existing_id != agent_id {
                let coupling = AgentCoupling {
                    from_agent: agent_id.to_string(),
                    to_agent: existing_id.clone(),
                    strength: self.config.initial_coupling,
                    trace: 0.0,
                    correlation: 0.0,
                };
                self.couplings.insert((agent_id.to_string(), existing_id.clone()), coupling.clone());
                self.couplings.insert((existing_id.clone(), agent_id.to_string()), AgentCoupling {
                    from_agent: existing_id.clone(),
                    to_agent: agent_id.to_string(),
                    strength: self.config.initial_coupling,
                    trace: 0.0,
                    correlation: 0.0,
                });

                // Add pBit coupling
                self.pbit_state.add_coupling(PBitCoupling::bell_coupling(
                    existing_node.pbit_index,
                    pbit_index,
                    self.config.initial_coupling,
                ));
            }
        }

        Ok(pbit_index)
    }

    /// Update agent state based on its signal/performance
    pub fn update_agent_state(
        &mut self,
        agent_id: &str,
        confidence: f64,
        performance: f64,
        spike: bool,
    ) -> Result<(), QuantumError> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            QuantumError::computation_error("update_agent", "Agent not found")
        })?;

        // Update agent state
        agent.confidence = confidence.clamp(0.0, 1.0);
        agent.performance_score = performance.clamp(0.0, 1.0);

        // Update corresponding pBit
        if let Some(pbit) = self.pbit_state.get_pbit_mut(agent.pbit_index) {
            pbit.probability_up = confidence;
            pbit.bias = (performance - 0.5) * 2.0 * self.config.initial_coupling;
        }

        // Handle spike for STDP
        if spike {
            let spike_time = self.time;
            let agent_id_owned = agent_id.to_string();
            let pbit_index = agent.pbit_index;
            agent.last_spike_time = spike_time;

            // Apply STDP with other agents that spiked recently
            let mut updates = Vec::new();
            for (other_id, other_node) in &self.agents {
                if other_id != agent_id {
                    let dt = spike_time - other_node.last_spike_time;
                    if dt.abs() < self.config.stdp_tau * 3.0 {
                        let dw = if dt > 0.0 {
                            // This agent spiked after other -> strengthen
                            self.config.stdp_a_plus * (-dt / self.config.stdp_tau).exp()
                        } else {
                            // This agent spiked before other -> weaken
                            -self.config.stdp_a_minus * (dt / self.config.stdp_tau).exp()
                        };
                        updates.push((other_id.clone(), other_node.pbit_index, dw));
                    }
                }
            }

            for (other_id, other_pbit_index, dw) in updates {
                // Update coupling
                if let Some(coupling) = self.couplings.get_mut(&(agent_id_owned.clone(), other_id.clone())) {
                    coupling.strength = (coupling.strength + dw).clamp(-2.0, 2.0);
                    coupling.trace = dw;
                }

                // Update pBit coupling
                let new_strength = self.couplings
                    .get(&(agent_id_owned.clone(), other_id))
                    .map(|c| c.strength)
                    .unwrap_or(0.0);

                if new_strength > 0.0 {
                    self.pbit_state.add_coupling(PBitCoupling::bell_coupling(
                        pbit_index,
                        other_pbit_index,
                        new_strength.abs(),
                    ));
                } else {
                    self.pbit_state.add_coupling(PBitCoupling::anti_bell_coupling(
                        pbit_index,
                        other_pbit_index,
                        new_strength.abs(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Perform one coordination step
    pub fn step(&mut self) -> Result<Vec<CoordinationSignal>, QuantumError> {
        // Perform Monte Carlo sweeps
        for _ in 0..self.config.sweeps_per_step {
            self.pbit_state.sweep();
        }

        // Apply annealing if enabled
        if self.config.auto_anneal && self.current_temperature > self.config.min_temperature {
            self.current_temperature *= self.config.annealing_rate;
        }

        // Update agent activations from pBit state
        for (_, agent) in &mut self.agents {
            if let Some(pbit) = self.pbit_state.get_pbit(agent.pbit_index) {
                agent.activation = pbit.spin;
                agent.confidence = pbit.probability_up;
            }
        }

        // Analyze coordination patterns
        let signals = self.analyze_coordination_patterns()?;

        // Update correlation in couplings
        self.update_correlations();

        self.time += 1.0;
        
        for signal in &signals {
            self.history.push(signal.clone());
        }

        Ok(signals)
    }

    /// Analyze patterns to generate coordination signals
    fn analyze_coordination_patterns(&self) -> Result<Vec<CoordinationSignal>, QuantumError> {
        let mut signals = Vec::new();

        // Check for consensus (most agents aligned)
        let total_activation: f64 = self.agents.values().map(|a| a.activation).sum();
        let avg_activation = total_activation / self.agents.len().max(1) as f64;

        if avg_activation.abs() > 0.7 {
            // Strong consensus
            let consensus_agents: Vec<String> = self.agents
                .iter()
                .filter(|(_, a)| a.activation.signum() == avg_activation.signum())
                .map(|(id, _)| id.clone())
                .collect();

            signals.push(CoordinationSignal {
                signal_type: CoordinationSignalType::Consensus,
                agents: consensus_agents.clone(),
                strength: avg_activation.abs(),
                action: CoordinationAction::IncreaseWeight(consensus_agents),
                timestamp: Utc::now(),
            });
        }

        // Check for strongly correlated agent pairs (opportunity/risk)
        for ((from, to), coupling) in &self.couplings {
            if coupling.strength.abs() > 1.5 && coupling.correlation.abs() > 0.8 {
                let signal_type = if coupling.strength > 0.0 {
                    CoordinationSignalType::OpportunityAlert
                } else {
                    CoordinationSignalType::RiskAlert
                };

                signals.push(CoordinationSignal {
                    signal_type,
                    agents: vec![from.clone(), to.clone()],
                    strength: coupling.correlation.abs(),
                    action: if coupling.strength > 0.0 {
                        CoordinationAction::StrengthenCoupling(from.clone(), to.clone())
                    } else {
                        CoordinationAction::CollectiveAction("hedge".to_string())
                    },
                    timestamp: Utc::now(),
                });
            }
        }

        // Check for leader election (one agent much more confident)
        let max_confidence_agent = self.agents
            .iter()
            .max_by(|a, b| a.1.confidence.partial_cmp(&b.1.confidence).unwrap());

        if let Some((leader_id, leader)) = max_confidence_agent {
            if leader.confidence > 0.9 {
                signals.push(CoordinationSignal {
                    signal_type: CoordinationSignalType::LeaderElection,
                    agents: vec![leader_id.clone()],
                    strength: leader.confidence,
                    action: CoordinationAction::IncreaseWeight(vec![leader_id.clone()]),
                    timestamp: Utc::now(),
                });
            }
        }

        Ok(signals)
    }

    /// Update correlation estimates between agents
    fn update_correlations(&mut self) {
        let alpha = 0.1; // EMA smoothing

        for ((from, to), coupling) in &mut self.couplings {
            if let (Some(from_agent), Some(to_agent)) = (
                self.agents.get(from),
                self.agents.get(to),
            ) {
                let correlation = from_agent.activation * to_agent.activation;
                coupling.correlation = (1.0 - alpha) * coupling.correlation + alpha * correlation;
            }
        }
    }

    /// Get current agent weights (based on lattice state)
    pub fn get_agent_weights(&self) -> HashMap<String, f64> {
        let total_confidence: f64 = self.agents.values().map(|a| a.confidence).sum();

        self.agents
            .iter()
            .map(|(id, agent)| {
                let weight = if total_confidence > 0.0 {
                    agent.confidence / total_confidence
                } else {
                    1.0 / self.agents.len() as f64
                };
                (id.clone(), weight * agent.weight)
            })
            .collect()
    }

    /// Get magnetization (overall consensus level)
    pub fn magnetization(&self) -> f64 {
        self.pbit_state.magnetization()
    }

    /// Get entropy (uncertainty level)
    pub fn entropy(&self) -> f64 {
        self.pbit_state.entropy()
    }

    /// Get number of registered agents
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get agent node by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&AgentNode> {
        self.agents.get(agent_id)
    }

    /// Get coupling between two agents
    pub fn get_coupling(&self, from: &str, to: &str) -> Option<&AgentCoupling> {
        self.couplings.get(&(from.to_string(), to.to_string()))
    }

    /// Reset coordinator to initial state
    pub fn reset(&mut self) -> Result<(), QuantumError> {
        self.current_temperature = self.config.temperature;
        self.time = 0.0;
        self.history.clear();

        // Reset all pBits to superposition
        for agent in self.agents.values() {
            if let Some(pbit) = self.pbit_state.get_pbit_mut(agent.pbit_index) {
                pbit.probability_up = 0.5;
                pbit.spin = 1.0;
                pbit.bias = 0.0;
            }
        }

        // Reset couplings
        for coupling in self.couplings.values_mut() {
            coupling.strength = self.config.initial_coupling;
            coupling.trace = 0.0;
            coupling.correlation = 0.0;
        }

        Ok(())
    }
}

/// Thread-safe wrapper for coordination
pub type SharedPBitCoordinator = Arc<RwLock<PBitLatticeCoordinator>>;

/// Create a shared pBit coordinator
pub fn create_shared_coordinator(config: PBitCoordinationConfig) -> Result<SharedPBitCoordinator, QuantumError> {
    Ok(Arc::new(RwLock::new(PBitLatticeCoordinator::new(config)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = PBitLatticeCoordinator::new(PBitCoordinationConfig::default()).unwrap();
        assert_eq!(coordinator.num_agents(), 0);
    }

    #[test]
    fn test_agent_registration() {
        let mut coordinator = PBitLatticeCoordinator::new(PBitCoordinationConfig::default()).unwrap();
        
        coordinator.register_agent("agent_1", 1.0).unwrap();
        coordinator.register_agent("agent_2", 1.0).unwrap();
        coordinator.register_agent("agent_3", 1.0).unwrap();

        assert_eq!(coordinator.num_agents(), 3);
        assert!(coordinator.get_agent("agent_1").is_some());
    }

    #[test]
    fn test_coordination_step() {
        let mut coordinator = PBitLatticeCoordinator::new(PBitCoordinationConfig::default()).unwrap();
        
        coordinator.register_agent("agent_1", 1.0).unwrap();
        coordinator.register_agent("agent_2", 1.0).unwrap();

        // Update states
        coordinator.update_agent_state("agent_1", 0.8, 0.7, true).unwrap();
        coordinator.update_agent_state("agent_2", 0.6, 0.5, false).unwrap();

        // Step coordination
        let signals = coordinator.step().unwrap();

        // Should have some coordination happening
        assert!(coordinator.time > 0.0);
    }

    #[test]
    fn test_stdp_learning() {
        let mut coordinator = PBitLatticeCoordinator::new(PBitCoordinationConfig {
            stdp_a_plus: 0.1,
            stdp_a_minus: 0.05,
            ..Default::default()
        }).unwrap();

        coordinator.register_agent("a", 1.0).unwrap();
        coordinator.register_agent("b", 1.0).unwrap();

        // Agent A spikes
        coordinator.update_agent_state("a", 0.9, 0.8, true).unwrap();
        coordinator.step().unwrap();

        // Agent B spikes shortly after
        coordinator.update_agent_state("b", 0.85, 0.75, true).unwrap();
        coordinator.step().unwrap();

        // Coupling from B to A should be strengthened
        let coupling = coordinator.get_coupling("b", "a");
        assert!(coupling.is_some());
    }
}
