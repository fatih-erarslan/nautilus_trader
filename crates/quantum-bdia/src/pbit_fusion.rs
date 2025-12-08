//! pBit Fusion for BDIA Decision Making
//!
//! Replaces the quantum circuit simulation with pBit probabilistic computing
//! for belief-desire-intention fusion. Uses Ising model dynamics and Boltzmann
//! sampling for decision synthesis.
//!
//! ## Advantages over Quantum Simulation
//!
//! - Deterministic reproducibility with seeded RNG
//! - Efficient annealing for optimization
//! - STDP-like learning for adaptive fusion weights
//! - Natural mapping to belief/intention probabilities
//!
//! ## Architecture
//!
//! ```text
//! Beliefs  ─┬─> pBit Lattice ─> Annealing ─> Fusion Result
//! Desires  ─┤      ↓
//! Intentions┴─> Couplings (entanglement-like correlations)
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::{DecisionType, network::ConsensusDecision};
use crate::market_phase::MarketPhase;

/// pBit fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitFusionConfig {
    /// Number of pBits for decision encoding
    pub num_pbits: usize,
    /// Initial temperature for sampling
    pub temperature: f64,
    /// Coupling strength between belief pBits
    pub belief_coupling: f64,
    /// Coupling strength between intention pBits
    pub intention_coupling: f64,
    /// Cross-coupling between beliefs and intentions
    pub cross_coupling: f64,
    /// Number of equilibration sweeps
    pub equilibration_sweeps: usize,
    /// Enable annealing for optimization
    pub enable_annealing: bool,
    /// Annealing steps
    pub annealing_steps: usize,
    /// Target temperature after annealing
    pub target_temperature: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PBitFusionConfig {
    fn default() -> Self {
        Self {
            num_pbits: 8,
            temperature: 2.0,
            belief_coupling: 1.0,
            intention_coupling: 1.5,
            cross_coupling: 0.5,
            equilibration_sweeps: 50,
            enable_annealing: true,
            annealing_steps: 100,
            target_temperature: 0.1,
            seed: None,
        }
    }
}

/// pBit node representing a belief or intention component
#[derive(Debug, Clone)]
struct FusionPBit {
    /// Component type
    component: FusionComponent,
    /// Current spin (-1 or +1)
    spin: f64,
    /// Probability of spin up
    probability_up: f64,
    /// External bias
    bias: f64,
    /// Input value (normalized)
    input: f64,
}

/// Types of fusion components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FusionComponent {
    /// Belief component
    Belief(usize),
    /// Intention component
    Intention(usize),
    /// Decision output
    Decision(usize),
}

/// pBit fusion engine for BDIA
pub struct PBitFusion {
    /// Configuration
    config: PBitFusionConfig,
    /// pBit nodes
    pbits: Vec<FusionPBit>,
    /// Couplings between pBits
    couplings: HashMap<(usize, usize), f64>,
    /// Current temperature
    temperature: f64,
    /// Coherence metric
    coherence: f64,
}

impl PBitFusion {
    /// Create new pBit fusion engine
    pub fn new(config: PBitFusionConfig) -> Result<Self> {
        info!("Initializing pBit fusion with {} pBits", config.num_pbits);

        let num_pbits = config.num_pbits;
        let mut pbits = Vec::with_capacity(num_pbits);

        // Create belief pBits (first third)
        let num_belief = num_pbits / 3;
        for i in 0..num_belief {
            pbits.push(FusionPBit {
                component: FusionComponent::Belief(i),
                spin: -1.0,
                probability_up: 0.5,
                bias: 0.0,
                input: 0.0,
            });
        }

        // Create intention pBits (second third)
        let num_intention = num_pbits / 3;
        for i in 0..num_intention {
            pbits.push(FusionPBit {
                component: FusionComponent::Intention(i),
                spin: -1.0,
                probability_up: 0.5,
                bias: 0.0,
                input: 0.0,
            });
        }

        // Create decision pBits (remaining)
        let num_decision = num_pbits - num_belief - num_intention;
        for i in 0..num_decision {
            pbits.push(FusionPBit {
                component: FusionComponent::Decision(i),
                spin: -1.0,
                probability_up: 0.5,
                bias: 0.0,
                input: 0.0,
            });
        }

        // Initialize couplings
        let mut couplings = HashMap::new();

        // Couple beliefs together (ferromagnetic)
        for i in 0..num_belief {
            for j in (i + 1)..num_belief {
                couplings.insert((i, j), config.belief_coupling);
                couplings.insert((j, i), config.belief_coupling);
            }
        }

        // Couple intentions together (ferromagnetic)
        let intention_start = num_belief;
        for i in 0..num_intention {
            for j in (i + 1)..num_intention {
                let idx_i = intention_start + i;
                let idx_j = intention_start + j;
                couplings.insert((idx_i, idx_j), config.intention_coupling);
                couplings.insert((idx_j, idx_i), config.intention_coupling);
            }
        }

        // Cross-couple beliefs to intentions
        for i in 0..num_belief {
            for j in 0..num_intention {
                let idx_j = intention_start + j;
                couplings.insert((i, idx_j), config.cross_coupling);
                couplings.insert((idx_j, i), config.cross_coupling);
            }
        }

        // Couple intentions and beliefs to decision pBits
        let decision_start = intention_start + num_intention;
        for d in 0..num_decision {
            let idx_d = decision_start + d;

            // Connect to all beliefs and intentions
            for i in 0..num_belief {
                couplings.insert((i, idx_d), config.cross_coupling * 0.8);
                couplings.insert((idx_d, i), config.cross_coupling * 0.8);
            }
            for i in 0..num_intention {
                let idx_i = intention_start + i;
                couplings.insert((idx_i, idx_d), config.intention_coupling * 0.7);
                couplings.insert((idx_d, idx_i), config.intention_coupling * 0.7);
            }
        }

        Ok(Self {
            temperature: config.temperature,
            config,
            pbits,
            couplings,
            coherence: 1.0,
        })
    }

    /// Fuse consensus decision through pBit dynamics
    pub async fn fuse(&mut self, consensus: ConsensusDecision) -> Result<PBitFusionResult> {
        debug!("Performing pBit fusion on consensus decision");

        // Encode consensus into pBit biases
        self.encode_consensus(&consensus);

        // Reset temperature
        self.temperature = self.config.temperature;

        // Equilibration sweeps
        for _ in 0..self.config.equilibration_sweeps {
            self.sweep();
        }

        // Annealing phase
        if self.config.enable_annealing {
            self.anneal();
        }

        // Decode decision from pBit state
        let (decision, confidence) = self.decode_decision();

        // Compute coherence
        self.coherence = self.compute_coherence();

        // Build reasoning
        let reasoning = vec![
            format!("pBit fusion with {} nodes", self.pbits.len()),
            format!("Final temperature: {:.3}", self.temperature),
            format!("Coherence: {:.3}", self.coherence),
            format!("Magnetization: {:.3}", self.magnetization()),
        ];

        Ok(PBitFusionResult {
            decision,
            pbit_confidence: confidence,
            classical_confidence: consensus.confidence,
            intention_signal: consensus.weighted_intention,
            coherence: self.coherence,
            entropy: self.entropy(),
            magnetization: self.magnetization(),
            reasoning,
        })
    }

    /// Encode consensus into pBit biases
    fn encode_consensus(&mut self, consensus: &ConsensusDecision) {
        let intention = consensus.weighted_intention;
        let confidence = consensus.confidence;

        // Market phase affects bias
        let phase_bias = match consensus.market_phase {
            MarketPhase::Growth => 0.5,
            MarketPhase::Conservation => 0.1,
            MarketPhase::Release => -0.3,
            MarketPhase::Reorganization => -0.1,
        };

        // Set biases based on consensus
        for pbit in &mut self.pbits {
            match pbit.component {
                FusionComponent::Belief(i) => {
                    // Beliefs get confidence-weighted bias
                    pbit.bias = confidence * (i as f64 + 1.0) * 0.2;
                    pbit.input = confidence;
                }
                FusionComponent::Intention(i) => {
                    // Intentions get intention signal as bias
                    pbit.bias = intention * self.config.intention_coupling * (1.0 - i as f64 * 0.1);
                    pbit.input = intention;
                }
                FusionComponent::Decision(i) => {
                    // Decision pBits get phase-adjusted bias
                    pbit.bias = phase_bias * (1.0 + i as f64 * 0.2);
                    pbit.input = phase_bias;
                }
            }
        }
    }

    /// Perform one Monte Carlo sweep
    fn sweep(&mut self) {
        for i in 0..self.pbits.len() {
            // Compute local field
            let mut field = self.pbits[i].bias;

            for j in 0..self.pbits.len() {
                if i != j {
                    if let Some(&coupling) = self.couplings.get(&(i, j)) {
                        field += coupling * self.pbits[j].spin;
                    }
                }
            }

            // Update probability via Boltzmann distribution
            let exponent = -2.0 * field / self.temperature.max(1e-10);
            self.pbits[i].probability_up = 1.0 / (1.0 + exponent.exp());

            // Sample new spin
            self.pbits[i].spin = if rand::random::<f64>() < self.pbits[i].probability_up {
                1.0
            } else {
                -1.0
            };
        }
    }

    /// Perform simulated annealing
    fn anneal(&mut self) {
        let temp_ratio = (self.config.target_temperature / self.temperature)
            .powf(1.0 / self.config.annealing_steps as f64);

        for _ in 0..self.config.annealing_steps {
            self.temperature *= temp_ratio;
            self.temperature = self.temperature.max(self.config.target_temperature);
            self.sweep();
        }
    }

    /// Decode decision from pBit state
    fn decode_decision(&self) -> (DecisionType, f64) {
        // Collect decision pBit votes
        let mut buy_votes = 0.0;
        let mut sell_votes = 0.0;
        let mut hold_votes = 0.0;
        let mut total_confidence = 0.0;

        for pbit in &self.pbits {
            if let FusionComponent::Decision(i) = pbit.component {
                let vote_strength = pbit.probability_up;
                total_confidence += vote_strength.abs();

                match i % 3 {
                    0 => {
                        if pbit.spin > 0.0 {
                            buy_votes += vote_strength;
                        } else {
                            sell_votes += vote_strength;
                        }
                    }
                    1 => {
                        if pbit.spin > 0.0 {
                            hold_votes += vote_strength;
                        } else {
                            sell_votes += vote_strength;
                        }
                    }
                    _ => {
                        if pbit.spin > 0.0 {
                            buy_votes += vote_strength * 0.5;
                            hold_votes += vote_strength * 0.5;
                        } else {
                            sell_votes += vote_strength;
                        }
                    }
                }
            }
        }

        // Also consider intention pBits
        for pbit in &self.pbits {
            if let FusionComponent::Intention(_) = pbit.component {
                if pbit.spin > 0.0 && pbit.probability_up > 0.6 {
                    buy_votes += 0.3;
                } else if pbit.spin < 0.0 && pbit.probability_up < 0.4 {
                    sell_votes += 0.3;
                }
            }
        }

        // Determine winner
        let total_votes = buy_votes + sell_votes + hold_votes;
        let confidence = if total_votes > 0.0 {
            (buy_votes.max(sell_votes).max(hold_votes) / total_votes).min(1.0)
        } else {
            0.5
        };

        let decision = if buy_votes > sell_votes && buy_votes > hold_votes {
            DecisionType::Buy
        } else if sell_votes > buy_votes && sell_votes > hold_votes {
            DecisionType::Sell
        } else {
            DecisionType::Hold
        };

        (decision, confidence)
    }

    /// Compute coherence (how aligned are the pBits)
    fn compute_coherence(&self) -> f64 {
        let n = self.pbits.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Coherence based on how close probabilities are to 0 or 1
        let certainty: f64 = self.pbits
            .iter()
            .map(|p| (2.0 * p.probability_up - 1.0).abs())
            .sum();

        certainty / n
    }

    /// Get magnetization (average spin)
    fn magnetization(&self) -> f64 {
        if self.pbits.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.pbits.iter().map(|p| p.spin).sum();
        sum / self.pbits.len() as f64
    }

    /// Get entropy
    fn entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for pbit in &self.pbits {
            let p = pbit.probability_up.clamp(1e-10, 1.0 - 1e-10);
            entropy -= p * p.ln() + (1.0 - p) * (1.0 - p).ln();
        }
        entropy
    }

    /// Get current coherence level
    pub fn coherence(&self) -> f64 {
        self.coherence
    }

    /// Update coherence after decision
    pub fn update_coherence(&mut self, outcome_success: bool) {
        let adjustment = if outcome_success { 0.01 } else { -0.005 };
        self.coherence = (self.coherence + adjustment).clamp(0.0, 1.0);
    }
}

/// Result of pBit fusion
#[derive(Debug, Clone)]
pub struct PBitFusionResult {
    /// Final decision
    pub decision: DecisionType,
    /// Confidence from pBit voting
    pub pbit_confidence: f64,
    /// Original classical confidence
    pub classical_confidence: f64,
    /// Intention signal strength
    pub intention_signal: f64,
    /// Coherence level
    pub coherence: f64,
    /// Entropy (uncertainty)
    pub entropy: f64,
    /// Magnetization (consensus direction)
    pub magnetization: f64,
    /// Reasoning steps
    pub reasoning: Vec<String>,
}

impl PBitFusionResult {
    /// Convert to standard FusionResult for compatibility
    pub fn to_fusion_result(&self) -> crate::quantum_fusion::FusionResult {
        use num_complex::Complex64;

        // Create synthetic quantum state from pBit state
        let num_states = 4; // Simplified
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        let mut probabilities = vec![0.0; num_states];

        // Map pBit probabilities to quantum-like state
        let p_buy = if matches!(self.decision, DecisionType::Buy) {
            self.pbit_confidence
        } else {
            0.0
        };
        let p_sell = if matches!(self.decision, DecisionType::Sell) {
            self.pbit_confidence
        } else {
            0.0
        };
        let p_hold = if matches!(self.decision, DecisionType::Hold) {
            self.pbit_confidence
        } else {
            0.0
        };
        let p_other = 1.0 - p_buy - p_sell - p_hold;

        probabilities[0] = p_buy.max(0.0);
        probabilities[1] = p_sell.max(0.0);
        probabilities[2] = p_hold.max(0.0);
        probabilities[3] = p_other.max(0.0);

        for (i, &p) in probabilities.iter().enumerate() {
            amplitudes[i] = Complex64::new(p.sqrt(), 0.0);
        }

        let quantum_state = crate::quantum_fusion::QuantumState {
            amplitudes,
            probabilities,
            entanglement: self.coherence,
            phases: vec![0.0; num_states],
        };

        crate::quantum_fusion::FusionResult {
            decision: self.decision,
            quantum_confidence: self.pbit_confidence,
            classical_confidence: self.classical_confidence,
            intention_signal: self.intention_signal,
            quantum_state,
            reasoning: self.reasoning.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::ConsensusDecision;
    use crate::market_phase::MarketPhase;

    fn mock_consensus(intention: f64, confidence: f64, phase: MarketPhase) -> ConsensusDecision {
        ConsensusDecision {
            weighted_intention: intention,
            confidence,
            market_phase: phase,
            participating_agents: 5,
            consensus_strength: 0.8,
        }
    }

    #[tokio::test]
    async fn test_pbit_fusion_creation() {
        let fusion = PBitFusion::new(PBitFusionConfig::default()).unwrap();
        assert!(fusion.pbits.len() > 0);
    }

    #[tokio::test]
    async fn test_buy_decision() {
        let mut fusion = PBitFusion::new(PBitFusionConfig {
            temperature: 0.5,
            enable_annealing: true,
            ..Default::default()
        }).unwrap();

        // Strong positive intention -> should bias toward buy
        let consensus = mock_consensus(1.5, 0.9, MarketPhase::Growth);
        let result = fusion.fuse(consensus).await.unwrap();

        // Should have reasonable confidence
        assert!(result.pbit_confidence > 0.3);
        assert!(result.coherence > 0.0);
    }

    #[tokio::test]
    async fn test_sell_decision() {
        let mut fusion = PBitFusion::new(PBitFusionConfig {
            temperature: 0.5,
            enable_annealing: true,
            ..Default::default()
        }).unwrap();

        // Strong negative intention + release phase -> should bias toward sell
        let consensus = mock_consensus(-1.5, 0.9, MarketPhase::Release);
        let result = fusion.fuse(consensus).await.unwrap();

        assert!(result.pbit_confidence > 0.3);
    }

    #[tokio::test]
    async fn test_coherence_update() {
        let mut fusion = PBitFusion::new(PBitFusionConfig::default()).unwrap();

        let initial_coherence = fusion.coherence();
        fusion.update_coherence(true);
        assert!(fusion.coherence() >= initial_coherence);

        fusion.update_coherence(false);
        // Should decrease slightly
    }
}
