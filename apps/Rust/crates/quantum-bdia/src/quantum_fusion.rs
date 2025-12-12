//! Quantum fusion mechanism for decision making

use std::sync::Arc;
use anyhow::Result;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use tracing::{info, debug};

use crate::{DecisionType, network::ConsensusDecision};
use quantum_hive::QuantumQueen;

/// Quantum fusion engine for decision synthesis
pub struct QuantumFusion {
    /// Number of qubits
    qubits: usize,
    /// Measurement shots
    shots: usize,
    /// Quantum state simulator
    quantum_sim: QuantumSimulator,
}

/// Quantum fusion result
#[derive(Debug, Clone)]
pub struct FusionResult {
    pub decision: DecisionType,
    pub quantum_confidence: f64,
    pub classical_confidence: f64,
    pub intention_signal: f64,
    pub quantum_state: QuantumState,
    pub reasoning: Vec<String>,
}

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    /// State vector amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Measurement probabilities
    pub probabilities: Vec<f64>,
    /// Entanglement measure
    pub entanglement: f64,
    /// Phase information
    pub phases: Vec<f64>,
}

/// Simple quantum simulator for BDIA
struct QuantumSimulator {
    /// Number of qubits
    qubits: usize,
    /// State vector
    state: Array1<Complex64>,
}

impl QuantumFusion {
    /// Create new quantum fusion engine
    pub fn new(config: crate::QuantumConfig) -> Result<Self> {
        info!("Initializing quantum fusion with {} qubits", config.qubits);
        
        let quantum_sim = QuantumSimulator::new(config.qubits);
        
        Ok(Self {
            qubits: config.qubits,
            shots: config.shots,
            quantum_sim,
        })
    }
    
    /// Fuse consensus decision through quantum circuit
    pub async fn fuse(&self, consensus: ConsensusDecision) -> Result<FusionResult> {
        debug!("Performing quantum fusion on consensus decision");
        
        // Normalize intention signal to rotation angle
        let angle = self.normalize_to_angle(consensus.weighted_intention);
        
        // Apply phase adjustment based on market phase
        let phase_angle = match consensus.market_phase {
            crate::market_phase::MarketPhase::Growth => 0.2,
            crate::market_phase::MarketPhase::Conservation => 0.1,
            crate::market_phase::MarketPhase::Release => -0.2,
            crate::market_phase::MarketPhase::Reorganization => -0.1,
        };
        
        // Create quantum circuit
        let circuit = self.create_fusion_circuit(angle, phase_angle);
        
        // Execute circuit and get quantum state
        let quantum_state = self.quantum_sim.execute_circuit(&circuit)?;
        
        // Interpret quantum state for decision
        let (decision, quantum_confidence) = self.interpret_quantum_state(&quantum_state, consensus.weighted_intention);
        
        let mut reasoning = vec![
            format!("Quantum fusion with {} qubits", self.qubits),
            format!("Intention angle: {:.3} rad", angle),
            format!("Market phase adjustment: {:.3} rad", phase_angle),
            format!("Quantum confidence: {:.3}", quantum_confidence),
        ];
        
        Ok(FusionResult {
            decision,
            quantum_confidence,
            classical_confidence: consensus.confidence,
            intention_signal: consensus.weighted_intention,
            quantum_state,
            reasoning,
        })
    }
    
    /// Normalize signal to angle in [-π, π]
    fn normalize_to_angle(&self, signal: f64) -> f64 {
        // Use same normalization as original
        let min_signal = -2.0;
        let max_signal = 2.0;
        std::f64::consts::PI * (2.0 * (signal - min_signal) / (max_signal - min_signal) - 1.0)
    }
    
    /// Create quantum fusion circuit
    fn create_fusion_circuit(&self, theta: f64, phase: f64) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(self.qubits);
        
        // Apply rotations based on intention
        circuit.ry(0, theta);
        
        if self.qubits >= 2 {
            circuit.ry(1, theta + phase);
            circuit.cnot(0, 1);
            circuit.rz(0, theta * 0.5);
            circuit.rx(1, phase);
            
            if self.qubits >= 3 {
                // Extended circuit for 3+ qubits
                circuit.ry(2, theta * 0.6);
                circuit.cnot(1, 2);
                circuit.rz(2, phase * 0.7);
                circuit.cnot(2, 0); // Create loop
            }
        }
        
        circuit
    }
    
    /// Interpret quantum state to make decision
    fn interpret_quantum_state(
        &self,
        state: &QuantumState,
        intention: f64,
    ) -> (DecisionType, f64) {
        let probs = &state.probabilities;
        
        // Decision mapping based on qubit count
        let (decision, confidence) = if self.qubits == 1 {
            // Single qubit: |0⟩ = Hold/Sell, |1⟩ = Buy/Increase
            if probs[0] > 0.6 {
                if intention < 0.0 {
                    (DecisionType::Sell, probs[0])
                } else {
                    (DecisionType::Hold, probs[0])
                }
            } else if probs[1] > 0.6 {
                (DecisionType::Buy, probs[1])
            } else {
                (DecisionType::Hold, 0.5)
            }
        } else if self.qubits == 2 {
            // Two qubits: more nuanced decisions
            if probs[0] > 0.4 {          // |00⟩
                (DecisionType::Buy, probs[0])
            } else if probs[3] > 0.4 {   // |11⟩
                (DecisionType::Sell, probs[3])
            } else if probs[1] > 0.4 {   // |01⟩
                if intention > 0.0 {
                    (DecisionType::Increase(10), probs[1])
                } else {
                    (DecisionType::Decrease(10), probs[1])
                }
            } else if probs[2] > 0.4 {   // |10⟩
                if intention > 0.0 {
                    (DecisionType::Hedge, probs[2])
                } else {
                    (DecisionType::Exit, probs[2])
                }
            } else {
                (DecisionType::Hold, 1.0 - probs.iter().take(4).sum::<f64>())
            }
        } else {
            // 3+ qubits: use dominant state
            self.interpret_multi_qubit(probs, intention)
        };
        
        (decision, confidence)
    }
    
    /// Interpret multi-qubit states (3+)
    fn interpret_multi_qubit(&self, probs: &[f64], intention: f64) -> (DecisionType, f64) {
        let n_states = 2_usize.pow(self.qubits as u32);
        
        // Find dominant state
        let (max_idx, &max_prob) = probs
            .iter()
            .take(n_states)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        // Map state index to decision
        let decision = match max_idx {
            0 => DecisionType::Buy,              // All zeros
            idx if idx == n_states - 1 => DecisionType::Sell,  // All ones
            idx if idx < n_states / 4 => {
                if intention > 0.0 {
                    DecisionType::Increase(10)
                } else {
                    DecisionType::Decrease(10)
                }
            }
            idx if idx < n_states / 2 => DecisionType::Hold,
            _ => {
                if intention > 0.0 {
                    DecisionType::Hedge
                } else {
                    DecisionType::Exit
                }
            }
        };
        
        (decision, max_prob)
    }
    
    /// Get quantum coherence measure
    pub fn coherence(&self) -> f64 {
        self.quantum_sim.coherence()
    }
}

/// Simple quantum circuit representation
struct QuantumCircuit {
    qubits: usize,
    gates: Vec<Gate>,
}

#[derive(Debug, Clone)]
enum Gate {
    RY(usize, f64),    // Y-rotation on qubit
    RX(usize, f64),    // X-rotation on qubit
    RZ(usize, f64),    // Z-rotation on qubit
    CNOT(usize, usize), // Controlled-NOT
}

impl QuantumCircuit {
    fn new(qubits: usize) -> Self {
        Self {
            qubits,
            gates: Vec::new(),
        }
    }
    
    fn ry(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::RY(qubit, angle));
    }
    
    fn rx(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::RX(qubit, angle));
    }
    
    fn rz(&mut self, qubit: usize, angle: f64) {
        self.gates.push(Gate::RZ(qubit, angle));
    }
    
    fn cnot(&mut self, control: usize, target: usize) {
        self.gates.push(Gate::CNOT(control, target));
    }
}

impl QuantumSimulator {
    fn new(qubits: usize) -> Self {
        let size = 2_usize.pow(qubits as u32);
        let mut state = Array1::zeros(size);
        state[0] = Complex64::new(1.0, 0.0); // |00...0⟩ state
        
        Self { qubits, state }
    }
    
    /// Execute quantum circuit
    fn execute_circuit(&self, circuit: &QuantumCircuit) -> Result<QuantumState> {
        let mut state = self.state.clone();
        
        // Apply gates
        for gate in &circuit.gates {
            match gate {
                Gate::RY(qubit, angle) => {
                    state = self.apply_ry(&state, *qubit, *angle);
                }
                Gate::RX(qubit, angle) => {
                    state = self.apply_rx(&state, *qubit, *angle);
                }
                Gate::RZ(qubit, angle) => {
                    state = self.apply_rz(&state, *qubit, *angle);
                }
                Gate::CNOT(control, target) => {
                    state = self.apply_cnot(&state, *control, *target);
                }
            }
        }
        
        // Calculate probabilities
        let probabilities: Vec<f64> = state
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Calculate phases
        let phases: Vec<f64> = state
            .iter()
            .map(|amp| amp.arg())
            .collect();
        
        // Calculate entanglement (simplified)
        let entanglement = self.calculate_entanglement(&state);
        
        Ok(QuantumState {
            amplitudes: state.to_vec(),
            probabilities,
            entanglement,
            phases,
        })
    }
    
    /// Apply RY gate
    fn apply_ry(&self, state: &Array1<Complex64>, qubit: usize, angle: f64) -> Array1<Complex64> {
        let mut new_state = state.clone();
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        let n = 2_usize.pow(self.qubits as u32);
        let qubit_mask = 1 << qubit;
        
        for i in 0..n {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                let amp0 = state[i];
                let amp1 = state[j];
                
                new_state[i] = amp0 * cos_half - amp1 * sin_half;
                new_state[j] = amp0 * sin_half + amp1 * cos_half;
            }
        }
        
        new_state
    }
    
    /// Apply RX gate
    fn apply_rx(&self, state: &Array1<Complex64>, qubit: usize, angle: f64) -> Array1<Complex64> {
        let mut new_state = state.clone();
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let i_unit = Complex64::new(0.0, 1.0);
        
        let n = 2_usize.pow(self.qubits as u32);
        let qubit_mask = 1 << qubit;
        
        for i in 0..n {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                let amp0 = state[i];
                let amp1 = state[j];
                
                new_state[i] = amp0 * cos_half - i_unit * amp1 * sin_half;
                new_state[j] = -i_unit * amp0 * sin_half + amp1 * cos_half;
            }
        }
        
        new_state
    }
    
    /// Apply RZ gate
    fn apply_rz(&self, state: &Array1<Complex64>, qubit: usize, angle: f64) -> Array1<Complex64> {
        let mut new_state = state.clone();
        let phase_pos = Complex64::from_polar(1.0, angle / 2.0);
        let phase_neg = Complex64::from_polar(1.0, -angle / 2.0);
        
        let n = 2_usize.pow(self.qubits as u32);
        let qubit_mask = 1 << qubit;
        
        for i in 0..n {
            if i & qubit_mask == 0 {
                new_state[i] *= phase_neg;
            } else {
                new_state[i] *= phase_pos;
            }
        }
        
        new_state
    }
    
    /// Apply CNOT gate
    fn apply_cnot(&self, state: &Array1<Complex64>, control: usize, target: usize) -> Array1<Complex64> {
        let mut new_state = state.clone();
        let n = 2_usize.pow(self.qubits as u32);
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        
        for i in 0..n {
            if (i & control_mask) != 0 {
                // Control is |1⟩, flip target
                let j = i ^ target_mask;
                new_state[i] = state[j];
                new_state[j] = state[i];
            }
        }
        
        new_state
    }
    
    /// Calculate entanglement measure
    fn calculate_entanglement(&self, state: &Array1<Complex64>) -> f64 {
        // Simplified entanglement measure based on state purity
        // For a pure state, entanglement can be estimated from reduced density matrix
        
        if self.qubits < 2 {
            return 0.0;
        }
        
        // Calculate purity of reduced density matrix
        let reduced_size = 2_usize.pow((self.qubits - 1) as u32);
        let mut purity = 0.0;
        
        for i in 0..reduced_size {
            let amp_sum = state[i].norm_sqr() + state[i + reduced_size].norm_sqr();
            purity += amp_sum * amp_sum;
        }
        
        // Convert purity to entanglement measure
        1.0 - purity
    }
    
    /// Get coherence measure
    fn coherence(&self) -> f64 {
        // Measure of superposition
        let probs: Vec<f64> = self.state.iter().map(|a| a.norm_sqr()).collect();
        let max_prob = probs.iter().cloned().fold(0.0, f64::max);
        
        1.0 - max_prob // Higher coherence when no single state dominates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_fusion_creation() {
        let config = crate::QuantumConfig::default();
        let fusion = QuantumFusion::new(config).unwrap();
        
        assert_eq!(fusion.qubits, 5);
        assert_eq!(fusion.shots, 1024);
    }
    
    #[test]
    fn test_angle_normalization() {
        let config = crate::QuantumConfig::default();
        let fusion = QuantumFusion::new(config).unwrap();
        
        let angle = fusion.normalize_to_angle(0.0);
        assert!((angle - 0.0).abs() < 1e-10);
        
        let angle_max = fusion.normalize_to_angle(2.0);
        assert!((angle_max - std::f64::consts::PI).abs() < 1e-10);
        
        let angle_min = fusion.normalize_to_angle(-2.0);
        assert!((angle_min + std::f64::consts::PI).abs() < 1e-10);
    }
    
    #[tokio::test]
    async fn test_quantum_fusion() {
        let config = crate::QuantumConfig {
            qubits: 2,
            shots: 100,
            use_hardware_acceleration: false,
        };
        
        let fusion = QuantumFusion::new(config).unwrap();
        
        let consensus = ConsensusDecision {
            decision: DecisionType::Buy,
            weighted_intention: 0.7,
            agent_decisions: vec![],
            confidence: 0.8,
            market_phase: crate::market_phase::MarketPhase::Growth,
            timestamp: chrono::Utc::now(),
        };
        
        let result = fusion.fuse(consensus).await.unwrap();
        
        assert!(result.quantum_confidence >= 0.0 && result.quantum_confidence <= 1.0);
        assert!(!result.reasoning.is_empty());
        assert_eq!(result.quantum_state.probabilities.len(), 4); // 2^2 states
    }
}