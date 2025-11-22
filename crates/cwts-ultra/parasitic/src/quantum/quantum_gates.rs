//! # Quantum Gate Operations
//! 
//! Quantum gate implementations with feature gating for different modes

use serde::{Serialize, Deserialize};
use num_complex::Complex64;
use crate::quantum::QuantumMode;
use crate::{quantum_gate, if_quantum, if_full_quantum};

/// Quantum gate factory for mode-dependent gate operations
#[derive(Debug, Clone)]
pub struct QuantumGateFactory;

impl QuantumGateFactory {
    /// Create a quantum gate operation based on current mode
    pub fn create_gate_operation(gate_type: GateType, params: GateParams) -> Box<dyn GateOperation> {
        quantum_gate!(
            // Classical mode: use classical implementations
            Box::new(ClassicalGateOperation::new(gate_type, params)),
            // Enhanced mode: use quantum-inspired implementations  
            Box::new(EnhancedGateOperation::new(gate_type, params)),
            // Full quantum mode: use full quantum gate operations
            Box::new(QuantumGateOperation::new(gate_type, params))
        )
    }
}

/// Gate operation trait
pub trait GateOperation: Send + Sync {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumGateError>;
    fn gate_type(&self) -> GateType;
    fn is_reversible(&self) -> bool;
    fn computational_cost(&self) -> u64;
}

/// Quantum state representation that adapts to current mode
#[derive(Debug, Clone)]
pub enum QuantumState {
    Classical(ClassicalState),
    Enhanced(EnhancedState), 
    FullQuantum(FullQuantumState),
}

impl QuantumState {
    pub fn new(num_qubits: u32) -> Self {
        quantum_gate!(
            QuantumState::Classical(ClassicalState::new(num_qubits)),
            QuantumState::Enhanced(EnhancedState::new(num_qubits)),
            QuantumState::FullQuantum(FullQuantumState::new(num_qubits))
        )
    }
    
    pub fn measure(&self) -> Vec<u8> {
        match self {
            QuantumState::Classical(state) => state.measure(),
            QuantumState::Enhanced(state) => state.measure(),
            QuantumState::FullQuantum(state) => state.measure(),
        }
    }
    
    pub fn get_probabilities(&self) -> Vec<f64> {
        match self {
            QuantumState::Classical(state) => state.get_probabilities(),
            QuantumState::Enhanced(state) => state.get_probabilities(), 
            QuantumState::FullQuantum(state) => state.get_probabilities(),
        }
    }
}

/// Classical state implementation
#[derive(Debug, Clone)]
pub struct ClassicalState {
    bits: Vec<u8>,
    probabilities: Vec<f64>,
}

impl ClassicalState {
    pub fn new(num_qubits: u32) -> Self {
        let num_states = 1usize << num_qubits;
        let mut probabilities = vec![0.0; num_states];
        probabilities[0] = 1.0; // Initialize to |0...0⟩
        
        Self {
            bits: vec![0; num_qubits as usize],
            probabilities,
        }
    }
    
    pub fn measure(&self) -> Vec<u8> {
        // Sample from probability distribution
        let rand_val = fastrand::f64();
        let mut cumulative = 0.0;
        
        for (state_idx, &prob) in self.probabilities.iter().enumerate() {
            cumulative += prob;
            if rand_val <= cumulative {
                // Convert state index to bit string
                let mut bits = vec![0u8; self.bits.len()];
                let mut idx = state_idx;
                for i in 0..bits.len() {
                    bits[i] = (idx & 1) as u8;
                    idx >>= 1;
                }
                return bits;
            }
        }
        
        self.bits.clone()
    }
    
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.probabilities.clone()
    }
}

/// Enhanced state with quantum-inspired features
#[derive(Debug, Clone)]
pub struct EnhancedState {
    amplitudes: Vec<Complex64>,
    coherence_matrix: Vec<Vec<f64>>,
}

impl EnhancedState {
    pub fn new(num_qubits: u32) -> Self {
        let num_states = 1usize << num_qubits;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); num_states];
        amplitudes[0] = Complex64::new(1.0, 0.0);
        
        let coherence_matrix = vec![vec![0.0; num_states]; num_states];
        
        Self {
            amplitudes,
            coherence_matrix,
        }
    }
    
    pub fn measure(&self) -> Vec<u8> {
        // Convert amplitudes to probabilities
        let probabilities: Vec<f64> = self.amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        // Sample from distribution
        let rand_val = fastrand::f64();
        let mut cumulative = 0.0;
        
        for (state_idx, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if rand_val <= cumulative {
                let mut bits = vec![0u8; (probabilities.len() as f64).log2() as usize];
                let mut idx = state_idx;
                for i in 0..bits.len() {
                    bits[i] = (idx & 1) as u8;
                    idx >>= 1;
                }
                return bits;
            }
        }
        
        vec![0; (probabilities.len() as f64).log2() as usize]
    }
    
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect()
    }
}

/// Full quantum state
#[derive(Debug, Clone)]
pub struct FullQuantumState {
    statevector: Vec<Complex64>,
    num_qubits: u32,
    decoherence_time: f64,
}

impl FullQuantumState {
    pub fn new(num_qubits: u32) -> Self {
        let num_states = 1usize << num_qubits;
        let mut statevector = vec![Complex64::new(0.0, 0.0); num_states];
        statevector[0] = Complex64::new(1.0, 0.0);
        
        Self {
            statevector,
            num_qubits,
            decoherence_time: 100.0, // microseconds
        }
    }
    
    pub fn measure(&self) -> Vec<u8> {
        let probabilities: Vec<f64> = self.statevector
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        let rand_val = fastrand::f64();
        let mut cumulative = 0.0;
        
        for (state_idx, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if rand_val <= cumulative {
                let mut bits = vec![0u8; self.num_qubits as usize];
                let mut idx = state_idx;
                for i in 0..bits.len() {
                    bits[i] = (idx & 1) as u8;
                    idx >>= 1;
                }
                return bits;
            }
        }
        
        vec![0; self.num_qubits as usize]
    }
    
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.statevector
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect()
    }
}

/// Gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    CNOT,
    Toffoli,
    Phase,
    RotationX,
    RotationY, 
    RotationZ,
    QuantumFourierTransform,
    GroverOracle,
}

/// Gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParams {
    pub target_qubits: Vec<u32>,
    pub control_qubits: Vec<u32>,
    pub angles: Vec<f64>,
    pub auxiliary_data: Option<serde_json::Value>,
}

impl GateParams {
    pub fn single_qubit(qubit: u32) -> Self {
        Self {
            target_qubits: vec![qubit],
            control_qubits: Vec::new(),
            angles: Vec::new(),
            auxiliary_data: None,
        }
    }
    
    pub fn two_qubit(control: u32, target: u32) -> Self {
        Self {
            target_qubits: vec![target],
            control_qubits: vec![control],
            angles: Vec::new(),
            auxiliary_data: None,
        }
    }
    
    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angles.push(angle);
        self
    }
}

/// Classical gate operation
#[derive(Debug, Clone)]
pub struct ClassicalGateOperation {
    gate_type: GateType,
    params: GateParams,
}

impl ClassicalGateOperation {
    pub fn new(gate_type: GateType, params: GateParams) -> Self {
        Self { gate_type, params }
    }
}

impl GateOperation for ClassicalGateOperation {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumGateError> {
        match state {
            QuantumState::Classical(classical_state) => {
                self.apply_classical_gate(classical_state)
            }
            _ => Err(QuantumGateError::StateMismatch),
        }
    }
    
    fn gate_type(&self) -> GateType {
        self.gate_type
    }
    
    fn is_reversible(&self) -> bool {
        matches!(self.gate_type, 
            GateType::PauliX | GateType::PauliY | GateType::PauliZ | 
            GateType::CNOT | GateType::Hadamard)
    }
    
    fn computational_cost(&self) -> u64 {
        match self.gate_type {
            GateType::Hadamard | GateType::PauliX | GateType::PauliY | GateType::PauliZ => 1,
            GateType::CNOT => 2,
            GateType::Toffoli => 5,
            _ => 1,
        }
    }
}

impl ClassicalGateOperation {
    fn apply_classical_gate(&self, state: &mut ClassicalState) -> Result<(), QuantumGateError> {
        match self.gate_type {
            GateType::PauliX => {
                if let Some(&qubit) = self.params.target_qubits.first() {
                    if (qubit as usize) < state.bits.len() {
                        state.bits[qubit as usize] = 1 - state.bits[qubit as usize];
                    }
                }
            }
            GateType::Hadamard => {
                // Classical approximation: create superposition in probability distribution
                if let Some(&qubit) = self.params.target_qubits.first() {
                    self.apply_classical_superposition(state, qubit)?;
                }
            }
            GateType::CNOT => {
                if let (Some(&control), Some(&target)) = 
                    (self.params.control_qubits.first(), self.params.target_qubits.first()) {
                    if state.bits[control as usize] == 1 {
                        state.bits[target as usize] = 1 - state.bits[target as usize];
                    }
                }
            }
            _ => {} // Other gates not implemented in classical mode
        }
        Ok(())
    }
    
    fn apply_classical_superposition(&self, state: &mut ClassicalState, qubit: u32) -> Result<(), QuantumGateError> {
        // Create probabilistic superposition
        let qubit_idx = qubit as usize;
        if qubit_idx >= state.bits.len() {
            return Err(QuantumGateError::InvalidQubit(qubit));
        }
        
        // Split probability between |0⟩ and |1⟩ states
        let mask = 1usize << qubit_idx;
        let mut new_probabilities = vec![0.0; state.probabilities.len()];
        
        for (i, &prob) in state.probabilities.iter().enumerate() {
            let prob_half = prob / 2.0;
            new_probabilities[i & !mask] += prob_half;
            new_probabilities[i | mask] += prob_half;
        }
        
        state.probabilities = new_probabilities;
        Ok(())
    }
}

/// Enhanced gate operation with quantum-inspired features
#[derive(Debug, Clone)]
pub struct EnhancedGateOperation {
    gate_type: GateType,
    params: GateParams,
}

impl EnhancedGateOperation {
    pub fn new(gate_type: GateType, params: GateParams) -> Self {
        Self { gate_type, params }
    }
}

impl GateOperation for EnhancedGateOperation {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumGateError> {
        match state {
            QuantumState::Enhanced(enhanced_state) => {
                self.apply_enhanced_gate(enhanced_state)
            }
            _ => Err(QuantumGateError::StateMismatch),
        }
    }
    
    fn gate_type(&self) -> GateType {
        self.gate_type
    }
    
    fn is_reversible(&self) -> bool {
        true // Enhanced gates are generally reversible
    }
    
    fn computational_cost(&self) -> u64 {
        match self.gate_type {
            GateType::QuantumFourierTransform => 100,
            GateType::GroverOracle => 50,
            _ => 5, // Enhanced operations cost more than classical
        }
    }
}

impl EnhancedGateOperation {
    fn apply_enhanced_gate(&self, state: &mut EnhancedState) -> Result<(), QuantumGateError> {
        match self.gate_type {
            GateType::Hadamard => {
                if let Some(&qubit) = self.params.target_qubits.first() {
                    self.apply_hadamard_enhanced(state, qubit)?;
                }
            }
            GateType::PauliX => {
                if let Some(&qubit) = self.params.target_qubits.first() {
                    self.apply_pauli_x_enhanced(state, qubit)?;
                }
            }
            GateType::CNOT => {
                if let (Some(&control), Some(&target)) = 
                    (self.params.control_qubits.first(), self.params.target_qubits.first()) {
                    self.apply_cnot_enhanced(state, control, target)?;
                }
            }
            GateType::QuantumFourierTransform => {
                self.apply_qft_enhanced(state)?;
            }
            _ => {}
        }
        Ok(())
    }
    
    fn apply_hadamard_enhanced(&self, state: &mut EnhancedState, qubit: u32) -> Result<(), QuantumGateError> {
        let sqrt2_inv = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        let mask = 1usize << qubit;
        
        for i in 0..state.amplitudes.len() {
            if i & mask == 0 {
                let j = i | mask;
                let amp0 = state.amplitudes[i];
                let amp1 = state.amplitudes[j];
                
                state.amplitudes[i] = sqrt2_inv * (amp0 + amp1);
                state.amplitudes[j] = sqrt2_inv * (amp0 - amp1);
            }
        }
        
        // Update coherence matrix
        self.update_coherence_matrix(state, qubit);
        
        Ok(())
    }
    
    fn apply_pauli_x_enhanced(&self, state: &mut EnhancedState, qubit: u32) -> Result<(), QuantumGateError> {
        let mask = 1usize << qubit;
        
        for i in 0..state.amplitudes.len() {
            if i & mask == 0 {
                let j = i | mask;
                state.amplitudes.swap(i, j);
            }
        }
        
        Ok(())
    }
    
    fn apply_cnot_enhanced(&self, state: &mut EnhancedState, control: u32, target: u32) -> Result<(), QuantumGateError> {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        
        for i in 0..state.amplitudes.len() {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                state.amplitudes.swap(i, j);
            }
        }
        
        // Update entanglement in coherence matrix
        self.update_entanglement(state, control, target);
        
        Ok(())
    }
    
    fn apply_qft_enhanced(&self, state: &mut EnhancedState) -> Result<(), QuantumGateError> {
        let n = (state.amplitudes.len() as f64).log2() as u32;
        
        // Apply QFT using enhanced quantum-inspired methods
        for k in 0..n {
            // Hadamard gate
            self.apply_hadamard_enhanced(state, k)?;
            
            // Controlled phase rotations
            for j in (k + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - k)) as f64;
                self.apply_controlled_phase_enhanced(state, j, k, angle)?;
            }
        }
        
        Ok(())
    }
    
    fn apply_controlled_phase_enhanced(&self, state: &mut EnhancedState, control: u32, target: u32, angle: f64) -> Result<(), QuantumGateError> {
        let control_mask = 1usize << control;
        let phase = Complex64::new(0.0, angle).exp();
        
        for i in 0..state.amplitudes.len() {
            if (i & control_mask) != 0 {
                state.amplitudes[i] *= phase;
            }
        }
        
        Ok(())
    }
    
    fn update_coherence_matrix(&self, state: &mut EnhancedState, qubit: u32) {
        // Update coherence relationships
        for i in 0..state.coherence_matrix.len() {
            for j in 0..state.coherence_matrix[i].len() {
                let qubit_diff = ((i ^ j) >> qubit) & 1;
                if qubit_diff == 1 {
                    state.coherence_matrix[i][j] *= 0.9; // Reduce coherence
                }
            }
        }
    }
    
    fn update_entanglement(&self, state: &mut EnhancedState, qubit1: u32, qubit2: u32) {
        // Increase entanglement between the two qubits
        let mask1 = 1usize << qubit1;
        let mask2 = 1usize << qubit2;
        
        for i in 0..state.coherence_matrix.len() {
            for j in 0..state.coherence_matrix[i].len() {
                let diff1 = ((i ^ j) & mask1) != 0;
                let diff2 = ((i ^ j) & mask2) != 0;
                
                if diff1 && diff2 {
                    state.coherence_matrix[i][j] += 0.1; // Increase entanglement
                }
            }
        }
    }
}

/// Full quantum gate operation
#[derive(Debug, Clone)]
pub struct QuantumGateOperation {
    gate_type: GateType,
    params: GateParams,
}

impl QuantumGateOperation {
    pub fn new(gate_type: GateType, params: GateParams) -> Self {
        Self { gate_type, params }
    }
}

impl GateOperation for QuantumGateOperation {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumGateError> {
        match state {
            QuantumState::FullQuantum(quantum_state) => {
                self.apply_quantum_gate(quantum_state)
            }
            _ => Err(QuantumGateError::StateMismatch),
        }
    }
    
    fn gate_type(&self) -> GateType {
        self.gate_type
    }
    
    fn is_reversible(&self) -> bool {
        true // All quantum operations are reversible
    }
    
    fn computational_cost(&self) -> u64 {
        match self.gate_type {
            GateType::QuantumFourierTransform => 1000,
            GateType::GroverOracle => 500,
            _ => 10, // Full quantum operations are most expensive
        }
    }
}

impl QuantumGateOperation {
    fn apply_quantum_gate(&self, state: &mut FullQuantumState) -> Result<(), QuantumGateError> {
        // Apply decoherence before gate operation
        self.apply_decoherence(state);
        
        match self.gate_type {
            GateType::Hadamard => {
                if let Some(&qubit) = self.params.target_qubits.first() {
                    self.apply_hadamard_quantum(state, qubit)?;
                }
            }
            GateType::PauliX => {
                if let Some(&qubit) = self.params.target_qubits.first() {
                    self.apply_pauli_x_quantum(state, qubit)?;
                }
            }
            GateType::CNOT => {
                if let (Some(&control), Some(&target)) = 
                    (self.params.control_qubits.first(), self.params.target_qubits.first()) {
                    self.apply_cnot_quantum(state, control, target)?;
                }
            }
            GateType::GroverOracle => {
                self.apply_grover_oracle_quantum(state)?;
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn apply_hadamard_quantum(&self, state: &mut FullQuantumState, qubit: u32) -> Result<(), QuantumGateError> {
        let sqrt2_inv = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        let mask = 1usize << qubit;
        
        for i in 0..state.statevector.len() {
            if i & mask == 0 {
                let j = i | mask;
                let amp0 = state.statevector[i];
                let amp1 = state.statevector[j];
                
                state.statevector[i] = sqrt2_inv * (amp0 + amp1);
                state.statevector[j] = sqrt2_inv * (amp0 - amp1);
            }
        }
        
        Ok(())
    }
    
    fn apply_pauli_x_quantum(&self, state: &mut FullQuantumState, qubit: u32) -> Result<(), QuantumGateError> {
        let mask = 1usize << qubit;
        
        for i in 0..state.statevector.len() {
            if i & mask == 0 {
                let j = i | mask;
                state.statevector.swap(i, j);
            }
        }
        
        Ok(())
    }
    
    fn apply_cnot_quantum(&self, state: &mut FullQuantumState, control: u32, target: u32) -> Result<(), QuantumGateError> {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;
        
        for i in 0..state.statevector.len() {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                state.statevector.swap(i, j);
            }
        }
        
        Ok(())
    }
    
    fn apply_grover_oracle_quantum(&self, state: &mut FullQuantumState) -> Result<(), QuantumGateError> {
        // Grover oracle flips the amplitude of marked states
        // For demonstration, mark the last state
        let marked_state = state.statevector.len() - 1;
        state.statevector[marked_state] = -state.statevector[marked_state];
        
        Ok(())
    }
    
    fn apply_decoherence(&self, state: &mut FullQuantumState) {
        // Simple decoherence model
        let decoherence_factor = (-0.01 / state.decoherence_time).exp();
        
        for amplitude in &mut state.statevector {
            *amplitude *= decoherence_factor;
        }
        
        // Renormalize
        let norm: f64 = state.statevector.iter().map(|a| a.norm_sqr()).sum();
        let norm_sqrt = norm.sqrt();
        
        if norm_sqrt > 0.0 {
            for amplitude in &mut state.statevector {
                *amplitude /= norm_sqrt;
            }
        }
    }
}

/// Quantum gate errors
#[derive(Debug, thiserror::Error)]
pub enum QuantumGateError {
    #[error("State type mismatch for gate operation")]
    StateMismatch,
    #[error("Invalid qubit index: {0}")]
    InvalidQubit(u32),
    #[error("Gate operation failed: {0}")]
    OperationFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::QuantumMode;
    
    #[test]
    fn test_quantum_state_creation() {
        QuantumMode::set_global(QuantumMode::Classical);
        let state = QuantumState::new(2);
        assert!(matches!(state, QuantumState::Classical(_)));
        
        QuantumMode::set_global(QuantumMode::Enhanced);
        let state = QuantumState::new(2);
        assert!(matches!(state, QuantumState::Enhanced(_)));
        
        QuantumMode::set_global(QuantumMode::Full);
        let state = QuantumState::new(2);
        assert!(matches!(state, QuantumState::FullQuantum(_)));
        
        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }
    
    #[test]
    fn test_gate_factory() {
        QuantumMode::set_global(QuantumMode::Classical);
        let gate = QuantumGateFactory::create_gate_operation(
            GateType::Hadamard,
            GateParams::single_qubit(0)
        );
        
        assert_eq!(gate.gate_type(), GateType::Hadamard);
        assert!(gate.computational_cost() > 0);
        
        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }
    
    #[test]
    fn test_classical_gate_operation() {
        let mut state = QuantumState::Classical(ClassicalState::new(1));
        let gate = ClassicalGateOperation::new(
            GateType::PauliX,
            GateParams::single_qubit(0)
        );
        
        // Apply X gate - should flip the bit
        let result = gate.apply(&mut state);
        assert!(result.is_ok());
        
        let measurements = state.measure();
        assert_eq!(measurements[0], 1);
    }
    
    #[test]
    fn test_enhanced_gate_operation() {
        let mut state = QuantumState::Enhanced(EnhancedState::new(1));
        let gate = EnhancedGateOperation::new(
            GateType::Hadamard,
            GateParams::single_qubit(0)
        );
        
        let result = gate.apply(&mut state);
        assert!(result.is_ok());
        
        let probabilities = state.get_probabilities();
        // After Hadamard, should have equal probabilities for |0⟩ and |1⟩
        assert!((probabilities[0] - 0.5).abs() < 0.01);
        assert!((probabilities[1] - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_quantum_gate_operation() {
        let mut state = QuantumState::FullQuantum(FullQuantumState::new(1));
        let gate = QuantumGateOperation::new(
            GateType::Hadamard,
            GateParams::single_qubit(0)
        );
        
        let result = gate.apply(&mut state);
        assert!(result.is_ok());
        
        let probabilities = state.get_probabilities();
        assert!((probabilities[0] - 0.5).abs() < 0.01);
        assert!((probabilities[1] - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_gate_params() {
        let params = GateParams::single_qubit(0);
        assert_eq!(params.target_qubits[0], 0);
        
        let params = GateParams::two_qubit(0, 1);
        assert_eq!(params.control_qubits[0], 0);
        assert_eq!(params.target_qubits[0], 1);
        
        let params = GateParams::single_qubit(0).with_angle(std::f64::consts::PI);
        assert_eq!(params.angles[0], std::f64::consts::PI);
    }
}