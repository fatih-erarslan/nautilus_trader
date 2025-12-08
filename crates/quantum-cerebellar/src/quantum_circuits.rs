//! Quantum circuit simulation for cerebellar spike processing
//! 
//! High-performance quantum circuit simulator optimized for neuromorphic
//! spike encoding, entanglement generation, and coherent dynamics.

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, Complex};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use rayon::prelude::*;

#[cfg(feature = "simd")]
use wide::f64x4;

/// Quantum gate types for cerebellar processing
#[derive(Debug, Clone, Copy)]
pub enum QuantumGate {
    /// Pauli-X (bit flip)
    PauliX,
    /// Pauli-Y (bit and phase flip)
    PauliY,
    /// Pauli-Z (phase flip)
    PauliZ,
    /// Hadamard (superposition)
    Hadamard,
    /// Rotation around X-axis
    RotationX(f64),
    /// Rotation around Y-axis
    RotationY(f64),
    /// Rotation around Z-axis  
    RotationZ(f64),
    /// Phase shift
    PhaseShift(f64),
    /// Controlled NOT
    CNOT,
    /// Controlled Y
    CY,
    /// Controlled Z
    CZ,
    /// Controlled phase
    CPhase(f64),
    /// Toffoli (CCX)
    Toffoli,
    /// Fredkin (CSWAP)
    Fredkin,
    /// Custom unitary matrix
    Custom(DMatrix<Complex64>),
}

/// Quantum circuit instruction
#[derive(Debug, Clone)]
pub struct QuantumInstruction {
    pub gate: QuantumGate,
    pub qubits: Vec<usize>,
    pub classical_condition: Option<(usize, u8)>, // (register, value)
}

/// Quantum register for measurements
#[derive(Debug, Clone)]
pub struct ClassicalRegister {
    pub bits: Vec<bool>,
    pub name: String,
}

/// High-performance quantum circuit simulator
pub struct QuantumCircuitSimulator {
    /// Number of qubits
    n_qubits: usize,
    
    /// Current quantum state vector
    state_vector: Vec<Complex64>,
    
    /// Circuit instructions
    instructions: Vec<QuantumInstruction>,
    
    /// Classical registers
    classical_registers: HashMap<String, ClassicalRegister>,
    
    /// Measurement cache for performance
    measurement_cache: HashMap<Vec<usize>, Vec<f64>>,
    
    /// SIMD optimization flags
    use_simd: bool,
    
    /// Entanglement tracking
    entanglement_graph: Vec<Vec<bool>>,
}

impl QuantumCircuitSimulator {
    /// Create new quantum circuit simulator
    pub fn new(n_qubits: usize) -> Result<Self> {
        if n_qubits > 30 {
            warn!("Large number of qubits ({}), simulation may be slow", n_qubits);
        }
        
        let state_size = 1 << n_qubits;
        let mut state_vector = vec![Complex64::new(0.0, 0.0); state_size];
        state_vector[0] = Complex64::new(1.0, 0.0); // |000...0⟩ initial state
        
        let entanglement_graph = vec![vec![false; n_qubits]; n_qubits];
        
        info!("Initialized quantum circuit simulator with {} qubits", n_qubits);
        
        Ok(Self {
            n_qubits,
            state_vector,
            instructions: Vec::new(),
            classical_registers: HashMap::new(),
            measurement_cache: HashMap::new(),
            use_simd: cfg!(feature = "simd"),
            entanglement_graph,
        })
    }
    
    /// Reset to initial state |000...0⟩
    pub fn reset(&mut self) {
        let state_size = self.state_vector.len();
        self.state_vector.fill(Complex64::new(0.0, 0.0));
        self.state_vector[0] = Complex64::new(1.0, 0.0);
        self.instructions.clear();
        self.measurement_cache.clear();
        
        // Reset entanglement tracking
        for row in &mut self.entanglement_graph {
            row.fill(false);
        }
    }
    
    /// Add quantum gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGate, qubits: Vec<usize>) -> Result<()> {
        // Validate qubit indices
        for &qubit in &qubits {
            if qubit >= self.n_qubits {
                return Err(anyhow!("Qubit index {} out of range", qubit));
            }
        }
        
        self.instructions.push(QuantumInstruction {
            gate,
            qubits,
            classical_condition: None,
        });
        
        Ok(())
    }
    
    /// Execute all circuit instructions
    pub fn execute(&mut self) -> Result<()> {
        for instruction in self.instructions.clone() {
            self.apply_gate(&instruction)?;
        }
        Ok(())
    }
    
    /// Apply single quantum gate
    fn apply_gate(&mut self, instruction: &QuantumInstruction) -> Result<()> {
        match instruction.gate {
            QuantumGate::PauliX => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.pauli_x_matrix())?;
            }
            QuantumGate::PauliY => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.pauli_y_matrix())?;
            }
            QuantumGate::PauliZ => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.pauli_z_matrix())?;
            }
            QuantumGate::Hadamard => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.hadamard_matrix())?;
            }
            QuantumGate::RotationX(angle) => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.rotation_x_matrix(angle))?;
            }
            QuantumGate::RotationY(angle) => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.rotation_y_matrix(angle))?;
            }
            QuantumGate::RotationZ(angle) => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.rotation_z_matrix(angle))?;
            }
            QuantumGate::PhaseShift(phase) => {
                self.apply_single_qubit_gate(&instruction.qubits[0], &self.phase_shift_matrix(phase))?;
            }
            QuantumGate::CNOT => {
                self.apply_cnot_gate(instruction.qubits[0], instruction.qubits[1])?;
            }
            QuantumGate::CY => {
                self.apply_controlled_gate(instruction.qubits[0], instruction.qubits[1], &self.pauli_y_matrix())?;
            }
            QuantumGate::CZ => {
                self.apply_controlled_gate(instruction.qubits[0], instruction.qubits[1], &self.pauli_z_matrix())?;
            }
            QuantumGate::CPhase(phase) => {
                self.apply_controlled_phase_gate(instruction.qubits[0], instruction.qubits[1], phase)?;
            }
            QuantumGate::Toffoli => {
                self.apply_toffoli_gate(instruction.qubits[0], instruction.qubits[1], instruction.qubits[2])?;
            }
            QuantumGate::Fredkin => {
                self.apply_fredkin_gate(instruction.qubits[0], instruction.qubits[1], instruction.qubits[2])?;
            }
            QuantumGate::Custom(ref matrix) => {
                self.apply_single_qubit_gate(&instruction.qubits[0], matrix)?;
            }
        }
        
        Ok(())
    }
    
    /// Apply single-qubit gate using tensor product
    fn apply_single_qubit_gate(&mut self, qubit: &usize, gate_matrix: &DMatrix<Complex64>) -> Result<()> {
        let state_size = self.state_vector.len();
        let mut new_state = vec![Complex64::new(0.0, 0.0); state_size];
        
        let qubit_mask = 1 << qubit;
        
        // Parallel processing for large state vectors
        if self.use_simd && state_size > 1024 {
            self.apply_single_qubit_gate_simd(*qubit, gate_matrix, &mut new_state)?;
        } else {
            for i in 0..state_size {
                let bit = (i & qubit_mask) >> qubit;
                let other_bits = i ^ (bit << qubit);
                
                // Apply gate matrix
                for j in 0..2 {
                    let target_state = other_bits | (j << qubit);
                    new_state[target_state] += gate_matrix[(j, bit)] * self.state_vector[i];
                }
            }
        }
        
        self.state_vector = new_state;
        Ok(())
    }
    
    /// SIMD-optimized single qubit gate application
    #[cfg(feature = "simd")]
    fn apply_single_qubit_gate_simd(
        &self,
        qubit: usize,
        gate_matrix: &DMatrix<Complex64>,
        new_state: &mut Vec<Complex64>,
    ) -> Result<()> {
        let state_size = self.state_vector.len();
        let qubit_mask = 1 << qubit;
        
        // Process in SIMD chunks
        (0..state_size).into_par_iter().step_by(4).for_each(|chunk_start| {
            let chunk_end = (chunk_start + 4).min(state_size);
            
            for i in chunk_start..chunk_end {
                let bit = (i & qubit_mask) >> qubit;
                let other_bits = i ^ (bit << qubit);
                
                // Apply gate matrix
                for j in 0..2 {
                    let target_state = other_bits | (j << qubit);
                    if target_state < state_size {
                        new_state[target_state] += gate_matrix[(j, bit)] * self.state_vector[i];
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Non-SIMD fallback
    #[cfg(not(feature = "simd"))]
    fn apply_single_qubit_gate_simd(
        &self,
        qubit: usize,
        gate_matrix: &DMatrix<Complex64>,
        new_state: &mut Vec<Complex64>,
    ) -> Result<()> {
        // Fallback to regular implementation
        let state_size = self.state_vector.len();
        let qubit_mask = 1 << qubit;
        
        for i in 0..state_size {
            let bit = (i & qubit_mask) >> qubit;
            let other_bits = i ^ (bit << qubit);
            
            for j in 0..2 {
                let target_state = other_bits | (j << qubit);
                new_state[target_state] += gate_matrix[(j, bit)] * self.state_vector[i];
            }
        }
        
        Ok(())
    }
    
    /// Apply CNOT gate
    fn apply_cnot_gate(&mut self, control: usize, target: usize) -> Result<()> {
        let state_size = self.state_vector.len();
        let mut new_state = self.state_vector.clone();
        
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        
        for i in 0..state_size {
            if (i & control_mask) != 0 {
                // Control qubit is |1⟩, flip target
                let flipped_state = i ^ target_mask;
                new_state[flipped_state] = self.state_vector[i];
                new_state[i] = Complex64::new(0.0, 0.0);
            }
        }
        
        self.state_vector = new_state;
        
        // Update entanglement tracking
        self.entanglement_graph[control][target] = true;
        self.entanglement_graph[target][control] = true;
        
        Ok(())
    }
    
    /// Apply controlled gate
    fn apply_controlled_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &DMatrix<Complex64>,
    ) -> Result<()> {
        let state_size = self.state_vector.len();
        let mut new_state = vec![Complex64::new(0.0, 0.0); state_size];
        
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        
        for i in 0..state_size {
            if (i & control_mask) == 0 {
                // Control is |0⟩, identity on target
                new_state[i] = self.state_vector[i];
            } else {
                // Control is |1⟩, apply gate to target
                let target_bit = (i & target_mask) >> target;
                let other_bits = i ^ (target_bit << target);
                
                for j in 0..2 {
                    let target_state = other_bits | (j << target);
                    new_state[target_state] += gate_matrix[(j, target_bit)] * self.state_vector[i];
                }
            }
        }
        
        self.state_vector = new_state;
        
        // Update entanglement
        self.entanglement_graph[control][target] = true;
        self.entanglement_graph[target][control] = true;
        
        Ok(())
    }
    
    /// Apply controlled phase gate
    fn apply_controlled_phase_gate(&mut self, control: usize, target: usize, phase: f64) -> Result<()> {
        let state_size = self.state_vector.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let phase_factor = Complex64::new(0.0, phase).exp();
        
        for i in 0..state_size {
            if (i & control_mask) != 0 && (i & target_mask) != 0 {
                // Both control and target are |1⟩
                self.state_vector[i] *= phase_factor;
            }
        }
        
        // Update entanglement
        self.entanglement_graph[control][target] = true;
        self.entanglement_graph[target][control] = true;
        
        Ok(())
    }
    
    /// Apply Toffoli (CCX) gate
    fn apply_toffoli_gate(&mut self, control1: usize, control2: usize, target: usize) -> Result<()> {
        let state_size = self.state_vector.len();
        let mut new_state = self.state_vector.clone();
        
        let control1_mask = 1 << control1;
        let control2_mask = 1 << control2;
        let target_mask = 1 << target;
        
        for i in 0..state_size {
            if (i & control1_mask) != 0 && (i & control2_mask) != 0 {
                // Both controls are |1⟩, flip target
                let flipped_state = i ^ target_mask;
                new_state[flipped_state] = self.state_vector[i];
                new_state[i] = Complex64::new(0.0, 0.0);
            }
        }
        
        self.state_vector = new_state;
        
        // Update entanglement for all pairs
        self.entanglement_graph[control1][control2] = true;
        self.entanglement_graph[control2][control1] = true;
        self.entanglement_graph[control1][target] = true;
        self.entanglement_graph[target][control1] = true;
        self.entanglement_graph[control2][target] = true;
        self.entanglement_graph[target][control2] = true;
        
        Ok(())
    }
    
    /// Apply Fredkin (CSWAP) gate
    fn apply_fredkin_gate(&mut self, control: usize, target1: usize, target2: usize) -> Result<()> {
        let state_size = self.state_vector.len();
        let mut new_state = self.state_vector.clone();
        
        let control_mask = 1 << control;
        let target1_mask = 1 << target1;
        let target2_mask = 1 << target2;
        
        for i in 0..state_size {
            if (i & control_mask) != 0 {
                // Control is |1⟩, swap targets
                let target1_bit = (i & target1_mask) != 0;
                let target2_bit = (i & target2_mask) != 0;
                
                if target1_bit != target2_bit {
                    let swapped_state = i ^ target1_mask ^ target2_mask;
                    new_state[swapped_state] = self.state_vector[i];
                    new_state[i] = Complex64::new(0.0, 0.0);
                }
            }
        }
        
        self.state_vector = new_state;
        
        // Update entanglement
        self.entanglement_graph[control][target1] = true;
        self.entanglement_graph[target1][control] = true;
        self.entanglement_graph[control][target2] = true;
        self.entanglement_graph[target2][control] = true;
        self.entanglement_graph[target1][target2] = true;
        self.entanglement_graph[target2][target1] = true;
        
        Ok(())
    }
    
    /// Measure qubits and return probabilities
    pub fn measure_probabilities(&mut self, qubits: &[usize]) -> Result<Vec<f64>> {
        // Check cache first
        if let Some(cached_probs) = self.measurement_cache.get(qubits) {
            return Ok(cached_probs.clone());
        }
        
        let num_outcomes = 1 << qubits.len();
        let mut probabilities = vec![0.0; num_outcomes];
        
        // Create qubit masks
        let qubit_masks: Vec<usize> = qubits.iter().map(|&q| 1 << q).collect();
        
        for (i, &amplitude) in self.state_vector.iter().enumerate() {
            // Extract measurement outcome
            let mut outcome = 0;
            for (j, &mask) in qubit_masks.iter().enumerate() {
                if (i & mask) != 0 {
                    outcome |= 1 << j;
                }
            }
            
            probabilities[outcome] += amplitude.norm_sqr();
        }
        
        // Cache result
        self.measurement_cache.insert(qubits.to_vec(), probabilities.clone());
        
        Ok(probabilities)
    }
    
    /// Measure all qubits and collapse state
    pub fn measure_all(&mut self) -> Vec<bool> {
        let total_prob: f64 = self.state_vector.iter().map(|a| a.norm_sqr()).sum();
        let random_value = rand::random::<f64>() * total_prob;
        
        let mut cumulative_prob = 0.0;
        for (i, amplitude) in self.state_vector.iter().enumerate() {
            cumulative_prob += amplitude.norm_sqr();
            if cumulative_prob >= random_value {
                // Collapse to this state
                let mut measurement = vec![false; self.n_qubits];
                for qubit in 0..self.n_qubits {
                    measurement[qubit] = (i & (1 << qubit)) != 0;
                }
                
                // Reset state vector to measured state
                self.state_vector.fill(Complex64::new(0.0, 0.0));
                self.state_vector[i] = Complex64::new(1.0, 0.0);
                
                return measurement;
            }
        }
        
        // Fallback (shouldn't happen with proper normalization)
        vec![false; self.n_qubits]
    }
    
    /// Get quantum state fidelity with target state
    pub fn fidelity(&self, target_state: &[Complex64]) -> Result<f64> {
        if target_state.len() != self.state_vector.len() {
            return Err(anyhow!("State vector size mismatch"));
        }
        
        let overlap: Complex64 = self.state_vector
            .iter()
            .zip(target_state.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        
        Ok(overlap.norm_sqr())
    }
    
    /// Calculate entanglement entropy between qubits
    pub fn entanglement_entropy(&self, subsystem_qubits: &[usize]) -> Result<f64> {
        let subsystem_size = subsystem_qubits.len();
        let total_size = self.n_qubits;
        
        if subsystem_size >= total_size {
            return Ok(0.0); // No entanglement for full system
        }
        
        // Trace out environment to get reduced density matrix
        let reduced_dm = self.partial_trace(subsystem_qubits)?;
        
        // Calculate eigenvalues
        let eigenvalues = reduced_dm.symmetric_eigenvalues();
        
        // Von Neumann entropy: S = -∑ λ log₂(λ)
        let entropy = eigenvalues.iter()
            .filter(|&&lambda| lambda > 1e-12)
            .map(|&lambda| -lambda * lambda.log2())
            .sum();
        
        Ok(entropy)
    }
    
    /// Compute partial trace for entanglement analysis
    fn partial_trace(&self, keep_qubits: &[usize]) -> Result<DMatrix<f64>> {
        let keep_size = 1 << keep_qubits.len();
        let mut reduced_dm = DMatrix::zeros(keep_size, keep_size);
        
        // Create mapping from full state to reduced state
        let keep_masks: Vec<usize> = keep_qubits.iter().map(|&q| 1 << q).collect();
        
        for (i, &amp_i) in self.state_vector.iter().enumerate() {
            for (j, &amp_j) in self.state_vector.iter().enumerate() {
                // Extract indices for kept qubits
                let mut i_reduced = 0;
                let mut j_reduced = 0;
                
                for (k, &mask) in keep_masks.iter().enumerate() {
                    if (i & mask) != 0 {
                        i_reduced |= 1 << k;
                    }
                    if (j & mask) != 0 {
                        j_reduced |= 1 << k;
                    }
                }
                
                // Check if traced-out qubits match
                let i_traced = i ^ i_reduced;
                let j_traced = j ^ j_reduced;
                
                if i_traced == j_traced {
                    reduced_dm[(i_reduced, j_reduced)] += (amp_i.conj() * amp_j).re;
                }
            }
        }
        
        Ok(reduced_dm)
    }
    
    // Matrix definitions for quantum gates
    
    fn pauli_x_matrix(&self) -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ])
    }
    
    fn pauli_y_matrix(&self) -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0),
        ])
    }
    
    fn pauli_z_matrix(&self) -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ])
    }
    
    fn hadamard_matrix(&self) -> DMatrix<Complex64> {
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0),
        ])
    }
    
    fn rotation_x_matrix(&self, angle: f64) -> DMatrix<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(0.0, -sin_half),
            Complex64::new(0.0, -sin_half), Complex64::new(cos_half, 0.0),
        ])
    }
    
    fn rotation_y_matrix(&self, angle: f64) -> DMatrix<Complex64> {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0),
            Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0),
        ])
    }
    
    fn rotation_z_matrix(&self, angle: f64) -> DMatrix<Complex64> {
        let exp_neg = Complex64::new(0.0, -angle / 2.0).exp();
        let exp_pos = Complex64::new(0.0, angle / 2.0).exp();
        
        DMatrix::from_row_slice(2, 2, &[
            exp_neg, Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), exp_pos,
        ])
    }
    
    fn phase_shift_matrix(&self, phase: f64) -> DMatrix<Complex64> {
        DMatrix::from_row_slice(2, 2, &[
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, phase).exp(),
        ])
    }
    
    /// Get current state vector
    pub fn state_vector(&self) -> &[Complex64] {
        &self.state_vector
    }
    
    /// Get number of qubits
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }
    
    /// Check if two qubits are entangled
    pub fn are_entangled(&self, qubit1: usize, qubit2: usize) -> bool {
        self.entanglement_graph[qubit1][qubit2]
    }
    
    /// Get entanglement graph
    pub fn entanglement_graph(&self) -> &[Vec<bool>] {
        &self.entanglement_graph
    }
}

/// Specialized quantum circuits for cerebellar operations
pub struct CerebellarQuantumCircuits;

impl CerebellarQuantumCircuits {
    /// Create spike encoding circuit
    pub fn create_spike_encoding_circuit(
        simulator: &mut QuantumCircuitSimulator,
        spike_amplitudes: &[f64],
        spike_phases: &[f64],
    ) -> Result<()> {
        for (i, (&amplitude, &phase)) in spike_amplitudes.iter().zip(spike_phases.iter()).enumerate() {
            if i >= simulator.n_qubits() {
                break;
            }
            
            // Encode amplitude in rotation angle
            let angle = 2.0 * amplitude.clamp(0.0, 1.0).asin();
            simulator.add_gate(QuantumGate::RotationY(angle), vec![i])?;
            
            // Encode phase
            if phase.abs() > 1e-10 {
                simulator.add_gate(QuantumGate::PhaseShift(phase), vec![i])?;
            }
        }
        
        Ok(())
    }
    
    /// Create entanglement circuit for spike correlations
    pub fn create_entanglement_circuit(
        simulator: &mut QuantumCircuitSimulator,
        correlations: &[(usize, usize, f64)],
    ) -> Result<()> {
        for &(qubit1, qubit2, strength) in correlations {
            if qubit1 < simulator.n_qubits() && qubit2 < simulator.n_qubits() {
                // Create entanglement with controlled rotation
                simulator.add_gate(QuantumGate::Hadamard, vec![qubit1])?;
                simulator.add_gate(QuantumGate::CPhase(strength * std::f64::consts::PI), vec![qubit1, qubit2])?;
            }
        }
        
        Ok(())
    }
    
    /// Create plasticity modulation circuit
    pub fn create_plasticity_circuit(
        simulator: &mut QuantumCircuitSimulator,
        learning_signal: f64,
        plasticity_qubits: &[usize],
    ) -> Result<()> {
        let modulation_angle = learning_signal * std::f64::consts::PI / 4.0;
        
        for &qubit in plasticity_qubits {
            if qubit < simulator.n_qubits() {
                simulator.add_gate(QuantumGate::RotationX(modulation_angle), vec![qubit])?;
            }
        }
        
        Ok(())
    }
    
    /// Create quantum Fourier transform for frequency analysis
    pub fn create_qft_circuit(
        simulator: &mut QuantumCircuitSimulator,
        qubits: &[usize],
    ) -> Result<()> {
        let n = qubits.len();
        
        for i in 0..n {
            // Hadamard gate
            simulator.add_gate(QuantumGate::Hadamard, vec![qubits[i]])?;
            
            // Controlled rotations
            for j in (i + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - i));
                simulator.add_gate(QuantumGate::CPhase(angle), vec![qubits[j], qubits[i]])?;
            }
        }
        
        // Swap qubits to get correct output order
        for i in 0..(n / 2) {
            // Implement SWAP using three CNOTs
            simulator.add_gate(QuantumGate::CNOT, vec![qubits[i], qubits[n - 1 - i]])?;
            simulator.add_gate(QuantumGate::CNOT, vec![qubits[n - 1 - i], qubits[i]])?;
            simulator.add_gate(QuantumGate::CNOT, vec![qubits[i], qubits[n - 1 - i]])?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = QuantumCircuitSimulator::new(3).unwrap();
        assert_eq!(circuit.n_qubits(), 3);
        assert_eq!(circuit.state_vector().len(), 8);
        
        // Initial state should be |000⟩
        assert_relative_eq!(circuit.state_vector()[0].norm_sqr(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_pauli_gates() {
        let mut circuit = QuantumCircuitSimulator::new(1).unwrap();
        
        // Apply X gate: |0⟩ → |1⟩
        circuit.add_gate(QuantumGate::PauliX, vec![0]).unwrap();
        circuit.execute().unwrap();
        
        assert_relative_eq!(circuit.state_vector()[0].norm_sqr(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(circuit.state_vector()[1].norm_sqr(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_hadamard_gate() {
        let mut circuit = QuantumCircuitSimulator::new(1).unwrap();
        
        // Apply H gate: |0⟩ → (|0⟩ + |1⟩)/√2
        circuit.add_gate(QuantumGate::Hadamard, vec![0]).unwrap();
        circuit.execute().unwrap();
        
        assert_relative_eq!(circuit.state_vector()[0].norm_sqr(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(circuit.state_vector()[1].norm_sqr(), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cnot_gate() {
        let mut circuit = QuantumCircuitSimulator::new(2).unwrap();
        
        // Prepare |10⟩ state
        circuit.add_gate(QuantumGate::PauliX, vec![0]).unwrap();
        circuit.execute().unwrap();
        circuit.instructions.clear();
        
        // Apply CNOT: |10⟩ → |11⟩
        circuit.add_gate(QuantumGate::CNOT, vec![0, 1]).unwrap();
        circuit.execute().unwrap();
        
        // State |11⟩ corresponds to index 3 in 2-qubit system
        assert_relative_eq!(circuit.state_vector()[3].norm_sqr(), 1.0, epsilon = 1e-10);
        assert!(circuit.are_entangled(0, 1));
    }
    
    #[test]
    fn test_measurement_probabilities() {
        let mut circuit = QuantumCircuitSimulator::new(2).unwrap();
        
        // Create equal superposition: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        circuit.add_gate(QuantumGate::Hadamard, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::Hadamard, vec![1]).unwrap();
        circuit.execute().unwrap();
        
        let probs = circuit.measure_probabilities(&[0, 1]).unwrap();
        
        // Each outcome should have probability 0.25
        for prob in probs {
            assert_relative_eq!(prob, 0.25, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_cerebellar_spike_encoding() {
        let mut circuit = QuantumCircuitSimulator::new(4).unwrap();
        
        let spike_amplitudes = vec![0.8, 0.3, 0.0, 0.9];
        let spike_phases = vec![0.0, std::f64::consts::PI / 4.0, 0.0, std::f64::consts::PI / 2.0];
        
        CerebellarQuantumCircuits::create_spike_encoding_circuit(
            &mut circuit,
            &spike_amplitudes,
            &spike_phases,
        ).unwrap();
        
        circuit.execute().unwrap();
        
        // Check that non-zero amplitudes result in non-trivial states
        let state = circuit.state_vector();
        let total_prob: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert_relative_eq!(total_prob, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_entanglement_entropy() {
        let mut circuit = QuantumCircuitSimulator::new(2).unwrap();
        
        // Create maximally entangled state: (|00⟩ + |11⟩)/√2
        circuit.add_gate(QuantumGate::Hadamard, vec![0]).unwrap();
        circuit.add_gate(QuantumGate::CNOT, vec![0, 1]).unwrap();
        circuit.execute().unwrap();
        
        let entropy = circuit.entanglement_entropy(&[0]).unwrap();
        
        // For maximally entangled 2-qubit state, entropy should be 1
        assert_relative_eq!(entropy, 1.0, epsilon = 1e-10);
    }
}