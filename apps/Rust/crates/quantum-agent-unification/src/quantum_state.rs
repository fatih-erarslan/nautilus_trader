//! Quantum state management and quantum bit operations

use num_complex::Complex64;
use nalgebra::{Matrix2, Vector2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Quantum bit representation with amplitude and phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumBit {
    /// Amplitude for |0⟩ state
    pub alpha: Complex64,
    /// Amplitude for |1⟩ state  
    pub beta: Complex64,
}

impl QuantumBit {
    /// Create a new quantum bit in |0⟩ state
    pub fn new() -> Self {
        Self {
            alpha: Complex64::new(1.0, 0.0),
            beta: Complex64::new(0.0, 0.0),
        }
    }
    
    /// Create quantum bit from angles (Bloch sphere representation)
    pub fn from_angles(theta: f64, phi: f64) -> Self {
        let half_theta = theta / 2.0;
        Self {
            alpha: Complex64::new(half_theta.cos(), 0.0),
            beta: Complex64::new(half_theta.sin() * phi.cos(), half_theta.sin() * phi.sin()),
        }
    }
    
    /// Apply rotation around X-axis (Pauli-X gate variation)
    pub fn rotate_x(&mut self, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        let new_alpha = self.alpha * cos_half - Complex64::i() * self.beta * sin_half;
        let new_beta = self.beta * cos_half - Complex64::i() * self.alpha * sin_half;
        
        self.alpha = new_alpha;
        self.beta = new_beta;
        self.normalize();
    }
    
    /// Apply rotation around Y-axis (Pauli-Y gate variation)
    pub fn rotate_y(&mut self, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        let new_alpha = self.alpha * cos_half - self.beta * sin_half;
        let new_beta = self.beta * cos_half + self.alpha * sin_half;
        
        self.alpha = new_alpha;
        self.beta = new_beta;
        self.normalize();
    }
    
    /// Apply rotation around Z-axis (Pauli-Z gate variation)
    pub fn rotate_z(&mut self, angle: f64) {
        let phase = Complex64::new(0.0, -angle / 2.0).exp();
        self.alpha *= phase.conj();
        self.beta *= phase;
        self.normalize();
    }
    
    /// Measure the quantum bit (collapse to classical state)
    pub fn measure(&mut self) -> bool {
        let prob_zero = self.alpha.norm_sqr();
        let random_val: f64 = rand::random();
        
        if random_val < prob_zero {
            self.alpha = Complex64::new(1.0, 0.0);
            self.beta = Complex64::new(0.0, 0.0);
            false // |0⟩ state
        } else {
            self.alpha = Complex64::new(0.0, 0.0);
            self.beta = Complex64::new(1.0, 0.0);
            true // |1⟩ state
        }
    }
    
    /// Get probability of measuring |0⟩
    pub fn prob_zero(&self) -> f64 {
        self.alpha.norm_sqr()
    }
    
    /// Get probability of measuring |1⟩
    pub fn prob_one(&self) -> f64 {
        self.beta.norm_sqr()
    }
    
    /// Normalize the quantum state
    fn normalize(&mut self) {
        let norm = (self.alpha.norm_sqr() + self.beta.norm_sqr()).sqrt();
        if norm > 0.0 {
            self.alpha /= norm;
            self.beta /= norm;
        }
    }
    
    /// Get Bloch sphere coordinates
    pub fn bloch_coordinates(&self) -> (f64, f64, f64) {
        let x = 2.0 * (self.alpha.conj() * self.beta).re;
        let y = 2.0 * (self.alpha.conj() * self.beta).im;
        let z = self.alpha.norm_sqr() - self.beta.norm_sqr();
        (x, y, z)
    }
}

/// Bloch sphere representation for quantum visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlochSphere {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub theta: f64,
    pub phi: f64,
}

impl BlochSphere {
    pub fn from_qubit(qubit: &QuantumBit) -> Self {
        let (x, y, z) = qubit.bloch_coordinates();
        let theta = z.acos();
        let phi = y.atan2(x);
        
        Self { x, y, z, theta, phi }
    }
    
    pub fn to_qubit(&self) -> QuantumBit {
        QuantumBit::from_angles(self.theta, self.phi)
    }
}

/// Multi-qubit quantum state with entanglement support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub qubits: Vec<QuantumBit>,
    pub entangled_pairs: Vec<(usize, usize)>,
    pub coherence_time: f64,
    pub phase: f64,
}

impl QuantumState {
    /// Create new quantum state with n qubits
    pub fn new(n_qubits: usize) -> Self {
        Self {
            qubits: vec![QuantumBit::new(); n_qubits],
            entangled_pairs: Vec::new(),
            coherence_time: 1.0,
            phase: 0.0,
        }
    }
    
    /// Create superposition state for all qubits
    pub fn create_superposition(&mut self) {
        for qubit in &mut self.qubits {
            qubit.rotate_y(PI / 2.0); // |+⟩ state
        }
    }
    
    /// Entangle two qubits (Bell state creation)
    pub fn entangle(&mut self, i: usize, j: usize) -> Result<(), String> {
        if i >= self.qubits.len() || j >= self.qubits.len() {
            return Err("Qubit index out of bounds".to_string());
        }
        
        // Create Bell state |00⟩ + |11⟩
        self.qubits[i].rotate_y(PI / 2.0); // Hadamard-like
        
        // CNOT-like operation
        if self.qubits[i].prob_one() > 0.5 {
            self.qubits[j].rotate_x(PI);
        }
        
        self.entangled_pairs.push((i, j));
        Ok(())
    }
    
    /// Apply quantum interference between qubits
    pub fn apply_interference(&mut self, pattern: &[f64]) {
        for (i, qubit) in self.qubits.iter_mut().enumerate() {
            if i < pattern.len() {
                qubit.rotate_z(pattern[i] * self.phase);
            }
        }
    }
    
    /// Quantum tunneling effect simulation
    pub fn quantum_tunnel(&mut self, barrier_height: f64, energy: f64) -> f64 {
        let transmission_prob = (-2.0 * (barrier_height - energy).sqrt()).exp();
        
        for qubit in &mut self.qubits {
            if rand::random::<f64>() < transmission_prob {
                qubit.rotate_x(PI); // Tunnel through barrier
            }
        }
        
        transmission_prob
    }
    
    /// Measure all qubits and collapse the state
    pub fn measure_all(&mut self) -> Vec<bool> {
        // Handle entangled measurements
        let mut measured = vec![None; self.qubits.len()];
        
        for &(i, j) in &self.entangled_pairs {
            if measured[i].is_none() && measured[j].is_none() {
                let result_i = self.qubits[i].measure();
                measured[i] = Some(result_i);
                measured[j] = Some(result_i); // Entangled correlation
                
                // Update the entangled qubit
                if result_i {
                    self.qubits[j].alpha = Complex64::new(0.0, 0.0);
                    self.qubits[j].beta = Complex64::new(1.0, 0.0);
                } else {
                    self.qubits[j].alpha = Complex64::new(1.0, 0.0);
                    self.qubits[j].beta = Complex64::new(0.0, 0.0);
                }
            }
        }
        
        // Measure remaining qubits
        for (i, qubit) in self.qubits.iter_mut().enumerate() {
            if measured[i].is_none() {
                measured[i] = Some(qubit.measure());
            }
        }
        
        measured.into_iter().map(|x| x.unwrap()).collect()
    }
    
    /// Get quantum coherence measure
    pub fn coherence(&self) -> f64 {
        let mut total_coherence = 0.0;
        
        for qubit in &self.qubits {
            let (x, y, _) = qubit.bloch_coordinates();
            total_coherence += (x * x + y * y).sqrt();
        }
        
        total_coherence / self.qubits.len() as f64
    }
    
    /// Apply decoherence over time
    pub fn apply_decoherence(&mut self, time_step: f64) {
        let decoherence_rate = time_step / self.coherence_time;
        
        for qubit in &mut self.qubits {
            // Exponential decay of off-diagonal terms
            let decay_factor = (-decoherence_rate).exp();
            qubit.alpha *= decay_factor;
            qubit.beta *= decay_factor;
            
            // Random phase noise
            let noise = rand::random::<f64>() * 0.1 * decoherence_rate;
            qubit.rotate_z(noise);
            
            qubit.normalize();
        }
    }
    
    /// Get entanglement entropy
    pub fn entanglement_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        
        for qubit in &self.qubits {
            let p0 = qubit.prob_zero();
            let p1 = qubit.prob_one();
            
            if p0 > 0.0 {
                entropy -= p0 * p0.ln();
            }
            if p1 > 0.0 {
                entropy -= p1 * p1.ln();
            }
        }
        
        entropy
    }
}

impl Default for QuantumBit {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for QuantumState {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qubit_creation() {
        let qubit = QuantumBit::new();
        assert!((qubit.prob_zero() - 1.0).abs() < 1e-10);
        assert!(qubit.prob_one().abs() < 1e-10);
    }
    
    #[test]
    fn test_superposition() {
        let mut qubit = QuantumBit::new();
        qubit.rotate_y(PI / 2.0);
        
        let p0 = qubit.prob_zero();
        let p1 = qubit.prob_one();
        
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_entanglement() {
        let mut state = QuantumState::new(2);
        state.entangle(0, 1).unwrap();
        
        assert_eq!(state.entangled_pairs.len(), 1);
        assert_eq!(state.entangled_pairs[0], (0, 1));
    }
}