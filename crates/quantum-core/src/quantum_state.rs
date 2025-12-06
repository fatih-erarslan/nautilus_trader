//! Quantum state management and operations
//!
//! This module provides comprehensive quantum state representation, manipulation,
//! and measurement capabilities for quantum algorithms in trading systems.

use crate::error::{QuantumError, QuantumResult};
// use nalgebra::DMatrix;
use num_complex::Complex64;
use num_traits::Zero;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Complex amplitude type for quantum states
pub type ComplexAmplitude = Complex64;

/// Quantum state representation with amplitude vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Number of qubits in the system
    num_qubits: usize,
    /// Complex amplitudes for each basis state
    amplitudes: Vec<ComplexAmplitude>,
    /// Optional normalization factor
    normalization: f64,
    /// State creation timestamp
    created_at: chrono::DateTime<chrono::Utc>,
    /// State modification timestamp
    modified_at: chrono::DateTime<chrono::Utc>,
    /// Optional state label
    label: Option<String>,
}

impl QuantumState {
    /// Create a new quantum state in |0...0> state
    pub fn new(num_qubits: usize) -> QuantumResult<Self> {
        if num_qubits == 0 {
            return Err(QuantumError::invalid_state("Number of qubits must be greater than 0"));
        }
        if num_qubits > crate::MAX_QUBITS {
            return Err(QuantumError::invalid_state(format!(
                "Number of qubits {} exceeds maximum {}",
                num_qubits,
                crate::MAX_QUBITS
            )));
        }

        let num_states = 1 << num_qubits; // 2^n states
        let mut amplitudes = vec![ComplexAmplitude::zero(); num_states];
        amplitudes[0] = ComplexAmplitude::new(1.0, 0.0); // |0...0> state

        let now = chrono::Utc::now();
        Ok(Self {
            num_qubits,
            amplitudes,
            normalization: 1.0,
            created_at: now,
            modified_at: now,
            label: None,
        })
    }

    /// Create a quantum state from amplitude vector
    pub fn from_amplitudes(amplitudes: Vec<ComplexAmplitude>) -> QuantumResult<Self> {
        let num_states = amplitudes.len();
        if !num_states.is_power_of_two() {
            return Err(QuantumError::invalid_state(
                "Number of amplitudes must be a power of 2"
            ));
        }

        let num_qubits = (num_states as f64).log2() as usize;
        if num_qubits > crate::MAX_QUBITS {
            return Err(QuantumError::invalid_state(format!(
                "State size exceeds maximum qubits: {}",
                crate::MAX_QUBITS
            )));
        }

        let mut state = Self {
            num_qubits,
            amplitudes,
            normalization: 1.0,
            created_at: chrono::Utc::now(),
            modified_at: chrono::Utc::now(),
            label: None,
        };

        state.normalize()?;
        Ok(state)
    }

    /// Create a superposition state with equal weights
    pub fn superposition(num_qubits: usize) -> QuantumResult<Self> {
        let num_states = 1 << num_qubits;
        let amplitude = ComplexAmplitude::new(1.0 / (num_states as f64).sqrt(), 0.0);
        let amplitudes = vec![amplitude; num_states];
        
        Self::from_amplitudes(amplitudes)
    }

    /// Create a Bell state (entangled state)
    pub fn bell_state() -> QuantumResult<Self> {
        let sqrt_half = (0.5_f64).sqrt();
        let amplitudes = vec![
            ComplexAmplitude::new(sqrt_half, 0.0), // |00>
            ComplexAmplitude::zero(),               // |01>
            ComplexAmplitude::zero(),               // |10>
            ComplexAmplitude::new(sqrt_half, 0.0), // |11>
        ];
        
        Self::from_amplitudes(amplitudes)
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get number of basis states
    pub fn num_states(&self) -> usize {
        self.amplitudes.len()
    }

    /// Get amplitude for a specific basis state
    pub fn get_amplitude(&self, state_index: usize) -> QuantumResult<ComplexAmplitude> {
        if state_index >= self.amplitudes.len() {
            return Err(QuantumError::invalid_state(format!(
                "State index {} out of bounds for {} states",
                state_index,
                self.amplitudes.len()
            )));
        }
        Ok(self.amplitudes[state_index])
    }

    /// Set amplitude for a specific basis state
    pub fn set_amplitude(&mut self, state_index: usize, amplitude: ComplexAmplitude) -> QuantumResult<()> {
        if state_index >= self.amplitudes.len() {
            return Err(QuantumError::invalid_state(format!(
                "State index {} out of bounds for {} states",
                state_index,
                self.amplitudes.len()
            )));
        }
        
        self.amplitudes[state_index] = amplitude;
        self.modified_at = chrono::Utc::now();
        Ok(())
    }

    /// Get all amplitudes
    pub fn amplitudes(&self) -> &[ComplexAmplitude] {
        &self.amplitudes
    }
    
    /// Get all amplitudes (alias for amplitudes)
    pub fn get_amplitudes(&self) -> &[ComplexAmplitude] {
        &self.amplitudes
    }

    /// Get mutable reference to amplitudes
    pub fn amplitudes_mut(&mut self) -> &mut [ComplexAmplitude] {
        self.modified_at = chrono::Utc::now();
        &mut self.amplitudes
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) -> QuantumResult<()> {
        let norm_squared: f64 = self.amplitudes
            .par_iter()
            .map(|amp| amp.norm_sqr())
            .sum();

        if norm_squared < crate::DEFAULT_PRECISION {
            return Err(QuantumError::invalid_state("State has zero norm"));
        }

        let norm = norm_squared.sqrt();
        self.amplitudes
            .par_iter_mut()
            .for_each(|amp| *amp /= norm);

        self.normalization = norm;
        self.modified_at = chrono::Utc::now();
        Ok(())
    }

    /// Check if state is normalized
    pub fn is_normalized(&self) -> bool {
        let norm_squared: f64 = self.amplitudes
            .par_iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        (norm_squared - 1.0).abs() < crate::DEFAULT_PRECISION
    }

    /// Calculate state fidelity with another state
    pub fn fidelity(&self, other: &QuantumState) -> QuantumResult<f64> {
        if self.num_qubits != other.num_qubits {
            return Err(QuantumError::computation_error(
                "fidelity",
                "States must have same number of qubits"
            ));
        }

        let overlap = self.amplitudes
            .par_iter()
            .zip(other.amplitudes.par_iter())
            .map(|(a, b)| a.conj() * b)
            .sum::<ComplexAmplitude>();

        Ok(overlap.norm())
    }

    /// Calculate probability of measuring a specific state
    pub fn probability(&self, state_index: usize) -> QuantumResult<f64> {
        if state_index >= self.amplitudes.len() {
            return Err(QuantumError::invalid_state(format!(
                "State index {} out of bounds",
                state_index
            )));
        }

        Ok(self.amplitudes[state_index].norm_sqr())
    }

    /// Get probability distribution over all basis states
    pub fn probability_distribution(&self) -> Vec<f64> {
        self.amplitudes
            .par_iter()
            .map(|amp| amp.norm_sqr())
            .collect()
    }

    /// Measure the quantum state (collapses to classical state)
    pub fn measure(&mut self) -> QuantumResult<usize> {
        use rand::prelude::*;
        
        let probabilities = self.probability_distribution();
        let mut rng = thread_rng();
        let random_value: f64 = rng.gen();
        
        let mut cumulative_prob = 0.0;
        for (state_index, prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                // Collapse to measured state
                self.amplitudes.fill(ComplexAmplitude::zero());
                self.amplitudes[state_index] = ComplexAmplitude::new(1.0, 0.0);
                self.modified_at = chrono::Utc::now();
                return Ok(state_index);
            }
        }

        // Fallback to last state (should not happen with proper probabilities)
        let last_state = self.amplitudes.len() - 1;
        self.amplitudes.fill(ComplexAmplitude::zero());
        self.amplitudes[last_state] = ComplexAmplitude::new(1.0, 0.0);
        self.modified_at = chrono::Utc::now();
        Ok(last_state)
    }

    /// Measure specific qubits
    pub fn measure_qubits(&mut self, qubits: &[usize]) -> QuantumResult<Vec<usize>> {
        if qubits.iter().any(|&q| q >= self.num_qubits) {
            return Err(QuantumError::measurement_error(
                "partial",
                "Qubit index out of bounds"
            ));
        }

        // For simplicity, implement full measurement and extract qubit values
        let measured_state = self.measure()?;
        let mut results = Vec::new();
        
        for &qubit in qubits {
            let bit_value = (measured_state >> qubit) & 1;
            results.push(bit_value);
        }

        Ok(results)
    }

    /// Measure all qubits and return single result
    pub fn measure_all(&self) -> QuantumResult<usize> {
        use rand::prelude::*;
        
        let probabilities = self.probability_distribution();
        let mut rng = thread_rng();
        let random_value: f64 = rng.gen();
        
        let mut cumulative_prob = 0.0;
        for (state_index, prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return Ok(state_index);
            }
        }

        // Fallback to last state (should not happen with proper probabilities)
        Ok(self.amplitudes.len() - 1)
    }

    /// Calculate entanglement entropy between subsystems
    pub fn entanglement_entropy(&self, subsystem_qubits: &[usize]) -> QuantumResult<f64> {
        if subsystem_qubits.is_empty() || subsystem_qubits.len() >= self.num_qubits {
            return Err(QuantumError::computation_error(
                "entanglement_entropy",
                "Invalid subsystem size"
            ));
        }

        // Simplified entropy calculation based on Schmidt decomposition
        // This is a placeholder for more sophisticated calculation
        let subsystem_size = subsystem_qubits.len();
        let reduced_dim = 1 << subsystem_size;
        
        // Calculate reduced density matrix (simplified)
        let mut reduced_probs = vec![0.0; reduced_dim];
        for (state_idx, amplitude) in self.amplitudes.iter().enumerate() {
            let subsystem_state = subsystem_qubits
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &qubit)| {
                    acc | (((state_idx >> qubit) & 1) << i)
                });
            reduced_probs[subsystem_state] += amplitude.norm_sqr();
        }

        // Calculate von Neumann entropy
        let entropy = reduced_probs
            .iter()
            .filter(|&&p| p > crate::DEFAULT_PRECISION)
            .map(|&p| -p * p.ln())
            .sum();

        Ok(entropy)
    }

    /// Apply decoherence model
    pub fn apply_decoherence(&mut self, decoherence_rate: f64, time_step: f64) -> QuantumResult<()> {
        if decoherence_rate < 0.0 || time_step < 0.0 {
            return Err(QuantumError::decoherence_error("Invalid decoherence parameters"));
        }

        let decay_factor = (-decoherence_rate * time_step).exp();
        
        // Apply exponential decay to off-diagonal elements (simplified model)
        for amplitude in &mut self.amplitudes {
            if amplitude.norm_sqr() < 1.0 {
                *amplitude *= decay_factor;
            }
        }

        self.normalize()?;
        self.modified_at = chrono::Utc::now();
        Ok(())
    }

    /// Get state age in milliseconds
    pub fn age_ms(&self) -> i64 {
        let now = chrono::Utc::now();
        now.signed_duration_since(self.created_at).num_milliseconds()
    }

    /// Set state label
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
        self.modified_at = chrono::Utc::now();
    }

    /// Get state label
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Convert to classical bit string (based on most probable state)
    pub fn to_classical_bitstring(&self) -> String {
        let probabilities = self.probability_distribution();
        let max_state = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(state, _)| state)
            .unwrap_or(0);

        format!("{:0width$b}", max_state, width = self.num_qubits)
    }

    /// Clone state with new label
    pub fn clone_with_label(&self, label: impl Into<String>) -> Self {
        let mut cloned = self.clone();
        cloned.set_label(label);
        cloned
    }
}

/// Thread-safe quantum state manager
#[derive(Debug)]
pub struct QuantumStateManager {
    /// Thread-safe storage for quantum states
    states: Arc<RwLock<HashMap<String, QuantumState>>>,
    /// Maximum number of states to store
    max_states: usize,
}

impl QuantumStateManager {
    /// Create a new quantum state manager
    pub fn new(max_states: usize) -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            max_states,
        }
    }

    /// Store a quantum state
    pub fn store_state(&self, id: String, state: QuantumState) -> QuantumResult<()> {
        let mut states = self.states.write()
            .map_err(|_| QuantumError::concurrency_error("Failed to acquire write lock"))?;

        if states.len() >= self.max_states && !states.contains_key(&id) {
            return Err(QuantumError::resource_exhausted(
                "quantum_states",
                "Maximum number of states reached"
            ));
        }

        states.insert(id, state);
        Ok(())
    }

    /// Retrieve a quantum state
    pub fn get_state(&self, id: &str) -> QuantumResult<Option<QuantumState>> {
        let states = self.states.read()
            .map_err(|_| QuantumError::concurrency_error("Failed to acquire read lock"))?;

        Ok(states.get(id).cloned())
    }

    /// Remove a quantum state
    pub fn remove_state(&self, id: &str) -> QuantumResult<Option<QuantumState>> {
        let mut states = self.states.write()
            .map_err(|_| QuantumError::concurrency_error("Failed to acquire write lock"))?;

        Ok(states.remove(id))
    }

    /// List all state IDs
    pub fn list_states(&self) -> QuantumResult<Vec<String>> {
        let states = self.states.read()
            .map_err(|_| QuantumError::concurrency_error("Failed to acquire read lock"))?;

        Ok(states.keys().cloned().collect())
    }

    /// Get number of stored states
    pub fn count(&self) -> QuantumResult<usize> {
        let states = self.states.read()
            .map_err(|_| QuantumError::concurrency_error("Failed to acquire read lock"))?;

        Ok(states.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.num_states(), 8);
        assert_eq!(state.get_amplitude(0).unwrap(), ComplexAmplitude::new(1.0, 0.0));
    }

    #[test]
    fn test_superposition_state() {
        let state = QuantumState::superposition(2).unwrap();
        assert_eq!(state.num_qubits(), 2);
        let expected_amplitude = ComplexAmplitude::new(0.5, 0.0);
        for i in 0..4 {
            assert_abs_diff_eq!(
                state.get_amplitude(i).unwrap().norm_sqr(),
                expected_amplitude.norm_sqr(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_bell_state() {
        let state = QuantumState::bell_state().unwrap();
        assert_eq!(state.num_qubits(), 2);
        assert_abs_diff_eq!(state.get_amplitude(0).unwrap().norm_sqr(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(3).unwrap().norm_sqr(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(1).unwrap().norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(2).unwrap().norm_sqr(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalization() {
        let mut state = QuantumState::new(2).unwrap();
        state.set_amplitude(1, ComplexAmplitude::new(1.0, 0.0)).unwrap();
        state.normalize().unwrap();
        assert!(state.is_normalized());
    }

    #[test]
    fn test_fidelity() {
        let state1 = QuantumState::new(2).unwrap();
        let state2 = QuantumState::new(2).unwrap();
        let fidelity = state1.fidelity(&state2).unwrap();
        assert_abs_diff_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measurement() {
        let mut state = QuantumState::new(1).unwrap();
        let result = state.measure().unwrap();
        assert_eq!(result, 0); // Should always measure |0> for initial state
    }

    #[test]
    fn test_probability_distribution() {
        let state = QuantumState::superposition(2).unwrap();
        let probs = state.probability_distribution();
        for prob in probs {
            assert_abs_diff_eq!(prob, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_state_manager() {
        let manager = QuantumStateManager::new(10);
        let state = QuantumState::new(2).unwrap();
        
        manager.store_state("test".to_string(), state.clone()).unwrap();
        let retrieved = manager.get_state("test").unwrap().unwrap();
        
        assert_eq!(retrieved.num_qubits(), state.num_qubits());
        assert_eq!(manager.count().unwrap(), 1);
    }

    #[test]
    fn test_classical_bitstring() {
        let state = QuantumState::new(3).unwrap();
        let bitstring = state.to_classical_bitstring();
        assert_eq!(bitstring, "000");
    }

    #[test]
    fn test_decoherence() {
        let mut state = QuantumState::superposition(2).unwrap();
        state.apply_decoherence(0.1, 1.0).unwrap();
        assert!(state.is_normalized());
    }
}