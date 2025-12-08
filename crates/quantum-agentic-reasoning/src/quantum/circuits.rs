//! Quantum circuit implementations for QAR system
//!
//! This module contains the quantum circuit implementations for:
//! - Quantum Fourier Transform (QFT) for market regime analysis
//! - Decision optimization circuits using amplitude amplification
//! - Pattern recognition circuits for similarity analysis

use crate::core::{QarResult, QarError, constants, CircuitParams, ExecutionContext, QuantumResult};
use crate::quantum::{QuantumState, Gate, StandardGates};
use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;
use super::traits::*;
use crate::core::traits::QuantumCircuit;

/// Quantum Fourier Transform circuit for market analysis
#[derive(Debug, Clone)]
pub struct QftCircuit {
    /// Number of qubits for the QFT
    pub num_qubits: usize,
    /// Circuit name
    pub name: String,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

impl QftCircuit {
    /// Create a new QFT circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            name: format!("QFT_{}", num_qubits),
            metadata: HashMap::new(),
        }
    }

    /// Apply QFT to the quantum state
    pub fn apply_qft(&self, state: &mut QuantumState) -> QarResult<()> {
        if state.num_qubits != self.num_qubits {
            return Err(QarError::QuantumError(
                "State and circuit qubit count mismatch".to_string()
            ));
        }

        // Apply QFT algorithm
        for j in 0..self.num_qubits {
            // Apply Hadamard gate
            let h_gate = StandardGates::hadamard();
            state.apply_single_qubit_gate(j, &h_gate)?;

            // Apply controlled phase gates
            for k in (j + 1)..self.num_qubits {
                let angle = constants::qft::ROTATION_MULTIPLIER / (2.0_f64.powi((k - j) as i32));
                let cp_gate = StandardGates::cphase(angle);
                state.apply_two_qubit_gate(k, j, &cp_gate)?;
            }
        }

        // Apply SWAP gates to reverse qubit order
        for i in 0..(self.num_qubits / 2) {
            let swap_gate = StandardGates::swap();
            state.apply_two_qubit_gate(i, self.num_qubits - 1 - i, &swap_gate)?;
        }

        Ok(())
    }

    /// Inverse QFT for restoring quantum state
    pub fn apply_inverse_qft(&self, state: &mut QuantumState) -> QarResult<()> {
        // Reverse the QFT process
        
        // First, apply SWAP gates
        for i in 0..(self.num_qubits / 2) {
            let swap_gate = StandardGates::swap();
            state.apply_two_qubit_gate(i, self.num_qubits - 1 - i, &swap_gate)?;
        }

        // Apply inverse controlled phase gates and Hadamard
        for j in (0..self.num_qubits).rev() {
            // Apply inverse controlled phase gates
            for k in ((j + 1)..self.num_qubits).rev() {
                let angle = -constants::qft::ROTATION_MULTIPLIER / (2.0_f64.powi((k - j) as i32));
                let cp_gate = StandardGates::cphase(angle);
                state.apply_two_qubit_gate(k, j, &cp_gate)?;
            }

            // Apply Hadamard gate
            let h_gate = StandardGates::hadamard();
            state.apply_single_qubit_gate(j, &h_gate)?;
        }

        Ok(())
    }

    /// Extract spectral information from QFT result
    pub fn extract_spectral_info(&self, state: &QuantumState) -> QarResult<Vec<f64>> {
        let probabilities = state.probabilities();
        let mut spectral_power = Vec::new();

        // Calculate power spectrum from probability amplitudes
        for i in 0..probabilities.len() {
            let frequency = i as f64 / probabilities.len() as f64;
            let power = probabilities[i];
            
            // Only include significant frequencies
            if power > constants::market::SPECTRAL_POWER_THRESHOLD {
                spectral_power.push(power);
            }
        }

        Ok(spectral_power)
    }
}

#[async_trait]
impl QuantumCircuit for QftCircuit {
    async fn execute(&self, params: &CircuitParams, context: &ExecutionContext) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();
        
        // Create initial state
        let mut state = if params.parameters.is_empty() {
            QuantumState::new(self.num_qubits)
        } else {
            // Encode parameters into quantum state
            self.encode_parameters(&params.parameters)?
        };

        // Apply QFT
        self.apply_qft(&mut state)?;

        // Extract results
        let spectral_info = self.extract_spectral_info(&state)?;
        let probabilities = state.probabilities();

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(spectral_info, execution_time, context.prefer_quantum)
            .with_probabilities(probabilities)
            .with_metadata("circuit_type".to_string(), "QFT".to_string()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn estimated_execution_time_ms(&self) -> u64 {
        // QFT complexity is O(n^2)
        (self.num_qubits * self.num_qubits * 10) as u64
    }

    fn supports_classical_fallback(&self) -> bool {
        true
    }

    async fn classical_fallback(&self, params: &CircuitParams) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();
        
        // Classical FFT implementation
        let spectral_info = self.classical_fft(&params.parameters)?;
        
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(spectral_info, execution_time, false)
            .with_metadata("fallback".to_string(), "classical_fft".to_string()))
    }

    fn validate_parameters(&self, params: &CircuitParams) -> QarResult<()> {
        if params.num_qubits != self.num_qubits {
            return Err(QarError::InvalidInput(
                format!("Expected {} qubits, got {}", self.num_qubits, params.num_qubits)
            ));
        }

        if !params.parameters.is_empty() && params.parameters.len() != (1 << self.num_qubits) {
            return Err(QarError::InvalidInput(
                "Parameter count must match 2^num_qubits for QFT".to_string()
            ));
        }

        Ok(())
    }
}

impl QftCircuit {
    /// Encode parameters into quantum state amplitudes
    fn encode_parameters(&self, parameters: &[f64]) -> QarResult<QuantumState> {
        let expected_size = 1 << self.num_qubits;
        if parameters.len() != expected_size {
            return Err(QarError::InvalidInput(
                format!("Expected {} parameters for {} qubits", expected_size, self.num_qubits)
            ));
        }

        let mut state = QuantumState::new(self.num_qubits);
        
        // Normalize and encode parameters as amplitudes
        let norm = parameters.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (i, &param) in parameters.iter().enumerate() {
                state.amplitudes[i] = num_complex::Complex64::new(param / norm, 0.0);
            }
        }

        Ok(state)
    }

    /// Classical FFT fallback implementation
    fn classical_fft(&self, data: &[f64]) -> QarResult<Vec<f64>> {
        use std::f64::consts::PI;

        let n = data.len();
        if n == 0 {
            return Ok(vec![0.0; 1 << self.num_qubits]);
        }

        // Pad data to power of 2 if necessary
        let padded_size = 1 << self.num_qubits;
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_size, 0.0);

        // Simple DFT implementation (not optimized)
        let mut result = Vec::with_capacity(padded_size);
        
        for k in 0..padded_size {
            let mut real = 0.0;
            let mut imag = 0.0;
            
            for n in 0..padded_size {
                let angle = -2.0 * PI * (k * n) as f64 / padded_size as f64;
                real += padded_data[n] * angle.cos();
                imag += padded_data[n] * angle.sin();
            }
            
            // Return power spectrum
            result.push((real * real + imag * imag).sqrt());
        }

        Ok(result)
    }
}

/// Decision optimization circuit using amplitude amplification
#[derive(Debug, Clone)]
pub struct DecisionOptimizationCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of amplitude amplification iterations
    pub iterations: usize,
    /// Oracle threshold for marking good states
    pub oracle_threshold: f64,
    /// Circuit name
    pub name: String,
}

impl DecisionOptimizationCircuit {
    /// Create a new decision optimization circuit
    pub fn new(num_qubits: usize, iterations: usize, oracle_threshold: f64) -> Self {
        Self {
            num_qubits,
            iterations,
            oracle_threshold,
            name: format!("DecisionOpt_{}q_{}i", num_qubits, iterations),
        }
    }

    /// Apply oracle to mark good decision states
    fn apply_oracle(&self, state: &mut QuantumState, factors: &[f64]) -> QarResult<()> {
        // Oracle marks states where the decision weight exceeds threshold
        for (i, amplitude) in state.amplitudes.iter_mut().enumerate() {
            let decision_weight = self.calculate_decision_weight(i, factors);
            if decision_weight < self.oracle_threshold {
                *amplitude = -*amplitude; // Phase flip for bad states
            }
        }
        Ok(())
    }

    /// Apply diffusion operator (inversion about average)
    fn apply_diffusion(&self, state: &mut QuantumState) -> QarResult<()> {
        // Calculate average amplitude
        let avg = state.amplitudes.iter().sum::<num_complex::Complex64>() 
                  / state.amplitudes.len() as f64;

        // Invert about average
        for amplitude in &mut state.amplitudes {
            *amplitude = 2.0 * avg - *amplitude;
        }

        Ok(())
    }

    /// Calculate decision weight for a given state
    fn calculate_decision_weight(&self, state_index: usize, factors: &[f64]) -> f64 {
        let mut weight = 0.0;
        
        // Use binary representation of state to weight factors
        for (i, &factor) in factors.iter().enumerate() {
            if i < self.num_qubits && (state_index & (1 << i)) != 0 {
                weight += factor;
            }
        }

        weight / factors.len() as f64
    }

    /// Perform amplitude amplification
    pub fn amplify_amplitudes(&self, state: &mut QuantumState, factors: &[f64]) -> QarResult<()> {
        for _ in 0..self.iterations {
            self.apply_oracle(state, factors)?;
            self.apply_diffusion(state)?;
        }
        Ok(())
    }
}

#[async_trait]
impl QuantumCircuit for DecisionOptimizationCircuit {
    async fn execute(&self, params: &CircuitParams, context: &ExecutionContext) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();

        // Create uniform superposition
        let mut state = QuantumState::superposition(self.num_qubits);

        // Apply amplitude amplification
        self.amplify_amplitudes(&mut state, &params.parameters)?;

        // Extract decision weights
        let mut decision_weights = Vec::new();
        let probabilities = state.probabilities();

        for (i, &prob) in probabilities.iter().enumerate() {
            let weight = self.calculate_decision_weight(i, &params.parameters);
            decision_weights.push(weight * prob);
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(decision_weights, execution_time, context.prefer_quantum)
            .with_probabilities(probabilities)
            .with_metadata("circuit_type".to_string(), "DecisionOptimization".to_string())
            .with_metadata("iterations".to_string(), self.iterations.to_string()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn estimated_execution_time_ms(&self) -> u64 {
        // Amplitude amplification scales with iterations and qubits
        (self.iterations * self.num_qubits * 20) as u64
    }

    fn supports_classical_fallback(&self) -> bool {
        true
    }

    async fn classical_fallback(&self, params: &CircuitParams) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();

        // Classical optimization using weighted selection
        let mut weights = Vec::new();
        let num_states = 1 << self.num_qubits;

        for i in 0..num_states {
            let weight = self.calculate_decision_weight(i, &params.parameters);
            
            // Apply threshold filtering
            if weight >= self.oracle_threshold {
                weights.push(weight * constants::decision::WEIGHT_NORMALIZATION_FACTOR);
            } else {
                weights.push(0.0);
            }
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(weights, execution_time, false)
            .with_metadata("fallback".to_string(), "classical_optimization".to_string()))
    }

    fn validate_parameters(&self, params: &CircuitParams) -> QarResult<()> {
        if params.parameters.is_empty() {
            return Err(QarError::InvalidInput("No parameters provided".to_string()));
        }

        if params.num_qubits != self.num_qubits {
            return Err(QarError::InvalidInput(
                format!("Expected {} qubits, got {}", self.num_qubits, params.num_qubits)
            ));
        }

        Ok(())
    }
}

/// Pattern recognition circuit using quantum similarity testing
#[derive(Debug, Clone)]
pub struct PatternRecognitionCircuit {
    /// Number of qubits for pattern encoding
    pub num_qubits: usize,
    /// Pattern encoding precision
    pub encoding_precision: usize,
    /// Circuit name
    pub name: String,
    /// Stored reference patterns
    pub reference_patterns: Vec<Vec<f64>>,
}

impl PatternRecognitionCircuit {
    /// Create a new pattern recognition circuit
    pub fn new(num_qubits: usize, encoding_precision: usize) -> Self {
        Self {
            num_qubits,
            encoding_precision,
            name: format!("PatternRec_{}q_{}p", num_qubits, encoding_precision),
            reference_patterns: Vec::new(),
        }
    }

    /// Add a reference pattern
    pub fn add_reference_pattern(&mut self, pattern: Vec<f64>) -> QarResult<()> {
        if pattern.len() != (1 << self.num_qubits) {
            return Err(QarError::InvalidInput(
                "Pattern size must match 2^num_qubits".to_string()
            ));
        }
        
        self.reference_patterns.push(pattern);
        Ok(())
    }

    /// Encode pattern into quantum state
    fn encode_pattern(&self, pattern: &[f64]) -> QarResult<QuantumState> {
        let mut state = QuantumState::new(self.num_qubits);
        
        // Normalize pattern
        let norm = pattern.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (i, &value) in pattern.iter().enumerate() {
                if i < state.amplitudes.len() {
                    state.amplitudes[i] = num_complex::Complex64::new(value / norm, 0.0);
                }
            }
        }

        Ok(state)
    }

    /// Calculate quantum similarity using state fidelity
    pub fn quantum_similarity(&self, pattern1: &[f64], pattern2: &[f64]) -> QarResult<f64> {
        let state1 = self.encode_pattern(pattern1)?;
        let state2 = self.encode_pattern(pattern2)?;
        
        state1.fidelity(&state2)
    }

    /// Perform pattern matching against all reference patterns
    pub fn match_patterns(&self, input_pattern: &[f64]) -> QarResult<Vec<(usize, f64)>> {
        let mut matches = Vec::new();

        for (i, reference) in self.reference_patterns.iter().enumerate() {
            let similarity = self.quantum_similarity(input_pattern, reference)?;
            
            if similarity >= constants::pattern::ORACLE_THRESHOLD {
                matches.push((i, similarity));
            }
        }

        // Sort by similarity (highest first)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(matches)
    }
}

#[async_trait]
impl QuantumCircuit for PatternRecognitionCircuit {
    async fn execute(&self, params: &CircuitParams, context: &ExecutionContext) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();

        if self.reference_patterns.is_empty() {
            return Err(QarError::QuantumError("No reference patterns loaded".to_string()));
        }

        // Find pattern matches
        let matches = self.match_patterns(&params.parameters)?;
        
        // Convert matches to result format
        let mut similarities = vec![0.0; self.reference_patterns.len()];
        for (pattern_idx, similarity) in matches {
            similarities[pattern_idx] = similarity;
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(similarities, execution_time, context.prefer_quantum)
            .with_metadata("circuit_type".to_string(), "PatternRecognition".to_string())
            .with_metadata("reference_patterns".to_string(), self.reference_patterns.len().to_string()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn estimated_execution_time_ms(&self) -> u64 {
        // Pattern recognition scales with number of reference patterns
        (self.reference_patterns.len() * self.num_qubits * 15) as u64
    }

    fn supports_classical_fallback(&self) -> bool {
        true
    }

    async fn classical_fallback(&self, params: &CircuitParams) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();

        // Classical cosine similarity
        let mut similarities = Vec::new();

        for reference in &self.reference_patterns {
            let similarity = self.classical_cosine_similarity(&params.parameters, reference);
            similarities.push(similarity);
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(QuantumResult::new(similarities, execution_time, false)
            .with_metadata("fallback".to_string(), "cosine_similarity".to_string()))
    }

    fn validate_parameters(&self, params: &CircuitParams) -> QarResult<()> {
        let expected_size = 1 << self.num_qubits;
        
        if params.parameters.len() != expected_size {
            return Err(QarError::InvalidInput(
                format!("Expected {} parameters for {} qubits", expected_size, self.num_qubits)
            ));
        }

        Ok(())
    }
}

impl PatternRecognitionCircuit {
    /// Classical cosine similarity fallback
    fn classical_cosine_similarity(&self, pattern1: &[f64], pattern2: &[f64]) -> f64 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }

        let dot_product: f64 = pattern1.iter().zip(pattern2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = pattern1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = pattern2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_circuit_creation() {
        let qft = QftCircuit::new(3);
        assert_eq!(qft.num_qubits, 3);
        assert_eq!(qft.name(), "QFT_3");
        assert!(qft.supports_classical_fallback());
    }

    #[test]
    fn test_qft_state_manipulation() {
        let qft = QftCircuit::new(2);
        let mut state = QuantumState::new(2);
        
        // Apply QFT should not error
        assert!(qft.apply_qft(&mut state).is_ok());
        
        // State should still be normalized
        assert!(state.is_normalized());
    }

    #[test]
    fn test_decision_optimization_circuit() {
        let circuit = DecisionOptimizationCircuit::new(3, 2, 0.5);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.iterations, 2);
        assert_eq!(circuit.oracle_threshold, 0.5);
    }

    #[test]
    fn test_pattern_recognition_circuit() {
        let mut circuit = PatternRecognitionCircuit::new(2, 16);
        assert_eq!(circuit.num_qubits, 2);
        
        // Add a reference pattern
        let pattern = vec![0.5, 0.3, 0.1, 0.1];
        assert!(circuit.add_reference_pattern(pattern).is_ok());
        assert_eq!(circuit.reference_patterns.len(), 1);
    }

    #[test]
    fn test_quantum_similarity() {
        let circuit = PatternRecognitionCircuit::new(2, 16);
        let pattern1 = vec![1.0, 0.0, 0.0, 0.0];
        let pattern2 = vec![1.0, 0.0, 0.0, 0.0];
        let pattern3 = vec![0.0, 1.0, 0.0, 0.0];
        
        // Identical patterns should have high similarity
        let sim1 = circuit.quantum_similarity(&pattern1, &pattern2).unwrap();
        assert!(sim1 > 0.99);
        
        // Different patterns should have lower similarity
        let sim2 = circuit.quantum_similarity(&pattern1, &pattern3).unwrap();
        assert!(sim2 < 0.5);
    }

    #[test]
    fn test_classical_cosine_similarity() {
        let circuit = PatternRecognitionCircuit::new(2, 16);
        let pattern1 = vec![1.0, 0.0, 0.0, 0.0];
        let pattern2 = vec![1.0, 0.0, 0.0, 0.0];
        
        let similarity = circuit.classical_cosine_similarity(&pattern1, &pattern2);
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_circuit_parameter_validation() {
        let qft = QftCircuit::new(3);
        
        // Valid parameters
        let valid_params = CircuitParams::new(vec![0.1; 8], 3);
        assert!(qft.validate_parameters(&valid_params).is_ok());
        
        // Invalid qubit count
        let invalid_params = CircuitParams::new(vec![0.1; 8], 2);
        assert!(qft.validate_parameters(&invalid_params).is_err());
        
        // Invalid parameter count
        let invalid_params2 = CircuitParams::new(vec![0.1; 4], 3);
        assert!(qft.validate_parameters(&invalid_params2).is_err());
    }
}