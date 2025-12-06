//! Quantum enhancement algorithms for hedge systems

use num_complex::Complex64;
use nalgebra::{DVector, DMatrix, Complex};
use std::collections::HashMap;
use rand::Rng;
use crate::{HedgeError, HedgeConfig, MarketData};

/// Quantum state representation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantumState {
    /// Quantum amplitudes
    pub amplitudes: Vec<Complex64>,
    /// State labels
    pub labels: Vec<String>,
    /// Measurement probabilities
    pub probabilities: Vec<f64>,
    /// Entanglement matrix
    pub entanglement_matrix: DMatrix<Complex64>,
    /// Decoherence rate
    pub decoherence_rate: f64,
    /// Time evolution parameter
    pub time_parameter: f64,
}

impl QuantumState {
    /// Create new quantum state
    pub fn new(labels: Vec<String>, decoherence_rate: f64) -> Self {
        let n = labels.len();
        let amplitudes = vec![Complex64::new(1.0 / (n as f64).sqrt(), 0.0); n];
        let probabilities = vec![1.0 / n as f64; n];
        let entanglement_matrix = DMatrix::identity(n, n);
        
        Self {
            amplitudes,
            labels,
            probabilities,
            entanglement_matrix,
            decoherence_rate,
            time_parameter: 0.0,
        }
    }
    
    /// Normalize quantum state
    pub fn normalize(&mut self) -> Result<(), HedgeError> {
        let norm_squared: f64 = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        if norm_squared == 0.0 {
            return Err(HedgeError::QuantumError("Cannot normalize zero state".to_string()));
        }
        
        let norm = norm_squared.sqrt();
        for amplitude in &mut self.amplitudes {
            *amplitude /= norm;
        }
        
        // Update probabilities
        self.probabilities = self.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        Ok(())
    }
    
    /// Apply quantum gate
    pub fn apply_gate(&mut self, gate: &QuantumGate) -> Result<(), HedgeError> {
        if gate.matrix.nrows() != self.amplitudes.len() {
            return Err(HedgeError::QuantumError("Gate dimension mismatch".to_string()));
        }
        
        let state_vector = DVector::from_vec(self.amplitudes.clone());
        let new_state = &gate.matrix * state_vector;
        
        self.amplitudes = new_state.as_slice().to_vec();
        self.normalize()?;
        
        Ok(())
    }
    
    /// Measure quantum state
    pub fn measure(&mut self) -> Result<usize, HedgeError> {
        let mut rng = rand::thread_rng();
        let random_value: f64 = rng.gen();
        
        let mut cumulative_prob = 0.0;
        for (i, prob) in self.probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                // Collapse to measured state
                self.amplitudes = vec![Complex64::new(0.0, 0.0); self.amplitudes.len()];
                self.amplitudes[i] = Complex64::new(1.0, 0.0);
                self.probabilities = vec![0.0; self.probabilities.len()];
                self.probabilities[i] = 1.0;
                
                return Ok(i);
            }
        }
        
        // Fallback to last state
        let last_idx = self.amplitudes.len() - 1;
        self.amplitudes = vec![Complex64::new(0.0, 0.0); self.amplitudes.len()];
        self.amplitudes[last_idx] = Complex64::new(1.0, 0.0);
        self.probabilities = vec![0.0; self.probabilities.len()];
        self.probabilities[last_idx] = 1.0;
        
        Ok(last_idx)
    }
    
    /// Apply decoherence
    pub fn apply_decoherence(&mut self, dt: f64) -> Result<(), HedgeError> {
        let decay_factor = (-self.decoherence_rate * dt).exp();
        
        for amplitude in &mut self.amplitudes {
            *amplitude *= decay_factor;
        }
        
        // Add random noise
        let mut rng = rand::thread_rng();
        let noise_strength = (1.0 - decay_factor * decay_factor).sqrt();
        
        for amplitude in &mut self.amplitudes {
            let noise_real = rng.gen::<f64>() * noise_strength;
            let noise_imag = rng.gen::<f64>() * noise_strength;
            *amplitude += Complex64::new(noise_real, noise_imag);
        }
        
        self.normalize()?;
        
        Ok(())
    }
    
    /// Get expectation value
    pub fn expectation_value(&self, observable: &DMatrix<Complex64>) -> Result<Complex64, HedgeError> {
        if observable.nrows() != self.amplitudes.len() {
            return Err(HedgeError::QuantumError("Observable dimension mismatch".to_string()));
        }
        
        let state_vector = DVector::from_vec(self.amplitudes.clone());
        let obs_state = observable * &state_vector;
        
        let expectation = state_vector.iter()
            .zip(obs_state.iter())
            .map(|(psi, obs_psi)| psi.conj() * obs_psi)
            .sum();
        
        Ok(expectation)
    }
    
    /// Get entropy
    pub fn entropy(&self) -> f64 {
        self.probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }
    
    /// Get purity
    pub fn purity(&self) -> f64 {
        self.probabilities.iter()
            .map(|&p| p * p)
            .sum()
    }
}

/// Quantum gate representation
#[derive(Debug, Clone)]
pub struct QuantumGate {
    /// Gate matrix
    pub matrix: DMatrix<Complex64>,
    /// Gate name
    pub name: String,
}

impl QuantumGate {
    /// Create Hadamard gate
    pub fn hadamard(n: usize) -> Self {
        let h_factor = 1.0 / (2.0_f64).sqrt();
        let h_matrix = DMatrix::from_fn(2, 2, |i, j| {
            if (i + j) % 2 == 0 {
                Complex64::new(h_factor, 0.0)
            } else {
                Complex64::new(-h_factor, 0.0)
            }
        });
        
        // Tensor product for n qubits
        let mut result = h_matrix.clone();
        for _ in 1..n {
            result = Self::tensor_product(&result, &h_matrix);
        }
        
        Self {
            matrix: result,
            name: format!("Hadamard({})", n),
        }
    }
    
    /// Create rotation gate
    pub fn rotation(theta: f64, n: usize) -> Self {
        let cos_theta = (theta / 2.0).cos();
        let sin_theta = (theta / 2.0).sin();
        
        let r_matrix = DMatrix::from_fn(2, 2, |i, j| {
            match (i, j) {
                (0, 0) => Complex64::new(cos_theta, 0.0),
                (0, 1) => Complex64::new(0.0, -sin_theta),
                (1, 0) => Complex64::new(0.0, sin_theta),
                (1, 1) => Complex64::new(cos_theta, 0.0),
                _ => Complex64::new(0.0, 0.0),
            }
        });
        
        // Tensor product for n qubits
        let mut result = r_matrix.clone();
        for _ in 1..n {
            result = Self::tensor_product(&result, &r_matrix);
        }
        
        Self {
            matrix: result,
            name: format!("Rotation({:.3}, {})", theta, n),
        }
    }
    
    /// Create phase gate
    pub fn phase(phi: f64, n: usize) -> Self {
        let phase_matrix = DMatrix::from_fn(2, 2, |i, j| {
            match (i, j) {
                (0, 0) => Complex64::new(1.0, 0.0),
                (1, 1) => Complex64::new(phi.cos(), phi.sin()),
                _ => Complex64::new(0.0, 0.0),
            }
        });
        
        // Tensor product for n qubits
        let mut result = phase_matrix.clone();
        for _ in 1..n {
            result = Self::tensor_product(&result, &phase_matrix);
        }
        
        Self {
            matrix: result,
            name: format!("Phase({:.3}, {})", phi, n),
        }
    }
    
    /// Tensor product of two matrices
    fn tensor_product(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let (a_rows, a_cols) = a.shape();
        let (b_rows, b_cols) = b.shape();
        
        DMatrix::from_fn(a_rows * b_rows, a_cols * b_cols, |i, j| {
            let a_i = i / b_rows;
            let a_j = j / b_cols;
            let b_i = i % b_rows;
            let b_j = j % b_cols;
            
            a[(a_i, a_j)] * b[(b_i, b_j)]
        })
    }
}

/// Quantum hedge algorithm
#[derive(Debug, Clone)]
pub struct QuantumHedgeAlgorithm {
    /// Quantum state
    quantum_state: QuantumState,
    /// Expert weights
    expert_weights: HashMap<String, f64>,
    /// Quantum gates
    gates: Vec<QuantumGate>,
    /// Configuration
    config: HedgeConfig,
    /// Time step
    time_step: usize,
    /// Performance history
    performance_history: Vec<f64>,
}

impl QuantumHedgeAlgorithm {
    /// Create new quantum hedge algorithm
    pub fn new(expert_names: Vec<String>, config: HedgeConfig) -> Result<Self, HedgeError> {
        let quantum_state = QuantumState::new(
            expert_names.clone(),
            config.quantum_config.decoherence_rate,
        );
        
        let expert_weights = expert_names.iter()
            .map(|name| (name.clone(), 1.0 / expert_names.len() as f64))
            .collect();
        
        Ok(Self {
            quantum_state,
            expert_weights,
            gates: Vec::new(),
            config,
            time_step: 0,
            performance_history: Vec::new(),
        })
    }
    
    /// Update quantum state with market data
    pub fn update(&mut self, market_data: &MarketData, expert_predictions: &HashMap<String, f64>) -> Result<(), HedgeError> {
        // Apply quantum gates based on market conditions
        self.apply_adaptive_gates(market_data)?;
        
        // Apply decoherence
        self.quantum_state.apply_decoherence(0.01)?;
        
        // Update expert weights based on quantum probabilities
        self.update_quantum_weights(expert_predictions)?;
        
        // Update time step
        self.time_step += 1;
        self.quantum_state.time_parameter = self.time_step as f64;
        
        Ok(())
    }
    
    /// Apply adaptive quantum gates
    fn apply_adaptive_gates(&mut self, market_data: &MarketData) -> Result<(), HedgeError> {
        let volatility = self.estimate_volatility(market_data)?;
        let momentum = self.estimate_momentum(market_data)?;
        
        // Apply Hadamard gate for superposition
        if volatility > 0.02 {
            let hadamard = QuantumGate::hadamard(1);
            self.quantum_state.apply_gate(&hadamard)?;
        }
        
        // Apply rotation gate based on momentum
        if momentum.abs() > 0.01 {
            let rotation_angle = momentum * std::f64::consts::PI;
            let rotation = QuantumGate::rotation(rotation_angle, 1);
            self.quantum_state.apply_gate(&rotation)?;
        }
        
        // Apply phase gate for market regime
        let phase_angle = market_data.typical_price().sin() * 0.1;
        let phase = QuantumGate::phase(phase_angle, 1);
        self.quantum_state.apply_gate(&phase)?;
        
        Ok(())
    }
    
    /// Estimate volatility from market data
    fn estimate_volatility(&self, market_data: &MarketData) -> Result<f64, HedgeError> {
        let high_low_ratio = market_data.high() / market_data.low();
        let volatility = (high_low_ratio.ln()).abs();
        Ok(volatility)
    }
    
    /// Estimate momentum from market data
    fn estimate_momentum(&self, market_data: &MarketData) -> Result<f64, HedgeError> {
        let price_change = market_data.close() - market_data.open();
        let momentum = price_change / market_data.open();
        Ok(momentum)
    }
    
    /// Update quantum weights
    fn update_quantum_weights(&mut self, expert_predictions: &HashMap<String, f64>) -> Result<(), HedgeError> {
        // Calculate quantum-enhanced weights
        for (i, label) in self.quantum_state.labels.iter().enumerate() {
            if let Some(prediction) = expert_predictions.get(label) {
                let quantum_prob = self.quantum_state.probabilities[i];
                let quantum_weight = quantum_prob * (1.0 + prediction.abs());
                
                self.expert_weights.insert(label.clone(), quantum_weight);
            }
        }
        
        // Normalize weights
        let total_weight: f64 = self.expert_weights.values().sum();
        if total_weight > 0.0 {
            for weight in self.expert_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        Ok(())
    }
    
    /// Get quantum-enhanced recommendation
    pub fn get_recommendation(&self) -> Result<(HashMap<String, f64>, f64), HedgeError> {
        let weights = self.expert_weights.clone();
        
        // Calculate quantum coherence as confidence measure
        let coherence = self.quantum_state.purity();
        let confidence = coherence.min(1.0);
        
        Ok((weights, confidence))
    }
    
    /// Perform quantum measurement
    pub fn measure(&mut self) -> Result<String, HedgeError> {
        let measured_idx = self.quantum_state.measure()?;
        
        if measured_idx < self.quantum_state.labels.len() {
            Ok(self.quantum_state.labels[measured_idx].clone())
        } else {
            Err(HedgeError::QuantumError("Invalid measurement result".to_string()))
        }
    }
    
    /// Get quantum state entropy
    pub fn get_entropy(&self) -> f64 {
        self.quantum_state.entropy()
    }
    
    /// Get quantum state purity
    pub fn get_purity(&self) -> f64 {
        self.quantum_state.purity()
    }
    
    /// Reset quantum state
    pub fn reset(&mut self) -> Result<(), HedgeError> {
        self.quantum_state = QuantumState::new(
            self.quantum_state.labels.clone(),
            self.config.quantum_config.decoherence_rate,
        );
        
        let n_experts = self.quantum_state.labels.len();
        for (_name, weight) in &mut self.expert_weights {
            *weight = 1.0 / n_experts as f64;
        }
        
        self.time_step = 0;
        self.performance_history.clear();
        
        Ok(())
    }
    
    /// Get quantum state amplitudes
    pub fn get_amplitudes(&self) -> &Vec<Complex64> {
        &self.quantum_state.amplitudes
    }
    
    /// Get quantum state probabilities
    pub fn get_probabilities(&self) -> &Vec<f64> {
        &self.quantum_state.probabilities
    }
}

/// Quantum annealing for optimization
#[derive(Debug, Clone)]
pub struct QuantumAnnealer {
    /// Problem Hamiltonian
    problem_hamiltonian: DMatrix<Complex64>,
    /// Transverse field Hamiltonian
    transverse_hamiltonian: DMatrix<Complex64>,
    /// Annealing schedule
    annealing_schedule: Vec<f64>,
    /// Current temperature
    temperature: f64,
    /// Number of qubits
    num_qubits: usize,
}

impl QuantumAnnealer {
    /// Create new quantum annealer
    pub fn new(num_qubits: usize) -> Self {
        let dimension = 2_usize.pow(num_qubits as u32);
        
        Self {
            problem_hamiltonian: DMatrix::zeros(dimension, dimension),
            transverse_hamiltonian: DMatrix::zeros(dimension, dimension),
            annealing_schedule: (0..100).map(|i| i as f64 / 100.0).collect(),
            temperature: 1.0,
            num_qubits,
        }
    }
    
    /// Set problem Hamiltonian
    pub fn set_problem_hamiltonian(&mut self, hamiltonian: DMatrix<Complex64>) -> Result<(), HedgeError> {
        if hamiltonian.nrows() != hamiltonian.ncols() {
            return Err(HedgeError::QuantumError("Hamiltonian must be square".to_string()));
        }
        
        self.problem_hamiltonian = hamiltonian;
        Ok(())
    }
    
    /// Set transverse field Hamiltonian
    pub fn set_transverse_hamiltonian(&mut self, hamiltonian: DMatrix<Complex64>) -> Result<(), HedgeError> {
        if hamiltonian.nrows() != hamiltonian.ncols() {
            return Err(HedgeError::QuantumError("Hamiltonian must be square".to_string()));
        }
        
        self.transverse_hamiltonian = hamiltonian;
        Ok(())
    }
    
    /// Perform quantum annealing
    pub fn anneal(&mut self, initial_state: &mut QuantumState) -> Result<Vec<f64>, HedgeError> {
        let mut results = Vec::new();
        
        for &s in &self.annealing_schedule {
            // Interpolate between Hamiltonians
            let hamiltonian = &self.transverse_hamiltonian * Complex64::new(1.0 - s, 0.0) + &self.problem_hamiltonian * Complex64::new(s, 0.0);
            
            // Time evolution (simplified)
            let dt = 0.01;
            let evolution_operator = (&hamiltonian * Complex64::new(0.0, -dt)).map(|x| x.exp());
            
            // Apply evolution
            let state_vector = DVector::from_vec(initial_state.amplitudes.clone());
            let evolved_state = &evolution_operator * state_vector;
            
            initial_state.amplitudes = evolved_state.as_slice().to_vec();
            initial_state.normalize()?;
            
            // Calculate energy expectation
            let energy = initial_state.expectation_value(&hamiltonian)?;
            results.push(energy.re);
        }
        
        Ok(results)
    }
}

/// Quantum error correction
#[derive(Debug, Clone)]
pub struct QuantumErrorCorrection {
    /// Error syndrome
    syndrome: Vec<bool>,
    /// Correction operations
    corrections: Vec<QuantumGate>,
}

impl QuantumErrorCorrection {
    /// Create new quantum error correction
    pub fn new() -> Self {
        Self {
            syndrome: Vec::new(),
            corrections: Vec::new(),
        }
    }
    
    /// Detect errors in quantum state
    pub fn detect_errors(&mut self, state: &QuantumState) -> Result<bool, HedgeError> {
        // Simplified error detection based on purity
        let purity = state.purity();
        let has_errors = purity < 0.9;
        
        if has_errors {
            self.syndrome = vec![true; state.amplitudes.len()];
        } else {
            self.syndrome = vec![false; state.amplitudes.len()];
        }
        
        Ok(has_errors)
    }
    
    /// Correct errors in quantum state
    pub fn correct_errors(&self, state: &mut QuantumState) -> Result<(), HedgeError> {
        if self.syndrome.iter().any(|&x| x) {
            // Apply correction by renormalizing
            state.normalize()?;
            
            // Apply additional correction gates if needed
            for correction in &self.corrections {
                state.apply_gate(correction)?;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let labels = vec!["expert1".to_string(), "expert2".to_string()];
        let state = QuantumState::new(labels, 0.01);
        
        assert_eq!(state.labels.len(), 2);
        assert_eq!(state.amplitudes.len(), 2);
        assert_eq!(state.probabilities.len(), 2);
    }

    #[test]
    fn test_quantum_gate_hadamard() {
        let hadamard = QuantumGate::hadamard(1);
        assert_eq!(hadamard.matrix.nrows(), 2);
        assert_eq!(hadamard.matrix.ncols(), 2);
        assert_eq!(hadamard.name, "Hadamard(1)");
    }

    #[test]
    fn test_quantum_hedge_algorithm() {
        let expert_names = vec!["expert1".to_string(), "expert2".to_string()];
        let config = HedgeConfig::default();
        let mut qha = QuantumHedgeAlgorithm::new(expert_names, config).unwrap();
        
        assert_eq!(qha.expert_weights.len(), 2);
        assert_eq!(qha.time_step, 0);
        
        let market_data = MarketData::new(
            "BTCUSD".to_string(),
            chrono::Utc::now(),
            [100.0, 105.0, 95.0, 102.0, 1000.0]
        );
        
        let mut predictions = HashMap::new();
        predictions.insert("expert1".to_string(), 0.05);
        predictions.insert("expert2".to_string(), -0.02);
        
        qha.update(&market_data, &predictions).unwrap();
        
        assert_eq!(qha.time_step, 1);
    }

    #[test]
    fn test_quantum_measurement() {
        let labels = vec!["expert1".to_string(), "expert2".to_string()];
        let mut state = QuantumState::new(labels, 0.01);
        
        let measurement = state.measure().unwrap();
        assert!(measurement < 2);
        
        // After measurement, state should be collapsed
        assert_eq!(state.probabilities.iter().filter(|&&p| p > 0.9).count(), 1);
    }

    #[test]
    fn test_quantum_state_entropy() {
        let labels = vec!["expert1".to_string(), "expert2".to_string()];
        let state = QuantumState::new(labels, 0.01);
        
        let entropy = state.entropy();
        assert!(entropy > 0.0);
        assert!(entropy <= 2.0_f64.ln());
    }
}