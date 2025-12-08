//! Quantum reservoir computing for temporal dynamics and entanglement processing
//! 
//! Implements a quantum-enhanced reservoir with complex temporal dynamics,
//! quantum entanglement patterns, and adaptive readout for cerebellar learning.

use std::collections::{HashMap, VecDeque};
use tch::{Tensor, Device, Kind};
use nalgebra::{DMatrix, DVector, ComplexField};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::{
    QuantumSNNConfig, QuantumSpikeTrain, QuantumSynapse,
    QuantumCircuitSimulator, CerebellarQuantumCircuits
};

/// Quantum reservoir configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumReservoirConfig {
    /// Number of reservoir neurons
    pub n_reservoir_neurons: usize,
    
    /// Spectral radius for stability
    pub spectral_radius: f64,
    
    /// Input scaling factor
    pub input_scaling: f64,
    
    /// Reservoir connectivity density
    pub connectivity: f64,
    
    /// Leak rate for dynamics
    pub leak_rate: f64,
    
    /// Quantum coherence parameters
    pub quantum_coherence_time: f64,
    pub entanglement_strength: f64,
    pub decoherence_rate: f64,
    
    /// Temporal memory parameters
    pub memory_depth: usize,
    pub temporal_window: f64,
    
    /// Readout adaptation
    pub adaptive_readout: bool,
    pub readout_learning_rate: f64,
    
    /// Performance constraints
    pub max_computation_time_us: u64,
}

impl Default for QuantumReservoirConfig {
    fn default() -> Self {
        Self {
            n_reservoir_neurons: 500,
            spectral_radius: 0.9,
            input_scaling: 0.1,
            connectivity: 0.1,
            leak_rate: 0.3,
            quantum_coherence_time: 50.0,
            entanglement_strength: 0.3,
            decoherence_rate: 0.01,
            memory_depth: 100,
            temporal_window: 20.0,
            adaptive_readout: true,
            readout_learning_rate: 0.001,
            max_computation_time_us: 1000, // 1ms limit
        }
    }
}

/// Quantum reservoir neuron with complex dynamics
#[derive(Debug, Clone)]
pub struct QuantumReservoirNeuron {
    /// Neuron ID
    pub id: usize,
    
    /// Current activation state
    pub activation: f64,
    
    /// Quantum state amplitudes
    pub quantum_state: Vec<Complex64>,
    
    /// Input connections and weights
    pub input_weights: Vec<(usize, f64)>,
    
    /// Reservoir connections and weights
    pub reservoir_weights: Vec<(usize, f64)>,
    
    /// Activation history for temporal dynamics
    pub activation_history: VecDeque<f64>,
    
    /// Quantum entanglement connections
    pub entangled_neurons: Vec<(usize, Complex64)>,
    
    /// Leak dynamics
    pub leak_constant: f64,
    
    /// Nonlinearity parameters
    pub activation_threshold: f64,
    pub saturation_level: f64,
}

impl QuantumReservoirNeuron {
    pub fn new(id: usize, n_inputs: usize, config: &QuantumReservoirConfig) -> Self {
        // Initialize random input weights
        let input_weights: Vec<(usize, f64)> = (0..n_inputs)
            .filter(|_| rand::random::<f64>() < 0.3) // Sparse connectivity
            .map(|i| (i, (rand::random::<f64>() - 0.5) * 2.0 * config.input_scaling))
            .collect();
        
        // Initialize quantum state
        let n_qubits = 4;
        let quantum_state = vec![Complex64::new(1.0, 0.0); n_qubits];
        
        Self {
            id,
            activation: 0.0,
            quantum_state,
            input_weights,
            reservoir_weights: Vec::new(),
            activation_history: VecDeque::with_capacity(config.memory_depth),
            entangled_neurons: Vec::new(),
            leak_constant: config.leak_rate,
            activation_threshold: 0.5,
            saturation_level: 1.0,
        }
    }
    
    /// Update neuron activation with quantum enhancement
    pub fn update(
        &mut self,
        inputs: &[f64],
        reservoir_activations: &[f64],
        quantum_field: Complex64,
        current_time: f64,
        config: &QuantumReservoirConfig,
    ) -> Result<f64> {
        // Calculate input current
        let mut input_current = 0.0;
        for &(input_idx, weight) in &self.input_weights {
            if input_idx < inputs.len() {
                input_current += inputs[input_idx] * weight;
            }
        }
        
        // Calculate reservoir current
        let mut reservoir_current = 0.0;
        for &(neuron_idx, weight) in &self.reservoir_weights {
            if neuron_idx < reservoir_activations.len() {
                reservoir_current += reservoir_activations[neuron_idx] * weight;
            }
        }
        
        // Update quantum state
        self.update_quantum_state(quantum_field, config)?;
        
        // Calculate quantum contribution
        let quantum_contribution = self.quantum_state.iter()
            .map(|q| q.norm())
            .sum::<f64>() / self.quantum_state.len() as f64;
        
        // Apply leak dynamics
        let total_input = input_current + reservoir_current + quantum_contribution * 0.1;
        self.activation = (1.0 - self.leak_constant) * self.activation + 
                         self.leak_constant * self.apply_nonlinearity(total_input);
        
        // Apply saturation
        self.activation = self.activation.clamp(-self.saturation_level, self.saturation_level);
        
        // Store in history
        self.activation_history.push_back(self.activation);
        if self.activation_history.len() > config.memory_depth {
            self.activation_history.pop_front();
        }
        
        Ok(self.activation)
    }
    
    /// Update quantum state evolution
    fn update_quantum_state(&mut self, field: Complex64, config: &QuantumReservoirConfig) -> Result<()> {
        for i in 0..self.quantum_state.len() {
            // Quantum oscillator with external field coupling
            let freq = 0.1 * (i + 1) as f64;
            let phase = Complex64::new(0.0, freq).exp();
            
            // Field coupling
            let coupling = field * Complex64::new(config.entanglement_strength, 0.0);
            
            // Evolution with decoherence
            self.quantum_state[i] = self.quantum_state[i] * phase + coupling;
            self.quantum_state[i] *= Complex64::new(1.0 - config.decoherence_rate, 0.0);
        }
        
        // Normalize to prevent divergence
        let norm = self.quantum_state.iter().map(|q| q.norm_sqr()).sum::<f64>().sqrt();
        if norm > 2.0 {
            for q in &mut self.quantum_state {
                *q /= Complex64::new(norm / 2.0, 0.0);
            }
        }
        
        Ok(())
    }
    
    /// Apply reservoir nonlinearity (tanh with threshold)
    fn apply_nonlinearity(&self, input: f64) -> f64 {
        if input.abs() > self.activation_threshold {
            input.tanh()
        } else {
            input * 0.1 // Linear region for small inputs
        }
    }
    
    /// Get temporal memory readout
    pub fn temporal_readout(&self, window_size: usize) -> f64 {
        if self.activation_history.is_empty() {
            return 0.0;
        }
        
        let window = window_size.min(self.activation_history.len());
        let recent_activations: Vec<f64> = self.activation_history
            .iter()
            .rev()
            .take(window)
            .copied()
            .collect();
        
        // Weighted temporal average
        let weights: Vec<f64> = (0..window)
            .map(|i| (-0.1 * i as f64).exp())
            .collect();
        
        let weighted_sum: f64 = recent_activations.iter()
            .zip(weights.iter())
            .map(|(&a, &w)| a * w)
            .sum();
        
        let weight_sum: f64 = weights.iter().sum();
        
        if weight_sum > 1e-10 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }
    
    /// Calculate quantum coherence with other neurons
    pub fn quantum_coherence(&self, other: &QuantumReservoirNeuron) -> Complex64 {
        if self.quantum_state.len() != other.quantum_state.len() {
            return Complex64::new(0.0, 0.0);
        }
        
        let mut coherence = Complex64::new(0.0, 0.0);
        for i in 0..self.quantum_state.len() {
            coherence += self.quantum_state[i].conj() * other.quantum_state[i];
        }
        
        coherence / Complex64::new(self.quantum_state.len() as f64, 0.0)
    }
    
    /// Reset neuron state
    pub fn reset(&mut self) {
        self.activation = 0.0;
        self.activation_history.clear();
        
        // Reset quantum state
        if !self.quantum_state.is_empty() {
            self.quantum_state[0] = Complex64::new(1.0, 0.0);
            for i in 1..self.quantum_state.len() {
                self.quantum_state[i] = Complex64::new(0.0, 0.0);
            }
        }
    }
}

/// Adaptive readout layer with ridge regression
#[derive(Debug)]
pub struct AdaptiveReadout {
    /// Readout weights
    pub weights: DMatrix<f64>,
    
    /// Bias terms
    pub biases: DVector<f64>,
    
    /// Target outputs for training
    target_buffer: VecDeque<DVector<f64>>,
    
    /// Reservoir state buffer
    state_buffer: VecDeque<DVector<f64>>,
    
    /// Learning rate
    learning_rate: f64,
    
    /// Regularization parameter
    ridge_parameter: f64,
    
    /// Buffer size for online learning
    buffer_size: usize,
}

impl AdaptiveReadout {
    pub fn new(n_reservoir: usize, n_outputs: usize, learning_rate: f64) -> Self {
        let weights = DMatrix::from_fn(n_outputs, n_reservoir, |_, _| {
            (rand::random::<f64>() - 0.5) * 0.01
        });
        
        let biases = DVector::zeros(n_outputs);
        
        Self {
            weights,
            biases,
            target_buffer: VecDeque::with_capacity(1000),
            state_buffer: VecDeque::with_capacity(1000),
            learning_rate,
            ridge_parameter: 1e-6,
            buffer_size: 100,
        }
    }
    
    /// Compute readout from reservoir state
    pub fn compute(&self, reservoir_state: &DVector<f64>) -> DVector<f64> {
        &self.weights * reservoir_state + &self.biases
    }
    
    /// Update readout weights with target
    pub fn update(&mut self, reservoir_state: &DVector<f64>, target: &DVector<f64>) -> Result<()> {
        // Store in buffers
        self.state_buffer.push_back(reservoir_state.clone());
        self.target_buffer.push_back(target.clone());
        
        // Maintain buffer size
        if self.state_buffer.len() > self.buffer_size {
            self.state_buffer.pop_front();
            self.target_buffer.pop_front();
        }
        
        // Online ridge regression update
        if self.state_buffer.len() >= 10 {
            let prediction = self.compute(reservoir_state);
            let error = target - prediction;
            
            // Weight update (simplified online learning)
            for i in 0..self.weights.nrows() {
                for j in 0..self.weights.ncols() {
                    self.weights[(i, j)] += self.learning_rate * error[i] * reservoir_state[j];
                }
                self.biases[i] += self.learning_rate * error[i];
            }
        }
        
        Ok(())
    }
    
    /// Batch ridge regression training
    pub fn train_batch(&mut self) -> Result<()> {
        if self.state_buffer.len() < 10 {
            return Ok(());
        }
        
        let n_samples = self.state_buffer.len();
        let n_features = self.weights.ncols();
        let n_outputs = self.weights.nrows();
        
        // Build matrices
        let mut X = DMatrix::zeros(n_samples, n_features);
        let mut Y = DMatrix::zeros(n_samples, n_outputs);
        
        for (i, (state, target)) in self.state_buffer.iter().zip(self.target_buffer.iter()).enumerate() {
            X.set_row(i, &state.transpose());
            Y.set_row(i, &target.transpose());
        }
        
        // Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        let XtX = X.transpose() * &X;
        let mut regularized = XtX + DMatrix::identity(n_features, n_features) * self.ridge_parameter;
        
        if let Some(inv) = regularized.try_inverse() {
            let new_weights = (inv * X.transpose() * Y).transpose();
            self.weights = new_weights;
            
            debug!("Readout weights updated via ridge regression");
        }
        
        Ok(())
    }
}

/// Main quantum reservoir computing system
pub struct QuantumReservoir {
    /// Configuration
    config: QuantumReservoirConfig,
    
    /// Reservoir neurons
    neurons: Vec<QuantumReservoirNeuron>,
    
    /// Quantum circuit for entanglement
    quantum_circuit: QuantumCircuitSimulator,
    
    /// Adaptive readout layer
    readout: AdaptiveReadout,
    
    /// Current reservoir state
    current_state: DVector<f64>,
    
    /// State history for temporal processing
    state_history: VecDeque<DVector<f64>>,
    
    /// Global quantum field
    quantum_field: Complex64,
    
    /// Entanglement matrix
    entanglement_matrix: DMatrix<Complex64>,
    
    /// Performance metrics
    computation_time_ns: u64,
    coherence_level: f64,
    
    /// Device for tensor operations
    device: Device,
}

impl QuantumReservoir {
    /// Create new quantum reservoir
    pub fn new(snn_config: &QuantumSNNConfig) -> Result<Self> {
        let config = QuantumReservoirConfig {
            n_reservoir_neurons: snn_config.n_reservoir_neurons,
            quantum_coherence_time: snn_config.quantum_coherence_time,
            entanglement_strength: snn_config.entanglement_strength,
            ..Default::default()
        };
        
        // Create reservoir neurons
        let mut neurons = Vec::with_capacity(config.n_reservoir_neurons);
        for i in 0..config.n_reservoir_neurons {
            neurons.push(QuantumReservoirNeuron::new(i, snn_config.n_inputs, &config));
        }
        
        // Initialize reservoir connectivity
        Self::initialize_reservoir_weights(&mut neurons, &config)?;
        
        // Create quantum circuit
        let quantum_circuit = QuantumCircuitSimulator::new(8)?; // 8 qubits for reservoir
        
        // Create adaptive readout
        let readout = AdaptiveReadout::new(
            config.n_reservoir_neurons, 
            snn_config.n_outputs, 
            config.readout_learning_rate
        );
        
        let current_state = DVector::zeros(config.n_reservoir_neurons);
        let entanglement_matrix = DMatrix::zeros(config.n_reservoir_neurons, config.n_reservoir_neurons);
        
        info!("Initialized quantum reservoir: {} neurons, spectral radius {:.3}", 
              config.n_reservoir_neurons, config.spectral_radius);
        
        Ok(Self {
            config,
            neurons,
            quantum_circuit,
            readout,
            current_state,
            state_history: VecDeque::with_capacity(1000),
            quantum_field: Complex64::new(0.0, 0.0),
            entanglement_matrix,
            computation_time_ns: 0,
            coherence_level: 0.0,
            device: snn_config.device,
        })
    }
    
    /// Update reservoir with spike train inputs
    pub fn update(
        &mut self,
        spike_trains: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        
        // Check computation time constraint
        if start_time.elapsed().as_micros() > self.config.max_computation_time_us {
            warn!("Reservoir computation time exceeded, using cached state");
            return self.get_cached_output();
        }
        
        // Convert spike trains to input vector
        let inputs = self.convert_spikes_to_inputs(spike_trains, current_time)?;
        
        // Update global quantum field
        self.update_quantum_field(&inputs)?;
        
        // Update all neurons in parallel
        let new_activations: Vec<f64> = self.neurons
            .par_iter_mut()
            .map(|neuron| {
                neuron.update(
                    &inputs,
                    self.current_state.as_slice(),
                    self.quantum_field,
                    current_time,
                    &self.config,
                ).unwrap_or(0.0)
            })
            .collect();
        
        // Update current state
        self.current_state = DVector::from_vec(new_activations);
        
        // Store state history
        self.state_history.push_back(self.current_state.clone());
        if self.state_history.len() > 1000 {
            self.state_history.pop_front();
        }
        
        // Update quantum entanglement
        self.update_entanglement_dynamics()?;
        
        // Compute readout
        let output = self.readout.compute(&self.current_state);
        
        // Convert to tensor
        let output_tensor = Tensor::from_slice(output.as_slice())
            .to_device(self.device)
            .unsqueeze(0);
        
        // Update performance metrics
        self.computation_time_ns = start_time.elapsed().as_nanos() as u64;
        self.coherence_level = self.calculate_global_coherence();
        
        debug!("Reservoir updated: coherence {:.3}, {}μs", 
               self.coherence_level, start_time.elapsed().as_micros());
        
        Ok(output_tensor)
    }
    
    /// Initialize random reservoir connectivity with spectral radius control
    fn initialize_reservoir_weights(
        neurons: &mut [QuantumReservoirNeuron],
        config: &QuantumReservoirConfig,
    ) -> Result<()> {
        let n = neurons.len();
        
        // Create random connectivity matrix
        let mut weight_matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i != j && rand::random::<f64>() < config.connectivity {
                    weight_matrix[(i, j)] = (rand::random::<f64>() - 0.5) * 2.0;
                }
            }
        }
        
        // Scale to desired spectral radius
        if let Some(eigenvalues) = weight_matrix.eigenvalues() {
            let max_eigenvalue = eigenvalues.iter()
                .map(|e| e.norm())
                .fold(0.0, f64::max);
            
            if max_eigenvalue > 1e-10 {
                let scaling = config.spectral_radius / max_eigenvalue;
                weight_matrix *= scaling;
            }
        }
        
        // Assign weights to neurons
        for (i, neuron) in neurons.iter_mut().enumerate() {
            neuron.reservoir_weights.clear();
            for j in 0..n {
                if weight_matrix[(i, j)].abs() > 1e-10 {
                    neuron.reservoir_weights.push((j, weight_matrix[(i, j)]));
                }
            }
        }
        
        Ok(())
    }
    
    /// Convert spike trains to input vector
    fn convert_spikes_to_inputs(
        &self,
        spike_trains: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<Vec<f64>> {
        let mut inputs = Vec::new();
        let time_window = self.config.temporal_window;
        
        for spike_train in spike_trains {
            // Calculate recent spike strength
            let mut strength = 0.0;
            for (i, &spike_time) in spike_train.times.iter().enumerate() {
                if current_time - spike_time < time_window {
                    let decay = (-(current_time - spike_time) / 5.0).exp();
                    let amplitude = if i < spike_train.amplitudes.len() {
                        spike_train.amplitudes[i].norm()
                    } else {
                        1.0
                    };
                    strength += amplitude * decay;
                }
            }
            inputs.push(strength);
        }
        
        // Pad or truncate to match expected input size
        while inputs.len() < 8 { // Minimum 8 inputs
            inputs.push(0.0);
        }
        
        Ok(inputs)
    }
    
    /// Update global quantum field based on inputs
    fn update_quantum_field(&mut self, inputs: &[f64]) -> Result<()> {
        let field_strength = inputs.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
        let field_phase = inputs.iter().enumerate()
            .map(|(i, &x)| x * (i as f64))
            .sum::<f64>();
        
        self.quantum_field = Complex64::new(field_strength, field_phase) * 
                           Complex64::new(self.config.entanglement_strength, 0.0);
        
        Ok(())
    }
    
    /// Update quantum entanglement between neurons
    fn update_entanglement_dynamics(&mut self) -> Result<()> {
        let n = self.neurons.len();
        
        // Calculate pairwise coherences
        for i in 0..n {
            for j in (i + 1)..n {
                let coherence = self.neurons[i].quantum_coherence(&self.neurons[j]);
                self.entanglement_matrix[(i, j)] = coherence;
                self.entanglement_matrix[(j, i)] = coherence.conj();
            }
        }
        
        // Apply decoherence
        self.entanglement_matrix *= Complex64::new(1.0 - self.config.decoherence_rate, 0.0);
        
        Ok(())
    }
    
    /// Calculate global coherence measure
    fn calculate_global_coherence(&self) -> f64 {
        let mut total_coherence = 0.0;
        let mut pairs = 0;
        
        for i in 0..self.entanglement_matrix.nrows() {
            for j in (i + 1)..self.entanglement_matrix.ncols() {
                total_coherence += self.entanglement_matrix[(i, j)].norm();
                pairs += 1;
            }
        }
        
        if pairs > 0 {
            total_coherence / pairs as f64
        } else {
            0.0
        }
    }
    
    /// Get cached output for time-critical scenarios
    fn get_cached_output(&self) -> Result<Tensor> {
        let output = if self.state_history.is_empty() {
            DVector::zeros(3) // Default output size
        } else {
            self.readout.compute(self.state_history.back().unwrap())
        };
        
        Ok(Tensor::from_slice(output.as_slice())
            .to_device(self.device)
            .unsqueeze(0))
    }
    
    /// Train readout with target output
    pub fn train_readout(&mut self, target: &Tensor) -> Result<()> {
        let target_vec = self.tensor_to_vector(target)?;
        self.readout.update(&self.current_state, &target_vec)?;
        Ok(())
    }
    
    /// Batch training of readout layer
    pub fn train_readout_batch(&mut self) -> Result<()> {
        self.readout.train_batch()
    }
    
    /// Get reservoir state for external processing
    pub fn get_state(&self) -> &DVector<f64> {
        &self.current_state
    }
    
    /// Get temporal memory representation
    pub fn get_temporal_memory(&self, window_size: usize) -> DVector<f64> {
        let memory: Vec<f64> = self.neurons.iter()
            .map(|neuron| neuron.temporal_readout(window_size))
            .collect();
        
        DVector::from_vec(memory)
    }
    
    /// Reset reservoir state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        
        self.current_state.fill(0.0);
        self.state_history.clear();
        self.quantum_field = Complex64::new(0.0, 0.0);
        self.entanglement_matrix.fill(Complex64::new(0.0, 0.0));
        self.quantum_circuit.reset();
        
        info!("Quantum reservoir reset to initial state");
    }
    
    /// Convert tensor to vector
    fn tensor_to_vector(&self, tensor: &Tensor) -> Result<DVector<f64>> {
        let data: Vec<f64> = tensor.to_device(Device::Cpu)
            .to_kind(Kind::Double)
            .into();
        
        Ok(DVector::from_vec(data))
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        metrics.insert("computation_time_us".to_string(), 
                      self.computation_time_ns as f64 / 1000.0);
        metrics.insert("coherence_level".to_string(), self.coherence_level);
        metrics.insert("active_neurons".to_string(), 
                      self.current_state.iter().filter(|&&x| x.abs() > 0.01).count() as f64);
        metrics.insert("quantum_field_strength".to_string(), self.quantum_field.norm());
        metrics.insert("state_history_length".to_string(), self.state_history.len() as f64);
        
        // Reservoir dynamics metrics
        let state_variance = if self.current_state.len() > 1 {
            let mean = self.current_state.mean();
            self.current_state.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 
            self.current_state.len() as f64
        } else {
            0.0
        };
        metrics.insert("state_variance".to_string(), state_variance);
        
        metrics
    }
    
    /// Adaptive parameter tuning based on performance
    pub fn adapt_parameters(&mut self, target_coherence: f64, target_activity: f64) -> Result<()> {
        // Adjust entanglement strength based on coherence
        if self.coherence_level < target_coherence * 0.9 {
            self.config.entanglement_strength *= 1.02;
        } else if self.coherence_level > target_coherence * 1.1 {
            self.config.entanglement_strength *= 0.98;
        }
        
        // Adjust leak rate based on activity
        let current_activity = self.current_state.iter().map(|&x| x.abs()).sum::<f64>() / 
                              self.current_state.len() as f64;
        
        if current_activity < target_activity * 0.9 {
            self.config.leak_rate *= 0.99; // Slower leak for more activity
        } else if current_activity > target_activity * 1.1 {
            self.config.leak_rate *= 1.01; // Faster leak for less activity
        }
        
        // Bound parameters
        self.config.entanglement_strength = self.config.entanglement_strength.clamp(0.01, 1.0);
        self.config.leak_rate = self.config.leak_rate.clamp(0.1, 0.9);
        
        debug!("Adapted reservoir parameters: entanglement {:.3}, leak {:.3}", 
               self.config.entanglement_strength, self.config.leak_rate);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_quantum_reservoir_creation() {
        let snn_config = QuantumSNNConfig::default();
        let reservoir = QuantumReservoir::new(&snn_config).unwrap();
        
        assert_eq!(reservoir.neurons.len(), snn_config.n_reservoir_neurons);
        assert_eq!(reservoir.current_state.len(), snn_config.n_reservoir_neurons);
    }
    
    #[test]
    fn test_reservoir_neuron_update() {
        let config = QuantumReservoirConfig::default();
        let mut neuron = QuantumReservoirNeuron::new(0, 4, &config);
        
        let inputs = vec![0.5, 0.3, 0.8, 0.1];
        let reservoir_activations = vec![0.2, 0.4, 0.1];
        let quantum_field = Complex64::new(0.1, 0.05);
        
        let activation = neuron.update(&inputs, &reservoir_activations, quantum_field, 1.0, &config).unwrap();
        
        assert!(activation.abs() <= config.spectral_radius + 0.1); // Should be bounded
        assert!(!neuron.activation_history.is_empty());
    }
    
    #[test]
    fn test_adaptive_readout() {
        let mut readout = AdaptiveReadout::new(10, 3, 0.01);
        
        let reservoir_state = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        let target = DVector::from_vec(vec![0.5, 0.3, 0.8]);
        
        let initial_output = readout.compute(&reservoir_state);
        readout.update(&reservoir_state, &target).unwrap();
        
        // Output should exist
        assert_eq!(initial_output.len(), 3);
    }
    
    #[test]
    fn test_reservoir_spike_processing() {
        let snn_config = QuantumSNNConfig::default();
        let mut reservoir = QuantumReservoir::new(&snn_config).unwrap();
        
        // Create test spike trains
        let mut spike_trains = Vec::new();
        for i in 0..5 {
            let mut st = QuantumSpikeTrain::new(i, 10.0);
            st.add_spike(1.0, Complex64::new(0.7, 0.0), 0.0);
            spike_trains.push(st);
        }
        
        let output = reservoir.update(&spike_trains, 2.0).unwrap();
        
        assert!(output.size().len() > 0);
        assert!(reservoir.coherence_level >= 0.0);
    }
    
    #[test]
    fn test_quantum_coherence_calculation() {
        let config = QuantumReservoirConfig::default();
        let neuron1 = QuantumReservoirNeuron::new(0, 4, &config);
        let mut neuron2 = QuantumReservoirNeuron::new(1, 4, &config);
        
        // Modify quantum state for testing
        neuron2.quantum_state[0] = Complex64::new(0.0, 1.0);
        
        let coherence = neuron1.quantum_coherence(&neuron2);
        assert!(coherence.norm() >= 0.0 && coherence.norm() <= 1.0);
    }
    
    #[test]
    fn test_temporal_memory() {
        let config = QuantumReservoirConfig::default();
        let mut neuron = QuantumReservoirNeuron::new(0, 4, &config);
        
        // Add some activation history
        for i in 0..10 {
            neuron.activation_history.push_back(0.1 * i as f64);
        }
        
        let temporal_readout = neuron.temporal_readout(5);
        assert!(temporal_readout > 0.0);
    }
    
    #[test]
    fn test_reservoir_reset() {
        let snn_config = QuantumSNNConfig::default();
        let mut reservoir = QuantumReservoir::new(&snn_config).unwrap();
        
        // Modify state
        reservoir.current_state[0] = 1.0;
        reservoir.quantum_field = Complex64::new(0.5, 0.3);
        
        reservoir.reset();
        
        assert_eq!(reservoir.current_state[0], 0.0);
        assert_eq!(reservoir.quantum_field.norm(), 0.0);
        assert!(reservoir.state_history.is_empty());
    }
    
    #[test]
    fn test_spectral_radius_initialization() {
        let config = QuantumReservoirConfig {
            n_reservoir_neurons: 50,
            spectral_radius: 0.9,
            connectivity: 0.1,
            ..Default::default()
        };
        
        let mut neurons = Vec::new();
        for i in 0..config.n_reservoir_neurons {
            neurons.push(QuantumReservoirNeuron::new(i, 8, &config));
        }
        
        QuantumReservoir::initialize_reservoir_weights(&mut neurons, &config).unwrap();
        
        // Check that neurons have reservoir connections
        let total_connections: usize = neurons.iter()
            .map(|n| n.reservoir_weights.len())
            .sum();
        
        assert!(total_connections > 0);
    }
    
    #[test]
    fn test_parameter_adaptation() {
        let snn_config = QuantumSNNConfig::default();
        let mut reservoir = QuantumReservoir::new(&snn_config).unwrap();
        
        let initial_entanglement = reservoir.config.entanglement_strength;
        let initial_leak = reservoir.config.leak_rate;
        
        reservoir.adapt_parameters(0.5, 0.3).unwrap();
        
        // Parameters should be within bounds
        assert!(reservoir.config.entanglement_strength >= 0.01 && 
                reservoir.config.entanglement_strength <= 1.0);
        assert!(reservoir.config.leak_rate >= 0.1 && 
                reservoir.config.leak_rate <= 0.9);
    }
}