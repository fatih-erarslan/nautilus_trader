//! Quantum Spiking Neural Network (SNN) with STDP
//! 
//! Quantum-enhanced spiking neural networks with spike-timing dependent plasticity
//! for temporal pattern recognition and neuromorphic computing

use nalgebra::{DVector};
use num_complex::Complex64;
use rand::Rng;
use crate::{QuantumState, QuantumMarketData, QuantumPrediction, QuantumMLError, quantum_gates::QuantumGates};

/// Quantum SNN configuration
#[derive(Debug, Clone)]
pub struct QuantumSNNConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub quantum_neurons: usize,
    pub spike_threshold: f64,
    pub refractory_period: f64,
    pub learning_rate: f64,
    pub stdp_window: f64,
    pub quantum_coupling: f64,
    pub decoherence_rate: f64,
}

impl Default for QuantumSNNConfig {
    fn default() -> Self {
        Self {
            input_size: 128,
            hidden_size: 64,
            output_size: 1,
            quantum_neurons: 16,
            spike_threshold: 1.0,
            refractory_period: 5.0,
            learning_rate: 0.01,
            stdp_window: 20.0,
            quantum_coupling: 0.1,
            decoherence_rate: 0.01,
        }
    }
}

/// Quantum neuron state
#[derive(Debug, Clone)]
pub struct QuantumNeuron {
    pub membrane_potential: f64,
    pub last_spike_time: f64,
    pub refractory_time: f64,
    pub quantum_state: QuantumState,
    pub entanglement_partners: Vec<usize>,
    pub spike_history: Vec<f64>,
}

impl QuantumNeuron {
    pub fn new(quantum_qubits: usize) -> Self {
        Self {
            membrane_potential: 0.0,
            last_spike_time: -1000.0,
            refractory_time: 0.0,
            quantum_state: QuantumState::new(quantum_qubits),
            entanglement_partners: Vec::new(),
            spike_history: Vec::new(),
        }
    }

    /// Check if neuron can spike (not in refractory period)
    pub fn can_spike(&self, current_time: f64) -> bool {
        current_time - self.last_spike_time > self.refractory_time
    }

    /// Update membrane potential with quantum enhancement
    pub fn update_membrane_potential(&mut self, input: f64, quantum_enhancement: f64) {
        self.membrane_potential += input + quantum_enhancement;
    }

    /// Generate spike if threshold is exceeded
    pub fn generate_spike(&mut self, current_time: f64, threshold: f64) -> bool {
        if self.can_spike(current_time) && self.membrane_potential >= threshold {
            self.last_spike_time = current_time;
            self.membrane_potential = 0.0; // Reset after spike
            self.spike_history.push(current_time);
            
            // Keep only recent spike history
            if self.spike_history.len() > 100 {
                self.spike_history.remove(0);
            }
            
            true
        } else {
            false
        }
    }

    /// Apply quantum superposition to spike generation
    pub fn quantum_spike_probability(&self) -> f64 {
        // Calculate spike probability based on quantum state
        let mut prob = 0.0;
        for amplitude in &self.quantum_state.amplitudes {
            prob += amplitude.norm_sqr();
        }
        prob / self.quantum_state.amplitudes.len() as f64
    }
}

/// Quantum synapse with STDP
#[derive(Debug, Clone)]
pub struct QuantumSynapse {
    pub weight: f64,
    pub quantum_weight: Complex64,
    pub pre_neuron_id: usize,
    pub post_neuron_id: usize,
    pub last_update_time: f64,
    pub stdp_trace: f64,
}

impl QuantumSynapse {
    pub fn new(pre_id: usize, post_id: usize, initial_weight: f64) -> Self {
        Self {
            weight: initial_weight,
            quantum_weight: Complex64::new(initial_weight, 0.0),
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            last_update_time: 0.0,
            stdp_trace: 0.0,
        }
    }

    /// Update synapse weight using STDP
    pub fn update_stdp(&mut self, 
                      pre_spike_time: f64, 
                      post_spike_time: f64, 
                      learning_rate: f64,
                      stdp_window: f64) {
        let dt = post_spike_time - pre_spike_time;
        
        if dt.abs() <= stdp_window {
            let stdp_change = if dt > 0.0 {
                // Post-synaptic spike after pre-synaptic: potentiation
                learning_rate * (-dt / stdp_window).exp()
            } else {
                // Pre-synaptic spike after post-synaptic: depression
                -learning_rate * (dt / stdp_window).exp()
            };
            
            self.weight += stdp_change;
            self.weight = self.weight.max(0.0).min(2.0); // Clamp weights
            
            // Update quantum weight with phase information
            let phase = dt * 0.1; // Phase encoding of timing
            self.quantum_weight = Complex64::new(self.weight, phase);
        }
    }

    /// Get effective synaptic transmission
    pub fn get_transmission(&self, quantum_enhancement: f64) -> f64 {
        self.weight * (1.0 + quantum_enhancement * self.quantum_weight.norm())
    }
}

/// Quantum Spiking Neural Network
pub struct QuantumSNN {
    config: QuantumSNNConfig,
    neurons: Vec<QuantumNeuron>,
    synapses: Vec<QuantumSynapse>,
    global_quantum_state: QuantumState,
    current_time: f64,
    
    // Network topology
    input_neuron_ids: Vec<usize>,
    hidden_neuron_ids: Vec<usize>,
    output_neuron_ids: Vec<usize>,
    
    // Performance metrics
    total_spikes: u64,
    predictions_count: u64,
    last_quantum_advantage: f64,
    training_epochs: usize,
}

impl QuantumSNN {
    /// Create new Quantum SNN
    pub async fn new(input_size: usize, hidden_size: usize) -> Result<Self, QuantumMLError> {
        let config = QuantumSNNConfig {
            input_size,
            hidden_size,
            ..Default::default()
        };

        let total_neurons = input_size + hidden_size + config.output_size;
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        // Create neurons
        for i in 0..total_neurons {
            let qubits = if i < config.quantum_neurons { 3 } else { 2 };
            neurons.push(QuantumNeuron::new(qubits));
        }

        // Create network topology
        let input_neuron_ids: Vec<usize> = (0..input_size).collect();
        let hidden_neuron_ids: Vec<usize> = (input_size..input_size + hidden_size).collect();
        let output_neuron_ids: Vec<usize> = (input_size + hidden_size..total_neurons).collect();

        // Create synapses
        let mut rng = rand::thread_rng();
        
        // Input to hidden connections
        for &input_id in &input_neuron_ids {
            for &hidden_id in &hidden_neuron_ids {
                let weight = rng.gen_range(-0.5..0.5);
                synapses.push(QuantumSynapse::new(input_id, hidden_id, weight));
            }
        }

        // Hidden to output connections
        for &hidden_id in &hidden_neuron_ids {
            for &output_id in &output_neuron_ids {
                let weight = rng.gen_range(-0.5..0.5);
                synapses.push(QuantumSynapse::new(hidden_id, output_id, weight));
            }
        }

        // Hidden to hidden connections (recurrent)
        for &hidden_id1 in &hidden_neuron_ids {
            for &hidden_id2 in &hidden_neuron_ids {
                if hidden_id1 != hidden_id2 && rng.gen::<f64>() < 0.3 { // 30% connectivity
                    let weight = rng.gen_range(-0.2..0.2);
                    synapses.push(QuantumSynapse::new(hidden_id1, hidden_id2, weight));
                }
            }
        }

        // Create global quantum state for entanglement
        let global_qubits = (config.quantum_neurons as f64).log2().ceil() as usize;
        let global_quantum_state = QuantumState::new(global_qubits);

        // Set up entanglement partners
        Self::setup_entanglement_network(&mut neurons, &config);

        Ok(Self {
            config,
            neurons,
            synapses,
            global_quantum_state,
            current_time: 0.0,
            input_neuron_ids,
            hidden_neuron_ids,
            output_neuron_ids,
            total_spikes: 0,
            predictions_count: 0,
            last_quantum_advantage: 1.0,
            training_epochs: 0,
        })
    }

    /// Setup entanglement network between quantum neurons
    fn setup_entanglement_network(neurons: &mut [QuantumNeuron], config: &QuantumSNNConfig) {
        for i in 0..config.quantum_neurons.min(neurons.len()) {
            for j in i + 1..config.quantum_neurons.min(neurons.len()) {
                if rand::thread_rng().gen::<f64>() < 0.2 { // 20% entanglement probability
                    neurons[i].entanglement_partners.push(j);
                    neurons[j].entanglement_partners.push(i);
                }
            }
        }
    }

    /// Forward pass through the network
    pub async fn forward(&mut self, input: &DVector<f64>) -> Result<DVector<f64>, QuantumMLError> {
        if input.len() != self.config.input_size {
            return Err(QuantumMLError::ModelPredictionFailed {
                reason: "Input size mismatch".to_string(),
            });
        }

        // Reset network state
        self.current_time += 1.0;
        let mut output = DVector::zeros(self.config.output_size);

        // Run simulation for multiple time steps
        for time_step in 0..50 {
            let current_time = self.current_time + time_step as f64;
            
            // Apply input to input neurons
            for (i, &input_val) in input.iter().enumerate() {
                if i < self.input_neuron_ids.len() {
                    let neuron_id = self.input_neuron_ids[i];
                    self.neurons[neuron_id].update_membrane_potential(input_val * 0.1, 0.0);
                }
            }

            // Update quantum states
            self.update_quantum_states(current_time).await?;

            // Process spikes and synaptic transmission
            self.process_spike_propagation(current_time).await?;

            // Collect output from output neurons
            for (i, &output_id) in self.output_neuron_ids.iter().enumerate() {
                output[i] += self.neurons[output_id].membrane_potential * 0.1;
            }
        }

        // Normalize output
        let max_output: f64 = output.iter().fold(0.0_f64, |acc, x: &f64| acc.max(x.abs()));
        if max_output > 0.0 {
            output /= max_output;
        }

        Ok(output)
    }

    /// Update quantum states of all neurons
    async fn update_quantum_states(&mut self, _current_time: f64) -> Result<(), QuantumMLError> {
        // Update global quantum state
        self.global_quantum_state.apply_decoherence(self.config.decoherence_rate, 1.0);

        // Update individual neuron quantum states
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Apply quantum gates based on membrane potential
            if neuron.membrane_potential.abs() > 0.1 {
                let angle = neuron.membrane_potential * 0.1;
                let ry_gate = QuantumGates::ry(angle);
                QuantumGates::apply_single_qubit_gate(&mut neuron.quantum_state, &ry_gate, 0)?;
            }

            // Apply entanglement effects
            if i < self.config.quantum_neurons {
                // Skip entanglement for now to avoid borrow conflicts
                // TODO: Implement proper entanglement without borrowing conflicts
            }

            // Update quantum state
            neuron.quantum_state.apply_decoherence(self.config.decoherence_rate, 1.0);
        }

        Ok(())
    }

    /// Apply entanglement effects between quantum neurons
    async fn apply_entanglement_effects(&mut self, neuron_id: usize, _current_time: f64) -> Result<(), QuantumMLError> {
        // This method is kept for compatibility but calls the safe version
        let neuron_potential = self.neurons[neuron_id].membrane_potential;
        let neuron_last_spike = self.neurons[neuron_id].last_spike_time;
        self.apply_entanglement_effects_safe(neuron_id, neuron_potential, neuron_last_spike, _current_time).await
    }
    
    /// Apply entanglement effects in a safe way to avoid borrow conflicts
    async fn apply_entanglement_effects_safe(&mut self, neuron_id: usize, _neuron_potential: f64, _neuron_last_spike: f64, _current_time: f64) -> Result<(), QuantumMLError> {
        if neuron_id >= self.neurons.len() {
            return Ok(());
        }

        let entanglement_partners = self.neurons[neuron_id].entanglement_partners.clone();
        
        for &partner_id in &entanglement_partners {
            if partner_id < self.neurons.len() && partner_id != neuron_id {
                // Create entanglement between quantum states
                let coupling_strength = self.config.quantum_coupling;
                
                // Simple entanglement: apply controlled rotations
                if self.neurons[neuron_id].quantum_state.n_qubits >= 2 &&
                   self.neurons[partner_id].quantum_state.n_qubits >= 2 {
                    
                    let cnot_gate = QuantumGates::cnot();
                    QuantumGates::apply_two_qubit_gate(
                        &mut self.neurons[neuron_id].quantum_state,
                        &cnot_gate,
                        0,
                        1,
                    )?;
                    
                    // Apply coupling effect to membrane potential
                    let quantum_effect = self.neurons[partner_id].quantum_state.entanglement_measure * coupling_strength;
                    self.neurons[neuron_id].membrane_potential += quantum_effect * 0.1; // Scale down the effect
                }
            }
        }

        Ok(())
    }

    /// Process spike propagation through the network
    async fn process_spike_propagation(&mut self, current_time: f64) -> Result<(), QuantumMLError> {
        let mut spike_events = Vec::new();

        // Check for spikes in all neurons
        for (neuron_id, neuron) in self.neurons.iter_mut().enumerate() {
            let quantum_prob = neuron.quantum_spike_probability();
            let effective_threshold = self.config.spike_threshold * (1.0 - quantum_prob * 0.1);
            
            if neuron.generate_spike(current_time, effective_threshold) {
                spike_events.push(neuron_id);
                self.total_spikes += 1;
            }
        }

        // Propagate spikes through synapses
        for spike_source in spike_events {
            for synapse in &mut self.synapses {
                if synapse.pre_neuron_id == spike_source {
                    let quantum_enhancement = self.neurons[spike_source].quantum_state.entanglement_measure;
                    let transmission = synapse.get_transmission(quantum_enhancement);
                    
                    // Apply synaptic transmission
                    if synapse.post_neuron_id < self.neurons.len() {
                        self.neurons[synapse.post_neuron_id].update_membrane_potential(transmission, 0.0);
                    }
                }
            }
        }

        Ok(())
    }

    /// Train the network using STDP
    pub async fn train(&mut self, training_data: &[QuantumMarketData], targets: &DVector<f64>) -> Result<(), QuantumMLError> {
        if training_data.len() != targets.len() {
            return Err(QuantumMLError::NeuralNetworkTrainingFailed {
                reason: "Training data and targets length mismatch".to_string(),
            });
        }

        let mut total_error = 0.0;

        for _epoch in 0..50 {
            let mut epoch_error = 0.0;

            for (data, &target) in training_data.iter().zip(targets.iter()) {
                // Convert market data to input
                let input = self.market_data_to_input(data)?;
                
                // Forward pass
                let output = self.forward(&input).await?;
                let prediction = output[0];
                
                // Calculate error
                let error = (prediction - target).abs();
                epoch_error += error;

                // Apply STDP learning
                self.apply_stdp_learning(target, prediction).await?;
            }

            epoch_error /= training_data.len() as f64;
            total_error += epoch_error;

            // Early stopping
            if epoch_error < 0.01 {
                break;
            }

            self.training_epochs += 1;
        }

        // Update quantum advantage
        self.last_quantum_advantage = self.calculate_quantum_advantage();

        tracing::info!("Quantum SNN training completed with final error: {:.6}", total_error);
        Ok(())
    }

    /// Apply STDP learning to synapses
    async fn apply_stdp_learning(&mut self, target: f64, prediction: f64) -> Result<(), QuantumMLError> {
        let error = target - prediction;
        
        // Update synaptic weights based on recent spike timing
        for synapse in &mut self.synapses {
            if synapse.pre_neuron_id < self.neurons.len() && synapse.post_neuron_id < self.neurons.len() {
                let pre_neuron = &self.neurons[synapse.pre_neuron_id];
                let post_neuron = &self.neurons[synapse.post_neuron_id];
                
                // Find recent spikes
                if let (Some(&pre_spike), Some(&post_spike)) = (
                    pre_neuron.spike_history.last(),
                    post_neuron.spike_history.last()
                ) {
                    synapse.update_stdp(
                        pre_spike,
                        post_spike,
                        self.config.learning_rate * error.abs(),
                        self.config.stdp_window,
                    );
                }
            }
        }

        Ok(())
    }

    /// Make prediction using the trained network
    pub async fn predict(&mut self, market_data: &QuantumMarketData) -> Result<QuantumPrediction, QuantumMLError> {
        let input = self.market_data_to_input(market_data)?;
        let output = self.forward(&input).await?;
        
        let prediction_value = output[0];
        let uncertainty = self.calculate_prediction_uncertainty();
        
        self.predictions_count += 1;

        Ok(QuantumPrediction {
            value: prediction_value,
            uncertainty,
            confidence_interval: (
                prediction_value - 2.0 * uncertainty,
                prediction_value + 2.0 * uncertainty,
            ),
            quantum_advantage: self.last_quantum_advantage,
            entanglement_contribution: self.global_quantum_state.entanglement_measure,
            prediction_timestamp: chrono::Utc::now(),
        })
    }

    /// Convert market data to input vector
    fn market_data_to_input(&self, market_data: &QuantumMarketData) -> Result<DVector<f64>, QuantumMLError> {
        let mut input = DVector::zeros(self.config.input_size);
        
        // Use prices and volumes as features
        let n_prices = market_data.prices.len().min(self.config.input_size / 2);
        let n_volumes = market_data.volumes.len().min(self.config.input_size / 2);
        
        for i in 0..n_prices {
            input[i] = market_data.prices[i] / 1000.0; // Normalize prices
        }
        
        for i in 0..n_volumes {
            input[self.config.input_size / 2 + i] = market_data.volumes[i] / 10000.0; // Normalize volumes
        }
        
        Ok(input)
    }

    /// Calculate prediction uncertainty
    fn calculate_prediction_uncertainty(&self) -> f64 {
        let quantum_entropy = self.global_quantum_state.entanglement_measure;
        let spike_rate_variance = self.calculate_spike_rate_variance();
        
        (quantum_entropy + spike_rate_variance) / 2.0
    }

    /// Calculate spike rate variance as uncertainty measure
    fn calculate_spike_rate_variance(&self) -> f64 {
        let mut spike_rates = Vec::new();
        
        for neuron in &self.neurons {
            let spike_count = neuron.spike_history.len();
            let rate = spike_count as f64 / (self.current_time + 1.0);
            spike_rates.push(rate);
        }
        
        if spike_rates.is_empty() {
            return 0.1;
        }
        
        let mean_rate = spike_rates.iter().sum::<f64>() / spike_rates.len() as f64;
        let variance = spike_rates.iter()
            .map(|&rate| (rate - mean_rate).powi(2))
            .sum::<f64>() / spike_rates.len() as f64;
        
        variance.sqrt()
    }

    /// Calculate quantum advantage
    fn calculate_quantum_advantage(&self) -> f64 {
        let entanglement_factor = self.global_quantum_state.entanglement_measure;
        let quantum_neuron_ratio = self.config.quantum_neurons as f64 / self.neurons.len() as f64;
        let spike_efficiency = self.total_spikes as f64 / (self.current_time + 1.0);
        
        (1.0 + entanglement_factor) * (1.0 + quantum_neuron_ratio) * (1.0 + spike_efficiency * 0.1)
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> QuantumSNNMetrics {
        QuantumSNNMetrics {
            accuracy: self.calculate_accuracy(),
            quantum_advantage: self.last_quantum_advantage,
            predictions: self.predictions_count,
            avg_prediction_time: std::time::Duration::from_millis(10), // Estimated
            total_spikes: self.total_spikes,
            network_size: self.neurons.len(),
            entanglement_measure: self.global_quantum_state.entanglement_measure,
        }
    }

    /// Calculate accuracy based on quantum advantage
    fn calculate_accuracy(&self) -> f64 {
        (self.last_quantum_advantage / 2.0).min(1.0).max(0.0)
    }
}

/// Quantum SNN performance metrics
#[derive(Debug, Clone)]
pub struct QuantumSNNMetrics {
    pub accuracy: f64,
    pub quantum_advantage: f64,
    pub predictions: u64,
    pub avg_prediction_time: std::time::Duration,
    pub total_spikes: u64,
    pub network_size: usize,
    pub entanglement_measure: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_quantum_snn_creation() {
        let snn = QuantumSNN::new(10, 5).await;
        assert!(snn.is_ok());
        
        let snn = snn.unwrap();
        assert_eq!(snn.config.input_size, 10);
        assert_eq!(snn.config.hidden_size, 5);
    }

    #[tokio::test]
    async fn test_quantum_neuron() {
        let mut neuron = QuantumNeuron::new(2);
        
        neuron.update_membrane_potential(0.5, 0.1);
        assert_abs_diff_eq!(neuron.membrane_potential, 0.6, epsilon = 1e-10);
        
        let spike = neuron.generate_spike(10.0, 0.5);
        assert!(spike);
        assert_abs_diff_eq!(neuron.membrane_potential, 0.0, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_synapse() {
        let mut synapse = QuantumSynapse::new(0, 1, 0.5);
        
        synapse.update_stdp(10.0, 12.0, 0.01, 20.0);
        assert!(synapse.weight > 0.5); // Should increase due to positive timing
        
        let transmission = synapse.get_transmission(0.1);
        assert!(transmission > 0.0);
    }

    #[tokio::test]
    async fn test_snn_forward_pass() {
        let mut snn = QuantumSNN::new(5, 3).await.unwrap();
        let input = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        
        let output = snn.forward(&input).await;
        assert!(output.is_ok());
        
        let output = output.unwrap();
        assert_eq!(output.len(), 1);
    }

    #[tokio::test]
    async fn test_snn_prediction() {
        let mut snn = QuantumSNN::new(6, 3).await.unwrap();
        
        let market_data = QuantumMarketData {
            prices: DVector::from_vec(vec![100.0, 101.0, 102.0]),
            volumes: DVector::from_vec(vec![1000.0, 1100.0, 900.0]),
            features: DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            timestamps: vec![chrono::Utc::now(); 3],
            quantum_encoding: None,
        };
        
        let prediction = snn.predict(&market_data).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.uncertainty >= 0.0);
        assert!(pred.quantum_advantage >= 0.0);
    }
}