//! Quantum Spiking Neural Networks (QSNN) with STDP Plasticity
//!
//! Implements quantum-enhanced spiking neural networks with quantum-enhanced
//! spike-timing-dependent plasticity for high-frequency trading predictions.

use crate::TENGRIError;
use super::quantum_gates::{QuantumState, QuantumCircuit, QuantumGateOp};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, VecDeque};
use rayon::prelude::*;
use chrono::{DateTime, Utc};

/// Quantum spiking neuron
#[derive(Debug, Clone)]
pub struct QuantumSpikingNeuron {
    pub id: usize,
    pub membrane_potential: f64,
    pub threshold: f64,
    pub reset_potential: f64,
    pub refractory_period: f64,
    pub refractory_timer: f64,
    pub leak_rate: f64,
    
    // Quantum components
    pub quantum_state: QuantumState,
    pub quantum_phase: f64,
    pub coherence_time: f64,
    pub quantum_interference: f64,
    
    // Spike history
    pub spike_history: VecDeque<f64>,
    pub last_spike_time: f64,
    
    // Adaptation parameters
    pub adaptation_current: f64,
    pub adaptation_conductance: f64,
}

impl QuantumSpikingNeuron {
    /// Create new quantum spiking neuron
    pub fn new(id: usize, n_qubits: usize) -> Result<Self, TENGRIError> {
        Ok(Self {
            id,
            membrane_potential: -70.0, // Resting potential in mV
            threshold: -55.0,          // Spike threshold in mV
            reset_potential: -75.0,    // Reset potential in mV
            refractory_period: 2.0,    // Refractory period in ms
            refractory_timer: 0.0,
            leak_rate: 0.1,
            
            quantum_state: QuantumState::new(n_qubits),
            quantum_phase: 0.0,
            coherence_time: 100.0,     // Quantum coherence time in ms
            quantum_interference: 0.0,
            
            spike_history: VecDeque::with_capacity(1000),
            last_spike_time: 0.0,
            
            adaptation_current: 0.0,
            adaptation_conductance: 0.02,
        })
    }

    /// Update neuron state with quantum dynamics
    pub fn update(&mut self, input_current: f64, dt: f64, time: f64) -> Result<bool, TENGRIError> {
        // Update quantum state
        self.update_quantum_state(input_current, dt)?;
        
        // Check if in refractory period
        if self.refractory_timer > 0.0 {
            self.refractory_timer -= dt;
            return Ok(false);
        }
        
        // Quantum-enhanced membrane dynamics
        let quantum_influence = self.compute_quantum_influence()?;
        
        // Leaky integrate-and-fire with quantum enhancement
        let leak_current = self.leak_rate * (self.membrane_potential - self.reset_potential);
        let adaptation_current = self.adaptation_current * self.adaptation_conductance;
        
        let membrane_derivative = -leak_current + input_current - adaptation_current + quantum_influence;
        self.membrane_potential += membrane_derivative * dt;
        
        // Update adaptation current
        self.adaptation_current *= (1.0 - dt / 20.0); // Exponential decay
        
        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.spike(time)?;
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Update quantum state based on input
    fn update_quantum_state(&mut self, input_current: f64, dt: f64) -> Result<(), TENGRIError> {
        // Encode input into quantum rotation
        let n_qubits = self.quantum_state.n_qubits;
        let mut quantum_circuit = QuantumCircuit::new(n_qubits);
        
        // Phase encoding based on input current
        let phase_angle = input_current * dt * 0.01; // Scale factor
        self.quantum_phase += phase_angle;
        
        // Apply rotation gates to encode input
        for i in 0..n_qubits {
            let qubit_angle = self.quantum_phase * (i as f64 + 1.0) / (n_qubits as f64);
            quantum_circuit.add_gate(QuantumGateOp::RZ(i, qubit_angle));
        }
        
        // Add entanglement for quantum correlations
        for i in 0..n_qubits-1 {
            if (input_current * (i as f64 + 1.0)).abs() > 0.1 {
                quantum_circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % n_qubits));
            }
        }
        
        // Execute quantum circuit
        quantum_circuit.execute(&mut self.quantum_state)?;
        
        // Apply decoherence
        self.apply_quantum_decoherence(dt)?;
        
        Ok(())
    }

    /// Compute quantum influence on membrane potential
    fn compute_quantum_influence(&self) -> Result<f64, TENGRIError> {
        let n_qubits = self.quantum_state.n_qubits;
        let mut quantum_expectation = 0.0;
        
        // Compute quantum expectation value
        let dim = 1usize << n_qubits;
        for i in 0..dim {
            let amplitude = self.quantum_state.amplitudes[i];
            let probability = amplitude.norm_sqr();
            let phase = amplitude.arg();
            
            // Quantum interference contribution
            quantum_expectation += probability * phase.cos();
        }
        
        // Scale quantum influence
        let quantum_influence = quantum_expectation * 10.0 * self.quantum_state.fidelity;
        
        Ok(quantum_influence)
    }

    /// Apply quantum decoherence
    fn apply_quantum_decoherence(&mut self, dt: f64) -> Result<(), TENGRIError> {
        let decoherence_rate = dt / self.coherence_time;
        let decoherence_factor = (-decoherence_rate).exp();
        
        // Apply decoherence to quantum amplitudes
        for amplitude in &mut self.quantum_state.amplitudes {
            *amplitude *= decoherence_factor;
        }
        
        // Renormalize quantum state
        let norm_squared: f64 = self.quantum_state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        if norm_squared > 0.0 {
            let norm = norm_squared.sqrt();
            for amplitude in &mut self.quantum_state.amplitudes {
                *amplitude /= norm;
            }
        }
        
        self.quantum_state.fidelity = self.quantum_state.calculate_fidelity();
        
        Ok(())
    }

    /// Process spike event
    fn spike(&mut self, time: f64) -> Result<(), TENGRIError> {
        // Reset membrane potential
        self.membrane_potential = self.reset_potential;
        
        // Set refractory period
        self.refractory_timer = self.refractory_period;
        
        // Add spike to history
        self.spike_history.push_back(time);
        if self.spike_history.len() > 1000 {
            self.spike_history.pop_front();
        }
        
        // Update last spike time
        self.last_spike_time = time;
        
        // Increase adaptation current
        self.adaptation_current += 10.0;
        
        // Quantum state collapse simulation
        self.quantum_state.measure()?;
        
        Ok(())
    }

    /// Get spike rate over time window
    pub fn get_spike_rate(&self, window_ms: f64, current_time: f64) -> f64 {
        let start_time = current_time - window_ms;
        let spike_count = self.spike_history.iter()
            .filter(|&&spike_time| spike_time >= start_time)
            .count();
        
        spike_count as f64 / (window_ms / 1000.0) // Convert to Hz
    }
}

/// Quantum STDP (Spike-Timing-Dependent Plasticity) synapse
#[derive(Debug, Clone)]
pub struct QuantumSTDPSynapse {
    pub pre_neuron_id: usize,
    pub post_neuron_id: usize,
    pub weight: f64,
    pub max_weight: f64,
    pub min_weight: f64,
    
    // Classical STDP parameters
    pub tau_plus: f64,      // LTP time constant
    pub tau_minus: f64,     // LTD time constant
    pub a_plus: f64,        // LTP amplitude
    pub a_minus: f64,       // LTD amplitude
    
    // Quantum STDP parameters
    pub quantum_correlation: f64,
    pub quantum_coherence: f64,
    pub entanglement_strength: f64,
    
    // Spike timing traces
    pub pre_trace: f64,
    pub post_trace: f64,
}

impl QuantumSTDPSynapse {
    /// Create new quantum STDP synapse
    pub fn new(pre_neuron_id: usize, post_neuron_id: usize, initial_weight: f64) -> Self {
        Self {
            pre_neuron_id,
            post_neuron_id,
            weight: initial_weight,
            max_weight: 5.0,
            min_weight: -5.0,
            
            tau_plus: 20.0,   // ms
            tau_minus: 20.0,  // ms
            a_plus: 0.01,
            a_minus: 0.005,
            
            quantum_correlation: 0.0,
            quantum_coherence: 1.0,
            entanglement_strength: 0.1,
            
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }

    /// Update synapse with quantum STDP
    pub fn update_quantum_stdp(
        &mut self,
        pre_spike: bool,
        post_spike: bool,
        pre_neuron: &QuantumSpikingNeuron,
        post_neuron: &QuantumSpikingNeuron,
        dt: f64,
    ) -> Result<(), TENGRIError> {
        // Update spike traces
        self.pre_trace *= (-dt / self.tau_plus).exp();
        self.post_trace *= (-dt / self.tau_minus).exp();
        
        // Compute quantum correlation
        self.quantum_correlation = self.compute_quantum_correlation(pre_neuron, post_neuron)?;
        
        // Classical STDP updates
        if pre_spike {
            self.pre_trace += 1.0;
            // Depression: post-before-pre
            let depression = self.a_minus * self.post_trace;
            self.weight -= depression;
        }
        
        if post_spike {
            self.post_trace += 1.0;
            // Potentiation: pre-before-post
            let potentiation = self.a_plus * self.pre_trace;
            self.weight += potentiation;
        }
        
        // Quantum enhancement
        let quantum_enhancement = self.quantum_correlation * self.entanglement_strength;
        if pre_spike && post_spike {
            // Quantum coherent plasticity
            self.weight += quantum_enhancement * self.quantum_coherence;
        }
        
        // Bound weights
        self.weight = self.weight.max(self.min_weight).min(self.max_weight);
        
        Ok(())
    }

    /// Compute quantum correlation between neurons
    fn compute_quantum_correlation(
        &self,
        pre_neuron: &QuantumSpikingNeuron,
        post_neuron: &QuantumSpikingNeuron,
    ) -> Result<f64, TENGRIError> {
        let pre_qubits = pre_neuron.quantum_state.n_qubits;
        let post_qubits = post_neuron.quantum_state.n_qubits;
        
        // Compute quantum fidelity-based correlation
        let pre_fidelity = pre_neuron.quantum_state.fidelity;
        let post_fidelity = post_neuron.quantum_state.fidelity;
        
        // Phase correlation
        let phase_correlation = (pre_neuron.quantum_phase - post_neuron.quantum_phase).cos();
        
        // Combined quantum correlation
        let quantum_correlation = (pre_fidelity * post_fidelity).sqrt() * phase_correlation;
        
        Ok(quantum_correlation)
    }
}

/// Quantum Spiking Neural Network
pub struct QuantumSNN {
    pub neurons: Vec<QuantumSpikingNeuron>,
    pub synapses: Vec<QuantumSTDPSynapse>,
    pub input_neurons: Vec<usize>,
    pub output_neurons: Vec<usize>,
    pub connectivity_matrix: DMatrix<f64>,
    
    // Network parameters
    pub n_qubits: usize,
    pub time_step: f64,
    pub current_time: f64,
    pub batch_size: usize,
    
    // Performance metrics
    pub spike_coherence: f64,
    pub quantum_advantage: f64,
    pub network_activity: f64,
}

impl QuantumSNN {
    /// Create new quantum SNN
    pub async fn new(n_qubits: usize, batch_size: usize) -> Result<Self, TENGRIError> {
        let n_neurons = 100;
        let n_input = 10;
        let n_output = 1;
        
        // Create neurons
        let mut neurons = Vec::new();
        for i in 0..n_neurons {
            let neuron = QuantumSpikingNeuron::new(i, n_qubits)?;
            neurons.push(neuron);
        }
        
        // Define input and output neurons
        let input_neurons: Vec<usize> = (0..n_input).collect();
        let output_neurons: Vec<usize> = (n_neurons - n_output..n_neurons).collect();
        
        // Create connectivity matrix
        let connectivity_matrix = DMatrix::from_fn(n_neurons, n_neurons, |i, j| {
            if i != j {
                (rand::random::<f64>() - 0.5) * 2.0 // Random weights [-1, 1]
            } else {
                0.0 // No self-connections
            }
        });
        
        // Create synapses
        let mut synapses = Vec::new();
        for i in 0..n_neurons {
            for j in 0..n_neurons {
                if i != j && connectivity_matrix[(i, j)].abs() > 0.1 {
                    let synapse = QuantumSTDPSynapse::new(i, j, connectivity_matrix[(i, j)]);
                    synapses.push(synapse);
                }
            }
        }
        
        Ok(Self {
            neurons,
            synapses,
            input_neurons,
            output_neurons,
            connectivity_matrix,
            n_qubits,
            time_step: 0.1, // ms
            current_time: 0.0,
            batch_size,
            spike_coherence: 0.0,
            quantum_advantage: 0.0,
            network_activity: 0.0,
        })
    }

    /// Predict using quantum SNN
    pub async fn predict(&mut self, input: &DMatrix<f64>) -> Result<f64, TENGRIError> {
        let start_time = std::time::Instant::now();
        
        let sequence_length = input.ncols();
        let mut output_spikes = Vec::new();
        
        // Reset network state
        self.reset_network()?;
        
        // Process input sequence
        for t in 0..sequence_length {
            let input_vector = input.column(t);
            
            // Set input currents
            for (&neuron_id, &current) in self.input_neurons.iter().zip(input_vector.iter()) {
                // Scale input current
                let scaled_current = current * 100.0; // Scale to physiological range
                
                // Update neuron with input
                self.neurons[neuron_id].update(scaled_current, self.time_step, self.current_time)?;
            }
            
            // Update all neurons
            let mut spike_pattern = Vec::new();
            for neuron in &mut self.neurons {
                let spiked = neuron.update(0.0, self.time_step, self.current_time)?;
                spike_pattern.push(spiked);
            }
            
            // Update synapses with quantum STDP
            self.update_synapses(&spike_pattern)?;
            
            // Collect output spikes
            for &output_id in &self.output_neurons {
                if spike_pattern[output_id] {
                    output_spikes.push(self.current_time);
                }
            }
            
            self.current_time += self.time_step;
        }
        
        // Compute output as spike rate
        let spike_rate = output_spikes.len() as f64 / (sequence_length as f64 * self.time_step / 1000.0);
        
        // Normalize to [0, 1] range
        let prediction = (spike_rate / 100.0).min(1.0).max(0.0);
        
        // Update network metrics
        self.update_network_metrics()?;
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 100 {
            tracing::warn!(
                "Quantum SNN prediction time: {}μs (target: <100μs)",
                elapsed.as_micros()
            );
        }
        
        Ok(prediction)
    }

    /// Reset network state
    fn reset_network(&mut self) -> Result<(), TENGRIError> {
        for neuron in &mut self.neurons {
            neuron.membrane_potential = neuron.reset_potential;
            neuron.refractory_timer = 0.0;
            neuron.adaptation_current = 0.0;
            neuron.quantum_phase = 0.0;
            neuron.quantum_state = QuantumState::new(self.n_qubits);
        }
        
        self.current_time = 0.0;
        Ok(())
    }

    /// Update synapses with quantum STDP
    fn update_synapses(&mut self, spike_pattern: &[bool]) -> Result<(), TENGRIError> {
        for synapse in &mut self.synapses {
            let pre_spike = spike_pattern[synapse.pre_neuron_id];
            let post_spike = spike_pattern[synapse.post_neuron_id];
            
            let pre_neuron = &self.neurons[synapse.pre_neuron_id];
            let post_neuron = &self.neurons[synapse.post_neuron_id];
            
            synapse.update_quantum_stdp(
                pre_spike,
                post_spike,
                pre_neuron,
                post_neuron,
                self.time_step,
            )?;
            
            // Update connectivity matrix
            self.connectivity_matrix[(synapse.pre_neuron_id, synapse.post_neuron_id)] = synapse.weight;
        }
        
        Ok(())
    }

    /// Update network metrics
    fn update_network_metrics(&mut self) -> Result<(), TENGRIError> {
        // Calculate spike coherence
        let mut coherence_sum = 0.0;
        let mut active_neurons = 0;
        
        for neuron in &self.neurons {
            if neuron.spike_history.len() > 0 {
                coherence_sum += neuron.quantum_state.fidelity;
                active_neurons += 1;
            }
        }
        
        self.spike_coherence = if active_neurons > 0 {
            coherence_sum / active_neurons as f64
        } else {
            0.0
        };
        
        // Calculate quantum advantage
        let quantum_correlations: f64 = self.synapses.iter()
            .map(|synapse| synapse.quantum_correlation)
            .sum();
        
        self.quantum_advantage = quantum_correlations / self.synapses.len() as f64;
        
        // Calculate network activity
        let total_spikes: usize = self.neurons.iter()
            .map(|neuron| neuron.spike_history.len())
            .sum();
        
        self.network_activity = total_spikes as f64 / self.neurons.len() as f64;
        
        Ok(())
    }

    /// Update network with training data
    pub async fn update(&mut self, training_data: &DMatrix<f64>, targets: &DVector<f64>) -> Result<(), TENGRIError> {
        let batch_size = training_data.ncols();
        let mut total_error = 0.0;
        
        for batch_idx in 0..batch_size {
            let input_sequence = training_data.column(batch_idx);
            let target = targets[batch_idx];
            
            // Create input matrix for prediction
            let mut input_matrix = DMatrix::zeros(input_sequence.len(), 1);
            input_matrix.set_column(0, &input_sequence);
            
            // Get prediction
            let prediction = self.predict(&input_matrix).await?;
            
            // Calculate error
            let error = (prediction - target).powi(2);
            total_error += error;
            
            // Update synaptic weights based on error
            let learning_rate = 0.001;
            let weight_adjustment = learning_rate * (target - prediction);
            
            for synapse in &mut self.synapses {
                synapse.weight += weight_adjustment * 0.1;
                synapse.weight = synapse.weight.max(synapse.min_weight).min(synapse.max_weight);
            }
        }
        
        let avg_error = total_error / batch_size as f64;
        tracing::info!("Quantum SNN training error: {:.6}", avg_error);
        
        Ok(())
    }

    /// Get quantum SNN metrics
    pub async fn get_metrics(&self) -> Result<QuantumSNNMetrics, TENGRIError> {
        Ok(QuantumSNNMetrics {
            spike_coherence: self.spike_coherence,
            quantum_advantage: self.quantum_advantage,
            network_activity: self.network_activity,
            average_quantum_fidelity: self.neurons.iter()
                .map(|n| n.quantum_state.fidelity)
                .sum::<f64>() / self.neurons.len() as f64,
        })
    }
}

/// Quantum SNN metrics
#[derive(Debug, Clone)]
pub struct QuantumSNNMetrics {
    pub spike_coherence: f64,
    pub quantum_advantage: f64,
    pub network_activity: f64,
    pub average_quantum_fidelity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantum_spiking_neuron_creation() {
        let neuron = QuantumSpikingNeuron::new(0, 4).unwrap();
        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.quantum_state.n_qubits, 4);
        assert_abs_diff_eq!(neuron.membrane_potential, -70.0);
    }

    #[test]
    fn test_quantum_spiking_neuron_update() {
        let mut neuron = QuantumSpikingNeuron::new(0, 4).unwrap();
        let input_current = 50.0; // Strong input
        let dt = 0.1;
        let time = 0.0;
        
        // Should not spike immediately
        let spiked = neuron.update(input_current, dt, time).unwrap();
        assert!(neuron.membrane_potential > -70.0); // Should depolarize
        
        // Apply multiple updates to potentially reach threshold
        for i in 1..200 {
            let spiked = neuron.update(input_current, dt, i as f64 * dt).unwrap();
            if spiked {
                assert!(neuron.spike_history.len() > 0);
                break;
            }
        }
    }

    #[test]
    fn test_quantum_stdp_synapse_creation() {
        let synapse = QuantumSTDPSynapse::new(0, 1, 0.5);
        assert_eq!(synapse.pre_neuron_id, 0);
        assert_eq!(synapse.post_neuron_id, 1);
        assert_abs_diff_eq!(synapse.weight, 0.5);
    }

    #[tokio::test]
    async fn test_quantum_snn_creation() {
        let qsnn = QuantumSNN::new(4, 32).await.unwrap();
        assert_eq!(qsnn.neurons.len(), 100);
        assert_eq!(qsnn.input_neurons.len(), 10);
        assert_eq!(qsnn.output_neurons.len(), 1);
        assert!(qsnn.synapses.len() > 0);
    }

    #[tokio::test]
    async fn test_quantum_snn_prediction() {
        let mut qsnn = QuantumSNN::new(4, 32).await.unwrap();
        
        // Create sample input
        let input = DMatrix::from_fn(10, 5, |i, j| {
            (i as f64 + j as f64) * 0.1
        });
        
        let prediction = qsnn.predict(&input).await.unwrap();
        assert!(prediction >= 0.0 && prediction <= 1.0);
    }

    #[tokio::test]
    async fn test_quantum_snn_metrics() {
        let qsnn = QuantumSNN::new(4, 32).await.unwrap();
        let metrics = qsnn.get_metrics().await.unwrap();
        
        assert!(metrics.spike_coherence >= 0.0);
        assert!(metrics.quantum_advantage >= 0.0);
        assert!(metrics.network_activity >= 0.0);
        assert!(metrics.average_quantum_fidelity >= 0.0 && metrics.average_quantum_fidelity <= 1.0);
    }
}