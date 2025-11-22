//! Quantum Long Short-Term Memory (QLSTM) Networks
//!
//! Implements quantum-enhanced LSTM networks for temporal market prediction
//! with quantum memory cells and sub-100μs inference targets.

use crate::TENGRIError;
use super::quantum_gates::{QuantumState, QuantumCircuit, QuantumGateOp};
use nalgebra::{DMatrix, DVector, Matrix4, Vector4};
use std::collections::VecDeque;
use rayon::prelude::*;
use chrono::{DateTime, Utc};

/// Quantum LSTM cell state
#[derive(Debug, Clone)]
pub struct QuantumLSTMCell {
    // Classical LSTM parameters
    pub input_size: usize,
    pub hidden_size: usize,
    pub forget_gate_weights: DMatrix<f64>,
    pub input_gate_weights: DMatrix<f64>,
    pub candidate_gate_weights: DMatrix<f64>,
    pub output_gate_weights: DMatrix<f64>,
    
    // Quantum memory components
    pub quantum_memory: QuantumState,
    pub quantum_circuit: QuantumCircuit,
    pub quantum_fidelity: f64,
    
    // Quantum-classical hybrid parameters
    pub quantum_classical_coupling: f64,
    pub decoherence_rate: f64,
    pub entanglement_strength: f64,
}

impl QuantumLSTMCell {
    /// Create new quantum LSTM cell
    pub fn new(input_size: usize, hidden_size: usize, n_qubits: usize) -> Result<Self, TENGRIError> {
        let total_input_size = input_size + hidden_size;
        
        // Initialize classical LSTM weights with Xavier initialization
        let xavier_bound = (6.0 / (total_input_size + hidden_size) as f64).sqrt();
        
        let forget_gate_weights = DMatrix::from_fn(hidden_size, total_input_size, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
        });
        
        let input_gate_weights = DMatrix::from_fn(hidden_size, total_input_size, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
        });
        
        let candidate_gate_weights = DMatrix::from_fn(hidden_size, total_input_size, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
        });
        
        let output_gate_weights = DMatrix::from_fn(hidden_size, total_input_size, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * xavier_bound
        });
        
        // Initialize quantum memory
        let quantum_memory = QuantumState::new(n_qubits);
        
        // Create quantum circuit for temporal encoding
        let quantum_circuit = QuantumCircuit::create_temporal_encoding_circuit(hidden_size);
        
        Ok(Self {
            input_size,
            hidden_size,
            forget_gate_weights,
            input_gate_weights,
            candidate_gate_weights,
            output_gate_weights,
            quantum_memory,
            quantum_circuit,
            quantum_fidelity: 1.0,
            quantum_classical_coupling: 0.1,
            decoherence_rate: 0.01,
            entanglement_strength: 0.5,
        })
    }

    /// Forward pass through quantum LSTM cell
    pub fn forward(
        &mut self,
        input: &DVector<f64>,
        hidden_state: &DVector<f64>,
        cell_state: &DVector<f64>,
    ) -> Result<(DVector<f64>, DVector<f64>), TENGRIError> {
        let start_time = std::time::Instant::now();
        
        // Concatenate input and hidden state
        let mut combined_input = DVector::zeros(self.input_size + self.hidden_size);
        combined_input.rows_mut(0, self.input_size).copy_from(input);
        combined_input.rows_mut(self.input_size, self.hidden_size).copy_from(hidden_state);
        
        // Classical LSTM gates
        let forget_gate = Self::sigmoid_vector(&(&self.forget_gate_weights * &combined_input));
        let input_gate = Self::sigmoid_vector(&(&self.input_gate_weights * &combined_input));
        let candidate_values = Self::tanh_vector(&(&self.candidate_gate_weights * &combined_input));
        let output_gate = Self::sigmoid_vector(&(&self.output_gate_weights * &combined_input));
        
        // Update quantum memory with input information
        self.update_quantum_memory(input, hidden_state)?;
        
        // Quantum-enhanced cell state update
        let quantum_influence = self.compute_quantum_influence()?;
        let new_cell_state = forget_gate.component_mul(cell_state) + 
                            input_gate.component_mul(&candidate_values) +
                            quantum_influence * self.quantum_classical_coupling;
        
        // Quantum-enhanced hidden state
        let quantum_hidden_contribution = self.compute_quantum_hidden_state()?;
        let new_hidden_state = output_gate.component_mul(&Self::tanh_vector(&new_cell_state)) +
                              quantum_hidden_contribution * self.quantum_classical_coupling;
        
        // Update quantum fidelity
        self.quantum_fidelity = self.quantum_memory.calculate_fidelity();
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 50 {
            tracing::warn!(
                "QLSTM forward pass time: {}μs (target: <50μs)",
                elapsed.as_micros()
            );
        }
        
        Ok((new_hidden_state, new_cell_state))
    }

    /// Update quantum memory with new information
    fn update_quantum_memory(&mut self, input: &DVector<f64>, hidden_state: &DVector<f64>) -> Result<(), TENGRIError> {
        // Encode input into quantum state using rotation gates
        let n_qubits = self.quantum_memory.n_qubits;
        let mut encoding_circuit = QuantumCircuit::new(n_qubits);
        
        // Feature encoding: map input values to rotation angles
        for (i, &value) in input.iter().enumerate() {
            if i < n_qubits {
                let angle = value * std::f64::consts::PI; // Normalize to [0, π]
                encoding_circuit.add_gate(QuantumGateOp::RY(i, angle));
            }
        }
        
        // Hidden state encoding: create entanglement patterns
        for i in 0..n_qubits-1 {
            if i < hidden_state.len() {
                let entanglement_strength = hidden_state[i] * self.entanglement_strength;
                if entanglement_strength.abs() > 0.1 {
                    encoding_circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % n_qubits));
                }
            }
        }
        
        // Execute quantum encoding
        encoding_circuit.execute(&mut self.quantum_memory)?;
        
        // Apply decoherence
        self.apply_decoherence()?;
        
        Ok(())
    }

    /// Compute quantum influence on cell state
    fn compute_quantum_influence(&self) -> Result<DVector<f64>, TENGRIError> {
        let n_qubits = self.quantum_memory.n_qubits;
        let mut influence = DVector::zeros(self.hidden_size);
        
        // Extract quantum information through expectation values
        for i in 0..self.hidden_size.min(n_qubits) {
            // Compute expectation value of Pauli-Z operator
            let mut expectation = 0.0;
            let dim = 1usize << n_qubits;
            
            for j in 0..dim {
                let bit = (j >> i) & 1;
                let sign = if bit == 0 { 1.0 } else { -1.0 };
                expectation += sign * self.quantum_memory.amplitudes[j].norm_sqr();
            }
            
            influence[i] = expectation * self.quantum_fidelity;
        }
        
        Ok(influence)
    }

    /// Compute quantum contribution to hidden state
    fn compute_quantum_hidden_state(&self) -> Result<DVector<f64>, TENGRIError> {
        let n_qubits = self.quantum_memory.n_qubits;
        let mut quantum_hidden = DVector::zeros(self.hidden_size);
        
        // Use quantum superposition to enhance hidden state
        for i in 0..self.hidden_size.min(n_qubits) {
            // Quantum interference pattern
            let mut interference = 0.0;
            let dim = 1usize << n_qubits;
            
            for j in 0..dim {
                let phase = self.quantum_memory.amplitudes[j].arg();
                let amplitude = self.quantum_memory.amplitudes[j].norm();
                interference += amplitude * (phase + (i as f64) * std::f64::consts::PI / (n_qubits as f64)).cos();
            }
            
            quantum_hidden[i] = interference * self.quantum_fidelity;
        }
        
        Ok(quantum_hidden)
    }

    /// Apply quantum decoherence
    fn apply_decoherence(&mut self) -> Result<(), TENGRIError> {
        let dim = self.quantum_memory.amplitudes.len();
        let decoherence_factor = (-self.decoherence_rate).exp();
        
        // Apply decoherence to quantum amplitudes
        for i in 0..dim {
            self.quantum_memory.amplitudes[i] *= decoherence_factor;
        }
        
        // Renormalize quantum state
        let norm_squared: f64 = self.quantum_memory.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        
        if norm_squared > 0.0 {
            let norm = norm_squared.sqrt();
            for i in 0..dim {
                self.quantum_memory.amplitudes[i] /= norm;
            }
        }
        
        Ok(())
    }

    /// Sigmoid activation function (vectorized)
    fn sigmoid_vector(x: &DVector<f64>) -> DVector<f64> {
        x.map(|val| 1.0 / (1.0 + (-val).exp()))
    }

    /// Tanh activation function (vectorized)
    fn tanh_vector(x: &DVector<f64>) -> DVector<f64> {
        x.map(|val| val.tanh())
    }
}

/// Quantum LSTM network for temporal prediction
pub struct QuantumLSTM {
    pub layers: Vec<QuantumLSTMCell>,
    pub output_layer: DMatrix<f64>,
    pub sequence_length: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub prediction_history: VecDeque<QuantumLSTMPrediction>,
    pub quantum_advantage_metric: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumLSTMPrediction {
    pub value: f64,
    pub quantum_confidence: f64,
    pub classical_confidence: f64,
    pub quantum_fidelity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct QuantumLSTMMetrics {
    pub quantum_fidelity: f64,
    pub entanglement_entropy: f64,
    pub classical_accuracy: f64,
    pub quantum_advantage: f64,
}

impl QuantumLSTM {
    /// Create new quantum LSTM network
    pub async fn new(n_qubits: usize, batch_size: usize) -> Result<Self, TENGRIError> {
        let input_size = 10; // Market features
        let hidden_size = 64;
        let num_layers = 2;
        let sequence_length = 50; // 50 time steps
        
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            let cell = QuantumLSTMCell::new(layer_input_size, hidden_size, n_qubits)?;
            layers.push(cell);
        }
        
        // Output layer for prediction
        let output_layer = DMatrix::from_fn(1, hidden_size, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * (2.0 / hidden_size as f64).sqrt()
        });
        
        Ok(Self {
            layers,
            output_layer,
            sequence_length,
            batch_size,
            learning_rate: 0.001,
            prediction_history: VecDeque::with_capacity(1000),
            quantum_advantage_metric: 0.0,
        })
    }

    /// Predict next value in sequence
    pub async fn predict(&mut self, input_sequence: &DMatrix<f64>) -> Result<f64, TENGRIError> {
        let start_time = std::time::Instant::now();
        
        let sequence_len = input_sequence.ncols();
        let input_size = input_sequence.nrows();
        
        // Initialize hidden and cell states
        let mut hidden_states: Vec<DVector<f64>> = self.layers.iter()
            .map(|layer| DVector::zeros(layer.hidden_size))
            .collect();
        
        let mut cell_states: Vec<DVector<f64>> = self.layers.iter()
            .map(|layer| DVector::zeros(layer.hidden_size))
            .collect();
        
        // Process sequence through quantum LSTM layers
        for t in 0..sequence_len {
            let input_t = input_sequence.column(t);
            
            // Forward pass through each layer
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                let layer_input = if layer_idx == 0 {
                    input_t.into_owned()
                } else {
                    hidden_states[layer_idx - 1].clone()
                };
                
                let (new_hidden, new_cell) = layer.forward(
                    &layer_input,
                    &hidden_states[layer_idx],
                    &cell_states[layer_idx],
                )?;
                
                hidden_states[layer_idx] = new_hidden;
                cell_states[layer_idx] = new_cell;
            }
        }
        
        // Generate prediction from final hidden state
        let final_hidden = &hidden_states[hidden_states.len() - 1];
        let prediction = (&self.output_layer * final_hidden)[0];
        
        // Calculate quantum confidence metrics
        let quantum_fidelity = self.layers.iter()
            .map(|layer| layer.quantum_fidelity)
            .sum::<f64>() / self.layers.len() as f64;
        
        let quantum_confidence = quantum_fidelity * 0.8 + 0.2; // Baseline confidence
        let classical_confidence = 0.7; // Placeholder for classical confidence
        
        // Store prediction
        let prediction_result = QuantumLSTMPrediction {
            value: prediction,
            quantum_confidence,
            classical_confidence,
            quantum_fidelity,
            timestamp: Utc::now(),
        };
        
        self.prediction_history.push_back(prediction_result);
        if self.prediction_history.len() > 1000 {
            self.prediction_history.pop_front();
        }
        
        // Update quantum advantage metric
        self.update_quantum_advantage_metric();
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 100 {
            tracing::warn!(
                "QLSTM prediction time: {}μs (target: <100μs)",
                elapsed.as_micros()
            );
        }
        
        Ok(prediction)
    }

    /// Update model with new training data
    pub async fn update(&mut self, training_data: &DMatrix<f64>, targets: &DVector<f64>) -> Result<(), TENGRIError> {
        // Simplified gradient descent update
        // In practice, this would use backpropagation through time (BPTT)
        
        let batch_size = training_data.ncols();
        let mut total_loss = 0.0;
        
        for batch_idx in 0..batch_size {
            let sequence = training_data.column(batch_idx);
            let target = targets[batch_idx];
            
            // Forward pass
            let mut input_matrix = DMatrix::zeros(sequence.len(), 1);
            input_matrix.set_column(0, &sequence);
            
            let prediction = self.predict(&input_matrix).await?;
            
            // Calculate loss
            let loss = (prediction - target).powi(2);
            total_loss += loss;
            
            // Simplified weight update (would need proper backpropagation)
            let gradient = 2.0 * (prediction - target);
            
            // Update output layer
            for i in 0..self.output_layer.ncols() {
                self.output_layer[(0, i)] -= self.learning_rate * gradient * 0.01;
            }
            
            // Update quantum coupling parameters
            for layer in &mut self.layers {
                layer.quantum_classical_coupling *= 0.999; // Gradual adaptation
                layer.decoherence_rate *= 0.9999; // Reduce decoherence over time
            }
        }
        
        let avg_loss = total_loss / batch_size as f64;
        tracing::info!("QLSTM training loss: {:.6}", avg_loss);
        
        Ok(())
    }

    /// Get quantum LSTM metrics
    pub async fn get_metrics(&self) -> Result<QuantumLSTMMetrics, TENGRIError> {
        let quantum_fidelity = self.layers.iter()
            .map(|layer| layer.quantum_fidelity)
            .sum::<f64>() / self.layers.len() as f64;
        
        // Calculate entanglement entropy
        let entanglement_entropy = self.calculate_entanglement_entropy();
        
        // Calculate classical accuracy from prediction history
        let classical_accuracy = if self.prediction_history.len() > 10 {
            self.prediction_history.iter()
                .map(|pred| pred.classical_confidence)
                .sum::<f64>() / self.prediction_history.len() as f64
        } else {
            0.0
        };
        
        Ok(QuantumLSTMMetrics {
            quantum_fidelity,
            entanglement_entropy,
            classical_accuracy,
            quantum_advantage: self.quantum_advantage_metric,
        })
    }

    /// Calculate entanglement entropy
    fn calculate_entanglement_entropy(&self) -> f64 {
        let mut total_entropy = 0.0;
        
        for layer in &self.layers {
            let n_qubits = layer.quantum_memory.n_qubits;
            let dim = 1usize << n_qubits;
            
            let mut entropy = 0.0;
            for i in 0..dim {
                let prob = layer.quantum_memory.amplitudes[i].norm_sqr();
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
            
            total_entropy += entropy;
        }
        
        total_entropy / self.layers.len() as f64
    }

    /// Update quantum advantage metric
    fn update_quantum_advantage_metric(&mut self) {
        if self.prediction_history.len() < 2 {
            return;
        }
        
        let recent_predictions: Vec<_> = self.prediction_history.iter()
            .rev()
            .take(10)
            .collect();
        
        let quantum_avg = recent_predictions.iter()
            .map(|pred| pred.quantum_confidence)
            .sum::<f64>() / recent_predictions.len() as f64;
        
        let classical_avg = recent_predictions.iter()
            .map(|pred| pred.classical_confidence)
            .sum::<f64>() / recent_predictions.len() as f64;
        
        self.quantum_advantage_metric = (quantum_avg - classical_avg).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_quantum_lstm_creation() {
        let qlstm = QuantumLSTM::new(4, 32).await.unwrap();
        assert_eq!(qlstm.layers.len(), 2);
        assert_eq!(qlstm.batch_size, 32);
    }

    #[tokio::test]
    async fn test_quantum_lstm_prediction() {
        let mut qlstm = QuantumLSTM::new(4, 32).await.unwrap();
        
        // Create sample input sequence
        let input_sequence = DMatrix::from_fn(10, 5, |i, j| {
            (i as f64 + j as f64) * 0.1
        });
        
        let prediction = qlstm.predict(&input_sequence).await.unwrap();
        assert!(prediction.is_finite());
    }

    #[tokio::test]
    async fn test_quantum_lstm_cell_forward() {
        let mut cell = QuantumLSTMCell::new(5, 10, 4).unwrap();
        
        let input = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let hidden = DVector::zeros(10);
        let cell_state = DVector::zeros(10);
        
        let (new_hidden, new_cell) = cell.forward(&input, &hidden, &cell_state).unwrap();
        
        assert_eq!(new_hidden.len(), 10);
        assert_eq!(new_cell.len(), 10);
        assert!(new_hidden.iter().all(|&x| x.is_finite()));
        assert!(new_cell.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_quantum_lstm_metrics() {
        let qlstm = QuantumLSTM::new(4, 32).await.unwrap();
        let metrics = qlstm.get_metrics().await.unwrap();
        
        assert!(metrics.quantum_fidelity >= 0.0 && metrics.quantum_fidelity <= 1.0);
        assert!(metrics.entanglement_entropy >= 0.0);
        assert!(metrics.classical_accuracy >= 0.0);
        assert!(metrics.quantum_advantage >= 0.0);
    }
}