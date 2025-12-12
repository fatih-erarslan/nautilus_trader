//! Quantum Long Short-Term Memory (QLSTM) Networks
//! 
//! Quantum-enhanced LSTM for time series prediction with quantum superposition
//! and entanglement for enhanced pattern recognition

// use std::collections::HashMap; // Unused
use nalgebra::{DMatrix, DVector};
// use num_complex::Complex64; // Unused
use rand::Rng;
use crate::{QuantumState, QuantumMarketData, QuantumPrediction, QuantumMLError, quantum_gates::QuantumGates};

/// QLSTM cell configuration
#[derive(Debug, Clone)]
pub struct QLSTMConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub quantum_layers: usize,
    pub learning_rate: f64,
    pub quantum_noise: f64,
    pub entanglement_strength: f64,
    pub dropout_rate: f64,
}

impl Default for QLSTMConfig {
    fn default() -> Self {
        Self {
            input_size: 64,
            hidden_size: 32,
            quantum_layers: 3,
            learning_rate: 0.001,
            quantum_noise: 0.01,
            entanglement_strength: 0.1,
            dropout_rate: 0.1,
        }
    }
}

/// QLSTM cell state
#[derive(Debug, Clone)]
pub struct QLSTMState {
    pub hidden_state: DVector<f64>,
    pub cell_state: DVector<f64>,
    pub quantum_state: QuantumState,
    pub entanglement_memory: f64,
}

impl QLSTMState {
    pub fn new(hidden_size: usize) -> Self {
        let n_qubits = (hidden_size as f64).log2().ceil() as usize;
        Self {
            hidden_state: DVector::zeros(hidden_size),
            cell_state: DVector::zeros(hidden_size),
            quantum_state: QuantumState::new(n_qubits),
            entanglement_memory: 0.0,
        }
    }
}

/// QLSTM model
pub struct QLSTMModel {
    config: QLSTMConfig,
    
    // Classical LSTM weights
    weight_ih: DMatrix<f64>, // Input to hidden weights
    weight_hh: DMatrix<f64>, // Hidden to hidden weights
    bias_ih: DVector<f64>,   // Input to hidden bias
    bias_hh: DVector<f64>,   // Hidden to hidden bias
    
    // Quantum enhancement parameters
    quantum_params: Vec<f64>,
    quantum_weight_matrix: DMatrix<f64>,
    
    // Model state
    training_history: Vec<f64>,
    current_state: Option<QLSTMState>,
    
    // Performance metrics
    predictions_count: u64,
    total_training_time: std::time::Duration,
    last_quantum_advantage: f64,
}

impl QLSTMModel {
    /// Create new QLSTM model
    pub async fn new(input_size: usize, hidden_size: usize) -> Result<Self, QuantumMLError> {
        let config = QLSTMConfig {
            input_size,
            hidden_size,
            ..Default::default()
        };
        
        let mut rng = rand::thread_rng();
        
        // Initialize classical LSTM weights with Xavier initialization
        let weight_ih = DMatrix::from_fn(4 * hidden_size, input_size, |_, _| {
            rng.gen_range(-1.0..1.0) / (input_size as f64).sqrt()
        });
        
        let weight_hh = DMatrix::from_fn(4 * hidden_size, hidden_size, |_, _| {
            rng.gen_range(-1.0..1.0) / (hidden_size as f64).sqrt()
        });
        
        let bias_ih = DVector::zeros(4 * hidden_size);
        let bias_hh = DVector::zeros(4 * hidden_size);
        
        // Initialize quantum parameters
        let n_qubits = (hidden_size as f64).log2().ceil() as usize;
        let quantum_params = (0..n_qubits * 3 * config.quantum_layers)
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();
        
        let quantum_weight_matrix = DMatrix::from_fn(hidden_size, hidden_size, |_, _| {
            rng.gen_range(-0.1..0.1)
        });
        
        Ok(Self {
            config,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            quantum_params,
            quantum_weight_matrix,
            training_history: Vec::new(),
            current_state: None,
            predictions_count: 0,
            total_training_time: std::time::Duration::new(0, 0),
            last_quantum_advantage: 1.0,
        })
    }
    
    /// Forward pass through QLSTM
    pub async fn forward(
        &mut self,
        input: &DVector<f64>,
        state: Option<QLSTMState>,
    ) -> Result<(DVector<f64>, QLSTMState), QuantumMLError> {
        let mut current_state = state.unwrap_or_else(|| QLSTMState::new(self.config.hidden_size));
        
        // Classical LSTM gates computation
        let input_transform = &self.weight_ih * input + &self.bias_ih;
        let hidden_transform = &self.weight_hh * &current_state.hidden_state + &self.bias_hh;
        let gates = input_transform + hidden_transform;
        
        // Extract gate values
        let (forget_gate, input_gate, output_gate, candidate_gate) = self.extract_gates(&gates);
        
        // Quantum enhancement
        let quantum_enhancement = self.compute_quantum_enhancement(&current_state, input).await?;
        
        // Update cell state with quantum enhancement
        let new_cell_state = forget_gate.component_mul(&current_state.cell_state) +
                           input_gate.component_mul(&candidate_gate) +
                           &quantum_enhancement * self.config.entanglement_strength;
        
        // Update hidden state
        let new_hidden_state = output_gate.component_mul(&new_cell_state.map(|x| x.tanh()));
        
        // Update quantum state
        self.update_quantum_state(&mut current_state, &new_hidden_state, &new_cell_state).await?;
        
        // Update state
        current_state.hidden_state = new_hidden_state.clone();
        current_state.cell_state = new_cell_state;
        
        Ok((new_hidden_state, current_state))
    }
    
    /// Extract LSTM gates from concatenated gate values
    fn extract_gates(&self, gates: &DVector<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>, DVector<f64>) {
        let gate_size = self.config.hidden_size;
        
        let forget_gate = gates.rows(0, gate_size).into_owned().map(|x| 1.0 / (1.0 + (-x).exp()));
        let input_gate = gates.rows(gate_size, gate_size).into_owned().map(|x| 1.0 / (1.0 + (-x).exp()));
        let output_gate = gates.rows(2 * gate_size, gate_size).into_owned().map(|x| 1.0 / (1.0 + (-x).exp()));
        let candidate_gate = gates.rows(3 * gate_size, gate_size).into_owned().map(|x| x.tanh());
        
        (forget_gate, input_gate, output_gate, candidate_gate)
    }
    
    /// Compute quantum enhancement for LSTM
    async fn compute_quantum_enhancement(
        &self,
        state: &QLSTMState,
        input: &DVector<f64>,
    ) -> Result<DVector<f64>, QuantumMLError> {
        let mut quantum_state = state.quantum_state.clone();
        
        // Encode input into quantum state
        let scaled_input: Vec<f64> = input.iter()
            .take(quantum_state.n_qubits)
            .map(|&x| x * 0.5) // Scale to prevent overflow
            .collect();
        
        QuantumGates::create_feature_map(&mut quantum_state, &scaled_input)?;
        
        // Apply variational quantum circuit
        QuantumGates::create_variational_circuit(&mut quantum_state, &self.quantum_params)?;
        
        // Extract quantum features
        let mut quantum_features = DVector::zeros(self.config.hidden_size);
        
        // Convert quantum amplitudes to classical features
        for i in 0..self.config.hidden_size {
            if i < quantum_state.amplitudes.len() {
                let amplitude = quantum_state.amplitudes[i];
                quantum_features[i] = amplitude.norm_sqr(); // Probability amplitudes
            }
        }
        
        // Apply quantum weight matrix
        let enhanced_features = &self.quantum_weight_matrix * &quantum_features;
        
        // Add quantum noise for regularization
        let mut rng = rand::thread_rng();
        let noise: DVector<f64> = DVector::from_fn(self.config.hidden_size, |_, _| {
            rng.gen_range(-self.config.quantum_noise..self.config.quantum_noise)
        });
        
        Ok(enhanced_features + noise)
    }
    
    /// Update quantum state based on LSTM output
    async fn update_quantum_state(
        &self,
        state: &mut QLSTMState,
        hidden_state: &DVector<f64>,
        cell_state: &DVector<f64>,
    ) -> Result<(), QuantumMLError> {
        // Create feedback parameters from LSTM states
        let feedback_params: Vec<f64> = hidden_state.iter()
            .zip(cell_state.iter())
            .take(state.quantum_state.n_qubits)
            .map(|(&h, &c)| (h + c) * 0.1) // Scale feedback
            .collect();
        
        // Apply feedback to quantum state
        if !feedback_params.is_empty() {
            for (i, &param) in feedback_params.iter().enumerate() {
                if i < state.quantum_state.n_qubits {
                    let ry_gate = QuantumGates::ry(param);
                    QuantumGates::apply_single_qubit_gate(&mut state.quantum_state, &ry_gate, i)?;
                }
            }
        }
        
        // Update entanglement memory
        state.quantum_state.calculate_entanglement();
        state.entanglement_memory = 0.9 * state.entanglement_memory + 
                                   0.1 * state.quantum_state.entanglement_measure;
        
        // Apply decoherence
        state.quantum_state.apply_decoherence(0.01, 1.0);
        
        Ok(())
    }
    
    /// Train QLSTM model
    pub async fn train(
        &mut self,
        training_data: &[QuantumMarketData],
        targets: &DVector<f64>,
    ) -> Result<(), QuantumMLError> {
        let start_time = std::time::Instant::now();
        
        if training_data.len() != targets.len() {
            return Err(QuantumMLError::NeuralNetworkTrainingFailed {
                reason: "Training data and targets length mismatch".to_string(),
            });
        }
        
        let mut total_loss = 0.0;
        let mut state = QLSTMState::new(self.config.hidden_size);
        
        // Training loop
        for _epoch in 0..100 { // Fixed number of epochs for now
            let mut epoch_loss = 0.0;
            
            for (_i, (data, &target)) in training_data.iter().zip(targets.iter()).enumerate() {
                // Convert market data to input vector
                let input = self.market_data_to_input(data)?;
                
                // Forward pass
                let (output, new_state) = self.forward(&input, Some(state.clone())).await?;
                state = new_state;
                
                // Compute loss (MSE)
                let prediction = output.iter().sum::<f64>() / output.len() as f64;
                let loss = (prediction - target).powi(2);
                epoch_loss += loss;
                
                // Backward pass (simplified gradient descent)
                self.backward_pass(&output, target, &input, &state).await?;
            }
            
            epoch_loss /= training_data.len() as f64;
            total_loss += epoch_loss;
            
            // Update training history
            self.training_history.push(epoch_loss);
            
            // Early stopping check
            if epoch_loss < 1e-6 {
                break;
            }
        }
        
        // Update metrics
        self.total_training_time += start_time.elapsed();
        
        // Calculate quantum advantage
        self.last_quantum_advantage = self.calculate_quantum_advantage();
        
        tracing::info!("QLSTM training completed with final loss: {:.6}", total_loss);
        
        Ok(())
    }
    
    /// Simplified backward pass
    async fn backward_pass(
        &mut self,
        output: &DVector<f64>,
        target: f64,
        input: &DVector<f64>,
        _state: &QLSTMState,
    ) -> Result<(), QuantumMLError> {
        let prediction = output.iter().sum::<f64>() / output.len() as f64;
        let error = prediction - target;
        
        // Update quantum parameters (simplified)
        for param in &mut self.quantum_params {
            *param -= self.config.learning_rate * error * 0.01;
        }
        
        // Update quantum weight matrix
        for i in 0..self.quantum_weight_matrix.nrows() {
            for j in 0..self.quantum_weight_matrix.ncols() {
                self.quantum_weight_matrix[(i, j)] -= self.config.learning_rate * error * 0.001;
            }
        }
        
        // Update classical weights (simplified)
        for i in 0..self.weight_ih.nrows() {
            for j in 0..self.weight_ih.ncols() {
                if j < input.len() {
                    self.weight_ih[(i, j)] -= self.config.learning_rate * error * input[j] * 0.01;
                }
            }
        }
        
        Ok(())
    }
    
    /// Make prediction using QLSTM
    pub async fn predict(&self, market_data: &QuantumMarketData) -> Result<QuantumPrediction, QuantumMLError> {
        let input = self.market_data_to_input(market_data)?;
        
        // Create a mutable copy for prediction
        let mut model_copy = self.clone();
        let state = QLSTMState::new(self.config.hidden_size);
        
        let (output, final_state) = model_copy.forward(&input, Some(state)).await?;
        
        let prediction_value = output.iter().sum::<f64>() / output.len() as f64;
        
        // Calculate uncertainty based on quantum state
        let uncertainty = self.calculate_prediction_uncertainty(&final_state);
        
        // Update prediction count
        model_copy.predictions_count += 1;
        
        Ok(QuantumPrediction {
            value: prediction_value,
            uncertainty,
            confidence_interval: (
                prediction_value - 2.0 * uncertainty,
                prediction_value + 2.0 * uncertainty,
            ),
            quantum_advantage: self.last_quantum_advantage,
            entanglement_contribution: final_state.entanglement_memory,
            prediction_timestamp: chrono::Utc::now(),
        })
    }
    
    /// Convert market data to input vector
    fn market_data_to_input(&self, market_data: &QuantumMarketData) -> Result<DVector<f64>, QuantumMLError> {
        let mut input = DVector::zeros(self.config.input_size);
        
        // Use prices as primary features
        let n_prices = market_data.prices.len().min(self.config.input_size / 2);
        for i in 0..n_prices {
            input[i] = market_data.prices[i];
        }
        
        // Use volumes as secondary features
        let n_volumes = market_data.volumes.len().min(self.config.input_size / 2);
        for i in 0..n_volumes {
            input[self.config.input_size / 2 + i] = market_data.volumes[i];
        }
        
        // Normalize input
        let max_val = input.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        if max_val > 0.0 {
            input /= max_val;
        }
        
        Ok(input)
    }
    
    /// Calculate prediction uncertainty
    fn calculate_prediction_uncertainty(&self, state: &QLSTMState) -> f64 {
        // Base uncertainty from quantum state entropy
        let entropy = state.quantum_state.entanglement_measure;
        
        // Add uncertainty from model training history
        let training_variance = if self.training_history.len() > 1 {
            let mean = self.training_history.iter().sum::<f64>() / self.training_history.len() as f64;
            let variance = self.training_history.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / self.training_history.len() as f64;
            variance.sqrt()
        } else {
            0.1
        };
        
        // Combine uncertainties
        entropy * 0.5 + training_variance * 0.5
    }
    
    /// Calculate quantum advantage metric
    fn calculate_quantum_advantage(&self) -> f64 {
        // Simplified quantum advantage calculation
        // Based on entanglement utilization and training performance
        
        let entanglement_factor = if !self.training_history.is_empty() {
            // Better training (lower loss) with quantum enhancement
            let final_loss = self.training_history.last().unwrap_or(&1.0);
            let classical_baseline = 0.1; // Assumed classical baseline
            (classical_baseline / final_loss).min(10.0) // Cap at 10x advantage
        } else {
            1.0
        };
        
        // Factor in quantum parameter utilization
        let param_utilization = self.quantum_params.iter()
            .map(|&p| p.abs())
            .sum::<f64>() / self.quantum_params.len() as f64;
        
        entanglement_factor * (1.0 + param_utilization * 0.1)
    }
    
    /// Get model performance metrics
    pub async fn get_metrics(&self) -> QLSTMMetrics {
        QLSTMMetrics {
            accuracy: self.calculate_accuracy(),
            quantum_advantage: self.last_quantum_advantage,
            predictions: self.predictions_count,
            avg_prediction_time: self.total_training_time / self.predictions_count.max(1) as u32,
            training_loss: self.training_history.last().copied().unwrap_or(0.0),
            quantum_parameter_count: self.quantum_params.len(),
        }
    }
    
    /// Calculate model accuracy
    fn calculate_accuracy(&self) -> f64 {
        // Simplified accuracy based on training loss
        if let Some(&final_loss) = self.training_history.last() {
            (1.0 - final_loss).max(0.0).min(1.0)
        } else {
            0.0
        }
    }
}

/// QLSTM performance metrics
#[derive(Debug, Clone)]
pub struct QLSTMMetrics {
    pub accuracy: f64,
    pub quantum_advantage: f64,
    pub predictions: u64,
    pub avg_prediction_time: std::time::Duration,
    pub training_loss: f64,
    pub quantum_parameter_count: usize,
}

// Implement Clone for QLSTMModel
impl Clone for QLSTMModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            weight_ih: self.weight_ih.clone(),
            weight_hh: self.weight_hh.clone(),
            bias_ih: self.bias_ih.clone(),
            bias_hh: self.bias_hh.clone(),
            quantum_params: self.quantum_params.clone(),
            quantum_weight_matrix: self.quantum_weight_matrix.clone(),
            training_history: self.training_history.clone(),
            current_state: self.current_state.clone(),
            predictions_count: self.predictions_count,
            total_training_time: self.total_training_time,
            last_quantum_advantage: self.last_quantum_advantage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_qlstm_creation() {
        let model = QLSTMModel::new(64, 32).await;
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.config.input_size, 64);
        assert_eq!(model.config.hidden_size, 32);
    }

    #[tokio::test]
    async fn test_qlstm_forward_pass() {
        let mut model = QLSTMModel::new(10, 5).await.unwrap();
        let input = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
        
        let result = model.forward(&input, None).await;
        assert!(result.is_ok());
        
        let (output, state) = result.unwrap();
        assert_eq!(output.len(), 5);
        assert_eq!(state.hidden_state.len(), 5);
        assert_eq!(state.cell_state.len(), 5);
    }

    #[tokio::test]
    async fn test_qlstm_prediction() {
        let model = QLSTMModel::new(10, 5).await.unwrap();
        
        let market_data = QuantumMarketData {
            prices: DVector::from_vec(vec![100.0, 101.0, 102.0]),
            volumes: DVector::from_vec(vec![1000.0, 1100.0, 900.0]),
            features: DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            timestamps: vec![chrono::Utc::now(); 3],
            quantum_encoding: None,
        };
        
        let prediction = model.predict(&market_data).await;
        assert!(prediction.is_ok());
        
        let pred = prediction.unwrap();
        assert!(pred.uncertainty >= 0.0);
        assert!(pred.quantum_advantage >= 0.0);
    }

    #[tokio::test]
    async fn test_quantum_enhancement() {
        let model = QLSTMModel::new(8, 4).await.unwrap();
        let state = QLSTMState::new(4);
        let input = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        
        let enhancement = model.compute_quantum_enhancement(&state, &input).await;
        assert!(enhancement.is_ok());
        
        let enhancement = enhancement.unwrap();
        assert_eq!(enhancement.len(), 4);
    }
}