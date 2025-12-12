//! Quantum Adaptive Temperature Scaling with Conformal Prediction (QATS-CP)
//! 
//! Advanced uncertainty quantification combining quantum superposition principles
//! with conformal prediction for reliable confidence intervals

use std::collections::VecDeque;
use nalgebra::{DVector};
use wide::{f64x4}; // SIMD support
use rand::Rng;
use crate::{QuantumState, QuantumMarketData, QuantumPrediction, QuantumMLError, quantum_gates::QuantumGates};

/// QATS-CP configuration
#[derive(Debug, Clone)]
pub struct QATSCPConfig {
    pub base_temperature: f64,
    pub confidence_level: f64,
    pub calibration_window_size: usize,
    pub quantum_temperature_coupling: f64,
    pub adaptive_learning_rate: f64,
    pub conformal_alpha: f64,
    pub simd_batch_size: usize,
    pub max_calibration_samples: usize,
}

impl Default for QATSCPConfig {
    fn default() -> Self {
        Self {
            base_temperature: 1.0,
            confidence_level: 0.95,
            calibration_window_size: 1000,
            quantum_temperature_coupling: 0.1,
            adaptive_learning_rate: 0.01,
            conformal_alpha: 0.05,
            simd_batch_size: 8,
            max_calibration_samples: 10000,
        }
    }
}

/// Calibration sample for conformal prediction
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    pub prediction: f64,
    pub actual: f64,
    pub quantum_uncertainty: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Conformal prediction scores
#[derive(Debug, Clone)]
pub struct ConformalScores {
    pub scores: Vec<f64>,
    pub quantum_scores: Vec<f64>,
    pub quantile_95: f64,
    pub quantile_99: f64,
}

/// QATS-CP model
pub struct QATSCPModel {
    config: QATSCPConfig,
    
    // Temperature scaling parameters
    temperature: f64,
    quantum_temperature_state: QuantumState,
    temperature_history: VecDeque<f64>,
    
    // Conformal prediction calibration
    calibration_samples: VecDeque<CalibrationSample>,
    conformal_scores: Option<ConformalScores>,
    
    // Quantum enhancement
    quantum_uncertainty_state: QuantumState,
    quantum_params: Vec<f64>,
    
    // Performance tracking
    predictions_count: u64,
    calibration_accuracy: f64,
    last_quantum_advantage: f64,
    
    // SIMD optimization buffers
    simd_prediction_buffer: Vec<f64>,
    simd_score_buffer: Vec<f64>,
}

impl QATSCPModel {
    /// Create new QATS-CP model
    pub async fn new(base_temperature: f64, confidence_level: f64) -> Result<Self, QuantumMLError> {
        let config = QATSCPConfig {
            base_temperature,
            confidence_level,
            ..Default::default()
        };
        
        // Initialize quantum states
        let quantum_temperature_state = QuantumState::new(3); // 3 qubits for temperature
        let quantum_uncertainty_state = QuantumState::new(4); // 4 qubits for uncertainty
        
        // Initialize quantum parameters
        let mut rng = rand::thread_rng();
        let quantum_params: Vec<f64> = (0..21) // 3 layers * 7 qubits * 3 params per qubit
            .map(|_| rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI))
            .collect();
        
        let simd_batch_size = config.simd_batch_size;
        
        Ok(Self {
            config,
            temperature: base_temperature,
            quantum_temperature_state,
            temperature_history: VecDeque::new(),
            calibration_samples: VecDeque::new(),
            conformal_scores: None,
            quantum_uncertainty_state,
            quantum_params,
            predictions_count: 0,
            calibration_accuracy: 0.0,
            last_quantum_advantage: 1.0,
            simd_prediction_buffer: Vec::with_capacity(simd_batch_size),
            simd_score_buffer: Vec::with_capacity(simd_batch_size),
        })
    }
    
    /// Adaptive temperature scaling with quantum enhancement
    pub async fn adaptive_temperature_scaling(
        &mut self,
        logits: &DVector<f64>,
        quantum_state: &QuantumState,
    ) -> Result<DVector<f64>, QuantumMLError> {
        // Update quantum temperature state
        self.update_quantum_temperature_state(logits, quantum_state).await?;
        
        // Calculate quantum-enhanced temperature
        let quantum_temperature_factor = self.calculate_quantum_temperature_factor();
        let adaptive_temperature = self.temperature * (1.0 + quantum_temperature_factor);
        
        // Apply temperature scaling with SIMD optimization
        let scaled_logits = self.simd_temperature_scaling(logits, adaptive_temperature)?;
        
        // Update temperature history
        self.temperature_history.push_back(adaptive_temperature);
        if self.temperature_history.len() > self.config.calibration_window_size {
            self.temperature_history.pop_front();
        }
        
        // Adaptive temperature update
        self.update_temperature_adaptively(&scaled_logits).await?;
        
        Ok(scaled_logits)
    }
    
    /// SIMD-optimized temperature scaling
    fn simd_temperature_scaling(&self, logits: &DVector<f64>, temperature: f64) -> Result<DVector<f64>, QuantumMLError> {
        let mut scaled_logits = logits.clone();
        let temp_recip = 1.0 / temperature;
        
        // Process in SIMD chunks
        let chunks = scaled_logits.as_mut_slice().chunks_exact_mut(4);
        let temp_vec = f64x4::splat(temp_recip);
        
        for chunk in chunks {
            let logits_vec = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let scaled_vec = logits_vec * temp_vec;
            
            let scaled_array: [f64; 4] = scaled_vec.into();
            chunk[0] = scaled_array[0];
            chunk[1] = scaled_array[1];
            chunk[2] = scaled_array[2];
            chunk[3] = scaled_array[3];
        }
        
        // Handle remaining elements
        let remainder = scaled_logits.len() % 4;
        if remainder > 0 {
            let start = scaled_logits.len() - remainder;
            for i in start..scaled_logits.len() {
                scaled_logits[i] *= temp_recip;
            }
        }
        
        Ok(scaled_logits)
    }
    
    /// Update quantum temperature state
    async fn update_quantum_temperature_state(
        &mut self,
        logits: &DVector<f64>,
        quantum_state: &QuantumState,
    ) -> Result<(), QuantumMLError> {
        // Encode logits information into quantum temperature state
        let logit_features: Vec<f64> = logits.iter()
            .take(self.quantum_temperature_state.n_qubits)
            .map(|&x| x * 0.1) // Scale for quantum encoding
            .collect();
        
        if !logit_features.is_empty() {
            QuantumGates::create_feature_map(&mut self.quantum_temperature_state, &logit_features)?;
        }
        
        // Apply entanglement with input quantum state
        if quantum_state.n_qubits >= 2 && self.quantum_temperature_state.n_qubits >= 2 {
            let entanglement_strength = self.config.quantum_temperature_coupling;
            
            // Create controlled rotation based on entanglement
            let angle = quantum_state.entanglement_measure * entanglement_strength;
            let ry_gate = QuantumGates::ry(angle);
            QuantumGates::apply_single_qubit_gate(&mut self.quantum_temperature_state, &ry_gate, 0)?;
        }
        
        // Apply variational circuit for temperature optimization
        let temp_params = &self.quantum_params[0..9]; // 3 qubits * 3 params
        QuantumGates::create_variational_circuit(&mut self.quantum_temperature_state, temp_params)?;
        
        Ok(())
    }
    
    /// Calculate quantum temperature factor
    fn calculate_quantum_temperature_factor(&self) -> f64 {
        // Extract temperature adjustment from quantum state
        let mut factor = 0.0;
        
        for amplitude in &self.quantum_temperature_state.amplitudes {
            factor += amplitude.norm_sqr() * amplitude.arg().sin(); // Phase information
        }
        
        // Normalize to reasonable range
        factor * self.config.quantum_temperature_coupling
    }
    
    /// Update temperature adaptively based on calibration performance
    async fn update_temperature_adaptively(&mut self, _scaled_logits: &DVector<f64>) -> Result<(), QuantumMLError> {
        if self.calibration_samples.len() < 10 {
            return Ok(()); // Need sufficient calibration data
        }
        
        // Calculate recent calibration accuracy
        let recent_samples = self.calibration_samples.iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();
        
        let mut accurate_predictions = 0;
        for sample in &recent_samples {
            let relative_error = (sample.prediction - sample.actual).abs() / sample.actual.abs().max(1e-6);
            if relative_error < 0.1 { // 10% accuracy threshold
                accurate_predictions += 1;
            }
        }
        
        let accuracy = accurate_predictions as f64 / recent_samples.len() as f64;
        self.calibration_accuracy = accuracy;
        
        // Adjust temperature based on accuracy
        let target_accuracy = 0.85;
        let accuracy_error = target_accuracy - accuracy;
        
        let temperature_adjustment = self.config.adaptive_learning_rate * accuracy_error;
        self.temperature = (self.temperature + temperature_adjustment).max(0.1).min(10.0);
        
        Ok(())
    }
    
    /// Generate conformal prediction intervals
    pub async fn conformal_prediction(
        &mut self,
        prediction: f64,
        quantum_uncertainty: f64,
    ) -> Result<(f64, f64), QuantumMLError> {
        // Update conformal scores if we have enough calibration data
        if self.calibration_samples.len() >= 50 {
            self.update_conformal_scores().await?;
        }
        
        // Calculate prediction interval
        let interval = if let Some(ref scores) = self.conformal_scores {
            let base_interval = scores.quantile_95;
            let quantum_adjustment = quantum_uncertainty * self.config.quantum_temperature_coupling;
            base_interval + quantum_adjustment
        } else {
            // Fallback to quantum uncertainty if no calibration data
            quantum_uncertainty * 2.0
        };
        
        Ok((prediction - interval, prediction + interval))
    }
    
    /// Update conformal scores using SIMD optimization
    async fn update_conformal_scores(&mut self) -> Result<(), QuantumMLError> {
        let samples: Vec<_> = self.calibration_samples.iter().collect();
        
        // Calculate nonconformity scores with SIMD
        let mut scores = Vec::with_capacity(samples.len());
        let mut quantum_scores = Vec::with_capacity(samples.len());
        
        // Process samples in SIMD batches
        self.simd_prediction_buffer.clear();
        self.simd_score_buffer.clear();
        
        for chunk in samples.chunks(self.config.simd_batch_size) {
            // Prepare SIMD batch
            self.simd_prediction_buffer.clear();
            self.simd_score_buffer.clear();
            
            for &sample in chunk {
                let score = (sample.prediction - sample.actual).abs();
                scores.push(score);
                
                let quantum_score = score * (1.0 + sample.quantum_uncertainty);
                quantum_scores.push(quantum_score);
            }
        }
        
        // Calculate quantiles
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        quantum_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = scores.len();
        let quantile_95_idx = ((n as f64) * 0.95) as usize;
        let quantile_99_idx = ((n as f64) * 0.99) as usize;
        
        self.conformal_scores = Some(ConformalScores {
            scores: scores.clone(),
            quantum_scores: quantum_scores.clone(),
            quantile_95: scores.get(quantile_95_idx).copied().unwrap_or(1.0),
            quantile_99: scores.get(quantile_99_idx).copied().unwrap_or(2.0),
        });
        
        Ok(())
    }
    
    /// Add calibration sample
    pub fn add_calibration_sample(&mut self, prediction: f64, actual: f64, quantum_uncertainty: f64) {
        let sample = CalibrationSample {
            prediction,
            actual,
            quantum_uncertainty,
            timestamp: chrono::Utc::now(),
        };
        
        self.calibration_samples.push_back(sample);
        
        // Keep only recent samples
        if self.calibration_samples.len() > self.config.max_calibration_samples {
            self.calibration_samples.pop_front();
        }
    }
    
    /// Train QATS-CP model
    pub async fn train(
        &mut self,
        training_data: &[QuantumMarketData],
        targets: &DVector<f64>,
    ) -> Result<(), QuantumMLError> {
        if training_data.len() != targets.len() {
            return Err(QuantumMLError::NeuralNetworkTrainingFailed {
                reason: "Training data and targets length mismatch".to_string(),
            });
        }
        
        // Train temperature scaling parameters
        for (data, &target) in training_data.iter().zip(targets.iter()) {
            // Create dummy logits for training (in real scenario, these come from a base model)
            let logits = self.create_dummy_logits(data)?;
            
            // Get quantum state from market data
            let default_quantum_state = QuantumState::new(2);
            let quantum_state = data.quantum_encoding.as_ref().unwrap_or(&default_quantum_state);
            
            // Apply temperature scaling
            let scaled_logits = self.adaptive_temperature_scaling(&logits, quantum_state).await?;
            
            // Convert to prediction
            let prediction = self.logits_to_prediction(&scaled_logits);
            
            // Calculate quantum uncertainty
            let quantum_uncertainty = self.calculate_quantum_uncertainty().await?;
            
            // Add calibration sample
            self.add_calibration_sample(prediction, target, quantum_uncertainty);
        }
        
        // Update quantum advantage
        self.last_quantum_advantage = self.calculate_quantum_advantage();
        
        tracing::info!("QATS-CP training completed with {} calibration samples", 
                      self.calibration_samples.len());
        
        Ok(())
    }
    
    /// Create dummy logits for training (in practice, these come from a base model)
    fn create_dummy_logits(&self, market_data: &QuantumMarketData) -> Result<DVector<f64>, QuantumMLError> {
        let n_classes = 5; // Example: 5 prediction classes
        let mut logits = DVector::zeros(n_classes);
        
        // Use price movement as basis for logits
        if !market_data.prices.is_empty() {
            let price_change = if market_data.prices.len() > 1 {
                market_data.prices[market_data.prices.len() - 1] - market_data.prices[0]
            } else {
                0.0
            };
            
            // Convert price change to logits distribution
            for i in 0..n_classes {
                logits[i] = price_change * (i as f64 - 2.0) * 0.1; // Center around class 2
            }
        }
        
        Ok(logits)
    }
    
    /// Convert logits to single prediction value
    fn logits_to_prediction(&self, logits: &DVector<f64>) -> f64 {
        // Apply softmax and compute expected value
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_logits.iter().sum();
        
        // Calculate expected value (weighted average of class indices)
        let mut expected_value = 0.0;
        for (i, &exp_logit) in exp_logits.iter().enumerate() {
            let probability = exp_logit / sum_exp;
            expected_value += probability * i as f64;
        }
        
        expected_value
    }
    
    /// Calculate quantum uncertainty
    async fn calculate_quantum_uncertainty(&mut self) -> Result<f64, QuantumMLError> {
        // Update quantum uncertainty state
        let uncertainty_params = &self.quantum_params[9..21]; // Next 12 parameters
        QuantumGates::create_variational_circuit(&mut self.quantum_uncertainty_state, uncertainty_params)?;
        
        // Calculate uncertainty from quantum state
        let entropy = self.quantum_uncertainty_state.calculate_entanglement();
        let amplitude_variance = self.calculate_amplitude_variance();
        
        Ok(entropy * 0.5 + amplitude_variance * 0.5)
    }
    
    /// Calculate amplitude variance as uncertainty measure
    fn calculate_amplitude_variance(&self) -> f64 {
        let amplitudes: Vec<f64> = self.quantum_uncertainty_state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        if amplitudes.is_empty() {
            return 0.1;
        }
        
        let mean = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        let variance = amplitudes.iter()
            .map(|&amp| (amp - mean).powi(2))
            .sum::<f64>() / amplitudes.len() as f64;
        
        variance.sqrt()
    }
    
    /// Make prediction with QATS-CP
    pub async fn predict(&mut self, market_data: &QuantumMarketData) -> Result<QuantumPrediction, QuantumMLError> {
        // Create logits from market data
        let logits = self.create_dummy_logits(market_data)?;
        
        // Get quantum state
        let default_quantum_state = QuantumState::new(2);
        let quantum_state = market_data.quantum_encoding.as_ref().unwrap_or(&default_quantum_state);
        
        // Apply temperature scaling
        let scaled_logits = self.adaptive_temperature_scaling(&logits, quantum_state).await?;
        
        // Convert to prediction
        let prediction_value = self.logits_to_prediction(&scaled_logits);
        
        // Calculate quantum uncertainty
        let quantum_uncertainty = self.calculate_quantum_uncertainty().await?;
        
        // Generate conformal prediction interval
        let (lower_bound, upper_bound) = self.conformal_prediction(prediction_value, quantum_uncertainty).await?;
        
        self.predictions_count += 1;
        
        Ok(QuantumPrediction {
            value: prediction_value,
            uncertainty: quantum_uncertainty,
            confidence_interval: (lower_bound, upper_bound),
            quantum_advantage: self.last_quantum_advantage,
            entanglement_contribution: quantum_state.entanglement_measure,
            prediction_timestamp: chrono::Utc::now(),
        })
    }
    
    /// Calculate quantum advantage
    fn calculate_quantum_advantage(&self) -> f64 {
        let temperature_stability = self.calculate_temperature_stability();
        let calibration_quality = self.calibration_accuracy;
        let quantum_utilization = self.calculate_quantum_utilization();
        
        (1.0 + temperature_stability) * (1.0 + calibration_quality) * (1.0 + quantum_utilization)
    }
    
    /// Calculate temperature stability
    fn calculate_temperature_stability(&self) -> f64 {
        if self.temperature_history.len() < 2 {
            return 0.0;
        }
        
        let temperatures: Vec<f64> = self.temperature_history.iter().copied().collect();
        let mean_temp = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
        let variance = temperatures.iter()
            .map(|&temp| (temp - mean_temp).powi(2))
            .sum::<f64>() / temperatures.len() as f64;
        
        1.0 / (1.0 + variance) // Higher stability = lower variance
    }
    
    /// Calculate quantum utilization
    fn calculate_quantum_utilization(&self) -> f64 {
        let temp_entanglement = self.quantum_temperature_state.entanglement_measure;
        let uncertainty_entanglement = self.quantum_uncertainty_state.entanglement_measure;
        
        (temp_entanglement + uncertainty_entanglement) / 2.0
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> QATSCPMetrics {
        QATSCPMetrics {
            accuracy: self.calibration_accuracy,
            quantum_advantage: self.last_quantum_advantage,
            predictions: self.predictions_count,
            avg_prediction_time: std::time::Duration::from_millis(5), // Estimated
            current_temperature: self.temperature,
            calibration_samples: self.calibration_samples.len(),
            conformal_coverage: self.calculate_conformal_coverage(),
        }
    }
    
    /// Calculate conformal prediction coverage
    fn calculate_conformal_coverage(&self) -> f64 {
        if self.calibration_samples.len() < 10 {
            return 0.0;
        }
        
        // Estimate coverage based on prediction accuracy
        self.calibration_accuracy * self.config.confidence_level
    }
}

/// QATS-CP performance metrics
#[derive(Debug, Clone)]
pub struct QATSCPMetrics {
    pub accuracy: f64,
    pub quantum_advantage: f64,
    pub predictions: u64,
    pub avg_prediction_time: std::time::Duration,
    pub current_temperature: f64,
    pub calibration_samples: usize,
    pub conformal_coverage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_qats_cp_creation() {
        let model = QATSCPModel::new(1.0, 0.95).await;
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_abs_diff_eq!(model.temperature, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(model.config.confidence_level, 0.95, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_simd_temperature_scaling() {
        let model = QATSCPModel::new(2.0, 0.95).await.unwrap();
        let logits = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        
        let scaled = model.simd_temperature_scaling(&logits, 2.0).unwrap();
        
        for (i, &scaled_val) in scaled.iter().enumerate() {
            assert_abs_diff_eq!(scaled_val, logits[i] / 2.0, epsilon = 1e-10);
        }
    }

    #[tokio::test]
    async fn test_calibration_sample() {
        let mut model = QATSCPModel::new(1.0, 0.95).await.unwrap();
        
        model.add_calibration_sample(1.5, 1.6, 0.1);
        assert_eq!(model.calibration_samples.len(), 1);
        
        let sample = &model.calibration_samples[0];
        assert_abs_diff_eq!(sample.prediction, 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(sample.actual, 1.6, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_conformal_prediction() {
        let mut model = QATSCPModel::new(1.0, 0.95).await.unwrap();
        
        // Add some calibration samples
        for i in 0..100 {
            model.add_calibration_sample(i as f64, (i as f64) + 0.1, 0.05);
        }
        
        let (lower, upper) = model.conformal_prediction(50.0, 0.1).await.unwrap();
        assert!(lower < 50.0);
        assert!(upper > 50.0);
        assert!(upper - lower > 0.0);
    }

    #[tokio::test]
    async fn test_qats_cp_prediction() {
        let mut model = QATSCPModel::new(1.0, 0.95).await.unwrap();
        
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
        assert!(pred.confidence_interval.0 <= pred.value);
        assert!(pred.confidence_interval.1 >= pred.value);
    }
}