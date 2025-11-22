//! Quantum Adaptive Temperature Scaling with Conformal Prediction (QATS-CP)
//!
//! Extends classical ATS-CP with quantum uncertainty quantification and
//! quantum-enhanced conformal prediction intervals for trading decisions.

use crate::TENGRIError;

/// Quantum Attention Trading System
#[derive(Debug)]
pub struct QuantumAttentionTradingSystem {
    quantum_state: QuantumState,
    temperature_params: QuantumTemperatureParameters,
    conformal_params: QuantumConformalParameters,
}

impl QuantumAttentionTradingSystem {
    pub fn new() -> Self {
        Self {
            quantum_state: QuantumState::new(8), // 8 qubits for demonstration
            temperature_params: QuantumTemperatureParameters::default(),
            conformal_params: QuantumConformalParameters::default(),
        }
    }
}
use super::quantum_gates::{QuantumState, QuantumCircuit, QuantumGateOp};
use super::{CombinedPrediction, QuantumUncertainty};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{Normal, ContinuousCDF};
use std::collections::VecDeque;
use rayon::prelude::*;
use chrono::{DateTime, Utc};

/// Quantum temperature scaling parameters
#[derive(Debug, Clone)]
pub struct QuantumTemperatureParameters {
    pub classical_temperature: f64,
    pub quantum_temperature: f64,
    pub quantum_coherence_factor: f64,
    pub entanglement_scaling: f64,
    pub decoherence_rate: f64,
    pub quantum_confidence_boost: f64,
}

impl Default for QuantumTemperatureParameters {
    fn default() -> Self {
        Self {
            classical_temperature: 1.0,
            quantum_temperature: 1.0,
            quantum_coherence_factor: 0.1,
            entanglement_scaling: 0.05,
            decoherence_rate: 0.01,
            quantum_confidence_boost: 0.1,
        }
    }
}

/// Quantum conformal prediction parameters
#[derive(Debug, Clone)]
pub struct QuantumConformalParameters {
    pub significance_level: f64,
    pub quantum_correction_factor: f64,
    pub entanglement_bonus: f64,
    pub coherence_threshold: f64,
    pub adaptive_window_size: usize,
    pub quantum_interval_expansion: f64,
}

impl Default for QuantumConformalParameters {
    fn default() -> Self {
        Self {
            significance_level: 0.05, // 95% confidence
            quantum_correction_factor: 0.1,
            entanglement_bonus: 0.05,
            coherence_threshold: 0.8,
            adaptive_window_size: 100,
            quantum_interval_expansion: 0.02,
        }
    }
}

/// Quantum ATS-CP calibration data
#[derive(Debug, Clone)]
pub struct QuantumCalibrationData {
    pub predictions: Vec<f64>,
    pub actual_values: Vec<f64>,
    pub quantum_confidences: Vec<f64>,
    pub quantum_fidelities: Vec<f64>,
    pub timestamps: Vec<DateTime<Utc>>,
}

impl QuantumCalibrationData {
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            actual_values: Vec::new(),
            quantum_confidences: Vec::new(),
            quantum_fidelities: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, prediction: f64, actual: f64, quantum_confidence: f64, quantum_fidelity: f64) {
        self.predictions.push(prediction);
        self.actual_values.push(actual);
        self.quantum_confidences.push(quantum_confidence);
        self.quantum_fidelities.push(quantum_fidelity);
        self.timestamps.push(Utc::now());
    }

    pub fn len(&self) -> usize {
        self.predictions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predictions.is_empty()
    }
}

/// Quantum Adaptive Temperature Scaling with Conformal Prediction
pub struct QuantumATS {
    pub temperature_params: QuantumTemperatureParameters,
    pub conformal_params: QuantumConformalParameters,
    pub quantum_state: QuantumState,
    pub calibration_data: QuantumCalibrationData,
    pub uncertainty_threshold: f64,
    
    // Quantum circuits for uncertainty quantification
    pub uncertainty_circuit: QuantumCircuit,
    pub temperature_circuit: QuantumCircuit,
    
    // Adaptive parameters
    pub adaptive_temperature_history: VecDeque<f64>,
    pub quantum_advantage_history: VecDeque<f64>,
    pub last_calibration_time: DateTime<Utc>,
    pub calibration_interval_hours: i64,
    
    // Performance metrics
    pub calibration_error: f64,
    pub coverage_probability: f64,
    pub quantum_enhancement_factor: f64,
}

impl QuantumATS {
    /// Create new quantum ATS-CP system
    pub async fn new(uncertainty_threshold: f64) -> Result<Self, TENGRIError> {
        let n_qubits = 8; // Quantum register size
        let quantum_state = QuantumState::new(n_qubits);
        
        // Create quantum circuits
        let uncertainty_circuit = Self::create_uncertainty_circuit(n_qubits);
        let temperature_circuit = Self::create_temperature_circuit(n_qubits);
        
        Ok(Self {
            temperature_params: QuantumTemperatureParameters::default(),
            conformal_params: QuantumConformalParameters::default(),
            quantum_state,
            calibration_data: QuantumCalibrationData::new(),
            uncertainty_threshold,
            uncertainty_circuit,
            temperature_circuit,
            adaptive_temperature_history: VecDeque::with_capacity(1000),
            quantum_advantage_history: VecDeque::with_capacity(1000),
            last_calibration_time: Utc::now(),
            calibration_interval_hours: 1,
            calibration_error: 0.0,
            coverage_probability: 0.95,
            quantum_enhancement_factor: 1.0,
        })
    }

    /// Combine predictions with quantum ATS-CP
    pub async fn combine_predictions(
        &mut self,
        predictions: &[f64],
        quantum_uncertainty: &QuantumUncertainty,
    ) -> Result<CombinedPrediction, TENGRIError> {
        let start_time = std::time::Instant::now();
        
        // Update quantum state with prediction data
        self.update_quantum_state(predictions, quantum_uncertainty).await?;
        
        // Compute quantum-enhanced temperature
        let quantum_temperature = self.compute_quantum_temperature(predictions, quantum_uncertainty).await?;
        
        // Apply temperature scaling to predictions
        let scaled_predictions = self.apply_temperature_scaling(predictions, quantum_temperature)?;
        
        // Compute combined prediction
        let combined_prediction = self.compute_ensemble_prediction(&scaled_predictions)?;
        
        // Generate quantum-enhanced conformal prediction intervals
        let (lower_bound, upper_bound) = self.compute_quantum_conformal_intervals(
            combined_prediction,
            quantum_uncertainty,
        ).await?;
        
        // Compute quantum-enhanced confidence
        let quantum_confidence = self.compute_quantum_confidence(
            combined_prediction,
            quantum_uncertainty,
            &scaled_predictions,
        ).await?;
        
        // Update adaptive parameters
        self.update_adaptive_parameters(quantum_temperature, quantum_confidence).await?;
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 50 {
            tracing::warn!(
                "Quantum ATS-CP processing time: {}μs (target: <50μs)",
                elapsed.as_micros()
            );
        }
        
        Ok(CombinedPrediction {
            value: combined_prediction,
            confidence: quantum_confidence,
            uncertainty_bounds: (lower_bound, upper_bound),
        })
    }

    /// Update quantum state with prediction information
    async fn update_quantum_state(
        &mut self,
        predictions: &[f64],
        quantum_uncertainty: &QuantumUncertainty,
    ) -> Result<(), TENGRIError> {
        let n_qubits = self.quantum_state.n_qubits;
        let mut encoding_circuit = QuantumCircuit::new(n_qubits);
        
        // Encode predictions into quantum state
        for (i, &prediction) in predictions.iter().enumerate() {
            if i < n_qubits {
                let angle = prediction * std::f64::consts::PI; // Normalize to [0, π]
                encoding_circuit.add_gate(QuantumGateOp::RY(i, angle));
            }
        }
        
        // Encode uncertainty information
        let uncertainty_angle = quantum_uncertainty.entropy * std::f64::consts::PI / 4.0;
        for i in 0..n_qubits.min(4) {
            encoding_circuit.add_gate(QuantumGateOp::RZ(i, uncertainty_angle));
        }
        
        // Create entanglement for correlation encoding
        for i in 0..n_qubits-1 {
            if quantum_uncertainty.variance > 0.1 {
                encoding_circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % n_qubits));
            }
        }
        
        // Execute quantum encoding
        encoding_circuit.execute(&mut self.quantum_state)?;
        
        Ok(())
    }

    /// Compute quantum-enhanced temperature
    async fn compute_quantum_temperature(
        &self,
        predictions: &[f64],
        quantum_uncertainty: &QuantumUncertainty,
    ) -> Result<f64, TENGRIError> {
        // Classical temperature based on prediction variance
        let prediction_variance = if predictions.len() > 1 {
            let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
            predictions.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / predictions.len() as f64
        } else {
            0.1
        };
        
        let classical_temperature = (1.0 + prediction_variance).ln();
        
        // Quantum temperature enhancement
        let quantum_fidelity = self.quantum_state.fidelity;
        let quantum_entropy = quantum_uncertainty.entropy;
        
        let quantum_enhancement = quantum_fidelity * quantum_entropy * 
                                 self.temperature_params.quantum_coherence_factor;
        
        let quantum_temperature = classical_temperature + quantum_enhancement;
        
        // Bound temperature to reasonable range
        let bounded_temperature = quantum_temperature.max(0.1).min(5.0);
        
        Ok(bounded_temperature)
    }

    /// Apply temperature scaling to predictions
    fn apply_temperature_scaling(&self, predictions: &[f64], temperature: f64) -> Result<Vec<f64>, TENGRIError> {
        let scaled_predictions = predictions.iter()
            .map(|&p| p / temperature)
            .collect();
        
        Ok(scaled_predictions)
    }

    /// Compute ensemble prediction
    fn compute_ensemble_prediction(&self, predictions: &[f64]) -> Result<f64, TENGRIError> {
        if predictions.is_empty() {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "No predictions to combine".to_string(),
            });
        }
        
        // Weighted average based on quantum confidence
        let quantum_fidelity = self.quantum_state.fidelity;
        let weights: Vec<f64> = predictions.iter()
            .enumerate()
            .map(|(i, _)| {
                let base_weight = 1.0 / predictions.len() as f64;
                let quantum_weight = quantum_fidelity * (1.0 + i as f64 * 0.1);
                base_weight * quantum_weight
            })
            .collect();
        
        let weight_sum: f64 = weights.iter().sum();
        let weighted_sum: f64 = predictions.iter()
            .zip(weights.iter())
            .map(|(pred, weight)| pred * weight)
            .sum();
        
        Ok(weighted_sum / weight_sum)
    }

    /// Compute quantum-enhanced conformal prediction intervals
    async fn compute_quantum_conformal_intervals(
        &self,
        prediction: f64,
        quantum_uncertainty: &QuantumUncertainty,
    ) -> Result<(f64, f64), TENGRIError> {
        // Classical conformal prediction interval
        let alpha = self.conformal_params.significance_level;
        let z_score = Normal::new(0.0, 1.0).unwrap()
            .inverse_cdf(1.0 - alpha / 2.0);
        
        // Base interval from calibration data
        let base_interval = if self.calibration_data.len() > 10 {
            let errors: Vec<f64> = self.calibration_data.predictions.iter()
                .zip(self.calibration_data.actual_values.iter())
                .map(|(pred, actual)| (pred - actual).abs())
                .collect();
            
            let percentile_idx = ((1.0 - alpha) * errors.len() as f64) as usize;
            let mut sorted_errors = errors.clone();
            sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            if percentile_idx < sorted_errors.len() {
                sorted_errors[percentile_idx]
            } else {
                0.1
            }
        } else {
            0.1 // Default interval
        };
        
        // Quantum enhancement to interval
        let quantum_fidelity = self.quantum_state.fidelity;
        let quantum_entropy = quantum_uncertainty.entropy;
        
        let quantum_interval_adjustment = quantum_entropy * 
                                        self.conformal_params.quantum_correction_factor *
                                        (1.0 + quantum_fidelity * self.conformal_params.entanglement_bonus);
        
        let adjusted_interval = base_interval + quantum_interval_adjustment;
        
        // Compute final bounds
        let lower_bound = prediction - adjusted_interval;
        let upper_bound = prediction + adjusted_interval;
        
        Ok((lower_bound, upper_bound))
    }

    /// Compute quantum-enhanced confidence
    async fn compute_quantum_confidence(
        &self,
        prediction: f64,
        quantum_uncertainty: &QuantumUncertainty,
        scaled_predictions: &[f64],
    ) -> Result<f64, TENGRIError> {
        // Classical confidence based on prediction consistency
        let prediction_std = if scaled_predictions.len() > 1 {
            let mean = scaled_predictions.iter().sum::<f64>() / scaled_predictions.len() as f64;
            let variance = scaled_predictions.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / scaled_predictions.len() as f64;
            variance.sqrt()
        } else {
            0.1
        };
        
        let classical_confidence = (-prediction_std).exp();
        
        // Quantum confidence enhancement
        let quantum_fidelity = self.quantum_state.fidelity;
        let quantum_coherence = 1.0 - quantum_uncertainty.entropy;
        
        let quantum_confidence_boost = quantum_fidelity * quantum_coherence * 
                                     self.temperature_params.quantum_confidence_boost;
        
        let quantum_confidence = classical_confidence + quantum_confidence_boost;
        
        // Bound confidence to [0, 1]
        let bounded_confidence = quantum_confidence.max(0.0).min(1.0);
        
        Ok(bounded_confidence)
    }

    /// Update adaptive parameters
    async fn update_adaptive_parameters(
        &mut self,
        quantum_temperature: f64,
        quantum_confidence: f64,
    ) -> Result<(), TENGRIError> {
        // Update temperature history
        self.adaptive_temperature_history.push_back(quantum_temperature);
        if self.adaptive_temperature_history.len() > 1000 {
            self.adaptive_temperature_history.pop_front();
        }
        
        // Update quantum advantage history
        let quantum_advantage = quantum_confidence - 0.5; // Baseline confidence
        self.quantum_advantage_history.push_back(quantum_advantage);
        if self.quantum_advantage_history.len() > 1000 {
            self.quantum_advantage_history.pop_front();
        }
        
        // Adapt temperature parameters
        if self.adaptive_temperature_history.len() > 10 {
            let recent_temps: Vec<f64> = self.adaptive_temperature_history.iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            
            let temp_mean = recent_temps.iter().sum::<f64>() / recent_temps.len() as f64;
            let temp_std = {
                let variance = recent_temps.iter()
                    .map(|&t| (t - temp_mean).powi(2))
                    .sum::<f64>() / recent_temps.len() as f64;
                variance.sqrt()
            };
            
            // Adaptive adjustment
            self.temperature_params.classical_temperature = temp_mean;
            self.temperature_params.quantum_temperature = 1.0 + temp_std;
        }
        
        // Update quantum enhancement factor
        if self.quantum_advantage_history.len() > 10 {
            let recent_advantages: Vec<f64> = self.quantum_advantage_history.iter()
                .rev()
                .take(10)
                .cloned()
                .collect();
            
            let advantage_mean = recent_advantages.iter().sum::<f64>() / recent_advantages.len() as f64;
            self.quantum_enhancement_factor = 1.0 + advantage_mean.max(0.0);
        }
        
        Ok(())
    }

    /// Update with new training data
    pub async fn update(&mut self, training_data: &DMatrix<f64>, targets: &DVector<f64>) -> Result<(), TENGRIError> {
        // Add training samples to calibration data
        for (i, &target) in targets.iter().enumerate() {
            if i < training_data.ncols() {
                let input = training_data.column(i);
                let prediction = input.mean(); // Simple prediction for calibration
                let quantum_confidence = self.quantum_state.fidelity;
                let quantum_fidelity = self.quantum_state.fidelity;
                
                self.calibration_data.add_sample(prediction, target, quantum_confidence, quantum_fidelity);
            }
        }
        
        // Recalibrate if needed
        let time_since_calibration = Utc::now().signed_duration_since(self.last_calibration_time);
        if time_since_calibration.num_hours() >= self.calibration_interval_hours {
            self.recalibrate().await?;
            self.last_calibration_time = Utc::now();
        }
        
        Ok(())
    }

    /// Recalibrate the quantum ATS-CP system
    async fn recalibrate(&mut self) -> Result<(), TENGRIError> {
        if self.calibration_data.len() < 10 {
            return Ok(()); // Need sufficient data for calibration
        }
        
        // Compute calibration error
        let errors: Vec<f64> = self.calibration_data.predictions.iter()
            .zip(self.calibration_data.actual_values.iter())
            .map(|(pred, actual)| (pred - actual).abs())
            .collect();
        
        self.calibration_error = errors.iter().sum::<f64>() / errors.len() as f64;
        
        // Compute coverage probability
        let within_bounds = errors.iter()
            .filter(|&&error| error <= self.calibration_error * 2.0)
            .count();
        
        self.coverage_probability = within_bounds as f64 / errors.len() as f64;
        
        // Adjust conformal parameters based on calibration
        if self.coverage_probability < 0.9 {
            self.conformal_params.quantum_correction_factor *= 1.1;
        } else if self.coverage_probability > 0.98 {
            self.conformal_params.quantum_correction_factor *= 0.9;
        }
        
        tracing::info!(
            "Quantum ATS-CP recalibration: error={:.4}, coverage={:.3}",
            self.calibration_error,
            self.coverage_probability
        );
        
        Ok(())
    }

    /// Create quantum circuit for uncertainty quantification
    fn create_uncertainty_circuit(n_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(n_qubits);
        
        // Create superposition for uncertainty encoding
        for i in 0..n_qubits {
            circuit.add_gate(QuantumGateOp::H(i));
        }
        
        // Add controlled rotations for uncertainty patterns
        for i in 0..n_qubits-1 {
            circuit.add_gate(QuantumGateOp::CNOT(i, i+1));
            circuit.add_gate(QuantumGateOp::RZ(i+1, std::f64::consts::PI / 4.0));
        }
        
        circuit
    }

    /// Create quantum circuit for temperature scaling
    fn create_temperature_circuit(n_qubits: usize) -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(n_qubits);
        
        // Temperature encoding through rotations
        for i in 0..n_qubits {
            let angle = (i as f64 + 1.0) * std::f64::consts::PI / (n_qubits as f64);
            circuit.add_gate(QuantumGateOp::RY(i, angle));
        }
        
        // Add entanglement for temperature correlations
        for i in 0..n_qubits-1 {
            circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % n_qubits));
        }
        
        circuit
    }

    /// Get quantum ATS-CP metrics
    pub async fn get_metrics(&self) -> Result<QuantumATSMetrics, TENGRIError> {
        Ok(QuantumATSMetrics {
            calibration_error: self.calibration_error,
            coverage_probability: self.coverage_probability,
            quantum_enhancement_factor: self.quantum_enhancement_factor,
            quantum_fidelity: self.quantum_state.fidelity,
            adaptive_temperature: self.adaptive_temperature_history.back().copied().unwrap_or(1.0),
        })
    }
}

/// Quantum ATS-CP metrics
#[derive(Debug, Clone)]
pub struct QuantumATSMetrics {
    pub calibration_error: f64,
    pub coverage_probability: f64,
    pub quantum_enhancement_factor: f64,
    pub quantum_fidelity: f64,
    pub adaptive_temperature: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_quantum_ats_creation() {
        let qats = QuantumATS::new(0.95).await.unwrap();
        assert_abs_diff_eq!(qats.uncertainty_threshold, 0.95);
        assert_eq!(qats.quantum_state.n_qubits, 8);
    }

    #[tokio::test]
    async fn test_quantum_ats_combine_predictions() {
        let mut qats = QuantumATS::new(0.95).await.unwrap();
        
        let predictions = vec![0.5, 0.6, 0.55];
        let quantum_uncertainty = QuantumUncertainty {
            entropy: 0.1,
            variance: 0.05,
            confidence_interval: (0.45, 0.65),
        };
        
        let result = qats.combine_predictions(&predictions, &quantum_uncertainty).await.unwrap();
        
        assert!(result.value >= 0.0 && result.value <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.uncertainty_bounds.0 < result.uncertainty_bounds.1);
    }

    #[tokio::test]
    async fn test_quantum_ats_temperature_scaling() {
        let qats = QuantumATS::new(0.95).await.unwrap();
        
        let predictions = vec![0.5, 0.6, 0.55];
        let temperature = 2.0;
        
        let scaled = qats.apply_temperature_scaling(&predictions, temperature).unwrap();
        
        assert_eq!(scaled.len(), predictions.len());
        assert_abs_diff_eq!(scaled[0], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[1], 0.30, epsilon = 1e-10);
        assert_abs_diff_eq!(scaled[2], 0.275, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_ats_ensemble_prediction() {
        let qats = QuantumATS::new(0.95).await.unwrap();
        
        let predictions = vec![0.5, 0.6, 0.55];
        let ensemble = qats.compute_ensemble_prediction(&predictions).unwrap();
        
        assert!(ensemble >= 0.45 && ensemble <= 0.65);
    }

    #[tokio::test]
    async fn test_quantum_ats_metrics() {
        let qats = QuantumATS::new(0.95).await.unwrap();
        let metrics = qats.get_metrics().await.unwrap();
        
        assert!(metrics.calibration_error >= 0.0);
        assert!(metrics.coverage_probability >= 0.0 && metrics.coverage_probability <= 1.0);
        assert!(metrics.quantum_enhancement_factor >= 0.0);
        assert!(metrics.quantum_fidelity >= 0.0 && metrics.quantum_fidelity <= 1.0);
        assert!(metrics.adaptive_temperature > 0.0);
    }

    #[tokio::test]
    async fn test_quantum_ats_calibration_data() {
        let mut calibration_data = QuantumCalibrationData::new();
        
        calibration_data.add_sample(0.5, 0.55, 0.8, 0.9);
        calibration_data.add_sample(0.6, 0.58, 0.7, 0.85);
        
        assert_eq!(calibration_data.len(), 2);
        assert!(!calibration_data.is_empty());
        assert_abs_diff_eq!(calibration_data.predictions[0], 0.5);
        assert_abs_diff_eq!(calibration_data.actual_values[1], 0.58);
    }
}