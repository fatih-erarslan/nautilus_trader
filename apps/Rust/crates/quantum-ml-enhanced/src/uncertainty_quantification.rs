//! Uncertainty Quantification for Quantum ML
//! 
//! Advanced uncertainty estimation using quantum principles, Monte Carlo methods,
//! and ensemble techniques with high-performance optimizations

use std::collections::HashMap;
use nalgebra::{DVector};
use wide::{f64x4}; // SIMD support
use rayon::prelude::*; // Parallel processing
use rand::Rng; // Add Rng trait
use rand_distr::Distribution; // Add Distribution trait
use crate::{QuantumState, QuantumMLError, quantum_gates::QuantumGates};

/// Uncertainty quantification configuration
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    pub monte_carlo_samples: usize,
    pub ensemble_size: usize,
    pub confidence_levels: Vec<f64>,
    pub quantum_noise_level: f64,
    pub epistemic_weight: f64,
    pub aleatoric_weight: f64,
    pub simd_batch_size: usize,
    pub parallel_threshold: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            monte_carlo_samples: 1000,
            ensemble_size: 10,
            confidence_levels: vec![0.68, 0.95, 0.99],
            quantum_noise_level: 0.01,
            epistemic_weight: 0.6,
            aleatoric_weight: 0.4,
            simd_batch_size: 8,
            parallel_threshold: 100,
        }
    }
}

/// Uncertainty components
#[derive(Debug, Clone)]
pub struct UncertaintyComponents {
    pub aleatoric: f64,    // Data/observation uncertainty
    pub epistemic: f64,    // Model/knowledge uncertainty
    pub quantum: f64,      // Quantum superposition uncertainty
    pub total: f64,        // Combined uncertainty
}

/// Uncertainty estimation result
#[derive(Debug, Clone)]
pub struct UncertaintyEstimation {
    pub components: UncertaintyComponents,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub prediction_distribution: Vec<f64>,
    pub entropy: f64,
    pub mutual_information: f64,
}

/// Monte Carlo sample
#[derive(Debug, Clone)]
struct MCSample {
    prediction: f64,
    quantum_state: QuantumState,
    noise_realization: f64,
}

/// Uncertainty quantifier
pub struct UncertaintyQuantifier {
    config: UncertaintyConfig,
    
    // Quantum uncertainty state
    quantum_uncertainty_state: QuantumState,
    
    // Historical uncertainty data for calibration
    historical_uncertainties: Vec<f64>,
    historical_errors: Vec<f64>,
    
    // Performance optimization
    simd_buffer: Vec<f64>,
    parallel_buffer: Vec<Vec<f64>>,
    
    // Uncertainty model parameters
    epistemic_model_params: DVector<f64>,
    aleatoric_model_params: DVector<f64>,
    
    // Calibration metrics
    calibration_accuracy: f64,
    coverage_statistics: HashMap<String, f64>,
}

impl UncertaintyQuantifier {
    /// Create new uncertainty quantifier
    pub async fn new() -> Result<Self, QuantumMLError> {
        let config = UncertaintyConfig::default();
        
        // Initialize quantum state for uncertainty modeling
        let quantum_uncertainty_state = QuantumState::new(4); // 4 qubits for uncertainty
        
        // Initialize model parameters
        let epistemic_model_params = DVector::from_fn(10, |_, _| rand::random::<f64>() * 0.1);
        let aleatoric_model_params = DVector::from_fn(10, |_, _| rand::random::<f64>() * 0.1);
        
        // Initialize performance buffers
        let simd_buffer = Vec::with_capacity(config.simd_batch_size);
        let parallel_buffer = vec![Vec::new(); rayon::current_num_threads()];
        
        // Initialize coverage statistics
        let mut coverage_statistics = HashMap::new();
        for &level in &config.confidence_levels {
            coverage_statistics.insert(format!("{:.1}%", level * 100.0), 0.0);
        }
        
        Ok(Self {
            config,
            quantum_uncertainty_state,
            historical_uncertainties: Vec::new(),
            historical_errors: Vec::new(),
            simd_buffer,
            parallel_buffer,
            epistemic_model_params,
            aleatoric_model_params,
            calibration_accuracy: 0.0,
            coverage_statistics,
        })
    }
    
    /// Estimate uncertainty for a single prediction
    pub async fn estimate_uncertainty(
        &mut self,
        prediction: f64,
        input_features: &DVector<f64>,
        quantum_state: Option<&QuantumState>,
    ) -> Result<UncertaintyEstimation, QuantumMLError> {
        // Update quantum uncertainty state
        self.update_quantum_uncertainty_state(input_features, quantum_state).await?;
        
        // Estimate uncertainty components
        let aleatoric = self.estimate_aleatoric_uncertainty(input_features, prediction).await?;
        let epistemic = self.estimate_epistemic_uncertainty(input_features, prediction).await?;
        let quantum = self.estimate_quantum_uncertainty().await?;
        
        // Combine uncertainties
        let total = self.combine_uncertainties(aleatoric, epistemic, quantum);
        
        let components = UncertaintyComponents {
            aleatoric,
            epistemic,
            quantum,
            total,
        };
        
        // Generate prediction distribution using Monte Carlo
        let distribution = self.generate_prediction_distribution(prediction, &components).await?;
        
        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&distribution);
        
        // Calculate information-theoretic measures
        let entropy = self.calculate_entropy(&distribution);
        let mutual_information = self.calculate_mutual_information(&distribution);
        
        Ok(UncertaintyEstimation {
            components,
            confidence_intervals,
            prediction_distribution: distribution,
            entropy,
            mutual_information,
        })
    }
    
    /// Update quantum uncertainty state
    async fn update_quantum_uncertainty_state(
        &mut self,
        features: &DVector<f64>,
        quantum_state: Option<&QuantumState>,
    ) -> Result<(), QuantumMLError> {
        // Encode features into quantum state
        let feature_subset: Vec<f64> = features.iter()
            .take(self.quantum_uncertainty_state.n_qubits)
            .map(|&x| x * 0.1) // Scale for quantum encoding
            .collect();
        
        if !feature_subset.is_empty() {
            QuantumGates::create_feature_map(&mut self.quantum_uncertainty_state, &feature_subset)?;
        }
        
        // Apply entanglement if input quantum state is provided
        if let Some(input_state) = quantum_state {
            // Create entanglement between uncertainty state and input state
            let entanglement_strength = 0.1;
            let angle = input_state.entanglement_measure * entanglement_strength;
            
            if self.quantum_uncertainty_state.n_qubits >= 2 {
                let ry_gate = QuantumGates::ry(angle);
                QuantumGates::apply_single_qubit_gate(&mut self.quantum_uncertainty_state, &ry_gate, 0)?;
                
                let cnot_gate = QuantumGates::cnot();
                QuantumGates::apply_two_qubit_gate(&mut self.quantum_uncertainty_state, &cnot_gate, 0, 1)?;
            }
        }
        
        // Apply noise for quantum uncertainty modeling
        self.quantum_uncertainty_state.apply_decoherence(self.config.quantum_noise_level, 1.0);
        
        Ok(())
    }
    
    /// Estimate aleatoric (data) uncertainty
    async fn estimate_aleatoric_uncertainty(
        &self,
        features: &DVector<f64>,
        prediction: f64,
    ) -> Result<f64, QuantumMLError> {
        // Model aleatoric uncertainty as feature-dependent noise
        let feature_norm = features.norm();
        let base_noise = self.config.quantum_noise_level;
        
        // Use learned parameters to estimate data-dependent uncertainty
        let feature_effect = if features.len() <= self.aleatoric_model_params.len() {
            features.dot(&self.aleatoric_model_params.rows(0, features.len()).into_owned())
        } else {
            let truncated_features = features.rows(0, self.aleatoric_model_params.len()).into_owned();
            truncated_features.dot(&self.aleatoric_model_params)
        };
        
        let aleatoric = base_noise * (1.0 + feature_norm * 0.1) * (1.0 + feature_effect.abs() * 0.1);
        
        // Add prediction-dependent scaling
        let prediction_scale = (prediction.abs() * 0.01).max(0.001);
        
        Ok(aleatoric * prediction_scale)
    }
    
    /// Estimate epistemic (model) uncertainty using ensemble approximation
    async fn estimate_epistemic_uncertainty(
        &self,
        features: &DVector<f64>,
        prediction: f64,
    ) -> Result<f64, QuantumMLError> {
        // Simulate ensemble predictions with parameter perturbations
        let ensemble_predictions = if features.len() >= self.config.parallel_threshold {
            // Use parallel processing for large feature vectors
            self.parallel_ensemble_predictions(features, prediction).await?
        } else {
            // Use SIMD for smaller feature vectors
            self.simd_ensemble_predictions(features, prediction).await?
        };
        
        // Calculate variance of ensemble predictions as epistemic uncertainty
        let mean_prediction = ensemble_predictions.iter().sum::<f64>() / ensemble_predictions.len() as f64;
        let variance = ensemble_predictions.iter()
            .map(|&pred| (pred - mean_prediction).powi(2))
            .sum::<f64>() / ensemble_predictions.len() as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Generate ensemble predictions using parallel processing
    async fn parallel_ensemble_predictions(
        &self,
        features: &DVector<f64>,
        base_prediction: f64,
    ) -> Result<Vec<f64>, QuantumMLError> {
        // Use parallel processing for ensemble generation
        let predictions: Vec<f64> = (0..self.config.ensemble_size)
            .into_par_iter()
            .map(|i| {
                // Simulate model uncertainty by perturbing parameters
                let param_noise = (i as f64 / self.config.ensemble_size as f64 - 0.5) * 0.1;
                let feature_perturbation = features.map(|x| x * (1.0 + param_noise));
                
                // Simplified ensemble prediction
                let feature_sum = feature_perturbation.sum();
                base_prediction * (1.0 + feature_sum * 0.001 + param_noise)
            })
            .collect();
        
        Ok(predictions)
    }
    
    /// Generate ensemble predictions using SIMD
    async fn simd_ensemble_predictions(
        &self,
        features: &DVector<f64>,
        base_prediction: f64,
    ) -> Result<Vec<f64>, QuantumMLError> {
        let mut predictions = Vec::with_capacity(self.config.ensemble_size);
        
        // Process ensemble members in SIMD batches
        for chunk_start in (0..self.config.ensemble_size).step_by(4) {
            let chunk_end = (chunk_start + 4).min(self.config.ensemble_size);
            
            // Create SIMD vectors for parameter perturbations
            let indices = [
                chunk_start as f64,
                (chunk_start + 1) as f64,
                (chunk_start + 2) as f64,
                (chunk_start + 3) as f64,
            ];
            
            let indices_vec = f64x4::from(indices);
            let ensemble_size_vec = f64x4::splat(self.config.ensemble_size as f64);
            let base_pred_vec = f64x4::splat(base_prediction);
            
            // Calculate parameter noise
            let param_noise = (indices_vec / ensemble_size_vec - f64x4::splat(0.5)) * f64x4::splat(0.1);
            
            // Calculate feature perturbation effect (simplified)
            let feature_sum = features.sum();
            let feature_effect = f64x4::splat(feature_sum * 0.001);
            
            // Generate predictions
            let pred_vec = base_pred_vec * (f64x4::splat(1.0) + feature_effect + param_noise);
            
            // Extract results
            let pred_array: [f64; 4] = pred_vec.into();
            for i in 0..(chunk_end - chunk_start) {
                predictions.push(pred_array[i]);
            }
        }
        
        Ok(predictions)
    }
    
    /// Estimate quantum uncertainty
    async fn estimate_quantum_uncertainty(&self) -> Result<f64, QuantumMLError> {
        // Quantum uncertainty from superposition and entanglement
        let entanglement_entropy = self.quantum_uncertainty_state.entanglement_measure;
        
        // Calculate amplitude distribution uncertainty
        let amplitude_uncertainties: Vec<f64> = self.quantum_uncertainty_state.amplitudes.iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        let mean_amplitude = amplitude_uncertainties.iter().sum::<f64>() / amplitude_uncertainties.len() as f64;
        let amplitude_variance = amplitude_uncertainties.iter()
            .map(|&amp| (amp - mean_amplitude).powi(2))
            .sum::<f64>() / amplitude_uncertainties.len() as f64;
        
        // Combine quantum uncertainty sources
        let quantum_uncertainty = entanglement_entropy * 0.5 + amplitude_variance.sqrt() * 0.5;
        
        Ok(quantum_uncertainty)
    }
    
    /// Combine different uncertainty components
    fn combine_uncertainties(&self, aleatoric: f64, epistemic: f64, quantum: f64) -> f64 {
        // Weighted combination of uncertainty sources
        let weighted_sum = self.config.aleatoric_weight * aleatoric.powi(2) +
                          self.config.epistemic_weight * epistemic.powi(2) +
                          0.2 * quantum.powi(2); // Fixed weight for quantum uncertainty
        
        weighted_sum.sqrt()
    }
    
    /// Generate prediction distribution using Monte Carlo sampling
    async fn generate_prediction_distribution(
        &self,
        base_prediction: f64,
        components: &UncertaintyComponents,
    ) -> Result<Vec<f64>, QuantumMLError> {
        let mut distribution = Vec::with_capacity(self.config.monte_carlo_samples);
        let mut rng = rand::thread_rng();
        
        // Generate samples using different uncertainty sources
        for _ in 0..self.config.monte_carlo_samples {
            // Sample aleatoric noise (Gaussian)
            let aleatoric_noise = rand_distr::Normal::new(0.0, components.aleatoric)
                .map_err(|e| QuantumMLError::UncertaintyQuantificationFailed {
                    reason: format!("Failed to create normal distribution: {}", e),
                })?
                .sample(&mut rng);
            
            // Sample epistemic uncertainty (Gaussian)
            let epistemic_noise = rand_distr::Normal::new(0.0, components.epistemic)
                .map_err(|e| QuantumMLError::UncertaintyQuantificationFailed {
                    reason: format!("Failed to create normal distribution: {}", e),
                })?
                .sample(&mut rng);
            
            // Sample quantum uncertainty (non-Gaussian, using quantum state)
            let quantum_noise = self.sample_quantum_uncertainty(components.quantum);
            
            // Combine samples
            let sample = base_prediction + aleatoric_noise + epistemic_noise + quantum_noise;
            distribution.push(sample);
        }
        
        Ok(distribution)
    }
    
    /// Sample from quantum uncertainty distribution
    fn sample_quantum_uncertainty(&self, quantum_uncertainty: f64) -> f64 {
        // Sample from quantum amplitude distribution
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.quantum_uncertainty_state.amplitudes.len());
        let amplitude = self.quantum_uncertainty_state.amplitudes[idx];
        
        // Convert complex amplitude to real sample
        let phase_contribution = amplitude.arg().sin();
        let magnitude_contribution = amplitude.norm() - 0.5; // Center around 0
        
        quantum_uncertainty * (phase_contribution + magnitude_contribution)
    }
    
    /// Calculate confidence intervals from distribution
    fn calculate_confidence_intervals(&self, distribution: &[f64]) -> HashMap<String, (f64, f64)> {
        let mut sorted_dist = distribution.to_vec();
        sorted_dist.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut intervals = HashMap::new();
        
        for &confidence_level in &self.config.confidence_levels {
            let alpha = 1.0 - confidence_level;
            let lower_idx = ((alpha / 2.0) * sorted_dist.len() as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * sorted_dist.len() as f64) as usize;
            
            let lower_bound = sorted_dist.get(lower_idx).copied().unwrap_or(sorted_dist[0]);
            let upper_bound = sorted_dist.get(upper_idx).copied()
                .unwrap_or(sorted_dist[sorted_dist.len() - 1]);
            
            intervals.insert(
                format!("{:.1}%", confidence_level * 100.0),
                (lower_bound, upper_bound),
            );
        }
        
        intervals
    }
    
    /// Calculate entropy of prediction distribution
    fn calculate_entropy(&self, distribution: &[f64]) -> f64 {
        // Discretize distribution for entropy calculation
        let n_bins = 50;
        let min_val = distribution.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = distribution.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let bin_width = (max_val - min_val) / n_bins as f64;
        
        if bin_width <= 0.0 {
            return 0.0;
        }
        
        // Create histogram
        let mut histogram = vec![0; n_bins];
        for &value in distribution {
            let bin_idx = ((value - min_val) / bin_width) as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            histogram[bin_idx] += 1;
        }
        
        // Calculate entropy
        let total_samples = distribution.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &histogram {
            if count > 0 {
                let probability = count as f64 / total_samples;
                entropy -= probability * probability.ln();
            }
        }
        
        entropy
    }
    
    /// Calculate mutual information (simplified)
    fn calculate_mutual_information(&self, distribution: &[f64]) -> f64 {
        // Simplified mutual information calculation
        // In practice, this would require joint distributions
        let entropy = self.calculate_entropy(distribution);
        let max_entropy = (distribution.len() as f64).ln();
        
        // Return normalized mutual information
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }
    
    /// Combine multiple uncertainties from ensemble predictions
    pub async fn combine_ensemble_uncertainties(&self, uncertainties: &[f64]) -> Result<f64, QuantumMLError> {
        if uncertainties.is_empty() {
            return Ok(0.0);
        }
        
        // Use different combination strategies based on size
        let combined = if uncertainties.len() <= 4 {
            // Use SIMD for small arrays
            self.simd_combine_uncertainties(uncertainties)?
        } else {
            // Use parallel processing for larger arrays
            self.parallel_combine_uncertainties(uncertainties)?
        };
        
        Ok(combined)
    }
    
    /// Combine uncertainties using SIMD
    fn simd_combine_uncertainties(&self, uncertainties: &[f64]) -> Result<f64, QuantumMLError> {
        if uncertainties.is_empty() {
            return Ok(0.0);
        }
        
        // Pad to multiple of 4 for SIMD
        let mut padded = uncertainties.to_vec();
        while padded.len() % 4 != 0 {
            padded.push(0.0);
        }
        
        let mut sum_squares = f64x4::splat(0.0);
        
        for chunk in padded.chunks_exact(4) {
            let uncertainty_vec = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sum_squares += uncertainty_vec * uncertainty_vec;
        }
        
        // Sum the SIMD vector elements (using array access instead of extract)
        let array: [f64; 4] = sum_squares.into();
        let total_sum = array[0] + array[1] + array[2] + array[3];
        
        // Return RMS combination (root mean square)
        Ok((total_sum / uncertainties.len() as f64).sqrt())
    }
    
    /// Combine uncertainties using parallel processing
    fn parallel_combine_uncertainties(&self, uncertainties: &[f64]) -> Result<f64, QuantumMLError> {
        let sum_squares: f64 = uncertainties
            .par_iter()
            .map(|&u| u * u)
            .sum();
        
        Ok((sum_squares / uncertainties.len() as f64).sqrt())
    }
    
    /// Update calibration statistics
    pub fn update_calibration(&mut self, predicted_uncertainty: f64, actual_error: f64) {
        self.historical_uncertainties.push(predicted_uncertainty);
        self.historical_errors.push(actual_error);
        
        // Keep only recent samples
        if self.historical_uncertainties.len() > 10000 {
            self.historical_uncertainties.remove(0);
            self.historical_errors.remove(0);
        }
        
        // Update calibration accuracy
        if self.historical_uncertainties.len() >= 100 {
            self.update_calibration_metrics();
        }
    }
    
    /// Update calibration metrics
    fn update_calibration_metrics(&mut self) {
        let n = self.historical_uncertainties.len();
        let mut accurate_predictions = 0;
        
        for i in 0..n {
            let predicted_uncertainty = self.historical_uncertainties[i];
            let actual_error = self.historical_errors[i];
            
            // Check if actual error is within predicted uncertainty bounds
            if actual_error <= predicted_uncertainty * 2.0 { // 2-sigma confidence
                accurate_predictions += 1;
            }
        }
        
        self.calibration_accuracy = accurate_predictions as f64 / n as f64;
        
        // Update coverage statistics for different confidence levels
        for &level in &self.config.confidence_levels {
            let mut covered = 0;
            let threshold = predicted_uncertainty_for_level(level);
            
            for i in 0..n {
                if self.historical_errors[i] <= self.historical_uncertainties[i] * threshold {
                    covered += 1;
                }
            }
            
            let coverage = covered as f64 / n as f64;
            self.coverage_statistics.insert(format!("{:.1}%", level * 100.0), coverage);
        }
    }
    
    /// Get calibration quality metrics
    pub fn get_calibration_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("calibration_accuracy".to_string(), self.calibration_accuracy);
        
        for (level, coverage) in &self.coverage_statistics {
            metrics.insert(format!("coverage_{}", level), *coverage);
        }
        
        metrics
    }
}

/// Helper function to map confidence level to uncertainty threshold
fn predicted_uncertainty_for_level(confidence_level: f64) -> f64 {
    match confidence_level {
        x if x >= 0.99 => 3.0,  // 3-sigma
        x if x >= 0.95 => 2.0,  // 2-sigma
        x if x >= 0.68 => 1.0,  // 1-sigma
        _ => 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_uncertainty_quantifier_creation() {
        let quantifier = UncertaintyQuantifier::new().await;
        assert!(quantifier.is_ok());
    }

    #[tokio::test]
    async fn test_uncertainty_estimation() {
        let mut quantifier = UncertaintyQuantifier::new().await.unwrap();
        let features = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        let estimation = quantifier.estimate_uncertainty(1.5, &features, None).await;
        assert!(estimation.is_ok());
        
        let est = estimation.unwrap();
        assert!(est.components.total > 0.0);
        assert!(est.components.aleatoric >= 0.0);
        assert!(est.components.epistemic >= 0.0);
        assert!(est.components.quantum >= 0.0);
    }

    #[tokio::test]
    async fn test_simd_uncertainty_combination() {
        let quantifier = UncertaintyQuantifier::new().await.unwrap();
        let uncertainties = vec![0.1, 0.2, 0.15, 0.25];
        
        let combined = quantifier.simd_combine_uncertainties(&uncertainties).unwrap();
        
        // Should be RMS combination
        let expected = (uncertainties.iter().map(|&u| u * u).sum::<f64>() / uncertainties.len() as f64).sqrt();
        assert_abs_diff_eq!(combined, expected, epsilon = 1e-10);
    }

    #[tokio::test]
    async fn test_confidence_intervals() {
        let quantifier = UncertaintyQuantifier::new().await.unwrap();
        let distribution = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        
        let intervals = quantifier.calculate_confidence_intervals(&distribution);
        
        assert!(intervals.contains_key("95.0%"));
        let (lower, upper) = intervals["95.0%"];
        assert!(lower < upper);
        assert!(lower >= 1.0);
        assert!(upper <= 5.0);
    }

    #[tokio::test]
    async fn test_entropy_calculation() {
        let quantifier = UncertaintyQuantifier::new().await.unwrap();
        let uniform_dist = vec![1.0; 100]; // Uniform distribution should have low entropy
        let varied_dist: Vec<f64> = (0..100).map(|i| i as f64).collect(); // Varied distribution
        
        let uniform_entropy = quantifier.calculate_entropy(&uniform_dist);
        let varied_entropy = quantifier.calculate_entropy(&varied_dist);
        
        assert!(varied_entropy > uniform_entropy);
    }

    #[tokio::test]
    async fn test_calibration_update() {
        let mut quantifier = UncertaintyQuantifier::new().await.unwrap();
        
        // Add some calibration data
        for i in 0..150 {
            let predicted_uncertainty = 0.1 + (i as f64) * 0.01;
            let actual_error = 0.05 + (i as f64) * 0.005;
            quantifier.update_calibration(predicted_uncertainty, actual_error);
        }
        
        let metrics = quantifier.get_calibration_metrics();
        assert!(metrics.contains_key("calibration_accuracy"));
        assert!(metrics["calibration_accuracy"] >= 0.0);
        assert!(metrics["calibration_accuracy"] <= 1.0);
    }
}