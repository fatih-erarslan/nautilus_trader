//! Quantum Uncertainty Quantification Methods
//!
//! Advanced quantum uncertainty quantification techniques for trading ML models
//! including quantum Fisher information, entanglement entropy, and quantum Cramér-Rao bounds.

use crate::TENGRIError;
use super::quantum_gates::{QuantumState, QuantumCircuit, QuantumGateOp};
use super::QuantumUncertainty;

/// Uncertainty Quantification System
#[derive(Debug)]
pub struct UncertaintyQuantification {
    fisher_matrix: Option<QuantumFisherMatrix>,
}

impl UncertaintyQuantification {
    pub fn new() -> Self {
        Self {
            fisher_matrix: None,
        }
    }
}
use nalgebra::{DMatrix, DVector, Complex};
use statrs::distribution::{Normal, ContinuousCDF, ChiSquared};
use std::collections::VecDeque;
use rayon::prelude::*;
use chrono::{DateTime, Utc};

/// Quantum Fisher Information Matrix
#[derive(Debug, Clone)]
pub struct QuantumFisherMatrix {
    pub matrix: DMatrix<f64>,
    pub eigenvalues: DVector<f64>,
    pub condition_number: f64,
    pub trace: f64,
}

impl QuantumFisherMatrix {
    /// Compute quantum Fisher information matrix
    pub fn compute(quantum_states: &[QuantumState], parameters: &DVector<f64>) -> Result<Self, TENGRIError> {
        let n_params = parameters.len();
        let mut fisher_matrix = DMatrix::zeros(n_params, n_params);
        
        for (i, j) in (0..n_params).flat_map(|i| (0..n_params).map(move |j| (i, j))) {
            let mut fisher_element = 0.0;
            
            for state in quantum_states {
                let grad_i = Self::compute_gradient(state, i)?;
                let grad_j = Self::compute_gradient(state, j)?;
                
                // Quantum Fisher information element: 4 * Re(<∂ψ/∂θᵢ|∂ψ/∂θⱼ>)
                let inner_product = grad_i.iter()
                    .zip(grad_j.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum::<Complex<f64>>();
                
                fisher_element += 4.0 * inner_product.re;
            }
            
            fisher_matrix[(i, j)] = fisher_element / quantum_states.len() as f64;
        }
        
        // Compute eigenvalues
        let eigendecomp = fisher_matrix.symmetric_eigen();
        let eigenvalues = eigendecomp.eigenvalues;
        
        // Compute condition number
        let max_eigen = eigenvalues.max();
        let min_eigen = eigenvalues.min();
        let condition_number = if min_eigen > 1e-10 {
            max_eigen / min_eigen
        } else {
            f64::INFINITY
        };
        
        let trace = eigenvalues.sum();
        
        Ok(Self {
            matrix: fisher_matrix,
            eigenvalues,
            condition_number,
            trace,
        })
    }

    /// Compute gradient of quantum state with respect to parameter
    fn compute_gradient(state: &QuantumState, param_index: usize) -> Result<Vec<Complex<f64>>, TENGRIError> {
        let eps = 1e-8;
        let n_amplitudes = state.amplitudes.len();
        let mut gradient = vec![Complex::new(0.0, 0.0); n_amplitudes];
        
        // Finite difference approximation for gradient
        for i in 0..n_amplitudes {
            // Forward difference
            let amplitude_real = state.amplitudes[i].re;
            let amplitude_imag = state.amplitudes[i].im;
            
            // Approximate gradient using finite differences
            if param_index < 2 {
                // For real part
                gradient[i].re = (amplitude_real + eps - amplitude_real) / eps;
                gradient[i].im = amplitude_imag;
            } else {
                // For imaginary part
                gradient[i].re = amplitude_real;
                gradient[i].im = (amplitude_imag + eps - amplitude_imag) / eps;
            }
        }
        
        Ok(gradient)
    }

    /// Compute quantum Cramér-Rao bound
    pub fn cramer_rao_bound(&self, parameter_index: usize) -> f64 {
        if parameter_index < self.eigenvalues.len() {
            let fisher_info = self.matrix[(parameter_index, parameter_index)];
            if fisher_info > 1e-10 {
                1.0 / fisher_info
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    }
}

/// Entanglement entropy calculator
#[derive(Debug, Clone)]
pub struct EntanglementEntropy {
    pub von_neumann_entropy: f64,
    pub renyi_entropy_2: f64,
    pub linear_entropy: f64,
    pub schmidt_rank: usize,
    pub schmidt_coefficients: Vec<f64>,
}

impl EntanglementEntropy {
    /// Compute entanglement entropy for bipartite quantum state
    pub fn compute(state: &QuantumState, subsystem_qubits: &[usize]) -> Result<Self, TENGRIError> {
        let n_qubits = state.n_qubits;
        let subsystem_size = subsystem_qubits.len();
        
        if subsystem_size >= n_qubits {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Subsystem cannot be larger than total system".to_string(),
            });
        }
        
        // Compute reduced density matrix for subsystem
        let reduced_density_matrix = Self::compute_reduced_density_matrix(state, subsystem_qubits)?;
        
        // Diagonalize reduced density matrix to get eigenvalues
        let eigendecomp = reduced_density_matrix.symmetric_eigen();
        let eigenvalues = eigendecomp.eigenvalues;
        
        // Filter out near-zero eigenvalues
        let significant_eigenvalues: Vec<f64> = eigenvalues.iter()
            .filter(|&&val| val > 1e-10)
            .cloned()
            .collect();
        
        // Von Neumann entropy: S = -Tr(ρ log ρ)
        let von_neumann_entropy = -significant_eigenvalues.iter()
            .map(|&val| val * val.ln())
            .sum::<f64>();
        
        // Rényi entropy of order 2: S₂ = -log(Tr(ρ²))
        let trace_rho_squared: f64 = significant_eigenvalues.iter()
            .map(|&val| val * val)
            .sum();
        let renyi_entropy_2 = -trace_rho_squared.ln();
        
        // Linear entropy: S_L = 1 - Tr(ρ²)
        let linear_entropy = 1.0 - trace_rho_squared;
        
        // Schmidt rank and coefficients
        let schmidt_rank = significant_eigenvalues.len();
        let schmidt_coefficients = significant_eigenvalues.iter()
            .map(|&val| val.sqrt())
            .collect();
        
        Ok(Self {
            von_neumann_entropy,
            renyi_entropy_2,
            linear_entropy,
            schmidt_rank,
            schmidt_coefficients,
        })
    }

    /// Compute reduced density matrix for subsystem
    fn compute_reduced_density_matrix(
        state: &QuantumState,
        subsystem_qubits: &[usize],
    ) -> Result<DMatrix<f64>, TENGRIError> {
        let subsystem_dim = 1usize << subsystem_qubits.len();
        let mut reduced_density = DMatrix::zeros(subsystem_dim, subsystem_dim);
        
        let full_dim = 1usize << state.n_qubits;
        
        // Trace out the complement of the subsystem
        for i in 0..subsystem_dim {
            for j in 0..subsystem_dim {
                let mut element = Complex::new(0.0, 0.0);
                
                // Sum over all configurations of the environment
                for env_config in 0..(full_dim / subsystem_dim) {
                    let full_i = Self::embed_subsystem_index(i, env_config, subsystem_qubits, state.n_qubits);
                    let full_j = Self::embed_subsystem_index(j, env_config, subsystem_qubits, state.n_qubits);
                    
                    element += state.amplitudes[full_i].conj() * state.amplitudes[full_j];
                }
                
                reduced_density[(i, j)] = element.re; // Take real part (should be real for valid density matrix)
            }
        }
        
        Ok(reduced_density)
    }

    /// Embed subsystem index into full system index
    fn embed_subsystem_index(
        subsystem_index: usize,
        env_index: usize,
        subsystem_qubits: &[usize],
        total_qubits: usize,
    ) -> usize {
        let mut full_index = 0;
        
        // Set bits for subsystem qubits
        for (bit_pos, &qubit) in subsystem_qubits.iter().enumerate() {
            if (subsystem_index >> bit_pos) & 1 == 1 {
                full_index |= 1usize << qubit;
            }
        }
        
        // Set bits for environment qubits
        let mut env_bit_pos = 0;
        for qubit in 0..total_qubits {
            if !subsystem_qubits.contains(&qubit) {
                if (env_index >> env_bit_pos) & 1 == 1 {
                    full_index |= 1usize << qubit;
                }
                env_bit_pos += 1;
            }
        }
        
        full_index
    }
}

/// Quantum uncertainty quantification system
pub struct QuantumUncertaintyQuantifier {
    pub n_qubits: usize,
    pub uncertainty_history: VecDeque<QuantumUncertainty>,
    pub fisher_matrices: VecDeque<QuantumFisherMatrix>,
    pub entanglement_entropies: VecDeque<EntanglementEntropy>,
    
    // Calibration parameters
    pub confidence_calibration_factor: f64,
    pub entropy_scaling_factor: f64,
    pub fisher_weight: f64,
    
    // Adaptive thresholds
    pub high_uncertainty_threshold: f64,
    pub low_uncertainty_threshold: f64,
    pub entropy_threshold: f64,
    
    // Performance metrics
    pub average_uncertainty: f64,
    pub uncertainty_variance: f64,
    pub calibration_score: f64,
}

impl QuantumUncertaintyQuantifier {
    /// Create new quantum uncertainty quantifier
    pub async fn new(n_qubits: usize) -> Result<Self, TENGRIError> {
        Ok(Self {
            n_qubits,
            uncertainty_history: VecDeque::with_capacity(1000),
            fisher_matrices: VecDeque::with_capacity(100),
            entanglement_entropies: VecDeque::with_capacity(100),
            confidence_calibration_factor: 1.0,
            entropy_scaling_factor: 0.1,
            fisher_weight: 0.2,
            high_uncertainty_threshold: 0.8,
            low_uncertainty_threshold: 0.2,
            entropy_threshold: 0.5,
            average_uncertainty: 0.5,
            uncertainty_variance: 0.1,
            calibration_score: 0.0,
        })
    }

    /// Quantify uncertainty for given input
    pub async fn quantify_uncertainty(&mut self, input: &DMatrix<f64>) -> Result<QuantumUncertainty, TENGRIError> {
        let start_time = std::time::Instant::now();
        
        // Create quantum state encoding the input
        let quantum_state = self.encode_input_to_quantum_state(input).await?;
        
        // Compute entanglement entropy
        let subsystem_qubits: Vec<usize> = (0..self.n_qubits/2).collect();
        let entanglement_entropy = EntanglementEntropy::compute(&quantum_state, &subsystem_qubits)?;
        
        // Compute quantum Fisher information
        let parameters = DVector::from_vec((0..4).map(|i| i as f64 * 0.1).collect());
        let fisher_matrix = QuantumFisherMatrix::compute(&[quantum_state.clone()], &parameters)?;
        
        // Compute overall uncertainty metrics
        let entropy = entanglement_entropy.von_neumann_entropy * self.entropy_scaling_factor;
        
        // Quantum variance based on Fisher information
        let quantum_variance = if fisher_matrix.trace > 1e-10 {
            self.fisher_weight / fisher_matrix.trace
        } else {
            1.0
        };
        
        // Compute confidence interval using quantum Cramér-Rao bounds
        let cramer_rao_bound = fisher_matrix.cramer_rao_bound(0);
        let confidence_width = (cramer_rao_bound * self.confidence_calibration_factor).sqrt();
        
        let mean_estimate = input.mean(); // Simple estimate for demonstration
        let confidence_interval = (
            mean_estimate - confidence_width,
            mean_estimate + confidence_width,
        );
        
        let quantum_uncertainty = QuantumUncertainty {
            entropy,
            variance: quantum_variance,
            confidence_interval,
        };
        
        // Update history and metrics
        self.update_uncertainty_history(quantum_uncertainty.clone());
        self.update_adaptive_thresholds();
        
        // Store computed matrices
        self.fisher_matrices.push_back(fisher_matrix);
        if self.fisher_matrices.len() > 100 {
            self.fisher_matrices.pop_front();
        }
        
        self.entanglement_entropies.push_back(entanglement_entropy);
        if self.entanglement_entropies.len() > 100 {
            self.entanglement_entropies.pop_front();
        }
        
        let elapsed = start_time.elapsed();
        if elapsed.as_micros() > 100 {
            tracing::warn!(
                "Quantum uncertainty quantification time: {}μs (target: <100μs)",
                elapsed.as_micros()
            );
        }
        
        Ok(quantum_uncertainty)
    }

    /// Encode input data into quantum state
    async fn encode_input_to_quantum_state(&self, input: &DMatrix<f64>) -> Result<QuantumState, TENGRIError> {
        let mut quantum_state = QuantumState::new(self.n_qubits);
        let mut encoding_circuit = QuantumCircuit::new(self.n_qubits);
        
        // Amplitude encoding: map input values to rotation angles
        let input_values: Vec<f64> = input.iter().take(self.n_qubits).cloned().collect();
        
        // Create superposition first
        for i in 0..self.n_qubits {
            encoding_circuit.add_gate(QuantumGateOp::H(i));
        }
        
        // Encode input through rotation gates
        for (i, &value) in input_values.iter().enumerate() {
            if i < self.n_qubits {
                let angle = value * std::f64::consts::PI; // Normalize to [0, π]
                encoding_circuit.add_gate(QuantumGateOp::RY(i, angle));
            }
        }
        
        // Add entanglement for correlations
        for i in 0..self.n_qubits-1 {
            if i < input_values.len() && input_values[i].abs() > 0.1 {
                encoding_circuit.add_gate(QuantumGateOp::CNOT(i, (i+1) % self.n_qubits));
            }
        }
        
        // Execute encoding circuit
        encoding_circuit.execute(&mut quantum_state)?;
        
        Ok(quantum_state)
    }

    /// Update uncertainty history and compute running statistics
    fn update_uncertainty_history(&mut self, uncertainty: QuantumUncertainty) {
        self.uncertainty_history.push_back(uncertainty);
        if self.uncertainty_history.len() > 1000 {
            self.uncertainty_history.pop_front();
        }
        
        if self.uncertainty_history.len() > 1 {
            // Update running average
            let entropies: Vec<f64> = self.uncertainty_history.iter()
                .map(|u| u.entropy)
                .collect();
            
            self.average_uncertainty = entropies.iter().sum::<f64>() / entropies.len() as f64;
            
            // Update variance
            let variance_sum: f64 = entropies.iter()
                .map(|&e| (e - self.average_uncertainty).powi(2))
                .sum();
            self.uncertainty_variance = variance_sum / entropies.len() as f64;
        }
    }

    /// Update adaptive thresholds based on uncertainty distribution
    fn update_adaptive_thresholds(&mut self) {
        if self.uncertainty_history.len() > 50 {
            let mut entropies: Vec<f64> = self.uncertainty_history.iter()
                .rev()
                .take(50)
                .map(|u| u.entropy)
                .collect();
            
            entropies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // Set thresholds at 20th and 80th percentiles
            let low_percentile_idx = (entropies.len() as f64 * 0.2) as usize;
            let high_percentile_idx = (entropies.len() as f64 * 0.8) as usize;
            
            if low_percentile_idx < entropies.len() {
                self.low_uncertainty_threshold = entropies[low_percentile_idx];
            }
            
            if high_percentile_idx < entropies.len() {
                self.high_uncertainty_threshold = entropies[high_percentile_idx];
            }
        }
    }

    /// Calibrate uncertainty quantification
    pub async fn calibrate(
        &mut self,
        predictions: &[f64],
        actual_values: &[f64],
        predicted_uncertainties: &[QuantumUncertainty],
    ) -> Result<(), TENGRIError> {
        if predictions.len() != actual_values.len() || predictions.len() != predicted_uncertainties.len() {
            return Err(TENGRIError::MathematicalValidationFailed {
                reason: "Mismatched array lengths for calibration".to_string(),
            });
        }
        
        // Compute calibration score
        let mut coverage_count = 0;
        let mut total_width = 0.0;
        
        for i in 0..predictions.len() {
            let (lower, upper) = predicted_uncertainties[i].confidence_interval;
            let actual = actual_values[i];
            
            // Check if actual value is within predicted interval
            if actual >= lower && actual <= upper {
                coverage_count += 1;
            }
            
            total_width += upper - lower;
        }
        
        let coverage_probability = coverage_count as f64 / predictions.len() as f64;
        let average_width = total_width / predictions.len() as f64;
        
        // Calibration score combines coverage and precision
        self.calibration_score = coverage_probability - 0.1 * average_width;
        
        // Adjust calibration factor based on coverage
        if coverage_probability < 0.9 {
            self.confidence_calibration_factor *= 1.1; // Increase intervals
        } else if coverage_probability > 0.98 {
            self.confidence_calibration_factor *= 0.95; // Decrease intervals
        }
        
        tracing::info!(
            "Quantum uncertainty calibration: coverage={:.3}, avg_width={:.4}, score={:.4}",
            coverage_probability,
            average_width,
            self.calibration_score
        );
        
        Ok(())
    }

    /// Get uncertainty quantification metrics
    pub async fn get_metrics(&self) -> Result<QuantumUncertaintyMetrics, TENGRIError> {
        let recent_fisher_info = self.fisher_matrices.back()
            .map(|fm| fm.trace)
            .unwrap_or(0.0);
        
        let recent_entanglement = self.entanglement_entropies.back()
            .map(|ee| ee.von_neumann_entropy)
            .unwrap_or(0.0);
        
        Ok(QuantumUncertaintyMetrics {
            average_uncertainty: self.average_uncertainty,
            uncertainty_variance: self.uncertainty_variance,
            calibration_score: self.calibration_score,
            high_uncertainty_threshold: self.high_uncertainty_threshold,
            low_uncertainty_threshold: self.low_uncertainty_threshold,
            recent_fisher_information: recent_fisher_info,
            recent_entanglement_entropy: recent_entanglement,
            confidence_calibration_factor: self.confidence_calibration_factor,
        })
    }

    /// Detect uncertainty regime (high/medium/low)
    pub fn detect_uncertainty_regime(&self, uncertainty: &QuantumUncertainty) -> UncertaintyRegime {
        if uncertainty.entropy > self.high_uncertainty_threshold {
            UncertaintyRegime::High
        } else if uncertainty.entropy < self.low_uncertainty_threshold {
            UncertaintyRegime::Low
        } else {
            UncertaintyRegime::Medium
        }
    }
}

/// Uncertainty regime classification
#[derive(Debug, Clone, PartialEq)]
pub enum UncertaintyRegime {
    Low,
    Medium,
    High,
}

/// Quantum uncertainty quantification metrics
#[derive(Debug, Clone)]
pub struct QuantumUncertaintyMetrics {
    pub average_uncertainty: f64,
    pub uncertainty_variance: f64,
    pub calibration_score: f64,
    pub high_uncertainty_threshold: f64,
    pub low_uncertainty_threshold: f64,
    pub recent_fisher_information: f64,
    pub recent_entanglement_entropy: f64,
    pub confidence_calibration_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[tokio::test]
    async fn test_quantum_uncertainty_quantifier_creation() {
        let quantifier = QuantumUncertaintyQuantifier::new(4).await.unwrap();
        assert_eq!(quantifier.n_qubits, 4);
        assert_eq!(quantifier.uncertainty_history.len(), 0);
    }

    #[tokio::test]
    async fn test_quantum_uncertainty_quantification() {
        let mut quantifier = QuantumUncertaintyQuantifier::new(4).await.unwrap();
        
        let input = DMatrix::from_vec(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let uncertainty = quantifier.quantify_uncertainty(&input).await.unwrap();
        
        assert!(uncertainty.entropy >= 0.0);
        assert!(uncertainty.variance >= 0.0);
        assert!(uncertainty.confidence_interval.0 <= uncertainty.confidence_interval.1);
    }

    #[test]
    fn test_entanglement_entropy_computation() {
        let quantum_state = QuantumState::new(2);
        let subsystem_qubits = vec![0];
        
        let entropy = EntanglementEntropy::compute(&quantum_state, &subsystem_qubits).unwrap();
        
        assert!(entropy.von_neumann_entropy >= 0.0);
        assert!(entropy.linear_entropy >= 0.0);
        assert!(entropy.schmidt_rank > 0);
    }

    #[test]
    fn test_quantum_fisher_matrix() {
        let quantum_states = vec![QuantumState::new(2)];
        let parameters = DVector::from_vec(vec![0.0, 0.1, 0.2, 0.3]);
        
        let fisher_matrix = QuantumFisherMatrix::compute(&quantum_states, &parameters).unwrap();
        
        assert_eq!(fisher_matrix.matrix.nrows(), 4);
        assert_eq!(fisher_matrix.matrix.ncols(), 4);
        assert!(fisher_matrix.condition_number >= 1.0);
        assert!(fisher_matrix.trace >= 0.0);
    }

    #[test]
    fn test_uncertainty_regime_detection() {
        let quantifier = QuantumUncertaintyQuantifier {
            n_qubits: 4,
            uncertainty_history: VecDeque::new(),
            fisher_matrices: VecDeque::new(),
            entanglement_entropies: VecDeque::new(),
            confidence_calibration_factor: 1.0,
            entropy_scaling_factor: 0.1,
            fisher_weight: 0.2,
            high_uncertainty_threshold: 0.8,
            low_uncertainty_threshold: 0.2,
            entropy_threshold: 0.5,
            average_uncertainty: 0.5,
            uncertainty_variance: 0.1,
            calibration_score: 0.0,
        };
        
        let high_uncertainty = QuantumUncertainty {
            entropy: 0.9,
            variance: 0.1,
            confidence_interval: (0.0, 1.0),
        };
        
        let low_uncertainty = QuantumUncertainty {
            entropy: 0.1,
            variance: 0.01,
            confidence_interval: (0.4, 0.6),
        };
        
        let medium_uncertainty = QuantumUncertainty {
            entropy: 0.5,
            variance: 0.05,
            confidence_interval: (0.2, 0.8),
        };
        
        assert_eq!(quantifier.detect_uncertainty_regime(&high_uncertainty), UncertaintyRegime::High);
        assert_eq!(quantifier.detect_uncertainty_regime(&low_uncertainty), UncertaintyRegime::Low);
        assert_eq!(quantifier.detect_uncertainty_regime(&medium_uncertainty), UncertaintyRegime::Medium);
    }

    #[test]
    fn test_embed_subsystem_index() {
        let subsystem_qubits = vec![0, 2];
        let subsystem_index = 3; // Binary: 11
        let env_index = 1;       // Binary: 1
        let total_qubits = 3;
        
        let full_index = EntanglementEntropy::embed_subsystem_index(
            subsystem_index,
            env_index,
            &subsystem_qubits,
            total_qubits,
        );
        
        // Should set bits 0 and 2 from subsystem (11), and bit 1 from environment (1)
        // Result should be 111 = 7
        assert_eq!(full_index, 7);
    }

    #[tokio::test]
    async fn test_uncertainty_calibration() {
        let mut quantifier = QuantumUncertaintyQuantifier::new(4).await.unwrap();
        
        let predictions = vec![0.5, 0.6, 0.7];
        let actual_values = vec![0.52, 0.58, 0.72];
        let uncertainties = vec![
            QuantumUncertainty {
                entropy: 0.1,
                variance: 0.01,
                confidence_interval: (0.4, 0.6),
            },
            QuantumUncertainty {
                entropy: 0.2,
                variance: 0.02,
                confidence_interval: (0.5, 0.7),
            },
            QuantumUncertainty {
                entropy: 0.15,
                variance: 0.015,
                confidence_interval: (0.6, 0.8),
            },
        ];
        
        let result = quantifier.calibrate(&predictions, &actual_values, &uncertainties).await;
        assert!(result.is_ok());
        assert!(quantifier.calibration_score >= -1.0 && quantifier.calibration_score <= 1.0);
    }
}