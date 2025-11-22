//! Quantum-inspired feature embeddings and encodings
//!
//! This module provides quantum-inspired methods for encoding classical data
//! into quantum-enhanced feature spaces for machine learning applications.

use crate::{Complex, StateVector, Result, QuantumError, Circuit};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
// Removed unused HashMap import
use std::f64::consts::PI;

/// Trait for quantum-inspired feature embeddings
pub trait QuantumEmbedding {
    /// Embed classical data into quantum-enhanced feature space
    fn embed(&self, data: &[f64]) -> Result<StateVector>;
    
    /// Get the dimension of the embedded space
    fn embedded_dimension(&self) -> usize;
    
    /// Get the original data dimension
    fn input_dimension(&self) -> usize;
}

/// Amplitude embedding - encode classical data as quantum state amplitudes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplitudeEmbedding {
    /// Dimension of the input data
    input_dim: usize,
    /// Number of qubits needed
    n_qubits: usize,
    /// Normalization method
    normalization: NormalizationMethod,
    /// Padding value for incomplete dimensions
    padding_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    L2,
    L1,
    Max,
    None,
}

impl AmplitudeEmbedding {
    /// Create a new amplitude embedding
    pub fn new(input_dim: usize, normalization: NormalizationMethod) -> Self {
        let n_qubits = (input_dim as f64).log2().ceil() as usize;
        
        Self {
            input_dim,
            n_qubits,
            normalization,
            padding_value: 0.0,
        }
    }
    
    /// Set the padding value for incomplete dimensions
    pub fn with_padding(mut self, padding_value: f64) -> Self {
        self.padding_value = padding_value;
        self
    }
    
    /// Normalize data according to the specified method
    fn normalize_data(&self, data: &[f64]) -> Vec<f64> {
        match self.normalization {
            NormalizationMethod::L2 => {
                let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    data.iter().map(|x| x / norm).collect()
                } else {
                    data.to_vec()
                }
            },
            NormalizationMethod::L1 => {
                let norm = data.iter().map(|x| x.abs()).sum::<f64>();
                if norm > 1e-12 {
                    data.iter().map(|x| x / norm).collect()
                } else {
                    data.to_vec()
                }
            },
            NormalizationMethod::Max => {
                let max_val = data.iter().map(|x| x.abs()).fold(0.0, f64::max);
                if max_val > 1e-12 {
                    data.iter().map(|x| x / max_val).collect()
                } else {
                    data.to_vec()
                }
            },
            NormalizationMethod::None => data.to_vec(),
        }
    }
}

impl QuantumEmbedding for AmplitudeEmbedding {
    fn embed(&self, data: &[f64]) -> Result<StateVector> {
        if data.len() > self.input_dim {
            return Err(QuantumError::InvalidParameter(
                format!("Input data dimension {} exceeds expected {}", data.len(), self.input_dim)
            ));
        }
        
        // Pad data to power of 2 if necessary
        let mut padded_data = data.to_vec();
        let target_dim = 1 << self.n_qubits;
        
        while padded_data.len() < target_dim {
            padded_data.push(self.padding_value);
        }
        
        // Normalize the data
        let normalized_data = self.normalize_data(&padded_data);
        
        // Create quantum state vector
        let mut state = Array1::zeros(target_dim);
        for (i, &value) in normalized_data.iter().enumerate() {
            state[i] = Complex::new(value, 0.0);
        }
        
        Ok(state)
    }
    
    fn embedded_dimension(&self) -> usize {
        1 << self.n_qubits
    }
    
    fn input_dimension(&self) -> usize {
        self.input_dim
    }
}

/// Angle embedding - encode classical data as rotation angles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleEmbedding {
    /// Dimension of the input data
    input_dim: usize,
    /// Number of qubits (should match input dimension for 1:1 encoding)
    n_qubits: usize,
    /// Scaling factor for the angles
    angle_scale: f64,
    /// Base circuit for initialization
    base_circuit: Option<String>,
}

impl AngleEmbedding {
    /// Create a new angle embedding
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            n_qubits: input_dim,
            angle_scale: 1.0,
            base_circuit: None,
        }
    }
    
    /// Set the angle scaling factor
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.angle_scale = scale;
        self
    }
    
    /// Set the number of qubits (for repeated encoding patterns)
    pub fn with_qubits(mut self, n_qubits: usize) -> Self {
        self.n_qubits = n_qubits;
        self
    }
    
    /// Build the embedding circuit
    pub fn build_circuit(&self, data: &[f64]) -> Result<Circuit> {
        if data.len() != self.input_dim {
            return Err(QuantumError::InvalidParameter(
                format!("Input data dimension {} doesn't match expected {}", data.len(), self.input_dim)
            ));
        }
        
        let mut circuit = Circuit::new(self.n_qubits);
        
        // Apply rotations based on data
        for (i, &value) in data.iter().enumerate() {
            let qubit = i % self.n_qubits;
            let angle = self.angle_scale * value;
            
            circuit.add_gate(Box::new(crate::gates::RY::new(qubit, angle)))?;
        }
        
        Ok(circuit)
    }
}

impl QuantumEmbedding for AngleEmbedding {
    fn embed(&self, data: &[f64]) -> Result<StateVector> {
        let circuit = self.build_circuit(data)?;
        circuit.execute()
    }
    
    fn embedded_dimension(&self) -> usize {
        1 << self.n_qubits
    }
    
    fn input_dimension(&self) -> usize {
        self.input_dim
    }
}

/// Parametric quantum embedding with learnable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricEmbedding {
    /// Number of data features
    n_features: usize,
    /// Number of qubits
    n_qubits: usize,
    /// Number of encoding layers
    n_layers: usize,
    /// Learnable parameters for the embedding
    parameters: Vec<f64>,
    /// Parameter names for tracking
    parameter_names: Vec<String>,
    /// Entanglement pattern
    entanglement: EntanglementPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    Full,
    Custom(Vec<(usize, usize)>),
    None,
}

impl ParametricEmbedding {
    /// Create a new parametric embedding
    pub fn new(n_features: usize, n_qubits: usize, n_layers: usize) -> Self {
        // Initialize parameters randomly
        let mut rng = rand::thread_rng();
        let n_params = n_layers * n_qubits * 2; // 2 rotations per qubit per layer
        let parameters: Vec<f64> = (0..n_params)
            .map(|_| rng.gen_range(-PI..PI))
            .collect();
        
        let parameter_names: Vec<String> = (0..n_layers)
            .flat_map(|layer| {
                (0..n_qubits).flat_map(move |qubit| {
                    vec![
                        format!("layer_{}_qubit_{}_theta", layer, qubit),
                        format!("layer_{}_qubit_{}_phi", layer, qubit),
                    ]
                })
            })
            .collect();
        
        Self {
            n_features,
            n_qubits,
            n_layers,
            parameters,
            parameter_names,
            entanglement: EntanglementPattern::Linear,
        }
    }
    
    /// Set the entanglement pattern
    pub fn with_entanglement(mut self, entanglement: EntanglementPattern) -> Self {
        self.entanglement = entanglement;
        self
    }
    
    /// Set the embedding parameters
    pub fn set_parameters(&mut self, params: Vec<f64>) -> Result<()> {
        if params.len() != self.parameters.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Expected {} parameters, got {}", self.parameters.len(), params.len())
            ));
        }
        self.parameters = params;
        Ok(())
    }
    
    /// Get the current parameters
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    
    /// Build the embedding circuit
    pub fn build_circuit(&self, data: &[f64]) -> Result<Circuit> {
        if data.len() != self.n_features {
            return Err(QuantumError::InvalidParameter(
                format!("Input data dimension {} doesn't match expected {}", data.len(), self.n_features)
            ));
        }
        
        let mut circuit = Circuit::new(self.n_qubits);
        let mut param_idx = 0;
        
        for _layer in 0..self.n_layers {
            // Data encoding layer
            for (feature_idx, &data_val) in data.iter().enumerate() {
                let qubit = feature_idx % self.n_qubits;
                circuit.add_gate(Box::new(crate::gates::RY::new(qubit, data_val)))?;
            }
            
            // Parametric layer
            for qubit in 0..self.n_qubits {
                let theta = self.parameters[param_idx];
                let phi = self.parameters[param_idx + 1];
                param_idx += 2;
                
                circuit.add_gate(Box::new(crate::gates::RY::new(qubit, theta)))?;
                circuit.add_gate(Box::new(crate::gates::RZ::new(qubit, phi)))?;
            }
            
            // Entangling layer
            self.add_entangling_gates(&mut circuit)?;
        }
        
        Ok(circuit)
    }
    
    /// Add entangling gates based on the pattern
    fn add_entangling_gates(&self, circuit: &mut Circuit) -> Result<()> {
        let pairs = self.get_entangling_pairs();
        
        for (control, target) in pairs {
            circuit.add_gate(Box::new(crate::gates::CNOT::new(control, target)))?;
        }
        
        Ok(())
    }
    
    /// Get entangling pairs based on the pattern
    fn get_entangling_pairs(&self) -> Vec<(usize, usize)> {
        match &self.entanglement {
            EntanglementPattern::Linear => {
                (0..self.n_qubits - 1).map(|i| (i, i + 1)).collect()
            },
            EntanglementPattern::Circular => {
                let mut pairs: Vec<_> = (0..self.n_qubits - 1).map(|i| (i, i + 1)).collect();
                if self.n_qubits > 2 {
                    pairs.push((self.n_qubits - 1, 0));
                }
                pairs
            },
            EntanglementPattern::Full => {
                let mut pairs = Vec::new();
                for i in 0..self.n_qubits {
                    for j in (i + 1)..self.n_qubits {
                        pairs.push((i, j));
                    }
                }
                pairs
            },
            EntanglementPattern::Custom(pairs) => pairs.clone(),
            EntanglementPattern::None => Vec::new(),
        }
    }
}

impl QuantumEmbedding for ParametricEmbedding {
    fn embed(&self, data: &[f64]) -> Result<StateVector> {
        let circuit = self.build_circuit(data)?;
        circuit.execute()
    }
    
    fn embedded_dimension(&self) -> usize {
        1 << self.n_qubits
    }
    
    fn input_dimension(&self) -> usize {
        self.n_features
    }
}

/// Quantum kernel embedding using quantum feature maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumKernelEmbedding {
    /// Base embedding method
    embedding: ParametricEmbedding,
    /// Kernel type
    kernel_type: KernelType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelType {
    /// Linear kernel in quantum space
    Linear,
    /// RBF-like kernel using quantum fidelity
    RBF { gamma: f64 },
    /// Custom polynomial kernel
    Polynomial { degree: usize, coeff: f64 },
}

impl QuantumKernelEmbedding {
    /// Create a new quantum kernel embedding
    pub fn new(
        n_features: usize,
        n_qubits: usize,
        n_layers: usize,
        kernel_type: KernelType,
    ) -> Self {
        Self {
            embedding: ParametricEmbedding::new(n_features, n_qubits, n_layers),
            kernel_type,
        }
    }
    
    /// Compute quantum kernel between two data points
    pub fn kernel(&self, x1: &[f64], x2: &[f64]) -> Result<f64> {
        let state1 = self.embedding.embed(x1)?;
        let state2 = self.embedding.embed(x2)?;
        
        match &self.kernel_type {
            KernelType::Linear => {
                // Inner product in quantum space
                let overlap = state1.iter()
                    .zip(state2.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum::<Complex>();
                Ok(overlap.re)
            },
            KernelType::RBF { gamma } => {
                // Quantum fidelity-based RBF kernel
                let fidelity = crate::utils::fidelity(&state1, &state2)?;
                Ok((-gamma * (1.0 - fidelity)).exp())
            },
            KernelType::Polynomial { degree, coeff } => {
                // Polynomial kernel in quantum space
                let overlap = state1.iter()
                    .zip(state2.iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum::<Complex>();
                Ok((coeff + overlap.re).powi(*degree as i32))
            },
        }
    }
    
    /// Compute kernel matrix for a set of data points
    pub fn kernel_matrix(&self, data: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n = data.len();
        let mut kernel_matrix = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in i..n {
                let kernel_val = self.kernel(&data[i], &data[j])?;
                kernel_matrix[[i, j]] = kernel_val;
                kernel_matrix[[j, i]] = kernel_val; // Symmetric
            }
        }
        
        Ok(kernel_matrix)
    }
}

/// Quantum-enhanced dimensionality reduction
pub struct QuantumPCA {
    /// Number of components to keep
    n_components: usize,
    /// Quantum embedding for data
    embedding: ParametricEmbedding,
    /// Learned transformation matrix
    components: Option<Array2<f64>>,
    /// Mean of the training data
    mean: Option<Array1<f64>>,
}

impl QuantumPCA {
    /// Create a new quantum PCA
    pub fn new(n_features: usize, n_components: usize, n_qubits: usize) -> Self {
        Self {
            n_components,
            embedding: ParametricEmbedding::new(n_features, n_qubits, 2),
            components: None,
            mean: None,
        }
    }
    
    /// Fit the quantum PCA on training data
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        let n_samples = data.len();
        let _n_features = data[0].len();
        
        // Embed all data points
        let mut embedded_data = Vec::with_capacity(n_samples);
        for sample in data {
            let embedded = self.embedding.embed(sample)?;
            // Convert complex amplitudes to real features (magnitude and phase)
            let mut real_features = Vec::with_capacity(embedded.len() * 2);
            for amplitude in embedded.iter() {
                real_features.push(amplitude.norm());
                real_features.push(amplitude.arg());
            }
            embedded_data.push(real_features);
        }
        
        // Convert to matrix
        let data_matrix = Array2::from_shape_fn(
            (n_samples, embedded_data[0].len()),
            |(i, j)| embedded_data[i][j]
        );
        
        // Compute mean and center the data
        let mean = data_matrix.mean_axis(Axis(0)).unwrap();
        let centered_data = &data_matrix - &mean.clone().insert_axis(Axis(0));
        
        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples as f64 - 1.0);
        
        // For now, use a simplified PCA (in practice, would use eigendecomposition)
        // This is a placeholder - real implementation would need eigenvalue decomposition
        let components = Array2::eye(self.n_components.min(cov_matrix.nrows()));
        
        self.components = Some(components);
        self.mean = Some(mean);
        
        Ok(())
    }
    
    /// Transform data using the fitted quantum PCA
    pub fn transform(&self, data: &[Vec<f64>]) -> Result<Array2<f64>> {
        let _components = self.components.as_ref()
            .ok_or_else(|| QuantumError::InvalidState)?;
        let mean = self.mean.as_ref()
            .ok_or_else(|| QuantumError::InvalidState)?;
        
        // Embed and transform data
        let mut transformed_data = Vec::new();
        
        for sample in data {
            let embedded = self.embedding.embed(sample)?;
            let mut real_features = Vec::with_capacity(embedded.len() * 2);
            for amplitude in embedded.iter() {
                real_features.push(amplitude.norm());
                real_features.push(amplitude.arg());
            }
            
            // Center the features
            let centered: Vec<f64> = real_features.iter()
                .zip(mean.iter())
                .map(|(x, m)| x - m)
                .collect();
            
            // Project onto principal components (simplified)
            let projected: Vec<f64> = centered[..self.n_components].to_vec();
            transformed_data.push(projected);
        }
        
        // Convert to array
        let n_samples = transformed_data.len();
        let transformed_matrix = Array2::from_shape_fn(
            (n_samples, self.n_components),
            |(i, j)| transformed_data[i][j]
        );
        
        Ok(transformed_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_amplitude_embedding() {
        let embedding = AmplitudeEmbedding::new(3, NormalizationMethod::L2);
        let data = vec![0.6, 0.8, 0.0];
        
        let state = embedding.embed(&data).unwrap();
        assert_eq!(state.len(), 4); // 2 qubits needed for 3 features
        
        // Check normalization
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_angle_embedding() {
        let embedding = AngleEmbedding::new(2);
        let data = vec![PI / 4.0, PI / 2.0];
        
        let state = embedding.embed(&data).unwrap();
        assert_eq!(state.len(), 4); // 2 qubits
        
        // State should be normalized
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parametric_embedding() {
        let embedding = ParametricEmbedding::new(2, 2, 1);
        let data = vec![0.5, 0.3];
        
        let state = embedding.embed(&data).unwrap();
        assert_eq!(state.len(), 4); // 2 qubits
        assert!(embedding.parameters().len() > 0);
    }
    
    #[test]
    fn test_quantum_kernel() {
        let kernel = QuantumKernelEmbedding::new(
            2, 2, 1,
            KernelType::RBF { gamma: 1.0 }
        );
        
        let x1 = vec![0.5, 0.3];
        let x2 = vec![0.5, 0.3];
        let x3 = vec![0.1, 0.9];
        
        let k11 = kernel.kernel(&x1, &x1).unwrap();
        let k12 = kernel.kernel(&x1, &x2).unwrap();
        let k13 = kernel.kernel(&x1, &x3).unwrap();
        
        // Kernel with itself should be close to 1
        assert!(k11 > 0.9);
        // Kernel with identical point should be close to k11
        assert_abs_diff_eq!(k11, k12, epsilon = 0.1);
        // Kernel with different point should be smaller
        assert!(k13 < k11);
    }
    
    #[test]
    fn test_quantum_pca() {
        let mut qpca = QuantumPCA::new(2, 1, 2);
        
        let data = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
        ];
        
        qpca.fit(&data).unwrap();
        let transformed = qpca.transform(&data).unwrap();
        
        assert_eq!(transformed.shape(), &[4, 1]);
    }
    
    #[test]
    fn test_kernel_matrix() {
        let kernel = QuantumKernelEmbedding::new(
            2, 2, 1,
            KernelType::Linear
        );
        
        let data = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
        ];
        
        let k_matrix = kernel.kernel_matrix(&data).unwrap();
        assert_eq!(k_matrix.shape(), &[3, 3]);
        
        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(k_matrix[[i, j]], k_matrix[[j, i]], epsilon = 1e-10);
            }
        }
        
        // Diagonal should be positive
        for i in 0..3 {
            assert!(k_matrix[[i, i]] > 0.0);
        }
    }
}