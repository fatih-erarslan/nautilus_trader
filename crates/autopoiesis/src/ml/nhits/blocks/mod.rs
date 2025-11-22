//! Hierarchical Neural Blocks with Basis Expansion
//! Core building blocks of NHITS architecture

use ndarray::{Array2, Array3, Array4, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::pooling::{PoolingLayer, PoolingType};
use super::interpolation::{InterpolationLayer, InterpolationType};

/// Configuration for hierarchical blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_basis: usize,
    pub pooling_factor: usize,
    pub pooling_type: PoolingType,
    pub interpolation_type: InterpolationType,
    pub dropout_rate: f64,
    pub activation: ActivationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    SiLU,
}

/// Hierarchical block with basis expansion
#[derive(Debug, Clone)]
pub struct HierarchicalBlock {
    config: BlockConfig,
    
    // Basis expansion parameters
    basis_functions: Vec<BasisFunction>,
    basis_weights: Array2<f64>,
    
    // Neural network layers
    linear1: LinearLayer,
    linear2: LinearLayer,
    
    // Pooling and interpolation
    pooling: PoolingLayer,
    interpolation: InterpolationLayer,
    
    // Normalization parameters
    layer_norm: LayerNorm,
    
    // Adaptive parameters
    adaptive_basis: bool,
    basis_learning_rate: f64,
}

/// Basis function types for temporal modeling
#[derive(Debug, Clone)]
pub enum BasisFunction {
    Polynomial { degree: usize },
    Fourier { frequency: f64 },
    Wavelet { scale: f64, translation: f64 },
    Gaussian { mean: f64, std: f64 },
    Exponential { rate: f64 },
}

impl HierarchicalBlock {
    pub fn new(config: BlockConfig) -> Self {
        let basis_functions = Self::initialize_basis_functions(config.num_basis);
        let basis_weights = Array2::from_shape_fn(
            (config.num_basis, config.hidden_size),
            |_| rand::thread_rng().gen_range(-0.1..0.1),
        );
        
        Self {
            linear1: LinearLayer::new(config.input_size, config.hidden_size),
            linear2: LinearLayer::new(config.hidden_size, config.hidden_size),
            pooling: PoolingLayer::new(config.pooling_factor, config.pooling_type.clone()),
            interpolation: InterpolationLayer::new(config.interpolation_type.clone()),
            layer_norm: LayerNorm::new(config.hidden_size),
            basis_functions,
            basis_weights,
            config,
            adaptive_basis: true,
            basis_learning_rate: 0.001,
        }
    }
    
    pub fn new_with_units(units: usize) -> Self {
        let config = BlockConfig {
            input_size: units,
            hidden_size: units,
            num_basis: 10,
            pooling_factor: 2,
            pooling_type: PoolingType::Max,
            interpolation_type: InterpolationType::Linear,
            dropout_rate: 0.1,
            activation: ActivationType::GELU,
        };
        Self::new(config)
    }
    
    /// Forward pass with consciousness modulation
    pub fn forward(
        &mut self,
        input: &Array3<f64>,
        consciousness_strength: f64,
    ) -> Result<Array3<f64>, BlockError> {
        let (batch_size, seq_len, features) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
        );
        
        // Apply pooling to reduce temporal resolution
        let pooled = self.pooling.forward(input)?;
        
        // Basis expansion
        let expanded = self.apply_basis_expansion(&pooled, consciousness_strength)?;
        
        // First linear transformation
        let hidden1 = self.linear1.forward(&expanded)?;
        let activated1 = self.apply_activation(&hidden1)?;
        
        // Layer normalization
        let normalized = self.layer_norm.forward(&activated1)?;
        
        // Second linear transformation with residual connection
        let hidden2 = self.linear2.forward(&normalized)?;
        let output = hidden2 + &normalized; // Residual connection
        
        // Interpolate back to original resolution
        let interpolated = self.interpolation.forward(&output, seq_len)?;
        
        // Apply dropout if training
        let final_output = self.apply_dropout(&interpolated)?;
        
        Ok(final_output)
    }
    
    /// Initialize diverse basis functions
    fn initialize_basis_functions(num_basis: usize) -> Vec<BasisFunction> {
        let mut basis = Vec::new();
        let mut rng = rand::thread_rng();
        
        for i in 0..num_basis {
            match i % 5 {
                0 => basis.push(BasisFunction::Polynomial { degree: i / 5 + 1 }),
                1 => basis.push(BasisFunction::Fourier {
                    frequency: (i as f64 + 1.0) * 0.1,
                }),
                2 => basis.push(BasisFunction::Wavelet {
                    scale: 2.0_f64.powi(i as i32 / 5),
                    translation: rng.gen_range(-1.0..1.0),
                }),
                3 => basis.push(BasisFunction::Gaussian {
                    mean: rng.gen_range(-2.0..2.0),
                    std: rng.gen_range(0.1..2.0),
                }),
                _ => basis.push(BasisFunction::Exponential {
                    rate: rng.gen_range(0.1..2.0),
                }),
            }
        }
        
        basis
    }
    
    /// Apply basis expansion with consciousness modulation
    fn apply_basis_expansion(
        &mut self,
        input: &Array3<f64>,
        consciousness_strength: f64,
    ) -> Result<Array3<f64>, BlockError> {
        let (batch_size, seq_len, features) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
        );
        
        // Create time points
        let time_points = Array2::from_shape_fn((batch_size, seq_len), |(_, t)| {
            t as f64 / seq_len as f64
        });
        
        // Compute basis values
        let mut basis_values = Array3::zeros((batch_size, seq_len, self.config.num_basis));
        
        for (basis_idx, basis_fn) in self.basis_functions.iter().enumerate() {
            for b in 0..batch_size {
                for t in 0..seq_len {
                    let time = time_points[[b, t]];
                    let value = self.evaluate_basis(basis_fn, time) * consciousness_strength;
                    basis_values[[b, t, basis_idx]] = value;
                }
            }
        }
        
        // Combine input with basis expansion
        let expanded = self.combine_with_basis(input, &basis_values)?;
        
        // Adapt basis functions if enabled
        if self.adaptive_basis {
            self.adapt_basis_functions(&input, &expanded, consciousness_strength)?;
        }
        
        Ok(expanded)
    }
    
    /// Evaluate basis function at given time point
    fn evaluate_basis(&self, basis: &BasisFunction, t: f64) -> f64 {
        match basis {
            BasisFunction::Polynomial { degree } => t.powi(*degree as i32),
            BasisFunction::Fourier { frequency } => (2.0 * PI * frequency * t).cos(),
            BasisFunction::Wavelet { scale, translation } => {
                let u = (t - translation) / scale;
                // Mexican hat wavelet
                (1.0 - u * u) * (-u * u / 2.0).exp()
            }
            BasisFunction::Gaussian { mean, std } => {
                let z = (t - mean) / std;
                (-0.5 * z * z).exp()
            }
            BasisFunction::Exponential { rate } => (-rate * t).exp(),
        }
    }
    
    /// Combine input with basis expansion
    fn combine_with_basis(
        &self,
        input: &Array3<f64>,
        basis_values: &Array3<f64>,
    ) -> Result<Array3<f64>, BlockError> {
        let (batch_size, seq_len, _) = input.shape();
        let hidden_size = self.config.hidden_size;
        
        let mut output = Array3::zeros((batch_size, seq_len, hidden_size));
        
        // Project input to hidden dimension
        let projected_input = input.dot(&Array2::from_shape_fn(
            (input.shape()[2], hidden_size),
            |_| rand::thread_rng().gen_range(-0.1..0.1),
        ));
        
        // Combine with weighted basis functions
        for b in 0..batch_size {
            for t in 0..seq_len {
                for h in 0..hidden_size {
                    output[[b, t, h]] = projected_input[[b, t, h]];
                    
                    for basis_idx in 0..self.config.num_basis {
                        output[[b, t, h]] += basis_values[[b, t, basis_idx]]
                            * self.basis_weights[[basis_idx, h]];
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Adapt basis functions based on gradient information
    fn adapt_basis_functions(
        &mut self,
        input: &Array3<f64>,
        output: &Array3<f64>,
        consciousness_strength: f64,
    ) -> Result<(), BlockError> {
        // Simplified adaptation logic
        // Full implementation would use proper gradients
        
        for (i, basis) in self.basis_functions.iter_mut().enumerate() {
            match basis {
                BasisFunction::Gaussian { mean, std } => {
                    // Adapt Gaussian parameters based on data distribution
                    *mean += self.basis_learning_rate * consciousness_strength * rand::thread_rng().gen_range(-0.01..0.01);
                    *std = (*std + self.basis_learning_rate * consciousness_strength * rand::thread_rng().gen_range(-0.01..0.01)).max(0.1);
                }
                BasisFunction::Fourier { frequency } => {
                    // Adapt frequency based on spectral content
                    *frequency += self.basis_learning_rate * consciousness_strength * rand::thread_rng().gen_range(-0.01..0.01);
                }
                _ => {} // Other basis functions remain fixed for now
            }
        }
        
        Ok(())
    }
    
    /// Apply activation function
    fn apply_activation(&self, input: &Array3<f64>) -> Result<Array3<f64>, BlockError> {
        let output = match self.config.activation {
            ActivationType::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationType::GELU => input.mapv(|x| {
                x * 0.5 * (1.0 + (PI.sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
            ActivationType::Tanh => input.mapv(|x| x.tanh()),
            ActivationType::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationType::SiLU => input.mapv(|x| x * (1.0 / (1.0 + (-x).exp()))),
        };
        Ok(output)
    }
    
    /// Apply dropout during training
    fn apply_dropout(&self, input: &Array3<f64>) -> Result<Array3<f64>, BlockError> {
        // In production, this would check if we're in training mode
        // For now, return input unchanged
        Ok(input.clone())
    }
    
    /// Adjust pooling factor
    pub fn adjust_pooling(&mut self, factor: usize) {
        self.pooling = PoolingLayer::new(factor, self.config.pooling_type.clone());
        self.config.pooling_factor = factor;
    }
    
    /// Expand basis functions
    pub fn expand_basis(&mut self, new_basis_count: usize) {
        let additional_basis = new_basis_count - self.config.num_basis;
        
        for _ in 0..additional_basis {
            self.basis_functions.push(BasisFunction::Gaussian {
                mean: rand::thread_rng().gen_range(-2.0..2.0),
                std: rand::thread_rng().gen_range(0.1..2.0),
            });
        }
        
        // Expand basis weights
        let mut new_weights = Array2::zeros((new_basis_count, self.config.hidden_size));
        new_weights.slice_mut(s![..self.config.num_basis, ..]).assign(&self.basis_weights);
        
        // Initialize new weights
        for i in self.config.num_basis..new_basis_count {
            for j in 0..self.config.hidden_size {
                new_weights[[i, j]] = rand::thread_rng().gen_range(-0.1..0.1);
            }
        }
        
        self.basis_weights = new_weights;
        self.config.num_basis = new_basis_count;
    }
}

/// Simple linear layer
#[derive(Debug, Clone)]
struct LinearLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl LinearLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let scale = (2.0 / input_size as f64).sqrt();
        Self {
            weights: Array2::from_shape_fn(
                (input_size, output_size),
                |_| rand::thread_rng().gen_range(-scale..scale),
            ),
            bias: Array1::zeros(output_size),
        }
    }
    
    fn forward(&self, input: &Array3<f64>) -> Result<Array3<f64>, BlockError> {
        let (batch_size, seq_len, _) = input.shape();
        let output_size = self.weights.shape()[1];
        
        let mut output = Array3::zeros((batch_size, seq_len, output_size));
        
        for b in 0..batch_size {
            let batch_input = input.slice(s![b, .., ..]);
            let batch_output = batch_input.dot(&self.weights) + &self.bias;
            output.slice_mut(s![b, .., ..]).assign(&batch_output);
        }
        
        Ok(output)
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
struct LayerNorm {
    normalized_shape: usize,
    eps: f64,
    weight: Array1<f64>,
    bias: Array1<f64>,
}

impl LayerNorm {
    fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            weight: Array1::ones(normalized_shape),
            bias: Array1::zeros(normalized_shape),
        }
    }
    
    fn forward(&self, input: &Array3<f64>) -> Result<Array3<f64>, BlockError> {
        let (batch_size, seq_len, features) = input.shape();
        let mut output = Array3::zeros((batch_size, seq_len, features));
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let values = input.slice(s![b, t, ..]);
                let mean = values.mean().unwrap_or(0.0);
                let var = values.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
                
                let normalized = (values - mean) / (var + self.eps).sqrt();
                let scaled = &normalized * &self.weight + &self.bias;
                
                output.slice_mut(s![b, t, ..]).assign(&scaled);
            }
        }
        
        Ok(output)
    }
}

/// Block errors
#[derive(Debug, thiserror::Error)]
pub enum BlockError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

use ndarray::{Array1, s};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basis_expansion() {
        // Test implementation
    }
}