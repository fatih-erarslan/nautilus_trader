//! Normalization layers (LayerNorm, BatchNorm)

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{DType, Tensor, TensorOps};
use super::Layer;
use serde::{Deserialize, Serialize};

/// Layer Normalization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Normalized shape (typically last dimension)
    pub normalized_shape: Vec<usize>,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Whether to use learnable affine parameters
    pub elementwise_affine: bool,
}

impl LayerNormConfig {
    /// Create config for normalizing over last dimension
    pub fn new(dim: usize) -> Self {
        Self {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// Create config with custom shape
    pub fn with_shape(shape: Vec<usize>) -> Self {
        Self {
            normalized_shape: shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// Set epsilon
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set affine parameters
    pub fn with_affine(mut self, affine: bool) -> Self {
        self.elementwise_affine = affine;
        self
    }
}

/// Layer Normalization
///
/// Normalizes input across the last dimension(s):
/// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
#[derive(Debug, Clone)]
pub struct LayerNorm {
    config: LayerNormConfig,
    /// Scale parameter gamma
    gamma: Option<Tensor>,
    /// Shift parameter beta
    beta: Option<Tensor>,
    device: Device,
}

impl LayerNorm {
    /// Create a new LayerNorm
    pub fn new(config: LayerNormConfig, device: &Device) -> MlResult<Self> {
        let (gamma, beta) = if config.elementwise_affine {
            let size: usize = config.normalized_shape.iter().product();
            let gamma = Tensor::ones(vec![size], DType::F32, device)?;
            let beta = Tensor::zeros(vec![size], DType::F32, device)?;
            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            gamma,
            beta,
            device: device.clone(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &LayerNormConfig {
        &self.config
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let input_shape = input.shape().dims();
        let norm_size: usize = self.config.normalized_shape.iter().product();

        // Validate shape
        let input_suffix: usize = input_shape
            .iter()
            .rev()
            .take(self.config.normalized_shape.len())
            .product();

        if input_suffix != norm_size {
            return Err(crate::error::MlError::shape_mismatch(
                self.config.normalized_shape.clone(),
                input_shape[input_shape.len() - self.config.normalized_shape.len()..].to_vec(),
            ));
        }

        // Compute mean and variance over normalized dimensions
        let output = layer_norm_forward(
            input,
            self.gamma.as_ref(),
            self.beta.as_ref(),
            self.config.eps,
            norm_size,
        )?;

        Ok(output)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        if self.config.elementwise_affine {
            let size: usize = self.config.normalized_shape.iter().product();
            size * 2 // gamma + beta
        } else {
            0
        }
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        if let Some(g) = &self.gamma {
            self.gamma = Some(g.to_device(device)?);
        }
        if let Some(b) = &self.beta {
            self.beta = Some(b.to_device(device)?);
        }
        self.device = device.clone();
        Ok(())
    }
}

/// Batch Normalization 1D configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm1dConfig {
    /// Number of features
    pub num_features: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Momentum for running mean/var
    pub momentum: f32,
    /// Whether to use learnable affine parameters
    pub affine: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
}

impl BatchNorm1dConfig {
    /// Create new config
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }

    /// Set epsilon
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

/// Batch Normalization 1D
///
/// Normalizes input across batch dimension:
/// y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
///
/// During training, uses batch statistics.
/// During inference, uses running statistics.
#[derive(Debug, Clone)]
pub struct BatchNorm1d {
    config: BatchNorm1dConfig,
    /// Scale parameter gamma
    gamma: Option<Tensor>,
    /// Shift parameter beta
    beta: Option<Tensor>,
    /// Running mean
    running_mean: Option<Tensor>,
    /// Running variance
    running_var: Option<Tensor>,
    /// Number of batches seen (for running stats)
    num_batches_tracked: usize,
    /// Training mode
    training: bool,
    device: Device,
}

impl BatchNorm1d {
    /// Create a new BatchNorm1d
    pub fn new(config: BatchNorm1dConfig, device: &Device) -> MlResult<Self> {
        let (gamma, beta) = if config.affine {
            let gamma = Tensor::ones(vec![config.num_features], DType::F32, device)?;
            let beta = Tensor::zeros(vec![config.num_features], DType::F32, device)?;
            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        let (running_mean, running_var) = if config.track_running_stats {
            let mean = Tensor::zeros(vec![config.num_features], DType::F32, device)?;
            let var = Tensor::ones(vec![config.num_features], DType::F32, device)?;
            (Some(mean), Some(var))
        } else {
            (None, None)
        };

        Ok(Self {
            config,
            gamma,
            beta,
            running_mean,
            running_var,
            num_batches_tracked: 0,
            training: true,
            device: device.clone(),
        })
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Layer for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        let input_shape = input.shape().dims();

        // Input can be [batch, features] or [batch, features, length]
        let is_3d = input_shape.len() == 3;
        let num_features = input_shape[1];

        if num_features != self.config.num_features {
            return Err(crate::error::MlError::shape_mismatch(
                vec![self.config.num_features],
                vec![num_features],
            ));
        }

        // Use running stats in eval mode, batch stats in train mode
        let (mean, var) = if self.training {
            // Compute batch statistics
            compute_batch_stats(input, is_3d)?
        } else if let (Some(rm), Some(rv)) = (&self.running_mean, &self.running_var) {
            (rm.clone(), rv.clone())
        } else {
            compute_batch_stats(input, is_3d)?
        };

        // Normalize
        batch_norm_forward(
            input,
            &mean,
            &var,
            self.gamma.as_ref(),
            self.beta.as_ref(),
            self.config.eps,
            is_3d,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        if self.config.affine {
            self.config.num_features * 2
        } else {
            0
        }
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        if let Some(g) = &self.gamma {
            self.gamma = Some(g.to_device(device)?);
        }
        if let Some(b) = &self.beta {
            self.beta = Some(b.to_device(device)?);
        }
        if let Some(rm) = &self.running_mean {
            self.running_mean = Some(rm.to_device(device)?);
        }
        if let Some(rv) = &self.running_var {
            self.running_var = Some(rv.to_device(device)?);
        }
        self.device = device.clone();
        Ok(())
    }
}

// Helper functions

fn layer_norm_forward(
    input: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
    norm_size: usize,
) -> MlResult<Tensor> {
    let input_shape = input.shape().dims();
    let total_size: usize = input_shape.iter().product();
    let num_instances = total_size / norm_size;

    #[cfg(feature = "cpu")]
    {
        if let Some(input_data) = input.as_slice() {
            let mut output_data = vec![0.0_f32; total_size];

            for i in 0..num_instances {
                let start = i * norm_size;
                let end = start + norm_size;
                let slice = &input_data[start..end];

                // Compute mean
                let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

                // Compute variance
                let var: f32 = slice
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>() / norm_size as f32;

                // Normalize
                let inv_std = 1.0 / (var + eps).sqrt();

                for j in 0..norm_size {
                    let normalized = (input_data[start + j] - mean) * inv_std;

                    // Apply affine transformation
                    let scaled = if let Some(g) = gamma {
                        if let Some(g_data) = g.as_slice() {
                            normalized * g_data[j]
                        } else {
                            normalized
                        }
                    } else {
                        normalized
                    };

                    let shifted = if let Some(b) = beta {
                        if let Some(b_data) = b.as_slice() {
                            scaled + b_data[j]
                        } else {
                            scaled
                        }
                    } else {
                        scaled
                    };

                    output_data[start + j] = shifted;
                }
            }

            return Tensor::from_slice(&output_data, input_shape.to_vec(), input.device());
        }
    }

    Err(crate::error::MlError::Other(
        "LayerNorm not implemented for this backend".to_string()
    ))
}

fn compute_batch_stats(input: &Tensor, is_3d: bool) -> MlResult<(Tensor, Tensor)> {
    let input_shape = input.shape().dims();
    let num_features = input_shape[1];

    #[cfg(feature = "cpu")]
    {
        if let Some(input_data) = input.as_slice() {
            let batch_size = input_shape[0];
            let length = if is_3d { input_shape[2] } else { 1 };
            let n = (batch_size * length) as f32;

            let mut means = vec![0.0_f32; num_features];
            let mut vars = vec![0.0_f32; num_features];

            // Compute means
            for b in 0..batch_size {
                for f in 0..num_features {
                    if is_3d {
                        for l in 0..length {
                            let idx = b * num_features * length + f * length + l;
                            means[f] += input_data[idx];
                        }
                    } else {
                        let idx = b * num_features + f;
                        means[f] += input_data[idx];
                    }
                }
            }
            for m in &mut means {
                *m /= n;
            }

            // Compute variances
            for b in 0..batch_size {
                for f in 0..num_features {
                    if is_3d {
                        for l in 0..length {
                            let idx = b * num_features * length + f * length + l;
                            vars[f] += (input_data[idx] - means[f]).powi(2);
                        }
                    } else {
                        let idx = b * num_features + f;
                        vars[f] += (input_data[idx] - means[f]).powi(2);
                    }
                }
            }
            for v in &mut vars {
                *v /= n;
            }

            let device = input.device();
            let mean_tensor = Tensor::from_slice(&means, vec![num_features], device)?;
            let var_tensor = Tensor::from_slice(&vars, vec![num_features], device)?;

            return Ok((mean_tensor, var_tensor));
        }
    }

    Err(crate::error::MlError::Other(
        "batch_stats not implemented for this backend".to_string()
    ))
}

fn batch_norm_forward(
    input: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
    is_3d: bool,
) -> MlResult<Tensor> {
    let input_shape = input.shape().dims();

    #[cfg(feature = "cpu")]
    {
        if let (Some(input_data), Some(mean_data), Some(var_data)) =
            (input.as_slice(), mean.as_slice(), var.as_slice())
        {
            let batch_size = input_shape[0];
            let num_features = input_shape[1];
            let length = if is_3d { input_shape[2] } else { 1 };
            let total_size: usize = input_shape.iter().product();

            let mut output_data = vec![0.0_f32; total_size];

            for b in 0..batch_size {
                for f in 0..num_features {
                    let inv_std = 1.0 / (var_data[f] + eps).sqrt();
                    let m = mean_data[f];

                    let g = gamma.and_then(|g| g.as_slice()).map(|d| d[f]).unwrap_or(1.0);
                    let bt = beta.and_then(|b| b.as_slice()).map(|d| d[f]).unwrap_or(0.0);

                    if is_3d {
                        for l in 0..length {
                            let idx = b * num_features * length + f * length + l;
                            let normalized = (input_data[idx] - m) * inv_std;
                            output_data[idx] = normalized * g + bt;
                        }
                    } else {
                        let idx = b * num_features + f;
                        let normalized = (input_data[idx] - m) * inv_std;
                        output_data[idx] = normalized * g + bt;
                    }
                }
            }

            return Tensor::from_slice(&output_data, input_shape.to_vec(), input.device());
        }
    }

    Err(crate::error::MlError::Other(
        "BatchNorm not implemented for this backend".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_creation() {
        let config = LayerNormConfig::new(512);
        let device = Device::Cpu;
        let norm = LayerNorm::new(config, &device).unwrap();

        assert_eq!(norm.num_parameters(), 1024); // gamma + beta
    }

    #[test]
    fn test_batch_norm_creation() {
        let config = BatchNorm1dConfig::new(64);
        let device = Device::Cpu;
        let norm = BatchNorm1d::new(config, &device).unwrap();

        assert_eq!(norm.num_parameters(), 128); // gamma + beta
        assert!(norm.is_training());
    }

    #[test]
    fn test_batch_norm_modes() {
        let config = BatchNorm1dConfig::new(64);
        let device = Device::Cpu;
        let mut norm = BatchNorm1d::new(config, &device).unwrap();

        assert!(norm.is_training());
        norm.eval();
        assert!(!norm.is_training());
        norm.train();
        assert!(norm.is_training());
    }
}
