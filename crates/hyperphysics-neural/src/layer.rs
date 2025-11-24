//! Neural network layers
//!
//! Provides dense, convolutional, and specialized layers for HFT applications.

use serde::{Deserialize, Serialize};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::activation::Activation;
use crate::core::{Tensor, TensorShape};
use crate::error::{NeuralError, NeuralResult};

/// Layer type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Fully connected / Dense layer
    Dense,
    /// 1D Convolution (for time series)
    Conv1D,
    /// Batch normalization
    BatchNorm,
    /// Layer normalization
    LayerNorm,
    /// Dropout
    Dropout,
    /// LSTM recurrent layer
    LSTM,
    /// GRU recurrent layer
    GRU,
    /// Multi-head attention
    MultiHeadAttention,
    /// Embedding layer
    Embedding,
    /// Residual connection wrapper
    Residual,
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: LayerType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: Activation,
    /// Use bias
    pub use_bias: bool,
    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f64,
    /// Layer-specific parameters
    pub params: LayerParams,
}

/// Layer-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayerParams {
    /// Kernel size for convolutions
    pub kernel_size: Option<usize>,
    /// Stride for convolutions
    pub stride: Option<usize>,
    /// Padding for convolutions
    pub padding: Option<usize>,
    /// Number of attention heads
    pub num_heads: Option<usize>,
    /// Hidden dimension for attention
    pub hidden_dim: Option<usize>,
    /// Epsilon for normalization layers
    pub eps: Option<f64>,
    /// Momentum for batch norm
    pub momentum: Option<f64>,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            layer_type: LayerType::Dense,
            input_dim: 0,
            output_dim: 0,
            activation: Activation::ReLU,
            use_bias: true,
            dropout_rate: 0.0,
            params: LayerParams::default(),
        }
    }
}

impl LayerConfig {
    /// Create dense layer config
    pub fn dense(input_dim: usize, output_dim: usize) -> Self {
        Self {
            layer_type: LayerType::Dense,
            input_dim,
            output_dim,
            activation: Activation::ReLU,
            use_bias: true,
            dropout_rate: 0.0,
            params: LayerParams::default(),
        }
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Disable bias
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }
}

/// Neural network layer with weights and biases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Layer configuration
    pub config: LayerConfig,
    /// Weight matrix
    weights: Tensor,
    /// Bias vector (optional)
    bias: Option<Tensor>,
    /// Cached input for backprop
    #[serde(skip)]
    cached_input: Option<Tensor>,
    /// Cached pre-activation output
    #[serde(skip)]
    cached_preact: Option<Tensor>,
}

impl Layer {
    /// Create new layer with random initialization
    pub fn new(config: LayerConfig) -> NeuralResult<Self> {
        Self::with_seed(config, 42)
    }

    /// Create new layer with specific seed
    pub fn with_seed(config: LayerConfig, seed: u64) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Initialize weights based on recommended initialization
        let weight_shape = TensorShape::d2(config.input_dim, config.output_dim);
        let weights = match config.activation.recommended_init() {
            "he" => Tensor::he(weight_shape, &mut rng),
            "lecun" => {
                // LeCun initialization: std = sqrt(1 / fan_in)
                use rand_distr::{Distribution, Normal};
                let std = (1.0 / config.input_dim as f64).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                let data: Vec<f64> = (0..config.input_dim * config.output_dim)
                    .map(|_| normal.sample(&mut rng))
                    .collect();
                Tensor::new(data, weight_shape)?
            },
            _ => Tensor::xavier(weight_shape, &mut rng),
        };

        let bias = if config.use_bias {
            Some(Tensor::zeros(TensorShape::d1(config.output_dim)))
        } else {
            None
        };

        Ok(Self {
            config,
            weights,
            bias,
            cached_input: None,
            cached_preact: None,
        })
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Tensor) -> NeuralResult<Tensor> {
        match self.config.layer_type {
            LayerType::Dense => self.forward_dense(input),
            LayerType::BatchNorm => self.forward_batchnorm(input),
            LayerType::LayerNorm => self.forward_layernorm(input),
            LayerType::Dropout => self.forward_dropout(input, true),
            _ => Err(NeuralError::InvalidLayerConfig(
                format!("Layer type {:?} not yet implemented", self.config.layer_type)
            )),
        }
    }

    /// Forward pass without caching (inference mode)
    pub fn forward_inference(&self, input: &Tensor) -> NeuralResult<Tensor> {
        match self.config.layer_type {
            LayerType::Dense => {
                // Linear transformation: y = xW + b
                let preact = input.matmul(&self.weights)?;
                let preact = if let Some(ref bias) = self.bias {
                    // Broadcast add bias
                    let mut result = preact;
                    for (i, val) in result.data_mut().iter_mut().enumerate() {
                        *val += bias.data()[i % bias.numel()];
                    }
                    result
                } else {
                    preact
                };
                Ok(self.config.activation.forward(&preact))
            },
            LayerType::Dropout => Ok(input.clone()), // No dropout in inference
            _ => self.forward_batchnorm(input),
        }
    }

    fn forward_dense(&mut self, input: &Tensor) -> NeuralResult<Tensor> {
        // Cache input for backprop
        self.cached_input = Some(input.clone());

        // Linear transformation: y = xW + b
        let preact = input.matmul(&self.weights)?;

        let preact = if let Some(ref bias) = self.bias {
            // Broadcast add bias
            let mut result = preact;
            for (i, val) in result.data_mut().iter_mut().enumerate() {
                *val += bias.data()[i % bias.numel()];
            }
            result
        } else {
            preact
        };

        self.cached_preact = Some(preact.clone());
        Ok(self.config.activation.forward(&preact))
    }

    fn forward_batchnorm(&self, input: &Tensor) -> NeuralResult<Tensor> {
        let eps = self.config.params.eps.unwrap_or(1e-5);
        let mean = input.mean();
        let variance = input.data().iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / input.numel() as f64;

        Ok(input.map(|x| (x - mean) / (variance + eps).sqrt()))
    }

    fn forward_layernorm(&self, input: &Tensor) -> NeuralResult<Tensor> {
        self.forward_batchnorm(input) // Same computation for single sample
    }

    fn forward_dropout(&self, input: &Tensor, training: bool) -> NeuralResult<Tensor> {
        if !training || self.config.dropout_rate <= 0.0 {
            return Ok(input.clone());
        }

        let mut rng = ChaCha8Rng::seed_from_u64(rand::random());
        let scale = 1.0 / (1.0 - self.config.dropout_rate);

        let data: Vec<f64> = input.data().iter().map(|&x| {
            if rand::Rng::gen::<f64>(&mut rng) < self.config.dropout_rate {
                0.0
            } else {
                x * scale
            }
        }).collect();

        Tensor::new(data, input.shape().clone())
    }

    /// Backward pass (returns gradient w.r.t. input)
    pub fn backward(&mut self, grad_output: &Tensor) -> NeuralResult<(Tensor, Tensor, Option<Tensor>)> {
        let cached_preact = self.cached_preact.as_ref()
            .ok_or_else(|| NeuralError::TrainingError("No cached preactivation".into()))?;
        let cached_input = self.cached_input.as_ref()
            .ok_or_else(|| NeuralError::TrainingError("No cached input".into()))?;

        // Gradient through activation
        let activation_grad = self.config.activation.backward(cached_preact);
        let grad_preact = grad_output.mul(&activation_grad)?;

        // Gradient w.r.t. weights: dL/dW = x^T * dL/dpreact
        let input_t = cached_input.transpose()?;
        let grad_weights = input_t.matmul(&grad_preact)?;

        // Gradient w.r.t. bias: sum of grad_preact
        let grad_bias = if self.bias.is_some() {
            Some(grad_preact.sum_axis(0)?)
        } else {
            None
        };

        // Gradient w.r.t. input: dL/dx = dL/dpreact * W^T
        let weights_t = self.weights.transpose()?;
        let grad_input = grad_preact.matmul(&weights_t)?;

        Ok((grad_input, grad_weights, grad_bias))
    }

    /// Update weights with gradients
    pub fn update(&mut self, grad_weights: &Tensor, grad_bias: Option<&Tensor>, lr: f64) {
        // SGD update: W = W - lr * grad_W
        let scaled_grad = grad_weights.scale(-lr);
        self.weights = self.weights.add(&scaled_grad).unwrap();

        if let (Some(ref mut bias), Some(grad_b)) = (&mut self.bias, grad_bias) {
            let scaled_grad_b = grad_b.scale(-lr);
            *bias = bias.add(&scaled_grad_b).unwrap();
        }
    }

    /// Get weights
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get bias
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Set weights (for loading)
    pub fn set_weights(&mut self, weights: Tensor) -> NeuralResult<()> {
        if weights.shape() != self.weights.shape() {
            return Err(NeuralError::ShapeMismatch {
                expected: self.weights.shape().0.clone(),
                actual: weights.shape().0.clone(),
            });
        }
        self.weights = weights;
        Ok(())
    }

    /// Set bias (for loading)
    pub fn set_bias(&mut self, bias: Tensor) -> NeuralResult<()> {
        if let Some(ref existing) = self.bias {
            if bias.shape() != existing.shape() {
                return Err(NeuralError::ShapeMismatch {
                    expected: existing.shape().0.clone(),
                    actual: bias.shape().0.clone(),
                });
            }
        }
        self.bias = Some(bias);
        Ok(())
    }

    /// Number of trainable parameters
    pub fn num_params(&self) -> usize {
        let w_params = self.weights.numel();
        let b_params = self.bias.as_ref().map_or(0, |b| b.numel());
        w_params + b_params
    }

    /// Input dimension
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let config = LayerConfig::dense(10, 5).with_activation(Activation::ReLU);
        let layer = Layer::new(config).unwrap();

        assert_eq!(layer.input_dim(), 10);
        assert_eq!(layer.output_dim(), 5);
        assert_eq!(layer.num_params(), 10 * 5 + 5); // weights + bias
    }

    #[test]
    fn test_layer_forward() {
        let config = LayerConfig::dense(4, 3).with_activation(Activation::Linear);
        let mut layer = Layer::new(config).unwrap();

        let input = Tensor::ones(TensorShape::d2(2, 4));
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().0, vec![2, 3]);
    }

    #[test]
    fn test_layer_without_bias() {
        let config = LayerConfig::dense(4, 3).without_bias();
        let layer = Layer::new(config).unwrap();

        assert!(layer.bias().is_none());
        assert_eq!(layer.num_params(), 4 * 3); // weights only
    }

    #[test]
    fn test_layer_gradient_shapes() {
        let config = LayerConfig::dense(4, 3).with_activation(Activation::ReLU);
        let mut layer = Layer::new(config).unwrap();

        let input = Tensor::ones(TensorShape::d2(2, 4));
        let _ = layer.forward(&input).unwrap();

        let grad_output = Tensor::ones(TensorShape::d2(2, 3));
        let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape().0, vec![2, 4]);
        assert_eq!(grad_weights.shape().0, vec![4, 3]);
        assert!(grad_bias.is_some());
    }
}
