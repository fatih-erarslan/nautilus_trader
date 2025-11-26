//! Linear (fully connected) layer

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{DType, Tensor, TensorOps};
use super::Layer;
use serde::{Deserialize, Serialize};

/// Linear layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// Whether to use bias
    pub bias: bool,
}

impl LinearConfig {
    /// Create new config
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    /// Set bias
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// Linear (fully connected) layer: y = xW^T + b
#[derive(Debug, Clone)]
pub struct Linear {
    /// Configuration
    config: LinearConfig,
    /// Weight matrix [out_features, in_features]
    weight: Tensor,
    /// Bias vector [out_features]
    bias: Option<Tensor>,
    /// Device
    device: Device,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(config: LinearConfig, device: &Device) -> MlResult<Self> {
        // Xavier initialization
        let scale = (2.0 / (config.in_features + config.out_features) as f32).sqrt();
        let mut weight = Tensor::randn(
            vec![config.out_features, config.in_features],
            device,
        )?;

        // Scale weights
        #[cfg(feature = "cpu")]
        {
            if let Some(data) = weight.as_slice() {
                let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
                weight = Tensor::from_slice(
                    &scaled,
                    vec![config.out_features, config.in_features],
                    device,
                )?;
            }
        }

        let bias = if config.bias {
            Some(Tensor::zeros(vec![config.out_features], DType::F32, device)?)
        } else {
            None
        };

        Ok(Self {
            config,
            weight,
            bias,
            device: device.clone(),
        })
    }

    /// Get weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get bias tensor
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // input: [batch, ..., in_features]
        // output: [batch, ..., out_features]

        let input_shape = input.shape().dims();
        let in_features = *input_shape.last().unwrap();

        if in_features != self.config.in_features {
            return Err(crate::error::MlError::shape_mismatch(
                vec![self.config.in_features],
                vec![in_features],
            ));
        }

        // Compute xW^T
        let output = input.matmul(&self.weight.transpose()?)?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        let weight_params = self.config.in_features * self.config.out_features;
        let bias_params = if self.config.bias {
            self.config.out_features
        } else {
            0
        };
        weight_params + bias_params
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(bias) = &self.bias {
            self.bias = Some(bias.to_device(device)?);
        }
        self.device = device.clone();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let config = LinearConfig::new(64, 128);
        let device = Device::Cpu;
        let linear = Linear::new(config, &device).unwrap();

        assert_eq!(linear.weight().shape().dims(), &[128, 64]);
        assert_eq!(linear.num_parameters(), 64 * 128 + 128);
    }

    #[test]
    fn test_linear_forward() {
        let config = LinearConfig::new(4, 8);
        let device = Device::Cpu;
        let linear = Linear::new(config, &device).unwrap();

        let input = Tensor::randn(vec![2, 4], &device).unwrap();
        let output = linear.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 8]);
    }
}
