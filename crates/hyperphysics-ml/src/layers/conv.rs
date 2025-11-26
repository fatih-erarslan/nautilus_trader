//! Convolutional layers for temporal pattern recognition

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{DType, Tensor, TensorOps};
use super::Layer;
use serde::{Deserialize, Serialize};

/// 1D Convolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1dConfig {
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Dilation factor
    pub dilation: usize,
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Whether to use bias
    pub bias: bool,
}

impl Conv1dConfig {
    /// Create new Conv1d config
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
        }
    }

    /// Set stride
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set dilation
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set bias
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Compute output length for given input length
    pub fn output_length(&self, input_length: usize) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        (input_length + 2 * self.padding - effective_kernel) / self.stride + 1
    }

    /// Compute padding for 'same' output size (stride=1)
    pub fn same_padding(&self) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        (effective_kernel - 1) / 2
    }

    /// Compute causal padding (left-only for autoregressive)
    pub fn causal_padding(&self) -> usize {
        self.dilation * (self.kernel_size - 1)
    }
}

/// 1D Convolutional layer
///
/// Input: [batch, channels, length]
/// Output: [batch, out_channels, output_length]
#[derive(Debug, Clone)]
pub struct Conv1d {
    config: Conv1dConfig,
    /// Weight [out_channels, in_channels/groups, kernel_size]
    weight: Tensor,
    /// Bias [out_channels]
    bias: Option<Tensor>,
    device: Device,
}

impl Conv1d {
    /// Create a new Conv1d layer
    pub fn new(config: Conv1dConfig, device: &Device) -> MlResult<Self> {
        let in_channels_per_group = config.in_channels / config.groups;

        // Kaiming initialization
        let fan_in = in_channels_per_group * config.kernel_size;
        let scale = (2.0 / fan_in as f32).sqrt();

        let weight = Tensor::randn(
            vec![config.out_channels, in_channels_per_group, config.kernel_size],
            device,
        )?;
        let weight = scale_tensor(&weight, scale, device)?;

        let bias = if config.bias {
            Some(Tensor::zeros(vec![config.out_channels], DType::F32, device)?)
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

    /// Get config
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }
}

impl Layer for Conv1d {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // Input: [batch, in_channels, length]
        let input_shape = input.shape().dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_length = input_shape[2];

        if in_channels != self.config.in_channels {
            return Err(crate::error::MlError::shape_mismatch(
                vec![batch_size, self.config.in_channels, input_length],
                input_shape.to_vec(),
            ));
        }

        // Apply padding
        let padded = if self.config.padding > 0 {
            input.pad_1d(self.config.padding)?
        } else {
            input.clone()
        };

        // Compute convolution via im2col (unfolding)
        let output = conv1d_forward(
            &padded,
            &self.weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;

        Ok(output)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        let weight_params = self.config.out_channels
            * (self.config.in_channels / self.config.groups)
            * self.config.kernel_size;
        let bias_params = if self.config.bias {
            self.config.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(b) = &self.bias {
            self.bias = Some(b.to_device(device)?);
        }
        self.device = device.clone();
        Ok(())
    }
}

/// Temporal Convolutional Network (TCN) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcnConfig {
    /// Input feature dimension
    pub input_size: usize,
    /// Hidden channels per layer
    pub hidden_channels: usize,
    /// Output feature dimension
    pub output_size: usize,
    /// Kernel size for all layers
    pub kernel_size: usize,
    /// Number of TCN layers (receptive field grows exponentially)
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl TcnConfig {
    /// Create new TCN config
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_channels: 64,
            output_size,
            kernel_size: 3,
            num_layers: 4,
            dropout: 0.1,
        }
    }

    /// Set hidden channels
    pub fn with_hidden_channels(mut self, channels: usize) -> Self {
        self.hidden_channels = channels;
        self
    }

    /// Set kernel size
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = size;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }

    /// Compute receptive field
    pub fn receptive_field(&self) -> usize {
        // With exponential dilation: 1, 2, 4, 8, ...
        // RF = 1 + (kernel_size - 1) * sum_{i=0}^{n-1} 2^i
        // RF = 1 + (kernel_size - 1) * (2^n - 1)
        1 + (self.kernel_size - 1) * ((1 << self.num_layers) - 1)
    }
}

impl Default for TcnConfig {
    fn default() -> Self {
        Self::new(64, 32)
    }
}

/// Temporal Block: Conv1d -> Normalization -> ReLU -> Dropout -> Conv1d -> Residual
#[derive(Debug, Clone)]
pub struct TemporalBlock {
    /// First convolution
    conv1: Conv1d,
    /// Second convolution
    conv2: Conv1d,
    /// Residual connection (if dimensions differ)
    residual: Option<Conv1d>,
    /// Dilation factor
    dilation: usize,
    /// Padding (for causal convolution)
    padding: usize,
    device: Device,
}

impl TemporalBlock {
    /// Create a new temporal block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        device: &Device,
    ) -> MlResult<Self> {
        let padding = dilation * (kernel_size - 1);

        let conv1 = Conv1d::new(
            Conv1dConfig::new(in_channels, out_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(padding),
            device,
        )?;

        let conv2 = Conv1d::new(
            Conv1dConfig::new(out_channels, out_channels, kernel_size)
                .with_dilation(dilation)
                .with_padding(padding),
            device,
        )?;

        // Residual connection if dimensions differ
        let residual = if in_channels != out_channels {
            Some(Conv1d::new(
                Conv1dConfig::new(in_channels, out_channels, 1).with_bias(false),
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            residual,
            dilation,
            padding,
            device: device.clone(),
        })
    }
}

impl Layer for TemporalBlock {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // First conv -> ReLU
        let mut out = self.conv1.forward(input)?.relu()?;

        // Chomp to maintain causal property (remove future timesteps)
        let out_len = out.shape().dims()[2];
        if self.padding > 0 && out_len > self.padding {
            out = out.slice(2, 0, out_len - self.padding)?;
        }

        // Second conv -> ReLU
        out = self.conv2.forward(&out)?.relu()?;

        let out_len = out.shape().dims()[2];
        if self.padding > 0 && out_len > self.padding {
            out = out.slice(2, 0, out_len - self.padding)?;
        }

        // Residual connection
        let residual = if let Some(res_conv) = &self.residual {
            res_conv.forward(input)?
        } else {
            input.clone()
        };

        // Match dimensions for residual add
        let res_len = residual.shape().dims()[2];
        let out_len = out.shape().dims()[2];
        let residual = if res_len > out_len {
            residual.slice(2, 0, out_len)?
        } else {
            residual
        };

        out.add(&residual)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        let mut count = self.conv1.num_parameters() + self.conv2.num_parameters();
        if let Some(res) = &self.residual {
            count += res.num_parameters();
        }
        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.conv1.to_device(device)?;
        self.conv2.to_device(device)?;
        if let Some(res) = &mut self.residual {
            res.to_device(device)?;
        }
        self.device = device.clone();
        Ok(())
    }
}

/// Temporal Convolutional Network (TCN)
///
/// Implements WaveNet-style architecture with exponentially increasing dilations
/// for efficient long-range temporal dependencies.
#[derive(Debug, Clone)]
pub struct Tcn {
    config: TcnConfig,
    /// Input projection
    input_proj: Conv1d,
    /// Temporal blocks with increasing dilation
    blocks: Vec<TemporalBlock>,
    /// Output projection
    output_proj: Conv1d,
    device: Device,
}

impl Tcn {
    /// Create a new TCN
    pub fn new(config: TcnConfig, device: &Device) -> MlResult<Self> {
        // Input projection
        let input_proj = Conv1d::new(
            Conv1dConfig::new(config.input_size, config.hidden_channels, 1),
            device,
        )?;

        // Temporal blocks with exponential dilation
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let dilation = 1 << i; // 1, 2, 4, 8, ...
            let block = TemporalBlock::new(
                config.hidden_channels,
                config.hidden_channels,
                config.kernel_size,
                dilation,
                device,
            )?;
            blocks.push(block);
        }

        // Output projection
        let output_proj = Conv1d::new(
            Conv1dConfig::new(config.hidden_channels, config.output_size, 1),
            device,
        )?;

        Ok(Self {
            config,
            input_proj,
            blocks,
            output_proj,
            device: device.clone(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &TcnConfig {
        &self.config
    }

    /// Get receptive field
    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field()
    }
}

impl Layer for Tcn {
    fn forward(&self, input: &Tensor) -> MlResult<Tensor> {
        // Input: [batch, length, features] -> [batch, features, length]
        let x = input.transpose_dims(1, 2)?;

        // Input projection
        let mut x = self.input_proj.forward(&x)?;

        // Process through temporal blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Output projection
        let x = self.output_proj.forward(&x)?;

        // Back to [batch, length, features]
        x.transpose_dims(1, 2)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn num_parameters(&self) -> usize {
        let mut count = self.input_proj.num_parameters() + self.output_proj.num_parameters();
        for block in &self.blocks {
            count += block.num_parameters();
        }
        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.input_proj.to_device(device)?;
        for block in &mut self.blocks {
            block.to_device(device)?;
        }
        self.output_proj.to_device(device)?;
        self.device = device.clone();
        Ok(())
    }
}

// Helper functions

fn scale_tensor(tensor: &Tensor, scale: f32, device: &Device) -> MlResult<Tensor> {
    #[cfg(feature = "cpu")]
    {
        if let Some(data) = tensor.as_slice() {
            let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
            return Tensor::from_slice(&scaled, tensor.shape().dims().to_vec(), device);
        }
    }
    Ok(tensor.clone())
}

/// Conv1d forward using im2col approach
fn conv1d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> MlResult<Tensor> {
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let output_length = (input_length - effective_kernel) / stride + 1;

    // For simplicity, implement direct convolution for CPU
    // GPU backends would use optimized kernels

    #[cfg(feature = "cpu")]
    {
        let mut output_data = vec![0.0_f32; batch_size * out_channels * output_length];

        if let (Some(input_data), Some(weight_data)) = (input.as_slice(), weight.as_slice()) {
            for b in 0..batch_size {
                for oc in 0..out_channels {
                    let group = oc / (out_channels / groups);
                    let ic_start = group * (in_channels / groups);
                    let ic_end = ic_start + (in_channels / groups);

                    for o in 0..output_length {
                        let mut sum = 0.0_f32;

                        for ic in ic_start..ic_end {
                            let ic_local = ic - ic_start;
                            for k in 0..kernel_size {
                                let i = o * stride + k * dilation;
                                if i < input_length {
                                    let input_idx = b * in_channels * input_length
                                        + ic * input_length + i;
                                    let weight_idx = oc * (in_channels / groups) * kernel_size
                                        + ic_local * kernel_size + k;
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }

                        output_data[b * out_channels * output_length + oc * output_length + o] = sum;
                    }
                }
            }

            // Add bias
            if let Some(b) = bias {
                if let Some(bias_data) = b.as_slice() {
                    for batch in 0..batch_size {
                        for oc in 0..out_channels {
                            for o in 0..output_length {
                                let idx = batch * out_channels * output_length + oc * output_length + o;
                                output_data[idx] += bias_data[oc];
                            }
                        }
                    }
                }
            }

            return Tensor::from_slice(
                &output_data,
                vec![batch_size, out_channels, output_length],
                input.device(),
            );
        }
    }

    // Fallback for non-CPU backends
    Err(crate::error::MlError::Other(
        "Conv1d not implemented for this backend".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_creation() {
        let config = Conv1dConfig::new(64, 128, 3);
        let device = Device::Cpu;
        let conv = Conv1d::new(config, &device).unwrap();

        assert_eq!(conv.config().out_channels, 128);
        assert_eq!(conv.config().kernel_size, 3);
    }

    #[test]
    fn test_tcn_receptive_field() {
        let config = TcnConfig::new(64, 32)
            .with_kernel_size(3)
            .with_num_layers(4);

        // RF = 1 + 2 * (16 - 1) = 31
        assert_eq!(config.receptive_field(), 31);
    }

    #[test]
    fn test_conv_output_length() {
        let config = Conv1dConfig::new(64, 128, 3)
            .with_stride(1)
            .with_padding(1);

        // With padding=1, stride=1, kernel=3: out = in
        assert_eq!(config.output_length(100), 100);
    }
}
