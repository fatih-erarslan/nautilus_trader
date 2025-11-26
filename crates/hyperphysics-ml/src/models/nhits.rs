//! N-HiTS (Neural Hierarchical Interpolation for Time Series)
//!
//! Reference: Challu et al. (2022) "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting"
//! https://arxiv.org/abs/2201.12886
//!
//! Key innovations:
//! - Multi-rate signal sampling via hierarchical interpolation
//! - Maxpool input downsampling for each stack
//! - Interpolation-based output upsampling
//! - Enables handling of long sequences efficiently

use crate::backends::Device;
use crate::error::MlResult;
use crate::layers::{Layer, Linear, LinearConfig};
use crate::tensor::{DType, Tensor, TensorOps};
use super::{ForecastOutput, Forecaster};
use serde::{Deserialize, Serialize};

/// N-HiTS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHitsConfig {
    /// Input sequence length
    pub input_length: usize,
    /// Forecast horizon
    pub horizon: usize,
    /// Number of input features
    pub input_features: usize,
    /// Stack configurations (hidden_size, num_blocks, pooling_kernel)
    pub stacks: Vec<NHitsStackConfig>,
    /// Dropout probability
    pub dropout: f32,
    /// Activation function
    pub activation: Activation,
    /// Shared weights between blocks in a stack
    pub shared_weights: bool,
}

/// Configuration for a single N-HiTS stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHitsStackConfig {
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of blocks in this stack
    pub num_blocks: usize,
    /// Pooling kernel size (downsampling factor)
    pub pooling_kernel: usize,
    /// Number of basis coefficients for interpolation
    pub n_coefficients: usize,
}

impl Default for NHitsStackConfig {
    fn default() -> Self {
        Self {
            hidden_size: 256,
            num_blocks: 2,
            pooling_kernel: 1,
            n_coefficients: 8,
        }
    }
}

/// Activation function type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    ReLU,
    GELU,
    SiLU,
    Tanh,
}

impl Default for NHitsConfig {
    fn default() -> Self {
        Self {
            input_length: 96,
            horizon: 24,
            input_features: 1,
            stacks: vec![
                NHitsStackConfig {
                    hidden_size: 256,
                    num_blocks: 2,
                    pooling_kernel: 2,
                    n_coefficients: 8,
                },
                NHitsStackConfig {
                    hidden_size: 256,
                    num_blocks: 2,
                    pooling_kernel: 4,
                    n_coefficients: 4,
                },
                NHitsStackConfig {
                    hidden_size: 256,
                    num_blocks: 2,
                    pooling_kernel: 8,
                    n_coefficients: 2,
                },
            ],
            dropout: 0.1,
            activation: Activation::ReLU,
            shared_weights: false,
        }
    }
}

impl NHitsConfig {
    /// Create config with custom parameters
    pub fn new(input_length: usize, horizon: usize, input_features: usize) -> Self {
        Self {
            input_length,
            horizon,
            input_features,
            ..Default::default()
        }
    }

    /// Set stack configurations
    pub fn with_stacks(mut self, stacks: Vec<NHitsStackConfig>) -> Self {
        self.stacks = stacks;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// N-HiTS Block: MLP with residual connection
#[derive(Debug, Clone)]
pub struct NHitsBlock {
    /// Input projection
    fc1: Linear,
    /// Hidden layers
    fc2: Linear,
    fc3: Linear,
    /// Backcast output
    backcast_fc: Linear,
    /// Forecast coefficients
    forecast_fc: Linear,
    /// Pooled input size
    pooled_input_size: usize,
    /// Number of coefficients
    n_coefficients: usize,
    /// Horizon
    horizon: usize,
    device: Device,
}

impl NHitsBlock {
    /// Create a new N-HiTS block
    pub fn new(
        pooled_input_size: usize,
        hidden_size: usize,
        n_coefficients: usize,
        horizon: usize,
        device: &Device,
    ) -> MlResult<Self> {
        let fc1 = Linear::new(LinearConfig::new(pooled_input_size, hidden_size), device)?;
        let fc2 = Linear::new(LinearConfig::new(hidden_size, hidden_size), device)?;
        let fc3 = Linear::new(LinearConfig::new(hidden_size, hidden_size), device)?;

        // Backcast: predict coefficients for input reconstruction
        let backcast_fc = Linear::new(
            LinearConfig::new(hidden_size, pooled_input_size),
            device,
        )?;

        // Forecast: predict coefficients for output interpolation
        let forecast_fc = Linear::new(
            LinearConfig::new(hidden_size, n_coefficients),
            device,
        )?;

        Ok(Self {
            fc1,
            fc2,
            fc3,
            backcast_fc,
            forecast_fc,
            pooled_input_size,
            n_coefficients,
            horizon,
            device: device.clone(),
        })
    }

    /// Forward pass returning (backcast, forecast_coefficients)
    pub fn forward(&self, x: &Tensor) -> MlResult<(Tensor, Tensor)> {
        // x: [batch, pooled_input_size]
        let h = self.fc1.forward(x)?.relu()?;
        let h = self.fc2.forward(&h)?.relu()?;
        let h = self.fc3.forward(&h)?.relu()?;

        let backcast = self.backcast_fc.forward(&h)?;
        let forecast_coeff = self.forecast_fc.forward(&h)?;

        Ok((backcast, forecast_coeff))
    }
}

/// N-HiTS Stack: Collection of blocks with shared pooling
#[derive(Debug, Clone)]
pub struct NHitsStack {
    /// Blocks in this stack
    blocks: Vec<NHitsBlock>,
    /// Pooling kernel size
    pooling_kernel: usize,
    /// Number of coefficients
    n_coefficients: usize,
    /// Horizon
    horizon: usize,
    /// Interpolation basis (precomputed)
    basis: Tensor,
    device: Device,
}

impl NHitsStack {
    /// Create a new N-HiTS stack
    pub fn new(
        config: &NHitsStackConfig,
        input_length: usize,
        input_features: usize,
        horizon: usize,
        device: &Device,
    ) -> MlResult<Self> {
        // Compute pooled input size
        let pooled_length = input_length / config.pooling_kernel;
        let pooled_input_size = pooled_length * input_features;

        // Create blocks
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            let block = NHitsBlock::new(
                pooled_input_size,
                config.hidden_size,
                config.n_coefficients,
                horizon,
                device,
            )?;
            blocks.push(block);
        }

        // Precompute interpolation basis
        let basis = create_interpolation_basis(config.n_coefficients, horizon, device)?;

        Ok(Self {
            blocks,
            pooling_kernel: config.pooling_kernel,
            n_coefficients: config.n_coefficients,
            horizon,
            basis,
            device: device.clone(),
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> MlResult<(Tensor, Tensor)> {
        let shape = x.shape().dims();
        let batch_size = shape[0];

        // Apply maxpooling along sequence dimension
        let pooled = maxpool_1d(x, self.pooling_kernel)?;

        // Flatten for MLP
        let pooled_flat = pooled.reshape(vec![batch_size, pooled.shape().numel() / batch_size])?;

        // Process through blocks with residual
        let mut residual = pooled_flat.clone();
        let mut forecast_sum = Tensor::zeros(
            vec![batch_size, self.n_coefficients],
            DType::F32,
            &self.device,
        )?;

        for block in &self.blocks {
            let (backcast, forecast_coeff) = block.forward(&residual)?;
            residual = residual.sub(&backcast)?;
            forecast_sum = forecast_sum.add(&forecast_coeff)?;
        }

        // Interpolate forecast coefficients to full horizon
        // forecast = coefficients @ basis
        let forecast = forecast_sum.matmul(&self.basis)?;

        Ok((residual, forecast))
    }
}

/// N-HiTS Model
#[derive(Debug, Clone)]
pub struct NHits {
    config: NHitsConfig,
    /// Stacks for hierarchical processing
    stacks: Vec<NHitsStack>,
    /// Output projection
    output_proj: Linear,
    device: Device,
}

impl NHits {
    /// Create a new N-HiTS model
    pub fn new(config: NHitsConfig, device: &Device) -> MlResult<Self> {
        let mut stacks = Vec::with_capacity(config.stacks.len());

        for stack_config in &config.stacks {
            let stack = NHitsStack::new(
                stack_config,
                config.input_length,
                config.input_features,
                config.horizon,
                device,
            )?;
            stacks.push(stack);
        }

        // Output projection to match output features
        let output_proj = Linear::new(
            LinearConfig::new(config.horizon, config.horizon),
            device,
        )?;

        Ok(Self {
            config,
            stacks,
            output_proj,
            device: device.clone(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &NHitsConfig {
        &self.config
    }
}

impl Forecaster for NHits {
    fn forecast(&self, x: &Tensor) -> MlResult<Tensor> {
        let shape = x.shape().dims();
        let batch_size = shape[0];

        // Initialize forecast accumulator
        let mut forecast = Tensor::zeros(
            vec![batch_size, self.config.horizon],
            DType::F32,
            &self.device,
        )?;

        // Process through each stack
        for stack in &self.stacks {
            let (_, stack_forecast) = stack.forward(x)?;
            forecast = forecast.add(&stack_forecast)?;
        }

        // Reshape to [batch, horizon, 1]
        forecast.reshape(vec![batch_size, self.config.horizon, 1])
    }

    fn forecast_with_intervals(
        &self,
        x: &Tensor,
        _confidence: f32,
    ) -> MlResult<ForecastOutput> {
        // N-HiTS is deterministic, intervals would require ensemble or dropout
        let point = self.forecast(x)?;
        Ok(ForecastOutput {
            point,
            lower: None,
            upper: None,
            distribution_params: None,
        })
    }

    fn horizon(&self) -> usize {
        self.config.horizon
    }

    fn input_length(&self) -> usize {
        self.config.input_length
    }

    fn num_parameters(&self) -> usize {
        let mut count = 0;
        for stack in &self.stacks {
            for block in &stack.blocks {
                count += block.fc1.num_parameters();
                count += block.fc2.num_parameters();
                count += block.fc3.num_parameters();
                count += block.backcast_fc.num_parameters();
                count += block.forecast_fc.num_parameters();
            }
        }
        count += self.output_proj.num_parameters();
        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        for stack in &mut self.stacks {
            for block in &mut stack.blocks {
                block.fc1.to_device(device)?;
                block.fc2.to_device(device)?;
                block.fc3.to_device(device)?;
                block.backcast_fc.to_device(device)?;
                block.forecast_fc.to_device(device)?;
            }
            stack.basis = stack.basis.to_device(device)?;
            stack.device = device.clone();
        }
        self.output_proj.to_device(device)?;
        self.device = device.clone();
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// Helper functions

/// Create interpolation basis matrix for upsampling
fn create_interpolation_basis(n_coefficients: usize, horizon: usize, device: &Device) -> MlResult<Tensor> {
    // Use polynomial basis for interpolation
    let mut basis_data = vec![0.0_f32; n_coefficients * horizon];

    for c in 0..n_coefficients {
        for h in 0..horizon {
            let t = h as f32 / horizon as f32;
            // Legendre-like polynomial basis
            basis_data[c * horizon + h] = chebyshev_basis(c, t);
        }
    }

    Tensor::from_slice(&basis_data, vec![n_coefficients, horizon], device)
}

/// Chebyshev polynomial basis
fn chebyshev_basis(n: usize, t: f32) -> f32 {
    let x = 2.0 * t - 1.0; // Map to [-1, 1]
    match n {
        0 => 1.0,
        1 => x,
        _ => 2.0 * x * chebyshev_basis(n - 1, t) - chebyshev_basis(n - 2, t),
    }
}

/// 1D max pooling
fn maxpool_1d(x: &Tensor, kernel_size: usize) -> MlResult<Tensor> {
    if kernel_size == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape().dims();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let features = shape[2];
    let output_len = seq_len / kernel_size;

    #[cfg(feature = "cpu")]
    {
        if let Some(data) = x.as_slice() {
            let mut output = vec![f32::NEG_INFINITY; batch_size * output_len * features];

            for b in 0..batch_size {
                for o in 0..output_len {
                    for f in 0..features {
                        let mut max_val = f32::NEG_INFINITY;
                        for k in 0..kernel_size {
                            let t = o * kernel_size + k;
                            if t < seq_len {
                                let val = data[b * seq_len * features + t * features + f];
                                max_val = max_val.max(val);
                            }
                        }
                        output[b * output_len * features + o * features + f] = max_val;
                    }
                }
            }

            return Tensor::from_slice(&output, vec![batch_size, output_len, features], x.device());
        }
    }

    Err(crate::error::MlError::Other(
        "maxpool_1d not implemented for this backend".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nhits_config_default() {
        let config = NHitsConfig::default();
        assert_eq!(config.input_length, 96);
        assert_eq!(config.horizon, 24);
        assert_eq!(config.stacks.len(), 3);
    }

    #[test]
    fn test_chebyshev_basis() {
        assert!((chebyshev_basis(0, 0.5) - 1.0).abs() < 1e-6);
        assert!((chebyshev_basis(1, 0.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolation_basis() {
        let device = Device::Cpu;
        let basis = create_interpolation_basis(4, 24, &device).unwrap();
        assert_eq!(basis.shape().dims(), &[4, 24]);
    }
}
