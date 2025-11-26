//! N-BEATS (Neural Basis Expansion Analysis for Time Series)
//!
//! Reference: Oreshkin et al. (2019) "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
//! https://arxiv.org/abs/1905.10437
//!
//! Key innovations:
//! - Pure deep learning (no time series specific components)
//! - Interpretable architecture with trend/seasonality decomposition
//! - Doubly residual stacking
//! - Generic and interpretable variants

use crate::backends::Device;
use crate::error::MlResult;
use crate::layers::{Layer, Linear, LinearConfig};
use crate::tensor::{DType, Tensor, TensorOps};
use super::{ForecastOutput, Forecaster};
use serde::{Deserialize, Serialize};

/// N-BEATS stack type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum NBeatsStackType {
    /// Generic stack (learns any pattern)
    Generic,
    /// Trend stack (polynomial basis)
    Trend,
    /// Seasonality stack (Fourier basis)
    Seasonality,
}

/// N-BEATS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBeatsConfig {
    /// Input sequence length (lookback)
    pub input_length: usize,
    /// Forecast horizon
    pub horizon: usize,
    /// Stack configurations
    pub stacks: Vec<NBeatsStackConfig>,
    /// Shared weights between blocks
    pub shared_weights: bool,
}

/// Configuration for a single N-BEATS stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBeatsStackConfig {
    /// Stack type
    pub stack_type: NBeatsStackType,
    /// Number of blocks
    pub num_blocks: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of hidden layers per block
    pub num_layers: usize,
    /// Expansion coefficient dimension (for interpretable stacks)
    pub expansion_dim: usize,
}

impl Default for NBeatsStackConfig {
    fn default() -> Self {
        Self {
            stack_type: NBeatsStackType::Generic,
            num_blocks: 3,
            hidden_size: 256,
            num_layers: 4,
            expansion_dim: 32,
        }
    }
}

impl NBeatsStackConfig {
    /// Create generic stack config
    pub fn generic(num_blocks: usize, hidden_size: usize) -> Self {
        Self {
            stack_type: NBeatsStackType::Generic,
            num_blocks,
            hidden_size,
            num_layers: 4,
            expansion_dim: hidden_size,
        }
    }

    /// Create trend stack config
    pub fn trend(num_blocks: usize, hidden_size: usize, poly_degree: usize) -> Self {
        Self {
            stack_type: NBeatsStackType::Trend,
            num_blocks,
            hidden_size,
            num_layers: 4,
            expansion_dim: poly_degree + 1,  // Polynomial degree + constant
        }
    }

    /// Create seasonality stack config
    pub fn seasonality(num_blocks: usize, hidden_size: usize, num_harmonics: usize) -> Self {
        Self {
            stack_type: NBeatsStackType::Seasonality,
            num_blocks,
            hidden_size,
            num_layers: 4,
            expansion_dim: 2 * num_harmonics,  // Sin + Cos per harmonic
        }
    }
}

impl Default for NBeatsConfig {
    fn default() -> Self {
        Self {
            input_length: 96,
            horizon: 24,
            stacks: vec![
                NBeatsStackConfig::trend(3, 256, 3),
                NBeatsStackConfig::seasonality(3, 256, 8),
                NBeatsStackConfig::generic(3, 256),
            ],
            shared_weights: false,
        }
    }
}

impl NBeatsConfig {
    /// Create interpretable config (trend + seasonality)
    pub fn interpretable(input_length: usize, horizon: usize) -> Self {
        Self {
            input_length,
            horizon,
            stacks: vec![
                NBeatsStackConfig::trend(3, 256, 3),
                NBeatsStackConfig::seasonality(3, 256, 8),
            ],
            shared_weights: false,
        }
    }

    /// Create generic config
    pub fn generic(input_length: usize, horizon: usize, num_stacks: usize) -> Self {
        Self {
            input_length,
            horizon,
            stacks: (0..num_stacks)
                .map(|_| NBeatsStackConfig::generic(3, 256))
                .collect(),
            shared_weights: false,
        }
    }
}

/// N-BEATS Block
#[derive(Debug, Clone)]
pub struct NBeatsBlock {
    /// MLP layers
    fc_layers: Vec<Linear>,
    /// Backcast linear
    theta_backcast: Linear,
    /// Forecast linear
    theta_forecast: Linear,
    /// Stack type
    stack_type: NBeatsStackType,
    /// Backcast basis (precomputed for interpretable)
    backcast_basis: Option<Tensor>,
    /// Forecast basis (precomputed for interpretable)
    forecast_basis: Option<Tensor>,
    /// Input length
    input_length: usize,
    /// Horizon
    horizon: usize,
    device: Device,
}

impl NBeatsBlock {
    /// Create a new N-BEATS block
    pub fn new(
        config: &NBeatsStackConfig,
        input_length: usize,
        horizon: usize,
        device: &Device,
    ) -> MlResult<Self> {
        // Create MLP layers
        let mut fc_layers = Vec::with_capacity(config.num_layers);
        let mut input_dim = input_length;

        for _ in 0..config.num_layers {
            let layer = Linear::new(
                LinearConfig::new(input_dim, config.hidden_size),
                device,
            )?;
            fc_layers.push(layer);
            input_dim = config.hidden_size;
        }

        // Output dimensions depend on stack type
        let (backcast_dim, forecast_dim) = match config.stack_type {
            NBeatsStackType::Generic => (input_length, horizon),
            NBeatsStackType::Trend | NBeatsStackType::Seasonality => {
                (config.expansion_dim, config.expansion_dim)
            }
        };

        let theta_backcast = Linear::new(
            LinearConfig::new(config.hidden_size, backcast_dim).with_bias(false),
            device,
        )?;

        let theta_forecast = Linear::new(
            LinearConfig::new(config.hidden_size, forecast_dim).with_bias(false),
            device,
        )?;

        // Create basis for interpretable stacks
        let (backcast_basis, forecast_basis) = match config.stack_type {
            NBeatsStackType::Generic => (None, None),
            NBeatsStackType::Trend => {
                let b_basis = create_polynomial_basis(config.expansion_dim, input_length, device)?;
                let f_basis = create_polynomial_basis(config.expansion_dim, horizon, device)?;
                (Some(b_basis), Some(f_basis))
            }
            NBeatsStackType::Seasonality => {
                let b_basis = create_fourier_basis(config.expansion_dim / 2, input_length, device)?;
                let f_basis = create_fourier_basis(config.expansion_dim / 2, horizon, device)?;
                (Some(b_basis), Some(f_basis))
            }
        };

        Ok(Self {
            fc_layers,
            theta_backcast,
            theta_forecast,
            stack_type: config.stack_type,
            backcast_basis,
            forecast_basis,
            input_length,
            horizon,
            device: device.clone(),
        })
    }

    /// Forward pass returning (backcast, forecast)
    pub fn forward(&self, x: &Tensor) -> MlResult<(Tensor, Tensor)> {
        // x: [batch, input_length]
        let mut h = x.clone();

        // MLP forward
        for layer in &self.fc_layers {
            h = layer.forward(&h)?.relu()?;
        }

        // Compute theta parameters
        let theta_b = self.theta_backcast.forward(&h)?;
        let theta_f = self.theta_forecast.forward(&h)?;

        // Apply basis expansion for interpretable stacks
        let (backcast, forecast) = match self.stack_type {
            NBeatsStackType::Generic => {
                (theta_b, theta_f)
            }
            NBeatsStackType::Trend | NBeatsStackType::Seasonality => {
                let b_basis = self.backcast_basis.as_ref().unwrap();
                let f_basis = self.forecast_basis.as_ref().unwrap();

                // backcast = theta_b @ basis_b
                let backcast = theta_b.matmul(b_basis)?;
                let forecast = theta_f.matmul(f_basis)?;

                (backcast, forecast)
            }
        };

        Ok((backcast, forecast))
    }
}

/// N-BEATS Stack
#[derive(Debug, Clone)]
pub struct NBeatsStack {
    /// Blocks in this stack
    blocks: Vec<NBeatsBlock>,
    /// Stack type
    stack_type: NBeatsStackType,
    device: Device,
}

impl NBeatsStack {
    /// Create a new N-BEATS stack
    pub fn new(
        config: &NBeatsStackConfig,
        input_length: usize,
        horizon: usize,
        device: &Device,
    ) -> MlResult<Self> {
        let mut blocks = Vec::with_capacity(config.num_blocks);

        for _ in 0..config.num_blocks {
            let block = NBeatsBlock::new(config, input_length, horizon, device)?;
            blocks.push(block);
        }

        Ok(Self {
            blocks,
            stack_type: config.stack_type,
            device: device.clone(),
        })
    }

    /// Forward pass with doubly residual connections
    pub fn forward(&self, x: &Tensor) -> MlResult<(Tensor, Tensor)> {
        let shape = x.shape().dims();
        let batch_size = shape[0];

        let mut residual = x.clone();
        let mut forecast_sum = Tensor::zeros(
            vec![batch_size, self.blocks[0].horizon],
            DType::F32,
            &self.device,
        )?;

        for block in &self.blocks {
            let (backcast, forecast) = block.forward(&residual)?;

            // Subtract backcast from residual
            residual = residual.sub(&backcast)?;

            // Add forecast
            forecast_sum = forecast_sum.add(&forecast)?;
        }

        Ok((residual, forecast_sum))
    }
}

/// N-BEATS Model
#[derive(Debug, Clone)]
pub struct NBeats {
    config: NBeatsConfig,
    /// Stacks
    stacks: Vec<NBeatsStack>,
    device: Device,
}

impl NBeats {
    /// Create a new N-BEATS model
    pub fn new(config: NBeatsConfig, device: &Device) -> MlResult<Self> {
        let mut stacks = Vec::with_capacity(config.stacks.len());

        for stack_config in &config.stacks {
            let stack = NBeatsStack::new(
                stack_config,
                config.input_length,
                config.horizon,
                device,
            )?;
            stacks.push(stack);
        }

        Ok(Self {
            config,
            stacks,
            device: device.clone(),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &NBeatsConfig {
        &self.config
    }

    /// Get stack decomposition for interpretability
    pub fn decompose(&self, x: &Tensor) -> MlResult<Vec<Tensor>> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Flatten if needed
        let input = if shape.len() == 3 {
            x.reshape(vec![batch_size, seq_len * shape[2]])?
        } else {
            x.clone()
        };

        let mut decomposition = Vec::with_capacity(self.stacks.len());
        let mut residual = input.clone();

        for stack in &self.stacks {
            let (new_residual, forecast) = stack.forward(&residual)?;
            decomposition.push(forecast);
            residual = new_residual;
        }

        Ok(decomposition)
    }
}

impl Forecaster for NBeats {
    fn forecast(&self, x: &Tensor) -> MlResult<Tensor> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Flatten input: [batch, seq, features] -> [batch, seq * features]
        let input = if shape.len() == 3 {
            x.reshape(vec![batch_size, seq_len * shape[2]])?
        } else {
            x.clone()
        };

        // Process through stacks with doubly residual
        let mut residual = input.clone();
        let mut forecast = Tensor::zeros(
            vec![batch_size, self.config.horizon],
            DType::F32,
            &self.device,
        )?;

        for stack in &self.stacks {
            let (new_residual, stack_forecast) = stack.forward(&residual)?;
            residual = new_residual;
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
        // N-BEATS is deterministic
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
                for layer in &block.fc_layers {
                    count += layer.num_parameters();
                }
                count += block.theta_backcast.num_parameters();
                count += block.theta_forecast.num_parameters();
            }
        }
        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        for stack in &mut self.stacks {
            for block in &mut stack.blocks {
                for layer in &mut block.fc_layers {
                    layer.to_device(device)?;
                }
                block.theta_backcast.to_device(device)?;
                block.theta_forecast.to_device(device)?;
                if let Some(ref b) = block.backcast_basis {
                    block.backcast_basis = Some(b.to_device(device)?);
                }
                if let Some(ref f) = block.forecast_basis {
                    block.forecast_basis = Some(f.to_device(device)?);
                }
                block.device = device.clone();
            }
            stack.device = device.clone();
        }
        self.device = device.clone();
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// Helper functions for basis creation

/// Create polynomial basis for trend decomposition
fn create_polynomial_basis(degree: usize, length: usize, device: &Device) -> MlResult<Tensor> {
    let mut basis_data = vec![0.0_f32; degree * length];

    for d in 0..degree {
        for t in 0..length {
            let x = t as f32 / length as f32;
            basis_data[d * length + t] = x.powi(d as i32);
        }
    }

    Tensor::from_slice(&basis_data, vec![degree, length], device)
}

/// Create Fourier basis for seasonality decomposition
fn create_fourier_basis(num_harmonics: usize, length: usize, device: &Device) -> MlResult<Tensor> {
    let num_coeffs = 2 * num_harmonics;
    let mut basis_data = vec![0.0_f32; num_coeffs * length];

    for h in 0..num_harmonics {
        let freq = 2.0 * std::f32::consts::PI * (h + 1) as f32;
        for t in 0..length {
            let x = t as f32 / length as f32;
            // Sin component
            basis_data[(2 * h) * length + t] = (freq * x).sin();
            // Cos component
            basis_data[(2 * h + 1) * length + t] = (freq * x).cos();
        }
    }

    Tensor::from_slice(&basis_data, vec![num_coeffs, length], device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbeats_config_interpretable() {
        let config = NBeatsConfig::interpretable(96, 24);
        assert_eq!(config.stacks.len(), 2);
        assert_eq!(config.stacks[0].stack_type, NBeatsStackType::Trend);
        assert_eq!(config.stacks[1].stack_type, NBeatsStackType::Seasonality);
    }

    #[test]
    fn test_polynomial_basis() {
        let device = Device::Cpu;
        let basis = create_polynomial_basis(4, 24, &device).unwrap();
        assert_eq!(basis.shape().dims(), &[4, 24]);

        // First row (degree 0) should be all 1s... no wait, x^0 = 1
        // Actually for normalized x in [0, 1], first row is 1^0 = 1
    }

    #[test]
    fn test_fourier_basis() {
        let device = Device::Cpu;
        let basis = create_fourier_basis(4, 24, &device).unwrap();
        // 4 harmonics = 8 coefficients (sin + cos each)
        assert_eq!(basis.shape().dims(), &[8, 24]);
    }

    #[test]
    fn test_nbeats_creation() {
        let device = Device::Cpu;
        let config = NBeatsConfig::generic(96, 24, 3);
        let model = NBeats::new(config, &device).unwrap();

        assert_eq!(model.horizon(), 24);
        assert_eq!(model.input_length(), 96);
    }
}
