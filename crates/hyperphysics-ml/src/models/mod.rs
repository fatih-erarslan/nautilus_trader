//! Forecasting models for HFT neural predictions
//!
//! Implements state-of-the-art time-series forecasting architectures:
//! - N-HiTS (Neural Hierarchical Interpolation for Time Series)
//! - DeepAR (Autoregressive Probabilistic Forecasting)
//! - N-BEATS (Neural Basis Expansion Analysis)

mod nhits;
mod deepar;
mod nbeats;

pub use nhits::{NHits, NHitsConfig, NHitsBlock, NHitsStack};
pub use deepar::{DeepAR, DeepARConfig};
pub use nbeats::{NBeats, NBeatsConfig, NBeatsBlock, NBeatsStack, NBeatsStackType};

use crate::backends::Device;
use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};

/// Forecast output with point predictions and optional intervals
#[derive(Debug, Clone)]
pub struct ForecastOutput {
    /// Point forecast [batch, horizon, features]
    pub point: Tensor,
    /// Lower bound (if probabilistic) [batch, horizon, features]
    pub lower: Option<Tensor>,
    /// Upper bound (if probabilistic) [batch, horizon, features]
    pub upper: Option<Tensor>,
    /// Full distribution parameters (for DeepAR)
    pub distribution_params: Option<DistributionParams>,
}

/// Parameters for output distribution
#[derive(Debug, Clone)]
pub struct DistributionParams {
    /// Mean/location parameter
    pub mu: Tensor,
    /// Scale/variance parameter
    pub sigma: Tensor,
    /// Distribution type
    pub distribution: DistributionType,
}

/// Supported output distributions
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistributionType {
    /// Gaussian (normal) distribution
    Gaussian,
    /// Student's t-distribution (heavier tails)
    StudentT,
    /// Negative binomial (for count data)
    NegativeBinomial,
}

/// Common trait for all forecasting models
pub trait Forecaster: Send + Sync {
    /// Generate point forecasts
    ///
    /// Args:
    ///     x: Input sequence [batch, seq_len, features]
    ///
    /// Returns:
    ///     Forecast tensor [batch, horizon, features]
    fn forecast(&self, x: &Tensor) -> MlResult<Tensor>;

    /// Generate forecasts with uncertainty estimates
    fn forecast_with_intervals(
        &self,
        x: &Tensor,
        confidence: f32,
    ) -> MlResult<ForecastOutput>;

    /// Get forecast horizon
    fn horizon(&self) -> usize;

    /// Get required input length
    fn input_length(&self) -> usize;

    /// Get number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Move model to device
    fn to_device(&mut self, device: &Device) -> MlResult<()>;

    /// Get current device
    fn device(&self) -> &Device;
}

/// Model configuration common to all forecasters
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Input sequence length (lookback window)
    pub input_length: usize,
    /// Forecast horizon
    pub horizon: usize,
    /// Number of input features
    pub input_features: usize,
    /// Number of output features
    pub output_features: usize,
    /// Whether to use quantile forecasts
    pub quantile_forecasting: bool,
    /// Quantiles to predict (e.g., [0.1, 0.5, 0.9])
    pub quantiles: Vec<f32>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            input_length: 96,    // ~1.5 hours at minute frequency
            horizon: 24,         // 24 steps ahead
            input_features: 5,   // OHLCV
            output_features: 1,  // Predict close price
            quantile_forecasting: false,
            quantiles: vec![0.1, 0.5, 0.9],
        }
    }
}

/// Utility: Generate positional encoding for sequence models
pub fn positional_encoding(seq_len: usize, d_model: usize, device: &Device) -> MlResult<Tensor> {
    let mut pe_data = vec![0.0_f32; seq_len * d_model];

    for pos in 0..seq_len {
        for i in 0..d_model {
            let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
            pe_data[pos * d_model + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }

    Tensor::from_slice(&pe_data, vec![seq_len, d_model], device)
}

/// Utility: Apply reversible instance normalization for better training
pub fn reversible_instance_norm(x: &Tensor, eps: f32) -> MlResult<(Tensor, Tensor, Tensor)> {
    // Compute mean and std along time dimension
    let shape = x.shape().dims();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let features = shape[2];

    #[cfg(feature = "cpu")]
    {
        if let Some(data) = x.as_slice() {
            let mut means = vec![0.0_f32; batch_size * features];
            let mut stds = vec![0.0_f32; batch_size * features];
            let mut normalized = vec![0.0_f32; batch_size * seq_len * features];

            // Compute mean per (batch, feature)
            for b in 0..batch_size {
                for f in 0..features {
                    let mut sum = 0.0_f32;
                    for t in 0..seq_len {
                        sum += data[b * seq_len * features + t * features + f];
                    }
                    means[b * features + f] = sum / seq_len as f32;
                }
            }

            // Compute std per (batch, feature)
            for b in 0..batch_size {
                for f in 0..features {
                    let mean = means[b * features + f];
                    let mut var_sum = 0.0_f32;
                    for t in 0..seq_len {
                        let val = data[b * seq_len * features + t * features + f];
                        var_sum += (val - mean).powi(2);
                    }
                    stds[b * features + f] = (var_sum / seq_len as f32 + eps).sqrt();
                }
            }

            // Normalize
            for b in 0..batch_size {
                for t in 0..seq_len {
                    for f in 0..features {
                        let idx = b * seq_len * features + t * features + f;
                        let mean = means[b * features + f];
                        let std = stds[b * features + f];
                        normalized[idx] = (data[idx] - mean) / std;
                    }
                }
            }

            let device = x.device();
            let normalized_tensor = Tensor::from_slice(&normalized, shape.to_vec(), device)?;
            let mean_tensor = Tensor::from_slice(&means, vec![batch_size, features], device)?;
            let std_tensor = Tensor::from_slice(&stds, vec![batch_size, features], device)?;

            return Ok((normalized_tensor, mean_tensor, std_tensor));
        }
    }

    Err(crate::error::MlError::Other(
        "reversible_instance_norm not implemented for this backend".to_string()
    ))
}

/// Utility: Reverse instance normalization
pub fn reverse_instance_norm(x: &Tensor, mean: &Tensor, std: &Tensor) -> MlResult<Tensor> {
    let shape = x.shape().dims();
    let batch_size = shape[0];
    let horizon = shape[1];
    let features = shape[2];

    #[cfg(feature = "cpu")]
    {
        if let (Some(x_data), Some(mean_data), Some(std_data)) =
            (x.as_slice(), mean.as_slice(), std.as_slice())
        {
            let mut denormalized = vec![0.0_f32; batch_size * horizon * features];

            for b in 0..batch_size {
                for t in 0..horizon {
                    for f in 0..features {
                        let idx = b * horizon * features + t * features + f;
                        let m = mean_data[b * features + f];
                        let s = std_data[b * features + f];
                        denormalized[idx] = x_data[idx] * s + m;
                    }
                }
            }

            return Tensor::from_slice(&denormalized, shape.to_vec(), x.device());
        }
    }

    Err(crate::error::MlError::Other(
        "reverse_instance_norm not implemented for this backend".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_config_default() {
        let config = ForecastConfig::default();
        assert_eq!(config.input_length, 96);
        assert_eq!(config.horizon, 24);
    }

    #[test]
    fn test_positional_encoding() {
        let device = Device::Cpu;
        let pe = positional_encoding(100, 64, &device).unwrap();
        assert_eq!(pe.shape().dims(), &[100, 64]);
    }
}
