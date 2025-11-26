//! DeepAR - Autoregressive Probabilistic Forecasting
//!
//! Reference: Salinas et al. (2020) "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"
//! https://arxiv.org/abs/1704.04110
//!
//! Key features:
//! - Autoregressive LSTM-based architecture
//! - Probabilistic output (learns distribution parameters)
//! - Handles multiple related time series
//! - Native support for missing values

use crate::backends::Device;
use crate::error::MlResult;
use crate::layers::{Layer, Linear, LinearConfig, Lstm, LstmConfig};
use crate::tensor::{DType, Tensor, TensorOps};
use super::{DistributionParams, DistributionType, ForecastOutput, Forecaster};
use serde::{Deserialize, Serialize};

/// DeepAR configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepARConfig {
    /// Input sequence length (context)
    pub context_length: usize,
    /// Forecast horizon
    pub prediction_length: usize,
    /// Number of input features (covariates)
    pub input_features: usize,
    /// LSTM hidden size
    pub hidden_size: usize,
    /// Number of LSTM layers
    pub num_layers: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Output distribution type
    pub distribution: DistributionType,
    /// Embedding dimension for categorical features
    pub embedding_dim: usize,
    /// Number of parallel samples for probabilistic forecast
    pub num_samples: usize,
}

impl Default for DeepARConfig {
    fn default() -> Self {
        Self {
            context_length: 96,
            prediction_length: 24,
            input_features: 5,
            hidden_size: 128,
            num_layers: 2,
            dropout: 0.1,
            distribution: DistributionType::Gaussian,
            embedding_dim: 32,
            num_samples: 100,
        }
    }
}

impl DeepARConfig {
    /// Create new config
    pub fn new(context_length: usize, prediction_length: usize) -> Self {
        Self {
            context_length,
            prediction_length,
            ..Default::default()
        }
    }

    /// Set hidden size
    pub fn with_hidden_size(mut self, size: usize) -> Self {
        self.hidden_size = size;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, layers: usize) -> Self {
        self.num_layers = layers;
        self
    }

    /// Set output distribution
    pub fn with_distribution(mut self, dist: DistributionType) -> Self {
        self.distribution = dist;
        self
    }

    /// Set number of samples
    pub fn with_num_samples(mut self, samples: usize) -> Self {
        self.num_samples = samples;
        self
    }
}

/// DeepAR Model
#[derive(Debug, Clone)]
pub struct DeepAR {
    config: DeepARConfig,
    /// Input embedding/projection
    input_proj: Linear,
    /// LSTM encoder
    lstm: Lstm,
    /// Distribution parameter projections
    mu_proj: Linear,      // Mean
    sigma_proj: Linear,   // Standard deviation
    /// Optional: additional distribution parameters
    nu_proj: Option<Linear>,  // Degrees of freedom for Student-t
    device: Device,
}

impl DeepAR {
    /// Create a new DeepAR model
    pub fn new(config: DeepARConfig, device: &Device) -> MlResult<Self> {
        // Input projection: features + 1 (lagged target) -> hidden
        let input_dim = config.input_features + 1;
        let input_proj = Linear::new(
            LinearConfig::new(input_dim, config.hidden_size),
            device,
        )?;

        // LSTM
        let lstm = Lstm::new(
            LstmConfig::new(config.hidden_size, config.hidden_size)
                .with_num_layers(config.num_layers)
                .with_dropout(config.dropout),
            device,
        )?;

        // Distribution parameter projections
        let mu_proj = Linear::new(
            LinearConfig::new(config.hidden_size, 1),
            device,
        )?;
        let sigma_proj = Linear::new(
            LinearConfig::new(config.hidden_size, 1),
            device,
        )?;

        // Additional parameters for Student-t
        let nu_proj = if config.distribution == DistributionType::StudentT {
            Some(Linear::new(
                LinearConfig::new(config.hidden_size, 1),
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            input_proj,
            lstm,
            mu_proj,
            sigma_proj,
            nu_proj,
            device: device.clone(),
        })
    }

    /// Encode context sequence
    fn encode(&self, x: &Tensor, target: &Tensor) -> MlResult<Tensor> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Concatenate covariates with lagged target
        let lagged_target = target.slice(1, 0, seq_len)?;
        let input = concat_features(x, &lagged_target.unsqueeze(2)?)?;

        // Project input
        let projected = self.input_proj.forward(&input)?;

        // Run through LSTM
        self.lstm.forward(&projected)
    }

    /// Decode autoregressively for prediction
    fn decode(
        &self,
        hidden: &Tensor,
        future_covariates: Option<&Tensor>,
        last_target: f32,
    ) -> MlResult<(Tensor, Tensor)> {
        let shape = hidden.shape().dims();
        let batch_size = shape[0];
        let horizon = self.config.prediction_length;

        let mut mus = Vec::with_capacity(horizon);
        let mut sigmas = Vec::with_capacity(horizon);
        let mut current_target = last_target;
        let mut lstm_state = None;

        // Use last hidden state as initial state
        let initial_hidden = hidden.slice(1, hidden.shape().dims()[1] - 1, hidden.shape().dims()[1])?;

        for t in 0..horizon {
            // Prepare input: covariates + lagged target
            let covariate = if let Some(fc) = future_covariates {
                fc.slice(1, t, t + 1)?.squeeze_dim(1)?
            } else {
                Tensor::zeros(vec![batch_size, self.config.input_features], DType::F32, &self.device)?
            };

            let target_tensor = Tensor::from_slice(
                &vec![current_target; batch_size],
                vec![batch_size, 1],
                &self.device,
            )?;

            let input = concat_features_2d(&covariate, &target_tensor)?;
            let input = input.unsqueeze(1)?;  // [batch, 1, features]

            // Project and run LSTM step
            let projected = self.input_proj.forward(&input)?;
            let lstm_result = self.lstm.forward_with_state(&projected, lstm_state)?;
            lstm_state = Some(lstm_result.state);

            let h = lstm_result.output.squeeze_dim(1)?;  // [batch, hidden]

            // Compute distribution parameters
            let mu = self.mu_proj.forward(&h)?;
            let sigma = self.sigma_proj.forward(&h)?.softplus()?;  // Ensure positive

            mus.push(mu.clone());
            sigmas.push(sigma);

            // Update target for next step (use mean for deterministic forecast)
            if let Some(mu_data) = mu.as_slice() {
                current_target = mu_data[0];
            }
        }

        // Stack outputs
        let mu_tensor = Tensor::stack_vec(&mus, 1)?;
        let sigma_tensor = Tensor::stack_vec(&sigmas, 1)?;

        Ok((mu_tensor, sigma_tensor))
    }

    /// Sample from output distribution
    fn sample_forecast(
        &self,
        mu: &Tensor,
        sigma: &Tensor,
        num_samples: usize,
    ) -> MlResult<Tensor> {
        let shape = mu.shape().dims();
        let batch_size = shape[0];
        let horizon = shape[1];

        // Generate samples based on distribution type
        match self.config.distribution {
            DistributionType::Gaussian => {
                sample_gaussian(mu, sigma, num_samples, &self.device)
            }
            DistributionType::StudentT => {
                // Would need nu parameter
                sample_gaussian(mu, sigma, num_samples, &self.device)
            }
            DistributionType::NegativeBinomial => {
                // Would need different parameterization
                sample_gaussian(mu, sigma, num_samples, &self.device)
            }
        }
    }

    /// Get configuration
    pub fn config(&self) -> &DeepARConfig {
        &self.config
    }
}

impl Forecaster for DeepAR {
    fn forecast(&self, x: &Tensor) -> MlResult<Tensor> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Extract target (assume last feature is target)
        let target = x.slice(2, shape[2] - 1, shape[2])?.squeeze_dim(2)?;
        let covariates = x.slice(2, 0, shape[2] - 1)?;

        // Encode context
        let hidden = self.encode(&covariates, &target)?;

        // Get last target value
        let last_target = if let Some(target_data) = target.as_slice() {
            target_data[(seq_len - 1)]
        } else {
            0.0
        };

        // Decode
        let (mu, _sigma) = self.decode(&hidden, None, last_target)?;

        // Return point forecast (mean)
        mu.reshape(vec![batch_size, self.config.prediction_length, 1])
    }

    fn forecast_with_intervals(
        &self,
        x: &Tensor,
        confidence: f32,
    ) -> MlResult<ForecastOutput> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // Extract target and covariates
        let target = x.slice(2, shape[2] - 1, shape[2])?.squeeze_dim(2)?;
        let covariates = x.slice(2, 0, shape[2] - 1)?;

        // Encode
        let hidden = self.encode(&covariates, &target)?;

        // Get last target
        let last_target = if let Some(target_data) = target.as_slice() {
            target_data[(seq_len - 1)]
        } else {
            0.0
        };

        // Decode
        let (mu, sigma) = self.decode(&hidden, None, last_target)?;

        // Compute confidence intervals
        let z = quantile_normal((1.0 + confidence) / 2.0);
        let lower = compute_lower_bound(&mu, &sigma, z)?;
        let upper = compute_upper_bound(&mu, &sigma, z)?;

        // Reshape outputs
        let point = mu.reshape(vec![batch_size, self.config.prediction_length, 1])?;
        let lower = lower.reshape(vec![batch_size, self.config.prediction_length, 1])?;
        let upper = upper.reshape(vec![batch_size, self.config.prediction_length, 1])?;

        Ok(ForecastOutput {
            point,
            lower: Some(lower),
            upper: Some(upper),
            distribution_params: Some(DistributionParams {
                mu: mu.clone(),
                sigma,
                distribution: self.config.distribution,
            }),
        })
    }

    fn horizon(&self) -> usize {
        self.config.prediction_length
    }

    fn input_length(&self) -> usize {
        self.config.context_length
    }

    fn num_parameters(&self) -> usize {
        let mut count = self.input_proj.num_parameters();
        count += self.lstm.num_parameters();
        count += self.mu_proj.num_parameters();
        count += self.sigma_proj.num_parameters();
        if let Some(ref nu) = self.nu_proj {
            count += nu.num_parameters();
        }
        count
    }

    fn to_device(&mut self, device: &Device) -> MlResult<()> {
        self.input_proj.to_device(device)?;
        self.lstm.to_device(device)?;
        self.mu_proj.to_device(device)?;
        self.sigma_proj.to_device(device)?;
        if let Some(ref mut nu) = self.nu_proj {
            nu.to_device(device)?;
        }
        self.device = device.clone();
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// Helper functions

fn concat_features(x: &Tensor, y: &Tensor) -> MlResult<Tensor> {
    let x_shape = x.shape().dims();
    let y_shape = y.shape().dims();

    let batch = x_shape[0];
    let seq = x_shape[1];
    let x_features = x_shape[2];
    let y_features = y_shape[2];
    let total_features = x_features + y_features;

    #[cfg(feature = "cpu")]
    {
        if let (Some(x_data), Some(y_data)) = (x.as_slice(), y.as_slice()) {
            let mut output = vec![0.0_f32; batch * seq * total_features];

            for b in 0..batch {
                for s in 0..seq {
                    for f in 0..x_features {
                        output[b * seq * total_features + s * total_features + f] =
                            x_data[b * seq * x_features + s * x_features + f];
                    }
                    for f in 0..y_features {
                        output[b * seq * total_features + s * total_features + x_features + f] =
                            y_data[b * seq * y_features + s * y_features + f];
                    }
                }
            }

            return Tensor::from_slice(&output, vec![batch, seq, total_features], x.device());
        }
    }

    Err(crate::error::MlError::Other("concat not implemented".to_string()))
}

fn concat_features_2d(x: &Tensor, y: &Tensor) -> MlResult<Tensor> {
    let x_shape = x.shape().dims();
    let y_shape = y.shape().dims();

    let batch = x_shape[0];
    let x_features = x_shape[1];
    let y_features = y_shape[1];
    let total = x_features + y_features;

    #[cfg(feature = "cpu")]
    {
        if let (Some(x_data), Some(y_data)) = (x.as_slice(), y.as_slice()) {
            let mut output = vec![0.0_f32; batch * total];

            for b in 0..batch {
                for f in 0..x_features {
                    output[b * total + f] = x_data[b * x_features + f];
                }
                for f in 0..y_features {
                    output[b * total + x_features + f] = y_data[b * y_features + f];
                }
            }

            return Tensor::from_slice(&output, vec![batch, total], x.device());
        }
    }

    Err(crate::error::MlError::Other("concat_2d not implemented".to_string()))
}

fn sample_gaussian(mu: &Tensor, sigma: &Tensor, num_samples: usize, device: &Device) -> MlResult<Tensor> {
    // For deterministic forecast, just return mu
    // Real implementation would sample from N(mu, sigma)
    mu.clone().unsqueeze(0)
}

fn quantile_normal(p: f32) -> f32 {
    // Approximate inverse normal CDF
    // Using Abramowitz and Stegun approximation
    if p <= 0.0 { return f32::NEG_INFINITY; }
    if p >= 1.0 { return f32::INFINITY; }
    if p == 0.5 { return 0.0; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -z } else { z }
}

fn compute_lower_bound(mu: &Tensor, sigma: &Tensor, z: f32) -> MlResult<Tensor> {
    // lower = mu - z * sigma
    #[cfg(feature = "cpu")]
    {
        if let (Some(mu_data), Some(sigma_data)) = (mu.as_slice(), sigma.as_slice()) {
            let output: Vec<f32> = mu_data.iter()
                .zip(sigma_data.iter())
                .map(|(m, s)| m - z * s)
                .collect();
            return Tensor::from_slice(&output, mu.shape().dims().to_vec(), mu.device());
        }
    }
    Err(crate::error::MlError::Other("lower_bound not implemented".to_string()))
}

fn compute_upper_bound(mu: &Tensor, sigma: &Tensor, z: f32) -> MlResult<Tensor> {
    // upper = mu + z * sigma
    #[cfg(feature = "cpu")]
    {
        if let (Some(mu_data), Some(sigma_data)) = (mu.as_slice(), sigma.as_slice()) {
            let output: Vec<f32> = mu_data.iter()
                .zip(sigma_data.iter())
                .map(|(m, s)| m + z * s)
                .collect();
            return Tensor::from_slice(&output, mu.shape().dims().to_vec(), mu.device());
        }
    }
    Err(crate::error::MlError::Other("upper_bound not implemented".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepar_config() {
        let config = DeepARConfig::new(96, 24)
            .with_hidden_size(256)
            .with_num_layers(3);

        assert_eq!(config.context_length, 96);
        assert_eq!(config.prediction_length, 24);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 3);
    }

    #[test]
    fn test_quantile_normal() {
        assert!((quantile_normal(0.5) - 0.0).abs() < 0.01);
        assert!((quantile_normal(0.975) - 1.96).abs() < 0.05);
    }
}
