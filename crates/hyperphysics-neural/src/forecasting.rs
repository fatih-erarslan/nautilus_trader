//! Time-series forecasting models for HFT applications
//!
//! Implements Neuro-Divergent forecasting models optimized for market prediction.
//!
//! ## Models
//!
//! - **N-BEATS**: Neural Basis Expansion Analysis for Time Series
//! - **LSTM**: Long Short-Term Memory networks
//! - **Transformer**: Attention-based sequence modeling
//! - **TCN**: Temporal Convolutional Networks
//! - **DeepAR**: Probabilistic forecasting with autoregressive RNN

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::activation::Activation;
use crate::core::{Tensor, TensorShape};
use crate::error::{NeuralError, NeuralResult};
use crate::network::Network;

/// Time series model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeSeriesModel {
    /// N-BEATS (Neural Basis Expansion Analysis)
    NBEATS,
    /// Long Short-Term Memory
    LSTM,
    /// Transformer encoder
    Transformer,
    /// Temporal Convolutional Network
    TCN,
    /// DeepAR probabilistic forecasting
    DeepAR,
    /// Simple MLP baseline
    MLP,
    /// Exponential smoothing neural hybrid
    ESHybrid,
}

impl Default for TimeSeriesModel {
    fn default() -> Self {
        TimeSeriesModel::MLP
    }
}

/// Forecasting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastConfig {
    /// Model type to use
    pub model: TimeSeriesModel,
    /// Lookback window (history length)
    pub lookback: usize,
    /// Forecast horizon (prediction length)
    pub horizon: usize,
    /// Number of input features (multivariate)
    pub num_features: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Use probabilistic output (mean + variance)
    pub probabilistic: bool,
    /// Maximum inference latency target (microseconds)
    pub max_latency_us: Option<u64>,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        Self {
            model: TimeSeriesModel::MLP,
            lookback: 60, // 1 minute at 1Hz, or 60 ticks
            horizon: 10,  // Predict next 10 steps
            num_features: 1,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.1,
            probabilistic: false,
            max_latency_us: Some(100), // 100μs target for HFT
        }
    }
}

impl ForecastConfig {
    /// Create config for HFT applications (ultra-low latency)
    pub fn hft() -> Self {
        Self {
            model: TimeSeriesModel::MLP,
            lookback: 20,
            horizon: 1,
            num_features: 5, // OHLCV
            hidden_dim: 32,
            num_layers: 2,
            dropout: 0.0,
            probabilistic: false,
            max_latency_us: Some(10), // 10μs target
        }
    }

    /// Create config for market regime detection
    pub fn regime_detection() -> Self {
        Self {
            model: TimeSeriesModel::LSTM,
            lookback: 100,
            horizon: 1,
            num_features: 10,
            hidden_dim: 64,
            num_layers: 2,
            dropout: 0.2,
            probabilistic: true,
            max_latency_us: Some(1000), // 1ms acceptable
        }
    }
}

/// Time series forecaster
#[derive(Debug)]
pub struct Forecaster {
    /// Configuration
    config: ForecastConfig,
    /// Underlying network
    network: Network,
    /// Inference statistics
    inference_count: u64,
    total_latency_us: u64,
}

impl Forecaster {
    /// Create new forecaster from configuration
    pub fn new(config: ForecastConfig) -> NeuralResult<Self> {
        let network = Self::build_network(&config)?;
        Ok(Self {
            config,
            network,
            inference_count: 0,
            total_latency_us: 0,
        })
    }

    /// Build network based on model type
    fn build_network(config: &ForecastConfig) -> NeuralResult<Network> {
        let input_dim = config.lookback * config.num_features;
        let output_dim = if config.probabilistic {
            config.horizon * 2 // mean + variance
        } else {
            config.horizon
        };

        match config.model {
            TimeSeriesModel::MLP | TimeSeriesModel::NBEATS => {
                // Simple MLP or N-BEATS-like stacked architecture
                let mut builder = Network::builder()
                    .name(format!("{:?}", config.model))
                    .input_dim(input_dim)
                    .output_dim(output_dim)
                    .hidden_activation(Activation::ReLU)
                    .output_activation(Activation::Linear)
                    .dropout(config.dropout);

                for _ in 0..config.num_layers {
                    builder = builder.hidden(config.hidden_dim);
                }

                builder.build()
            },
            TimeSeriesModel::LSTM => {
                // For LSTM/GRU, we use MLP as simplified version
                // Full RNN implementation would require separate recurrent layer
                Network::builder()
                    .name("LSTM-like")
                    .input_dim(input_dim)
                    .hidden(config.hidden_dim * 2)
                    .hidden(config.hidden_dim)
                    .output_dim(output_dim)
                    .hidden_activation(Activation::Tanh)
                    .output_activation(Activation::Linear)
                    .build()
            },
            TimeSeriesModel::Transformer => {
                // Simplified transformer (feedforward approximation)
                // Full attention would require dedicated attention layer
                Network::builder()
                    .name("Transformer-like")
                    .input_dim(input_dim)
                    .hidden(config.hidden_dim * 4)
                    .hidden(config.hidden_dim * 2)
                    .hidden(config.hidden_dim)
                    .output_dim(output_dim)
                    .hidden_activation(Activation::GELU)
                    .output_activation(Activation::Linear)
                    .build()
            },
            TimeSeriesModel::TCN => {
                // TCN approximation with dilated-like structure
                Network::builder()
                    .name("TCN-like")
                    .input_dim(input_dim)
                    .hidden(config.hidden_dim)
                    .hidden(config.hidden_dim)
                    .hidden(config.hidden_dim)
                    .output_dim(output_dim)
                    .hidden_activation(Activation::ReLU)
                    .output_activation(Activation::Linear)
                    .dropout(config.dropout)
                    .build()
            },
            TimeSeriesModel::DeepAR | TimeSeriesModel::ESHybrid => {
                // Probabilistic output network
                Network::builder()
                    .name("DeepAR-like")
                    .input_dim(input_dim)
                    .hidden(config.hidden_dim)
                    .hidden(config.hidden_dim)
                    .output_dim(output_dim)
                    .hidden_activation(Activation::ReLU)
                    .output_activation(Activation::Linear)
                    .build()
            }
        }
    }

    /// Forecast future values from historical data
    pub fn forecast(&mut self, history: &[f64]) -> NeuralResult<ForecastResult> {
        let start = Instant::now();

        // Validate input
        let expected_len = self.config.lookback * self.config.num_features;
        if history.len() < expected_len {
            return Err(NeuralError::DimensionMismatch {
                input_dim: history.len(),
                expected_dim: expected_len,
            });
        }

        // Take last lookback * num_features values
        let input_data: Vec<f64> = history.iter()
            .rev()
            .take(expected_len)
            .rev()
            .copied()
            .collect();

        let input = Tensor::new(input_data, TensorShape::d2(1, expected_len))?;
        let output = self.network.forward(&input)?;

        let latency_us = start.elapsed().as_micros() as u64;
        self.inference_count += 1;
        self.total_latency_us += latency_us;

        // Check latency constraint
        if let Some(max_us) = self.config.max_latency_us {
            if latency_us > max_us {
                tracing::warn!(
                    "Forecast latency {}μs exceeded target {}μs",
                    latency_us,
                    max_us
                );
            }
        }

        // Parse output
        let output_data = output.data();
        if self.config.probabilistic {
            let mid = output_data.len() / 2;
            let predictions = output_data[..mid].to_vec();
            let variances: Vec<f64> = output_data[mid..].iter()
                .map(|&v| v.exp()) // Variance is predicted in log-space
                .collect();

            Ok(ForecastResult {
                predictions,
                variances: Some(variances),
                latency_us,
            })
        } else {
            Ok(ForecastResult {
                predictions: output_data.to_vec(),
                variances: None,
                latency_us,
            })
        }
    }

    /// Forecast with confidence intervals
    pub fn forecast_with_confidence(
        &mut self,
        history: &[f64],
        confidence_level: f64,
    ) -> NeuralResult<ForecastWithConfidence> {
        let result = self.forecast(history)?;

        let (lower, upper) = if let Some(ref vars) = result.variances {
            // Use predicted variance for confidence intervals
            let z = Self::z_score(confidence_level);
            let lower: Vec<f64> = result.predictions.iter()
                .zip(vars.iter())
                .map(|(p, v)| p - z * v.sqrt())
                .collect();
            let upper: Vec<f64> = result.predictions.iter()
                .zip(vars.iter())
                .map(|(p, v)| p + z * v.sqrt())
                .collect();
            (lower, upper)
        } else {
            // No variance prediction, use fixed uncertainty
            let uncertainty = 0.1; // 10% of prediction magnitude
            let lower: Vec<f64> = result.predictions.iter()
                .map(|p| p * (1.0 - uncertainty))
                .collect();
            let upper: Vec<f64> = result.predictions.iter()
                .map(|p| p * (1.0 + uncertainty))
                .collect();
            (lower, upper)
        };

        Ok(ForecastWithConfidence {
            predictions: result.predictions,
            lower_bound: lower,
            upper_bound: upper,
            confidence_level,
            latency_us: result.latency_us,
        })
    }

    /// Z-score for confidence level
    fn z_score(confidence: f64) -> f64 {
        match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            c if c >= 0.80 => 1.282,
            _ => 1.0,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ForecastConfig {
        &self.config
    }

    /// Get average inference latency
    pub fn avg_latency_us(&self) -> f64 {
        if self.inference_count == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.inference_count as f64
        }
    }

    /// Get underlying network
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Get mutable network (for training)
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }
}

/// Forecast result
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Predicted values for each horizon step
    pub predictions: Vec<f64>,
    /// Optional variance estimates (for probabilistic models)
    pub variances: Option<Vec<f64>>,
    /// Inference latency in microseconds
    pub latency_us: u64,
}

/// Forecast with confidence intervals
#[derive(Debug, Clone)]
pub struct ForecastWithConfidence {
    /// Point predictions
    pub predictions: Vec<f64>,
    /// Lower confidence bound
    pub lower_bound: Vec<f64>,
    /// Upper confidence bound
    pub upper_bound: Vec<f64>,
    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,
    /// Inference latency in microseconds
    pub latency_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecaster_creation() {
        let config = ForecastConfig::default();
        let forecaster = Forecaster::new(config).unwrap();

        assert_eq!(forecaster.config().lookback, 60);
        assert_eq!(forecaster.config().horizon, 10);
    }

    #[test]
    fn test_forecaster_predict() {
        let config = ForecastConfig {
            lookback: 10,
            horizon: 3,
            num_features: 1,
            hidden_dim: 16,
            num_layers: 1,
            ..Default::default()
        };

        let mut forecaster = Forecaster::new(config).unwrap();
        let history: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();

        let result = forecaster.forecast(&history).unwrap();
        assert_eq!(result.predictions.len(), 3);
        assert!(result.latency_us > 0);
    }

    #[test]
    fn test_hft_config() {
        let config = ForecastConfig::hft();
        let forecaster = Forecaster::new(config).unwrap();

        assert_eq!(forecaster.config().max_latency_us, Some(10));
        assert_eq!(forecaster.config().lookback, 20);
        assert_eq!(forecaster.config().horizon, 1);
    }

    #[test]
    fn test_probabilistic_forecast() {
        let config = ForecastConfig {
            lookback: 10,
            horizon: 3,
            num_features: 1,
            hidden_dim: 16,
            probabilistic: true,
            ..Default::default()
        };

        let mut forecaster = Forecaster::new(config).unwrap();
        let history: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();

        let result = forecaster.forecast(&history).unwrap();
        assert!(result.variances.is_some());
        assert_eq!(result.variances.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_forecast_with_confidence() {
        let config = ForecastConfig {
            lookback: 10,
            horizon: 3,
            num_features: 1,
            hidden_dim: 16,
            probabilistic: true,
            ..Default::default()
        };

        let mut forecaster = Forecaster::new(config).unwrap();
        let history: Vec<f64> = (0..20).map(|i| i as f64).collect();

        let result = forecaster.forecast_with_confidence(&history, 0.95).unwrap();
        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.lower_bound.len(), 3);
        assert_eq!(result.upper_bound.len(), 3);

        // Upper should be >= predictions >= lower
        for i in 0..3 {
            assert!(result.upper_bound[i] >= result.predictions[i]);
            assert!(result.predictions[i] >= result.lower_bound[i]);
        }
    }
}
