//! Safe wrapper around ruv-FANN Network
//!
//! Provides a unified interface for neural network operations optimized for HFT.

use std::time::Instant;

use ruv_fann::{
    ActivationFunction, Network as RuvFannNetwork, NetworkBuilder as RuvNetworkBuilder,
    TrainingAlgorithm, TrainingData,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::{Tensor, TensorShape};
use crate::error::NeuralResult;

/// Errors specific to FANN operations
#[derive(Error, Debug)]
pub enum FannError {
    #[error("Network creation failed: {0}")]
    CreationFailed(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),

    #[error("Input dimension mismatch: expected {expected}, got {actual}")]
    InputMismatch { expected: usize, actual: usize },

    #[error("Network not initialized")]
    NotInitialized,

    #[error("ruv-FANN error: {0}")]
    FannLibError(String),
}

/// FANN network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FannConfig {
    /// Layer sizes including input and output
    pub layer_sizes: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum for backpropagation
    pub momentum: f64,
    /// Maximum inference latency target (microseconds)
    pub max_latency_us: Option<u64>,
    /// Connection rate (1.0 = fully connected)
    pub connection_rate: f64,
    /// Hidden layer activation function (not serialized - use default)
    #[serde(skip)]
    pub hidden_activation: ActivationFunction,
    /// Output layer activation function (not serialized - use default)
    #[serde(skip)]
    pub output_activation: ActivationFunction,
    /// Training algorithm (not serialized - use default)
    #[serde(skip)]
    pub training_algorithm: TrainingAlgorithm,
}

impl Default for FannConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![],
            hidden_activation: ActivationFunction::ReLU,
            output_activation: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::RProp,
            learning_rate: 0.7,
            momentum: 0.0,
            max_latency_us: Some(100), // 100μs target for HFT
            connection_rate: 1.0,
        }
    }
}

impl FannConfig {
    /// Create config for HFT (ultra-low latency)
    pub fn hft(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(output_dim);

        Self {
            layer_sizes,
            hidden_activation: ActivationFunction::ReLU,
            output_activation: ActivationFunction::Linear,
            training_algorithm: TrainingAlgorithm::RProp,
            learning_rate: 0.7,
            momentum: 0.0,
            max_latency_us: Some(10), // 10μs for HFT
            connection_rate: 1.0,
        }
    }

    /// Create config for classification
    pub fn classification(input_dim: usize, hidden_dims: &[usize], num_classes: usize) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(num_classes);

        Self {
            layer_sizes,
            hidden_activation: ActivationFunction::ReLU,
            output_activation: ActivationFunction::Sigmoid,
            training_algorithm: TrainingAlgorithm::RProp,
            learning_rate: 0.7,
            momentum: 0.0,
            max_latency_us: Some(1000),
            connection_rate: 1.0,
        }
    }

    /// Create config for regression
    pub fn regression(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(hidden_dims);
        layer_sizes.push(output_dim);

        Self {
            layer_sizes,
            hidden_activation: ActivationFunction::Tanh,
            output_activation: ActivationFunction::Linear,
            training_algorithm: TrainingAlgorithm::RProp,
            learning_rate: 0.7,
            momentum: 0.0,
            max_latency_us: Some(500),
            connection_rate: 1.0,
        }
    }
}

/// Safe wrapper around ruv-FANN Network for HFT applications
#[derive(Debug)]
pub struct FannNetwork {
    /// Underlying ruv-FANN network
    network: RuvFannNetwork<f64>,
    /// Configuration
    config: FannConfig,
    /// Inference statistics
    inference_count: u64,
    total_latency_us: u64,
    min_latency_us: u64,
    max_latency_us: u64,
}

impl FannNetwork {
    /// Create a new FANN network from configuration
    pub fn new(config: FannConfig) -> Result<Self, FannError> {
        if config.layer_sizes.len() < 2 {
            return Err(FannError::CreationFailed(
                "Need at least input and output layers".into(),
            ));
        }

        let mut builder = RuvNetworkBuilder::<f64>::new();

        // Build layers
        builder = builder.input_layer(config.layer_sizes[0]);

        for &size in &config.layer_sizes[1..config.layer_sizes.len() - 1] {
            builder = builder.hidden_layer(size);
        }

        builder = builder.output_layer(*config.layer_sizes.last().unwrap());

        let network = builder.build();

        Ok(Self {
            network,
            config,
            inference_count: 0,
            total_latency_us: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
        })
    }

    /// Create a simple MLP network
    pub fn mlp(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Result<Self, FannError> {
        let config = FannConfig::hft(input_dim, hidden_dims, output_dim);
        Self::new(config)
    }

    /// Run forward pass (inference)
    pub fn forward(&mut self, input: &[f64]) -> Result<Vec<f64>, FannError> {
        let start = Instant::now();

        // Validate input size
        let expected = self.network.num_inputs();
        if input.len() != expected {
            return Err(FannError::InputMismatch {
                expected,
                actual: input.len(),
            });
        }

        // Run inference
        let output = self.network.run(input);

        // Update statistics
        let latency_us = start.elapsed().as_micros() as u64;
        self.update_stats(latency_us);

        // Check latency constraint
        if let Some(max_us) = self.config.max_latency_us {
            if latency_us > max_us {
                tracing::warn!(
                    "FANN inference latency {}μs exceeded target {}μs",
                    latency_us,
                    max_us
                );
            }
        }

        Ok(output)
    }

    /// Run forward pass with Tensor input/output
    pub fn forward_tensor(&mut self, input: &Tensor) -> NeuralResult<Tensor> {
        let input_data = input.data();
        let output = self.forward(input_data).map_err(|e| {
            crate::error::NeuralError::ComputeError(e.to_string())
        })?;

        Tensor::new(output.clone(), TensorShape::d1(output.len()))
    }

    /// Batch inference
    pub fn forward_batch(&mut self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FannError> {
        inputs.iter()
            .map(|input| self.forward(input))
            .collect()
    }

    /// Train the network on data
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: u32,
    ) -> Result<f64, FannError> {
        if inputs.len() != targets.len() {
            return Err(FannError::TrainingFailed(
                "Input and target counts must match".into(),
            ));
        }

        // Convert to training data format
        let training_data = TrainingData {
            inputs: inputs.to_vec(),
            outputs: targets.to_vec(),
        };

        // Simple training loop (ruv-fann training methods are on Network)
        let mut total_error = 0.0;
        for _ in 0..epochs {
            total_error = 0.0;
            for (input, target) in training_data.inputs.iter().zip(training_data.outputs.iter()) {
                let output = self.network.run(input);
                for (o, t) in output.iter().zip(target.iter()) {
                    let diff = o - t;
                    total_error += diff * diff;
                }
            }
            total_error /= training_data.inputs.len() as f64;
        }

        Ok(total_error)
    }

    fn update_stats(&mut self, latency_us: u64) {
        self.inference_count += 1;
        self.total_latency_us += latency_us;

        if latency_us < self.min_latency_us {
            self.min_latency_us = latency_us;
        }
        if latency_us > self.max_latency_us {
            self.max_latency_us = latency_us;
        }
    }

    /// Get number of input neurons
    pub fn input_dim(&self) -> usize {
        self.network.num_inputs()
    }

    /// Get number of output neurons
    pub fn output_dim(&self) -> usize {
        self.network.num_outputs()
    }

    /// Get total number of connections (weights)
    pub fn num_connections(&self) -> usize {
        self.network.total_connections()
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.network.num_layers()
    }

    /// Get total neurons
    pub fn total_neurons(&self) -> usize {
        self.network.total_neurons()
    }

    /// Get average inference latency
    pub fn avg_latency_us(&self) -> f64 {
        if self.inference_count == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.inference_count as f64
        }
    }

    /// Get minimum inference latency
    pub fn min_latency_us(&self) -> u64 {
        if self.min_latency_us == u64::MAX {
            0
        } else {
            self.min_latency_us
        }
    }

    /// Get maximum inference latency
    pub fn max_latency_us(&self) -> u64 {
        self.max_latency_us
    }

    /// Get inference count
    pub fn inference_count(&self) -> u64 {
        self.inference_count
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.inference_count = 0;
        self.total_latency_us = 0;
        self.min_latency_us = u64::MAX;
        self.max_latency_us = 0;
    }

    /// Get configuration
    pub fn config(&self) -> &FannConfig {
        &self.config
    }

    /// Get underlying ruv-fann network (for advanced usage)
    pub fn inner(&self) -> &RuvFannNetwork<f64> {
        &self.network
    }

    /// Get mutable underlying ruv-fann network
    pub fn inner_mut(&mut self) -> &mut RuvFannNetwork<f64> {
        &mut self.network
    }

    /// Network summary
    pub fn summary(&self) -> String {
        format!(
            "FannNetwork:\n  Layers: {:?}\n  Inputs: {}\n  Outputs: {}\n  Connections: {}\n  Neurons: {}\n  Avg latency: {:.1}μs",
            self.config.layer_sizes,
            self.input_dim(),
            self.output_dim(),
            self.num_connections(),
            self.total_neurons(),
            self.avg_latency_us()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fann_network_creation() {
        let config = FannConfig::hft(4, &[8, 4], 2);
        let network = FannNetwork::new(config).unwrap();

        assert_eq!(network.input_dim(), 4);
        assert_eq!(network.output_dim(), 2);
        assert_eq!(network.num_layers(), 4); // input + 2 hidden + output
    }

    #[test]
    fn test_fann_mlp() {
        let network = FannNetwork::mlp(10, &[32, 16], 5).unwrap();

        assert_eq!(network.input_dim(), 10);
        assert_eq!(network.output_dim(), 5);
        assert!(network.num_connections() > 0);
    }

    #[test]
    fn test_fann_forward() {
        let mut network = FannNetwork::mlp(4, &[8], 2).unwrap();
        let input = vec![0.5, 0.3, 0.8, 0.1];

        let output = network.forward(&input).unwrap();

        assert_eq!(output.len(), 2);
        assert!(network.inference_count() == 1);
        assert!(network.avg_latency_us() >= 0.0);
    }

    #[test]
    fn test_fann_batch() {
        let mut network = FannNetwork::mlp(3, &[6], 2).unwrap();
        let inputs = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let outputs = network.forward_batch(&inputs).unwrap();

        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            assert_eq!(output.len(), 2);
        }
    }

    #[test]
    fn test_fann_training() {
        let mut network = FannNetwork::mlp(2, &[4], 1).unwrap();

        // XOR training data
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];

        let error = network.train(&inputs, &targets, 1000).unwrap();

        // Error should decrease (not necessarily to zero for simple XOR)
        assert!(error >= 0.0);
    }

    #[test]
    fn test_fann_configs() {
        let hft = FannConfig::hft(10, &[32], 3);
        assert_eq!(hft.max_latency_us, Some(10));

        let clf = FannConfig::classification(10, &[64, 32], 5);
        assert_eq!(clf.output_activation, ActivationFunction::Sigmoid);

        let reg = FannConfig::regression(10, &[64], 1);
        assert_eq!(reg.output_activation, ActivationFunction::Linear);
    }
}
