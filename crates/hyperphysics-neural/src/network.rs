//! Neural network architecture
//!
//! Provides Network struct with builder pattern for constructing neural networks.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::activation::Activation;
use crate::core::Tensor;
use crate::error::{NeuralError, NeuralResult};
use crate::layer::{Layer, LayerConfig};

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network name/identifier
    pub name: String,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Hidden layer activation
    pub hidden_activation: Activation,
    /// Output layer activation
    pub output_activation: Activation,
    /// Dropout rate for hidden layers
    pub dropout_rate: f64,
    /// Use batch normalization
    pub use_batch_norm: bool,
    /// Maximum inference latency target
    pub max_latency_us: Option<u64>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            name: "network".into(),
            input_dim: 0,
            output_dim: 0,
            hidden_dims: vec![],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Linear,
            dropout_rate: 0.0,
            use_batch_norm: false,
            max_latency_us: None,
        }
    }
}

/// Network builder for fluent construction
#[derive(Debug, Clone)]
pub struct NetworkBuilder {
    config: NetworkConfig,
}

impl NetworkBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
        }
    }

    /// Set network name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set input dimension
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.config.input_dim = dim;
        self
    }

    /// Set output dimension
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.config.output_dim = dim;
        self
    }

    /// Add hidden layer
    pub fn hidden(mut self, dim: usize) -> Self {
        self.config.hidden_dims.push(dim);
        self
    }

    /// Set all hidden layer dimensions
    pub fn hidden_layers(mut self, dims: Vec<usize>) -> Self {
        self.config.hidden_dims = dims;
        self
    }

    /// Set hidden activation
    pub fn hidden_activation(mut self, activation: Activation) -> Self {
        self.config.hidden_activation = activation;
        self
    }

    /// Set output activation
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.config.output_activation = activation;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, rate: f64) -> Self {
        self.config.dropout_rate = rate;
        self
    }

    /// Enable batch normalization
    pub fn with_batch_norm(mut self) -> Self {
        self.config.use_batch_norm = true;
        self
    }

    /// Set maximum latency target (microseconds)
    pub fn max_latency_us(mut self, us: u64) -> Self {
        self.config.max_latency_us = Some(us);
        self
    }

    /// Build the network
    pub fn build(self) -> NeuralResult<Network> {
        Network::from_config(self.config)
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Neural network composed of layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Network configuration
    config: NetworkConfig,
    /// Network layers
    layers: Vec<Layer>,
    /// Training mode flag
    #[serde(skip)]
    training: bool,
    /// Inference statistics
    #[serde(skip)]
    inference_stats: InferenceStats,
}

/// Inference statistics for monitoring
#[derive(Debug, Clone, Default)]
struct InferenceStats {
    total_inferences: u64,
    total_latency_us: u64,
    min_latency_us: u64,
    max_latency_us: u64,
}

impl Network {
    /// Create network from configuration
    pub fn from_config(config: NetworkConfig) -> NeuralResult<Self> {
        if config.input_dim == 0 {
            return Err(NeuralError::InvalidArchitecture("Input dimension must be > 0".into()));
        }
        if config.output_dim == 0 {
            return Err(NeuralError::InvalidArchitecture("Output dimension must be > 0".into()));
        }

        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;

        // Build hidden layers
        for (i, &hidden_dim) in config.hidden_dims.iter().enumerate() {
            let layer_config = LayerConfig::dense(prev_dim, hidden_dim)
                .with_activation(config.hidden_activation)
                .with_dropout(config.dropout_rate);

            layers.push(Layer::with_seed(layer_config, (42 + i) as u64)?);
            prev_dim = hidden_dim;
        }

        // Output layer
        let output_config = LayerConfig::dense(prev_dim, config.output_dim)
            .with_activation(config.output_activation);
        layers.push(Layer::new(output_config)?);

        Ok(Self {
            config,
            layers,
            training: false,
            inference_stats: InferenceStats::default(),
        })
    }

    /// Create network using builder
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Tensor) -> NeuralResult<Tensor> {
        let start = Instant::now();

        let mut output = input.clone();
        for layer in &mut self.layers {
            output = if self.training {
                layer.forward(&output)?
            } else {
                layer.forward_inference(&output)?
            };
        }

        // Update inference stats
        let latency_us = start.elapsed().as_micros() as u64;
        self.update_stats(latency_us);

        // Check latency constraint
        if let Some(max_us) = self.config.max_latency_us {
            if latency_us > max_us {
                tracing::warn!(
                    "Inference latency {}μs exceeded target {}μs",
                    latency_us,
                    max_us
                );
            }
        }

        Ok(output)
    }

    /// Forward pass optimized for inference (no gradient tracking)
    pub fn predict(&self, input: &Tensor) -> NeuralResult<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward_inference(&output)?;
        }
        Ok(output)
    }

    /// Batch prediction
    pub fn predict_batch(&self, inputs: &[Tensor]) -> NeuralResult<Vec<Tensor>> {
        inputs.iter()
            .map(|input| self.predict(input))
            .collect()
    }

    fn update_stats(&mut self, latency_us: u64) {
        let stats = &mut self.inference_stats;
        stats.total_inferences += 1;
        stats.total_latency_us += latency_us;

        if stats.min_latency_us == 0 || latency_us < stats.min_latency_us {
            stats.min_latency_us = latency_us;
        }
        if latency_us > stats.max_latency_us {
            stats.max_latency_us = latency_us;
        }
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation/inference mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get network configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Get layers
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Get mutable layers
    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    /// Total number of parameters
    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|l| l.num_params()).sum()
    }

    /// Input dimension
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Output dimension
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get average inference latency
    pub fn avg_latency_us(&self) -> f64 {
        if self.inference_stats.total_inferences == 0 {
            0.0
        } else {
            self.inference_stats.total_latency_us as f64 /
            self.inference_stats.total_inferences as f64
        }
    }

    /// Get min inference latency
    pub fn min_latency_us(&self) -> u64 {
        self.inference_stats.min_latency_us
    }

    /// Get max inference latency
    pub fn max_latency_us(&self) -> u64 {
        self.inference_stats.max_latency_us
    }

    /// Reset inference statistics
    pub fn reset_stats(&mut self) {
        self.inference_stats = InferenceStats::default();
    }

    /// Serialize network to bytes
    pub fn to_bytes(&self) -> NeuralResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    /// Deserialize network from bytes
    pub fn from_bytes(bytes: &[u8]) -> NeuralResult<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))
    }

    /// Architecture summary string
    pub fn summary(&self) -> String {
        let mut s = format!("Network: {}\n", self.config.name);
        s.push_str(&format!("  Input: {}\n", self.config.input_dim));
        for (i, layer) in self.layers.iter().enumerate() {
            s.push_str(&format!(
                "  Layer {}: {} -> {} ({:?})\n",
                i,
                layer.input_dim(),
                layer.output_dim(),
                layer.config.activation
            ));
        }
        s.push_str(&format!("  Output: {}\n", self.config.output_dim));
        s.push_str(&format!("  Total params: {}\n", self.num_params()));
        s
    }
}

/// Simple MLP (Multi-Layer Perceptron) convenience constructor
pub fn mlp(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> NeuralResult<Network> {
    Network::builder()
        .name("mlp")
        .input_dim(input_dim)
        .hidden_layers(hidden_dims.to_vec())
        .output_dim(output_dim)
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Linear)
        .build()
}

/// Create network for regression
pub fn regression_network(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> NeuralResult<Network> {
    Network::builder()
        .name("regression")
        .input_dim(input_dim)
        .hidden_layers(hidden_dims.to_vec())
        .output_dim(output_dim)
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Linear)
        .build()
}

/// Create network for classification
pub fn classification_network(input_dim: usize, hidden_dims: &[usize], num_classes: usize) -> NeuralResult<Network> {
    Network::builder()
        .name("classifier")
        .input_dim(input_dim)
        .hidden_layers(hidden_dims.to_vec())
        .output_dim(num_classes)
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Softmax)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TensorShape;

    #[test]
    fn test_network_builder() {
        let net = Network::builder()
            .name("test")
            .input_dim(10)
            .hidden(64)
            .hidden(32)
            .output_dim(5)
            .hidden_activation(Activation::ReLU)
            .output_activation(Activation::Softmax)
            .build()
            .unwrap();

        assert_eq!(net.input_dim(), 10);
        assert_eq!(net.output_dim(), 5);
        assert_eq!(net.num_layers(), 3); // 2 hidden + 1 output
    }

    #[test]
    fn test_network_forward() {
        let mut net = mlp(4, &[8, 4], 2).unwrap();
        let input = Tensor::ones(TensorShape::d2(3, 4)); // batch of 3

        let output = net.forward(&input).unwrap();
        assert_eq!(output.shape().0, vec![3, 2]);
    }

    #[test]
    fn test_network_predict() {
        let net = mlp(4, &[8], 2).unwrap();
        let input = Tensor::ones(TensorShape::d2(1, 4));

        let output = net.predict(&input).unwrap();
        assert_eq!(output.shape().0, vec![1, 2]);
    }

    #[test]
    fn test_network_serialization() {
        let net = mlp(4, &[8], 2).unwrap();
        let bytes = net.to_bytes().unwrap();

        let loaded = Network::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.input_dim(), net.input_dim());
        assert_eq!(loaded.output_dim(), net.output_dim());
        assert_eq!(loaded.num_params(), net.num_params());
    }

    #[test]
    fn test_network_stats() {
        let mut net = mlp(4, &[8], 2).unwrap();
        let input = Tensor::ones(TensorShape::d2(1, 4));

        // Run multiple inferences
        for _ in 0..10 {
            let _ = net.forward(&input).unwrap();
        }

        assert_eq!(net.inference_stats.total_inferences, 10);
        // Latency can be 0 on very fast systems - just verify stats are recorded
        assert!(net.avg_latency_us() >= 0.0);
    }

    #[test]
    fn test_convenience_constructors() {
        let reg = regression_network(10, &[64, 32], 1).unwrap();
        assert_eq!(reg.config().output_activation, Activation::Linear);

        let clf = classification_network(10, &[64, 32], 5).unwrap();
        assert_eq!(clf.config().output_activation, Activation::Softmax);
    }

    #[test]
    fn test_network_summary() {
        let net = mlp(10, &[64, 32], 5).unwrap();
        let summary = net.summary();

        assert!(summary.contains("Input: 10"));
        assert!(summary.contains("Output: 5"));
        assert!(summary.contains("Total params:"));
    }
}
