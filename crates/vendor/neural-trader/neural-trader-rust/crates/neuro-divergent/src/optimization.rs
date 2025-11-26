//! Hyperparameter optimization utilities

use crate::ModelConfig;

/// Hyperparameter search space
#[derive(Debug, Clone)]
pub struct HyperparameterSpace {
    pub hidden_sizes: Vec<usize>,
    pub num_layers: Vec<usize>,
    pub dropout_rates: Vec<f64>,
    pub learning_rates: Vec<f64>,
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            hidden_sizes: vec![32, 64, 128, 256],
            num_layers: vec![1, 2, 3, 4],
            dropout_rates: vec![0.0, 0.1, 0.2, 0.3],
            learning_rates: vec![0.0001, 0.001, 0.01],
        }
    }
}

/// Generate random configuration from space
pub fn random_config(space: &HyperparameterSpace, base: &ModelConfig) -> ModelConfig {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    ModelConfig {
        hidden_size: *space.hidden_sizes.get(rng.gen_range(0..space.hidden_sizes.len())).unwrap(),
        num_layers: *space.num_layers.get(rng.gen_range(0..space.num_layers.len())).unwrap(),
        dropout: *space.dropout_rates.get(rng.gen_range(0..space.dropout_rates.len())).unwrap(),
        learning_rate: *space.learning_rates.get(rng.gen_range(0..space.learning_rates.len())).unwrap(),
        ..base.clone()
    }
}
