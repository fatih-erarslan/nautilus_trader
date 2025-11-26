//! Configuration structures for models and training

use serde::{Deserialize, Serialize};

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input sequence length
    pub input_size: usize,

    /// Forecast horizon
    pub horizon: usize,

    /// Number of hidden units
    pub hidden_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Dropout rate
    pub dropout: f64,

    /// Learning rate
    pub learning_rate: f64,

    /// Batch size
    pub batch_size: usize,

    /// Number of features (for multivariate)
    pub num_features: usize,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Enable Flash Attention (1000-5000x memory reduction)
    pub use_flash_attention: bool,

    /// Flash Attention block size (default: 64)
    pub flash_block_size: usize,

    /// Use causal masking for autoregressive models
    pub flash_causal: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 168,      // 1 week of hourly data
            horizon: 24,          // 24 hour forecast
            hidden_size: 512,
            num_layers: 3,
            dropout: 0.1,
            learning_rate: 0.001,
            batch_size: 32,
            num_features: 1,
            seed: Some(42),
            use_flash_attention: true,   // Enable by default for memory efficiency
            flash_block_size: 64,        // Balanced default
            flash_causal: false,
        }
    }
}

impl ModelConfig {
    /// Create a new configuration with builder pattern
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_input_size(mut self, input_size: usize) -> Self {
        self.input_size = input_size;
        self
    }

    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_num_features(mut self, num_features: usize) -> Self {
        self.num_features = num_features;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_flash_attention(mut self, enabled: bool) -> Self {
        self.use_flash_attention = enabled;
        self
    }

    pub fn with_flash_block_size(mut self, block_size: usize) -> Self {
        self.flash_block_size = block_size;
        self
    }

    pub fn with_flash_causal(mut self, causal: bool) -> Self {
        self.flash_causal = causal;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::Result<()> {
        if self.input_size == 0 {
            return Err(crate::error::NeuroDivergentError::ConfigError(
                "input_size must be greater than 0".to_string(),
            ));
        }
        if self.horizon == 0 {
            return Err(crate::error::NeuroDivergentError::ConfigError(
                "horizon must be greater than 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(crate::error::NeuroDivergentError::ConfigError(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(crate::error::NeuroDivergentError::ConfigError(
                "dropout must be in range [0, 1)".to_string(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(crate::error::NeuroDivergentError::ConfigError(
                "learning_rate must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Early stopping patience
    pub patience: usize,

    /// Validation split ratio
    pub validation_split: f64,

    /// Enable gradient clipping
    pub gradient_clip: Option<f64>,

    /// Learning rate scheduler
    pub lr_scheduler: LRScheduler,

    /// Optimizer type
    pub optimizer: OptimizerType,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Enable mixed precision training
    pub mixed_precision: bool,

    /// Checkpoint frequency (epochs)
    pub checkpoint_freq: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            patience: 10,
            validation_split: 0.2,
            gradient_clip: Some(1.0),
            lr_scheduler: LRScheduler::ReduceOnPlateau {
                factor: 0.5,
                patience: 5,
            },
            optimizer: OptimizerType::Adam,
            weight_decay: 0.0001,
            mixed_precision: false,
            checkpoint_freq: 10,
        }
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduler {
    /// Constant learning rate
    Constant,

    /// Step decay
    StepDecay {
        step_size: usize,
        gamma: f64,
    },

    /// Exponential decay
    ExponentialDecay {
        gamma: f64,
    },

    /// Reduce on plateau
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
    },

    /// Cosine annealing
    CosineAnnealing {
        t_max: usize,
        eta_min: f64,
    },
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD {
        momentum: f64,
        nesterov: bool,
    },

    /// Adam optimizer
    Adam,

    /// AdamW optimizer
    AdamW,

    /// RMSprop optimizer
    RMSprop {
        alpha: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ModelConfig::default();
        assert_eq!(config.input_size, 168);
        assert_eq!(config.horizon, 24);
    }

    #[test]
    fn test_builder_pattern() {
        let config = ModelConfig::new()
            .with_input_size(100)
            .with_horizon(10)
            .with_hidden_size(256);

        assert_eq!(config.input_size, 100);
        assert_eq!(config.horizon, 10);
        assert_eq!(config.hidden_size, 256);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = ModelConfig::default();
        assert!(valid_config.validate().is_ok());

        let invalid_config = ModelConfig::default().with_input_size(0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.patience, 10);
    }
}
