//! Configuration System for NHITS Model
//! Comprehensive configuration for all model components

use serde::{Deserialize, Serialize};

use super::blocks::{BlockConfig, ActivationType};
use super::attention::{AttentionConfig, AttentionType};
use super::decomposition::{DecomposerConfig, DecompositionType};
use super::adaptation::{AdaptationConfig, AdaptationStrategy};
use super::pooling::PoolingType;
use super::interpolation::InterpolationType;

/// Main NHITS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NHITSConfig {
    // Model architecture
    pub block_configs: Vec<BlockConfig>,
    pub attention_config: AttentionConfig,
    pub decomposer_config: DecomposerConfig,
    pub adaptation_config: AdaptationConfig,
    
    // Time series parameters
    pub lookback_window: usize,
    pub forecast_horizon: usize,
    pub input_features: usize,
    pub output_features: usize,
    
    // Training parameters
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gradient_clip: f64,
    pub weight_decay: f64,
    
    // Consciousness integration
    pub consciousness_enabled: bool,
    pub coherence_weight: f64,
    pub min_coherence_threshold: f64,
    
    // Early stopping
    pub early_stop_patience: usize,
    pub early_stop_threshold: f64,
    
    // Advanced features
    pub use_residual_connections: bool,
    pub use_layer_norm: bool,
    pub use_dropout: bool,
    pub dropout_rate: f64,
}

impl NHITSConfig {
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            block_configs: Self::default_blocks(),
            attention_config: Self::default_attention(),
            decomposer_config: Self::default_decomposer(),
            adaptation_config: Self::default_adaptation(),
            lookback_window: 168, // One week for hourly data
            forecast_horizon: 24, // One day ahead
            input_features: 1,
            output_features: 1,
            learning_rate: 0.001,
            batch_size: 32,
            gradient_clip: 1.0,
            weight_decay: 0.0001,
            consciousness_enabled: true,
            coherence_weight: 0.1,
            min_coherence_threshold: 0.3,
            early_stop_patience: 10,
            early_stop_threshold: 0.0001,
            use_residual_connections: true,
            use_layer_norm: true,
            use_dropout: true,
            dropout_rate: 0.1,
        }
    }
    
    /// Default block configuration
    fn default_blocks() -> Vec<BlockConfig> {
        vec![
            BlockConfig {
                input_size: 512,
                hidden_size: 512,
                num_basis: 10,
                pooling_factor: 2,
                pooling_type: PoolingType::Average,
                interpolation_type: InterpolationType::Linear,
                dropout_rate: 0.1,
                activation: ActivationType::GELU,
            },
            BlockConfig {
                input_size: 512,
                hidden_size: 256,
                num_basis: 8,
                pooling_factor: 2,
                pooling_type: PoolingType::Max,
                interpolation_type: InterpolationType::Cubic,
                dropout_rate: 0.1,
                activation: ActivationType::GELU,
            },
            BlockConfig {
                input_size: 256,
                hidden_size: 128,
                num_basis: 6,
                pooling_factor: 2,
                pooling_type: PoolingType::Adaptive,
                interpolation_type: InterpolationType::Spline { tension: 0.5 },
                dropout_rate: 0.1,
                activation: ActivationType::GELU,
            },
        ]
    }
    
    /// Default attention configuration
    fn default_attention() -> AttentionConfig {
        AttentionConfig {
            num_heads: 8,
            head_dim: 64,
            dropout_rate: 0.1,
            temperature: 1.0,
            use_causal_mask: false,
            attention_type: AttentionType::Relative,
            consciousness_integration: true,
        }
    }
    
    /// Default decomposer configuration
    fn default_decomposer() -> DecomposerConfig {
        DecomposerConfig {
            decomposition_type: DecompositionType::Hybrid,
            num_scales: 4,
            seasonal_periods: vec![24, 168], // Daily and weekly for hourly data
            trend_filter_size: 13,
            use_stl: true,
            robust: true,
        }
    }
    
    /// Default adaptation configuration
    fn default_adaptation() -> AdaptationConfig {
        AdaptationConfig {
            adaptation_rate: 0.01,
            performance_window: 20,
            change_threshold: 0.01,
            max_depth: 8,
            min_depth: 2,
            consciousness_weight: 0.5,
            exploration_rate: 0.1,
            adaptation_strategy: AdaptationStrategy::ConsciousnessGuided,
        }
    }
    
    /// Create configuration for specific use case
    pub fn for_use_case(use_case: UseCase) -> Self {
        match use_case {
            UseCase::ShortTermForecasting => Self::short_term_config(),
            UseCase::LongTermForecasting => Self::long_term_config(),
            UseCase::MultivariateSeries => Self::multivariate_config(),
            UseCase::HighFrequencyTrading => Self::high_frequency_config(),
            UseCase::AnomalyDetection => Self::anomaly_detection_config(),
            UseCase::SeasonalDecomposition => Self::seasonal_config(),
        }
    }
    
    /// Configuration for short-term forecasting
    fn short_term_config() -> Self {
        let mut config = Self::default();
        config.lookback_window = 48;
        config.forecast_horizon = 12;
        config.block_configs = vec![
            BlockConfig {
                input_size: 256,
                hidden_size: 256,
                num_basis: 5,
                pooling_factor: 2,
                pooling_type: PoolingType::Average,
                interpolation_type: InterpolationType::Linear,
                dropout_rate: 0.05,
                activation: ActivationType::ReLU,
            },
            BlockConfig {
                input_size: 256,
                hidden_size: 128,
                num_basis: 4,
                pooling_factor: 2,
                pooling_type: PoolingType::Max,
                interpolation_type: InterpolationType::Linear,
                dropout_rate: 0.05,
                activation: ActivationType::ReLU,
            },
        ];
        config
    }
    
    /// Configuration for long-term forecasting
    fn long_term_config() -> Self {
        let mut config = Self::default();
        config.lookback_window = 720; // 30 days for hourly
        config.forecast_horizon = 168; // 7 days ahead
        config.decomposer_config.num_scales = 6;
        config.decomposer_config.seasonal_periods = vec![24, 168, 720]; // Daily, weekly, monthly
        config.attention_config.attention_type = AttentionType::Sparse { sparsity_factor: 0.8 };
        config
    }
    
    /// Configuration for multivariate series
    fn multivariate_config() -> Self {
        let mut config = Self::default();
        config.input_features = 10;
        config.output_features = 10;
        config.attention_config.num_heads = 10;
        config.block_configs[0].hidden_size = 1024;
        config
    }
    
    /// Configuration for high-frequency trading
    fn high_frequency_config() -> Self {
        let mut config = Self::default();
        config.lookback_window = 100;
        config.forecast_horizon = 1;
        config.batch_size = 128;
        config.adaptation_config.adaptation_strategy = AdaptationStrategy::Aggressive;
        config.adaptation_config.adaptation_rate = 0.1;
        config.attention_config.attention_type = AttentionType::LocalWindow { window_size: 10 };
        config
    }
    
    /// Configuration for anomaly detection
    fn anomaly_detection_config() -> Self {
        let mut config = Self::default();
        config.decomposer_config.decomposition_type = DecompositionType::EMD;
        config.decomposer_config.robust = true;
        config.adaptation_config.adaptation_strategy = AdaptationStrategy::Conservative;
        config.consciousness_enabled = true;
        config.coherence_weight = 0.3;
        config
    }
    
    /// Configuration for seasonal decomposition
    fn seasonal_config() -> Self {
        let mut config = Self::default();
        config.decomposer_config.decomposition_type = DecompositionType::STL;
        config.decomposer_config.seasonal_periods = vec![12, 24, 168, 8760]; // Various periods
        config.decomposer_config.use_stl = true;
        config.decomposer_config.robust = true;
        config
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check basic constraints
        if self.lookback_window == 0 {
            return Err(ConfigError::InvalidParameter("lookback_window must be > 0".to_string()));
        }
        
        if self.forecast_horizon == 0 {
            return Err(ConfigError::InvalidParameter("forecast_horizon must be > 0".to_string()));
        }
        
        if self.block_configs.is_empty() {
            return Err(ConfigError::InvalidParameter("Must have at least one block".to_string()));
        }
        
        // Validate block configurations
        for (i, block) in self.block_configs.iter().enumerate() {
            if block.hidden_size == 0 {
                return Err(ConfigError::InvalidParameter(
                    format!("Block {} hidden_size must be > 0", i)
                ));
            }
            
            if block.num_basis == 0 {
                return Err(ConfigError::InvalidParameter(
                    format!("Block {} num_basis must be > 0", i)
                ));
            }
            
            if block.pooling_factor == 0 {
                return Err(ConfigError::InvalidParameter(
                    format!("Block {} pooling_factor must be > 0", i)
                ));
            }
        }
        
        // Validate attention configuration
        if self.attention_config.num_heads == 0 {
            return Err(ConfigError::InvalidParameter("num_heads must be > 0".to_string()));
        }
        
        if self.attention_config.head_dim == 0 {
            return Err(ConfigError::InvalidParameter("head_dim must be > 0".to_string()));
        }
        
        // Validate adaptation configuration
        if self.adaptation_config.max_depth <= self.adaptation_config.min_depth {
            return Err(ConfigError::InvalidParameter(
                "max_depth must be > min_depth".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            block_configs: vec![BlockConfig {
                input_size: 64,
                hidden_size: 64,
                num_basis: 3,
                pooling_factor: 2,
                pooling_type: PoolingType::Average,
                interpolation_type: InterpolationType::Linear,
                dropout_rate: 0.0,
                activation: ActivationType::ReLU,
            }],
            attention_config: AttentionConfig {
                num_heads: 2,
                head_dim: 32,
                dropout_rate: 0.0,
                temperature: 1.0,
                use_causal_mask: false,
                attention_type: AttentionType::Standard,
                consciousness_integration: false,
            },
            decomposer_config: DecomposerConfig {
                decomposition_type: DecompositionType::Additive,
                num_scales: 2,
                seasonal_periods: vec![12],
                trend_filter_size: 5,
                use_stl: false,
                robust: false,
            },
            adaptation_config: AdaptationConfig {
                adaptation_rate: 0.0,
                performance_window: 10,
                change_threshold: 1.0,
                max_depth: 3,
                min_depth: 1,
                consciousness_weight: 0.0,
                exploration_rate: 0.0,
                adaptation_strategy: AdaptationStrategy::Conservative,
            },
            lookback_window: 24,
            forecast_horizon: 6,
            input_features: 1,
            output_features: 1,
            learning_rate: 0.01,
            batch_size: 16,
            gradient_clip: 1.0,
            weight_decay: 0.0,
            consciousness_enabled: false,
            coherence_weight: 0.0,
            min_coherence_threshold: 0.0,
            early_stop_patience: 5,
            early_stop_threshold: 0.001,
            use_residual_connections: true,
            use_layer_norm: false,
            use_dropout: false,
            dropout_rate: 0.0,
        }
    }
}

/// Predefined use cases
#[derive(Debug, Clone, Copy)]
pub enum UseCase {
    ShortTermForecasting,
    LongTermForecasting,
    MultivariateSeries,
    HighFrequencyTrading,
    AnomalyDetection,
    SeasonalDecomposition,
}

/// Configuration builder for fluent API
#[derive(Debug, Clone)]
pub struct NHITSConfigBuilder {
    config: NHITSConfig,
}

impl NHITSConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: NHITSConfig::default(),
        }
    }
    
    pub fn with_lookback(mut self, window: usize) -> Self {
        self.config.lookback_window = window;
        self
    }
    
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.config.forecast_horizon = horizon;
        self
    }
    
    pub fn with_features(mut self, input: usize, output: usize) -> Self {
        self.config.input_features = input;
        self.config.output_features = output;
        self
    }
    
    pub fn with_consciousness(mut self, enabled: bool, weight: f64) -> Self {
        self.config.consciousness_enabled = enabled;
        self.config.coherence_weight = weight;
        self
    }
    
    pub fn with_adaptation(mut self, strategy: AdaptationStrategy, rate: f64) -> Self {
        self.config.adaptation_config.adaptation_strategy = strategy;
        self.config.adaptation_config.adaptation_rate = rate;
        self
    }
    
    pub fn with_blocks(mut self, blocks: Vec<BlockConfig>) -> Self {
        self.config.block_configs = blocks;
        self
    }
    
    pub fn build(self) -> Result<NHITSConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Incompatible configuration: {0}")]
    IncompatibleConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_validation() {
        let config = NHITSConfig::default();
        assert!(config.validate().is_ok());
        
        let minimal = NHITSConfig::minimal();
        assert!(minimal.validate().is_ok());
    }
    
    #[test]
    fn test_config_builder() {
        let config = NHITSConfigBuilder::new()
            .with_lookback(100)
            .with_horizon(20)
            .with_features(5, 3)
            .with_consciousness(true, 0.2)
            .build()
            .unwrap();
        
        assert_eq!(config.lookback_window, 100);
        assert_eq!(config.forecast_horizon, 20);
        assert_eq!(config.input_features, 5);
        assert_eq!(config.output_features, 3);
        assert!(config.consciousness_enabled);
        assert_eq!(config.coherence_weight, 0.2);
    }
}