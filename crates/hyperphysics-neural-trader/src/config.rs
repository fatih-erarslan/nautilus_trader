//! Configuration for Neural Trader bridge.

use serde::{Deserialize, Serialize};

/// Configuration for neural bridge integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBridgeConfig {
    /// Minimum input sequence length for forecasting
    pub min_sequence_length: usize,

    /// Default forecast horizon (number of steps ahead)
    pub forecast_horizon: usize,

    /// Hidden layer size for neural models
    pub hidden_size: usize,

    /// Number of attention heads (for transformer models)
    pub num_attention_heads: usize,

    /// Dropout rate during inference (usually 0.0)
    pub dropout: f64,

    /// Confidence level for prediction intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,

    /// Enable ensemble prediction (combine multiple models)
    pub enable_ensemble: bool,

    /// Models to include in ensemble
    pub ensemble_models: Vec<NeuralModelType>,

    /// Weight averaging method for ensemble
    pub ensemble_method: EnsembleMethod,

    /// Enable GPU acceleration if available
    pub use_gpu: bool,

    /// Maximum batch size for inference
    pub max_batch_size: usize,

    /// Enable conformal prediction for uncertainty quantification
    pub enable_conformal: bool,

    /// Rolling window size for feature calculation
    pub feature_window: usize,
}

impl Default for NeuralBridgeConfig {
    fn default() -> Self {
        Self {
            min_sequence_length: 24,
            forecast_horizon: 12,
            hidden_size: 256,
            num_attention_heads: 4,
            dropout: 0.0,
            confidence_level: 0.95,
            enable_ensemble: true,
            ensemble_models: vec![
                NeuralModelType::NHITS,
                NeuralModelType::LSTMAttention,
            ],
            ensemble_method: EnsembleMethod::WeightedAverage,
            use_gpu: true,
            max_batch_size: 32,
            enable_conformal: true,
            feature_window: 168, // 1 week of hourly data
        }
    }
}

impl NeuralBridgeConfig {
    /// Create config optimized for HFT (low latency)
    pub fn hft_optimized() -> Self {
        Self {
            min_sequence_length: 12,
            forecast_horizon: 6,
            hidden_size: 128,
            num_attention_heads: 2,
            dropout: 0.0,
            confidence_level: 0.90,
            enable_ensemble: false, // Single model for speed
            ensemble_models: vec![NeuralModelType::NHITS],
            ensemble_method: EnsembleMethod::Single,
            use_gpu: true,
            max_batch_size: 1, // Minimize latency
            enable_conformal: false,
            feature_window: 48,
        }
    }

    /// Create config for high accuracy (longer computation)
    pub fn high_accuracy() -> Self {
        Self {
            min_sequence_length: 48,
            forecast_horizon: 24,
            hidden_size: 512,
            num_attention_heads: 8,
            dropout: 0.0,
            confidence_level: 0.99,
            enable_ensemble: true,
            ensemble_models: vec![
                NeuralModelType::NHITS,
                NeuralModelType::LSTMAttention,
                NeuralModelType::Transformer,
                NeuralModelType::GRU,
            ],
            ensemble_method: EnsembleMethod::StackedGeneralization,
            use_gpu: true,
            max_batch_size: 64,
            enable_conformal: true,
            feature_window: 336, // 2 weeks of hourly data
        }
    }
}

/// Neural model types available for forecasting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuralModelType {
    /// Neural Hierarchical Interpolation for Time Series
    NHITS,
    /// LSTM with multi-head attention
    LSTMAttention,
    /// Transformer architecture
    Transformer,
    /// Gated Recurrent Unit
    GRU,
    /// Temporal Convolutional Network
    TCN,
    /// Probabilistic forecasting with LSTM
    DeepAR,
    /// Pure MLP with interpretable decomposition
    NBeats,
    /// Time series decomposition (trend + seasonality)
    Prophet,
}

impl std::fmt::Display for NeuralModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeuralModelType::NHITS => write!(f, "NHITS"),
            NeuralModelType::LSTMAttention => write!(f, "LSTM-Attention"),
            NeuralModelType::Transformer => write!(f, "Transformer"),
            NeuralModelType::GRU => write!(f, "GRU"),
            NeuralModelType::TCN => write!(f, "TCN"),
            NeuralModelType::DeepAR => write!(f, "DeepAR"),
            NeuralModelType::NBeats => write!(f, "N-BEATS"),
            NeuralModelType::Prophet => write!(f, "Prophet"),
        }
    }
}

/// Ensemble aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Use single model (no ensemble)
    Single,
    /// Simple average of all model predictions
    SimpleAverage,
    /// Weighted average based on historical performance
    WeightedAverage,
    /// Median of predictions (robust to outliers)
    Median,
    /// Stacked generalization (meta-learner)
    StackedGeneralization,
}
