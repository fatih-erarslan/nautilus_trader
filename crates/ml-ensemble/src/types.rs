use ats_core::types::{MarketData, Signal};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Model types in the ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    Transformer,
    LSTM,
    XGBoost,
    LightGBM,
    IsolationForest,
    GRU,
    NBeats,
    NHits,
}

/// Market condition types for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketCondition {
    Trending,
    Ranging,
    HighVolatility,
    LowVolatility,
    Breakout,
    Reversal,
    Anomalous,
}

/// Prediction with confidence and interpretability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    /// Final ensemble prediction
    pub value: f64,
    
    /// Confidence score (0-1)
    pub confidence: f64,
    
    /// Individual model predictions
    pub model_predictions: Vec<ModelPrediction>,
    
    /// Market condition detected
    pub market_condition: MarketCondition,
    
    /// Feature importance scores
    pub feature_importance: Vec<(String, f64)>,
    
    /// Prediction timestamp (nanoseconds)
    pub timestamp_ns: u64,
    
    /// Inference latency (microseconds)
    pub latency_us: f64,
}

/// Individual model prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_type: ModelType,
    pub prediction: f64,
    pub confidence: f64,
    pub weight: f64,
    pub latency_us: f64,
}

/// Ensemble configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Maximum inference latency in microseconds
    pub max_latency_us: f64,
    
    /// Minimum confidence threshold
    pub min_confidence: f64,
    
    /// Enable GPU acceleration
    pub use_gpu: bool,
    
    /// Number of parallel inference threads
    pub inference_threads: usize,
    
    /// Model weights configuration
    pub model_weights: ModelWeightsConfig,
    
    /// Calibration configuration
    pub calibration: CalibrationConfig,
    
    /// Feature engineering configuration
    pub features: FeatureConfig,
}

/// Model weights configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightsConfig {
    /// Use dynamic weights based on recent performance
    pub dynamic_weights: bool,
    
    /// Weight update frequency (seconds)
    pub update_frequency: u64,
    
    /// Performance lookback period (seconds)
    pub lookback_period: u64,
    
    /// Initial model weights
    pub initial_weights: Vec<(ModelType, f64)>,
}

/// Confidence calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Enable isotonic regression calibration
    pub isotonic_regression: bool,
    
    /// Enable Platt scaling
    pub platt_scaling: bool,
    
    /// Calibration window size
    pub window_size: usize,
    
    /// Minimum samples for calibration
    pub min_samples: usize,
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Number of price lags
    pub price_lags: usize,
    
    /// Technical indicators to compute
    pub technical_indicators: Vec<String>,
    
    /// Market microstructure features
    pub microstructure_features: bool,
    
    /// Order flow features
    pub order_flow_features: bool,
    
    /// Sentiment features
    pub sentiment_features: bool,
}

/// Model performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_type: ModelType,
    pub accuracy: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub avg_latency_us: f64,
    pub prediction_count: u64,
    pub last_update: u64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            max_latency_us: 500.0,
            min_confidence: 0.6,
            use_gpu: true,
            inference_threads: 4,
            model_weights: ModelWeightsConfig::default(),
            calibration: CalibrationConfig::default(),
            features: FeatureConfig::default(),
        }
    }
}

impl Default for ModelWeightsConfig {
    fn default() -> Self {
        Self {
            dynamic_weights: true,
            update_frequency: 300, // 5 minutes
            lookback_period: 3600, // 1 hour
            initial_weights: vec![
                (ModelType::Transformer, 0.25),
                (ModelType::LSTM, 0.20),
                (ModelType::XGBoost, 0.20),
                (ModelType::LightGBM, 0.20),
                (ModelType::IsolationForest, 0.15),
            ],
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            isotonic_regression: true,
            platt_scaling: false,
            window_size: 1000,
            min_samples: 100,
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            price_lags: 20,
            technical_indicators: vec![
                "RSI".to_string(),
                "MACD".to_string(),
                "BB".to_string(),
                "ATR".to_string(),
            ],
            microstructure_features: true,
            order_flow_features: true,
            sentiment_features: false,
        }
    }
}