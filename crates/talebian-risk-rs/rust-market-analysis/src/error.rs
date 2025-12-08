//! Error handling for the market analysis engine

use thiserror::Error;

/// Main error type for market analysis operations
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Insufficient data for analysis
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Calculation error
    #[error("Calculation error: {0}")]
    CalculationError(String),
    
    /// Model training error
    #[error("Model training error: {0}")]
    ModelError(String),
    
    /// Data validation error
    #[error("Data validation error: {0}")]
    ValidationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    /// Async task error
    #[error("Async task error: {0}")]
    TaskError(#[from] tokio::task::JoinError),
    
    /// External crate errors
    #[error("External error: {0}")]
    ExternalError(String),
    
    /// Feature extraction error
    #[error("Feature extraction error: {0}")]
    FeatureError(String),
    
    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
    
    /// Memory allocation error
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    /// Timeout error
    #[error("Operation timed out: {0}")]
    TimeoutError(String),
    
    /// Concurrent access error
    #[error("Concurrent access error: {0}")]
    ConcurrencyError(String),
}

/// Result type alias for market analysis operations
pub type Result<T> = std::result::Result<T, AnalysisError>;

impl AnalysisError {
    /// Create an insufficient data error
    pub fn insufficient_data(msg: impl Into<String>) -> Self {
        Self::InsufficientData(msg.into())
    }
    
    /// Create an invalid configuration error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
    
    /// Create a calculation error
    pub fn calculation_error(msg: impl Into<String>) -> Self {
        Self::CalculationError(msg.into())
    }
    
    /// Create a model error
    pub fn model_error(msg: impl Into<String>) -> Self {
        Self::ModelError(msg.into())
    }
    
    /// Create a validation error
    pub fn validation_error(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }
    
    /// Create a feature extraction error
    pub fn feature_error(msg: impl Into<String>) -> Self {
        Self::FeatureError(msg.into())
    }
    
    /// Create a cache error
    pub fn cache_error(msg: impl Into<String>) -> Self {
        Self::CacheError(msg.into())
    }
    
    /// Create a memory error
    pub fn memory_error(msg: impl Into<String>) -> Self {
        Self::MemoryError(msg.into())
    }
    
    /// Create a timeout error
    pub fn timeout_error(msg: impl Into<String>) -> Self {
        Self::TimeoutError(msg.into())
    }
    
    /// Create a concurrency error
    pub fn concurrency_error(msg: impl Into<String>) -> Self {
        Self::ConcurrencyError(msg.into())
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InsufficientData(_) => true,
            Self::ValidationError(_) => true,
            Self::NetworkError(_) => true,
            Self::TimeoutError(_) => true,
            Self::CacheError(_) => true,
            Self::ConcurrencyError(_) => true,
            _ => false,
        }
    }
    
    /// Check if this is a critical error
    pub fn is_critical(&self) -> bool {
        match self {
            Self::InvalidConfig(_) => true,
            Self::ModelError(_) => true,
            Self::MemoryError(_) => true,
            Self::IoError(_) => true,
            _ => false,
        }
    }
    
    /// Get error category for logging/monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::InsufficientData(_) => "data",
            Self::InvalidConfig(_) => "config",
            Self::CalculationError(_) => "calculation",
            Self::ModelError(_) => "model",
            Self::ValidationError(_) => "validation",
            Self::IoError(_) => "io",
            Self::SerializationError(_) => "serialization",
            Self::NetworkError(_) => "network",
            Self::TaskError(_) => "async",
            Self::ExternalError(_) => "external",
            Self::FeatureError(_) => "feature",
            Self::CacheError(_) => "cache",
            Self::MemoryError(_) => "memory",
            Self::TimeoutError(_) => "timeout",
            Self::ConcurrencyError(_) => "concurrency",
        }
    }
    
    /// Get suggested retry delay in seconds
    pub fn retry_delay(&self) -> Option<u64> {
        match self {
            Self::NetworkError(_) => Some(5),
            Self::TimeoutError(_) => Some(1),
            Self::CacheError(_) => Some(1),
            Self::ConcurrencyError(_) => Some(1),
            _ => None,
        }
    }
}

/// Conversion from external crate errors
impl From<ndarray::ShapeError> for AnalysisError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::CalculationError(format!("Array shape error: {}", err))
    }
}

impl From<chrono::ParseError> for AnalysisError {
    fn from(err: chrono::ParseError) -> Self {
        Self::ValidationError(format!("Date parsing error: {}", err))
    }
}

impl From<std::num::ParseFloatError> for AnalysisError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::ValidationError(format!("Float parsing error: {}", err))
    }
}

impl From<std::num::ParseIntError> for AnalysisError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::ValidationError(format!("Integer parsing error: {}", err))
    }
}

impl From<config::ConfigError> for AnalysisError {
    fn from(err: config::ConfigError) -> Self {
        Self::InvalidConfig(format!("Configuration error: {}", err))
    }
}

// Custom error types for specific modules

/// Whale analysis specific errors
#[derive(Error, Debug)]
pub enum WhaleAnalysisError {
    #[error("Volume profile calculation failed: {0}")]
    VolumeProfileError(String),
    
    #[error("Order flow analysis failed: {0}")]
    OrderFlowError(String),
    
    #[error("Smart money detection failed: {0}")]
    SmartMoneyError(String),
    
    #[error("Whale classification failed: {0}")]
    ClassificationError(String),
}

/// Regime detection specific errors
#[derive(Error, Debug)]
pub enum RegimeDetectionError {
    #[error("Feature extraction failed: {0}")]
    FeatureExtractionError(String),
    
    #[error("Model prediction failed: {0}")]
    PredictionError(String),
    
    #[error("Regime transition invalid: {0}")]
    TransitionError(String),
    
    #[error("Ensemble consensus failed: {0}")]
    ConsensusError(String),
}

/// Pattern recognition specific errors
#[derive(Error, Debug)]
pub enum PatternRecognitionError {
    #[error("Pattern detection failed: {0}")]
    DetectionError(String),
    
    #[error("Pattern validation failed: {0}")]
    ValidationError(String),
    
    #[error("Support/resistance calculation failed: {0}")]
    SupportResistanceError(String),
    
    #[error("Technical pattern analysis failed: {0}")]
    TechnicalPatternError(String),
}

/// Predictive modeling specific errors
#[derive(Error, Debug)]
pub enum PredictiveModelError {
    #[error("Time series forecasting failed: {0}")]
    TimeSeriesError(String),
    
    #[error("Neural network prediction failed: {0}")]
    NeuralNetworkError(String),
    
    #[error("Model training failed: {0}")]
    TrainingError(String),
    
    #[error("Prediction validation failed: {0}")]
    ValidationError(String),
}

/// Market microstructure specific errors
#[derive(Error, Debug)]
pub enum MicrostructureError {
    #[error("Order book analysis failed: {0}")]
    OrderBookError(String),
    
    #[error("Trade classification failed: {0}")]
    TradeClassificationError(String),
    
    #[error("Liquidity measurement failed: {0}")]
    LiquidityError(String),
    
    #[error("Market efficiency test failed: {0}")]
    EfficiencyError(String),
}

// Convert specific errors to general AnalysisError
impl From<WhaleAnalysisError> for AnalysisError {
    fn from(err: WhaleAnalysisError) -> Self {
        Self::ExternalError(err.to_string())
    }
}

impl From<RegimeDetectionError> for AnalysisError {
    fn from(err: RegimeDetectionError) -> Self {
        Self::ExternalError(err.to_string())
    }
}

impl From<PatternRecognitionError> for AnalysisError {
    fn from(err: PatternRecognitionError) -> Self {
        Self::ExternalError(err.to_string())
    }
}

impl From<PredictiveModelError> for AnalysisError {
    fn from(err: PredictiveModelError) -> Self {
        Self::ExternalError(err.to_string())
    }
}

impl From<MicrostructureError> for AnalysisError {
    fn from(err: MicrostructureError) -> Self {
        Self::ExternalError(err.to_string())
    }
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub symbol: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            symbol: None,
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }
    
    pub fn with_info(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }
}

/// Result with context for better error reporting
pub type ContextResult<T> = std::result::Result<T, (AnalysisError, ErrorContext)>;

/// Helper trait for adding context to results
pub trait ResultExt<T> {
    fn with_context(self, context: ErrorContext) -> ContextResult<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, context: ErrorContext) -> ContextResult<T> {
        self.map_err(|err| (err, context))
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    Retry { max_attempts: u32, delay_seconds: u64 },
    FallbackToDefault,
    FallbackToCache,
    SkipOperation,
    EmergencyStop,
}

impl AnalysisError {
    /// Get suggested recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::NetworkError(_) => RecoveryStrategy::Retry { max_attempts: 3, delay_seconds: 5 },
            Self::TimeoutError(_) => RecoveryStrategy::Retry { max_attempts: 2, delay_seconds: 1 },
            Self::CacheError(_) => RecoveryStrategy::FallbackToDefault,
            Self::InsufficientData(_) => RecoveryStrategy::SkipOperation,
            Self::ValidationError(_) => RecoveryStrategy::FallbackToDefault,
            Self::ConcurrencyError(_) => RecoveryStrategy::Retry { max_attempts: 3, delay_seconds: 1 },
            Self::MemoryError(_) => RecoveryStrategy::EmergencyStop,
            Self::InvalidConfig(_) => RecoveryStrategy::EmergencyStop,
            _ => RecoveryStrategy::FallbackToDefault,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_categories() {
        assert_eq!(AnalysisError::insufficient_data("test").category(), "data");
        assert_eq!(AnalysisError::calculation_error("test").category(), "calculation");
        assert_eq!(AnalysisError::model_error("test").category(), "model");
    }
    
    #[test]
    fn test_error_recoverability() {
        assert!(AnalysisError::insufficient_data("test").is_recoverable());
        assert!(AnalysisError::network_error("test").is_recoverable());
        assert!(!AnalysisError::model_error("test").is_recoverable());
        assert!(!AnalysisError::invalid_config("test").is_recoverable());
    }
    
    #[test]
    fn test_error_criticality() {
        assert!(AnalysisError::invalid_config("test").is_critical());
        assert!(AnalysisError::model_error("test").is_critical());
        assert!(!AnalysisError::insufficient_data("test").is_critical());
        assert!(!AnalysisError::validation_error("test").is_critical());
    }
    
    #[test]
    fn test_retry_delays() {
        assert_eq!(AnalysisError::network_error("test").retry_delay(), Some(5));
        assert_eq!(AnalysisError::timeout_error("test").retry_delay(), Some(1));
        assert_eq!(AnalysisError::model_error("test").retry_delay(), None);
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_symbol("BTCUSDT")
            .with_info("data_points", "100");
            
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.symbol, Some("BTCUSDT".to_string()));
        assert!(context.additional_info.contains_key("data_points"));
    }
}