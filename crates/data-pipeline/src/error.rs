//! Error types for the data pipeline

use thiserror::Error;

/// Main error type for the data pipeline
#[derive(Error, Debug)]
pub enum DataPipelineError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Streaming error: {0}")]
    Streaming(#[from] StreamingError),
    
    #[error("Sentiment analysis error: {0}")]
    Sentiment(#[from] SentimentError),
    
    #[error("Indicator calculation error: {0}")]
    Indicator(#[from] IndicatorError),
    
    #[error("Data fusion error: {0}")]
    Fusion(#[from] FusionError),
    
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
    
    #[error("Feature extraction error: {0}")]
    Feature(#[from] FeatureError),
    
    #[error("Monitoring error: {0}")]
    Monitoring(#[from] MonitoringError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    #[cfg(feature = "kafka")]
    #[error("Kafka error: {0}")]
    Kafka(#[from] rdkafka::error::KafkaError),
    
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    
    #[error("Polars error: {0}")]
    Polars(#[from] polars::error::PolarsError),
    
    #[cfg(feature = "nlp")]
    #[error("Tokenization error: {0}")]
    Tokenization(#[from] tokenizers::Error),
    
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Data quality error: {0}")]
    DataQuality(String),
    
    #[error("Schema validation error: {0}")]
    SchemaValidation(String),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Streaming-specific errors
#[derive(Error, Debug)]
pub enum StreamingError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Consumer error: {0}")]
    Consumer(String),
    
    #[error("Producer error: {0}")]
    Producer(String),
    
    #[error("Topic not found: {0}")]
    TopicNotFound(String),
    
    #[error("Partition error: {0}")]
    Partition(String),
    
    #[error("Offset error: {0}")]
    Offset(String),
    
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    #[error("Compression error: {0}")]
    Compression(String),
    
    #[error("Authentication error: {0}")]
    Authentication(String),
    
    #[error("Authorization error: {0}")]
    Authorization(String),
    
    #[error("Network timeout: {0}")]
    NetworkTimeout(String),
    
    #[error("Buffer overflow: {0}")]
    BufferOverflow(String),
    
    #[error("Schema registry error: {0}")]
    SchemaRegistry(String),
}

/// Sentiment analysis errors
#[derive(Error, Debug)]
pub enum SentimentError {
    #[error("Model loading error: {0}")]
    ModelLoading(String),
    
    #[error("Tokenization error: {0}")]
    Tokenization(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
    
    #[error("Preprocessing error: {0}")]
    Preprocessing(String),
    
    #[error("Language detection error: {0}")]
    LanguageDetection(String),
    
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    
    #[error("Text too long: {0}")]
    TextTooLong(String),
    
    #[error("Empty text")]
    EmptyText,
    
    #[error("GPU error: {0}")]
    Gpu(String),
    
    #[error("Memory error: {0}")]
    Memory(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Batch processing error: {0}")]
    BatchProcessing(String),
}

/// Technical indicator errors
#[derive(Error, Debug)]
pub enum IndicatorError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Calculation error: {0}")]
    Calculation(String),
    
    #[error("SIMD error: {0}")]
    Simd(String),
    
    #[error("Window size error: {0}")]
    WindowSize(String),
    
    #[error("Data type error: {0}")]
    DataType(String),
    
    #[error("Missing data: {0}")]
    MissingData(String),
    
    #[error("Overflow error: {0}")]
    Overflow(String),
    
    #[error("Underflow error: {0}")]
    Underflow(String),
    
    #[error("Division by zero")]
    DivisionByZero,
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
}

/// Data fusion errors
#[derive(Error, Debug)]
pub enum FusionError {
    #[error("Alignment error: {0}")]
    Alignment(String),
    
    #[error("Interpolation error: {0}")]
    Interpolation(String),
    
    #[error("Weight error: {0}")]
    Weight(String),
    
    #[error("Missing source: {0}")]
    MissingSource(String),
    
    #[error("Timestamp mismatch: {0}")]
    TimestampMismatch(String),
    
    #[error("Data type mismatch: {0}")]
    DataTypeMismatch(String),
    
    #[error("Quality threshold not met: {0}")]
    QualityThreshold(String),
    
    #[error("Outlier detection error: {0}")]
    OutlierDetection(String),
    
    #[error("Kalman filter error: {0}")]
    KalmanFilter(String),
    
    #[error("Bayesian fusion error: {0}")]
    BayesianFusion(String),
    
    #[error("Neural fusion error: {0}")]
    NeuralFusion(String),
    
    #[error("Algorithm error: {0}")]
    Algorithm(String),
}

/// Validation errors
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),
    
    #[error("Range validation failed: {0}")]
    RangeValidation(String),
    
    #[error("Type validation failed: {0}")]
    TypeValidation(String),
    
    #[error("Format validation failed: {0}")]
    FormatValidation(String),
    
    #[error("Duplicate detected: {0}")]
    Duplicate(String),
    
    #[error("Anomaly detected: {0}")]
    Anomaly(String),
    
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    #[error("Checksum mismatch: {0}")]
    ChecksumMismatch(String),
    
    #[error("Timestamp validation failed: {0}")]
    TimestampValidation(String),
    
    #[error("Quality score too low: {0}")]
    QualityScore(String),
}

/// Feature extraction errors
#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Feature calculation error: {0}")]
    Calculation(String),
    
    #[error("Feature selection error: {0}")]
    Selection(String),
    
    #[error("Feature scaling error: {0}")]
    Scaling(String),
    
    #[error("Feature transformation error: {0}")]
    Transformation(String),
    
    #[error("Dimensionality reduction error: {0}")]
    DimensionalityReduction(String),
    
    #[error("Statistical feature error: {0}")]
    Statistical(String),
    
    #[error("Frequency domain error: {0}")]
    FrequencyDomain(String),
    
    #[error("Wavelet transform error: {0}")]
    WaveletTransform(String),
    
    #[error("Time series feature error: {0}")]
    TimeSeries(String),
    
    #[error("Polynomial feature error: {0}")]
    Polynomial(String),
    
    #[error("Interaction feature error: {0}")]
    Interaction(String),
    
    #[error("Feature encoding error: {0}")]
    Encoding(String),
}

/// Monitoring errors
#[derive(Error, Debug)]
pub enum MonitoringError {
    #[error("Metrics collection error: {0}")]
    MetricsCollection(String),
    
    #[error("Health check error: {0}")]
    HealthCheck(String),
    
    #[error("Alert error: {0}")]
    Alert(String),
    
    #[error("Logging error: {0}")]
    Logging(String),
    
    #[error("Tracing error: {0}")]
    Tracing(String),
    
    #[error("Dashboard error: {0}")]
    Dashboard(String),
    
    #[error("Notification error: {0}")]
    Notification(String),
    
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Query error: {0}")]
    Query(String),
    
    #[error("Aggregation error: {0}")]
    Aggregation(String),
    
    #[error("Export error: {0}")]
    Export(String),
    
    #[error("Import error: {0}")]
    Import(String),
}

/// Result type for the data pipeline
pub type DataPipelineResult<T> = Result<T, DataPipelineError>;

/// Result type for streaming operations
pub type StreamingResult<T> = Result<T, StreamingError>;

/// Result type for sentiment analysis
pub type SentimentResult<T> = Result<T, SentimentError>;

/// Result type for indicator calculations
pub type IndicatorResult<T> = Result<T, IndicatorError>;

/// Result type for data fusion
pub type FusionResult<T> = Result<T, FusionError>;

/// Result type for validation
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Result type for feature extraction
pub type FeatureResult<T> = Result<T, FeatureError>;

/// Result type for monitoring
pub type MonitoringResult<T> = Result<T, MonitoringError>;

/// Error context for debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub component: String,
    pub operation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub thread_id: String,
    pub additional_info: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            component: component.to_string(),
            operation: operation.to_string(),
            timestamp: chrono::Utc::now(),
            thread_id: format!("{:?}", std::thread::current().id()),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    pub fn add_info(mut self, key: &str, value: &str) -> Self {
        self.additional_info.insert(key.to_string(), value.to_string());
        self
    }
}

/// Error reporting trait
pub trait ErrorReporter {
    fn report_error(&self, error: &DataPipelineError, context: &ErrorContext);
    fn report_warning(&self, message: &str, context: &ErrorContext);
    fn report_info(&self, message: &str, context: &ErrorContext);
}

/// Default error reporter
pub struct DefaultErrorReporter;

impl ErrorReporter for DefaultErrorReporter {
    fn report_error(&self, error: &DataPipelineError, context: &ErrorContext) {
        tracing::error!(
            component = %context.component,
            operation = %context.operation,
            timestamp = %context.timestamp,
            thread_id = %context.thread_id,
            error = %error,
            "Data pipeline error occurred"
        );
    }
    
    fn report_warning(&self, message: &str, context: &ErrorContext) {
        tracing::warn!(
            component = %context.component,
            operation = %context.operation,
            timestamp = %context.timestamp,
            thread_id = %context.thread_id,
            message = %message,
            "Data pipeline warning"
        );
    }
    
    fn report_info(&self, message: &str, context: &ErrorContext) {
        tracing::info!(
            component = %context.component,
            operation = %context.operation,
            timestamp = %context.timestamp,
            thread_id = %context.thread_id,
            message = %message,
            "Data pipeline info"
        );
    }
}

/// Utility functions for error handling
pub mod utils {
    use super::*;
    
    /// Convert any error to DataPipelineError
    pub fn to_pipeline_error<E: std::error::Error + Send + Sync + 'static>(
        error: E,
        component: &str,
    ) -> DataPipelineError {
        DataPipelineError::Unknown(format!("{}: {}", component, error))
    }
    
    /// Create a timeout error
    pub fn timeout_error(operation: &str, timeout: std::time::Duration) -> DataPipelineError {
        DataPipelineError::Timeout(format!("{} timed out after {:?}", operation, timeout))
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted_error(resource: &str, details: &str) -> DataPipelineError {
        DataPipelineError::ResourceExhausted(format!("{} exhausted: {}", resource, details))
    }
    
    /// Create a data quality error
    pub fn data_quality_error(issue: &str, details: &str) -> DataPipelineError {
        DataPipelineError::DataQuality(format!("{}: {}", issue, details))
    }
    
    /// Create a schema validation error
    pub fn schema_validation_error(field: &str, expected: &str, actual: &str) -> DataPipelineError {
        DataPipelineError::SchemaValidation(format!(
            "Field '{}' expected {} but got {}",
            field, expected, actual
        ))
    }
    
    /// Check if error is retryable
    pub fn is_retryable(error: &DataPipelineError) -> bool {
        match error {
            DataPipelineError::Streaming(StreamingError::NetworkTimeout(_)) => true,
            DataPipelineError::Streaming(StreamingError::ConnectionFailed(_)) => true,
            DataPipelineError::Network(_) => true,
            DataPipelineError::Timeout(_) => true,
            DataPipelineError::ResourceExhausted(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity
    pub fn get_severity(error: &DataPipelineError) -> ErrorSeverity {
        match error {
            DataPipelineError::Configuration(_) => ErrorSeverity::Critical,
            DataPipelineError::Streaming(StreamingError::ConnectionFailed(_)) => ErrorSeverity::High,
            DataPipelineError::Streaming(StreamingError::Authentication(_)) => ErrorSeverity::Critical,
            DataPipelineError::Streaming(StreamingError::Authorization(_)) => ErrorSeverity::Critical,
            DataPipelineError::Validation(ValidationError::SchemaValidation(_)) => ErrorSeverity::High,
            DataPipelineError::Validation(ValidationError::Anomaly(_)) => ErrorSeverity::Medium,
            DataPipelineError::DataQuality(_) => ErrorSeverity::Medium,
            DataPipelineError::Timeout(_) => ErrorSeverity::Low,
            DataPipelineError::ResourceExhausted(_) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = DataPipelineError::Configuration("Test error".to_string());
        assert!(matches!(error, DataPipelineError::Configuration(_)));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_component", "test_operation")
            .add_info("key1", "value1")
            .add_info("key2", "value2");
        
        assert_eq!(context.component, "test_component");
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.additional_info.get("key1"), Some(&"value1".to_string()));
        assert_eq!(context.additional_info.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_error_severity() {
        let error = DataPipelineError::Configuration("Test".to_string());
        let severity = utils::get_severity(&error);
        assert_eq!(severity, ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_retryable() {
        let error = DataPipelineError::Timeout("Test timeout".to_string());
        assert!(utils::is_retryable(&error));
        
        let error = DataPipelineError::Configuration("Test config".to_string());
        assert!(!utils::is_retryable(&error));
    }
}