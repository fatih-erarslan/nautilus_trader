// Error Types for Quantum Pair Analyzer
// Copyright (c) 2025 TENGRI Trading Swarm

use thiserror::Error;

/// Main error type for the analyzer
#[derive(Error, Debug)]
pub enum AnalyzerError {
    #[error("🚫 Mock data detected: {0}")]
    MockDataDetected(String),
    
    #[error("Data source error: {0}")]
    DataSourceError(#[from] anyhow::Error),
    
    #[error("Correlation calculation failed: {0}")]
    CorrelationError(String),
    
    #[error("Quantum optimization failed: {0}")]
    QuantumError(String),
    
    #[error("Swarm optimization failed: {0}")]
    SwarmError(String),
    
    #[error("Sentiment analysis failed: {0}")]
    SentimentError(String),
    
    #[error("Regime detection failed: {0}")]
    RegimeError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(#[from] ConfigError),
    
    #[error("Insufficient market data: need {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Performance optimization failed: {0}")]
    PerformanceError(String),
    
    // Memory-related errors
    #[error("Memory pool exhausted")]
    MemoryPoolExhausted,
    #[error("Memory allocation error")]
    MemoryAllocationError,
    #[error("Memory layout error: {0}")]
    MemoryLayoutError(String),
    #[error("Memory mapping error: {0}")]
    MemoryMappingError(String),
    #[error("System error: {0}")]
    SystemError(String),
    #[error("Quantum disabled")]
    QuantumDisabled,
    #[error("Connection not found: {0}")]
    ConnectionNotFound(String),
    #[error("Unsupported connection type: {0:?}")]
    UnsupportedConnectionType(crate::performance::network_io_optimizer::ConnectionType),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("Message queue full")]
    MessageQueueFull,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Authentication failed: {0}")]
    AuthError(String),
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Configuration-specific errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("🚫 Mock enforcement disabled - PRODUCTION VIOLATION")]
    MockEnforcementDisabled,
    
    #[error("🚫 Mock endpoint detected in exchange: {0}")]
    MockEndpointDetected(String),
    
    #[error("🚫 Test API key detected in exchange: {0}")]
    TestKeyDetected(String),
    
    #[error("Invalid configuration parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Missing required configuration: {0}")]
    MissingConfiguration(String),
    
    #[error("Configuration validation failed: {0}")]
    ValidationFailed(String),
}

/// Exchange-specific errors
#[derive(Error, Debug)]
pub enum ExchangeError {
    #[error("Connection failed to {exchange}: {reason}")]
    ConnectionFailed { exchange: String, reason: String },
    
    #[error("Authentication failed for {exchange}")]
    AuthenticationFailed { exchange: String },
    
    #[error("API rate limit exceeded for {exchange}")]
    RateLimitExceeded { exchange: String },
    
    #[error("Invalid symbol {symbol} on {exchange}")]
    InvalidSymbol { exchange: String, symbol: String },
    
    #[error("WebSocket error for {exchange}: {reason}")]
    WebSocketError { exchange: String, reason: String },
    
    #[error("Data parsing error for {exchange}: {reason}")]
    DataParsingError { exchange: String, reason: String },
}

/// Quantum computing errors
#[derive(Error, Debug)]
pub enum QuantumError {
    #[error("Quantum circuit compilation failed: {0}")]
    CircuitCompilationFailed(String),
    
    #[error("Quantum device unavailable: {0}")]
    DeviceUnavailable(String),
    
    #[error("Quantum measurement failed: {0}")]
    MeasurementFailed(String),
    
    #[error("Quantum optimization convergence failed")]
    ConvergenceFailed,
    
    #[error("Invalid quantum parameters: {0}")]
    InvalidParameters(String),
}

/// Swarm algorithm errors
#[derive(Error, Debug)]
pub enum SwarmError {
    #[error("Swarm convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: u32 },
    
    #[error("Invalid swarm parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Population initialization failed: {0}")]
    PopulationInitFailed(String),
    
    #[error("Fitness evaluation failed: {0}")]
    FitnessEvaluationFailed(String),
    
    #[error("Algorithm-specific error for {algorithm}: {reason}")]
    AlgorithmError { algorithm: String, reason: String },
}

/// Performance-related errors
#[derive(Error, Debug)]
pub enum PerformanceError {
    #[error("SIMD optimization failed: {0}")]
    SIMDFailed(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
    
    #[error("Parallel processing failed: {0}")]
    ParallelProcessingFailed(String),
    
    #[error("Benchmark target not met: expected {expected}, got {actual}")]
    BenchmarkFailed { expected: String, actual: String },
    
    #[error("Latency target exceeded: {target_ms}ms target, {actual_ms}ms actual")]
    LatencyExceeded { target_ms: u64, actual_ms: u64 },
}

/// Result type alias for convenience
pub type AnalyzerResult<T> = Result<T, AnalyzerError>;

impl From<ExchangeError> for AnalyzerError {
    fn from(err: ExchangeError) -> Self {
        AnalyzerError::DataSourceError(err.into())
    }
}

impl From<QuantumError> for AnalyzerError {
    fn from(err: QuantumError) -> Self {
        AnalyzerError::QuantumError(err.to_string())
    }
}

impl From<SwarmError> for AnalyzerError {
    fn from(err: SwarmError) -> Self {
        AnalyzerError::SwarmError(err.to_string())
    }
}

impl From<PerformanceError> for AnalyzerError {
    fn from(err: PerformanceError) -> Self {
        AnalyzerError::PerformanceError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversion() {
        let config_error = ConfigError::MockEnforcementDisabled;
        let analyzer_error: AnalyzerError = config_error.into();
        
        match analyzer_error {
            AnalyzerError::ConfigError(_) => assert!(true),
            _ => panic!("Expected ConfigError conversion"),
        }
    }
    
    #[test]
    fn test_error_display() {
        let error = AnalyzerError::MockDataDetected("test_source".to_string());
        let display = format!("{}", error);
        assert!(display.contains("Mock data detected"));
        assert!(display.contains("test_source"));
    }
    
    #[test]
    fn test_performance_error() {
        let error = PerformanceError::LatencyExceeded {
            target_ms: 100,
            actual_ms: 150,
        };
        let display = format!("{}", error);
        assert!(display.contains("100ms target"));
        assert!(display.contains("150ms actual"));
    }
}