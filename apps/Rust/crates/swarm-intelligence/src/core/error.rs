//! Error types for swarm intelligence framework

use thiserror::Error;

/// Main error type for swarm intelligence operations
#[derive(Error, Debug)]
pub enum SwarmError {
    /// Algorithm initialization failed
    #[error("Algorithm initialization failed: {0}")]
    InitializationError(String),
    
    /// Optimization process failed
    #[error("Optimization failed: {0}")]
    OptimizationError(String),
    
    /// Parameter validation failed
    #[error("Invalid parameter: {0}")]
    ParameterError(String),
    
    /// Population management error
    #[error("Population error: {0}")]
    PopulationError(String),
    
    /// Mathematical computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    /// Parallel processing error
    #[error("Parallel processing error: {0}")]
    ParallelError(String),
    
    /// CDFA integration error
    #[error("CDFA integration error: {0}")]
    CdfaError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// IO operation failed
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Thread pool error
    #[error("Thread pool error: {0}")]
    ThreadPoolError(String),
    
    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    /// SIMD operation error
    #[cfg(feature = "simd")]
    #[error("SIMD operation error: {0}")]
    SimdError(String),
    
    /// GPU operation error
    #[cfg(feature = "gpu")]
    #[error("GPU operation error: {0}")]
    GpuError(String),
}

impl SwarmError {
    /// Create a new initialization error
    pub fn initialization<S: Into<String>>(msg: S) -> Self {
        Self::InitializationError(msg.into())
    }
    
    /// Create a new optimization error
    pub fn optimization<S: Into<String>>(msg: S) -> Self {
        Self::OptimizationError(msg.into())
    }
    
    /// Create a new parameter error
    pub fn parameter<S: Into<String>>(msg: S) -> Self {
        Self::ParameterError(msg.into())
    }
    
    /// Create a new population error
    pub fn population<S: Into<String>>(msg: S) -> Self {
        Self::PopulationError(msg.into())
    }
    
    /// Create a new computation error
    pub fn computation<S: Into<String>>(msg: S) -> Self {
        Self::ComputationError(msg.into())
    }
    
    /// Create a new parallel error
    pub fn parallel<S: Into<String>>(msg: S) -> Self {
        Self::ParallelError(msg.into())
    }
    
    /// Create a new CDFA error
    pub fn cdfa<S: Into<String>>(msg: S) -> Self {
        Self::CdfaError(msg.into())
    }
    
    /// Create a new configuration error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(msg.into())
    }
    
    /// Create a new thread pool error
    pub fn thread_pool<S: Into<String>>(msg: S) -> Self {
        Self::ThreadPoolError(msg.into())
    }
    
    /// Create a new memory error
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        Self::MemoryError(msg.into())
    }
    
    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InitializationError(_) => false,
            Self::OptimizationError(_) => false,
            Self::ParameterError(_) => true,
            Self::PopulationError(_) => true,
            Self::ComputationError(_) => true,
            Self::ParallelError(_) => true,
            Self::CdfaError(_) => true,
            Self::ConfigError(_) => true,
            Self::IoError(_) => false,
            Self::SerializationError(_) => false,
            Self::ThreadPoolError(_) => false,
            Self::MemoryError(_) => false,
            #[cfg(feature = "simd")]
            Self::SimdError(_) => true,
            #[cfg(feature = "gpu")]
            Self::GpuError(_) => true,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InitializationError(_) => ErrorSeverity::Critical,
            Self::OptimizationError(_) => ErrorSeverity::High,
            Self::ParameterError(_) => ErrorSeverity::Medium,
            Self::PopulationError(_) => ErrorSeverity::Medium,
            Self::ComputationError(_) => ErrorSeverity::Medium,
            Self::ParallelError(_) => ErrorSeverity::Medium,
            Self::CdfaError(_) => ErrorSeverity::High,
            Self::ConfigError(_) => ErrorSeverity::Medium,
            Self::IoError(_) => ErrorSeverity::High,
            Self::SerializationError(_) => ErrorSeverity::Medium,
            Self::ThreadPoolError(_) => ErrorSeverity::High,
            Self::MemoryError(_) => ErrorSeverity::Critical,
            #[cfg(feature = "simd")]
            Self::SimdError(_) => ErrorSeverity::Low,
            #[cfg(feature = "gpu")]
            Self::GpuError(_) => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Result type for swarm operations
pub type SwarmResult<T> = Result<T, SwarmError>;

/// Validation errors for algorithm parameters
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Parameter {name} value {value} is outside valid range [{min}, {max}]")]
    OutOfRange {
        name: String,
        value: f64,
        min: f64,
        max: f64,
    },
    
    #[error("Parameter {name} with value {value} failed validation: {reason}")]
    ValidationFailed {
        name: String,
        value: String,
        reason: String,
    },
    
    #[error("Required parameter {name} is missing")]
    MissingParameter { name: String },
    
    #[error("Parameter {name} has incompatible type")]
    TypeMismatch { name: String },
}

impl ValidationError {
    pub fn out_of_range<S: Into<String>>(name: S, value: f64, min: f64, max: f64) -> Self {
        Self::OutOfRange {
            name: name.into(),
            value,
            min,
            max,
        }
    }
    
    pub fn validation_failed<S: Into<String>>(name: S, value: S, reason: S) -> Self {
        Self::ValidationFailed {
            name: name.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }
    
    pub fn missing_parameter<S: Into<String>>(name: S) -> Self {
        Self::MissingParameter {
            name: name.into(),
        }
    }
    
    pub fn type_mismatch<S: Into<String>>(name: S) -> Self {
        Self::TypeMismatch {
            name: name.into(),
        }
    }
}

impl From<ValidationError> for SwarmError {
    fn from(err: ValidationError) -> Self {
        Self::ParameterError(err.to_string())
    }
}

/// Helper macro for parameter validation
#[macro_export]
macro_rules! validate_parameter {
    ($param:expr, $name:expr, $min:expr, $max:expr) => {
        if $param < $min || $param > $max {
            return Err(crate::core::error::ValidationError::out_of_range(
                $name, $param, $min, $max
            ).into());
        }
    };
    
    ($param:expr, $name:expr, $condition:expr, $reason:expr) => {
        if !$condition {
            return Err(crate::core::error::ValidationError::validation_failed(
                $name, $param.to_string(), $reason
            ).into());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = SwarmError::initialization("Test error");
        assert!(matches!(err, SwarmError::InitializationError(_)));
        assert!(!err.is_recoverable());
        assert_eq!(err.severity(), ErrorSeverity::Critical);
    }
    
    #[test]
    fn test_validation_error() {
        let err = ValidationError::out_of_range("test_param", 15.0, 0.0, 10.0);
        assert!(err.to_string().contains("test_param"));
        assert!(err.to_string().contains("15"));
        
        let swarm_err: SwarmError = err.into();
        assert!(matches!(swarm_err, SwarmError::ParameterError(_)));
    }
    
    #[test]
    fn test_parameter_validation_macro() {
        fn test_function(value: f64) -> SwarmResult<()> {
            validate_parameter!(value, "test_value", 0.0, 10.0);
            Ok(())
        }
        
        assert!(test_function(5.0).is_ok());
        assert!(test_function(15.0).is_err());
        assert!(test_function(-5.0).is_err());
    }
}