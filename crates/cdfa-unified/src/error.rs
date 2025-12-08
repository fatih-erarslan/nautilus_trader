//! Error types for the unified CDFA library

use thiserror::Error;

/// Result type for CDFA operations
pub type Result<T> = std::result::Result<T, CdfaError>;

/// Alias for convenience
pub type CdfaResult<T> = Result<T>;

/// Comprehensive error type for all CDFA operations
#[derive(Error, Debug)]
pub enum CdfaError {
    /// Input validation errors
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    /// Mathematical computation errors
    #[error("Mathematical error: {message}")]
    MathError { message: String },
    
    /// Numerical instability or convergence errors
    #[error("Numerical error: {message}")]
    NumericalError { message: String },
    
    /// SIMD/parallel processing errors
    #[error("SIMD processing error: {message}")]
    SimdError { message: String },
    
    /// Parallel processing errors
    #[error("Parallel processing error: {message}")]
    ParallelError { message: String },
    
    /// GPU processing errors
    #[cfg(feature = "gpu")]
    #[error("GPU processing error: {message}")]
    GpuError { message: String },
    
    /// Machine learning errors
    #[cfg(feature = "ml")]
    #[error("ML processing error: {message}")]
    MlError { message: String },
    
    /// Pattern detection errors
    #[cfg(feature = "detectors")]
    #[error("Pattern detection error: {message}")]
    DetectionError { message: String },
    
    /// Serialization/deserialization errors
    #[cfg(feature = "serde")]
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Bincode serialization errors
    #[cfg(feature = "serde")]
    #[error("Bincode serialization error: {0}")]
    BincodeError(#[from] bincode::Error),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    /// Resource allocation errors
    #[error("Resource allocation error: {message}")]
    ResourceError { message: String },
    
    /// Timeout errors
    #[error("Operation timed out after {duration_ms}ms")]
    TimeoutError { duration_ms: u64 },
    
    /// Unsupported operation errors
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },
    
    /// Feature not enabled errors
    #[error("Feature '{feature}' is not enabled")]
    FeatureNotEnabled { feature: String },
    
    /// External library errors
    #[error("External library error: {library}: {message}")]
    ExternalError { library: String, message: String },
    
    /// Generic errors with context
    #[error("Error in {context}: {source}")]
    ContextError { context: String, source: Box<CdfaError> },
    
    /// Analysis errors
    #[error("Analysis error: {message}")]
    AnalysisError { message: String },
    
    /// Computation failed errors
    #[error("Computation failed: {message}")]
    ComputationFailed { message: String },
}

impl CdfaError {
    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput { message: message.into() }
    }
    
    /// Create a new dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
    
    /// Create a new mathematical error
    pub fn math_error<S: Into<String>>(message: S) -> Self {
        Self::MathError { message: message.into() }
    }
    
    /// Create a new numerical error
    pub fn numerical_error<S: Into<String>>(message: S) -> Self {
        Self::NumericalError { message: message.into() }
    }
    
    /// Create a new SIMD error
    pub fn simd_error<S: Into<String>>(message: S) -> Self {
        Self::SimdError { message: message.into() }
    }
    
    /// Create a new parallel processing error
    pub fn parallel_error<S: Into<String>>(message: S) -> Self {
        Self::ParallelError { message: message.into() }
    }
    
    /// Create a new GPU error
    #[cfg(feature = "gpu")]
    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuError { message: message.into() }
    }
    
    /// Create a new ML error
    #[cfg(feature = "ml")]
    pub fn ml_error<S: Into<String>>(message: S) -> Self {
        Self::MlError { message: message.into() }
    }
    
    /// Create a new detection error
    #[cfg(feature = "detectors")]
    pub fn detection_error<S: Into<String>>(message: S) -> Self {
        Self::DetectionError { message: message.into() }
    }
    
    /// Create a new configuration error
    pub fn config_error<S: Into<String>>(message: S) -> Self {
        Self::ConfigError { message: message.into() }
    }
    
    /// Create a new analysis error
    pub fn analysis_error<S: Into<String>>(message: S) -> Self {
        Self::AnalysisError { message: message.into() }
    }
    
    /// Create a new computation failed error
    pub fn computation_failed<S: Into<String>>(message: S) -> Self {
        Self::ComputationFailed { message: message.into() }
    }
    
    /// Create a new resource error
    pub fn resource_error<S: Into<String>>(message: S) -> Self {
        Self::ResourceError { message: message.into() }
    }
    
    /// Create a new timeout error
    pub fn timeout_error(duration_ms: u64) -> Self {
        Self::TimeoutError { duration_ms }
    }
    
    /// Create a new unsupported operation error
    pub fn unsupported_operation<S: Into<String>>(operation: S) -> Self {
        Self::UnsupportedOperation { operation: operation.into() }
    }
    
    /// Create a new feature not enabled error
    pub fn feature_not_enabled<S: Into<String>>(feature: S) -> Self {
        Self::FeatureNotEnabled { feature: feature.into() }
    }
    
    /// Create a new external library error
    pub fn external_error<S: Into<String>>(library: S, message: S) -> Self {
        Self::ExternalError {
            library: library.into(),
            message: message.into(),
        }
    }
    
    /// Add context to an error
    pub fn with_context<S: Into<String>>(self, context: S) -> Self {
        Self::ContextError {
            context: context.into(),
            source: Box::new(self),
        }
    }
    
    /// Check if this is a critical error that should halt processing
    pub fn is_critical(&self) -> bool {
        #[cfg(feature = "gpu")]
        let gpu_critical = matches!(self, Self::GpuError { .. });
        #[cfg(not(feature = "gpu"))]
        let gpu_critical = false;
        
        matches!(
            self,
            Self::ResourceError { .. } |
            Self::NumericalError { .. } |
            Self::IoError(_)
        ) || gpu_critical
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput { .. } |
            Self::DimensionMismatch { .. } |
            Self::ConfigError { .. } |
            Self::TimeoutError { .. } |
            Self::UnsupportedOperation { .. } |
            Self::FeatureNotEnabled { .. }
        )
    }
}

/// Macro for creating context errors
#[macro_export]
macro_rules! cdfa_context {
    ($result:expr, $context:expr) => {
        $result.map_err(|e| e.with_context($context))
    };
}

/// Macro for ensuring feature is enabled
#[macro_export]
macro_rules! require_feature {
    ($feature:literal) => {
        #[cfg(not(feature = $feature))]
        return Err(CdfaError::feature_not_enabled($feature));
    };
}

/// Macro for validating input dimensions
#[macro_export]
macro_rules! validate_dimensions {
    ($actual:expr, $expected:expr) => {
        if $actual != $expected {
            return Err(CdfaError::dimension_mismatch($expected, $actual));
        }
    };
}

/// Macro for validating input arrays are not empty
#[macro_export]
macro_rules! validate_not_empty {
    ($array:expr, $name:expr) => {
        if $array.is_empty() {
            return Err(CdfaError::invalid_input(format!("{} cannot be empty", $name)));
        }
    };
}

/// Macro for validating input arrays have same length
#[macro_export]
macro_rules! validate_same_length {
    ($array1:expr, $array2:expr, $name1:expr, $name2:expr) => {
        if $array1.len() != $array2.len() {
            return Err(CdfaError::invalid_input(format!(
                "{} and {} must have the same length: {} vs {}",
                $name1, $name2, $array1.len(), $array2.len()
            )));
        }
    };
}

/// Convert various external errors to CdfaError
impl From<ndarray::ShapeError> for CdfaError {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::invalid_input(format!("NDArray shape error: {}", err))
    }
}

#[cfg(feature = "parallel")]
impl From<rayon::ThreadPoolBuildError> for CdfaError {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        Self::parallel_error(format!("Thread pool build error: {}", err))
    }
}

#[cfg(feature = "tokio")]
impl From<tokio::time::error::Elapsed> for CdfaError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        Self::timeout_error(0) // Duration will be set by the caller
    }
}

#[cfg(all(feature = "gpu", feature = "webgpu"))]
impl From<wgpu::RequestDeviceError> for CdfaError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        Self::gpu_error(format!("GPU device request error: {:?}", err))
    }
}

#[cfg(all(feature = "gpu", feature = "webgpu"))]
impl From<wgpu::CreateSurfaceError> for CdfaError {
    fn from(err: wgpu::CreateSurfaceError) -> Self {
        Self::gpu_error(format!("GPU surface creation error: {:?}", err))
    }
}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for CdfaError {
    fn from(err: candle_core::Error) -> Self {
        Self::ml_error(format!("Candle error: {}", err))
    }
}

// Add From implementations for combinatorial errors
#[derive(Debug, thiserror::Error)]
pub enum CombinatorialError {
    #[error("Invalid combination size: {0}")]
    InvalidCombinationSize(usize),
    #[error("Insufficient diversity in combination")]
    InsufficientDiversity,
    #[error("Excessive redundancy in combination")]
    ExcessiveRedundancy,
    #[error("Synergy detection failed: {0}")]
    SynergyDetectionFailed(String),
    #[error("Combination evaluation failed: {0}")]
    EvaluationFailed(String),
}

impl From<CombinatorialError> for CdfaError {
    fn from(err: CombinatorialError) -> Self {
        Self::AnalysisError { message: err.to_string() }
    }
}

// Add From implementations for machine learning errors  
#[cfg(feature = "ml")]
#[derive(Debug, thiserror::Error)]
pub enum MLError {
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Model error: {0}")]
    ModelError(String),
}

#[cfg(feature = "ml")]
impl From<MLError> for CdfaError {
    fn from(err: MLError) -> Self {
        Self::MlError { message: err.to_string() }
    }
}

// Add From implementations for antifragility errors
#[derive(Debug, thiserror::Error)]
pub enum AntifragilityError {
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
}

impl From<AntifragilityError> for CdfaError {
    fn from(err: AntifragilityError) -> Self {
        Self::AnalysisError { message: err.to_string() }
    }
}

// Add From implementation for futures io errors
// Futures error handling removed - using std::io::Error conversion

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = CdfaError::invalid_input("test message");
        assert!(matches!(err, CdfaError::InvalidInput { .. }));
        assert!(err.is_recoverable());
        assert!(!err.is_critical());
    }
    
    #[test]
    fn test_error_context() {
        let err = CdfaError::math_error("division by zero")
            .with_context("calculating correlation");
        
        assert!(matches!(err, CdfaError::ContextError { .. }));
        assert!(err.to_string().contains("calculating correlation"));
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let err = CdfaError::dimension_mismatch(5, 3);
        assert!(matches!(err, CdfaError::DimensionMismatch { expected: 5, actual: 3 }));
    }
    
    #[test]
    fn test_macros() {
        // Test validate_dimensions macro
        let result: Result<()> = (|| {
            validate_dimensions!(3, 5);
            Ok(())
        })();
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CdfaError::DimensionMismatch { .. }));
    }
}