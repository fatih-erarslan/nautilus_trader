//! Error handling for TorchScript Fusion operations
//!
//! This module provides comprehensive error handling for all fusion operations,
//! including hardware acceleration errors, dimension mismatches, and numerical issues.

use thiserror::Error;

/// Result type for TorchScript Fusion operations
pub type Result<T> = std::result::Result<T, FusionError>;

/// Error types for TorchScript Fusion operations
#[derive(Error, Debug)]
pub enum FusionError {
    /// Candle framework errors
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Device-related errors
    #[error("Device error: {message}")]
    Device { message: String },

    /// GPU acceleration errors
    #[error("GPU acceleration error: {message}")]
    GpuAcceleration { message: String },

    /// CUDA-specific errors
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {message}")]
    Cuda { message: String },

    /// Metal-specific errors
    #[cfg(feature = "metal")]
    #[error("Metal error: {message}")]
    Metal { message: String },

    /// ROCm-specific errors
    #[cfg(feature = "rocm")]
    #[error("ROCm error: {message}")]
    Rocm { message: String },

    /// Input validation errors
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Empty input errors
    #[error("Empty input provided")]
    EmptyInput,

    /// Unsupported fusion type
    #[error("Unsupported fusion type: {fusion_type}")]
    UnsupportedFusionType { fusion_type: String },

    /// Numerical computation errors
    #[error("Numerical error: {message}")]
    Numerical { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Model compilation errors
    #[error("Model compilation error: {message}")]
    Compilation { message: String },

    /// Memory allocation errors
    #[error("Memory allocation error: {message}")]
    Memory { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic errors
    #[error("Error: {message}")]
    Generic { message: String },
}

impl FusionError {
    /// Create a new device error
    pub fn device(message: impl Into<String>) -> Self {
        Self::Device {
            message: message.into(),
        }
    }

    /// Create a new GPU acceleration error
    pub fn gpu_acceleration(message: impl Into<String>) -> Self {
        Self::GpuAcceleration {
            message: message.into(),
        }
    }

    /// Create a new CUDA error
    #[cfg(feature = "cuda")]
    pub fn cuda(message: impl Into<String>) -> Self {
        Self::Cuda {
            message: message.into(),
        }
    }

    /// Create a new Metal error
    #[cfg(feature = "metal")]
    pub fn metal(message: impl Into<String>) -> Self {
        Self::Metal {
            message: message.into(),
        }
    }

    /// Create a new ROCm error
    #[cfg(feature = "rocm")]
    pub fn rocm(message: impl Into<String>) -> Self {
        Self::Rocm {
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new dimension mismatch error
    pub fn dimension_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a new unsupported fusion type error
    pub fn unsupported_fusion_type(fusion_type: impl Into<String>) -> Self {
        Self::UnsupportedFusionType {
            fusion_type: fusion_type.into(),
        }
    }

    /// Create a new numerical error
    pub fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new compilation error
    pub fn compilation(message: impl Into<String>) -> Self {
        Self::Compilation {
            message: message.into(),
        }
    }

    /// Create a new memory allocation error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a new generic error
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Hardware errors might be recoverable by falling back to CPU
            Self::GpuAcceleration { .. } 
            | Self::Device { .. } => true,
            
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => true,
            
            #[cfg(feature = "metal")]
            Self::Metal { .. } => true,
            
            #[cfg(feature = "rocm")]
            Self::Rocm { .. } => true,
            
            // Memory errors might be recoverable with smaller batch sizes
            Self::Memory { .. } => true,
            
            // Numerical errors might be recoverable with different parameters
            Self::Numerical { .. } => true,
            
            // These are typically not recoverable
            Self::EmptyInput
            | Self::DimensionMismatch { .. }
            | Self::InvalidInput { .. }
            | Self::UnsupportedFusionType { .. }
            | Self::Configuration { .. }
            | Self::Compilation { .. }
            | Self::Serialization(_)
            | Self::Io(_)
            | Self::Candle(_)
            | Self::Generic { .. } => false,
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Candle(_) => ErrorCategory::Framework,
            Self::Device { .. } 
            | Self::GpuAcceleration { .. } => ErrorCategory::Hardware,
            
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => ErrorCategory::Hardware,
            
            #[cfg(feature = "metal")]
            Self::Metal { .. } => ErrorCategory::Hardware,
            
            #[cfg(feature = "rocm")]
            Self::Rocm { .. } => ErrorCategory::Hardware,
            
            Self::InvalidInput { .. }
            | Self::DimensionMismatch { .. }
            | Self::EmptyInput => ErrorCategory::Input,
            
            Self::UnsupportedFusionType { .. } => ErrorCategory::Configuration,
            Self::Configuration { .. } => ErrorCategory::Configuration,
            Self::Compilation { .. } => ErrorCategory::Compilation,
            Self::Numerical { .. } => ErrorCategory::Numerical,
            Self::Memory { .. } => ErrorCategory::Memory,
            Self::Serialization(_) => ErrorCategory::Serialization,
            Self::Io(_) => ErrorCategory::Io,
            Self::Generic { .. } => ErrorCategory::Generic,
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Framework-related errors
    Framework,
    /// Hardware/device errors
    Hardware,
    /// Input validation errors
    Input,
    /// Configuration errors
    Configuration,
    /// Model compilation errors
    Compilation,
    /// Numerical computation errors
    Numerical,
    /// Memory allocation errors
    Memory,
    /// Serialization errors
    Serialization,
    /// I/O errors
    Io,
    /// Generic errors
    Generic,
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorCategory::Framework => write!(f, "Framework"),
            ErrorCategory::Hardware => write!(f, "Hardware"),
            ErrorCategory::Input => write!(f, "Input"),
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Compilation => write!(f, "Compilation"),
            ErrorCategory::Numerical => write!(f, "Numerical"),
            ErrorCategory::Memory => write!(f, "Memory"),
            ErrorCategory::Serialization => write!(f, "Serialization"),
            ErrorCategory::Io => write!(f, "I/O"),
            ErrorCategory::Generic => write!(f, "Generic"),
        }
    }
}

/// Extension trait for Result to provide additional error handling capabilities
pub trait ResultExt<T> {
    /// Add context to an error
    fn with_context(self, context: &str) -> Result<T>;
    
    /// Chain errors with additional information
    fn chain_err(self, f: impl FnOnce() -> FusionError) -> Result<T>;
    
    /// Convert to a recoverable error if possible
    fn make_recoverable(self) -> Result<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, context: &str) -> Result<T> {
        self.map_err(|e| FusionError::generic(format!("{}: {}", context, e)))
    }

    fn chain_err(self, f: impl FnOnce() -> FusionError) -> Result<T> {
        self.map_err(|_| f())
    }

    fn make_recoverable(self) -> Result<T> {
        match self {
            Ok(v) => Ok(v),
            Err(e) if e.is_recoverable() => Err(e),
            Err(e) => Err(FusionError::generic(format!("Non-recoverable error: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = FusionError::device("Test device error");
        assert!(matches!(err, FusionError::Device { .. }));
        assert!(err.is_recoverable());
        assert_eq!(err.category(), ErrorCategory::Hardware);
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = FusionError::gpu_acceleration("GPU error");
        assert!(recoverable.is_recoverable());

        let non_recoverable = FusionError::EmptyInput;
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(FusionError::device("test").category(), ErrorCategory::Hardware);
        assert_eq!(FusionError::EmptyInput.category(), ErrorCategory::Input);
        assert_eq!(FusionError::numerical("test").category(), ErrorCategory::Numerical);
    }

    #[test]
    fn test_result_extensions() {
        let result: Result<i32> = Err(FusionError::device("test"));
        let with_context = result.with_context("Additional context");
        assert!(with_context.is_err());
        
        let result2: Result<i32> = Ok(42);
        let with_context2 = result2.with_context("This shouldn't change anything");
        assert_eq!(with_context2.unwrap(), 42);
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = FusionError::dimension_mismatch("(3, 100)", "(2, 100)");
        assert!(matches!(err, FusionError::DimensionMismatch { .. }));
        assert!(!err.is_recoverable());
        assert_eq!(err.category(), ErrorCategory::Input);
    }
}